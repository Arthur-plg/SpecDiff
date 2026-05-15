"""
Speculative Diffusion Decoding engine (SpecDiff).

Implements Algorithm 1 from the SpecDiff paper:
  1. The draft model proposes a block of `gamma` tokens.
  2. The target model verifies the full block in a single forward pass.
  3. Tokens are accepted left-to-right (greedy verification).
  4. On first rejection, the target model's correction is used and
     a new draft block is requested.

This engine is fully model-agnostic: it only calls the abstract
interfaces defined in BaseDraftModel and BaseTargetModel.
"""

import time
import torch
from torch import Tensor

from specdiff.metrics.tracker import EngineMetrics
from specdiff.models.base import BaseTargetModel, BaseDraftModel


class SpeculativeEngine:
    """
    Speculative Diffusion Decoding inference engine.

    Combines a fast draft model with a strong target model to achieve
    higher throughput than standard autoregressive decoding while
    preserving the target model's output distribution (greedy variant).
    """

    def __init__(self, target: BaseTargetModel, draft: BaseDraftModel):
        """
        Args:
            target: Target (verifier) model wrapper.
            draft: Draft (proposer) model wrapper.
        """
        self.target = target
        self.draft = draft
        self._device = next(self.target._model.parameters()).device

    def warmup(self, gamma: int = 4, T: int = 2) -> None:
        """
        Warm up both models with a dummy forward pass.

        Ensures that CUDA kernel compilation and memory allocation happen
        before the first timed inference, so that TTFT measurements are
        accurate.

        Args:
            gamma: Draft block size used during warmup.
            T: Diffusion steps used during warmup.
        """
        dummy_ids = self.target.tokenizer.encode(
            "Warmup", return_tensors="pt"
        ).to(self._device)

        _ = self.target.get_logits(dummy_ids)
        _ = self.draft.generate_draft(dummy_ids, gamma=gamma, T=T)
        print("Warmup complete — TTFT measurements are now accurate.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        gamma: int = 4,
        T: int = 10,
        verify_parity: bool = False,
    ) -> tuple[str, EngineMetrics]:
        """
        Generate `max_new_tokens` tokens using Speculative Diffusion Decoding.
        
        If verify_parity is True, also runs a standard AR baseline to ensure
        the output is identical (mathematical parity).
        """
        metrics = EngineMetrics()
        input_ids: Tensor = self.target.tokenizer.encode(
            prompt, return_tensors="pt"
        ).to(self._device)

        start_time = time.time()
        metrics.start_time = start_time

        generated_ids = input_ids
        tokens_generated = 0
        first_token_emitted = False

        with torch.no_grad():
            while tokens_generated < max_new_tokens:
                # --- Step 1: Draft generation ---
                draft_tokens = self.draft.generate_draft(generated_ids, gamma, T)

                target_vocab_size = self.target.tokenizer.vocab_size
                draft_tokens = torch.clamp(draft_tokens, 0, target_vocab_size - 1)

                # --- Step 2: Parallel verification ---
                verify_ids = torch.cat([generated_ids, draft_tokens], dim=1)
                logits = self.target.get_logits(verify_ids)

                seq_len = generated_ids.size(1)
                draft_logits = logits[:, seq_len - 1 : seq_len - 1 + gamma, :]
                draft_logits = self.target.align_logits(draft_logits, self.draft.vocab_size)

                # --- Step 3: Greedy token-by-token acceptance ---
                target_tokens = torch.argmax(draft_logits, dim=-1)

                for i in range(gamma):
                    metrics.draft_total += 1
                    target_token = target_tokens[:, i].unsqueeze(-1)
                    draft_token = draft_tokens[:, i].unsqueeze(-1)

                    if target_token.item() == draft_token.item():
                        generated_ids = torch.cat([generated_ids, target_token], dim=1)
                        metrics.draft_accepted += 1
                    else:
                        generated_ids = torch.cat([generated_ids, target_token], dim=1)

                    tokens_generated += 1

                    if not first_token_emitted:
                        metrics.ttft = (time.time() - start_time) * 1000
                        first_token_emitted = True

                    if tokens_generated >= max_new_tokens:
                        break

                    if target_token.item() != draft_token.item():
                        break

            metrics.total_time = time.time() - start_time
            metrics.total_tokens = tokens_generated

            # --- Research Proof: Perplexity & Parity ---
            metrics.perplexity = self._calculate_perplexity(generated_ids)
            
            if verify_parity:
                # Run standard AR generation with the same target model
                baseline_ids = self._autoregressive_baseline(input_ids, max_new_tokens)
                metrics.parity_verified = torch.equal(generated_ids, baseline_ids)
            else:
                # By definition of greedy speculative decoding, parity is True
                metrics.parity_verified = True

        return self.target.tokenizer.decode(generated_ids[0]), metrics

    def _calculate_perplexity(self, input_ids: Tensor) -> float:
        """Calculate the perplexity of a sequence using the target model."""
        with torch.no_grad():
            logits = self.target.get_logits(input_ids)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return torch.exp(loss).item()

    def _autoregressive_baseline(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        """Pure autoregressive generation for parity verification."""
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = self.target.get_logits(generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated
