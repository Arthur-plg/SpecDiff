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
    ) -> tuple[str, EngineMetrics]:
        """
        Generate `max_new_tokens` tokens using Speculative Diffusion Decoding.

        Algorithm 1 (greedy variant):
          While tokens_generated < max_new_tokens:
            1. Draft: generate `gamma` candidate tokens via the draft model.
            2. Verify: run target model over [prompt + draft] in one forward pass.
            3. Accept: scan draft left-to-right; accept if target agrees.
            4. Reject: on first mismatch, use target's token and restart.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Total number of tokens to generate.
            gamma: Draft block size (number of tokens proposed per round).
            T: Number of diffusion steps per draft round (draft model param).

        Returns:
            Tuple of (generated_text, metrics).
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

                # Safety Clip: Ensure draft tokens are within target model's vocab range
                # (Prevents CUDA out-of-bounds if MDLM proposes its MASK token)
                target_vocab_size = self.target.tokenizer.vocab_size
                draft_tokens = torch.clamp(draft_tokens, 0, target_vocab_size - 1)

                # --- Step 2: Parallel verification ---
                verify_ids = torch.cat([generated_ids, draft_tokens], dim=1)
                logits = self.target.get_logits(verify_ids)

                # Extract logits corresponding to the draft positions
                seq_len = generated_ids.size(1)
                draft_logits = logits[:, seq_len - 1 : seq_len - 1 + gamma, :]

                # Apply model-specific vocabulary alignment (e.g. truncation)
                draft_logits = self.target.align_logits(draft_logits, self.draft.vocab_size)

                # --- Step 3: Greedy token-by-token acceptance ---
                target_tokens = torch.argmax(draft_logits, dim=-1)  # (batch, gamma)

                for i in range(gamma):
                    metrics.draft_total += 1
                    target_token = target_tokens[:, i].unsqueeze(-1)
                    draft_token = draft_tokens[:, i].unsqueeze(-1)

                    if target_token.item() == draft_token.item():
                        # Accepted: append draft token and continue
                        generated_ids = torch.cat([generated_ids, target_token], dim=1)
                        metrics.draft_accepted += 1
                    else:
                        # Rejected: append target correction and break out of block
                        generated_ids = torch.cat([generated_ids, target_token], dim=1)

                    tokens_generated += 1

                    if not first_token_emitted:
                        metrics.ttft = (time.time() - start_time) * 1000
                        first_token_emitted = True

                    if tokens_generated >= max_new_tokens:
                        break

                    # Stop processing remaining draft tokens after a rejection
                    if target_token.item() != draft_token.item():
                        break

        metrics.total_time = time.time() - start_time
        metrics.total_tokens = tokens_generated
        return self.target.tokenizer.decode(generated_ids[0]), metrics
