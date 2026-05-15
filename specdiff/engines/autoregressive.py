"""
Standard autoregressive inference engine.

Generates tokens sequentially, one at a time, using greedy decoding.
Serves as the baseline against which SpecDiff speedup is measured.
"""

import time
import torch
from torch import Tensor

from specdiff.metrics.tracker import EngineMetrics
from specdiff.models.base import BaseTargetModel, BaseDraftModel


class AREngine:
    """
    Standard autoregressive generation engine (baseline).

    Calls the target model once per token in a sequential loop.
    Used as the performance reference for computing SpecDiff speedup.
    """

    def __init__(self, target: BaseTargetModel, draft: BaseDraftModel):
        """
        Args:
            target: Target model wrapper (provides tokenizer and logits).
            draft: Draft model wrapper (only used here to read vocab_size
                   for logit alignment).
        """
        self.target = target
        self.draft = draft

    def generate(self, prompt: str, max_new_tokens: int = 32) -> tuple[str, EngineMetrics]:
        """
        Generate `max_new_tokens` tokens autoregressively with greedy decoding.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Number of tokens to generate.

        Returns:
            Tuple of (generated_text, metrics).
        """
        metrics = EngineMetrics()
        input_ids: Tensor = self.target.tokenizer.encode(
            prompt, return_tensors="pt"
        ).to(next(self.target._model.parameters()).device)

        start_time = time.time()
        metrics.start_time = start_time
        generated_ids = input_ids

        for i in range(max_new_tokens):
            logits = self.target.get_logits(generated_ids)
            next_token_logits = logits[:, -1, :]

            # Align to draft vocabulary if needed (model-specific hook)
            next_token_logits = self.target.align_logits(
                next_token_logits.unsqueeze(1), self.draft.vocab_size
            ).squeeze(1)

            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if i == 0:
                metrics.ttft = (time.time() - start_time) * 1000  # ms

            metrics.total_tokens += 1

        metrics.total_time = time.time() - start_time
        metrics.perplexity = self._calculate_perplexity(generated_ids)
        metrics.parity_verified = True # Standard AR is its own reference
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
