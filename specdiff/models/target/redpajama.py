"""
RedPajama-3B target model wrapper.

RedPajama-INCITE-Base-3B-v1 uses a vocabulary of size 50432, while the
MDLM draft model (trained on OpenWebText) uses GPT-2's vocabulary of
size 50257. To make token-level comparison possible during speculative
verification, we truncate the target logits to the draft vocabulary size.

This wrapper encapsulates that model-specific quirk so the SpecDiff
algorithm never needs to know about it.
"""

from torch import Tensor

from specdiff.models.target.default import DefaultTargetModel


class RedPajama3BModel(DefaultTargetModel):
    """
    Target model wrapper for togethercomputer/RedPajama-INCITE-Base-3B-v1.

    Overrides `align_logits` to truncate the logit dimension from the
    RedPajama vocabulary size (50432) down to the draft model's GPT-2
    vocabulary size (50257), enabling greedy verification across the
    shared token space.
    """

    def align_logits(self, logits: Tensor, draft_vocab_size: int) -> Tensor:
        """
        Truncate target logits to match the draft model vocabulary.

        Args:
            logits: Shape (batch, seq_len, 50432).
            draft_vocab_size: Expected to be 50257 for MDLM-OWT.

        Returns:
            Truncated logits of shape (batch, seq_len, draft_vocab_size).
        """
        return logits[:, :, :draft_vocab_size]
