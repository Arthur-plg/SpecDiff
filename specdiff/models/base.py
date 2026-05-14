"""
Abstract base classes for target and draft models.

Every new model integration must inherit from one of these two ABCs
and implement the required interface methods. The inference engines
only depend on these abstractions — never on concrete implementations.
"""

from abc import ABC, abstractmethod
import torch
from torch import Tensor


class BaseTargetModel(ABC):
    """
    Abstract wrapper for the target (verifier) language model.

    The target model is a standard causal LM that verifies draft tokens
    and produces the final accepted sequence.
    """

    def __init__(self, model_name: str, device: str, dtype: str = "float16"):
        """All target model wrappers must accept these three constructor args."""
        ...

    @abstractmethod
    def get_logits(self, input_ids: Tensor) -> Tensor:
        """
        Run a forward pass and return raw logits.

        Args:
            input_ids: Token ids of shape (batch, seq_len).

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size).
        """
        raise NotImplementedError

    def align_logits(self, logits: Tensor, draft_vocab_size: int) -> Tensor:
        """
        Optional vocabulary alignment hook.

        Override this method when the target model's vocabulary does not
        match the draft model's vocabulary (e.g. vocab truncation for
        RedPajama-3B vs. MDLM-OWT which both use GPT-2 tokens).

        Default implementation: no-op (vocabularies already aligned).

        Args:
            logits: Raw logits of shape (batch, seq_len, target_vocab_size).
            draft_vocab_size: Vocabulary size of the draft model.

        Returns:
            Aligned logits of shape (batch, seq_len, draft_vocab_size).
        """
        return logits

    @property
    @abstractmethod
    def tokenizer(self):
        """The tokenizer associated with this target model."""
        raise NotImplementedError


class BaseDraftModel(ABC):
    """
    Abstract wrapper for the draft (proposer) model.

    The draft model generates a block of `gamma` candidate tokens
    conditioned on a prompt. These candidates are then verified in
    parallel by the target model (Algorithm 1 of SpecDiff).
    """

    def __init__(self, model_name: str, device: str):
        """All draft model wrappers must accept these two constructor args."""
        ...

    @abstractmethod
    def generate_draft(self, prompt_ids: Tensor, gamma: int, T: int) -> Tensor:
        """
        Generate a block of `gamma` draft tokens.

        Args:
            prompt_ids: Prompt token ids of shape (batch, prompt_len).
            gamma: Number of tokens to draft (block size).
            T: Number of diffusion/generation steps (model-dependent).

        Returns:
            Draft token ids of shape (batch, gamma).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Vocabulary size of the draft model."""
        raise NotImplementedError
