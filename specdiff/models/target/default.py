"""
Default target model wrapper — for causal LMs with no vocabulary quirks.

Suitable for any model whose vocabulary is already aligned with the draft
model (e.g. GPT-2 XL, GPT-Neo, GPT variants trained on OWT).
"""

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from specdiff.models.base import BaseTargetModel


class DefaultTargetModel(BaseTargetModel):
    """
    Generic wrapper for any HuggingFace causal language model.

    Vocabulary alignment is a no-op: assumes the target model shares
    the same token space as the draft model (e.g. both use GPT-2 tokens).
    """

    def __init__(self, model_name: str, device: str, dtype: str = "float16"):
        self.device = device
        dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype_map.get(dtype, torch.float16)
        ).to(device)
        self._model.eval()

    @property
    def tokenizer(self):
        return self._tokenizer

    def get_logits(self, input_ids: Tensor) -> Tensor:
        """Forward pass — returns logits of shape (batch, seq_len, vocab_size)."""
        with torch.no_grad():
            return self._model(input_ids).logits

    # align_logits is inherited as a no-op from BaseTargetModel
