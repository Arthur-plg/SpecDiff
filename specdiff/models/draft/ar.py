"""
AR-based draft model — autoregressive token proposer.

A lightweight alternative to MDLM for ablation studies: uses a small
causal LM (e.g. GPT-2) to generate draft tokens autoregressively.
The `T` parameter is not applicable here and is silently ignored.
"""

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM

from specdiff.models.base import BaseDraftModel


class SmallARDraftModel(BaseDraftModel):
    """
    Autoregressive draft model backed by any causal language model.

    Generates `gamma` tokens greedily (argmax decoding) in a sequential
    AR loop. Intended for baseline comparisons against diffusion-based
    draft models like MDLM.

    Note:
        The `T` parameter (diffusion steps) is ignored — AR generation
        has no equivalent concept.
    """

    def __init__(self, model_name: str, device: str):
        self.device = device
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
        self._model.eval()
        self._vocab_size = self._model.config.vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def generate_draft(self, prompt_ids: Tensor, gamma: int, T: int) -> Tensor:
        """
        Greedily generate `gamma` draft tokens autoregressively.

        Args:
            prompt_ids: Shape (batch, prompt_len).
            gamma: Number of tokens to generate.
            T: Unused — included for interface compatibility.

        Returns:
            Draft token ids of shape (batch, gamma).
        """
        generated = prompt_ids

        with torch.no_grad():
            for _ in range(gamma):
                outputs = self._model(generated)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

        # Return only the newly generated tokens
        return generated[:, prompt_ids.size(1):]
