"""
MDLM draft model — discrete diffusion-based token proposer.

Wraps the MDLM (Masked Diffusion Language Model) from:
  kuleshov-group/mdlm-no_flashattn-fp32-owt

Key implementation notes:
  - flash_attn is mocked at import time because the HuggingFace model file
    contains a hard `import flash_attn` that cannot be disabled via config.
  - The RoPE (Rotary Position Embeddings) kernel from flash_attn is replaced
    with a pure PyTorch implementation, since the mocked module returns empty
    objects that break tensor shapes.
"""

import sys
import torch
from torch import Tensor
from unittest.mock import MagicMock
from transformers import AutoModelForMaskedLM

from specdiff.models.base import BaseDraftModel


def _patch_flash_attn() -> None:
    """
    Suppress the hard `import flash_attn` inside the MDLM model file.

    HuggingFace scans model source files (AST) for imports before executing
    them. We disable this scanner and inject a MagicMock so that the
    `import flash_attn` line becomes a no-op at runtime.
    """
    import transformers.dynamic_module_utils

    # Disable HuggingFace import scanner
    transformers.dynamic_module_utils.check_imports = lambda filename: []

    # Inject mock modules so `import flash_attn` succeeds silently
    if "flash_attn" not in sys.modules:
        sys.modules["flash_attn"] = MagicMock()
        sys.modules["flash_attn.layers"] = MagicMock()
        sys.modules["flash_attn.layers.rotary"] = MagicMock()


def _patch_rope(model) -> None:
    """
    Replace the flash_attn RoPE kernel with a pure PyTorch implementation.

    The MDLM codebase calls `apply_rotary_pos_emb` from flash_attn even when
    flash attention is disabled. Since we mocked flash_attn, the kernel returns
    a MagicMock object which causes shape errors in subsequent tensor ops.
    We inject a pure PyTorch equivalent directly into the model's module.
    """

    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(qkv: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        # Reconstruct full cos/sin tensors from the halved format MDLM uses
        cos_half = cos[0, :, 0, 0, : cos.shape[-1] // 2]
        sin_half = sin[0, :, 0, 0, : sin.shape[-1] // 2]
        cos_full = torch.cat([cos_half, cos_half], dim=-1).unsqueeze(0).unsqueeze(2)
        sin_full = torch.cat([sin_half, sin_half], dim=-1).unsqueeze(0).unsqueeze(2)

        q = qkv[:, :, 0]
        k = qkv[:, :, 1]

        q_embed = (q * cos_full) + (_rotate_half(q) * sin_full)
        k_embed = (k * cos_full) + (_rotate_half(k) * sin_full)

        qkv_out = qkv.clone()
        qkv_out[:, :, 0] = q_embed
        qkv_out[:, :, 1] = k_embed
        return qkv_out

    # Inject our PyTorch implementation into the model's own module namespace
    module = sys.modules[model.__module__]
    module.apply_rotary_pos_emb = _apply_rotary_pos_emb


class MDLMDraftModel(BaseDraftModel):
    """
    Draft model based on Masked Diffusion Language Model (MDLM).

    Generates a block of `gamma` tokens via `T` steps of reverse diffusion
    conditioned on a prompt using the absorbing-state (mask) noise schedule.

    Reference: Sahoo et al., "Simple and Effective Masked Diffusion Language
    Models" (NeurIPS 2024) — https://arxiv.org/abs/2406.07524
    """

    # Special token index used as the MASK token (matches GPT-2 vocab)
    MASK_INDEX = 50257
    # Small epsilon to avoid log(0) in the noise schedule
    EPS = 1e-3

    def __init__(self, model_name: str, device: str):
        self.device = device

        _patch_flash_attn()

        self._model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(device)
        self._model.eval()

        _patch_rope(self._model)

        self._vocab_size = self._model.config.vocab_size  # typically 50257 or 50258

    # ------------------------------------------------------------------
    # BaseDraftModel interface
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def generate_draft(self, prompt_ids: Tensor, gamma: int, T: int) -> Tensor:
        """
        Generate `gamma` draft tokens via `T` reverse-diffusion steps.

        Algorithm:
          1. Build input: [PROMPT_TOKENS] + [MASK] * gamma
          2. Run T denoising steps using the log-linear noise schedule.
          3. After each step, force the prompt region back to its original
             tokens (conditional infilling constraint).
          4. Return only the newly generated `gamma` tokens.

        Args:
            prompt_ids: Shape (batch, prompt_len).
            gamma: Number of tokens to generate.
            T: Number of diffusion denoising steps.

        Returns:
            Draft token ids of shape (batch, gamma).
        """
        bsz = prompt_ids.size(0)
        prompt_len = prompt_ids.size(1)

        # Initialize sequence with prompt + masked suffix
        x = torch.cat(
            [
                prompt_ids,
                torch.full((bsz, gamma), self.MASK_INDEX, dtype=torch.long, device=self.device),
            ],
            dim=1,
        )

        timesteps = torch.linspace(1, 1e-5, T + 1, device=self.device)
        dt = (1 - 1e-5) / T

        p_x0_cache = None

        with torch.no_grad():
            for i in range(T):
                t_tensor = timesteps[i] * torch.ones(bsz, 1, device=self.device)
                p_x0_cache, x_next = self._ddpm_caching_update(x, t_tensor, dt, p_x0=p_x0_cache)
                # Enforce conditional infilling: keep prompt tokens unchanged
                x_next[:, :prompt_len] = prompt_ids
                x = x_next

        return x[:, prompt_len:]

    # ------------------------------------------------------------------
    # Internal diffusion helpers
    # ------------------------------------------------------------------

    def _noise_total(self, t: Tensor) -> Tensor:
        """Log-linear noise schedule: σ(t) = -log(1 - (1-ε)·t)."""
        return -torch.log1p(-(1 - self.EPS) * t)

    def _sample_categorical(self, categorical_probs: Tensor) -> Tensor:
        """Gumbel-max sampling for discrete categorical distributions."""
        gumbel_noise = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_noise).argmax(dim=-1)

    def _subs_parameterization(self, logits: Tensor, xt: Tensor) -> Tensor:
        """
        Absorbing-state substitution parameterization for discrete diffusion.

        Masks out the MASK token from the predicted distribution and enforces
        that already-unmasked tokens remain unchanged (identity mapping).
        """
        # Suppress the mask token from predictions
        logits[:, :, self.MASK_INDEX] += -1_000_000.0
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        # For already-revealed tokens, lock prediction to the current token
        unmasked = xt != self.MASK_INDEX
        logits[unmasked] = -1_000_000.0
        logits[unmasked, xt[unmasked]] = 0.0

        return logits

    def _ddpm_caching_update(
        self,
        x: Tensor,
        t_tensor: Tensor,
        dt: float,
        p_x0: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Single DDPM denoising step with p(x0) caching.

        Caching avoids a redundant forward pass: if p(x0) was already computed
        in the previous step and no tokens changed, we reuse it directly.

        Args:
            x: Current (partially masked) sequence, shape (batch, seq_len).
            t_tensor: Current timestep, shape (batch, 1).
            dt: Step size (scalar).
            p_x0: Cached x0 predictions from the previous step (or None).

        Returns:
            (updated p_x0, next sequence x_s)
        """
        t_squeeze = t_tensor.squeeze(-1) if t_tensor.ndim > 1 else t_tensor
        move_chance_t = t_squeeze[:, None, None]
        move_chance_s = (t_squeeze - dt)[:, None, None]

        if p_x0 is None:
            sigma_t = self._noise_total(t_tensor)
            sigma_t_sq = sigma_t.squeeze(-1) if sigma_t.ndim > 1 else sigma_t

            out = self._model(x, sigma_t_sq)
            logits = out.logits if hasattr(out, "logits") else out
            if isinstance(logits, tuple):
                logits = logits[0]

            log_p_x0 = self._subs_parameterization(logits, x)
            p_x0 = log_p_x0.exp()

        # Compute transition probabilities for the x_t → x_s step
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.MASK_INDEX] = move_chance_s[:, :, 0]

        x_s = self._sample_categorical(q_xs)

        # Copy already-unmasked tokens from x (they should not change)
        copy_flag = (x != self.MASK_INDEX).to(x.dtype)
        x_s = copy_flag * x + (1 - copy_flag) * x_s

        return p_x0, x_s
