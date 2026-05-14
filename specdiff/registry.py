"""
Model registry — maps wrapper name strings to concrete classes.

The registry is the single place that connects YAML config strings
(e.g. "RedPajama3BModel") to their Python implementations. To add a
new model, register it here — no other file needs to change.
"""

from specdiff.models.target.default import DefaultTargetModel
from specdiff.models.target.redpajama import RedPajama3BModel
from specdiff.models.draft.mdlm import MDLMDraftModel
from specdiff.models.draft.ar import SmallARDraftModel
from specdiff.config import ExperimentConfig
from specdiff.models.base import BaseTargetModel, BaseDraftModel


# --- Registries ---

TARGET_REGISTRY: dict[str, type[BaseTargetModel]] = {
    "DefaultTargetModel": DefaultTargetModel,
    "RedPajama3BModel":   RedPajama3BModel,
}

DRAFT_REGISTRY: dict[str, type[BaseDraftModel]] = {
    "MDLMDraftModel":    MDLMDraftModel,
    "SmallARDraftModel": SmallARDraftModel,
}


# --- Factory functions ---

def build_target(config: ExperimentConfig) -> BaseTargetModel:
    """
    Instantiate the target model specified in the experiment config.

    Args:
        config: Parsed experiment configuration.

    Returns:
        Initialized BaseTargetModel instance.

    Raises:
        ValueError: If the wrapper name is not registered.
    """
    wrapper_name = config.target.wrapper
    if wrapper_name not in TARGET_REGISTRY:
        raise ValueError(
            f"Unknown target wrapper '{wrapper_name}'. "
            f"Available: {list(TARGET_REGISTRY.keys())}"
        )
    cls = TARGET_REGISTRY[wrapper_name]
    print(f"[Registry] Loading target model: {config.target.model_name} ({wrapper_name})")
    return cls(
        model_name=config.target.model_name,
        device=config.device,
        dtype=config.target.dtype,
    )


def build_draft(config: ExperimentConfig) -> BaseDraftModel:
    """
    Instantiate the draft model specified in the experiment config.

    Args:
        config: Parsed experiment configuration.

    Returns:
        Initialized BaseDraftModel instance.

    Raises:
        ValueError: If the wrapper name is not registered.
    """
    wrapper_name = config.draft.wrapper
    if wrapper_name not in DRAFT_REGISTRY:
        raise ValueError(
            f"Unknown draft wrapper '{wrapper_name}'. "
            f"Available: {list(DRAFT_REGISTRY.keys())}"
        )
    cls = DRAFT_REGISTRY[wrapper_name]
    print(f"[Registry] Loading draft model: {config.draft.model_name} ({wrapper_name})")
    return cls(
        model_name=config.draft.model_name,
        device=config.device,
    )
