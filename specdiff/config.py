"""
YAML-based experiment configuration system.

A single YAML file fully describes an experiment: which models to load,
which wrappers to use, and what generation hyperparameters to apply.
This makes experiments reproducible and shareable without code changes.

Example usage:
    config = load_config("configs/redpajama_mdlm.yaml")
    target = build_target(config)
    draft  = build_draft(config)
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TargetConfig:
    """Configuration for the target (verifier) model."""
    model_name: str
    wrapper: str = "DefaultTargetModel"
    dtype: str = "float16"


@dataclass
class DraftConfig:
    """Configuration for the draft (proposer) model."""
    model_name: str
    wrapper: str = "MDLMDraftModel"


@dataclass
class ExperimentConfig:
    """Full experiment specification loaded from a YAML file."""
    target: TargetConfig
    draft: DraftConfig
    device: str = "cuda"
    gamma: int = 4
    T: int = 10
    max_new_tokens: int = 256
    prompt: str = "The future of artificial intelligence is"
    results_dir: str = "results"


def load_config(path: str | Path) -> ExperimentConfig:
    """
    Parse a YAML config file into an ExperimentConfig dataclass.

    Args:
        path: Path to the YAML config file.

    Returns:
        Fully populated ExperimentConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If required fields are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return ExperimentConfig(
        target=TargetConfig(**raw["target"]),
        draft=DraftConfig(**raw["draft"]),
        device=raw.get("device", "cuda"),
        gamma=raw.get("gamma", 4),
        T=raw.get("T", 10),
        max_new_tokens=raw.get("max_new_tokens", 256),
        prompt=raw.get("prompt", "The future of artificial intelligence is"),
        results_dir=raw.get("results_dir", "results"),
    )
