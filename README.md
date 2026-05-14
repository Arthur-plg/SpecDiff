# SpecDiff — Speculative Diffusion Decoding Framework

A modular benchmarking framework for **Speculative Diffusion Decoding (SpecDiff)** — an inference acceleration technique that combines a discrete diffusion draft model (MDLM) with a standard autoregressive target model to achieve higher throughput while preserving output quality.

> **Paper:** *Speculative Diffusion Decoding: Accelerating Language Generation through Diffusion* — [arXiv](https://arxiv.org/abs/2408.05636)

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    SpecDiff — Algorithm 1                   │
│                                                             │
│  While tokens_generated < max_new_tokens:                   │
│                                                             │
│    1. DRAFT   → MDLM generates γ candidate tokens           │
│                 via T reverse-diffusion steps               │
│                                                             │
│    2. VERIFY  → Target model scores [prompt + draft]        │
│                 in a SINGLE forward pass                    │
│                                                             │
│    3. ACCEPT  → Tokens accepted left-to-right (greedy)      │
│                 On first mismatch: use target token, retry  │
└─────────────────────────────────────────────────────────────┘
```

**Key insight:** The target model runs once per block (not once per token), yielding a decode speedup proportional to the draft acceptance rate α.

---

## Project Structure

```
SpecDiff_project/
│
├── specdiff/                    # Core Python package
│   ├── config.py                # YAML → ExperimentConfig dataclass
│   ├── registry.py              # ModelRegistry + factory functions
│   ├── models/
│   │   ├── base.py              # BaseTargetModel, BaseDraftModel (ABCs)
│   │   ├── target/
│   │   │   ├── default.py       # DefaultTargetModel (no vocab quirks)
│   │   │   └── redpajama.py     # RedPajama3BModel (vocab truncation)
│   │   └── draft/
│   │       ├── mdlm.py          # MDLMDraftModel (discrete diffusion)
│   │       └── ar.py            # SmallARDraftModel (AR ablation baseline)
│   ├── engines/
│   │   ├── autoregressive.py    # AREngine — sequential baseline
│   │   └── speculative.py       # SpeculativeEngine — SpecDiff algorithm
│   └── metrics/
│       └── tracker.py           # EngineMetrics (throughput, TTFT, α, ITL)
│
├── configs/                     # One YAML file = one experiment
│   ├── gpt2xl_mdlm.yaml
│   ├── gptneo_mdlm.yaml
│   └── redpajama_mdlm.yaml
│
├── experiments/
│   └── grid_search.py           # Sweep (γ, T) grid, save CSV results
│
├── results/                     # Auto-generated CSV outputs (gitignored)
├── mdlm/                        # MDLM source repo (kuleshov-group)
├── run.py                       # CLI entry point (single run)
└── pyproject.toml               # pip install -e . support
```

---

## Installation

```bash
git clone <your-repo-url>
cd SpecDiff_project
pip install -e .
```

---

## Quick Start

### Single run

```bash
# RedPajama-3B as target + MDLM as draft
python run.py --config configs/redpajama_mdlm.yaml

# GPT-Neo-1.3B (same vocab, no alignment needed)
python run.py --config configs/gptneo_mdlm.yaml

# Override hyperparameters without editing the YAML
python run.py --config configs/redpajama_mdlm.yaml --gamma 6 --T 3 --max_new_tokens 128
```

### Grid search

```bash
# Sweep γ ∈ {3,4,5} × T ∈ {3,5,10} on RedPajama
python experiments/grid_search.py \
  --config configs/redpajama_mdlm.yaml \
  --gammas 3 4 5 \
  --T_values 3 5 10
```

Results are saved as timestamped CSVs in `results/`.

---

## Metrics

| Metric | Description |
|--------|-------------|
| **Throughput** | `total_tokens / total_time` (tok/s) |
| **Decode Throughput** | Throughput excluding TTFT warmup — best for speedup comparisons |
| **TTFT** | Time To First Token (ms) |
| **ITL** | Average Inter-Token Latency (ms) |
| **α (Acceptance Rate)** | Fraction of draft tokens accepted by the target model |
| **Decode Speedup** | `decode_throughput_specdiff / decode_throughput_ar` |

---

## Adding a New Model

### New target model (same vocabulary as MDLM)
Just create a YAML:

```yaml
# configs/my_new_model.yaml
target:
  model_name: "EleutherAI/gpt-j-6b"
  wrapper: "DefaultTargetModel"   # no changes needed
  dtype: "float16"
draft:
  model_name: "kuleshov-group/mdlm-no_flashattn-fp32-owt"
  wrapper: "MDLMDraftModel"
device: "cuda"
```

### New target model (different vocabulary)
1. Create a wrapper in `specdiff/models/target/my_model.py`:

```python
from specdiff.models.target.default import DefaultTargetModel

class MyModel(DefaultTargetModel):
    def align_logits(self, logits, draft_vocab_size):
        return logits[:, :, :draft_vocab_size]  # or any remapping
```

2. Register it in `specdiff/registry.py`:

```python
from specdiff.models.target.my_model import MyModel
TARGET_REGISTRY["MyModel"] = MyModel
```

3. Reference it in your YAML: `wrapper: "MyModel"`

### New draft model
Implement `BaseDraftModel` (one required method: `generate_draft()`), register it in `DRAFT_REGISTRY`.

---

## Supported Configurations

| Target Model | Draft Model | Wrapper | Notes |
|---|---|---|---|
| `gpt2-xl` | `mdlm-owt` | `DefaultTargetModel` | Reference config |
| `gpt-neo-1.3B` | `mdlm-owt` | `DefaultTargetModel` | Same GPT-2 vocab |
| `RedPajama-3B` | `mdlm-owt` | `RedPajama3BModel` | Vocab truncation applied |

---

## Implementation Notes

### flash_attn workaround
MDLM's HuggingFace model file contains a hard `import flash_attn` that cannot be disabled via config. The `MDLMDraftModel` wrapper handles this by:
1. Disabling HuggingFace's AST import scanner
2. Injecting a `MagicMock` for `flash_attn`
3. Replacing the RoPE kernel with a pure PyTorch implementation

This is fully encapsulated in `specdiff/models/draft/mdlm.py` and requires no user action.
