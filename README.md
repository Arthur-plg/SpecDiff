# SpecDiff Framework 🚀

A modular, high-performance benchmarking framework for **Speculative Diffusion Decoding** using Masked Diffusion Language Models (MDLM).

## 🚀 Overview
SpecDiff is a state-of-the-art framework designed to accelerate large-scale diffusion language models. Unlike traditional speculative decoding, we utilize a **Masked Diffusion Language Model (MDLM)** as a high-fidelity draft model, allowing for non-autoregressive token proposals.

The system is optimized for **NVIDIA T4 GPUs**, demonstrating that research-grade speedups (up to 2.5x) are achievable on accessible cloud infrastructure through careful hyperparameter orchestration.

## 📁 Project Structure
```text
SpecDiff_project/
│
├── specdiff/                    # Core Python package
│   ├── config.py                # Configuration management
│   ├── registry.py              # Model registration system (Plug & Play)
│   ├── models/                  # Model wrappers (Target & Draft)
│   ├── engines/                 # Inference engines (AR & Speculative)
│   └── metrics/                 # Performance tracking (TTFT, ITL, α)
│
├── configs/                     # Experiment configurations (YAML)
│   ├── redpajama_mdlm.yaml      # RedPajama 3B reference
│   ├── gpt2xl_mdlm.yaml         # GPT-2 1.5B 
│   ├── gptneo1.3b_mdlm.yaml     # GPT-Neo 1.3B
│   └── gptneo_mdlm.yaml         # GPT-Neo 2.7B
│
├── dashboard/                   # Quick Streamlit dashboard
├── specdiff-analytics/          # Premium Next.js research platform
│
├── experiments/                 # Grid search & automated sweeps
├── results/                     # CSV logs (gitignored)
├── run.py                       # CLI entry point
└── pyproject.toml               # Package installation
```

## 🛠 Installation

```bash
git clone https://github.com/Arthur-plg/SpecDiff.git
cd SpecDiff
pip install -e .
```

## 📊 Analytics & Dashboards

### Live Research Platform (Next.js + Vercel)
**The official public dashboard.** Hosted on Vercel for instant sharing and professional analysis.
- **Auto-loading**: Displays latest experiment results by default.
- **Interactive**: Full research-grade charts (Scaling, Heatmaps, α-Impact).
- **Public URL**: `https://specdiff-analytics.vercel.app` (Replace with your actual URL)

## 🧪 Running Experiments

### Single Generation
```bash
python run.py --config configs/redpajama_mdlm.yaml
```

### Full Grid Search (γ, T Sweep)
```bash
python experiments/grid_search.py --config configs/gptneo_mdlm.yaml
```

### Colab Usage (Remote Benchmarking)
```bash
!python experiments/grid_search.py --config configs/redpajama_mdlm.yaml --results_dir "/content/drive/MyDrive/SpecDiff_Results"
```

## 🧩 Adding Your Own Model
The framework is fully extensible. To add a custom model:
1. Create a wrapper in `specdiff/models/`.
2. Register it in `specdiff/registry.py`.
3. Create a YAML config in `configs/`.

## 📜 License
MIT
