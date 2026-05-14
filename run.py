"""
Main CLI entry point for single-run inference.

Usage:
    python run.py --config configs/redpajama_mdlm.yaml
    python run.py --config configs/gptneo_mdlm.yaml --gamma 6 --T 3
    python run.py --config configs/gpt2xl_mdlm.yaml --max_new_tokens 128 --prompt "Once upon a time"

On Google Colab (redirect results to Drive):
    !python run.py --config configs/redpajama_mdlm.yaml --results_dir /content/drive/MyDrive/specdiff_results
"""

import argparse
import os
import datetime
import pandas as pd
from pathlib import Path

from specdiff import load_config, build_target, build_draft, AREngine, SpeculativeEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpecDiff single-run benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML experiment config")
    parser.add_argument("--gamma", type=int, default=None, help="Override draft block size")
    parser.add_argument("--T", type=int, default=None, help="Override diffusion steps")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Override token budget")
    parser.add_argument("--prompt", type=str, default=None, help="Override generation prompt")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Override results directory (e.g. /content/drive/MyDrive/specdiff_results)")
    return parser.parse_args()


def save_results(run_data: list[dict], results_dir: str) -> str:
    """Append results to a CSV file in the results directory."""
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "specdiff_results.csv")
    df = pd.DataFrame(run_data)
    if Path(csv_path).exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, mode="w", header=True, index=False)
    return csv_path


def main():
    args = parse_args()

    # Load config and apply CLI overrides
    config = load_config(args.config)
    if args.gamma is not None:
        config.gamma = args.gamma
    if args.T is not None:
        config.T = args.T
    if args.max_new_tokens is not None:
        config.max_new_tokens = args.max_new_tokens
    if args.prompt is not None:
        config.prompt = args.prompt
    if args.results_dir is not None:
        config.results_dir = args.results_dir

    print(f"\n{'='*60}")
    print(f"  SpecDiff Benchmark — {args.config}")
    print(f"  Target : {config.target.model_name} ({config.target.wrapper})")
    print(f"  Draft  : {config.draft.model_name} ({config.draft.wrapper})")
    print(f"  gamma={config.gamma}, T={config.T}, max_new_tokens={config.max_new_tokens}")
    print(f"{'='*60}\n")

    # Build models from registry
    target = build_target(config)
    draft = build_draft(config)

    # Build engines
    ar_engine = AREngine(target, draft)
    spec_engine = SpeculativeEngine(target, draft)
    spec_engine.warmup(gamma=config.gamma, T=config.T)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = []

    # --- Baseline: Standard AR ---
    print(f"\n[1/2] Running Standard AR ({config.max_new_tokens} tokens)...")
    text_ar, metrics_ar = ar_engine.generate(config.prompt, config.max_new_tokens)
    print(
        f"  Throughput:        {metrics_ar.get_throughput():.2f} tok/s\n"
        f"  Decode Throughput: {metrics_ar.get_decode_throughput():.2f} tok/s\n"
        f"  TTFT:              {metrics_ar.ttft:.2f} ms"
    )
    results.append({
        "timestamp": timestamp,
        "config": args.config,
        "method": "Standard AR",
        "target_model": config.target.model_name,
        "draft_model": config.draft.model_name,
        "gamma": 0,
        "T_steps": 0,
        **metrics_ar.to_dict(),
    })

    # --- SpecDiff ---
    print(f"\n[2/2] Running SpecDiff (gamma={config.gamma}, T={config.T})...")
    text_spec, metrics_spec = spec_engine.generate(
        config.prompt, config.max_new_tokens, config.gamma, config.T
    )
    decode_speedup = (
        metrics_spec.get_decode_throughput() / metrics_ar.get_decode_throughput()
        if metrics_ar.get_decode_throughput() > 0
        else 0
    )
    print(
        f"  Throughput:        {metrics_spec.get_throughput():.2f} tok/s\n"
        f"  Decode Throughput: {metrics_spec.get_decode_throughput():.2f} tok/s\n"
        f"  Acceptance rate:   {metrics_spec.get_acceptance_rate()*100:.2f}%\n"
        f"  Decode Speedup:    {decode_speedup:.2f}x"
    )
    results.append({
        "timestamp": timestamp,
        "config": args.config,
        "method": "SpecDiff",
        "target_model": config.target.model_name,
        "draft_model": config.draft.model_name,
        "gamma": config.gamma,
        "T_steps": config.T,
        **metrics_spec.to_dict(),
    })

    # Save results
    csv_path = save_results(results, config.results_dir)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
