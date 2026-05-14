"""
Grid search experiment over (gamma, T) hyperparameters.

Runs AR baseline once, then sweeps the full (gamma, T) grid with SpecDiff.
All results are saved to a single CSV in the results directory.

Usage:
    python experiments/grid_search.py --config configs/redpajama_mdlm.yaml
    python experiments/grid_search.py --config configs/gptneo_mdlm.yaml --gammas 3 4 5 6 --T_values 3 5 10
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import datetime
import pandas as pd
from pathlib import Path
from itertools import product

from specdiff import load_config, build_target, build_draft, AREngine, SpeculativeEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpecDiff grid search experiment")
    parser.add_argument("--config", required=True, help="Path to YAML experiment config")
    parser.add_argument(
        "--gammas", nargs="+", type=int, default=[3, 4, 5],
        help="List of gamma (draft block size) values to sweep"
    )
    parser.add_argument(
        "--T_values", nargs="+", type=int, default=[3, 5, 10],
        help="List of T (diffusion steps) values to sweep"
    )
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Override results directory (e.g. /content/drive/MyDrive/specdiff_results)")
    return parser.parse_args()


def save_results(results: list[dict], results_dir: str) -> str:
    """Save all grid search results to a timestamped CSV."""
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"grid_search_{timestamp}.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    return csv_path


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.max_new_tokens is not None:
        config.max_new_tokens = args.max_new_tokens
    if args.prompt is not None:
        config.prompt = args.prompt
    if args.results_dir is not None:
        config.results_dir = args.results_dir

    grid = list(product(args.gammas, args.T_values))
    total_runs = 1 + len(grid)  # AR baseline + all (gamma, T) combinations

    print(f"\n{'='*60}")
    print(f"  SpecDiff Grid Search — {args.config}")
    print(f"  Target  : {config.target.model_name}")
    print(f"  Draft   : {config.draft.model_name}")
    print(f"  Grid    : gamma={args.gammas} × T={args.T_values} ({len(grid)} combos)")
    print(f"  Total   : {total_runs} runs")
    print(f"{'='*60}\n")

    target = build_target(config)
    draft = build_draft(config)

    ar_engine = AREngine(target, draft)
    spec_engine = SpeculativeEngine(target, draft)
    # Warmup with the smallest gamma/T to minimize warmup overhead
    spec_engine.warmup(gamma=min(args.gammas), T=min(args.T_values))

    results = []
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Baseline: Standard AR ---
    print(f"[1/{total_runs}] Standard AR baseline...")
    _, metrics_ar = ar_engine.generate(config.prompt, config.max_new_tokens)
    ar_decode_tp = metrics_ar.get_decode_throughput()
    print(f"  Decode Throughput: {ar_decode_tp:.2f} tok/s | TTFT: {metrics_ar.ttft:.2f} ms")

    results.append({
        "timestamp": run_timestamp,
        "config": args.config,
        "method": "Standard AR",
        "target_model": config.target.model_name,
        "draft_model": config.draft.model_name,
        "gamma": 0,
        "T_steps": 0,
        "decode_speedup": 1.0,
        **metrics_ar.to_dict(),
    })

    # --- Grid Search ---
    for run_idx, (gamma, T) in enumerate(grid, start=2):
        print(f"\n[{run_idx}/{total_runs}] SpecDiff — gamma={gamma}, T={T}")
        _, metrics_spec = spec_engine.generate(config.prompt, config.max_new_tokens, gamma, T)

        decode_speedup = (
            metrics_spec.get_decode_throughput() / ar_decode_tp
            if ar_decode_tp > 0 else 0
        )
        print(
            f"  Decode Throughput: {metrics_spec.get_decode_throughput():.2f} tok/s  |  "
            f"Acceptance: {metrics_spec.get_acceptance_rate()*100:.1f}%  |  "
            f"Speedup: {decode_speedup:.2f}x"
        )

        results.append({
            "timestamp": run_timestamp,
            "config": args.config,
            "method": "SpecDiff",
            "target_model": config.target.model_name,
            "draft_model": config.draft.model_name,
            "gamma": gamma,
            "T_steps": T,
            "decode_speedup": decode_speedup,
            **metrics_spec.to_dict(),
        })

    csv_path = save_results(results, config.results_dir)
    print(f"\n{'='*60}")
    print(f"  Grid search complete — {len(grid)} SpecDiff runs")
    print(f"  Results saved to: {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
