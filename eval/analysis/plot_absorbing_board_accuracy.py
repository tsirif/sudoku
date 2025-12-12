#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot absorbing evaluation board accuracy vs step for each checkpoint.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "eval_results_all"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent


def find_csv_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.csv"))


def load_results(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path))
        except Exception as err:
            print(f"Warning: failed to read {path}: {err}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_absorbing(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    subset = df[df["eval_mode"] == "absorbing"].copy()
    if subset.empty:
        return

    required_columns = {"mask_ratio", "step", "board_accuracy", "algorithm"}
    missing = required_columns - set(subset.columns)
    if missing:
        print(f"Skipping {output_dir} (missing columns: {missing})")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for mask_ratio, group in subset.groupby("mask_ratio"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for algorithm, algo_group in group.groupby("algorithm"):
            algo_group = algo_group.sort_values("step")
            ax.plot(
                algo_group["step"],
                algo_group["board_accuracy"],
                marker="o",
                label=algorithm,
            )
        ax.set_title(f"Mask ratio {mask_ratio}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Board accuracy")
        ax.set_ylim(0.0, 1.01)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend()

        filename = f"mask_{mask_ratio}.png"
        fig.tight_layout()
        fig.savefig(output_dir / filename)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot absorbing board-accuracy curves for each checkpoint"
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root directory containing eval_results",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory to save plots (defaults to eval/analysis)",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    if not results_root.exists():
        fallback = Path(__file__).resolve().parents[1] / "eval_results"
        if fallback.exists():
            print(
                f"Results root {results_root} not found; "
                f"falling back to {fallback}"
            )
            results_root = fallback
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    csv_paths = find_csv_files(results_root)
    if not csv_paths:
        print(f"No CSV files found under {results_root}")
        return

    by_checkpoint = {}
    for path in csv_paths:
        key = path.parent.relative_to(results_root)
        by_checkpoint.setdefault(key, []).append(path)

    for rel_path, paths in by_checkpoint.items():
        df = load_results(paths)
        if df.empty:
            continue
        checkpoint_output = output_root / rel_path
        plot_absorbing(df, checkpoint_output)

        metadata_path = checkpoint_output / "sources.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump({"csv_files": [str(p) for p in paths]}, f, indent=2)

    print(f"Plots saved to {output_root}")


if __name__ == "__main__":
    main()
