#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize uniform_noise_diffusion evaluation metrics per checkpoint.
For each checkpoint:
  - Create step-specific folders with board-accuracy bar charts comparing
    algorithms for every (noise_ratio, editable_ratio) pair.
  - Create a "lines" folder with line charts (steps on x-axis) for each
    (noise_ratio, editable_ratio) pair comparing algorithms.
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


def plot_bars_per_step(
    df: pd.DataFrame,
    checkpoint_output: Path,
) -> None:
    steps = sorted(df["steps"].unique())
    for step in steps:
        step_dir = checkpoint_output / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        df_step = df[df["steps"] == step]

        grouped = df_step.groupby([
            "noise_ratio",
            "editable_ratio",
        ])
        for (noise_ratio, editable_ratio), subset in grouped:
            pivot_df = subset.groupby("algorithm")["board_accuracy"].mean()
            pivot = pivot_df.sort_index()
            if pivot.empty:
                continue

            algorithms = list(pivot.index)
            values = [float(pivot[alg]) for alg in algorithms]
            max_val = max(values)

            fig, ax = plt.subplots(figsize=(6, 4))
            colors = [
                "#5bc0de" if "self_conf" in alg else "#d9534f"
                for alg in algorithms
            ]
            bars = ax.bar(algorithms, values, color=colors)
            ax.set_ylim(0.0, max(0.05, max_val) * 1.25)
            ax.set_ylabel("Board accuracy")
            ax.set_title(
                "noise_ratio={}, editable_ratio={}".format(
                    noise_ratio,
                    editable_ratio,
                )
            )
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    val + max_val * 0.03,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            fig.tight_layout()
            filename = f"noise_{noise_ratio}_editable_{editable_ratio}.png"
            fig.savefig(step_dir / filename)
            plt.close(fig)


def plot_lines(df: pd.DataFrame, checkpoint_output: Path) -> None:
    lines_dir = checkpoint_output / "lines"
    lines_dir.mkdir(parents=True, exist_ok=True)

    for (noise_ratio, editable_ratio), subset in df.groupby([
        "noise_ratio",
        "editable_ratio",
    ]):
        pivot = (
            subset.groupby(["steps", "algorithm"])["board_accuracy"]
            .mean()
            .reset_index()
        )
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 4))
        for algorithm, algo_df in pivot.groupby("algorithm"):
            algo_df = algo_df.sort_values("steps")
            ax.plot(
                algo_df["steps"],
                algo_df["board_accuracy"],
                marker="o",
                label=algorithm,
            )
        ax.set_xlabel("Step")
        ax.set_ylabel("Board accuracy")
        ax.set_title(
            f"noise_ratio={noise_ratio}, editable_ratio={editable_ratio}"
        )
        ax.set_ylim(0.0, 1.01)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend()

        fig.tight_layout()
        filename = f"noise_{noise_ratio}_editable_{editable_ratio}.png"
        fig.savefig(lines_dir / filename)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot uniform_noise_diffusion metrics per checkpoint"
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
        help="Directory to save plots",
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
        subset = df[df["eval_mode"] == "uniform_noise_diffusion"].copy()
        if "step" in subset.columns and "steps" not in subset.columns:
            subset.rename(columns={"step": "steps"}, inplace=True)
        if subset.empty:
            continue

        required_columns = {
            "noise_ratio",
            "editable_ratio",
            "steps",
            "algorithm",
            "board_accuracy",
        }
        missing = required_columns - set(subset.columns)
        if missing:
            print(f"Skipping {rel_path} (missing columns: {missing})")
            continue

        checkpoint_output = output_root / rel_path
        plot_bars_per_step(subset, checkpoint_output)
        plot_lines(subset, checkpoint_output)

        metadata_path = (
            checkpoint_output / "uniform_noise_diffusion_sources.json"
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump({"csv_files": [str(p) for p in paths]}, f, indent=2)

    print(f"Plots saved to {output_root}")


if __name__ == "__main__":
    main()
