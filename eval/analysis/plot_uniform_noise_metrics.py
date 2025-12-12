#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bar plots comparing clean vs noise token metrics for uniform_noise_only
evaluations.
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


def plot_uniform_noise(df: pd.DataFrame, output_dir: Path) -> None:
    subset = df[df["eval_mode"] == "uniform_noise_only"].copy()
    if subset.empty:
        return

    required = {
        "noise_ratio",
        "noise_token_confidence",
        "clean_token_confidence",
    }
    missing = required - set(subset.columns)
    if missing:
        print(f"Skipping {output_dir} (missing columns: {missing})")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for noise_ratio, group in subset.groupby("noise_ratio"):
        fig, ax = plt.subplots(figsize=(6, 4))

        noise_mean = float(group["noise_token_confidence"].mean())
        clean_mean = float(group["clean_token_confidence"].mean())
        labels = ["Noise", "Clean"]
        values = [noise_mean, clean_mean]
        colors = ["#d9534f", "#5bc0de"]

        bars = ax.bar(labels, values, color=colors)

        # Dynamic y scaling for better aesthetics
        max_val = max(values)
        ax.set_ylim(0.0, max(0.05, max_val) * 1.25)

        # Annotate bar values
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + max_val * 0.03,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Ratio annotation: clean vs noise
        ratio_text = "∞"
        if noise_mean > 1e-8:
            ratio_text = f"{(clean_mean / noise_mean):.2f}x"
        ax.text(
            0.5,
            max(0.05, max_val) * 1.14,
            f"clean/noise = {ratio_text}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

        ax.set_title(f"Confidence (noise_ratio={noise_ratio})")
        ax.set_ylabel("p(token = current)")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

        fig.tight_layout()
        filename = f"uniform_noise_confidence_ratio_{noise_ratio}.png"
        fig.savefig(output_dir / filename)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot uniform noise metrics per checkpoint"
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
        checkpoint_output = output_root / rel_path
        plot_uniform_noise(df, checkpoint_output)

        metadata_path = checkpoint_output / "uniform_noise_sources.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump({"csv_files": [str(p) for p in paths]}, f, indent=2)

    print(f"Plots saved to {output_root}")


if __name__ == "__main__":
    main()
