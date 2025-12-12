#!/usr/bin/env python3
"""
Run all analysis plotting scripts over a results directory tree.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR.parents[1] / "eval_results_all"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parents[1] / "analyiss_results_all"

PLOT_SCRIPTS = [
    "plot_absorbing_board_accuracy.py",
    "plot_uniform_noise_metrics.py",
    "plot_uniform_noise_diffusion_metrics.py",
]


def run_plot_script(
    script_name: str,
    results_root: Path,
    output_root: Path,
) -> None:
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Plot script not found: {script_path}")

    cmd = [
        sys.executable,
        str(script_path),
        "--results-root",
        str(results_root),
        "--output-root",
        str(output_root),
    ]
    print(f"Running analysis: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
    )

    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    if proc.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed with exit code {proc.returncode}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all analysis scripts for evaluation results."
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root directory containing evaluation CSV/JSON outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where analysis outputs will be saved.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    for script in PLOT_SCRIPTS:
        run_plot_script(script, results_root, output_root)

    print("All analyses completed.")


if __name__ == "__main__":
    main()
