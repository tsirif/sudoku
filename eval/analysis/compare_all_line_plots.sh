#!/bin/bash
# Compare uniform_noise_diffusion results across multiple checkpoints

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_BASE="/data/szhang967/dlm_agents/sudoku/eval_results_all_nov22"
OUTPUT_DIR="/data/szhang967/dlm_agents/sudoku/compare_results_all"

# List of checkpoint result directories to compare
RESULT_DIRS=(
    "${RESULTS_BASE}/checkpoints_absorbing/best_model"
    "${RESULTS_BASE}/uniform_absorbing_mixture_checkpoints/prob_0.1/best_model"
    # "${RESULTS_BASE}/uniform_absorbing_mixture_checkpoints/prob_0.2/best_model"
    # "${RESULTS_BASE}/uniform_absorbing_mixture_checkpoints/prob_0.02/best_model"
)

# Define noise_ratio and editable_ratio combinations to display
# Format: "noise,editable" for each combination
COMBINATIONS=(
    "0.1,0.4" "0.1,0.5" "0.1,0.6"
    "0.2,0.4" "0.2,0.5" "0.2,0.6"
)

# Run the comparison script
python3 "${SCRIPT_DIR}/compare_all_line_plots.py" \
    --result-dirs "${RESULT_DIRS[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    --combinations "${COMBINATIONS[@]}"

echo "Line plot comparison complete!"

