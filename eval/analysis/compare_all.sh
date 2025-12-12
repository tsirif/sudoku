#!/bin/bash
# Compare uniform_noise_only results across multiple checkpoints

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

# Run the comparison script
python3 "${SCRIPT_DIR}/compare_all.py" \
    --result-dirs "${RESULT_DIRS[@]}" \
    --output-dir "${OUTPUT_DIR}"

echo "Comparison complete!"

