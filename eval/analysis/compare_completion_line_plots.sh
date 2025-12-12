#!/bin/bash
# Compare absorbing results across multiple checkpoints

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_BASE="/data/szhang967/dlm_agents/sudoku/eval_results_absorbing_only_nov22"
OUTPUT_DIR="/data/szhang967/dlm_agents/sudoku/compare_results_all"

# List of checkpoint result directories to compare (best_model paths)
RESULT_DIRS=(
    "${RESULTS_BASE}/checkpoints_absorbing/best_model"
    "${RESULTS_BASE}/uniform_absorbing_mixture_checkpoints/prob_0.1/best_model"
    # "${RESULTS_BASE}/uniform_absorbing_mixture_checkpoints/prob_0.2/best_model"
    # "${RESULTS_BASE}/uniform_absorbing_mixture_checkpoints/prob_0.02/best_model"
)

# Define mask_ratios to display
# Default: (0.3 0.4 0.5)
MASK_RATIOS=(
    0.3
    0.4
    0.5
    0.6
)

# Define steps to display
# Default: (1 2 3 4 5 6 7 8)
STEPS=(
    1
    2
    3
    4
    5
    6
    7
    8
)

# Run the comparison script
python3 "${SCRIPT_DIR}/compare_completion_line_plots.py" \
    --result-dirs "${RESULT_DIRS[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    --mask-ratios "${MASK_RATIOS[@]}" \
    --steps "${STEPS[@]}"

echo "Absorbing completion line plot comparison complete!"

