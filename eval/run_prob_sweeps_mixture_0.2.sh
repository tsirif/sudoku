#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=5
# Evaluate specified checkpoints in a given prob directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration: prob directory and checkpoints to evaluate
PROB_DIR="${ROOT_DIR}/uniform_absorbing_mixture_checkpoints/prob_0.2"
CHECKPOINTS=(
  "best_model.pt"
  "checkpoint_100000.pt"
  "checkpoint_150000.pt"
  "checkpoint_190000.pt"
)

# Validate prob directory
if [ ! -d "${PROB_DIR}" ]; then
  echo "Error: Directory not found: ${PROB_DIR}" >&2
  exit 1
fi

# Default configuration
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
DEVICE="${DEVICE:-cuda}"
PROJECT="${PROJECT:-sudoku_eval_nov22}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/eval_results_all_nov22}"
MODES=${MODES:-"absorbing uniform_noise_only uniform_noise_diffusion"}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-2000}
NUM_SAMPLES=${NUM_SAMPLES:-2000}

# Evaluate each checkpoint
for checkpoint_name in "${CHECKPOINTS[@]}"; do
  checkpoint_path="${PROB_DIR}/${checkpoint_name}"
  
  if [ ! -f "${checkpoint_path}" ]; then
    echo "Warning: Checkpoint not found: ${checkpoint_path}, skipping..." >&2
    continue
  fi

  echo "=============================="
  echo "Evaluating: ${checkpoint_path}"
  echo "=============================="

  python "${ROOT_DIR}/eval/sweep_eval.py" \
    --checkpoint "${checkpoint_path}" \
    --data-dir "${DATA_DIR}" \
    --device "${DEVICE}" \
    --project "${PROJECT}" \
    --output-dir "${OUTPUT_DIR}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --num-samples "${NUM_SAMPLES}" \
    --modes ${MODES}
done

echo "=============================="
echo "Evaluation complete!"
echo "=============================="
