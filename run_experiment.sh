#!/bin/bash
# Complete experiment pipeline for Sudoku Diffusion Language Model

set -e  # Exit on error
export CUDA_VISIBLE_DEVICES=3
echo "======================================================"
echo "Sudoku Diffusion Language Model - Full Pipeline"
echo "======================================================"

# Configuration
DATA_DIR="./data"
# LOSS_MODE="absorbing"
# LOSS_MODE="uniform_absorbing_mixture"
LOSS_MODE="uniform_absorbing_mixture_with_clean"
UNIFORM_MIXTURE_PROB=0.02
UNIFORM_MIXTURE_LOSS_WEIGHT=1.0
UNIFORM_CLEAN_LOSS_WEIGHT=0

BASE_CHECKPOINT_DIR="./${LOSS_MODE}_checkpoints"
if [ "$LOSS_MODE" = "uniform_absorbing_mixture" ]; then
    CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR}/prob_${UNIFORM_MIXTURE_PROB}"
elif [ "$LOSS_MODE" = "uniform_absorbing_mixture_with_clean" ]; then
    CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR}/prob_${UNIFORM_MIXTURE_PROB}_clean_${UNIFORM_CLEAN_LOSS_WEIGHT}"
else
    CHECKPOINT_DIR="$BASE_CHECKPOINT_DIR"
fi
TRAIN_SIZE=36000
TEST_SIZE=2000
TRAIN_BATCH_SIZE=64
EVAL_BATCH_SIZE=2000
NUM_STEPS=200000
NUM_WORKERS=64

# Step 1: Generate Dataset
echo ""
echo "[1/3] Generating dataset..."
echo "------------------------------------------------------"
python generate_data.py \
    --train-size $TRAIN_SIZE \
    --test-size $TEST_SIZE \
    --output-dir $DATA_DIR \
    --num-workers $NUM_WORKERS

# Step 2: Train Model
echo ""
echo "[2/3] Training model..."
echo "------------------------------------------------------"
python train.py \
    --data-dir $DATA_DIR \
    --output-dir $CHECKPOINT_DIR \
    --train-batch-size $TRAIN_BATCH_SIZE \
    --eval-batch-size $EVAL_BATCH_SIZE \
    --num-steps $NUM_STEPS \
    --lr 1e-4 \
    --weight-decay 0.01 \
    --mask-ratio-min 0.2 \
    --mask-ratio-max 0.9 \
    --num-workers $NUM_WORKERS \
    --log-interval 100 \
    --eval-interval 1000 \
    --save-interval 10000 \
    --loss-mode $LOSS_MODE \
    --uniform-mixture-prob $UNIFORM_MIXTURE_PROB \
    --uniform-mixture-loss-weight $UNIFORM_MIXTURE_LOSS_WEIGHT \
    --uniform-clean-loss-weight $UNIFORM_CLEAN_LOSS_WEIGHT

# Step 3: Evaluate Model
echo ""
echo "[3/3] Evaluating model on test set..."
echo "------------------------------------------------------"
python inference_puzzles.py \
    --checkpoint $CHECKPOINT_DIR/best_model.pt \
    --data-dir $DATA_DIR \
    --num-samples $TEST_SIZE \
    --steps 10 \
    --algorithm random-remask \
    --device cuda

echo ""
echo "======================================================"
echo "✓ Experiment complete!"
echo "======================================================"
echo "Results:"
echo "  - Dataset: $DATA_DIR"
echo "  - Checkpoints: $CHECKPOINT_DIR"
echo "  - Loss mode: $LOSS_MODE"
if [ "$LOSS_MODE" = "uniform_absorbing_mixture" ]; then
  echo "  - Uniform mixture prob: $UNIFORM_MIXTURE_PROB"
  echo "  - Uniform mixture loss weight: $UNIFORM_MIXTURE_LOSS_WEIGHT"
elif [ "$LOSS_MODE" = "uniform_absorbing_mixture_with_clean" ]; then
  echo "  - Uniform mixture prob: $UNIFORM_MIXTURE_PROB"
  echo "  - Uniform mixture loss weight: $UNIFORM_MIXTURE_LOSS_WEIGHT"
  echo "  - Uniform clean loss weight: $UNIFORM_CLEAN_LOSS_WEIGHT"
fi
echo "  - Logging: wandb (see configured project)"
echo ""

