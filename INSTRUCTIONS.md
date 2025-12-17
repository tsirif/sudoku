# Sudoku Diffusion Model - Complete Workflow Instructions

This guide provides step-by-step instructions for generating data, training models, and evaluating them on Sudoku puzzles.

## Prerequisites

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Required packages:
# - torch (PyTorch for deep learning)
# - numpy (numerical operations)
# - wandb (experiment tracking, optional)
# - tqdm (progress bars)
```

### Verify Installation
```bash
# Test that the model can be created
python test_model.py

# Expected output: All tests pass, ~29M parameters
```

---

## Step 1: Generate Training and Validation Data

### Quick Start (Default Settings)
```bash
python generate_data.py \
    --train-size 36000 \
    --test-size 2000 \
    --output-dir ./data \
    --num-workers 64
```

### What This Does
1. Generates 36,000 training puzzles with solutions
2. Generates 2,000 test/validation puzzles with solutions
3. Uses multiprocessing (64 workers) across all CPU cores
4. Saves data in `./data/` directory

### Expected Runtime
- ~30-45 minutes on modern multi-core CPU
- Progress bar shows puzzle generation status

### Output Files
```
./data/
├── train_puzzles.npy        # (36000, 81) - puzzles as 81-token sequences
├── train_solutions.npy      # (36000, 81) - solutions as 81-token sequences
├── test_puzzles.npy         # (2000, 81) - test puzzles
├── test_solutions.npy       # (2000, 81) - test solutions
├── train_puzzles.txt        # Human-readable text format
├── train_solutions.txt      # Human-readable text format
├── test_puzzles.txt         # Human-readable text format
├── test_solutions.txt       # Human-readable text format
├── train_metadata.json      # Dataset metadata
└── test_metadata.json       # Dataset metadata
```

### Understanding the Data Format
- **Sequence length**: 81 tokens (9×9 grid, row-major order)
- **Token values**: 0 (MASK/empty), 1-9 (digits)
- **No separator tokens**: Position embeddings encode spatial structure
- Example: `[5, 3, 0, 0, 7, ...]` represents first row of Sudoku

### Custom Data Generation
```bash
# Generate smaller dataset for quick experiments
python generate_data.py --train-size 1000 --test-size 100 --output-dir ./data_small

# Specify number of parallel workers
python generate_data.py --train-size 36000 --test-size 2000 --num-workers 32

# Check if data already exists (skip regeneration)
python generate_data.py --train-size 36000 --test-size 2000 --output-dir ./data
# → Will skip if data exists and matches requested size
```

---

## Step 2: Train a Diffusion Model

### Quick Start (Absorbing Loss Mode)
```bash
python train.py \
    --data-dir ./data \
    --output-dir ./absorbing_checkpoints \
    --train-batch-size 64 \
    --eval-batch-size 2000 \
    --num-steps 200000 \
    --lr 1e-4 \
    --loss-mode absorbing
```

### What This Does
1. Loads training data from `./data/`
2. Creates a 29M parameter Diffusion Transformer model
3. Trains with absorbing (masking) noise for 200k steps
4. Evaluates on test set every 1000 steps
5. Saves best model based on test accuracy
6. Logs metrics to wandb (optional)

### Expected Runtime
- ~8-12 hours on single A100 GPU
- ~24-48 hours on consumer GPU (RTX 3090)

### Training Loss Modes

#### 1. Absorbing (Masking Only)
```bash
python train.py \
    --data-dir ./data \
    --output-dir ./absorbing_checkpoints \
    --loss-mode absorbing \
    --mask-ratio-min 0.2 \
    --mask-ratio-max 0.9
```
- Randomly masks 20-90% of tokens per sample
- Model learns to predict masked tokens
- **Best for**: Clean diffusion training baseline

#### 2. Uniform Absorbing Mixture (Masking + Noise)
```bash
python train.py \
    --data-dir ./data \
    --output-dir ./mixture_checkpoints \
    --loss-mode uniform_absorbing_mixture \
    --uniform-mixture-prob 0.1 \
    --uniform-mixture-loss-weight 1.0
```
- Applies masking (20-90% ratio)
- Additionally replaces 10% of non-masked tokens with random digits
- **Best for**: More robust denoising (handles both mask and corruption)

#### 3. Uniform Absorbing Mixture with Clean Loss
```bash
python train.py \
    --data-dir ./data \
    --output-dir ./mixture_clean_checkpoints \
    --loss-mode uniform_absorbing_mixture_with_clean \
    --uniform-mixture-prob 0.02 \
    --uniform-clean-loss-weight 1.0
```
- Same as above + loss on clean (non-corrupted) tokens
- **Best for**: Maximum supervision signal

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-batch-size` | 128 | Training batch size (reduce if OOM) |
| `--eval-batch-size` | 128 | Evaluation batch size (can be larger) |
| `--num-steps` | 100000 | Total training steps |
| `--lr` | 1e-4 | Learning rate (AdamW optimizer) |
| `--weight-decay` | 0.01 | L2 regularization |
| `--mask-ratio-min` | 0.2 | Minimum masking ratio |
| `--mask-ratio-max` | 0.9 | Maximum masking ratio |
| `--log-interval` | 100 | Log metrics every N steps |
| `--eval-interval` | 1000 | Evaluate on test set every N steps |
| `--save-interval` | 10000 | Save checkpoint every N steps |

### Output Files
```
./absorbing_checkpoints/
├── best_model.pt          # Best model (highest test accuracy)
├── final_model.pt         # Final model (last training step)
└── checkpoint_10000.pt    # Periodic checkpoints
    checkpoint_20000.pt
    ...
```

For mixture modes with hyperparameters, outputs are nested:
```
./uniform_absorbing_mixture_checkpoints/
└── prob_0.1/
    ├── best_model.pt
    └── ...
```

### Checkpoint Structure
Each `.pt` file contains:
```python
{
    'step': 50000,                     # Training step
    'model_state_dict': {...},         # Model weights
    'optimizer_state_dict': {...},     # Optimizer state
    'scheduler_state_dict': {...},     # LR scheduler state
    'best_test_acc': 0.856,           # Best test accuracy so far
    'config': {...}                    # Model configuration
}
```

### Monitoring Training

#### Option 1: Console Output
```
[Step 1000] Test Loss: 0.2345, Test Acc: 0.823
✓ Saved best model (acc=0.823)
```

#### Option 2: Wandb (Recommended)
```bash
# Install wandb
pip install wandb
wandb login

# Train with wandb logging (enabled by default)
python train.py --data-dir ./data --output-dir ./checkpoints

# Disable wandb
python train.py --data-dir ./data --output-dir ./checkpoints --no-wandb

# Custom wandb project/run name
python train.py \
    --wandb-project my-sudoku-experiments \
    --wandb-run-name absorbing-baseline
```

View metrics at: https://wandb.ai/your-username/my-sudoku-experiments

### Resume Training from Checkpoint
```bash
# Load checkpoint and continue training
python train.py \
    --data-dir ./data \
    --output-dir ./checkpoints \
    --resume-from ./checkpoints/checkpoint_50000.pt \
    --num-steps 100000  # Train to step 100k total
```
(Note: Resume functionality may need to be implemented - current version saves checkpoints but doesn't have explicit resume flag)

---

## Step 3: Evaluate a Trained Model

### Quick Evaluation on Test Set
```bash
python inference_puzzles.py \
    --checkpoint ./absorbing_checkpoints/best_model.pt \
    --data-dir ./data \
    --num-samples 2000 \
    --steps 10 \
    --algorithm random-remask \
    --device cuda
```

### What This Does
1. Loads trained model checkpoint
2. Loads test solutions from `./data/test_solutions.npy`
3. Runs diffusion sampling for 10 steps
4. Computes board accuracy (% fully solved puzzles)

### Output
```
Loading checkpoint: ./absorbing_checkpoints/best_model.pt
Model step: 200000, Best test acc: 0.876

Running evaluation on 2000 test samples...
Steps: 10, Algorithm: random-remask

Results:
  Board Accuracy: 0.892
  Token Accuracy: 0.967
  Average solving time: 0.45s per puzzle
```

### Sampling Algorithms

#### Random Remask (Default)
```bash
python inference_puzzles.py \
    --checkpoint ./checkpoints/best_model.pt \
    --algorithm random-remask \
    --steps 10
```
- Randomly re-mask tokens at each step
- **Best for**: General-purpose solving

#### Top-P Sampling
```bash
python inference_puzzles.py \
    --checkpoint ./checkpoints/best_model.pt \
    --algorithm top-p \
    --steps 10 \
    --confidence-threshold 0.8
```
- Keep high-confidence predictions (p > 0.8)
- Re-mask low-confidence tokens
- **Best for**: More conservative solving

#### MAD (Median Absolute Deviation)
```bash
python inference_puzzles.py \
    --checkpoint ./checkpoints/best_model.pt \
    --algorithm mad \
    --steps 10 \
    --mad-k 2.0
```
- Statistical outlier detection for low-confidence tokens
- **Best for**: Adaptive confidence thresholding

### Remask Schedulers

Control how many tokens to re-mask at each step:

```bash
# Linear decrease (default)
python inference_puzzles.py --remask-scheduler linear

# Cosine decay
python inference_puzzles.py --remask-scheduler cosine

# Constant ratio
python inference_puzzles.py --remask-scheduler constant
```

### Evaluation Scripts

#### Full Evaluation Suite
```bash
# Comprehensive evaluation with hyperparameter sweeps
bash eval/run_all_evals.py --checkpoint ./checkpoints/best_model.pt
```

#### Evaluation Under Different Noise Types

**Absorbing Noise (Masking)**:
```bash
python eval/absorbing_eval.py \
    --checkpoint ./checkpoints/best_model.pt \
    --data-dir ./data \
    --mask-ratio 0.5 \
    --step 10 \
    --algorithm random-remask
```

**Uniform Noise (Token Replacement)**:
```bash
python eval/uniform_noise_eval.py \
    --checkpoint ./checkpoints/best_model.pt \
    --data-dir ./data \
    --noise-ratio 0.3 \
    --step 10
```

#### Hyperparameter Sweeps
```bash
# Sweep over different mask ratios and sampling steps
bash eval/run_eval_absorbing.sh ./checkpoints/best_model.pt

# Sweep for mixture-trained models
bash eval/run_prob_sweeps_mixture_0.1.sh
```

### Analysis and Visualization

After running evaluations, analyze results:
```bash
cd eval/analysis

# Compare different checkpoints
python compare_all.py

# Plot board accuracy over training steps
python plot_absorbing_board_accuracy.py

# Generate comparison plots
bash compare_all_line_plots.sh
```

### Custom Puzzle Evaluation

Solve a specific puzzle (not from test set):
```bash
# Create puzzle file (0 = empty cell)
cat > my_puzzle.txt << EOF
5 3 0 0 7 0 0 0 0
6 0 0 1 9 5 0 0 0
0 9 8 0 0 0 0 6 0
8 0 0 0 6 0 0 0 3
4 0 0 8 0 3 0 0 1
7 0 0 0 2 0 0 0 6
0 6 0 0 0 0 2 8 0
0 0 0 4 1 9 0 0 5
0 0 0 0 8 0 0 7 9
EOF

python inference_puzzles.py \
    --checkpoint ./checkpoints/best_model.pt \
    --puzzle-file my_puzzle.txt \
    --steps 10 \
    --algorithm random-remask
```

---

## Complete End-to-End Workflow

### Option 1: Automated Script
```bash
# Edit run_experiment.sh to configure:
# - DATA_DIR, CHECKPOINT_DIR
# - LOSS_MODE, UNIFORM_MIXTURE_PROB
# - TRAIN_SIZE, NUM_STEPS

bash run_experiment.sh
```

This runs:
1. Data generation (if not exists)
2. Model training
3. Evaluation on test set

### Option 2: Manual Step-by-Step
```bash
# 1. Generate data
python generate_data.py --train-size 36000 --test-size 2000 --output-dir ./data

# 2. Train model
python train.py \
    --data-dir ./data \
    --output-dir ./checkpoints \
    --num-steps 200000 \
    --loss-mode absorbing

# 3. Evaluate
python inference_puzzles.py \
    --checkpoint ./checkpoints/best_model.pt \
    --data-dir ./data \
    --num-samples 2000 \
    --steps 10

# 4. Analyze results
python eval/run_all_evals.py --checkpoint ./checkpoints/best_model.pt
```

---

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python train.py --train-batch-size 32 --eval-batch-size 1000

# Enable gradient accumulation (manual implementation needed)
# Or train on smaller dataset first
```

### Slow Data Generation
```bash
# Reduce workers if causing issues
python generate_data.py --num-workers 16

# Generate smaller dataset for testing
python generate_data.py --train-size 1000 --test-size 100
```

### Training Not Improving
- Check initial evaluation (step 0) - should be ~10% accuracy (random)
- Ensure training on **solutions**, not puzzles
- Verify masking ratio range (0.2-0.9 is good)
- Try lower learning rate: `--lr 5e-5`

### Evaluation Accuracy Lower Than Expected
- Increase sampling steps: `--steps 20` or `--steps 50`
- Try different algorithms: `--algorithm mad` or `--algorithm top-p`
- Check if model trained long enough (test accuracy during training)

### Wandb Not Logging
```bash
# Login to wandb
wandb login

# Or disable wandb
python train.py --no-wandb
```

---

## Quick Reference

### Minimal Working Example
```bash
# 1. Generate small dataset (5 minutes)
python generate_data.py --train-size 1000 --test-size 100 --output-dir ./data_small

# 2. Train for 10k steps (30 minutes on GPU)
python train.py --data-dir ./data_small --output-dir ./checkpoints_test --num-steps 10000

# 3. Evaluate
python inference_puzzles.py --checkpoint ./checkpoints_test/best_model.pt --data-dir ./data_small --num-samples 100 --steps 5
```

### Recommended Production Settings
```bash
# Data
python generate_data.py --train-size 36000 --test-size 2000 --output-dir ./data

# Training (absorbing baseline)
python train.py \
    --data-dir ./data \
    --output-dir ./absorbing_checkpoints \
    --train-batch-size 64 \
    --num-steps 200000 \
    --loss-mode absorbing

# Training (mixture - more robust)
python train.py \
    --data-dir ./data \
    --output-dir ./mixture_checkpoints \
    --train-batch-size 64 \
    --num-steps 200000 \
    --loss-mode uniform_absorbing_mixture \
    --uniform-mixture-prob 0.1 \
    --uniform-mixture-loss-weight 1.0

# Evaluation
python inference_puzzles.py \
    --checkpoint ./absorbing_checkpoints/best_model.pt \
    --data-dir ./data \
    --num-samples 2000 \
    --steps 10 \
    --algorithm random-remask
```

---

## Further Reading

- [DLM_README.md](DLM_README.md) - Detailed architecture and theory
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Project overview
- [.github/copilot-instructions.md](.github/copilot-instructions.md) - AI agent coding guide
