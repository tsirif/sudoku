# Sudoku Diffusion Language Model - AI Agent Instructions

## Project Overview
This is a **discrete diffusion language model** for Sudoku puzzle solving using a Diffusion Transformer (DiT) architecture. The model learns to denoise masked/corrupted Sudoku grids through iterative refinement, treating Sudoku solving as a sequence denoising task.

## Architecture & Data Representation

### Token Vocabulary (10 tokens)
- `0`: MASK token (empty cells during training/inference)
- `1-9`: Sudoku digits
- No EOL tokens - flat 81-token sequence

### Sequence Format
- **Input**: 81 tokens representing a flattened 9×9 grid (row-major order)
- **Position encoding**: Learnable embeddings encode the 2D structure
- **Model**: 29M parameter DiT (12 layers, 512 hidden dim, 8 heads)

### Key Architectural Decisions
- No special row/column separator tokens - position embeddings capture spatial structure
- Bidirectional attention (not causal) - all cells can attend to all others
- Single forward pass predicts all positions simultaneously (not autoregressive)

## Training Paradigms

### Three Loss Modes (see [train.py](train.py))

1. **`absorbing`** (default): Pure masking noise
   - Randomly mask 20-90% of tokens with `MASK_ID`
   - Loss computed only on masked positions

2. **`uniform_absorbing_mixture`**: Masking + token replacement
   - Apply absorbing noise as above
   - Additionally replace non-masked tokens with random digits at probability `--uniform-mixture-prob`
   - Combined loss: `mask_loss + mixture_loss_weight * mix_loss`

3. **`uniform_absorbing_mixture_with_clean`**: Full supervision
   - Same as above + loss on clean (non-corrupted) tokens
   - Combined loss includes `clean_loss_weight * clean_loss`

### Dynamic Masking Strategy
```python
# From train.py: add_absorbing_noise()
mask_ratios = torch.rand(B) * (0.9 - 0.2) + 0.2  # Different ratio per sample
mask_positions = rand_values < mask_ratios.unsqueeze(1)
```
Each training sample gets a random mask ratio in [0.2, 0.9], forcing the model to handle varying corruption levels.

## Critical Workflows

### 1. Data Generation
```bash
python generate_data.py --train-size 36000 --test-size 2000 --output-dir ./data
```
- Uses `advanced_sudoku_generator.py` for puzzle generation
- Outputs: `{train,test}_{puzzles,solutions}.npy` (81-token sequences)
- Multiprocessing across all CPU cores (~30 min for 38k puzzles)

### 2. Training
```bash
python train.py \
    --data-dir ./data \
    --output-dir ./absorbing_checkpoints \
    --train-batch-size 64 \
    --num-steps 200000 \
    --loss-mode absorbing  # or uniform_absorbing_mixture
```
- **Output structure**: Checkpoints auto-nested by loss mode and hyperparams
  - `absorbing_checkpoints/best_model.pt`
  - `uniform_absorbing_mixture_checkpoints/prob_0.1/best_model.pt`
- Initial evaluation runs at step 0 (before training)
- Best model saved based on test accuracy

### 3. Evaluation
```bash
# Evaluate with diffusion sampling
python inference_puzzles.py \
    --checkpoint ./absorbing_checkpoints/best_model.pt \
    --data-dir ./data \
    --num-samples 2000 \
    --steps 10 \
    --algorithm random-remask

# Comprehensive eval suite
bash eval/run_eval_absorbing.sh  # Sweeps over steps, algorithms, mask ratios
```

### Key Evaluation Scripts
- `eval/absorbing_eval.py`: Evaluation under absorbing noise
- `eval/uniform_noise_eval.py`: Evaluation under uniform replacement noise
- `eval/utils.py`: Shared sampling utilities (`llada_sample`, metrics)

## Sampling Algorithms

Implemented in `llada_sample()` (from `eval/utils.py`):

1. **`random-remask`**: Randomly re-mask tokens at each step (default)
2. **`top-p`**: Keep high-confidence predictions, re-mask low-confidence
3. **`mad`**: Median Absolute Deviation-based confidence thresholding
4. **`simple-llada`**: Basic LLADA (Longest Lasting Active Diffusion Agents) sampling

**Remask schedulers**: `linear`, `cosine`, `constant` - control how many tokens to re-mask per step.

## Project-Specific Conventions

### File Organization
```
model.py              # DiT architecture (DiTConfig, create_sudoku_dit)
train.py              # Training loop (handles all 3 loss modes)
generate_data.py      # Dataset generation
inference_puzzles.py  # Batch evaluation on test set
eval/
  utils.py            # load_model_and_data, llada_sample, metrics
  absorbing_eval.py   # Evaluation under masking noise
  run_*.sh            # Hyperparameter sweep scripts
```

### Wandb Logging
- Project name: `sudoku-diffusion-lm` (default)
- Logs: `train/loss`, `train/accuracy`, `test/loss`, `test/board_accuracy`
- For mixture modes: Logs separate `mask_loss`, `mix_loss`, `clean_loss` components

### Checkpoint Structure
```python
{
    'step': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'scheduler_state_dict': OrderedDict,
    'best_test_acc': float,
    'config': dict  # DiTConfig as dict
}
```

## Integration Points

### Model Loading Pattern
```python
from model import create_sudoku_dit
model = create_sudoku_dit(vocab_size=10, seq_length=81)
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])
```

### Data Loading Pattern
```python
# From train.py: SudokuDataset
solutions = np.load(f'{data_dir}/train_solutions.npy')  # (N, 81)
solutions_tensor = torch.from_numpy(solutions).long()
```

### Evaluation Metrics
- **Token accuracy**: Correct token predictions on masked/noised positions
- **Board accuracy**: Percentage of fully solved puzzles (all 81 tokens correct)
- Always evaluate on **solutions**, not puzzles (model learns clean data distribution)

## Common Pitfalls

1. **Don't train on puzzles** - training data is `*_solutions.npy`, not `*_puzzles.npy`
2. **Loss mode hyperparameters**: When using `uniform_absorbing_mixture`, must specify both `--uniform-mixture-prob` and `--uniform-mixture-loss-weight`
3. **Evaluation requires diffusion sampling** - direct model output is NOT the solution; must use `llada_sample()` with multiple steps
4. **Checkpoint paths** - Output directories auto-nest by loss mode/hyperparams; check `train.py` line 730-737 for naming logic
5. **Grid validation** - Use functions from `advanced_sudoku_generator.py` to validate solutions (check rows/columns/boxes)

## Quick Reference Commands

```bash
# End-to-end experiment
bash run_experiment.sh

# Train with mixture noise
python train.py --loss-mode uniform_absorbing_mixture \
    --uniform-mixture-prob 0.1 --uniform-mixture-loss-weight 1.0

# Evaluate specific checkpoint
python eval/absorbing_eval.py \
    --checkpoint ./checkpoints/best_model.pt \
    --mask-ratio 0.5 --step 10 --algorithm random-remask

# Hyperparameter sweep (see eval/*.sh for examples)
bash eval/run_prob_sweeps_absorbing.sh
```

## Testing & Debugging

- `test_model.py`: Validates model creation, forward pass, masking, gradients (~29M params expected)
- `example_usage.py`: Demonstrates inference pipeline
- Model output: `model(input).logits` → shape `(B, 81, 10)` (logits over vocab for each position)
