# HuggingFace Dataset Integration - Usage Guide

This guide covers using the HuggingFace-based dataset generation for fair comparison with TinyRecursiveModels.

## Overview

The new `generate_data_from_hf.py` script downloads the `sapientinc/sudoku-extreme` dataset from HuggingFace, converts vocabulary from TinyRecursiveModels format (1-10) to our format (0-9), applies augmentation, and saves in HuggingFace or numpy format.

**Key Features:**
- ✅ Exact same dataset as TinyRecursiveModels
- ✅ Vocabulary conversion (TRM 1-10 → Sudoku 0-9)
- ✅ Symmetry-preserving augmentation
- ✅ Saves both puzzles and solutions
- ✅ Backward compatible with existing train.py

## Quick Start

### 1. Install Dependencies

```bash
# Activate environment and install HuggingFace tools
source setup_env.sh

# Or manually install
pip install huggingface_hub datasets pyarrow
```

### 2. Generate Dataset (Matching TinyRecursiveModels)

```bash
# Standard setup: 8 augmentations per training sample
python generate_data_from_hf.py \
    --output-dir data/sudoku-extreme-aug8 \
    --num-aug 8 \
    --save-format huggingface

# Expected output:
# - data/sudoku-extreme-aug8/
#   ├── train/ (HuggingFace dataset with augmentations)
#   ├── test/ (HuggingFace dataset, no augmentation)
#   └── dataset_dict.json
```

**Training Data:**
- Original puzzles: depends on CSV (typically 9M+)
- With 8x augmentation: original × 9 (1 original + 8 augmented)
- Test data: no augmentation

### 3. Train Model

```bash
# Train with HuggingFace dataset (auto-detected)
python train.py \
    --data-dir data/sudoku-extreme-aug8 \
    --output-dir checkpoints/hf-aug8-absorbing \
    --train-batch-size 64 \
    --num-steps 200000 \
    --loss-mode absorbing

# Explicitly specify format (optional)
python train.py \
    --data-dir data/sudoku-extreme-aug8 \
    --data-format huggingface \
    ...
```

## Vocabulary Conversion

**TinyRecursiveModels (TRM) Format:**
- `0` = PAD token (unused)
- `1` = blank/empty cell
- `2-10` = digits 1-9

**Our (Sudoku) Format:**
- `0` = MASK/empty cell
- `1-9` = digits

**Conversion:** Subtract 1 from all TRM values
- `1 → 0` (blank → MASK)
- `2 → 1, 3 → 2, ..., 10 → 9` (digits)

This is handled automatically by `generate_data_from_hf.py`.

## Augmentation

The script implements symmetry-preserving augmentation from TinyRecursiveModels:

**Transformations:**
1. **Digit permutation**: Randomly permute digits 1-9 (keep 0=MASK unchanged)
2. **Transposition**: Randomly transpose grid (rows ↔ columns)
3. **Row permutation**: Shuffle rows within 3×3 bands
4. **Column permutation**: Shuffle columns within 3×3 stacks

**Properties:**
- Preserves Sudoku validity
- Each augmentation is deterministic (seeded by puzzle index)
- Applied only to training set
- Test set never augmented

## Dataset Formats

### HuggingFace Format (Recommended)

**Structure:**
```
data/sudoku-extreme-aug8/
├── train/
│   └── data-00000-of-00001.arrow
├── test/
│   └── data-00000-of-00001.arrow
└── dataset_dict.json
```

**Columns:**
- `puzzle`: (81,) array with 0=MASK, 1-9=digits
- `solution`: (81,) array with 1-9=digits only
- `puzzle_identifier`: Always 0 (single-step puzzles)
- `puzzle_index`: Global index
- `group_index`: Augmentation group ID (same for original + augmented versions)
- `is_augmented`: Boolean flag
- `augmentation_id`: 0=original, 1-N=augmented

**Advantages:**
- Memory-mapped (efficient for large datasets)
- Includes metadata for filtering/analysis
- Easy to inspect and share

### Numpy Format (Backward Compatible)

**Structure:**
```
data/sudoku-extreme-aug8/
├── train_puzzles.npy
├── train_solutions.npy
├── train_puzzle_identifiers.npy
├── train_puzzle_indices.npy
├── train_group_indices.npy
├── train_is_augmented.npy
├── train_augmentation_ids.npy
├── train_metadata.json
├── test_*.npy
└── test_metadata.json
```

**Use when:**
- Compatibility with old scripts
- No HuggingFace dependencies
- Simpler file inspection

## Advanced Usage

### Subsampling Training Data

```bash
# Generate smaller dataset for quick experiments
python generate_data_from_hf.py \
    --output-dir data/sudoku-extreme-aug8-sub10k \
    --num-aug 8 \
    --subsample-size 10000

# Result: 10k original + 80k augmented = 90k total training samples
```

**Note:** Subsampling happens **before** augmentation.

### Difficulty Filtering

```bash
# Only use hard puzzles (rating ≥ 5)
python generate_data_from_hf.py \
    --output-dir data/sudoku-extreme-aug8-hard \
    --num-aug 8 \
    --min-difficulty 5
```

**Note:** Check TinyRecursiveModels config to see what they use (likely None = all difficulties).

### No Augmentation (Baseline)

```bash
# Train on original data only (no augmentation)
python generate_data_from_hf.py \
    --output-dir data/sudoku-extreme-no-aug \
    --num-aug 0
```

### Different Augmentation Counts

```bash
# Light augmentation (4x)
python generate_data_from_hf.py \
    --output-dir data/sudoku-extreme-aug4 \
    --num-aug 4

# Heavy augmentation (16x)
python generate_data_from_hf.py \
    --output-dir data/sudoku-extreme-aug16 \
    --num-aug 16
```

## Training with Both Formats

The updated `train.py` automatically detects format:

```bash
# HuggingFace format
python train.py --data-dir data/sudoku-extreme-aug8

# Old numpy format (still works!)
python train.py --data-dir data/old_generated_data

# Explicit format specification
python train.py --data-dir data/... --data-format huggingface
python train.py --data-dir data/... --data-format npy
```

## Dataset Statistics

After generation, verify dataset statistics:

```python
from datasets import load_from_disk

# Load dataset
ds = load_from_disk('data/sudoku-extreme-aug8/train')

# Basic stats
print(f"Total samples: {len(ds)}")
print(f"Original: {sum(1 for s in ds if not s['is_augmented'])}")
print(f"Augmented: {sum(1 for s in ds if s['is_augmented'])}")

# Check vocabulary
import numpy as np
sample = ds[0]
puzzle = np.array(sample['puzzle'])
solution = np.array(sample['solution'])
print(f"Puzzle range: [{puzzle.min()}, {puzzle.max()}]")  # Should be [0, 9]
print(f"Solution range: [{solution.min()}, {solution.max()}]")  # Should be [1, 9]
```

## Validation

The script includes automatic validation:

1. **Vocabulary check**: Ensures all values in [0, 9] after conversion
2. **Sudoku validity**: Validates solutions have unique digits in rows/columns/boxes
3. **Augmentation validity**: Checks augmented puzzles remain valid

**Manual validation:**

```bash
# Run test suite
bash test_hf_dataset.sh

# Expected output: All tests passed
```

## Comparison with TinyRecursiveModels

**Exact match:**
- ✅ Same source dataset (sapientinc/sudoku-extreme)
- ✅ Same train/test split
- ✅ Same augmentation strategy (shuffle_sudoku)
- ✅ Same number of augmentations (if you use same `--num-aug`)

**Differences:**
- Vocabulary: We use 0-9 (they use 1-10) → conversion is automatic
- Model architecture: Different models, fair comparison on same data
- Data structure: We include more metadata (augmentation tracking)

## Common Issues

### "datasets not installed"
```bash
pip install datasets pyarrow
```

### "HuggingFace authentication required"
Some datasets need login:
```bash
pip install huggingface_hub
huggingface-cli login
```

### Out of Memory
Use subsampling:
```bash
python generate_data_from_hf.py --subsample-size 50000 ...
```

### Slow generation
- Augmentation is CPU-intensive
- Large datasets take 10-30 minutes
- Use `--num-aug 0` for quick testing

## Example Workflows

### Full Training Run (Matching TRM)

```bash
# 1. Generate dataset (one-time, ~20 minutes)
python generate_data_from_hf.py \
    --output-dir data/sudoku-extreme-aug8 \
    --num-aug 8 \
    --seed 42

# 2. Train model (8-12 hours on A100)
python train.py \
    --data-dir data/sudoku-extreme-aug8 \
    --output-dir checkpoints/hf-aug8-absorbing \
    --train-batch-size 64 \
    --num-steps 200000 \
    --loss-mode absorbing

# 3. Evaluate
python inference_puzzles.py \
    --checkpoint checkpoints/hf-aug8-absorbing/best_model.pt \
    --data-dir data/sudoku-extreme-aug8 \
    --num-samples 2000 \
    --steps 10
```

### Quick Experiment (Small Dataset)

```bash
# 1. Generate small dataset (~2 minutes)
python generate_data_from_hf.py \
    --output-dir data/sudoku-extreme-aug4-sub5k \
    --num-aug 4 \
    --subsample-size 5000

# 2. Quick training (~2 hours)
python train.py \
    --data-dir data/sudoku-extreme-aug4-sub5k \
    --output-dir checkpoints/quick-test \
    --train-batch-size 128 \
    --num-steps 50000

# 3. Evaluate
python inference_puzzles.py \
    --checkpoint checkpoints/quick-test/best_model.pt \
    --data-dir data/sudoku-extreme-aug4-sub5k \
    --num-samples 500 \
    --steps 10
```

### Augmentation Ablation Study

```bash
# Generate datasets with different augmentation levels
for NUM_AUG in 0 2 4 8 16; do
    python generate_data_from_hf.py \
        --output-dir data/sudoku-extreme-aug${NUM_AUG} \
        --num-aug ${NUM_AUG}
done

# Train on each
for NUM_AUG in 0 2 4 8 16; do
    python train.py \
        --data-dir data/sudoku-extreme-aug${NUM_AUG} \
        --output-dir checkpoints/ablation-aug${NUM_AUG} \
        --num-steps 200000
done
```

## File Sizes

**Approximate storage requirements:**

| Dataset | Train Samples | Disk Space |
|---------|--------------|------------|
| No augmentation | ~9M | ~1.5 GB |
| 4x augmentation | ~45M | ~7 GB |
| 8x augmentation | ~81M | ~13 GB |
| 16x augmentation | ~153M | ~25 GB |

HuggingFace format is slightly larger than numpy due to metadata, but more flexible.

## Next Steps

1. **Generate your dataset** with desired augmentation
2. **Verify statistics** match expected counts
3. **Train model** using same hyperparameters as TRM
4. **Compare results** on same test set

For questions or issues, see main documentation in `INSTRUCTIONS.md` and `.github/copilot-instructions.md`.
