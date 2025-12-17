#!/bin/bash
# Test HuggingFace dataset generation with a small subset

set -e

echo "========================================"
echo "Testing HuggingFace Dataset Generation"
echo "========================================"

# Test 1: Generate small dataset with no augmentation
echo ""
echo "Test 1: Small dataset, no augmentation"
python generate_data_from_hf.py \
    --output-dir ./data/test_hf_no_aug \
    --num-aug 0 \
    --subsample-size 100 \
    --save-format huggingface

# Test 2: Generate small dataset with 2 augmentations
echo ""
echo "Test 2: Small dataset, 2 augmentations"
python generate_data_from_hf.py \
    --output-dir ./data/test_hf_aug2 \
    --num-aug 2 \
    --subsample-size 100 \
    --save-format huggingface

# Test 3: Generate small dataset with numpy format
echo ""
echo "Test 3: Small dataset, numpy format"
python generate_data_from_hf.py \
    --output-dir ./data/test_npy_aug2 \
    --num-aug 2 \
    --subsample-size 100 \
    --save-format npy

# Test loading in Python
echo ""
echo "Test 4: Load HuggingFace dataset in Python"
python << 'EOF'
from datasets import load_from_disk
import numpy as np

# Load dataset
ds = load_from_disk('./data/test_hf_aug2/train')
print(f"\nLoaded {len(ds)} training samples")
print(f"Columns: {ds.column_names}")

# Check first sample
sample = ds[0]
print(f"\nFirst sample:")
print(f"  Puzzle shape: {sample['puzzle'].shape}")
print(f"  Solution shape: {sample['solution'].shape}")
print(f"  Group index: {sample['group_index']}")
print(f"  Is augmented: {sample['is_augmented']}")
print(f"  Augmentation ID: {sample['augmentation_id']}")

# Verify vocabulary (should be 0-9)
puzzle = np.array(sample['puzzle'])
solution = np.array(sample['solution'])
print(f"\nVocabulary check:")
print(f"  Puzzle min: {puzzle.min()}, max: {puzzle.max()}")
print(f"  Solution min: {solution.min()}, max: {solution.max()}")

# Count augmentations
num_original = sum(1 for s in ds if not s['is_augmented'])
num_augmented = sum(1 for s in ds if s['is_augmented'])
print(f"\nAugmentation stats:")
print(f"  Original samples: {num_original}")
print(f"  Augmented samples: {num_augmented}")
print(f"  Total: {len(ds)}")
print(f"  Expected: {num_original * 3} (1 original + 2 augmented)")

EOF

# Test loading with train.py
echo ""
echo "Test 5: Test train.py data loading"
python << 'EOF'
import sys
sys.path.insert(0, '.')
from pathlib import Path
from train import SudokuDataset

# Test HuggingFace format
print("Testing HuggingFace format...")
ds_hf = SudokuDataset('./data/test_hf_aug2', split='train', format='huggingface')
print(f"  Loaded {len(ds_hf)} samples")
sample = ds_hf[0]
print(f"  Sample keys: {sample.keys()}")
print(f"  Solution shape: {sample['solution'].shape}")

# Test numpy format
print("\nTesting numpy format...")
ds_npy = SudokuDataset('./data/test_npy_aug2', split='train', format='npy')
print(f"  Loaded {len(ds_npy)} samples")
sample = ds_npy[0]
print(f"  Sample keys: {sample.keys()}")
print(f"  Solution shape: {sample['solution'].shape}")

# Test auto-detection
print("\nTesting auto-detection...")
ds_auto_hf = SudokuDataset('./data/test_hf_aug2', split='train', format='auto')
print(f"  Detected format: {ds_auto_hf.format}")
ds_auto_npy = SudokuDataset('./data/test_npy_aug2', split='train', format='auto')
print(f"  Detected format: {ds_auto_npy.format}")

print("\n✓ All tests passed!")

EOF

echo ""
echo "========================================"
echo "✓ All tests completed successfully!"
echo "========================================"
