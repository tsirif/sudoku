# Quick Reference: HuggingFace Dataset Integration

## 🚀 One-Line Commands

```bash
# Generate full dataset (8x augmentation, matching TRM)
python generate_data_from_hf.py --output-dir data/sudoku-extreme-aug8 --num-aug 8

# Train on HuggingFace dataset
python train.py --data-dir data/sudoku-extreme-aug8 --output-dir checkpoints/hf-aug8

# Test everything
bash test_hf_dataset.sh
```

## 📊 Vocabulary Mapping

| TRM Format | Sudoku Format | Meaning |
|------------|---------------|---------|
| 0 | (unused) | PAD token |
| 1 | 0 | MASK/blank |
| 2-10 | 1-9 | Digits |

**Conversion:** `sudoku_value = trm_value - 1`

## 🎲 Augmentation Levels

```bash
# No augmentation (baseline)
--num-aug 0

# Light (quick experiments)
--num-aug 4    # 5x total samples

# Standard (match TRM)
--num-aug 8    # 9x total samples

# Heavy (maximum diversity)
--num-aug 16   # 17x total samples
```

## 📁 Dataset Formats

### HuggingFace (Recommended)
```bash
--save-format huggingface
# Output: data/*/train/, data/*/test/, dataset_dict.json
# Use with: train.py --data-format huggingface (or auto)
```

### Numpy (Backward Compatible)
```bash
--save-format npy
# Output: data/*_{train,test}_{puzzles,solutions,...}.npy
# Use with: train.py --data-format npy (or auto)
```

## ⚡ Quick Experiments

```bash
# Small dataset (1k original → 5k with 4x aug)
python generate_data_from_hf.py \
    --output-dir data/quick-test \
    --num-aug 4 \
    --subsample-size 1000

# Quick training (30 min on GPU)
python train.py \
    --data-dir data/quick-test \
    --num-steps 10000 \
    --train-batch-size 128
```

## 🔍 Validation

```python
# Verify vocabulary
from datasets import load_from_disk
ds = load_from_disk('data/sudoku-extreme-aug8/train')
sample = ds[0]
print(f"Puzzle: {sample['puzzle'].min()}-{sample['puzzle'].max()}")  # 0-9
print(f"Solution: {sample['solution'].min()}-{sample['solution'].max()}")  # 1-9

# Check augmentation
print(f"Total: {len(ds)}")
print(f"Original: {sum(1 for s in ds if not s['is_augmented'])}")
print(f"Augmented: {sum(1 for s in ds if s['is_augmented'])}")
```

## 📋 Checklist for Fair Comparison with TRM

- [ ] Same dataset: `sapientinc/sudoku-extreme`
- [ ] Same `--num-aug` (check TRM config)
- [ ] Same `--min-difficulty` (likely None)
- [ ] No `--subsample-size` (use full dataset)
- [ ] Verify vocabulary: 0-9 for sudoku format
- [ ] Train/test split matches (automatic from HF)

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `datasets not installed` | `pip install datasets pyarrow` |
| Out of memory | Use `--subsample-size 50000` |
| Slow generation | Normal for large datasets (10-30 min) |
| Test fails | Check Python/dependencies with `python --version` |

## 📚 Documentation Files

- `HF_DATASET_GUIDE.md` - Comprehensive usage guide
- `HF_IMPLEMENTATION_SUMMARY.md` - Technical implementation details
- `INSTRUCTIONS.md` - General training instructions
- `.github/copilot-instructions.md` - AI agent guide

## 🎯 Production Workflow

```bash
# 1. Install (one-time)
source setup_env.sh

# 2. Generate dataset (one-time, ~20 min)
python generate_data_from_hf.py \
    --output-dir data/sudoku-extreme-aug8 \
    --num-aug 8

# 3. Train (8-12 hours)
python train.py \
    --data-dir data/sudoku-extreme-aug8 \
    --output-dir checkpoints/hf-aug8-absorbing \
    --train-batch-size 64 \
    --num-steps 200000 \
    --loss-mode absorbing

# 4. Evaluate
python inference_puzzles.py \
    --checkpoint checkpoints/hf-aug8-absorbing/best_model.pt \
    --data-dir data/sudoku-extreme-aug8 \
    --num-samples 2000 \
    --steps 10
```

## 💡 Key Points

1. **Vocabulary conversion is automatic** - Don't worry about TRM's 1-10 format
2. **Augmentation applied pre-training** - Not on-the-fly
3. **Test set never augmented** - Ensures fair evaluation
4. **Format auto-detection works** - train.py figures it out
5. **Backward compatible** - Old .npy datasets still work

---

**Ready to go!** Start with `bash test_hf_dataset.sh` to validate everything works.
