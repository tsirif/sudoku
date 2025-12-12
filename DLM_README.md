# Sudoku Diffusion Language Model

A lightweight **29M parameter** Diffusion Transformer trained from scratch to solve Sudoku puzzles using discrete diffusion modeling.

![Architecture Overview](https://img.shields.io/badge/Model-DiT-blue) ![Parameters](https://img.shields.io/badge/Params-29M-green) ![Training](https://img.shields.io/badge/Training-100k_steps-orange)

## Overview

This project implements a **Diffusion Language Model (DLM)** specifically designed for Sudoku puzzle solving. Unlike autoregressive models, diffusion models can iteratively refine solutions through parallel denoising steps, making them well-suited for constraint satisfaction problems like Sudoku.

### Key Features

- ✅ **Compact Architecture**: Only 29M parameters
- ✅ **Efficient Training**: ~530 epochs on 48k puzzles
- ✅ **Dynamic Masking**: Trains on varying mask ratios (0.2-0.9)
- ✅ **Compatible Inference**: Works with `llada_sample.py` interface
- ✅ **Parallel Generation**: Non-autoregressive decoding

## Architecture

### Model Specifications

| Component | Configuration |
|-----------|--------------|
| **Model Type** | Diffusion Transformer (DiT) |
| **Parameters** | 29.0M |
| **Layers** | 12 Transformer blocks |
| **Hidden Dim** | 448 |
| **Attention Heads** | 8 |
| **FFN Ratio** | 4 |
| **Dropout** | 0.1 |

### Input Representation

Sudoku puzzles are encoded as sequences of **89 tokens**:
- **81 cell tokens**: digits 0-9 (0 = empty/unknown)
- **8 EOL tokens**: row separators (token ID = 10)

**Vocabulary**: 12 tokens total
- `0-9`: Sudoku digits (0 for empty cells)
- `10`: End-of-line (EOL) separator
- `11`: MASK token for diffusion training

**Example Encoding**:
```
Input Grid (9×9):
5 3 0 | 0 7 0 | 0 0 0
6 0 0 | 1 9 5 | 0 0 0
...

Sequence (89 tokens):
[5,3,0,0,7,0,0,0,0, 10, 6,0,0,1,9,5,0,0,0, 10, ...]
 └─────── row 1 ────┘ EOL └─────── row 2 ────┘ EOL
```

## Installation

### Requirements

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install numpy tqdm tensorboard PyYAML

# For data generation (from parent directory)
pip install -r requirements.txt
```

### Project Structure

```
sudoku/
├── generate_data.py      # Generate training data
├── model.py              # Diffusion Transformer architecture
├── train.py              # Training script
├── inference.py          # Solve puzzles with trained model
├── train_config.yaml     # Training configuration
├── DLM_README.md         # This file
├── data/                 # Dataset (generated)
│   ├── train_puzzles.npy
│   ├── train_solutions.npy
│   ├── test_puzzles.npy
│   └── test_solutions.npy
└── checkpoints/          # Model checkpoints (saved during training)
    ├── best_model.pt
    ├── final_model.pt
    └── tensorboard/
```

## Quick Start

### 1. Generate Dataset

Generate 48,000 training puzzles and 2,000 test puzzles:

```bash
cd sudoku
python generate_data.py --train-size 48000 --test-size 2000 --output-dir ./data
```

This creates:
- `data/train_puzzles.npy`, `data/train_solutions.npy`
- `data/test_puzzles.npy`, `data/test_solutions.npy`

**Estimated time**: ~20-30 minutes (with multiprocessing)

### 2. Train Model

Train the 28.6M parameter Diffusion Transformer:

```bash
python train.py \
    --data-dir ./data \
    --output-dir ./checkpoints \
    --batch-size 128 \
    --num-steps 100000 \
    --lr 1e-4 \
    --mask-ratio-min 0.2 \
    --mask-ratio-max 0.9
```

**Key Training Parameters**:
- `--batch-size 128`: Batch size (adjust based on GPU memory)
- `--num-steps 100000`: Total training steps (~530 epochs)
- `--lr 1e-4`: Learning rate with cosine annealing
- `--mask-ratio-min/max`: Dynamic masking range

**Training Time**: ~8-12 hours on a single A100 GPU

**Monitoring**: View training progress with TensorBoard:
```bash
tensorboard --logdir ./checkpoints/tensorboard
```

### 3. Solve Puzzles

Solve puzzles from the test set:

```bash
python inference.py \
    --checkpoint ./checkpoints/best_model.pt \
    --data-dir ./data \
    --num-samples 10 \
    --steps 10 \
    --algorithm random-remask
```

**Sampling Algorithms** (compatible with `llada_sample.py`):
- `random-remask`: Random remasking strategy (default)
- `self_conf-remask:vanilla`: Confidence-based remasking (vanilla)
- `self_conf-remask:entropy`: Entropy-based confidence
- `self_conf-remask:topk_margin`: Top-k margin confidence

Solve a custom puzzle:

```bash
# Create puzzle file (9 lines, 9 space-separated digits, 0 for empty)
echo "5 3 0 0 7 0 0 0 0
6 0 0 1 9 5 0 0 0
0 9 8 0 0 0 0 6 0
8 0 0 0 6 0 0 0 3
4 0 0 8 0 3 0 0 1
7 0 0 0 2 0 0 0 6
0 6 0 0 0 0 2 8 0
0 0 0 4 1 9 0 0 5
0 0 0 0 8 0 0 7 9" > puzzle.txt

python inference.py \
    --checkpoint ./checkpoints/best_model.pt \
    --puzzle-file puzzle.txt \
    --steps 10
```

## Training Details

### Diffusion Training Process

1. **Forward Diffusion**: Randomly mask 20-90% of tokens in ground truth solutions
2. **Denoising**: Model predicts masked tokens given context
3. **Loss**: Cross-entropy on masked positions only
4. **Dynamic Masking**: Mask ratio varies per batch to improve robustness

### Training Configuration

```yaml
# From train_config.yaml
Batch Size:      128
Training Steps:  100,000 (~530 epochs)
Learning Rate:   1e-4 (cosine decay to 1e-5)
Optimizer:       AdamW (β₁=0.9, β₂=0.999)
Weight Decay:    0.01
Dropout:         0.1
Mask Ratio:      Uniform[0.2, 0.9] per sample
Gradient Clip:   1.0
```

### Expected Results

| Metric | Value |
|--------|-------|
| **Training Accuracy** | ~99%+ (on masked tokens) |
| **Test Accuracy** | ~95%+ (on masked tokens) |
| **Solution Validity** | ~80-90% (valid Sudoku solutions) |

*Note: Accuracy measures token-level predictions; validity measures complete Sudoku constraint satisfaction.*

## Inference Methods

### Compatible with `llada_sample.py`

The model is fully compatible with the diffusion sampling interface:

```python
from llada_sample import llada_sample
from model import create_sudoku_dit

# Load model
model = create_sudoku_dit()
model.load_state_dict(torch.load('checkpoints/best_model.pt')['model_state_dict'])
model.eval()

# Prepare input (puzzle with MASK tokens)
input_ids = torch.tensor([[5,3,11,11,7,11, ...]])  # 11 = MASK
fix_mask = (input_ids != 11)  # True for given clues

# Run diffusion sampling
result = llada_sample(
    model=model,
    input_ids=input_ids,
    fix_mask=fix_mask,
    mask_id=11,
    steps=10,
    algorithm='random-remask',
    temperature=0.0,
)

solution = result['sequences']
```

### Sampling Parameters

- **steps**: Number of iterative refinement steps (default: 10)
  - More steps → better quality but slower
  - Typically 5-20 steps is sufficient
  
- **temperature**: Sampling temperature (default: 0.0)
  - 0.0 = greedy (deterministic)
  - >0.0 = stochastic sampling
  
- **algorithm**: Remasking strategy
  - `random-remask`: Random selection
  - `self_conf-remask:*`: Confidence-based selection

## Model Architecture Details

### Diffusion Transformer (DiT)

```
Input: token_ids [batch, 89]
  ↓
Token Embedding [batch, 89, 512]
  + 
Position Embedding [1, 89, 512] (learnable)
  ↓
12× Transformer Block:
  ├─ LayerNorm
  ├─ Multi-Head Self-Attention (8 heads, 64 dim/head)
  ├─ Residual Connection
  ├─ LayerNorm
  ├─ FFN (512 → 2048 → 512)
  └─ Residual Connection
  ↓
LayerNorm
  ↓
Output Head [batch, 89, 12] (vocab logits)
```

**Parameter Count Breakdown**:
```
Token Embedding:     12 × 448    = 5,376
Position Embedding:  89 × 448    = 39,872
12× Transformer:     ~2.4M each  = 28.9M
Output Head:         448 × 12    = 5,376
─────────────────────────────────────────
Total:                             29.0M
```

## Tips & Tricks

### For Better Results

1. **More Training Steps**: Increase `--num-steps` to 150k-200k
2. **Larger Batch Size**: Use `--batch-size 256` if GPU memory allows
3. **More Inference Steps**: Use `--steps 20` for higher quality
4. **Ensemble**: Run multiple times with different seeds and vote

### For Faster Training

1. **Mixed Precision**: Add AMP (automatic mixed precision) in `train.py`
2. **Gradient Accumulation**: Simulate larger batch sizes
3. **Multi-GPU**: Use `torch.nn.DataParallel` or DDP

### Memory Optimization

If OOM (out of memory) errors occur:
```bash
# Reduce batch size
python train.py --batch-size 64 ...

# Or reduce model size
# Edit model.py: hidden_dim=384, num_layers=10
```

## Citation

If you use this code, please cite:

```bibtex
@software{sudoku_diffusion_lm,
  title={Sudoku Diffusion Language Model},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/sudoku-diffusion-lm}
}
```

## License

MIT License

## Acknowledgments

- Diffusion sampling interface inspired by `llada_sample.py`
- Sudoku generation from `advanced_sudoku_generator.py`
- Architecture inspired by DiT (Peebles & Xie, 2023)

---

**Questions or Issues?** Open an issue on GitHub or contact the author.

