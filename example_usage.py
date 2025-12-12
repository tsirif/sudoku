#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of Sudoku Diffusion Language Model
Demonstrates how to use the model for inference
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent directory for llada_sample
sys.path.append(str(Path(__file__).parent.parent))
from llada_sample import llada_sample
from model import create_sudoku_dit


def grid_to_sequence(grid):
    """Convert 9x9 grid to 81-token sequence (no EOL)."""
    return grid.flatten().tolist()


def sequence_to_grid(seq):
    """Convert 81-token sequence to 9x9 grid."""
    return np.array(seq[:81]).reshape(9, 9)


def print_grid(grid, title=""):
    """Pretty print Sudoku grid."""
    if title:
        print(f"\n{title}")
    print("╔═══════╤═══════╤═══════╗")
    for i, row in enumerate(grid):
        if i > 0 and i % 3 == 0:
            print("╟───────┼───────┼───────╢")
        row_str = "║ "
        for j, val in enumerate(row):
            if j > 0 and j % 3 == 0:
                row_str += "│ "
            row_str += f"{val if val != 0 else '.'} "
        row_str += "║"
        print(row_str)
    print("╚═══════╧═══════╧═══════╝")


def example_1_basic_inference():
    """Example 1: Basic inference with random model."""
    print("=" * 70)
    print("Example 1: Basic Inference (Untrained Model)")
    print("=" * 70)
    
    # Create model (randomly initialized)
    model = create_sudoku_dit()
    model.eval()
    
    # Example puzzle (0 = unknown)
    puzzle = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ])
    
    print_grid(puzzle, "Input Puzzle:")
    
    # Convert to sequence (0s are already MASK tokens, no conversion needed)
    MASK_ID = 0  # 0 is the MASK token
    seq = grid_to_sequence(puzzle)
    input_seq = torch.tensor([seq], dtype=torch.long)  # No conversion needed!
    
    # Create fix_mask (True for given clues)
    fix_mask = (input_seq != MASK_ID)
    
    print(f"\nRunning diffusion sampling (10 steps)...")
    print(f"  Masked positions: {(~fix_mask).sum().item()}")
    
    # Run diffusion sampling
    result = llada_sample(
        model=model,
        input_ids=input_seq,
        fix_mask=fix_mask,
        mask_id=MASK_ID,
        steps=10,
        algorithm='random-remask',
        temperature=0.0,
        return_history=False,
    )
    
    # Convert back to grid
    output_seq = result['sequences'][0].cpu().numpy()
    solution = sequence_to_grid(output_seq)
    
    print_grid(solution, "Model Output (Untrained):")
    print("\nNote: This is an untrained model, so output will be random.")
    print("After training, the model should produce valid Sudoku solutions.")


def example_2_with_checkpoint():
    """Example 2: Load and use a trained checkpoint."""
    print("\n" + "=" * 70)
    print("Example 2: Using Trained Checkpoint")
    print("=" * 70)
    
    checkpoint_path = "./checkpoints/best_model.pt"
    
    if not Path(checkpoint_path).exists():
        print(f"\n⚠️  Checkpoint not found: {checkpoint_path}")
        print("Train a model first with: python train.py")
        print("Then this example will load the trained weights.")
        return
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create and load model
    model = create_sudoku_dit()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from step {checkpoint.get('step', 'unknown')}")
    print(f"  Best test accuracy: {checkpoint.get('best_test_acc', 'unknown')}")
    
    # Test on a puzzle
    puzzle = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ])
    
    print_grid(puzzle, "Input Puzzle:")
    
    # Solve
    MASK_ID = 0  # 0 is the MASK token
    seq = grid_to_sequence(puzzle)
    input_seq = torch.tensor([seq], dtype=torch.long)  # No conversion needed!
    fix_mask = (input_seq != MASK_ID)
    
    print(f"\nSolving with trained model...")
    
    result = llada_sample(
        model=model,
        input_ids=input_seq,
        fix_mask=fix_mask,
        mask_id=MASK_ID,
        steps=10,
        algorithm='random-remask',
        temperature=0.0,
    )
    
    output_seq = result['sequences'][0].cpu().numpy()
    solution = sequence_to_grid(output_seq)
    
    print_grid(solution, "Predicted Solution:")
    
    # Validate
    def is_valid_sudoku(grid):
        for i in range(9):
            if len(set(grid[i, :])) != 9 or set(grid[i, :]) != set(range(1, 10)):
                return False
            if len(set(grid[:, i])) != 9 or set(grid[:, i]) != set(range(1, 10)):
                return False
        for bi in range(3):
            for bj in range(3):
                box = grid[bi*3:(bi+1)*3, bj*3:(bj+1)*3].flatten()
                if len(set(box)) != 9 or set(box) != set(range(1, 10)):
                    return False
        return True
    
    if is_valid_sudoku(solution):
        print("\n✅ Valid Sudoku solution!")
    else:
        print("\n❌ Invalid solution (training may not be complete)")


def example_3_sampling_strategies():
    """Example 3: Different sampling strategies."""
    print("\n" + "=" * 70)
    print("Example 3: Comparing Sampling Strategies")
    print("=" * 70)
    
    model = create_sudoku_dit()
    model.eval()
    
    puzzle = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ])
    
    MASK_ID = 0  # 0 is the MASK token
    seq = grid_to_sequence(puzzle_grid)
    input_seq = torch.tensor([seq], dtype=torch.long)  # No conversion needed!
    fix_mask = (input_seq != MASK_ID)
    
    algorithms = [
        'random-remask',
        'self_conf-remask:vanilla',
        'self_conf-remask:entropy',
    ]
    
    print_grid(puzzle, "Input Puzzle:")
    
    for algo in algorithms:
        print(f"\n{'─' * 70}")
        print(f"Algorithm: {algo}")
        print(f"{'─' * 70}")
        
        result = llada_sample(
            model=model,
            input_ids=input_seq.clone(),
            fix_mask=fix_mask,
            mask_id=MASK_ID,
            steps=5,
            algorithm=algo,
            temperature=0.0,
        )
        
        output_seq = result['sequences'][0].cpu().numpy()
        solution = sequence_to_grid(output_seq)
        
        print(f"First 3 rows of output:")
        for row in solution[:3]:
            print("  " + " ".join(str(x) for x in row))
        
        print(f"Actual steps: {result['actual_steps'][0].item()}")


def main():
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "Sudoku Diffusion Language Model" + " " * 22 + "║")
    print("║" + " " * 24 + "Example Usage" + " " * 31 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Run examples
    example_1_basic_inference()
    example_2_with_checkpoint()
    example_3_sampling_strategies()
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nFor full functionality:")
    print("  1. Generate data: python generate_data.py")
    print("  2. Train model: python train.py")
    print("  3. Solve puzzles: python inference.py --checkpoint ./checkpoints/best_model.pt")


if __name__ == '__main__':
    main()

