#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for Sudoku Diffusion Language Model
Compatible with llada_sample.py interface
"""

import sys
import torch
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path to import llada_sample
sys.path.append(str(Path(__file__).parent.parent))
from llada_sample import llada_sample

from model import create_sudoku_dit, DiffusionTransformer


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = create_sudoku_dit(vocab_size=10, seq_length=81)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  Step: {checkpoint.get('step', 'unknown')}")
    print(f"  Best test acc: {checkpoint.get('best_test_acc', 'unknown')}")
    
    return model


def sequence_to_grid(seq):
    """
    Convert 81-token sequence back to 9x9 grid.
    Input: [81 cells]
    Output: 9x9 numpy array
    """
    return np.array(seq[:81]).reshape(9, 9)


def grid_to_sequence(grid):
    """
    Convert 9x9 grid to sequence of 81 tokens.
    No EOL tokens - position embedding encodes structure.
    """
    return grid.flatten().tolist()


def puzzle_to_input(puzzle_grid, mask_id=0):
    """
    Convert puzzle (with 0s for empty cells) to model input.
    In the simplified design, 0 IS the MASK token, so no conversion needed.
    """
    seq = grid_to_sequence(puzzle_grid)
    # No conversion needed! 0 in puzzle = 0 (MASK) in model
    return torch.tensor(seq, dtype=torch.long)


def print_grid(grid, title=""):
    """Pretty print a Sudoku grid."""
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
            if val == 0:
                row_str += ". "
            else:
                row_str += f"{val} "
        row_str += "║"
        print(row_str)
    print("╚═══════╧═══════╧═══════╝")


def solve_sudoku(model, puzzle_grid, device='cuda', steps=10, algorithm='random-remask',
                 temperature=0.0, top_p=None, top_k=None, remask_scheduler='linear'):
    """Solve one or a batch of Sudoku puzzles using the diffusion model.

    Args:
        model: trained DiffusionTransformer
        puzzle_grid: array-like of shape (9, 9) or (B, 9, 9) with 0 denoting empty cells
        device: 'cuda' or 'cpu'
        steps: number of diffusion steps
        algorithm: sampling algorithm (see llada_sample.py)
        temperature: sampling temperature
        top_p: nucleus sampling parameter
        top_k: top-k sampling parameter

    Returns:
        If a single puzzle is provided, returns a (9, 9) numpy array.
        If a batch is provided, returns an array of shape (B, 9, 9).
    """
    MASK_ID = 0  # 0 is the MASK token (empty cells)

    puzzles = np.asarray(puzzle_grid)
    single_input = False
    if puzzles.ndim == 2:  # (9, 9)
        puzzles = puzzles[np.newaxis, ...]
        single_input = True
    elif puzzles.ndim != 3:
        raise ValueError("puzzle_grid must be of shape (9, 9) or (B, 9, 9)")

    batch_size = puzzles.shape[0]
    input_seq = torch.tensor(puzzles.reshape(batch_size, -1), dtype=torch.long, device=device)

    # Create fix_mask (True for given clues, False for unknowns)
    fix_mask = (input_seq != MASK_ID)

    # Run diffusion sampling
    result = llada_sample(
        model=model,
        input_ids=input_seq,
        fix_mask=fix_mask,
        mask_id=MASK_ID,
        attention_mask=None,
        steps=steps,
        algorithm=algorithm,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        remask_scheduler=remask_scheduler,
        return_history=False,
    )

    # Convert output sequence back to grid(s)
    output_seq = result['sequences'].cpu().numpy()
    solutions = np.array([sequence_to_grid(seq) for seq in output_seq])

    return solutions[0] if single_input else solutions


def validate_solution(solution_grid):
    """
    Validate if a Sudoku solution is correct.
    Returns: (is_valid, error_message)
    """
    # Check rows
    for i, row in enumerate(solution_grid):
        if len(set(row)) != 9 or set(row) != set(range(1, 10)):
            return False, f"Invalid row {i+1}: {row}"
    
    # Check columns
    for j in range(9):
        col = solution_grid[:, j]
        if len(set(col)) != 9 or set(col) != set(range(1, 10)):
            return False, f"Invalid column {j+1}: {col}"
    
    # Check 3x3 boxes
    for box_i in range(3):
        for box_j in range(3):
            box = solution_grid[box_i*3:(box_i+1)*3, box_j*3:(box_j+1)*3].flatten()
            if len(set(box)) != 9 or set(box) != set(range(1, 10)):
                return False, f"Invalid box ({box_i+1}, {box_j+1})"
    
    return True, "Valid solution!"


def main():
    parser = argparse.ArgumentParser(description='Solve Sudoku puzzles with Diffusion LM')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    # Input
    parser.add_argument('--puzzle-file', type=str, help='Path to puzzle file (9x9 space-separated)')
    parser.add_argument('--data-dir', type=str, help='Use test set from data directory')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of test samples to solve')
    
    # Sampling
    parser.add_argument('--steps', type=int, default=10, help='Number of diffusion steps')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for batched inference')
    parser.add_argument('--algorithm', type=str, default='self_conf-remask:vanilla',
                       choices=['random-remask', 'self_conf-remask:vanilla', 
                               'self_conf-remask:entropy', 'self_conf-remask:topk_margin'],
                       help='Sampling algorithm')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=None, help='Nucleus sampling')
    parser.add_argument('--top-k', type=int, default=None, help='Top-k sampling')
    parser.add_argument('--remask-scheduler', type=str, default='linear',
                       help='Remask scheduler (linear, cosine, fixed_X where X is ratio)')
    
    args = parser.parse_args()
    
    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device)
    
    # Load puzzles
    if args.puzzle_file:
        # Load single puzzle from file
        puzzle = np.loadtxt(args.puzzle_file, dtype=int)
        puzzles = np.expand_dims(puzzle, axis=0)
        ground_truth = None
        print(f"Loaded puzzle from {args.puzzle_file}")
    elif args.data_dir:
        # Load from test set
        test_puzzles = np.load(Path(args.data_dir) / 'test_puzzles.npy')
        test_solutions = np.load(Path(args.data_dir) / 'test_solutions.npy')

        max_samples = min(args.num_samples, len(test_puzzles))
        puzzles = np.stack([sequence_to_grid(test_puzzles[i]) for i in range(max_samples)])
        ground_truth = np.stack([sequence_to_grid(test_solutions[i]) for i in range(max_samples)])

        print(f"Loaded {len(puzzles)} puzzles from test set")
    else:
        print("Error: Please provide either --puzzle-file or --data-dir")
        return
 
    # Solve puzzles
    print(f"\nSolving {len(puzzles)} puzzle(s) with {args.steps} steps...")
    print(f"Algorithm: {args.algorithm}")
 
    num_correct = 0
    solutions = []
    batch_size = max(1, args.batch_size)
    for start in range(0, len(puzzles), batch_size):
        end = start + batch_size
        batch_puzzles = puzzles[start:end]
        batch_solutions = solve_sudoku(
            model,
            batch_puzzles,
            device,
            steps=args.steps,
            algorithm=args.algorithm,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            remask_scheduler=args.remask_scheduler,
        )
        solutions.append(batch_solutions)
    solutions = np.concatenate(solutions, axis=0)

    from tqdm import tqdm

    for i, (puzzle, solution) in enumerate(tqdm(zip(puzzles, solutions), total=len(puzzles), desc="Evaluating")):
        # print(f"\n{'='*50}")
        # print(f"Puzzle {i+1}/{len(puzzles)}")
        # print_grid(puzzle, "Input Puzzle:")
        
        # print_grid(solution, "Predicted Solution:")
         # Validate
        is_valid, _ = validate_solution(solution)
        if is_valid:
            # print(f"✓ {message}")
            num_correct += 1
        # else:
        #     print(f"✗ {message}")
 
    print(f"\n{'='*50}")
    total = len(puzzles)
    accuracy = (num_correct / total * 100) if total > 0 else 0.0
    print(f"Summary: {num_correct}/{total} valid solutions ({accuracy:.1f}%)")

 
if __name__ == '__main__':
    main()

