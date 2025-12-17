#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Sudoku dataset from HuggingFace hub (sapientinc/sudoku-extreme)
Converts vocabulary from TinyRecursiveModels format (1-10) to sudoku format (0-9)
Applies augmentation and saves as HuggingFace dataset or numpy arrays
"""

import os
import csv
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from datasets import Dataset, DatasetDict


def convert_vocabulary(arr):
    """
    Convert from TinyRecursiveModels format to sudoku format.
    
    TRM: 0=PAD (unused), 1=blank, 2-10=digits 1-9
    Sudoku: 0=MASK, 1-9=digits
    
    Args:
        arr: numpy array with values in range [1, 10]
    
    Returns:
        arr: numpy array with values in range [0, 9]
    """
    # Subtract 1: 1→0 (blank/MASK), 2→1, 3→2, ..., 10→9
    return arr - 1


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray, seed=None):
    """
    Apply symmetry-preserving augmentation to Sudoku puzzles.
    Adapted from TinyRecursiveModels to work with 0-9 vocabulary.
    
    Applies:
    - Random digit permutation (1-9, keeping 0=MASK unchanged)
    - Random transposition
    - Random row permutation (within 3×3 bands)
    - Random column permutation (within 3×3 stacks)
    
    Args:
        board: (9, 9) puzzle with 0=MASK, 1-9=digits
        solution: (9, 9) complete solution with 1-9=digits only
        seed: random seed for reproducibility
    
    Returns:
        augmented_board, augmented_solution
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create digit mapping: permute 1-9, keep 0 (MASK) unchanged
    digit_map = np.zeros(10, dtype=np.int64)
    digit_map[0] = 0  # MASK stays MASK
    digit_map[1:] = np.random.permutation(np.arange(1, 10))
    
    # Randomly decide whether to transpose
    transpose_flag = np.random.rand() < 0.5
    
    # Generate valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows)
    # - Within each band, shuffle its 3 rows
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])
    
    # Similarly for columns (stacks)
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])
    
    # Build 81→81 position mapping
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])
    
    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose if selected
        if transpose_flag:
            x = x.T
        # Apply position mapping
        new_grid = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_grid]
    
    return apply_transformation(board), apply_transformation(solution)


def validate_sudoku(solution: np.ndarray) -> bool:
    """
    Validate that a sudoku solution is correct.
    
    Args:
        solution: (9, 9) array with values 1-9
    
    Returns:
        True if valid, False otherwise
    """
    # Check all values are 1-9
    if not np.all((solution >= 1) & (solution <= 9)):
        return False
    
    # Check rows
    for i in range(9):
        if len(set(solution[i, :])) != 9:
            return False
    
    # Check columns
    for j in range(9):
        if len(set(solution[:, j])) != 9:
            return False
    
    # Check 3×3 boxes
    for box_i in range(3):
        for box_j in range(3):
            box = solution[box_i*3:(box_i+1)*3, box_j*3:(box_j+1)*3].flatten()
            if len(set(box)) != 9:
                return False
    
    return True


def load_csv_data(csv_path, min_difficulty=None):
    """
    Load puzzles and solutions from CSV file.
    
    CSV format: source,quizzes,solutions,rating
    - quizzes: 81-char string with '.' for blank
    - solutions: 81-char string with digits
    - rating: difficulty rating
    
    Args:
        csv_path: path to CSV file
        min_difficulty: minimum difficulty rating (None = no filter)
    
    Returns:
        inputs: list of (9,9) puzzle arrays (TRM format: 1=blank, 2-10=digits)
        labels: list of (9,9) solution arrays (TRM format: 2-10=digits)
    """
    inputs = []
    labels = []
    
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 4:
                continue
            
            source, q, a, rating = row[0], row[1], row[2], row[3]
            
            # Filter by difficulty if specified
            if min_difficulty is not None and int(rating) < min_difficulty:
                continue
            
            if len(q) != 81 or len(a) != 81:
                print(f"Warning: Invalid puzzle length, skipping")
                continue
            
            # Parse puzzle (. → '0', then convert to int)
            puzzle_str = q.replace('.', '0')
            puzzle = np.frombuffer(puzzle_str.encode(), dtype=np.uint8).reshape(9, 9) - ord('0')
            
            # Parse solution
            solution = np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord('0')
            
            # In TRM format: they add 1 to everything to get 1-10 range
            # We'll convert later, but for now match their format
            puzzle = puzzle + 1  # 0→1 (blank), 1-9→2-10 (digits)
            solution = solution + 1  # 1-9→2-10 (digits)
            
            inputs.append(puzzle)
            labels.append(solution)
    
    return inputs, labels


def process_dataset(
    set_name,
    source_repo,
    output_dir,
    num_aug=0,
    min_difficulty=None,
    subsample_size=None,
    seed=42,
    save_format='huggingface'
):
    """
    Process a dataset split (train or test).
    
    Args:
        set_name: 'train' or 'test'
        source_repo: HuggingFace repo ID
        output_dir: output directory
        num_aug: number of augmentations per sample (train only)
        min_difficulty: minimum difficulty filter
        subsample_size: subsample training set to this size (before augmentation)
        seed: random seed
        save_format: 'huggingface' or 'npy'
    """
    np.random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Processing {set_name} set")
    print(f"{'='*60}")
    
    # Download CSV from HuggingFace
    print(f"Downloading from {source_repo}...")
    csv_path = hf_hub_download(source_repo, f"{set_name}.csv", repo_type="dataset")
    
    # Load and filter data
    print(f"Loading data (min_difficulty={min_difficulty})...")
    inputs, labels = load_csv_data(csv_path, min_difficulty=min_difficulty)
    print(f"Loaded {len(inputs)} puzzles")
    
    # Subsample training set if requested
    if set_name == "train" and subsample_size is not None:
        if subsample_size < len(inputs):
            print(f"Subsampling to {subsample_size} puzzles...")
            indices = np.random.choice(len(inputs), size=subsample_size, replace=False)
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]
    
    # Convert from TRM format (1-10) to sudoku format (0-9)
    print("Converting vocabulary (TRM 1-10 → sudoku 0-9)...")
    inputs = [convert_vocabulary(inp) for inp in inputs]
    labels = [convert_vocabulary(lbl) for lbl in labels]
    
    # Validate conversion
    print("Validating conversions...")
    for i, solution in enumerate(labels[:10]):  # Check first 10
        if not validate_sudoku(solution):
            print(f"WARNING: Invalid solution at index {i}")
    
    # Generate augmentations
    num_augments = num_aug if set_name == "train" else 0
    print(f"Generating augmentations ({num_augments} per sample)...")
    
    all_puzzles = []
    all_solutions = []
    all_puzzle_identifiers = []
    all_puzzle_indices = []
    all_group_indices = []
    all_is_augmented = []
    all_augmentation_ids = []
    
    puzzle_idx = 0
    group_idx = 0
    
    for orig_puzzle, orig_solution in tqdm(zip(inputs, labels), total=len(inputs)):
        # Original (non-augmented)
        all_puzzles.append(orig_puzzle.flatten())
        all_solutions.append(orig_solution.flatten())
        all_puzzle_identifiers.append(0)
        all_puzzle_indices.append(puzzle_idx)
        all_group_indices.append(group_idx)
        all_is_augmented.append(False)
        all_augmentation_ids.append(0)
        puzzle_idx += 1
        
        # Augmented versions
        for aug_id in range(1, num_augments + 1):
            aug_puzzle, aug_solution = shuffle_sudoku(
                orig_puzzle.copy(), 
                orig_solution.copy(),
                seed=seed + puzzle_idx * 1000 + aug_id  # Deterministic but unique
            )
            
            # Validate augmentation
            if not validate_sudoku(aug_solution):
                print(f"WARNING: Invalid augmented solution at group {group_idx}, aug {aug_id}")
                continue
            
            all_puzzles.append(aug_puzzle.flatten())
            all_solutions.append(aug_solution.flatten())
            all_puzzle_identifiers.append(0)
            all_puzzle_indices.append(puzzle_idx)
            all_group_indices.append(group_idx)
            all_is_augmented.append(True)
            all_augmentation_ids.append(aug_id)
            puzzle_idx += 1
        
        group_idx += 1
    
    print(f"Generated {len(all_puzzles)} total samples ({len(inputs)} original + {len(all_puzzles) - len(inputs)} augmented)")
    
    # Convert to numpy arrays
    puzzles_array = np.array(all_puzzles, dtype=np.int16)
    solutions_array = np.array(all_solutions, dtype=np.int16)
    puzzle_identifiers = np.array(all_puzzle_identifiers, dtype=np.int32)
    puzzle_indices = np.array(all_puzzle_indices, dtype=np.int32)
    group_indices = np.array(all_group_indices, dtype=np.int32)
    is_augmented = np.array(all_is_augmented, dtype=bool)
    augmentation_ids = np.array(all_augmentation_ids, dtype=np.int32)
    
    # Save dataset
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if save_format == 'huggingface':
        # Save as HuggingFace dataset
        print(f"Saving as HuggingFace dataset to {output_path / set_name}...")
        
        dataset = Dataset.from_dict({
            'puzzle': puzzles_array,
            'solution': solutions_array,
            'puzzle_identifier': puzzle_identifiers,
            'puzzle_index': puzzle_indices,
            'group_index': group_indices,
            'is_augmented': is_augmented,
            'augmentation_id': augmentation_ids,
        })
        
        dataset.save_to_disk(str(output_path / set_name))
        
    elif save_format == 'npy':
        # Save as numpy arrays (backward compatible)
        print(f"Saving as numpy arrays to {output_path}...")
        
        np.save(output_path / f'{set_name}_puzzles.npy', puzzles_array)
        np.save(output_path / f'{set_name}_solutions.npy', solutions_array)
        np.save(output_path / f'{set_name}_puzzle_identifiers.npy', puzzle_identifiers)
        np.save(output_path / f'{set_name}_puzzle_indices.npy', puzzle_indices)
        np.save(output_path / f'{set_name}_group_indices.npy', group_indices)
        np.save(output_path / f'{set_name}_is_augmented.npy', is_augmented)
        np.save(output_path / f'{set_name}_augmentation_ids.npy', augmentation_ids)
        
        # Save metadata
        metadata = {
            'num_samples': len(all_puzzles),
            'num_original': len(inputs),
            'num_augmented': len(all_puzzles) - len(inputs),
            'num_augmentations_per_sample': num_augments,
            'seq_length': 81,
            'vocab_size': 10,  # 0: MASK, 1-9: digits
            'format': '81 cells (0=MASK, 1-9=digits), flat array',
            'split': set_name,
            'source': source_repo,
            'min_difficulty': min_difficulty,
            'subsample_size': subsample_size if set_name == 'train' else None,
            'seed': seed,
        }
        
        with open(output_path / f'{set_name}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved {set_name} dataset")
    print(f"  Total samples: {len(all_puzzles)}")
    print(f"  Original: {len(inputs)}")
    print(f"  Augmented: {len(all_puzzles) - len(inputs)}")
    
    return len(all_puzzles), len(inputs)


def main():
    parser = argparse.ArgumentParser(
        description='Generate Sudoku dataset from HuggingFace (sapientinc/sudoku-extreme)'
    )
    
    # Data source
    parser.add_argument(
        '--source-repo',
        type=str,
        default='sapientinc/sudoku-extreme',
        help='HuggingFace dataset repository'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for processed dataset'
    )
    
    # Augmentation
    parser.add_argument(
        '--num-aug',
        type=int,
        default=0,
        help='Number of augmentations per training sample (test set not augmented)'
    )
    
    # Filtering
    parser.add_argument(
        '--min-difficulty',
        type=int,
        default=None,
        help='Minimum difficulty rating (None = use all puzzles)'
    )
    
    parser.add_argument(
        '--subsample-size',
        type=int,
        default=None,
        help='Subsample training set to this size (before augmentation)'
    )
    
    # System
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--save-format',
        type=str,
        choices=['huggingface', 'npy'],
        default='huggingface',
        help='Save format: huggingface (arrow files) or npy (numpy arrays)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Sudoku Dataset Generator from HuggingFace")
    print("="*60)
    print(f"Source: {args.source_repo}")
    print(f"Output: {args.output_dir}")
    print(f"Augmentations: {args.num_aug} per training sample")
    print(f"Min difficulty: {args.min_difficulty}")
    print(f"Subsample size: {args.subsample_size}")
    print(f"Save format: {args.save_format}")
    print(f"Seed: {args.seed}")
    print("="*60)
    
    # Process train set
    train_total, train_orig = process_dataset(
        set_name='train',
        source_repo=args.source_repo,
        output_dir=args.output_dir,
        num_aug=args.num_aug,
        min_difficulty=args.min_difficulty,
        subsample_size=args.subsample_size,
        seed=args.seed,
        save_format=args.save_format,
    )
    
    # Process test set (no augmentation, no subsampling)
    test_total, test_orig = process_dataset(
        set_name='test',
        source_repo=args.source_repo,
        output_dir=args.output_dir,
        num_aug=0,  # Never augment test set
        min_difficulty=args.min_difficulty,
        subsample_size=None,  # Never subsample test set
        seed=args.seed,
        save_format=args.save_format,
    )
    
    # Save dataset dict for HuggingFace format
    if args.save_format == 'huggingface':
        print("\nCreating DatasetDict...")
        dataset_dict = DatasetDict({
            'train': Dataset.load_from_disk(str(Path(args.output_dir) / 'train')),
            'test': Dataset.load_from_disk(str(Path(args.output_dir) / 'test')),
        })
        
        dataset_dict.save_to_disk(args.output_dir)
        print(f"✓ Saved DatasetDict to {args.output_dir}")
    
    print("\n" + "="*60)
    print("Dataset generation complete!")
    print("="*60)
    print(f"Train: {train_total} samples ({train_orig} original)")
    print(f"Test:  {test_total} samples ({test_orig} original)")
    print(f"Output: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
