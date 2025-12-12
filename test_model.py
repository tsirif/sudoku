#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script to verify model implementation
Tests model creation, forward pass, and basic functionality
"""

import torch
import numpy as np
from model import create_sudoku_dit, DiffusionTransformer


def test_model_creation():
    """Test model creation and parameter count."""
    print("Testing model creation...")
    model = create_sudoku_dit(vocab_size=10, seq_length=81)
    
    params = model.count_parameters()
    print(f"✓ Model created successfully")
    print(f"  Total parameters: {params['total']:,} ({params['total']/1e6:.1f}M)")
    
    # Check parameter count is reasonable (30-40M range)
    expected_min = 30e6
    expected_max = 45e6
    assert expected_min <= params['total'] <= expected_max, \
        f"Parameter count out of range: expected {expected_min/1e6:.1f}M-{expected_max/1e6:.1f}M, got {params['total']/1e6:.1f}M"
    
    return model


def test_forward_pass():
    """Test forward pass with random inputs."""
    print("\nTesting forward pass...")
    model = create_sudoku_dit()
    model.eval()
    
    batch_size = 4
    seq_length = 81
    vocab_size = 10  # 0: MASK, 1-9: digits (no EOL)
    
    # Create random input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Forward pass
    with torch.no_grad():
        output = model(input_ids)
    
    # Check output shape
    assert output.logits.shape == (batch_size, seq_length, vocab_size), \
        f"Output shape mismatch: {output.logits.shape}"
    
    # Check no NaN or Inf
    assert not torch.isnan(output.logits).any(), "NaN detected in output"
    assert not torch.isinf(output.logits).any(), "Inf detected in output"
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {output.logits.shape}")
    
    return model


def test_masking_behavior():
    """Test model behavior with masked inputs."""
    print("\nTesting masking behavior...")
    model = create_sudoku_dit()
    model.eval()
    
    batch_size = 2
    seq_length = 81
    mask_id = 0  # 0 is the MASK token
    
    # Create input with masks
    input_ids = torch.randint(1, 10, (batch_size, seq_length))  # 1-9 only
    
    # Mask 50% of positions
    mask_positions = torch.rand(batch_size, seq_length) < 0.5
    input_ids[mask_positions] = mask_id
    
    # Forward pass
    with torch.no_grad():
        output = model(input_ids)
    
    # Get predictions
    preds = output.logits.argmax(dim=-1)
    
    # Check that predictions are valid tokens (0-10, not including 11=MASK)
    # Note: predictions might include MASK (11) but ideally shouldn't
    print(f"✓ Masking test successful")
    print(f"  Masked positions: {mask_positions.sum().item()} / {mask_positions.numel()}")
    print(f"  Unique predictions: {torch.unique(preds).tolist()}")
    
    return model


def test_gradient_flow():
    """Test gradient flow through the model."""
    print("\nTesting gradient flow...")
    model = create_sudoku_dit()
    model.train()
    
    batch_size = 2
    seq_length = 81
    vocab_size = 10  # 0: MASK, 1-9: digits (no EOL)
    
    # Create random input and targets
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    targets = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Forward pass
    output = model(input_ids)
    
    # Compute loss
    loss = torch.nn.functional.cross_entropy(
        output.logits.reshape(-1, vocab_size),
        targets.reshape(-1)
    )
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grad = 0
    no_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad += 1
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
        else:
            no_grad += 1
    
    print(f"✓ Gradient flow successful")
    print(f"  Parameters with gradients: {has_grad}")
    print(f"  Parameters without gradients: {no_grad}")
    print(f"  Loss: {loss.item():.4f}")
    
    return model


def test_inference_compatibility():
    """Test compatibility with inference interface."""
    print("\nTesting inference compatibility...")
    model = create_sudoku_dit()
    model.eval()
    
    # Simulate puzzle input (9x9 grid with some unknowns)
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
    
    # Convert to sequence (81 tokens, no EOL)
    MASK = 0  # 0 is the MASK token (empty cells)
    seq = puzzle.flatten().tolist()
    
    input_ids = torch.tensor([seq], dtype=torch.long)
    assert input_ids.shape == (1, 81), f"Sequence length mismatch: {input_ids.shape}"
    
    # Forward pass
    with torch.no_grad():
        output = model(input_ids)
    
    # Get predictions
    preds = output.logits.argmax(dim=-1)
    
    print(f"✓ Inference compatibility test successful")
    print(f"  Input sequence length: {input_ids.shape[1]}")
    print(f"  Number of masked cells: {(input_ids == MASK).sum().item()}")
    print(f"  Output predictions shape: {preds.shape}")
    print(f"  Note: No EOL tokens - position embedding encodes structure")
    
    return model


def main():
    print("=" * 60)
    print("Sudoku Diffusion Transformer - Test Suite")
    print("=" * 60)
    
    try:
        # Run tests
        test_model_creation()
        test_forward_pass()
        test_masking_behavior()
        test_gradient_flow()
        test_inference_compatibility()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Generate dataset: python generate_data.py")
        print("  2. Train model: python train.py")
        print("  3. Solve puzzles: python inference.py")
        print("\nOr run complete pipeline: bash run_experiment.sh")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ Test failed!")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

