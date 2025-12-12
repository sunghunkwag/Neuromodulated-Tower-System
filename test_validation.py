#!/usr/bin/env python3
"""PyTorch Validation Test for Neuromodulated Tower System.

This script validates the 5-tower neuromodulated system implementation.
Run: python test_validation.py
"""

import sys
import torch
import torch.nn as nn

from src.system import FiveTowerSystem


def print_header(title):
    """Print formatted header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def test_system_init():
    """Test 1: System initialization."""
    print_header("TEST 1: System Initialization")
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}")
        
        system = FiveTowerSystem(
            input_dim=256,
            hidden_dim=128,
            latent_dim=128,
            output_dim=4,
            device=device
        )
        
        total_params = sum(p.numel() for p in system.parameters())
        print(f"Total parameters: {total_params:,}")
        print("✓ System initialized successfully")
        return system, device
    
    except Exception as e:
        print(f"✗ FAILED: {e}")
        sys.exit(1)


def test_forward_pass(system, device):
    """Test 2: Forward pass."""
    print_header("TEST 2: Forward Pass")
    
    try:
        batch_size = 2
        state = torch.randn(batch_size, 256, device=device)
        
        with torch.no_grad():
            action, debug = system(state)
        
        assert action.shape == (batch_size, 4), f"Action shape mismatch: {action.shape}"
        assert -1 <= action.min() and action.max() <= 1, "Action out of range [-1, 1]"
        
        print(f"Input shape: {state.shape}")
        print(f"Output shape: {action.shape}")
        print(f"Output range: [{action.min().item():.4f}, {action.max().item():.4f}]")
        print("✓ Forward pass successful")
        return action, debug
    
    except Exception as e:
        print(f"✗ FAILED: {e}")
        sys.exit(1)


def test_tower_outputs(debug):
    """Test 3: Tower outputs."""
    print_header("TEST 3: 5-Tower Outputs")
    
    try:
        tower_outputs = debug['tower_outputs']
        assert len(tower_outputs) == 5, f"Expected 5 towers, got {len(tower_outputs)}"
        
        for i, out in enumerate(tower_outputs, 1):
            print(f"  Tower {i}: shape {out.shape}, dtype {out.dtype}")
        
        print("✓ All 5 towers working")
    
    except Exception as e:
        print(f"✗ FAILED: {e}")
        sys.exit(1)


def test_hormones(debug):
    """Test 4: Hormone generation."""
    print_header("TEST 4: Hormone System (Tower 3)")
    
    try:
        hormones = debug['hormones']
        required = {'dopamine', 'serotonin', 'cortisol'}
        assert set(hormones.keys()) == required, "Missing hormone keys"
        
        for name, val in hormones.items():
            assert 0 <= val.min() and val.max() <= 1, f"{name} out of [0, 1]"
            print(f"  {name:12s}: {val.item():.4f}")
        
        print("✓ 3-hormone system working")
    
    except Exception as e:
        print(f"✗ FAILED: {e}")
        sys.exit(1)


def test_nt_gates(debug):
    """Test 5: Neurotransmitter gates."""
    print_header("TEST 5: Neurotransmitter-Gated Integration")
    
    try:
        nt_weights = debug['nt_weights']
        required = {'NET', 'DAT', '5HTT'}
        assert set(nt_weights.keys()) == required, "Missing NT pathway keys"
        
        for nt_name, weights in nt_weights.items():
            # Check softmax constraint
            sums = weights.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
                f"{nt_name} weights don't sum to 1"
            print(f"  {nt_name}: shape {weights.shape}")
        
        print("✓ NT-gating layer working (3 pathways)")
    
    except Exception as e:
        print(f"✗ FAILED: {e}")
        sys.exit(1)


def test_gradients(system, device):
    """Test 6: Gradient flow."""
    print_header("TEST 6: Gradient Flow & Backprop")
    
    try:
        state = torch.randn(1, 256, device=device, requires_grad=True)
        action, _ = system(state)
        loss = action.sum()
        loss.backward()
        
        assert state.grad is not None, "No gradient for input"
        assert not torch.isnan(state.grad).any(), "NaN in gradients"
        
        grad_norm = state.grad.norm().item()
        print(f"Input gradient norm: {grad_norm:.6f}")
        print(f"Loss value: {loss.item():.6f}")
        print("✓ Gradient backpropagation working")
    
    except Exception as e:
        print(f"✗ FAILED: {e}")
        sys.exit(1)


def test_batch_processing(system, device):
    """Test 7: Batch processing."""
    print_header("TEST 7: Batch Processing Flexibility")
    
    try:
        batch_sizes = [1, 2, 4, 8]
        for bs in batch_sizes:
            state = torch.randn(bs, 256, device=device)
            action, _ = system(state)
            assert action.shape[0] == bs, f"Batch size mismatch"
            print(f"  Batch size {bs}: ✓")
        
        print("✓ All batch sizes working")
    
    except Exception as e:
        print(f"✗ FAILED: {e}")
        sys.exit(1)


def test_numerical_stability(system, device):
    """Test 8: Numerical stability under extreme inputs."""
    print_header("TEST 8: Numerical Stability")
    
    try:
        # Test 1: Very small inputs
        small_state = torch.randn(1, 256, device=device) * 1e-6
        action_small, debug_small = system(small_state)
        assert not torch.isnan(action_small).any(), "NaN with small inputs"
        assert not torch.isinf(action_small).any(), "Inf with small inputs"
        print(f"  Small input (1e-6): ✓")
        
        # Test 2: Large inputs
        large_state = torch.randn(1, 256, device=device) * 1e3
        action_large, debug_large = system(large_state)
        assert not torch.isnan(action_large).any(), "NaN with large inputs"
        assert not torch.isinf(action_large).any(), "Inf with large inputs"
        print(f"  Large input (1e3): ✓")
        
        # Test 3: Zero input
        zero_state = torch.zeros(1, 256, device=device)
        action_zero, debug_zero = system(zero_state)
        assert not torch.isnan(action_zero).any(), "NaN with zero input"
        print(f"  Zero input: ✓")
        
        print("✓ Numerical stability verified")
    
    except Exception as e:
        print(f"✗ FAILED: {e}")
        sys.exit(1)


def test_hormone_modulation(system, device):
    """Test 9: Hormone modulation effect on NT gates."""
    print_header("TEST 9: Hormone-NT Gate Interaction")
    
    try:
        # Run multiple forward passes
        states = [torch.randn(1, 256, device=device) for _ in range(5)]
        hormone_levels = []
        nt_gate_outputs = []
        
        for state in states:
            with torch.no_grad():
                _, debug = system(state)
            hormone_levels.append(debug['hormones'])
            nt_gate_outputs.append(debug['nt_weights'])
        
        # Check that NT weights vary with different hormone levels
        net_weights_list = [nts['NET'] for nts in nt_gate_outputs]
        net_weights_stack = torch.stack(net_weights_list)
        variance = net_weights_stack.var(dim=0).mean()
        
        assert variance > 1e-6, "NT weights not varying (might indicate no modulation)"
        print(f"  NET weight variance: {variance:.6f}")
        print("✓ Hormone modulation active")
    
    except Exception as e:
        print(f"✗ FAILED: {e}")
        sys.exit(1)


def test_tower_independence(system, device):
    """Test 10: Tower independence verification."""
    print_header("TEST 10: Tower Independence")
    
    try:
        state = torch.randn(1, 256, device=device)
        _, debug = system(state)
        tower_outputs = debug['tower_outputs']
        
        # Check that tower outputs are different
        for i in range(len(tower_outputs) - 1):
            for j in range(i + 1, len(tower_outputs)):
                diff = (tower_outputs[i] - tower_outputs[j]).abs().mean()
                assert diff > 1e-3, f"Tower {i+1} and {j+1} outputs too similar: {diff}"
        
        print(f"  Tower output diversity verified")
        print("✓ All towers processing independently")
    
    except Exception as e:
        print(f"✗ FAILED: {e}")
        sys.exit(1)


def main():
    """Run all tests."""
    print("\n" + "#"*70)
    print("#  Neuromodulated-Tower-System: Enhanced Validation Suite")
    print("#"*70)
    
    system, device = test_system_init()
    action, debug = test_forward_pass(system, device)
    test_tower_outputs(debug)
    test_hormones(debug)
    test_nt_gates(debug)
    test_gradients(system, device)
    test_batch_processing(system, device)
    test_numerical_stability(system, device)
    test_hormone_modulation(system, device)
    test_tower_independence(system, device)
    
    print_header("ALL TESTS PASSED ✅")
    print(f"Device used: {device}")
    print(f"Total parameters: {sum(p.numel() for p in system.parameters()):,}")
    print(f"\nSystem validation completed successfully!\n")


if __name__ == '__main__':
    main()
