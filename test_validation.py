#!/usr/bin/env python3
"""PyTorch Validation Test for Neuromodulated Tower System.

This script validates the 5-tower neuromodulated system implementation.
Run: python test_validation.py
"""

import contextlib
import pytest
import torch
from src.system import FiveTowerSystem


def print_header(title):
    """Print formatted header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


@contextlib.contextmanager
def eval_mode(module):
    """Temporarily switch a module to eval mode."""
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        if was_training:
            module.train()


def run_forward_pass(system, device, batch_size=2):
    """Run a deterministic forward pass in eval mode."""
    state = torch.randn(batch_size, 256, device=device)

    with torch.inference_mode():
        with eval_mode(system):
            action, debug = system(state)

    return action, debug, state


@pytest.fixture(scope="session", autouse=True)
def configure_determinism():
    """Set global determinism for repeatable CI runs."""
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@pytest.fixture(scope="module")
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(scope="module")
def system(device):
    return FiveTowerSystem(
        input_dim=256,
        hidden_dim=128,
        latent_dim=128,
        output_dim=4,
        device=device
    )


@pytest.fixture(autouse=True)
def reset_rng():
    """Keep random behavior deterministic across tests."""
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    yield


@pytest.fixture
def forward_result(system, device):
    return run_forward_pass(system, device)


def test_system_init(system, device):
    """Test 1: System initialization."""
    print_header("TEST 1: System Initialization")

    total_params = sum(p.numel() for p in system.parameters())
    print(f"Device: {device}")
    print(f"Total parameters: {total_params:,}")
    print("✓ System initialized successfully")


def test_forward_pass(forward_result):
    """Test 2: Forward pass."""
    print_header("TEST 2: Forward Pass")

    action, debug, state = forward_result

    assert action.shape == (state.shape[0], 4), f"Action shape mismatch: {action.shape}"
    assert -1 <= action.min() and action.max() <= 1, "Action out of range [-1, 1]"

    print(f"Input shape: {state.shape}")
    print(f"Output shape: {action.shape}")
    print(f"Output range: [{action.min().item():.4f}, {action.max().item():.4f}]")
    print("✓ Forward pass successful")


def test_tower_outputs(forward_result):
    """Test 3: Tower outputs."""
    print_header("TEST 3: 5-Tower Outputs")

    _, debug, _ = forward_result
    tower_outputs = debug['tower_outputs']
    assert len(tower_outputs) == 5, f"Expected 5 towers, got {len(tower_outputs)}"

    for i, out in enumerate(tower_outputs, 1):
        print(f"  Tower {i}: shape {out.shape}, dtype {out.dtype}")

    print("✓ All 5 towers working")


def test_hormones(forward_result):
    """Test 4: Hormone generation."""
    print_header("TEST 4: Hormone System (Tower 3)")

    _, debug, _ = forward_result
    hormones = debug['hormones']
    required = {'dopamine', 'serotonin', 'cortisol'}
    assert set(hormones.keys()) == required, "Missing hormone keys"

    for name, val in hormones.items():
        assert 0 <= val.min() and val.max() <= 1, f"{name} out of [0, 1]"
        val_scalar = float(val.mean()) if isinstance(val, torch.Tensor) else float(val)
        print(f" {name:12s}: {val_scalar:.4f}")
    print("✓ 3-hormone system working")


def test_nt_gates(forward_result):
    """Test 5: Neurotransmitter gates."""
    print_header("TEST 5: Neurotransmitter-Gated Integration")

    _, debug, _ = forward_result
    nt_weights = debug['nt_weights']
    required = {'NET', 'DAT', '5HTT', 'combined'}
    assert set(nt_weights.keys()) == required, "Missing NT pathway keys"

    for nt_name, weights in nt_weights.items():
        if nt_name == 'combined':
            assert torch.isfinite(weights).all(), "NaN or inf in combined weights"
            assert (weights >= 0).all(), "Negative values in combined weights"
        else:
            sums = weights.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
                f"{nt_name} weights don't sum to 1"
        print(f"  {nt_name}: shape {weights.shape}")

    print("✓ NT-gating layer working (3 pathways)")


def test_gradients(system, device):
    """Test 6: Gradient flow."""
    print_header("TEST 6: Gradient Flow & Backprop")

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


def test_batch_processing(system, device):
    """Test 7: Batch processing."""
    print_header("TEST 7: Batch Processing Flexibility")

    batch_sizes = [1, 2, 4, 8]
    for bs in batch_sizes:
        state = torch.randn(bs, 256, device=device)
        action, _ = system(state)
        assert action.shape[0] == bs, "Batch size mismatch"
        print(f"  Batch size {bs}: ✓")

    print("✓ All batch sizes working")


def test_numerical_stability(system, device):
    """Test 8: Numerical stability under extreme inputs."""
    print_header("TEST 8: Numerical Stability")

    # Test 1: Very small inputs
    small_state = torch.randn(1, 256, device=device) * 1e-6
    action_small, _ = system(small_state)
    assert not torch.isnan(action_small).any(), "NaN with small inputs"
    assert not torch.isinf(action_small).any(), "Inf with small inputs"
    print(f"  Small input (1e-6): ✓")

    # Test 2: Large inputs
    large_state = torch.randn(1, 256, device=device) * 1e3
    action_large, _ = system(large_state)
    assert not torch.isnan(action_large).any(), "NaN with large inputs"
    assert not torch.isinf(action_large).any(), "Inf with large inputs"
    print(f"  Large input (1e3): ✓")

    # Test 3: Zero input
    zero_state = torch.zeros(1, 256, device=device)
    action_zero, _ = system(zero_state)
    assert not torch.isnan(action_zero).any(), "NaN with zero input"
    print(f"  Zero input: ✓")

    print("✓ Numerical stability verified")


def test_hormone_modulation(system, device):
    """Test 9: Hormone modulation effect on NT gates."""
    print_header("TEST 9: Hormone-NT Gate Interaction")

    # Run multiple forward passes
    states = [torch.randn(1, 256, device=device) for _ in range(5)]
    nt_gate_outputs = []

    for state in states:
        with torch.no_grad():
            _, debug = system(state)
        nt_gate_outputs.append(debug['nt_weights'])

    # Check that NT weights vary with different hormone levels
    net_weights_list = [nts['NET'] for nts in nt_gate_outputs]
    net_weights_stack = torch.stack(net_weights_list)
    variance = net_weights_stack.var(dim=0).mean()

    assert variance > 1e-6, "NT weights not varying (might indicate no modulation)"
    print(f"  NET weight variance: {variance:.6f}")
    print("✓ Hormone modulation active")


def test_tower_independence(system, device):
    """Test 10: Tower independence verification."""
    print_header("TEST 10: Tower Independence")

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


def main():
    """Run the validation suite via pytest for parity with CI."""
    print("\n" + "#" * 70)
    print("#  Neuromodulated-Tower-System: Enhanced Validation Suite")
    print("#" * 70)
    print("Launching pytest runner...\n")

    # Execute the same collection path CI uses so local runs match pipelines.
    raise_on_fail = pytest.main([__file__])
    if raise_on_fail != 0:
        raise SystemExit(raise_on_fail)

    print_header("ALL TESTS PASSED ✅")
    print("Validation suite completed successfully!\n")


if __name__ == '__main__':
    main()
