"""Unit tests verifying TFT architecture resolves mode collapse and maintains feature variance."""

from __future__ import annotations

import pytest
import torch

from models.architectures import TFTScalper, VariableSelectionNetwork


def test_vsn_weight_magnitude_and_variance():
    """Verify VariableSelectionNetwork does not attenuate features by 1/F."""
    B, T, F = 8, 120, 584
    vsn = VariableSelectionNetwork(input_size=F, hidden=128)
    x = torch.randn(B, T, F)

    x_sel, weights = vsn(x)

    assert x_sel.shape == (B, T, F)
    assert weights.shape == (B, T, F)

    # Weights should be around 1.0 (mean between 0.5 and 1.5, not 1/584 ~ 0.0017)
    mean_weight = weights.mean().item()
    assert 0.5 <= mean_weight <= 1.5, f"Expected mean weight ~1.0, got {mean_weight}"

    # Variance of selected features should be preserved (~1.0, not ~1e-6)
    sel_var = x_sel.var().item()
    assert sel_var >= 0.5, f"Expected selected feature variance >= 0.5, got {sel_var}"


def test_tft_scalper_output_variance_and_gradients():
    """Verify TFTScalper produces healthy output standard deviation and gradient flow."""
    B, T, F = 16, 120, 584
    model = TFTScalper(input_size=F, hidden=128, heads=4, lstm_layers=2, max_seq_len=240)
    x = torch.randn(B, T, F, requires_grad=True)

    out = model(x)
    assert out.shape == (B,), f"Expected shape ({B},), got {out.shape}"

    # Output standard deviation across batch must be non-zero and healthy
    out_std = out.std().item()
    assert out_std >= 0.05, f"Expected output std >= 0.05, got {out_std} (mode collapse check)"

    # Backpropagation check
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0.0, "Gradients must flow back to input features"
    # Gradient norm on input should not be underflowing
    assert x.grad.norm().item() > 1e-5
