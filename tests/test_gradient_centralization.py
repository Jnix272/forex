"""Tests for shared Gradient Centralization helper."""

from __future__ import annotations

import torch
import torch.nn as nn

from training.supervised_loop import _centralize_gradients


def test_centralize_gradients_zeros_weight_mean():
    lin = nn.Linear(4, 3, bias=True)
    # Synthetic grads: weight mean along out-features should become ~0
    lin.weight.grad = torch.ones_like(lin.weight) * 2.0
    lin.bias.grad = torch.ones_like(lin.bias) * 5.0  # dim=1 → untouched

    _centralize_gradients(lin)

    assert torch.allclose(lin.weight.grad.mean(dim=1), torch.zeros(3), atol=1e-6)
    assert torch.allclose(lin.bias.grad, torch.ones_like(lin.bias) * 5.0)


def test_centralize_gradients_accepts_param_iterable():
    a = nn.Linear(2, 2, bias=False)
    b = nn.Linear(2, 2, bias=False)
    a.weight.grad = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    b.weight.grad = torch.ones(2, 2)

    _centralize_gradients([a.weight, b.weight])

    assert torch.allclose(a.weight.grad.mean(dim=1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(b.weight.grad.mean(dim=1), torch.zeros(2), atol=1e-6)


def test_optimizer_step_applies_gc_on_scaler_and_non_scaler_paths():
    """Both AMP (scaler) and non-AMP paths must call the shared GC helper."""
    import inspect

    from training import supervised_loop as sl

    src = inspect.getsource(sl._optimizer_step)
    assert "_centralize_gradients(model)" in src
    # Must not be gated only on the non-scaler branch
    assert "if not use_fp16_scaler:\n        _centralize_gradients" not in src
