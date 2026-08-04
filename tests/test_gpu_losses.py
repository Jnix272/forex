"""Unit tests for trading-aware GPU losses."""
from __future__ import annotations

import argparse

import pytest
import torch

from training.gpu_losses import SharpeProxyLoss, soft_direction


def test_soft_direction_bounds_and_grad_at_large_pred():
    pred = torch.tensor([-50.0, -2.0, 0.0, 2.0, 50.0], requires_grad=True)
    out = soft_direction(pred)
    assert torch.all(out.abs() < 1.0)
    assert float(out[2].detach()) == 0.0
    out.sum().backward()
    # softsign grad = 1/(1+|x|)^2 — still material at |x|=50 (~4e-4), unlike tanh (~0)
    assert pred.grad is not None
    assert float(pred.grad[0].abs()) > 1e-5
    assert float(pred.grad[-1].abs()) > 1e-5


def test_sharpe_proxy_uses_softsign_not_tanh_saturation():
    """Confident preds must still move the Sharpe term (non-vanishing grads)."""
    loss_fn = SharpeProxyLoss(delta=1.0, sharpe_weight=1.0, ann=1.0)
    pred = torch.tensor([20.0, 25.0, -18.0, -22.0], requires_grad=True)
    target = torch.tensor([1.0, 1.0, -1.0, -1.0])
    loss = loss_fn(pred, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert float(pred.grad.abs().mean()) > 1e-6


def test_apply_yaml_maps_distillation_student_when_enabled(tmp_path):
    from training.gpu_cli import _apply_yaml_config

    cfg = tmp_path / "kd.yaml"
    cfg.write_text(
        "distillation:\n"
        "  enabled: true\n"
        "  student_model: tft\n"
        "  teacher_model: mamba\n"
        "  alpha: 0.4\n",
        encoding="utf-8",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--model", dest="model", default="haelt")
    p.add_argument("--teacher-model", dest="teacher_model", default=None)
    p.add_argument("--distill-weight", dest="distill_weight", type=float, default=0.5)
    _apply_yaml_config(p, str(cfg))
    args, _ = p.parse_known_args([])
    assert args.model == "tft"
    assert args.teacher_model == "mamba"
    assert args.distill_weight == pytest.approx(0.4)


def test_apply_yaml_skips_distillation_when_disabled(tmp_path):
    from training.gpu_cli import _apply_yaml_config

    cfg = tmp_path / "kd_off.yaml"
    cfg.write_text(
        "distillation:\n"
        "  enabled: false\n"
        "  student_model: tft\n"
        "  teacher_model: mamba\n",
        encoding="utf-8",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--model", dest="model", default="haelt")
    p.add_argument("--teacher-model", dest="teacher_model", default=None)
    _apply_yaml_config(p, str(cfg))
    args, _ = p.parse_known_args([])
    assert args.model == "haelt"
    assert args.teacher_model is None
