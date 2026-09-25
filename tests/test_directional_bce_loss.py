"""
tests/test_directional_bce_loss.py
==================================
Unit tests for Auxiliary Directional Loss:
1. Verifies that DirectionalHuberLoss delivers non-vanishing gradients when predictions
   have the wrong sign on micro-targets (y = +/- 1e-4), where standard Huber gradient ~ 0.
2. Verifies MultiTaskLoss incorporates directional BCE loss on logits.
"""

import pytest
import torch
import torch.nn as nn
from training.gpu_losses import DirectionalHuberLoss
from models.architectures import MultiTaskLoss


def test_directional_huber_bce_non_vanishing_gradient():
    """Verify that DirectionalHuberLoss produces non-zero gradient on micro sign errors."""
    crit_huber = nn.HuberLoss()
    crit_dir = DirectionalHuberLoss(direction_weight=0.5, bce_weight=0.5)

    # Micro target: market went down by 0.0001 (-1 pip)
    target = torch.tensor([-0.0001], dtype=torch.float32)

    # Pred 1 with standard Huber (wrong sign: predicted +0.0001)
    pred_huber = torch.tensor([0.0001], dtype=torch.float32, requires_grad=True)
    loss_h = crit_huber(pred_huber, target)
    loss_h.backward()
    grad_huber = pred_huber.grad.item()

    # Pred 2 with DirectionalHuberLoss (wrong sign: predicted +0.0001)
    pred_dir = torch.tensor([0.0001], dtype=torch.float32, requires_grad=True)
    loss_d = crit_dir(pred_dir, target)
    loss_d.backward()
    grad_dir = pred_dir.grad.item()

    # Standard Huber gradient is only ~2e-4 (virtually vanishing)
    assert abs(grad_huber) < 0.001
    # Directional BCE loss delivers steep gradient (> 0.1) pulling prediction downward
    assert grad_dir > 0.1, f"Expected strong positive gradient to pull prediction down, got {grad_dir}"
    assert grad_dir > abs(grad_huber) * 100


def test_directional_huber_correct_direction_lower_loss():
    """Verify that predictions with correct direction have strictly lower loss."""
    crit = DirectionalHuberLoss(direction_weight=0.5, bce_weight=0.5)
    target = torch.tensor([0.0005], dtype=torch.float32)

    pred_correct = torch.tensor([0.0005], dtype=torch.float32)
    pred_wrong = torch.tensor([-0.0005], dtype=torch.float32)

    loss_correct = crit(pred_correct, target).item()
    loss_wrong = crit(pred_wrong, target).item()

    assert loss_correct < loss_wrong


def test_multitask_loss_directional_bce():
    """Verify MultiTaskLoss includes directional BCE penalty."""
    mt_loss = MultiTaskLoss(w_dir=1.0, w_ret=0.5, w_conf=0.3)

    target = torch.tensor([0.0010], dtype=torch.float32)
    ret_hat = torch.tensor([0.0010], dtype=torch.float32)
    conf = torch.tensor([1.0], dtype=torch.float32)

    # Logit matching direction (positive)
    logits_pos = torch.tensor([2.0], dtype=torch.float32)
    loss_pos = mt_loss(logits_pos, ret_hat, conf, None, target).item()

    # Logit contradicting direction (negative)
    logits_neg = torch.tensor([-2.0], dtype=torch.float32)
    loss_neg = mt_loss(logits_neg, ret_hat, conf, None, target).item()

    assert loss_neg > loss_pos, f"Expected higher loss for wrong directional logit: {loss_neg} vs {loss_pos}"
