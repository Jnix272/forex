"""Trading-aware loss functions used by GPU supervised training."""
from __future__ import annotations

import torch
import torch.nn as nn

from models.architectures import HuberLoss

# Non-finite targets seen by match_target_shape (never silently zeroed).
_MATCH_SHAPE_NONFINITE = 0


def match_target_shape(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return target reshaped to pred for scalar regression heads.

    Does **not** silently ``nan_to_num`` targets to 0 — callers must drop or
    fail on non-finite labels (see ``_sanitize_batch_tensors``).
    """
    global _MATCH_SHAPE_NONFINITE
    target = target.float()
    if not torch.isfinite(target).all():
        n_bad = int((~torch.isfinite(target)).sum().item())
        _MATCH_SHAPE_NONFINITE += 1
        if _MATCH_SHAPE_NONFINITE <= 3 or _MATCH_SHAPE_NONFINITE % 50 == 0:
            print(
                f"[match_target_shape] WARN: {n_bad} non-finite target value(s) "
                f"(left unchanged; count={_MATCH_SHAPE_NONFINITE})"
            )
    if pred.shape == target.shape:
        return target
    if pred.ndim == 2 and pred.shape[-1] == 1 and target.ndim == 1:
        return target.unsqueeze(-1)
    return target


# Back-compat alias used throughout train_gpu.py
_match_target_shape = match_target_shape


def soft_direction(pred: torch.Tensor) -> torch.Tensor:
    """Map continuous predictions to (-1, 1) without tanh saturation.

    Softsign ``x / (1 + |x|)`` keeps usable gradients for large |pred|
    (decay ~1/x²) unlike ``tanh``, which vanishes exponentially and stalls
    the Sharpe proxy once the model becomes confident.
    """
    return pred / (1.0 + pred.abs())


class DirectionalHuberLoss(nn.Module):
    """Huber magnitude loss + extra penalty when direction is wrong."""

    def __init__(self, delta: float = 1.0, direction_weight: float = 0.5):
        super().__init__()
        self.huber = HuberLoss(delta=delta)
        self.direction_weight = float(direction_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = match_target_shape(pred, target)
        base = self.huber(pred, target)
        wrong_sign = (pred * target) < 0
        dir_pen = wrong_sign.float() * (pred - target).abs()
        return base + self.direction_weight * dir_pen.mean()


class SharpeProxyLoss(nn.Module):
    """Minimize -Sharpe proxy while keeping pointwise stability via Huber.

    The annualized Sharpe is stored for logging, but the gradient term uses the
    config-supplied ``ann`` scale directly (already a sqrt-style factor from
    config) so the Sharpe component stays in the same magnitude band as Huber.
    """

    def __init__(
        self,
        delta: float = 1.0,
        sharpe_weight: float = 0.2,
        eps: float = 1e-8,
        ann: float = 1.0,
    ):
        super().__init__()
        self.huber = HuberLoss(delta=delta)
        self.sharpe_weight = float(sharpe_weight)
        self.eps = float(eps)
        self.ann = float(ann)
        self._ann_sqrt = float(ann)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight=None) -> torch.Tensor:
        target = match_target_shape(pred, target)
        try:
            base = self.huber(pred, target, weight=weight)
        except TypeError:
            base = self.huber(pred, target)
        direction = soft_direction(pred)
        returns = (direction * target).flatten()
        mean = returns.mean()
        var = returns.var(unbiased=False)
        std = torch.sqrt(var + self.eps)
        sharpe_gradient = (mean / std) * self._ann_sqrt
        return base - self.sharpe_weight * sharpe_gradient
