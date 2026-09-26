"""One decision rule for validation, the promotion gate and live trading.

Audit 2026-09-25 S8: validation traded sign(logit) on every row, the gate
backtest used another threshold and live gated on ``proba.max() >= 0.45``, so
three different policies were scored. Everything now goes through
``action_proba`` / ``decide``:

    p_buy   = sigmoid(direction_logit)      (BUY vs SELL, trained on tradable rows)
    p_trade = sigmoid(confidence_logit)     (row clears costs; 1.0 if no conf head)
    proba   = [SELL, HOLD, BUY] = [(1 - p_buy) * p_trade, 1 - p_trade, p_buy * p_trade]
    action  = argmax(proba), or HOLD when max(proba) < threshold

Per-pair heads give (B, P) logits; the rule applies per pair.
"""

from __future__ import annotations

import torch

try:
    from config.settings import BACKTEST as _BT
except Exception:  # pragma: no cover
    _BT = {}

DEFAULT_THRESHOLD = float(_BT.get("min_confidence", 0.45) or 0.45)


def action_proba(outputs) -> torch.Tensor:
    """[..., 3] probabilities (SELL, HOLD, BUY) from model outputs."""
    if isinstance(outputs, (tuple, list)):
        logits = outputs[0]
        conf = outputs[2] if len(outputs) > 2 else None
    else:
        logits, conf = outputs, None
    logits = logits.float()
    if logits.dim() >= 2 and logits.shape[-1] == 3 and conf is None:
        return torch.softmax(logits, dim=-1)  # legacy 3-class head
    if logits.dim() == 2 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    p_buy = torch.sigmoid(logits)
    p_trade = torch.sigmoid(conf.float().reshape(logits.shape)) if conf is not None else torch.ones_like(p_buy)
    return torch.stack([(1 - p_buy) * p_trade, 1 - p_trade, p_buy * p_trade], dim=-1)


def decide(outputs, threshold: float | None = None) -> torch.Tensor:
    """Positions in {-1, 0, +1} (float), shaped like the direction logits."""
    thr = DEFAULT_THRESHOLD if threshold is None else float(threshold)
    proba = action_proba(outputs)
    top, idx = proba.max(dim=-1)
    d = idx.float() - 1.0
    return torch.where(top >= thr, d, torch.zeros_like(d))
