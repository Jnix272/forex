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


def decide(
    outputs,
    threshold: float | None = None,
    deadband: float = 0.0,
) -> torch.Tensor:
    """Positions in {-1, 0, +1} (float), shaped like the direction logits.

    Parameters
    ----------
    outputs : tuple | Tensor
        Model predictions. Either (logits, ret_hat, conf, ...) or a Tensor.
    threshold : float | None
        Confidence threshold on max(proba) (default 0.45 from settings).
    deadband : float
        Hurdle rate / conviction deadband:
        - If outputs is a tuple with return predictions ret_hat (outputs[1]):
          requires |ret_hat| > deadband, and ret_hat must agree in sign with
          direction d. Predictions below the hurdle collapse to 0 (HOLD).
        - If outputs is a Tensor (regression):
          requires |outputs| > deadband. Values in [-deadband, +deadband]
          become 0 (HOLD).
    """
    thr = DEFAULT_THRESHOLD if threshold is None else float(threshold)
    proba = action_proba(outputs)
    top, idx = proba.max(dim=-1)
    d = idx.float() - 1.0
    d = torch.where(top >= thr, d, torch.zeros_like(d))

    if deadband > 0.0:
        if isinstance(outputs, (tuple, list)) and len(outputs) > 1 and outputs[1] is not None:
            ret_hat = outputs[1].float()
            if ret_hat.shape != d.shape:
                ret_hat = ret_hat.reshape(d.shape)
            hurdle_mask = (ret_hat.abs() > float(deadband)) & ((ret_hat * d) >= 0)
            d = torch.where(hurdle_mask, d, torch.zeros_like(d))
        elif isinstance(outputs, torch.Tensor):
            out_f = outputs.float()
            d = torch.where(out_f.abs() > float(deadband), torch.sign(out_f), torch.zeros_like(out_f))
    return d

