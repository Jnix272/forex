"""Regression tests for docs/AUDIT_2026-09-25_sharpe_loss.md fixes."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _loss(**kw):
    from models.architectures import MultiTaskLoss

    return MultiTaskLoss(w_quantile=0.0, **kw)


def test_unknown_loss_options_raise():
    from models.architectures import MultiTaskLoss

    with pytest.raises(TypeError):
        MultiTaskLoss(direction_weight_floor=0.3)


def test_direction_uses_y_cls_and_masks_hold():
    loss = _loss(w_ret=0.0, w_conf=0.0)
    logits = torch.tensor([5.0, 5.0])
    y_cont = torch.tensor([1.0, 1.0])  # mid moved up on both rows...
    # ...but row 0 is SELL after costs and row 1 is HOLD (masked out).
    y_cls = torch.tensor([0, 1])
    bad = loss(logits, torch.zeros(2), torch.zeros(2), y_cls, y_cont)
    good = loss(-logits, torch.zeros(2), torch.zeros(2), y_cls, y_cont)
    assert good < bad


def test_confidence_learns_tradability():
    loss = _loss(w_dir=0.0, w_ret=0.0, w_conf=1.0)
    y_cls = torch.tensor([1, 1, 2, 0])
    conf_right = torch.tensor([-5.0, -5.0, 5.0, 5.0])
    assert loss(torch.zeros(4), torch.zeros(4), conf_right, y_cls, torch.ones(4)) < 0.05


def test_bet_size_weights_apply():
    loss = _loss(w_dir=0.0, w_conf=0.0, w_ret=1.0, target_clip=0.0)
    y = torch.tensor([0.0, 4.0])
    args = (torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.tensor([2, 2]), y)
    l_even = loss(*args, bet_size=torch.tensor([1.0, 1.0]))
    l_first = loss(*args, bet_size=torch.tensor([1.0, 0.0]))
    assert l_first < l_even


def test_regression_target_winsorised():
    loss = _loss(w_dir=0.0, w_conf=0.0, w_ret=1.0, target_clip=5.0)
    args = (torch.zeros(1), torch.tensor([5.0]), torch.zeros(1), torch.tensor([2]))
    assert loss(*args, torch.tensor([50.0])) == pytest.approx(loss(*args, torch.tensor([5.0])).item())


def test_decide_holds_when_not_confident():
    from training.decision import decide

    logits = torch.tensor([3.0, -3.0, 3.0])
    conf = torch.tensor([3.0, 3.0, -3.0])  # third row: not tradable
    d = decide((logits, torch.zeros(3), conf))
    assert d.tolist() == [1.0, -1.0, 0.0]


def test_honest_annualises_by_realised_trades():
    from training.honest_eval import net_pnl_metrics

    n, h = 20_000, 30
    rng = np.random.default_rng(0)
    close = np.cumprod(1 + rng.normal(0, 1e-4, n + h))
    dirs = np.zeros(n)
    dirs[::300] = 1.0  # trades rarely (1 in 10 horizons)
    m = net_pnl_metrics(dirs, np.arange(n), close, None, h, min_trades=5, extra_cost_pips=0.0)
    full = 288 * 260 / h
    assert m["trades_per_year"] < 0.2 * full


def test_pooled_counts_one_period_per_timestamp():
    from training.honest_eval import pooled_pair_metrics

    n, h = 3000, 5
    rng = np.random.default_rng(1)
    base = np.cumprod(1 + rng.normal(0, 1e-3, n + h))
    close = np.stack([base, base], axis=1)  # perfectly correlated pairs
    d = np.ones((n, 2))
    m = pooled_pair_metrics(d, np.arange(n), close, None, h, min_trades=5)
    assert m["n_portfolio_periods"] * 2 == m["n_trades"]
