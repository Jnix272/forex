"""Regression tests for docs/AUDIT_2026-09-25_backtest_gate.md fixes."""

from __future__ import annotations

import json

import numpy as np
import pytest


def test_psr_uses_per_observation_scale():
    from validation.promotion_gate import probabilistic_sharpe_ratio

    # Annualised Sharpe 0.5 from 300 five-minute trades per year is weak evidence;
    # unconverted, PSR came out ~1.0.
    ppy = 288 * 260 / 30
    psr = probabilistic_sharpe_ratio(0.5, 0.0, 300, periods_per_year=ppy)
    assert 0.5 < psr < 0.95
    assert probabilistic_sharpe_ratio(0.5, 0.0, 300) > 0.99  # the old, wrong reading


def test_gate_fails_closed_without_periods_per_year():
    from validation.promotion_gate import PromotionGate

    r = PromotionGate().evaluate(
        sharpe=3.0, profit_factor=2.0, max_drawdown=0.05, n_trades=2000, gross_pnl=1.0, n_obs=2000
    )
    assert r["gates"]["psr_ok"] is False and r["promoted"] is False


def test_deflation_grows_with_trials():
    from validation.promotion_gate import deflated_sharpe_ratio

    ppy = 288 * 260 / 30
    d1 = deflated_sharpe_ratio(2.0, 1, 2000, periods_per_year=ppy)
    d500 = deflated_sharpe_ratio(2.0, 500, 2000, periods_per_year=ppy)
    assert d500 < d1


def test_certificate_bound_to_artifact_hash(tmp_path):
    from validation.gate_policy import certificate_from_gate, check_gate_artifact

    ckpt = tmp_path / "m_best.pt"
    ckpt.write_bytes(b"weights-v1")
    cert = certificate_from_gate({"promoted": True, "gates": {"a": True}}, [ckpt])
    doc = json.loads(json.dumps(cert))
    assert check_gate_artifact(doc) == (True, "ok")
    ckpt.write_bytes(b"weights-v2")  # retrained / swapped checkpoint
    ok, why = check_gate_artifact(doc)
    assert not ok and "changed" in why
    unbound = dict(doc, artifact_hashes={})
    assert check_gate_artifact(unbound)[0] is False


def test_trade_stats_profit_factor_finite():
    from training.honest_eval import trade_stats

    assert trade_stats(np.array([0.01, 0.02]))["profit_factor"] == 100.0
    assert trade_stats(np.array([]))["profit_factor"] == 0.0
    st = trade_stats(np.array([0.02, -0.01, -0.01]))
    assert st["profit_factor"] == pytest.approx(1.0) and st["max_drawdown"] > 0


def test_holdout_gate_metrics_scores_cached_rows(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("zarr")
    from common.cache_io import _zarr_create, _zarr_open_group
    from training.honest_eval import holdout_gate_metrics

    n, h = 600, 5
    rng = np.random.default_rng(3)
    close = np.cumprod(1 + rng.normal(0, 1e-3, n)).astype(np.float32)
    fwd = np.sign(np.r_[close[h:] - close[:-h], np.zeros(h)])
    zs = _zarr_open_group(str(tmp_path / "c.zarr"), mode="w")
    _zarr_create(zs, "X", shape=(0, 4, 2), chunks=(64, 4, 2))
    X = np.zeros((n, 4, 2), dtype=np.float32)
    X[:, -1, 0] = fwd  # perfect-foresight feature
    zs["X"].append(X)
    for k, v in (("close", close), ("spread", np.full(n, 1e-6, dtype=np.float32))):
        _zarr_create(zs, k, shape=(0,), chunks=(64,))
        zs[k].append(v)

    class Oracle(torch.nn.Module):
        def forward(self, x):
            return x[:, -1, 0]

    m = holdout_gate_metrics(Oracle(), str(tmp_path / "c.zarr"), np.arange(300, n - h), horizon=h)
    # Realised trades per year: at most one per horizon.
    assert m["n_trades"] > 20 and m["sharpe"] > 0 and 0 < m["periods_per_year"] <= 288 * 260 / h + 1e-6


def test_backtest_pip_value_for_usd_base_pair():
    pd = pytest.importorskip("pandas")
    from backtesting.backtest import ForexScalingBacktest

    bars = pd.DataFrame({"open": [150.0] * 10, "high": [150.1] * 10, "low": [149.9] * 10, "close": [150.0] * 10})
    sig = pd.DataFrame({"action": [0] * 10, "stop_loss": [0.0] * 10, "take_profit": [0.0] * 10})
    bt = ForexScalingBacktest(bars, sig, pair="USDJPY")
    assert bt.pip_size == 0.01
    assert bt.pip_value_per_lot == pytest.approx(10_000 * 0.01 / 150.0)
