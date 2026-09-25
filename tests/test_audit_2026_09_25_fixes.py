"""Regression tests for fixes from docs/AUDIT_2026-09-25_*.md."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest


def test_cost_aware_label_holds_when_neither_side_pays():
    from labeling.cpar_labeling import _cost_aware_label

    assert _cost_aware_label(0.8, -0.9) == 1.0
    assert _cost_aware_label(-0.9, 0.8) == -1.0
    # Mid moved up but the long doesn't clear the spread: HOLD, not BUY.
    assert _cost_aware_label(-0.05, -0.15) == 0.0
    assert _cost_aware_label(0.0, 0.0) == 0.0


def test_seq_schedule_is_ordered_and_inside_run():
    from scripts.optuna_tune import _build_seq_schedule

    sched = _build_seq_schedule(30, 10, 90, total_epochs=6)
    starts = [e["epoch_start"] for e in sched]
    assert starts == sorted(starts)
    assert all(s < 6 for s in starts)
    lens = [e["seq_len"] for e in sched]
    assert lens == sorted(lens)


def test_relative_surprise_is_bounded_and_unit_free():
    from features.engineering.core import _relative_surprise

    df = pl.DataFrame({"actual": ["250K", "3.1%", None, "1"], "forecast": ["180K", "3.0%", "1", None]})
    out = df.select(_relative_surprise("forecast").alias("s"))["s"].to_numpy()
    assert np.all(np.abs(out) <= 1.0)
    assert out[0] > 0 and out[1] > 0
    assert out[2] == 0.0 and out[3] == 0.0


def test_lag_returns_are_log_returns_not_price_levels():
    from features.engineering import microstructure as m

    rng = np.random.default_rng(0)
    steps = rng.normal(0, 1e-4, 400)
    eur = pl.DataFrame({"close": 1.1 * np.exp(np.cumsum(steps))})
    jpy = pl.DataFrame({"close": 150.0 * np.exp(np.cumsum(steps))})
    r_eur = eur.select(m.lag_returns([20]))["ret_20"].to_numpy()
    r_jpy = jpy.select(m.lag_returns([20]))["ret_20"].to_numpy()
    ok = np.isfinite(r_eur) & np.isfinite(r_jpy)
    # Same relative moves -> same feature, whatever the price level (JPY-safe).
    np.testing.assert_allclose(r_eur[ok], r_jpy[ok], atol=1e-6)
    assert np.nanmax(np.abs(r_eur)) < 200  # bps, not ~20 x price
    rsi = eur.select(m.rsi(14))["rsi_14"].to_numpy()
    assert 20 < np.nanmean(rsi) < 80


def test_asia_london_gap_has_no_same_day_lookahead():
    from features.multipair import compute_asia_london_gap

    ts = pd.date_range("2024-01-02 00:00", "2024-01-03 23:55", freq="5min", tz="UTC")
    close = np.linspace(1.0, 2.0, len(ts))
    gap = compute_asia_london_gap(pd.DataFrame({"timestamp_utc": ts, "close": close})).to_numpy()
    day2_asia = (ts.date == ts[-1].date()) & (ts.hour < 7)
    day1_london_last = np.where((ts.date == ts[0].date()) & (ts.hour >= 7))[0][-1]
    # Day-2 Asia rows carry day 1's last gap; they can't know day 2's Asia close.
    assert np.allclose(gap[day2_asia], gap[day1_london_last])


def test_neutralize_price_level_columns():
    from sklearn.preprocessing import RobustScaler

    from training.dataset_builder import neutralize_price_level_columns

    X = np.random.default_rng(1).normal(size=(200, 3)) + [100.0, 0.0, 5.0]
    sc = RobustScaler().fit(X)
    hit = neutralize_price_level_columns(sc, ["EURUSD::close", "EURUSD::rsi_14", "USDJPY::bb_upper"])
    assert hit == ["EURUSD::close", "USDJPY::bb_upper"]
    out = sc.transform(X)
    assert np.abs(out[:, 0]).max() < 1e-6 and np.abs(out[:, 2]).max() < 1e-6
    assert np.abs(out[:, 1]).max() > 0.1


def test_merge_scalers_merges_robust_scalers():
    from sklearn.preprocessing import RobustScaler

    from training.dataset_builder import _merge_scalers

    a = RobustScaler().fit(np.zeros((50, 2)) + [0.0, 1.0] + np.random.default_rng(2).normal(size=(50, 2)))
    b = RobustScaler().fit(np.zeros((50, 2)) + [10.0, 1.0] + np.random.default_rng(3).normal(size=(50, 2)))
    m = _merge_scalers([a, b])
    assert m is not a and m is not b
    assert 3.0 < m.center_[0] < 7.0


def test_meta_learner_affine_and_legacy_state_dict():
    torch = pytest.importorskip("torch")
    from models.ensemble import EnsembleMetaLearner

    class Base(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k
            self.lin = torch.nn.Linear(4, 1)

        def forward(self, x):
            return x[:, -1, 0] * self.k

    meta = EnsembleMetaLearner([Base(1.0), Base(2.0)], context_dim=4, hidden=8)
    legacy = {k: v for k, v in meta.state_dict().items() if k not in ("out_scale", "out_bias")}
    fresh = EnsembleMetaLearner([Base(1.0), Base(2.0)], context_dim=4, hidden=8)
    fresh.load_state_dict(legacy)  # old checkpoints load as identity calibration
    assert float(fresh.out_scale) == 1.0 and float(fresh.out_bias) == 0.0

    x = torch.ones(3, 5, 4) * 7.0
    fresh.attach_base_scalers([(np.full(4, 7.0), np.ones(4)), (np.full(4, 7.0), np.ones(4))])
    out, w = fresh(x)
    # Both bases now see (7 - 7) / 1 = 0 in feature 0.
    assert torch.allclose(out, torch.zeros(3), atol=1e-6)
