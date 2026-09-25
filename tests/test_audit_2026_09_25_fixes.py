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


def test_bootstrap_sharpe_ci_brackets_point_estimate():
    from training.honest_eval import bootstrap_sharpe_ci

    r = np.random.default_rng(4).normal(0.001, 0.01, 400)
    ann = 10.0
    point = r.mean() / r.std(ddof=1) * ann
    lo, hi, p = bootstrap_sharpe_ci(r, ann)
    assert lo < point < hi
    noise = np.random.default_rng(5).normal(0.0, 0.01, 400)
    lo0, hi0, p0 = bootstrap_sharpe_ci(noise, ann)
    assert lo0 < 0 < hi0 and 0.05 < p0 < 0.95


def test_missingness_staleness_resets_on_new_value():
    from features.engineering.microstructure import missingness_flags

    df = pl.DataFrame({"x": [0.0, 0.0, 0.3, 0.3, 0.3, -0.1]})
    out = missingness_flags(df, ["x"], 0.9)
    assert out["x_missing"].to_list() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    st = out["x_staleness"].to_list()
    assert st[2] == 0.0 and st[4] > st[3] > 0.0 and st[5] == 0.0


def test_volatility_clock_infers_5min_bars():
    from features.engineering.microstructure import add_volatility_clock_features

    ts = pd.date_range("2024-01-01", periods=288 * 10, freq="5min", tz="UTC")
    close = 1.1 * np.exp(np.cumsum(np.random.default_rng(6).normal(0, 1e-4, len(ts))))
    out = add_volatility_clock_features(pl.DataFrame({"timestamp_utc": ts, "close": close}))
    assert out["vol_clock_pace"].tail(288).std() > 0  # was constant 0 on 5m bars


def test_store_pair_extras_keeps_rows_aligned(tmp_path):
    zarr = pytest.importorskip("zarr")
    from common.cache_io import _zarr_create, _zarr_open_group
    from training.dataset_builder import _store_pair_extras

    zs = _zarr_open_group(str(tmp_path / "c.zarr"), mode="w")
    _zarr_create(zs, "X", shape=(0, 2, 3), chunks=(4, 2, 3))
    zs["X"].append(np.zeros((5, 2, 3), dtype=np.float32))
    _store_pair_extras(zs, {"y_pairs": np.ones((5, 4)), "t_ns": np.arange(5)}, 5)
    zs["X"].append(np.zeros((3, 2, 3), dtype=np.float32))
    _store_pair_extras(zs, {"t_ns": np.arange(3)}, 3)  # window missing y_pairs
    assert zs["y_pairs"].shape == (8, 4) and zs["t_ns"].shape == (8,)
    assert np.isnan(zs["y_pairs"][5:]).all()


def _tiny_pair_cache(path, n=64, p=4):
    from common.cache_io import _zarr_create, _zarr_open_group
    from training.dataset_builder import _store_pair_extras

    zs = _zarr_open_group(str(path), mode="w")
    for name, shape in (("X", (0, 5, 3)), ("y", (0,)), ("y_cls", (0,))):
        _zarr_create(zs, name, shape=shape, chunks=(16, *shape[1:]))
    rng = np.random.default_rng(7)
    zs["X"].append(rng.normal(size=(n, 5, 3)).astype(np.float32))
    zs["y"].append(np.zeros(n, dtype=np.float32))
    zs["y_cls"].append(np.zeros(n, dtype=np.float32))
    yp = rng.normal(size=(n, p)).astype(np.float32)
    yp[3, 1] = np.nan  # a missing pair value must survive (not become 0)
    t_ns = (np.arange(n) * 180 * 86400 * 10**9).astype(np.int64)  # ~2 rows per year
    _store_pair_extras(
        zs, {"y_pairs": yp, "ycls_pairs": np.sign(yp), "close_pairs": np.ones((n, p)), "t_ns": t_ns}, n
    )
    return yp


def test_zarr_dataset_pair_targets_yields_vectors(tmp_path):
    pytest.importorskip("zarr")
    from training.gpu_datasets import ZarrStreamDataset

    cache = tmp_path / "c.zarr"
    yp = _tiny_pair_cache(cache)
    ds = ZarrStreamDataset(str(cache), np.arange(64), shuffle_chunks=False, multitask_targets=True,
                           return_indices=True, pair_targets=True)
    rows = {int(s[-1]): s for s in ds}
    assert rows[0][1].shape == (4,) and rows[0][2].shape == (4,)
    np.testing.assert_allclose(rows[5][1].numpy(), yp[5], rtol=1e-6)
    assert np.isnan(rows[3][1].numpy()[1])


def test_period_balance_weights_equalise_years(tmp_path):
    pytest.importorskip("zarr")
    from training.honest_eval import period_balance_weights

    cache = tmp_path / "c.zarr"
    _tiny_pair_cache(cache)
    w = period_balance_weights(str(cache), np.arange(64), 64)
    t_years = (np.arange(64) * 180 // 365.2425).astype(int)
    per_year = [w[t_years == y].sum() for y in np.unique(t_years)]
    assert np.allclose(per_year, per_year[0], rtol=0.05)
    assert abs(w.mean() - 1.0) < 1e-6


def test_pooled_pair_metrics_scores_each_pair_on_its_prices():
    from training.honest_eval import pooled_pair_metrics

    n, h = 4000, 5
    rng = np.random.default_rng(8)
    close = np.cumprod(1 + rng.normal(0, 1e-3, (n + h, 2)), axis=0)
    fwd = np.sign(close[h:, :] - close[:-h, :])[:n]  # perfect foresight on both pairs
    m = pooled_pair_metrics(fwd, np.arange(n), close, None, h, pair_names=["A", "B"])
    assert set(m["per_pair"]) == {"A", "B"}
    assert m["per_pair"]["A"]["sharpe_net"] > 0 and m["per_pair"]["B"]["sharpe_net"] > 0
    assert m["sharpe_net_ci_low"] > 0
