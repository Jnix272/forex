"""
tests/test_regime_detection.py
==============================
Tests for features/regime_detection.py — true HMM regime detection,
Hurst exponents (R/S and DFA), and Higuchi fractal dimension.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from features.regime_detection import (
    RegimeHMM,
    detect_regimes_polars,
    fit_regime_hmm,
    fractal_dimension,
    hurst_dfa,
    hurst_rs,
    vol_regime_probs_polars,
    vol_regime_quantile_probs,
)

# ─────────────────────────────────────────────────────────────────────────────
# Hurst exponents
# ─────────────────────────────────────────────────────────────────────────────

def test_hurst_rs_random_walk_approx_half():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 8000)
    h = hurst_rs(x)
    assert 0.35 < h < 0.70  # near 0.5 for white noise


def test_hurst_dfa_random_walk_approx_half():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 8000)
    h = hurst_dfa(x)
    assert 0.35 < h < 0.65


def test_hurst_dfa_trending_above_half():
    rng = np.random.default_rng(3)
    trend = np.cumsum(rng.normal(0.02, 1.0, 5000))
    assert hurst_dfa(trend) > 0.6


def test_hurst_short_series_returns_neutral():
    assert hurst_rs(np.ones(10)) == 0.5
    assert hurst_dfa(np.ones(10)) == 0.5


def test_hurst_handles_nan_and_inf():
    x = np.array([1.0, np.nan, np.inf, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] * 100)
    h = hurst_rs(x)
    assert np.isfinite(h)
    assert 0.0 <= h <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Fractal dimension (Higuchi)
# ─────────────────────────────────────────────────────────────────────────────

def test_fractal_dimension_white_noise_high():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, 5000)
    fd = fractal_dimension(x)
    assert 1.7 <= fd <= 2.0


def test_fractal_dimension_brownian_around_1_5():
    rng = np.random.default_rng(5)
    bm = np.cumsum(rng.normal(0, 1, 5000))
    fd = fractal_dimension(bm)
    assert 1.2 <= fd <= 1.8


def test_fractal_dimension_smooth_low():
    t = np.linspace(0, 200 * np.pi, 5000)
    fd = fractal_dimension(np.sin(t))
    assert fd <= 1.5


def test_fractal_dimension_short_series():
    assert fractal_dimension(np.ones(8)) == 1.5


# ─────────────────────────────────────────────────────────────────────────────
# HMM regime detection
# ─────────────────────────────────────────────────────────────────────────────

def test_regime_hmm_fit_and_states():
    rng = np.random.default_rng(6)
    features = np.column_stack([
        np.concatenate([rng.normal(0.0, 0.1, 500), rng.normal(0.0, 2.0, 500)]),
        np.concatenate([rng.normal(0.0, 1.0, 500), rng.normal(0.0, 1.0, 500)]),
    ])
    model = fit_regime_hmm(features, n_states=2)
    assert len(model.states) == 1000
    assert model.state_probs.shape == (1000, 2)
    assert model.transition_.shape == (2, 2)
    assert np.isclose(model.state_probs.sum(axis=1), 1.0).all()


def test_regime_hmm_requires_hmmlearn():
    try:
        import hmmlearn  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            RegimeHMM(n_states=2)
    else:
        model = RegimeHMM(n_states=2, random_state=0)
        model.fit(np.random.default_rng(0).normal(0, 1, (100, 2)))


def test_regime_hmm_predict_before_fit_raises():
    model = RegimeHMM(n_states=2)
    with pytest.raises(RuntimeError):
        _ = model.states


# ─────────────────────────────────────────────────────────────────────────────
# Polars builders
# ─────────────────────────────────────────────────────────────────────────────

def _make_bars(n: int = 3000) -> pl.DataFrame:
    rng = np.random.default_rng(7)
    ret = np.concatenate([
        rng.normal(0.0003, 0.0005, n // 3),
        rng.normal(0.0, 0.002, n // 3),
        rng.normal(-0.0003, 0.0005, n - 2 * (n // 3)),
    ])
    close = 100 * np.exp(np.cumsum(ret))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    ts = pl.datetime_range(
        start=start, end=start + timedelta(hours=3 * n), interval="90m",
        time_zone="UTC", eager=True,
    ).head(n)
    return pl.DataFrame({
        "timestamp_utc": ts,
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": rng.integers(10, 100, n),
    })


def test_detect_regimes_polars_output_columns():
    bars = _make_bars()
    out = detect_regimes_polars(bars, n_states=3, window=60,
                                hurst_window=120, fractal_window=60)
    assert len(out) == len(bars)
    for s in range(3):
        assert f"vol_regime_state_{s}_prob" in out.columns
    assert {"hurst_rs", "hurst_dfa", "fractal_dim", "regime_label",
            "regime_class"} <= set(out.columns)
    assert out["regime_class"].max() <= 2
    assert set(out["regime_label"].unique().to_list()) <= {-1.0, 0.0, 1.0}


def test_detect_regimes_polars_neutral_baseline_at_start():
    bars = _make_bars()
    out = detect_regimes_polars(bars, n_states=3, window=60,
                                hurst_window=120, fractal_window=60)
    # First row: no lookback yet -> Hurst baseline 0.5, fractal baseline 1.5.
    assert out["hurst_dfa"][0] == 0.5
    assert out["fractal_dim"][0] == 1.5


def test_detect_regimes_polars_step_keeps_length():
    bars = _make_bars()
    out = detect_regimes_polars(bars, n_states=3, window=60,
                                hurst_window=120, fractal_window=60, step=5)
    assert len(out) == len(bars)
    assert out["hurst_dfa"].is_null().sum() == 0
    assert set(out.columns) == {
        "vol_regime_state_0_prob", "vol_regime_state_1_prob",
        "vol_regime_state_2_prob", "hurst_rs", "hurst_dfa",
        "fractal_dim", "regime_label", "regime_class",
    }


def test_vol_regime_probs_polars_contract():
    bars = _make_bars()
    out = vol_regime_probs_polars(bars, close_col="close", n_states=3, window=60)
    assert len(out) == len(bars)
    assert [f"vol_regime_state_{s}_prob" for s in range(3)] == out.columns


def test_vol_regime_quantile_probs_expressions():
    bars = _make_bars()
    exprs = vol_regime_quantile_probs(n_states=3, window=60)
    assert len(exprs) == 3
    out = bars.select(exprs)
    assert [f"vol_regime_state_{s}_prob" for s in range(3)] == out.columns
    assert len(out) == len(bars)
