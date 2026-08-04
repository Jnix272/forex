"""
Tests for feature quality monitor (Improvement #4):
PSI, IV/WOE, stability index, leakage detection.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from features.feature_quality_monitor import (
    feature_quality_monitor,
    information_value,
    ks_statistic,
    leakage_scan,
    population_stability_index,
    stability_index_series,
    woe_iv,
)


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------

def test_psi_identical_distributions_low():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(0, 1, 5000)
    assert population_stability_index(a, b) < 0.1


def test_psi_shifted_distribution_high():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 5000)
    c = rng.normal(1.5, 1, 5000)
    assert population_stability_index(a, c) > 0.25


def test_psi_handles_empty_and_constant():
    assert population_stability_index([], [1.0, 2.0]) == 0.0
    assert population_stability_index(np.ones(100), np.ones(100)) == 0.0
    assert population_stability_index([1.0], [1.0, 2.0, 3.0]) == 0.0


# ---------------------------------------------------------------------------
# IV / WOE
# ---------------------------------------------------------------------------

def test_iv_strong_vs_noise():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 10000)
    p = 1 / (1 + np.exp(-2 * x))
    y = rng.binomial(1, p, 10000)
    assert information_value(x, y) > 0.3
    noise = rng.normal(0, 1, 10000)
    y2 = rng.binomial(1, 0.5, 10000)
    assert information_value(noise, y2) < 0.1


def test_woe_iv_shapes():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 2000)
    y = (x > 0).astype(float)
    woe, iv, edges = woe_iv(x, y, n_bins=10)
    assert iv > 0.5
    assert len(woe) == len(edges) - 1
    assert np.all(np.isfinite(woe))


def test_iv_degenerate_target_zero():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, 1000)
    y = np.ones(1000)  # single class
    assert information_value(x, y) == 0.0


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

def test_stability_static_series_low():
    rng = np.random.default_rng(5)
    s = rng.normal(0, 1, 2000)
    stab = stability_index_series(s, window=400, step=100)
    assert np.mean(stab) < 0.1
    assert len(stab) == len(s)


def test_stability_drifting_series_high():
    rng = np.random.default_rng(6)
    s = np.concatenate([rng.normal(0, 1, 600), rng.normal(3, 1, 1400)])
    stab = stability_index_series(s, window=400, step=100)
    assert np.mean(stab) > 0.1


def test_stability_short_series_zero():
    rng = np.random.default_rng(7)
    assert (stability_index_series(rng.normal(size=100), window=400) == 0).all()


def test_ks_statistic():
    rng = np.random.default_rng(8)
    a = rng.normal(0, 1, 2000)
    b = rng.normal(0, 1, 2000)
    c = rng.normal(2.0, 1, 2000)
    assert ks_statistic(a, b) < 0.1
    assert ks_statistic(a, c) > 0.5


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def test_leakage_scan_flags_target_derived_feature():
    rng = np.random.default_rng(9)
    f = rng.normal(size=2000)
    t = (f > 0).astype(float)
    noise = rng.normal(size=2000)
    L = leakage_scan(pl.DataFrame({"f": f, "noise": noise}), t)
    lf = L.filter(pl.col("feature") == "f")
    ln = L.filter(pl.col("feature") == "noise")
    assert lf["leak_flag"].item()
    assert lf["iv"].item() > 0.5
    assert not ln["leak_flag"].item()


def test_leakage_scan_no_features():
    L = leakage_scan(pl.DataFrame(), np.zeros(10))
    assert L is not None


# ---------------------------------------------------------------------------
# Master monitor
# ---------------------------------------------------------------------------

def _panel(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = rng.normal(size=n).tolist()
    for i in range(0, n, 10):
        vals[i] = None
    return pl.DataFrame({
        "clean": rng.normal(0, 1, n),
        "drift": np.concatenate([rng.normal(0, 1, n // 3), rng.normal(3, 1, n - n // 3)]),
        "const": np.ones(n),
        "nulls": vals,
        "timestamp_utc": np.arange(n),
    })


def test_quality_monitor_flags_issues():
    rep = feature_quality_monitor(_panel(), stability_window=400, stability_step=100)
    m = {r["feature"]: r for r in rep.to_dicts()}
    assert not m["clean"]["quality_flag"] or True  # sanity
    assert m["drift"]["psi_level"] == "severe"
    assert m["drift"]["quality_flag"] is False
    assert m["const"]["constant"] is True
    assert m["const"]["quality_flag"] is False
    assert m["nulls"]["null_pct"] > 0
    assert m["clean"]["quality_flag"] is True
    assert m["clean"]["psi_level"] == "stable"


def test_quality_monitor_with_target_leakage():
    rng = np.random.default_rng(1)
    feat = rng.normal(size=2000)
    t = (feat > 0).astype(float)
    df = pl.DataFrame({"feature_a": feat, "noise": rng.normal(size=2000)})
    rep = feature_quality_monitor(df, target_col=None, stability_window=400)
    rep2 = feature_quality_monitor(df.with_columns(pl.Series("target", t)),
                                   target_col="target", stability_window=400)
    lf = rep2.filter(pl.col("feature") == "feature_a")
    assert lf["leak_flag"].item()


def test_quality_monitor_with_reference_df():
    rng = np.random.default_rng(2)
    ref = pl.DataFrame({"x": rng.normal(0, 1, 2000)})
    cur = pl.DataFrame({"x": rng.normal(3, 1, 2000)})
    rep = feature_quality_monitor(cur, reference_df=ref, stability_window=400)
    assert rep["psi"].item() > 0.25


# ---------------------------------------------------------------------------
# filter_features gate
# ---------------------------------------------------------------------------

def test_filter_features_drops_bad_keeps_good():
    from features.feature_quality_monitor import filter_features
    rng = np.random.default_rng(3)
    sig = rng.normal(0, 1, 2000)
    t = (sig > 0).astype(float)
    df = pl.DataFrame({
        "good_feat": rng.normal(0, 1, 2000),          # genuine noise
        "leaky_feat": sig + rng.normal(0, 1e-4, 2000),  # near-perfect for t
        "const_feat": np.ones(2000),
        "drift_feat": np.concatenate([rng.normal(0, 1, 700), rng.normal(3, 1, 1300)]),
        "target": t,
    })
    kept, report, dropped = filter_features(df, target_col="target",
                                            stability_window=400, stability_step=100)
    assert "target" in kept.columns
    assert "const_feat" in dropped
    assert "drift_feat" in dropped
    assert "leaky_feat" in dropped
    assert "good_feat" in kept.columns
    assert "target" not in dropped
