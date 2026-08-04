"""
Tests for triple barrier labeling and RL reward computation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trending_up_bars() -> pd.DataFrame:
    """Price trending strongly upward — long TP should hit."""
    n = 100
    close = 1.1000 + np.arange(n) * 0.0005
    idx = pd.date_range("2024-01-02 10:00", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "close": close,
        "open": close - 0.0001,
        "high": close + 0.0002,
        "low": close - 0.0002,
        "volume": np.ones(n) * 100,
    }, index=idx)


@pytest.fixture
def trending_down_bars() -> pd.DataFrame:
    """Price trending strongly downward — short TP should hit."""
    n = 100
    close = 1.1000 - np.arange(n) * 0.0005
    idx = pd.date_range("2024-01-02 10:00", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "close": close,
        "open": close + 0.0001,
        "high": close + 0.0002,
        "low": close - 0.0002,
        "volume": np.ones(n) * 100,
    }, index=idx)


@pytest.fixture
def flat_bars() -> pd.DataFrame:
    """Price flat — neither TP nor SL should hit within short lookahead."""
    n = 100
    rng = np.random.default_rng(99)
    close = 1.1000 + rng.normal(0, 0.00001, n)
    idx = pd.date_range("2024-01-02 10:00", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "close": close,
        "open": close,
        "high": close + 0.00002,
        "low": close - 0.00002,
        "volume": np.ones(n) * 100,
    }, index=idx)


@pytest.fixture
def features_for_bars(trending_up_bars) -> pd.DataFrame:
    n = len(trending_up_bars)
    return pd.DataFrame({
        "atr_6": np.full(n, 0.0005),
        "spread_pips": np.full(n, 1.0),
    }, index=trending_up_bars.index)


# ---------------------------------------------------------------------------
# 1. Sequential scan correctness
# ---------------------------------------------------------------------------

class TestScanOutcomesSequential:
    def test_trending_up_produces_long_wins(self):
        from labeling.triple_barrier_labeling import _scan_outcomes_sequential
        n = 80
        close = 1.1000 + np.arange(n) * 0.0005
        atr = np.full(n, 0.0005)
        lo, tl, so, ts = _scan_outcomes_sequential(
            close, close, close, atr,
            profit_mult=1.5, stop_mult=1.0,
            vertical_bars=15, execution_delay_bars=0,
        )
        long_wins = (lo == 1).sum()
        assert long_wins > 0, "Strong uptrend should produce long wins"
        assert long_wins > (lo == -1).sum(), "More long wins than losses expected"

    def test_trending_down_produces_short_wins(self):
        from labeling.triple_barrier_labeling import _scan_outcomes_sequential
        n = 80
        close = 1.1000 - np.arange(n) * 0.0005
        atr = np.full(n, 0.0005)
        lo, tl, so, ts = _scan_outcomes_sequential(
            close, close, close, atr,
            profit_mult=1.5, stop_mult=1.0,
            vertical_bars=15, execution_delay_bars=0,
        )
        short_wins = (so == 1).sum()
        assert short_wins > 0, "Strong downtrend should produce short wins"

    def test_empty_on_insufficient_bars(self):
        from labeling.triple_barrier_labeling import _scan_outcomes_sequential
        close = np.array([1.1, 1.2, 1.3])
        atr = np.array([0.0005, 0.0005, 0.0005])
        lo, tl, so, ts = _scan_outcomes_sequential(
            close, close, close, atr,
            profit_mult=1.5, stop_mult=1.0,
            vertical_bars=10, execution_delay_bars=0,
        )
        assert len(lo) == 0

    def test_output_shapes_match(self):
        from labeling.triple_barrier_labeling import _scan_outcomes_sequential
        n = 50
        close = 1.1 + np.cumsum(np.random.default_rng(1).normal(0, 0.0002, n))
        atr = np.full(n, 0.0005)
        lo, tl, so, ts = _scan_outcomes_sequential(
            close, close, close, atr,
            profit_mult=1.5, stop_mult=1.0,
            vertical_bars=10, execution_delay_bars=1,
        )
        assert lo.shape == tl.shape == so.shape == ts.shape

    def test_outcomes_are_bounded(self):
        from labeling.triple_barrier_labeling import _scan_outcomes_sequential
        n = 60
        close = 1.1 + np.cumsum(np.random.default_rng(7).normal(0, 0.0003, n))
        atr = np.full(n, 0.0005)
        lo, tl, so, ts = _scan_outcomes_sequential(
            close, close, close, atr,
            profit_mult=1.5, stop_mult=1.0,
            vertical_bars=10, execution_delay_bars=0,
        )
        assert set(np.unique(lo)).issubset({-1, 0, 1})
        assert set(np.unique(so)).issubset({-1, 0, 1})
        assert tl.max() <= 10
        assert ts.max() <= 10

    def test_execution_delay_reduces_output_length(self):
        from labeling.triple_barrier_labeling import _scan_outcomes_sequential
        n = 50
        close = np.ones(n) * 1.1
        atr = np.full(n, 0.0005)
        lo0, *_ = _scan_outcomes_sequential(
            close, close, close, atr, 1.5, 1.0, 10, 0,
        )
        lo3, *_ = _scan_outcomes_sequential(
            close, close, close, atr, 1.5, 1.0, 10, 3,
        )
        assert len(lo3) == len(lo0) - 3


# ---------------------------------------------------------------------------
# 2. compute_triple_barrier_labels high-level
# ---------------------------------------------------------------------------

class TestComputeTripleBarrierLabels:
    def test_returns_dataframe(self, trending_up_bars, features_for_bars):
        from labeling.triple_barrier_labeling import compute_triple_barrier_labels
        result = compute_triple_barrier_labels(
            trending_up_bars, features_for_bars,
            vertical_bars=10, use_numba=False,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_has_required_columns(self, trending_up_bars, features_for_bars):
        from labeling.triple_barrier_labeling import compute_triple_barrier_labels
        result = compute_triple_barrier_labels(
            trending_up_bars, features_for_bars,
            vertical_bars=10, use_numba=False,
        )
        for col in ("reward_long", "reward_short", "reward", "label"):
            assert col in result.columns, f"Missing column '{col}'"

    def test_label_values_valid(self, trending_up_bars, features_for_bars):
        from labeling.triple_barrier_labeling import compute_triple_barrier_labels
        result = compute_triple_barrier_labels(
            trending_up_bars, features_for_bars,
            vertical_bars=10, use_numba=False,
        )
        valid_labels = {-1, 0, 1}
        actual = set(result["label"].unique())
        assert actual.issubset(valid_labels), f"Unexpected labels: {actual - valid_labels}"

    def test_empty_result_on_tiny_input(self):
        from labeling.triple_barrier_labeling import compute_triple_barrier_labels
        tiny_bars = pd.DataFrame({
            "close": [1.1, 1.2],
            "open": [1.1, 1.2],
            "high": [1.1, 1.2],
            "low": [1.1, 1.2],
        })
        tiny_feats = pd.DataFrame({
            "atr_6": [0.0005, 0.0005],
            "spread_pips": [1.0, 1.0],
        })
        result = compute_triple_barrier_labels(
            tiny_bars, tiny_feats, vertical_bars=10, use_numba=False,
        )
        assert len(result) == 0

    def test_uptrend_produces_long_labels(self, trending_up_bars, features_for_bars):
        from labeling.triple_barrier_labeling import compute_triple_barrier_labels
        result = compute_triple_barrier_labels(
            trending_up_bars, features_for_bars,
            vertical_bars=10, profit_atr_mult=1.5, stop_atr_mult=1.0,
            use_numba=False,
        )
        long_count = (result["label"] == 1).sum()
        short_count = (result["label"] == -1).sum()
        assert long_count > short_count, (
            f"Uptrend: expected more long({long_count}) than short({short_count}) labels"
        )


# ---------------------------------------------------------------------------
# 3. Directional label combination
# ---------------------------------------------------------------------------

class TestCombineDirectionalLabels:
    def test_combined_labels_use_best_outcome(self):
        from labeling.triple_barrier_labeling import _combine_directional_labels
        lo = np.array([1, -1, 0, 1, -1], dtype=np.int8)
        tl = np.array([3,  5, 10, 2, 7], dtype=np.int32)
        so = np.array([-1, 1, 0, 0, 0], dtype=np.int8)
        ts = np.array([5,  3, 10, 10, 10], dtype=np.int32)
        label = _combine_directional_labels(lo, tl, so, ts)
        assert len(label) == 5
        assert label[0] == 1    # long won (lo=1)
        assert label[1] == -1   # short won (so=1)
        assert label[2] == 0    # hold (both 0)
        assert label[3] == 1    # long won (lo=1, so=0)

    def test_both_tp_resolves_by_time(self):
        from labeling.triple_barrier_labeling import _combine_directional_labels
        lo = np.array([1, 1], dtype=np.int8)
        tl = np.array([2, 5], dtype=np.int32)
        so = np.array([1, 1], dtype=np.int8)
        ts = np.array([5, 2], dtype=np.int32)
        label = _combine_directional_labels(lo, tl, so, ts)
        assert label[0] == 1    # long hit first (tl=2 < ts=5)
        assert label[1] == -1   # short hit first (ts=2 < tl=5)
