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
    """Price trending strongly upward - long TP should hit."""
    n = 100
    close = 1.1000 + np.arange(n) * 0.0005
    idx = pd.date_range("2024-01-02 10:00", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "close": close,
            "open": close - 0.0001,
            "high": close + 0.0002,
            "low": close - 0.0002,
            "volume": np.ones(n) * 100,
        },
        index=idx,
    )


@pytest.fixture
def trending_down_bars() -> pd.DataFrame:
    """Price trending strongly downward - short TP should hit."""
    n = 100
    close = 1.1000 - np.arange(n) * 0.0005
    idx = pd.date_range("2024-01-02 10:00", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "close": close,
            "open": close + 0.0001,
            "high": close + 0.0002,
            "low": close - 0.0002,
            "volume": np.ones(n) * 100,
        },
        index=idx,
    )


@pytest.fixture
def flat_bars() -> pd.DataFrame:
    """Price flat - neither TP nor SL should hit within short lookahead."""
    n = 100
    rng = np.random.default_rng(99)
    close = 1.1000 + rng.normal(0, 0.00001, n)
    idx = pd.date_range("2024-01-02 10:00", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "close": close,
            "open": close,
            "high": close + 0.00002,
            "low": close - 0.00002,
            "volume": np.ones(n) * 100,
        },
        index=idx,
    )


@pytest.fixture
def features_for_bars(trending_up_bars) -> pd.DataFrame:
    n = len(trending_up_bars)
    return pd.DataFrame(
        {
            "atr_6": np.full(n, 0.0005),
            "spread_pips": np.full(n, 1.0),
        },
        index=trending_up_bars.index,
    )


# ---------------------------------------------------------------------------
# 1. Sequential scan correctness
# ---------------------------------------------------------------------------


class TestScanOutcomesSequential:
    def test_trending_up_produces_long_wins(self):
        from labeling.triple_barrier_labeling import _scan_outcomes_cpar_sequential
        n = 80
        close = 1.1000 + np.arange(n) * 0.0005
        atr = np.full(n, 0.0005)
        cpar_l, cpar_s, rew, lab = _scan_outcomes_cpar_sequential(
            close, close, close, close, atr, penalty=1.0, vertical_bars=15, execution_delay_bars=0
        )
        assert np.mean(cpar_l) > 0, "Strong uptrend should produce positive CPAR long"
        assert np.mean(cpar_l) > np.mean(cpar_s), "CPAR long should beat CPAR short"

    def test_trending_down_produces_short_wins(self):
        from labeling.triple_barrier_labeling import _scan_outcomes_cpar_sequential
        n = 80
        close = 1.1000 - np.arange(n) * 0.0005
        atr = np.full(n, 0.0005)
        cpar_l, cpar_s, rew, lab = _scan_outcomes_cpar_sequential(
            close, close, close, close, atr, penalty=1.0, vertical_bars=15, execution_delay_bars=0
        )
        assert np.mean(cpar_s) > 0, "Strong downtrend should produce positive CPAR short"

    def test_empty_on_insufficient_bars(self):
        from labeling.triple_barrier_labeling import _scan_outcomes_cpar_sequential
        close = np.array([1.1, 1.2, 1.3])
        atr = np.array([0.0005, 0.0005, 0.0005])
        cpar_l, _cpar_s, _rew, _lab = _scan_outcomes_cpar_sequential(
            close, close, close, close, atr, penalty=1.0, vertical_bars=10, execution_delay_bars=0
        )
        assert len(cpar_l) == 0

    def test_output_shapes_match(self):
        from labeling.triple_barrier_labeling import _scan_outcomes_cpar_sequential
        n = 50
        close = np.ones(n) * 1.1
        atr = np.full(n, 0.0005)
        cpar_l, cpar_s, rew, lab = _scan_outcomes_cpar_sequential(
            close, close, close, close, atr, penalty=1.0, vertical_bars=10, execution_delay_bars=0
        )
        assert len(cpar_l) == len(cpar_s) == len(rew) == len(lab) == (n - 10)

    def test_outcomes_are_bounded(self):
        from labeling.triple_barrier_labeling import _scan_outcomes_cpar_sequential
        n = 50
        close = np.ones(n) * 1.1
        atr = np.full(n, 0.0005)
        cpar_l, cpar_s, rew, lab = _scan_outcomes_cpar_sequential(
            close, close, close, close, atr, penalty=1.0, vertical_bars=5, execution_delay_bars=0
        )
        assert not np.isnan(cpar_l).any()
        assert not np.isnan(cpar_s).any()

    def test_execution_delay_reduces_output_length(self):
        from labeling.triple_barrier_labeling import _scan_outcomes_cpar_sequential
        n = 50
        close = np.ones(n) * 1.1
        atr = np.full(n, 0.0005)
        lo0, *_ = _scan_outcomes_cpar_sequential(
            close, close, close, close, atr, penalty=1.0, vertical_bars=10, execution_delay_bars=0
        )
        lo3, *_ = _scan_outcomes_cpar_sequential(
            close, close, close, close, atr, penalty=1.0, vertical_bars=10, execution_delay_bars=3
        )
        assert len(lo3) == len(lo0) - 3


# ---------------------------------------------------------------------------
# 2. compute_triple_barrier_labels high-level
# ---------------------------------------------------------------------------


class TestComputeTripleBarrierLabels:
    def test_returns_dataframe(self, trending_up_bars, features_for_bars):
        from labeling.triple_barrier_labeling import compute_triple_barrier_labels

        result = compute_triple_barrier_labels(
            trending_up_bars,
            features_for_bars,
            vertical_bars=10,
            use_numba=False,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_has_required_columns(self, trending_up_bars, features_for_bars):
        from labeling.triple_barrier_labeling import compute_triple_barrier_labels

        result = compute_triple_barrier_labels(
            trending_up_bars,
            features_for_bars,
            vertical_bars=10,
            use_numba=False,
        )
        for col in ("reward_long", "reward_short", "reward", "label"):
            assert col in result.columns, f"Missing column '{col}'"

    def test_label_values_valid(self, trending_up_bars, features_for_bars):
        from labeling.triple_barrier_labeling import compute_triple_barrier_labels

        result = compute_triple_barrier_labels(
            trending_up_bars,
            features_for_bars,
            vertical_bars=10,
            use_numba=False,
        )
        assert pd.api.types.is_numeric_dtype(result["label"])
        assert not result["label"].isna().any()

    def test_empty_result_on_tiny_input(self):
        from labeling.triple_barrier_labeling import compute_triple_barrier_labels

        tiny_bars = pd.DataFrame(
            {
                "close": [1.1, 1.2],
                "open": [1.1, 1.2],
                "high": [1.1, 1.2],
                "low": [1.1, 1.2],
            }
        )
        tiny_feats = pd.DataFrame(
            {
                "atr_6": [0.0005, 0.0005],
                "spread_pips": [1.0, 1.0],
            }
        )
        result = compute_triple_barrier_labels(
            tiny_bars,
            tiny_feats,
            vertical_bars=10,
            use_numba=False,
        )
        assert len(result) == 0

    def test_uptrend_produces_long_labels(self, trending_up_bars, features_for_bars):
        from labeling.triple_barrier_labeling import compute_triple_barrier_labels

        result = compute_triple_barrier_labels(
            trending_up_bars,
            features_for_bars,
            vertical_bars=5,
            use_numba=False,
        )
        long_labels = (result["label"] > 0).sum()
        assert long_labels > 0, "Uptrend should produce positive CPAR labels"


