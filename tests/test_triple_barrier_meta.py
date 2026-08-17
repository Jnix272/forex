"""
Tests for meta-labeling & Bayesian barrier search (Improvement #6)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from labeling.triple_barrier_meta import (
    BarrierSearchConfig,
    BarrierSearchSpace,
    MetaLabelConfig,
    MetaLabeler,
    evaluate_barrier_params,
)

# ---------------------------------------------------------------------------
# Meta-Labeling
# ---------------------------------------------------------------------------


def test_meta_labeler_basic():
    """Test basic meta-labeler fit and predict."""
    rng = np.random.default_rng(0)
    n = 1000

    # Primary model: predicts long/short/hold
    primary = np.zeros(n)
    primary[:400] = 1  # long
    primary[400:800] = -1  # short
    primary[800:] = 0  # hold

    # Labels: TBM outcomes
    labels = np.zeros(n)
    # For long trades (0-399): 70% hit TP (label=1), 30% hit SL (label=-1)
    labels[:280] = 1
    labels[280:400] = -1
    # For short trades (400-799): 60% hit TP (label=-1), 40% hit SL (label=1)
    labels[400:640] = -1
    labels[640:800] = 1
    # Holds are 0

    # Features for meta-model
    features = pd.DataFrame(
        {
            "feat1": rng.normal(size=n),
            "feat2": rng.normal(size=n),
        }
    )

    config = MetaLabelConfig(
        meta_features=["feat1", "feat2"],
        min_meta_samples=50,
        meta_train_frac=0.7,
        meta_prob_threshold=0.5,
        random_state=0,
    )
    meta = MetaLabeler(config)
    meta.fit(primary, labels, features)

    probs = meta.predict_proba(primary, features)
    assert len(probs) == n
    assert (probs >= 0).all() and (probs <= 1).all()

    trade_mask = meta.should_trade(primary, features)
    assert trade_mask.sum() > 0


def test_meta_labeler_no_trades():
    """Meta-labeler handles case with no trades signaled."""
    primary = np.zeros(100)
    labels = np.zeros(100)
    config = MetaLabelConfig(min_meta_samples=10)
    meta = MetaLabeler(config)
    meta.fit(primary, labels)
    probs = meta.predict_proba(primary)
    assert (probs == 0).all()


def test_meta_labeler_insufficient_samples():
    """Meta-labeler warns and returns early with insufficient trade samples."""
    primary = np.array([1, -1, 0, 0, 0])
    labels = np.array([1, -1, 0, 0, 0])
    config = MetaLabelConfig(min_meta_samples=10)
    meta = MetaLabeler(config)
    meta.fit(primary, labels)
    assert not meta._is_fitted


def test_meta_labeler_without_extra_features():
    """Meta-labeler works with only primary prediction as feature."""
    rng = np.random.default_rng(1)
    n = 500
    primary = np.where(rng.random(n) > 0.5, 1, -1)
    labels = np.where(rng.random(n) > 0.6, primary, -primary)  # noisy labels

    config = MetaLabelConfig(min_meta_samples=30)
    meta = MetaLabeler(config)
    meta.fit(primary, labels)

    probs = meta.predict_proba(primary)
    assert len(probs) == n


# ---------------------------------------------------------------------------
# Barrier Parameter Evaluation
# ---------------------------------------------------------------------------


def test_evaluate_barrier_params():
    """Quick evaluation of barrier parameters."""
    rng = np.random.default_rng(2)
    n = 500
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))

    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "close": close,
        }
    )
    bars.set_index("timestamp", inplace=True)

    features = pd.DataFrame({"atr_6": np.full(n, 0.5)}, index=bars.index)

    def dummy_pred(bars, feats):
        return np.ones(len(bars))  # always long

    metrics = evaluate_barrier_params(
        bars,
        features,
        profit_mult=1.5,
        stop_mult=1.0,
        vertical_bars=10,
        primary_pred_fn=dummy_pred,
    )

    assert "n_trades" in metrics
    assert metrics["n_trades"] > 0
    assert "win_rate" in metrics
    assert "sharpe" in metrics


# ---------------------------------------------------------------------------
# Bayesian Barrier Optimizer (requires optuna)
# ---------------------------------------------------------------------------


def test_barrier_search_space():
    space = BarrierSearchSpace(
        profit_mult=(0.5, 3.0),
        stop_mult=(0.3, 2.0),
        vertical_bars=(5, 40),
    )
    assert space.profit_mult == (0.5, 3.0)
    assert space.stop_mult == (0.3, 2.0)
    assert space.vertical_bars == (5, 40)


def test_barrier_search_config():
    config = BarrierSearchConfig(
        n_trials=10,
        objective="sharpe",
        direction="maximize",
    )
    assert config.n_trials == 10
    assert config.objective == "sharpe"
    assert config.direction == "maximize"


# ---------------------------------------------------------------------------
# Integration test (mocked primary model)
# ---------------------------------------------------------------------------


def test_evaluate_barrier_params_multiple():
    """Test multiple parameter evaluations."""
    rng = np.random.default_rng(3)
    n = 1000
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))

    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="30min"),
            "close": close,
            "high": close * 1.001,
            "low": close * 0.999,
        }
    )
    bars.set_index("timestamp", inplace=True)

    pd.DataFrame({"atr_6": np.full(n, 0.3)}, index=bars.index)

    def dummy_pred(bars, feats):
        return np.random.choice([-1, 1], len(bars))

    # Test multiple parameter sets
    params = [
        (1.0, 0.5, 5),
        (1.5, 1.0, 10),
        (2.0, 1.5, 20),
    ]
    results = []
    for pm, sm, vb in params:
        m = evaluate_barrier_params(bars, bars[["close"]].copy(), pm, sm, vb, dummy_pred)
        results.append(m)
        assert "n_trades" in m

    # At least one should produce trades
    assert any(r["n_trades"] > 0 for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
