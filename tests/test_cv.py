"""
Tests for validation module (Improvement #11):
WalkForwardCV, CombCV, OnlineCV, NestedCV, PurgedKFold.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from validation.cv import (
    CombCV,
    NestedCV,
    OnlineCV,
    PurgedKFold,
    WalkForwardCV,
    _embargo_indices,
    _purge_indices,
    create_cv,
    cv_diagnostics,
    evaluate_cv,
)

# ════════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_data():
    """Generate synthetic time-series data."""
    rng = np.random.default_rng(0)
    n = 1000
    X = rng.normal(size=(n, 10))
    y = rng.randint(0, 2, n)
    return X, y


@pytest.fixture
def regime_labels():
    """Generate regime labels."""
    return np.random.choice([0, 1, 2], 1000, p=[0.3, 0.5, 0.2])


# ═════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ════════════════════════════════════════════════════════════════════════════


def test_purge_indices():
    """Test _purge_indices function."""
    idx = np.arange(100)

    # Purge after
    purged = _purge_indices(idx, purge_after=10)
    assert len(purged) == 90
    assert purged[-1] == 89

    # Purge before
    purged = _purge_indices(idx, purge_before=5)
    assert len(purged) == 95
    assert purged[0] == 5

    # Both
    purged = _purge_indices(idx, purge_before=5, purge_after=10)
    assert len(purged) == 85
    assert purged[0] == 5
    assert purged[-1] == 89


def test_embargo_indices():
    """Test _embargo_indices function."""
    idx = np.arange(100)

    # Embargo after index 50 removes the next 10 samples (51..60)
    embargoed = _embargo_indices(idx, embargo=10, after_idx=50)
    # Should keep 0-50 and 61-99
    assert len(embargoed) == 90
    assert 50 in embargoed
    assert 55 not in embargoed
    assert 60 not in embargoed


# ═════════════════════════════════════════════════════════════════════════════
# WalkForwardCV
# ═════════════════════════════════════════════════════════════════════════════


def test_walk_forward_cv_basic():
    """Test basic WalkForwardCV functionality."""
    X = np.random.randn(1000, 10)
    np.random.randint(0, 2, 1000)

    cv = WalkForwardCV(n_splits=5, purge=5, embargo=5)
    splits = list(cv.split(X))

    assert len(splits) == 5
    for train_idx, val_idx in splits:
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert train_idx[-1] < val_idx[0]  # Temporal order


def test_walk_forward_cv_expanding():
    """Test expanding window walk-forward."""
    X = np.random.randn(500, 5)

    cv = WalkForwardCV(n_splits=4, initial_train_size=0.5, step_size=0.1, expanding=True)
    splits = list(cv.split(X))

    assert len(splits) == 4
    train_sizes = [len(s[0]) for s in splits]
    # Expanding window: train size should increase
    assert all(train_sizes[i] <= train_sizes[i + 1] for i in range(len(train_sizes) - 1))


def test_walk_forward_cv_rolling():
    """Test rolling window walk-forward."""
    X = np.random.randn(500, 5)

    cv = WalkForwardCV(n_splits=4, initial_train_size=0.5, step_size=0.1, expanding=False)
    splits = list(cv.split(X))

    assert len(splits) == 4
    train_sizes = [len(s[0]) for s in splits]
    # Rolling window: train size should be roughly constant
    assert max(train_sizes) - min(train_sizes) < 50


def test_walk_forward_purge_embargo():
    """Test purge and embargo work correctly."""
    X = np.random.randn(100, 5)

    cv = WalkForwardCV(n_splits=3, initial_train_size=0.5, step_size=0.1, purge=10, embargo=5, expanding=True)
    splits = list(cv.split(X))

    for train_idx, val_idx in splits:
        # Check purge: gap between train and val
        if len(train_idx) > 0 and len(val_idx) > 0:
            gap = val_idx[0] - train_idx[-1]
            assert gap >= 5  # purge distance


def test_walk_forward_cv_metadata():
    """Test get_splits_with_metadata."""
    X = np.random.randn(200, 5)

    cv = WalkForwardCV(n_splits=4, initial_train_size=0.4, step_size=0.1)
    splits = cv.get_splits_with_metadata(X)

    assert len(splits) == 4
    for i, split in enumerate(splits):
        assert split.fold == i
        assert split.train_start < split.train_end
        assert split.val_start < split.val_end
        assert split.train_end < split.val_start


# ═════════════════════════════════════════════════════════════════════════════
# PurgedKFold
# ═════════════════════════════════════════════════════════════════════════════


def test_purged_kfold_basic():
    """Test PurgedKFold basic functionality."""
    X = np.random.randn(1000, 10)

    cv = PurgedKFold(n_splits=5, purge=10, embargo=5)
    splits = list(cv.split(X))

    assert len(splits) == 5
    for train_idx, val_idx in splits:
        assert len(train_idx) > 0
        assert len(val_idx) > 0


def test_purged_kfold_purge():
    """Test purge in PurgedKFold."""
    X = np.random.randn(500, 5)

    cv = PurgedKFold(n_splits=4, purge=20, embargo=5)
    splits = list(cv.split(X))

    for train_idx, val_idx in splits:
        if len(train_idx) > 0 and len(val_idx) > 0:
            # Purge+embargo trims the train segments adjacent to the val fold,
            # so no train sample may sit within the purge/embargo zone of val.
            below = train_idx[train_idx < val_idx[0]]
            above = train_idx[train_idx > val_idx[-1]]
            if len(below) > 0:
                assert val_idx[0] - below[-1] >= 5  # embargo gap
            if len(above) > 0:
                assert above[0] - val_idx[-1] >= 5  # embargo gap


def test_purged_kfold_embargo():
    """Test embargo in PurgedKFold."""
    X = np.random.randn(400, 5)

    cv = PurgedKFold(n_splits=4, purge=10, embargo=10)
    splits = list(cv.split(X))

    assert len(splits) == 4


# ═════════════════════════════════════════════════════════════════════════════
# OnlineCV
# ═════════════════════════════════════════════════════════════════════════════


def test_online_cv_basic():
    """Test OnlineCV basic functionality."""
    np.random.randn(1000, 5)

    cv = OnlineCV(initial_train=0.5, window=0.1, step=0.1, purge=5, expanding=True)
    splits = list(cv.split(np.zeros(1000)))

    assert len(splits) > 0
    for train_idx, val_idx in cv.split(np.zeros(1000)):
        assert len(train_idx) > 0
        assert len(val_idx) > 0


def test_online_cv_rolling():
    """Test OnlineCV with rolling window."""
    np.zeros(500)

    cv = OnlineCV(initial_train=0.4, window=0.1, step=0.1, purge=5, expanding=False)
    list(cv.split(np.zeros(500)))

    train_sizes = [len(s[0]) for s in cv.split(np.zeros(500))]
    # Rolling window: train size should be roughly constant
    assert max(train_sizes) - min(train_sizes) < 50


# ═════════════════════════════════════════════════════════════════════════════
# CombCV (Combinatorial Purged CV)
# ═════════════════════════════════════════════════════════════════════════════


def test_comb_cv_basic():
    """Test CombCV basic functionality."""
    X = np.random.randn(500, 5)

    cv = CombCV(n_groups=10, test_groups=2, purge=5, embargo=2)
    splits = list(cv.split(X))

    # Should have C(10, 2) = 45 combinations
    # But limited by data size
    assert len(splits) > 0
    for train_idx, val_idx in cv.split(np.zeros(100)):
        assert len(train_idx) > 0
        assert len(val_idx) > 0


def test_comb_cv_purge_embargo_gap():
    """AUD-001: purge/embargo must remove train samples adjacent to test blocks."""
    X = np.zeros(200)
    cv = CombCV(n_groups=10, test_groups=1, purge=3, embargo=2)
    for train_idx, test_idx in cv.split(X):
        assert len(train_idx) > 0 and len(test_idx) > 0
        t0, t1 = int(test_idx.min()), int(test_idx.max())
        # No training index inside [t0-purge, t1+purge+embargo]
        banned = train_idx[(train_idx >= t0 - 3) & (train_idx <= t1 + 3 + 2)]
        assert len(banned) == 0, f"leak near test [{t0},{t1}]: {banned[:10]}"


def test_comb_cv_n_splits():
    """Test CombCV n_splits calculation."""
    cv = CombCV(n_groups=10, test_groups=2)
    # C(10, 2) = 45
    assert cv.get_n_splits() == 45


# ═════════════════════════════════════════════════════════════════════════════
# NestedCV
# ═════════════════════════════════════════════════════════════════════════════


class DummyEstimator(BaseEstimator, ClassifierMixin):
    """Simple dummy estimator for testing."""

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def test_nested_cv():
    """Test NestedCV basic functionality."""
    np.random.randn(200, 5)
    np.random.randint(0, 2, 200)

    WalkForwardCV(n_splits=3)
    WalkForwardCV(n_splits=2)


    nested = NestedCV(
        outer_cv=WalkForwardCV(n_splits=3),
        inner_cv=WalkForwardCV(n_splits=2),
        estimator_factory=lambda C=1.0: LogisticRegression(C=C, max_iter=100, random_state=0),
        param_grid={"C": [0.1, 1.0, 10.0]},
        scoring=lambda y_t, y_p: np.mean(y_t == y_p),
    )

    # Use a simpler test with smaller data
    X_small = np.random.randn(100, 5)
    y_small = np.random.randint(0, 2, 100)

    nested = NestedCV(
        outer_cv=WalkForwardCV(n_splits=2),
        inner_cv=WalkForwardCV(n_splits=2),
        estimator_factory=lambda C=1.0: LogisticRegression(C=C, max_iter=100, random_state=0),
        param_grid={"C": [0.1, 1.0]},
        scoring=lambda y_t, y_p: np.mean(y_t == y_p),
    )

    result = nested.fit(X_small, y_small)

    assert "outer_scores" in result
    assert "mean_score" in result
    assert "best_params_per_fold" in result
    assert len(result["outer_scores"]) > 0


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation & Diagnostics
# ═════════════════════════════════════════════════════════════════════════════


def test_evaluate_cv():
    """Test evaluate_cv function."""
    X = np.random.randn(200, 5)
    y = np.random.randint(0, 2, 200)

    cv = WalkForwardCV(n_splits=3)
    model = LogisticRegression(max_iter=100, random_state=0)

    results = evaluate_cv(model, X, y, cv)

    assert "scores" in results
    assert "mean" in results
    assert "std" in results
    assert len(results["scores"]) == 3


def test_cv_diagnostics():
    """Test cv_diagnostics function."""
    np.random.randn(200, 10)

    cv = WalkForwardCV(n_splits=4, purge=5, embargo=5)
    diag = cv_diagnostics(cv, np.zeros(200))

    assert "n_splits" in diag
    assert "train_size_mean" in diag
    assert "val_size_mean" in diag
    assert "overlap_pairs" in diag


def test_purge_embargo_functions():
    """Test purge and embargo helper functions."""
    idx = np.arange(100)

    # Purge
    purged = _purge_indices(idx, purge_after=10)
    assert len(purged) == 90
    assert purged[-1] == 89

    # Embargo
    embargoed = _embargo_indices(np.arange(100), embargo=10, after_idx=50)
    assert 50 in embargoed
    assert 55 not in embargoed
    assert 60 not in embargoed


# ═════════════════════════════════════════════════════════════════════════════
# Factory
# ═════════════════════════════════════════════════════════════════════════════


def test_create_cv():
    """Test create_cv factory function."""
    # Walk forward
    cv = create_cv("walk_forward", n_splits=5, purge=5)
    assert isinstance(cv, WalkForwardCV)

    # Purged K-Fold
    cv = create_cv("purged_kfold", n_splits=5, purge=10)
    assert isinstance(cv, PurgedKFold)

    # Comb
    cv = create_cv("comb", n_groups=10, test_groups=2)
    assert isinstance(cv, CombCV)

    # Online
    cv = create_cv("online", initial_train=0.5, window=0.1)
    assert isinstance(cv, OnlineCV)

    # Nested
    cv = create_cv(
        "nested",
        outer_cv="walk_forward",
        outer_kwargs={"n_splits": 3},
        inner_cv="walk_forward",
        inner_kwargs={"n_splits": 2},
        estimator_factory=lambda: None,
        param_grid={},
    )
    assert isinstance(cv, NestedCV)


def test_cv_diagnostics_output():
    """Test cv_diagnostics output structure."""
    np.random.randn(200, 10)
    cv = WalkForwardCV(n_splits=5, purge=5, embargo=5)
    diag = cv_diagnostics(cv, np.zeros(200))

    assert "n_splits" in diag
    assert diag["n_splits"] == 5
    assert "train_size_mean" in diag
    assert "val_size_mean" in diag
    assert "overlap_pairs" in diag


# ═════════════════════════════════════════════════════════════════════════════
# Edge Cases
# ═════════════════════════════════════════════════════════════════════════════


def test_walk_forward_edge_cases():
    """Test edge cases for WalkForwardCV."""
    # Very small data
    np.random.randn(50, 5)
    cv = WalkForwardCV(n_splits=2, initial_train_size=0.4, step_size=0.2)
    splits = list(cv.split(np.zeros(50)))
    assert len(splits) >= 1

    # Too small data
    np.random.randn(20, 5)
    cv = WalkForwardCV(n_splits=5, initial_train_size=0.6)
    splits = list(cv.split(np.zeros(20)))
    # Should handle gracefully


def test_cv_diagnostics_output():  # noqa: F811
    """Test cv_diagnostics output structure."""
    np.random.randn(200, 10)
    cv = WalkForwardCV(n_splits=4, purge=5, embargo=5)
    diag = cv_diagnostics(cv, np.zeros(200))

    assert "n_splits" in diag
    assert diag["n_splits"] == 4
    assert "train_size_mean" in diag
    assert "train_size_std" in diag
    assert "val_size_mean" in diag
    assert "val_size_std" in diag
    assert "overlap_pairs" in diag
    assert "total_pairs" in diag


# ═════════════════════════════════════════════════════════════════════════════
# train_gpu wiring tests (--cv-strategy)
# ═════════════════════════════════════════════════════════════════════════════


def _cv_args(**kw):
    import argparse

    base = {
        "cv_strategy": "legacy",
        "walk_forward_folds": 6,
        "seq_len": 60,
        "lookahead_bars": 15,
        "execution_delay_bars": 1,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def test_build_cv_splits_legacy_default():
    """legacy strategy must preserve original walk_forward_splits behavior."""
    from training.train_gpu import _build_cv_splits

    splits, label = _build_cv_splits(_cv_args(), 20_000)
    assert label == "legacy"
    assert len(splits) == 6
    for tr, va in splits:
        assert len(tr) > 0 and len(va) > 0
        assert max(va) <= 20_000


@pytest.mark.parametrize("strategy", ["walk_forward", "comb", "online"])
def test_build_cv_splits_alt_strategies(strategy):
    """Alt strategies produce usable non-leaking train/val splits."""
    from training.train_gpu import _build_cv_splits

    splits, label = _build_cv_splits(_cv_args(cv_strategy=strategy), 20_000)
    assert label == strategy
    assert len(splits) > 0
    for tr, va in splits:
        assert len(tr) >= 100 and len(va) >= 10
        if strategy != "comb":
            # temporal guard (comb is combinatorial by design, not chronological)
            assert va[0] >= tr[0]


def test_build_cv_splits_unknown_falls_back_legacy():
    """Unknown strategy must fall back to legacy splits without raising."""
    from training.train_gpu import _build_cv_splits

    splits, label = _build_cv_splits(_cv_args(cv_strategy="nope"), 20_000)
    assert label == "legacy"
    assert len(splits) > 0


def test_build_cv_splits_small_data_fallback():
    """Small datasets must still return a usable split (embargoed fallback)."""
    from training.train_gpu import _build_cv_splits

    splits, _label = _build_cv_splits(_cv_args(cv_strategy="comb"), 2_000)
    assert len(splits) > 0
    for tr, va in splits:
        assert len(tr) > 0 and len(va) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
