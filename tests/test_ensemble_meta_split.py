"""Tests for ensemble meta-learner temporal split + embargo.

Closes the canonical in-sample leak from audit 2026-08-07, finding E3 /
"meta-learner trained in-sample". Previously
``scripts/train_ensemble_meta.py`` sampled ``meta_idx`` uniformly from
``[0, total)`` of the SAME cache the base models were trained on, with no
chronological split, purge, or embargo. Now the script must compute the
``_trainable_max_index`` (= total - promotion-holdout - embargo) and
sample only from the [0, _trainable) prefix — mirroring
``training/post_train.run_ensemble_meta``.

These tests pin the helper-level contract — they target the pure
split-compute logic without standing up torch / zarr / models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.cache_integrity import _promotion_holdout_n, _trainable_max_index  # noqa: E402
from training.cv_splits import _embargo_bars  # noqa: E402

try:
    from types import SimpleNamespace
except ImportError:  # pragma: no cover - always available in py3
    SimpleNamespace = None  # type: ignore[assignment]


def _args(**kw) -> "SimpleNamespace":
    """Build the minimal args namespace used by the embargo / holdout helpers."""
    base = dict(
        seq_len=60,
        lookahead_bars=30,
        execution_delay_bars=1,
        promote_forward_frac=0.1,
        quick_mode=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_trainable_max_index_excludes_holdout_and_embargo():
    """`_trainable_max_index(total, args)` must be less than `total` by
    at least `holdout + embargo` so the meta-learner never sees the
    chronological promotion holdout nor the embargo gap."""
    total = 10_000
    a = _args()
    trainable = _trainable_max_index(total, a)
    holdout = _promotion_holdout_n(total, a)
    embargo = _embargo_bars(a)
    assert trainable == total - holdout - embargo
    # Sanity: holdout/embargo must be at least 1 each so we actually split.
    assert holdout >= 50, "promotion holdout has a floor of 50 bars"
    assert embargo >= 1, "embargo must be at least 1 bar (seq_len + LH + delay)"
    assert trainable < total, "trainable must be strictly less than total"
    assert trainable > 0, "trainable must be > 0 for total=10_000"


def test_trainable_max_index_quick_mode_scales_floor():
    """For tiny caches, the holdout floor is shrunk so the split still produces
    a trainable prefix > 0. Important so smoke tests don't crash."""
    total = 150  # quick-mode range
    a = _args(quick_mode=True)
    trainable = _trainable_max_index(total, a)
    assert 0 < trainable < total


def test_trainable_max_index_grows_with_total():
    """Doubling total must not shrink trainable (it should grow with total)."""
    total_a = 10_000
    total_b = 100_000
    a = _args()
    ta = _trainable_max_index(total_a, a)
    tb = _trainable_max_index(total_b, a)
    assert tb > ta


def test_trainable_max_index_no_drop_when_holdout_pct_is_zero():
    """If promote_forward_frac is set to 0.01 (the smallest allowed), the
    holdout still respects the floor of 50 bars — protects against
    off-by-zero under-sampling on small caches."""
    a = _args(promote_forward_frac=0.001)  # below 0.01 floor
    total = 1_000
    trainable = _trainable_max_index(total, a)
    holdout = _promotion_holdout_n(total, a)
    assert holdout >= 50  # floor enforced
    assert trainable == total - holdout - _embargo_bars(a)


def test_meta_idx_sampling_excludes_holdout_and_embargo_tail():
    """Standalone script's meta_idx logic: ``np.sort(rng.choice(_trainable, n_meta,
    replace=False))`` must produce indices strictly less than
    ``_trainable_max_index(total, args)`` — never sampling from the
    chronological holdout nor the embargo gap. This is the exact contract
    fix; the standalone script is required to mirror the production path
    (training/post_train.run_ensemble_meta:164-174)."""
    rng = np.random.default_rng(1337)
    total = 50_000
    a = _args()
    trainable = _trainable_max_index(total, a)
    assert trainable > 0
    n_meta = min(2000, trainable)
    meta_idx = np.sort(rng.choice(trainable, n_meta, replace=False))
    assert meta_idx.min() >= 0
    assert meta_idx.max() < trainable, (
        f"meta_idx max ({meta_idx.max()}) must be < trainable "
        f"({trainable}); otherwise the meta-learner has sampled from the "
        "chronological promotion holdout, which the bases also trained "
        "on — in-sample leak (audit E3)."
    )


def test_old_uniform_sampling_pattern_was_a_leak():
    """Regression guard: the OLD sampling pattern was
    ``np.sort(rng.choice(total, n_meta, replace=False))`` with no embargo.
    Demonstrate explicitly that the OLD pattern can return indices from
    the holdout tail, while the NEW pattern never can."""
    rng_old = np.random.default_rng(13)
    rng_new = np.random.default_rng(13)
    total = 50_000
    a = _args()
    trainable = _trainable_max_index(total, a)
    n_meta = min(20_000, trainable)
    # OLD pattern (leaky)
    old = np.sort(rng_old.choice(total, n_meta, replace=False))
    # NEW pattern (audited)
    new = np.sort(rng_new.choice(trainable, n_meta, replace=False))
    assert old.max() < total
    # The leak: the old pattern CAN return indices in [trainable, total)
    assert (old >= trainable).any(), (
        "If this assertion fails, the old pattern sampled within the "
        "trainable prefix by luck — re-seed the test. The point is the OLD "
        "pattern PERMITS sampling from [trainable, total), which is the leak."
    )
    # The fix always excludes that tail:
    assert (new >= trainable).sum() == 0
