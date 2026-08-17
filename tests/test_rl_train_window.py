"""Tests for holdout-safe RL/pretrain index helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.train_gpu import (
    _promotion_holdout_n,
    _rl_reward_weights,
    _rl_train_val_slices,
    _trainable_max_index,
)


def _args(**kw):
    base = {
        "promote_forward_frac": 0.1,
        "seq_len": 60,
        "lookahead_bars": 15,
        "execution_delay_bars": 1,
        "rl_val_frac": 0.15,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def test_trainable_max_index_excludes_holdout():
    n = 10_000
    a = _args()
    holdout = _promotion_holdout_n(n, a)
    max_idx = _trainable_max_index(n, a)
    assert max_idx <= n - holdout
    assert max_idx > 0


def test_rl_train_val_slices_no_overlap():
    train_start, train_n, val_start, val_n = _rl_train_val_slices(50_000, _args())
    assert train_start == 0
    assert train_n > 0
    assert val_n > 0
    assert val_start == train_n
    assert train_start + train_n + val_n <= _trainable_max_index(50_000, _args())


def test_rl_reward_weights_yaml_keys():
    a = _args(rl_reward_weights={"pnl": 1.1, "drawdown": 0.4, "tx_cost": 0.2, "overtrade": 0.3})
    w = _rl_reward_weights(a)
    assert w["pnl"] == pytest.approx(1.1)
    assert w["overtrade"] == pytest.approx(0.3)
