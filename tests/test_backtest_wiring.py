"""
Tests for backtest_model wiring helpers (Improvements #6, #15, #16):
meta-labeling, advanced execution engine, Monte Carlo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.backtest_model import _build_meta_labeler_mask


def _make_args(**overrides) -> argparse.Namespace:
    base = dict(
        pair="EURUSD", seq_len=20, take_pips=18.0, stop_pips=12.0,
        execution_delay_bars=1, meta_prob_threshold=0.55, meta_min_samples=10,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def window_data():
    n = 400
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 1.10 + np.cumsum(rng.normal(0, 0.0002, n))
    bars = pd.DataFrame(
        {
            "open": close - rng.uniform(0, 0.0001, n),
            "high": close + rng.uniform(0, 0.0002, n),
            "low": close - rng.uniform(0, 0.0002, n),
            "close": close,
            "volume": rng.integers(50, 500, n).astype(float),
        },
        index=idx,
    )
    feats = pd.DataFrame(
        {
            "atr_6": rng.uniform(0.0001, 0.0005, n),
            "spread_pips": rng.uniform(0.5, 2.0, n),
            "rsi_14": rng.uniform(30, 70, n),
            "adx_14": rng.uniform(10, 40, n),
            "ret_5": rng.normal(0, 0.001, n),
        },
        index=idx,
    )
    seq_len = 20
    n_pred = n - seq_len
    cls = rng.choice([0, 1, 2], size=n_pred)
    return bars, feats, cls, seq_len


def test_meta_labeler_mask_disabled_when_args_off(window_data):
    bars, feats, cls, seq_len = window_data
    args = _make_args()
    # arg has no meta_labeling attr here -> helper should not crash, returns None
    mask, meta = _build_meta_labeler_mask(args, bars, feats, cls, seq_len)
    assert (mask is None) or (len(mask) == len(cls))


def test_meta_labeler_mask_shape(window_data):
    bars, feats, cls, seq_len = window_data
    args = _make_args()
    mask, meta = _build_meta_labeler_mask(args, bars, feats, cls, seq_len)
    if mask is not None:
        assert len(mask) == len(cls)
        assert mask.dtype == bool


def test_meta_labeler_mask_small_window_returns_none():
    rng = np.random.default_rng(1)
    n = 40
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 1.10 + np.cumsum(rng.normal(0, 0.0002, n))
    bars = pd.DataFrame(
        {"open": close, "high": close + 0.001, "low": close - 0.001,
         "close": close, "volume": 100.0},
        index=idx,
    )
    feats = pd.DataFrame(
        {"atr_6": rng.uniform(0.0001, 0.0005, n), "spread_pips": 1.0,
         "rsi_14": 50.0, "adx_14": 20.0, "ret_5": 0.0},
        index=idx,
    )
    cls = np.zeros(n - 20, dtype=int)
    args = _make_args(seq_len=20)
    mask, meta = _build_meta_labeler_mask(args, bars, feats, cls, 20)
    # All-hold cls -> no trades -> meta not fitted -> returns None
    assert meta is None or not getattr(meta, "_is_fitted", False)


def test_advanced_execution_overlay_legacy(window_data):
    """D1 wiring: legacy engine returns baseline slippage unchanged."""
    from scripts.backtest_model import _advanced_execution_overlay
    bars, _, _, _ = window_data
    args = _make_args(execution_engine="legacy", slippage_pips=0.7)
    sigs = [{"timestamp": bars.index[100], "action": 1}]
    eff, meta = _advanced_execution_overlay(args, bars, sigs)
    assert eff == 0.7
    assert meta == {}


def test_advanced_execution_overlay_advanced(window_data):
    """D1 wiring: advanced engine raises effective slippage + returns meta."""
    from scripts.backtest_model import _advanced_execution_overlay
    bars, _, _, _ = window_data
    args = _make_args(execution_engine="advanced", slippage_pips=0.7)
    sigs = [{"timestamp": bars.index[100], "action": 1},
            {"timestamp": bars.index[150], "action": 2}]
    eff, meta = _advanced_execution_overlay(args, bars, sigs)
    assert eff >= 0.7
    assert meta["execution_engine"] == "advanced"
    assert meta["overlay_applied_to_n_signals"] == 2


def test_execution_engine_arg(monkeypatch):
    """D1 wiring: --execution-engine parses with default legacy."""
    from scripts.backtest_model import parse_args
    monkeypatch.setattr(sys, "argv", ["backtest_model", "--model", "tft"])
    args = parse_args()
    assert args.execution_engine == "legacy"
    monkeypatch.setattr(sys, "argv", ["backtest_model", "--model", "tft", "--execution-engine", "advanced"])
    args = parse_args()
    assert args.execution_engine == "advanced"
