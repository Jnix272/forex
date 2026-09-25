"""
labeling/triple_barrier_labeling.py
====================================
Replaced Triple Barrier Method (TBM) with Continuous Path-Adjusted Reward (CPAR).
The continuous label is calculated as:
(Forward_Return / Rolling_Volatility) - (Penalty * Maximum_Adverse_Excursion)

Ensures returning a continuous scalar for regression targets.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

try:
    from numba import njit, prange
    _NUMBA_IMPORT_OK = True
except ImportError:
    _NUMBA_IMPORT_OK = False

    def _identity_njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        def decorator(func):
            return func
        return decorator

    njit = _identity_njit  # type: ignore[assignment]
    prange = range  # type: ignore[assignment]


def _default_labeling() -> dict[str, Any]:
    try:
        from config.settings import LABELING as L
        return L
    except Exception:
        return {}


@njit(cache=True)
def _cost_aware_label(cpar_l: float, cpar_s: float) -> float:
    """BUY/SELL only when that side pays after costs, else HOLD.

    Each side's CPAR already enters at the ask (long) / bid (short), so it is net
    of spread. The old ``sign((cpar_l - cpar_s) / 2)`` cancelled the spread and
    labelled the mid-price move, leaving HOLD almost empty (127 of 418k EURUSD).
    """
    if cpar_l > 0.0 and cpar_l >= cpar_s:
        return 1.0
    if cpar_s > 0.0 and cpar_s > cpar_l:
        return -1.0
    return 0.0


@njit(cache=True, fastmath=True, parallel=True)
def _scan_outcomes_cpar_numba(
    exit_long_path: np.ndarray,
    exit_short_path: np.ndarray,
    entry_long: np.ndarray,
    entry_short: np.ndarray,
    atr: np.ndarray,
    penalty: float,
    vertical_bars: int,
    n_valid: int,
    execution_delay_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    cpar_long_out = np.zeros(n_valid, dtype=np.float32)
    cpar_short_out = np.zeros(n_valid, dtype=np.float32)
    label_out = np.zeros(n_valid, dtype=np.float32)
    reward_out = np.zeros(n_valid, dtype=np.float32)
    
    delay = max(0, int(execution_delay_bars))

    for i in prange(n_valid):
        ei = i + delay
        el = entry_long[ei]
        es = entry_short[ei]
        vol = atr[ei] if atr[ei] > 1e-8 else 1e-8

        # For long:
        end_idx = ei + vertical_bars
        fwd_ret_long = exit_long_path[end_idx] - el
        
        # MAE is maximum adverse excursion (lowest low during the path)
        min_p = el
        for t in range(1, vertical_bars + 1):
            if exit_long_path[ei + t] < min_p:
                min_p = exit_long_path[ei + t]
        mae_long = max(0.0, el - min_p)
        
        # CPAR Long
        cpar_l = (fwd_ret_long / vol) - (penalty * mae_long / vol)

        # For short:
        fwd_ret_short = es - exit_short_path[end_idx]
        
        max_p = es
        for t in range(1, vertical_bars + 1):
            if exit_short_path[ei + t] > max_p:
                max_p = exit_short_path[ei + t]
        mae_short = max(0.0, max_p - es)
        
        # CPAR Short
        cpar_s = (fwd_ret_short / vol) - (penalty * mae_short / vol)
        
        cpar_long_out[i] = cpar_l
        cpar_short_out[i] = cpar_s
        
        # Continuous scalar label for regression
        reward_val = (cpar_l - cpar_s) / 2.0
        reward_out[i] = reward_val
        label_out[i] = _cost_aware_label(cpar_l, cpar_s)

    return cpar_long_out, cpar_short_out, reward_out, label_out


def _scan_outcomes_cpar_sequential(
    exit_long_path: np.ndarray,
    exit_short_path: np.ndarray,
    entry_long: np.ndarray,
    entry_short: np.ndarray,
    atr: np.ndarray,
    penalty: float,
    vertical_bars: int,
    execution_delay_bars: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    n = exit_long_path.shape[0]
    delay = max(0, int(execution_delay_bars))
    n_valid = n - vertical_bars - delay
    
    if n_valid <= 0:
        z = np.zeros(0, dtype=np.float32)
        return z, z, z, z
        
    cpar_long_out = np.zeros(n_valid, dtype=np.float32)
    cpar_short_out = np.zeros(n_valid, dtype=np.float32)
    label_out = np.zeros(n_valid, dtype=np.float32)
    reward_out = np.zeros(n_valid, dtype=np.float32)
    
    for i in range(n_valid):
        ei = i + delay
        el = entry_long[ei]
        es = entry_short[ei]
        vol = atr[ei] if atr[ei] > 1e-8 else 1e-8

        # For long:
        end_idx = ei + vertical_bars
        fwd_ret_long = exit_long_path[end_idx] - el
        
        min_p = el
        for t in range(1, vertical_bars + 1):
            if exit_long_path[ei + t] < min_p:
                min_p = exit_long_path[ei + t]
        mae_long = max(0.0, el - min_p)
        cpar_l = (fwd_ret_long / vol) - (penalty * mae_long / vol)

        # For short:
        fwd_ret_short = es - exit_short_path[end_idx]
        
        max_p = es
        for t in range(1, vertical_bars + 1):
            if exit_short_path[ei + t] > max_p:
                max_p = exit_short_path[ei + t]
        mae_short = max(0.0, max_p - es)
        cpar_s = (fwd_ret_short / vol) - (penalty * mae_short / vol)
        
        cpar_long_out[i] = cpar_l
        cpar_short_out[i] = cpar_s
        reward_val = (cpar_l - cpar_s) / 2.0
        reward_out[i] = reward_val
        label_out[i] = _cost_aware_label(cpar_l, cpar_s)

    return cpar_long_out, cpar_short_out, reward_out, label_out


# Regime -> lookahead mapping: 0=low_vol(trending), 1=normal, 2=high_vol(volatile)
_REGIME_LOOKAHEAD: dict[int, int] = {0: 20, 1: 12, 2: 6}


def _build_lookahead_array(
    regime_class: np.ndarray | None,
    n: int,
    default: int,
) -> np.ndarray:
    """Return per-bar lookahead array based on regime_class integers."""
    if regime_class is None or len(regime_class) == 0:
        return np.full(n, default, dtype=np.int32)
    arr = np.full(n, default, dtype=np.int32)
    for regime_val, bars_val in _REGIME_LOOKAHEAD.items():
        mask = regime_class == regime_val
        arr[mask] = bars_val
    return arr


def _scan_outcomes_cpar_dynamic(
    exit_long_path: np.ndarray,
    exit_short_path: np.ndarray,
    entry_long: np.ndarray,
    entry_short: np.ndarray,
    atr: np.ndarray,
    penalty: float,
    lookahead_arr: np.ndarray,
    execution_delay_bars: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sequential CPAR scan with per-bar dynamic lookahead."""
    n = exit_long_path.shape[0]
    delay = max(0, int(execution_delay_bars))
    max_lookahead = int(lookahead_arr.max()) if len(lookahead_arr) > 0 else 1
    n_valid = n - max_lookahead - delay

    if n_valid <= 0:
        z = np.zeros(0, dtype=np.float32)
        return z, z, z, z

    cpar_long_out = np.zeros(n_valid, dtype=np.float32)
    cpar_short_out = np.zeros(n_valid, dtype=np.float32)
    label_out = np.zeros(n_valid, dtype=np.float32)
    reward_out = np.zeros(n_valid, dtype=np.float32)

    for i in range(n_valid):
        vb = int(lookahead_arr[i]) if i < len(lookahead_arr) else max_lookahead
        ei = i + delay
        if ei + vb >= n:
            vb = n - ei - 1
        if vb < 1:
            continue
        el = entry_long[ei]
        es = entry_short[ei]
        vol = atr[ei] if atr[ei] > 1e-8 else 1e-8

        end_idx = ei + vb
        fwd_ret_long = exit_long_path[end_idx] - el
        min_p = el
        for t in range(1, vb + 1):
            if exit_long_path[ei + t] < min_p:
                min_p = exit_long_path[ei + t]
        mae_long = max(0.0, el - min_p)
        cpar_l = (fwd_ret_long / vol) - (penalty * mae_long / vol)

        fwd_ret_short = es - exit_short_path[end_idx]
        max_p = es
        for t in range(1, vb + 1):
            if exit_short_path[ei + t] > max_p:
                max_p = exit_short_path[ei + t]
        mae_short = max(0.0, max_p - es)
        cpar_s = (fwd_ret_short / vol) - (penalty * mae_short / vol)

        cpar_long_out[i] = cpar_l
        cpar_short_out[i] = cpar_s
        reward_val = (cpar_l - cpar_s) / 2.0
        reward_out[i] = reward_val
        # The 'label' column must be discrete {-1, 0, 1} for row_quality checks
        label_out[i] = _cost_aware_label(cpar_l, cpar_s)

    return cpar_long_out, cpar_short_out, reward_out, label_out


def compute_triple_barrier_labels(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    atr_col: str = "atr_6",
    vertical_bars: int | None = None,
    profit_atr_mult: float | None = None,
    stop_atr_mult: float | None = None,
    pip_size: float | None = None,
    execution_delay_bars: int = 1,
    use_numba: bool | None = None,
    parallel: bool | None = None,
    pair: str | None = None,
    dynamic_lookahead: bool = True,
) -> pd.DataFrame:
    """
    Computes Continuous Path-Adjusted Reward (CPAR) labels for regression.
    Maintains function signature for compatibility with dataset_builder.py.
    """
    cfg = _default_labeling()
    if vertical_bars is None:
        vertical_bars = int(cfg.get("lookahead_bars", 30))
    if pip_size is None:
        pip_size = float(cfg.get("pip_size", 0.0001))
    if use_numba is None:
        use_numba = bool(cfg.get("tbm_numba", True))
    if parallel is None:
        parallel = bool(cfg.get("tbm_parallel", True))
    if pair and pip_size == 0.0001 and "JPY" in str(pair).upper():
        pip_size = 0.01

    # MAE penalty weight. Was hard-coded 0.5 regardless of config.
    penalty = float(cfg.get("cpar_mae_penalty", 0.5))

    if len(bars) < vertical_bars + 2:
        warnings.warn(
            f"[CPARLabeling] Only {len(bars)} bars but vertical_bars={vertical_bars}. "
            "Need at least vertical_bars + 2 rows. Returning empty DataFrame.", stacklevel=2
        )
        empty_idx = features.index[:0]
        return pd.DataFrame(
            {"reward_long": [], "reward_short": [], "reward": [], "label": []},
            index=empty_idx,
        )

    close = np.asarray(bars["close"].reindex(features.index).ffill(), dtype=np.float64)

    bid: np.ndarray | None = None
    ask: np.ndarray | None = None
    if "bid_close" in bars.columns and "ask_close" in bars.columns:
        bid = np.asarray(bars["bid_close"].reindex(features.index).ffill(), dtype=np.float64)
        ask = np.asarray(bars["ask_close"].reindex(features.index).ffill(), dtype=np.float64)
        entry_long = ask.copy()  
        entry_short = bid.copy() 
    elif "spread_pips" in features.columns:
        spread_half = np.asarray(features["spread_pips"], dtype=np.float64) * pip_size / 2.0
        entry_long = close + spread_half
        entry_short = close - spread_half
        bid = close - spread_half  
        ask = close + spread_half  
    else:
        entry_long = close.copy()
        entry_short = close.copy()
        bid = close.copy()
        ask = close.copy()

    atr = (
        np.asarray(features[atr_col], dtype=np.float64)
        if atr_col in features.columns
        else np.full(len(close), 0.0005, dtype=np.float64)
    )

    n = len(close)
    delay = max(0, int(execution_delay_bars))

    # Dynamic regime-based lookahead
    regime_arr: np.ndarray | None = None
    if dynamic_lookahead and "regime_class" in features.columns:
        regime_arr = np.asarray(features["regime_class"].reindex(features.index).ffill().fillna(1), dtype=np.int32)

    if regime_arr is not None:
        lookahead_arr = _build_lookahead_array(regime_arr, n, vertical_bars)
        max_lookahead = int(lookahead_arr.max())
        n_valid = n - max_lookahead - delay
    else:
        lookahead_arr = None
        n_valid = n - vertical_bars - delay

    if n_valid <= 0:
        z = np.zeros(0, dtype=np.float32)
        return pd.DataFrame(
            {"reward_long": z, "reward_short": z, "reward": z, "label": z},
            index=features.index,
        ).iloc[0:0]

    if lookahead_arr is not None:
        cpar_l, cpar_s, rew, lab = _scan_outcomes_cpar_dynamic(
            bid, ask, entry_long, entry_short, atr, penalty, lookahead_arr, delay
        )
        backend = "dynamic"
    elif use_numba and _NUMBA_IMPORT_OK:
        cpar_l, cpar_s, rew, lab = _scan_outcomes_cpar_numba(
            bid, ask, entry_long, entry_short, atr, penalty, vertical_bars, n_valid, delay
        )
        backend = "numba"
    else:
        cpar_l, cpar_s, rew, lab = _scan_outcomes_cpar_sequential(
            bid, ask, entry_long, entry_short, atr, penalty, vertical_bars, delay
        )
        backend = "sequential"

    result = pd.DataFrame(
        {
            "reward_long": cpar_l,
            "reward_short": cpar_s,
            "reward": rew,
            "label": lab,
        },
        index=features.index[:n_valid],
    )

    print(f"[CPARLabeling] {len(result):,} continuous labels generated | backend={backend}")
    return result
