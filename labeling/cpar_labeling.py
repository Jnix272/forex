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
        label_val = (cpar_l - cpar_s) / 2.0
        label_out[i] = label_val
        
        reward_out[i] = label_val

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
        label_val = (cpar_l - cpar_s) / 2.0
        label_out[i] = label_val
        reward_out[i] = label_val

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

    # Penalty for CPAR (use stop_loss_atr as a base or hardcode)
    penalty = 0.5 

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
    n_valid = n - vertical_bars - delay

    if n_valid <= 0:
        z = np.zeros(0, dtype=np.float32)
        return pd.DataFrame(
            {"reward_long": z, "reward_short": z, "reward": z, "label": z},
            index=features.index,
        ).iloc[0:0]

    if use_numba and _NUMBA_IMPORT_OK:
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
