"""
labeling/triple_barrier_labeling.py
====================================
Triple Barrier Method (TBM): ATR-scaled TP/SL + vertical horizon; first touch
wins. Parallel **Numba** scan over all bars (optional; falls back to a sequential
reference implementation if Numba is unavailable).

Fully automated: enable/disable via LABELING["tbm_numba"] / ["tbm_parallel"] in
config/settings.py (no manual steps in training).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

# ── Optional Numba (required for fast path on large series) ─────────────────
try:
    from numba import njit, prange
    _NUMBA_IMPORT_OK = True
except ImportError:
    _NUMBA_IMPORT_OK = False
    njit = None  # type: ignore
    prange = range  # type: ignore


def _default_labeling() -> dict[str, Any]:
    try:
        from config.settings import LABELING as L
        return L
    except Exception:
        return {}


def _scan_outcomes_sequential(
    close: np.ndarray,
    entry_long: np.ndarray,
    entry_short: np.ndarray,
    atr: np.ndarray,
    profit_mult: float,
    stop_mult: float,
    vertical_bars: int,
    execution_delay_bars: int = 0,
    bid: np.ndarray = None,
    ask: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation (single-threaded). Used for tests and fallback.

    DS-001: When bid/ask arrays are provided, long exits are evaluated at bid
    and short exits at ask (realistic execution pricing).
    """
    n = close.shape[0]
    delay = max(0, int(execution_delay_bars))
    n_valid = n - vertical_bars - delay
    if n_valid <= 0:
        z = np.zeros(0, dtype=np.int8)
        return z, z.astype(np.int32), z, z.astype(np.int32)

    # DS-001: use bid for long exits, ask for short exits
    exit_long_path = bid if bid is not None else close
    exit_short_path = ask if ask is not None else close

    lo_o = np.zeros(n_valid, dtype=np.int8)
    tl_o = np.zeros(n_valid, dtype=np.int32)
    so_o = np.zeros(n_valid, dtype=np.int8)
    ts_o = np.zeros(n_valid, dtype=np.int32)

    for i in range(n_valid):
        ei = i + delay
        el = entry_long[ei]
        tp_l = el + profit_mult * atr[ei]
        sl_l = el - stop_mult * atr[ei]
        es = entry_short[ei]
        tp_s = es - profit_mult * atr[ei]
        sl_s = es + stop_mult * atr[ei]

        lo = np.int8(0)
        tl = np.int32(vertical_bars)
        so = np.int8(0)
        ts = np.int32(vertical_bars)

        for t in range(vertical_bars):
            p_long = exit_long_path[ei + 1 + t]
            p_short = exit_short_path[ei + 1 + t]
            if lo == 0:
                if p_long >= tp_l:
                    lo = 1
                    tl = t
                elif p_long <= sl_l:
                    lo = -1
                    tl = t
            if so == 0:
                if p_short <= tp_s:
                    so = 1
                    ts = t
                elif p_short >= sl_s:
                    so = -1
                    ts = t
            if lo != 0 and so != 0:
                break

        lo_o[i] = lo
        tl_o[i] = tl
        so_o[i] = so
        ts_o[i] = ts

    return lo_o, tl_o, so_o, ts_o


if _NUMBA_IMPORT_OK:

    @njit(cache=True, fastmath=True, parallel=True)
    def _scan_outcomes_numba(
        exit_long_path,
        exit_short_path,
        entry_long,
        entry_short,
        atr,
        profit_mult,
        stop_mult,
        vertical_bars,
        n_valid,
        execution_delay_bars,
    ):
        """DS-001: exit_long_path=bid (long exits at bid), exit_short_path=ask (short exits at ask)."""
        lo_o = np.zeros(n_valid, dtype=np.int8)
        tl_o = np.zeros(n_valid, dtype=np.int32)
        so_o = np.zeros(n_valid, dtype=np.int8)
        ts_o = np.zeros(n_valid, dtype=np.int32)
        delay = max(0, int(execution_delay_bars))

        for i in prange(n_valid):
            ei = i + delay
            el = entry_long[ei]
            tp_l = el + profit_mult * atr[ei]
            sl_l = el - stop_mult * atr[ei]
            es = entry_short[ei]
            tp_s = es - profit_mult * atr[ei]
            sl_s = es + stop_mult * atr[ei]

            lo = np.int8(0)
            tl = vertical_bars
            so = np.int8(0)
            ts = vertical_bars

            for t in range(vertical_bars):
                p_long = exit_long_path[ei + 1 + t]
                p_short = exit_short_path[ei + 1 + t]
                if lo == 0:
                    if p_long >= tp_l:
                        lo = 1
                        tl = t
                    elif p_long <= sl_l:
                        lo = -1
                        tl = t
                if so == 0:
                    if p_short <= tp_s:
                        so = 1
                        ts = t
                    elif p_short >= sl_s:
                        so = -1
                        ts = t
                if lo != 0 and so != 0:
                    break

            lo_o[i] = lo
            tl_o[i] = tl
            so_o[i] = so
            ts_o[i] = ts

        return lo_o, tl_o, so_o, ts_o

    @njit(cache=True, fastmath=True)
    def _scan_outcomes_numba_serial(
        exit_long_path,
        exit_short_path,
        entry_long,
        entry_short,
        atr,
        profit_mult,
        stop_mult,
        vertical_bars,
        n_valid,
        execution_delay_bars,
    ):
        """DS-001: uses bid path for long exits, ask path for short exits."""
        lo_o = np.zeros(n_valid, dtype=np.int8)
        tl_o = np.zeros(n_valid, dtype=np.int32)
        so_o = np.zeros(n_valid, dtype=np.int8)
        ts_o = np.zeros(n_valid, dtype=np.int32)
        delay = max(0, int(execution_delay_bars))

        for i in range(n_valid):
            ei = i + delay
            el = entry_long[ei]
            tp_l = el + profit_mult * atr[ei]
            sl_l = el - stop_mult * atr[ei]
            es = entry_short[ei]
            tp_s = es - profit_mult * atr[ei]
            sl_s = es + stop_mult * atr[ei]

            lo = np.int8(0)
            tl = vertical_bars
            so = np.int8(0)
            ts = vertical_bars

            for t in range(vertical_bars):
                p_long = exit_long_path[ei + 1 + t]
                p_short = exit_short_path[ei + 1 + t]
                if lo == 0:
                    if p_long >= tp_l:
                        lo = 1
                        tl = t
                    elif p_long <= sl_l:
                        lo = -1
                        tl = t
                if so == 0:
                    if p_short <= tp_s:
                        so = 1
                        ts = t
                    elif p_short >= sl_s:
                        so = -1
                        ts = t
                if lo != 0 and so != 0:
                    break

            lo_o[i] = lo
            tl_o[i] = tl
            so_o[i] = so
            ts_o[i] = ts

        return lo_o, tl_o, so_o, ts_o


def _combine_directional_labels(
    lo: np.ndarray,
    tl: np.ndarray,
    so: np.ndarray,
    ts: np.ndarray,
) -> np.ndarray:
    """Vectorized merge: bullish/bearish/neutral from long/short barrier outcomes."""
    n = lo.shape[0]
    label = np.zeros(n, dtype=np.int8)
    both_tp = (lo == 1) & (so == 1)
    if np.any(both_tp):
        label[both_tp] = np.where(tl[both_tp] <= ts[both_tp], 1, -1).astype(np.int8)
    rest = ~both_tp
    label[rest & (lo == 1)] = 1
    label[rest & (so == 1)] = -1
    return label


def _run_barrier_scan(
    close: np.ndarray,
    entry_long: np.ndarray,
    entry_short: np.ndarray,
    atr: np.ndarray,
    profit_mult: float,
    stop_mult: float,
    vertical_bars: int,
    use_numba: bool,
    parallel: bool,
    execution_delay_bars: int = 0,
    bid: np.ndarray = None,
    ask: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Returns (lo, tl, so, ts) each length n_valid, and backend tag for logging.
    DS-001: bid/ask used for realistic exit pricing when available.
    """
    # DS-001: use bid for long exit path, ask for short exit path
    exit_long_path = bid if bid is not None else close
    exit_short_path = ask if ask is not None else close
    n = close.shape[0]
    delay = max(0, int(execution_delay_bars))
    n_valid = n - vertical_bars - delay
    if n_valid <= 0:
        z = np.zeros(0, dtype=np.int8)
        return z, z.astype(np.int32), z, z.astype(np.int32), "empty"

    if use_numba and _NUMBA_IMPORT_OK:
        try:
            if parallel:
                lo, tl, so, ts = _scan_outcomes_numba(
                    exit_long_path,
                    exit_short_path,
                    entry_long,
                    entry_short,
                    atr,
                    profit_mult,
                    stop_mult,
                    vertical_bars,
                    n_valid,
                    delay,
                )
                return lo, tl, so, ts, "numba_parallel"
            lo, tl, so, ts = _scan_outcomes_numba_serial(
                exit_long_path,
                exit_short_path,
                entry_long,
                entry_short,
                atr,
                profit_mult,
                stop_mult,
                vertical_bars,
                n_valid,
                delay,
            )
            return lo, tl, so, ts, "numba_serial"
        except Exception as ex:
            warnings.warn(f"[TBM] Numba scan failed ({ex}); using sequential scan.")
            return (
                *_scan_outcomes_sequential(
                    exit_long_path,
                    exit_short_path,
                    entry_long,
                    entry_short,
                    atr,
                    profit_mult,
                    stop_mult,
                    vertical_bars,
                    delay,
                    bid=bid,
                    ask=ask,
                ),
                "sequential_fallback",
            )

    return (
        *_scan_outcomes_sequential(
            close,
            entry_long,
            entry_short,
            atr,
            profit_mult,
            stop_mult,
            vertical_bars,
            delay,
            bid=bid,
            ask=ask,
        ),
        "sequential",
    )


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
    Per-bar triple-barrier outcomes; combined directional label.

    When ``use_numba`` / ``parallel`` are None, values are read from
    ``config.settings.LABELING`` (``tbm_numba``, ``tbm_parallel``).
    ``pair`` is accepted for trainer API compatibility (JPY pairs can use
    a larger default pip size when ``pip_size`` is left at the default).
    """
    cfg = _default_labeling()
    if vertical_bars is None:
        vertical_bars = int(cfg.get("lookahead_bars", 30))
    if profit_atr_mult is None:
        profit_atr_mult = float(cfg.get("profit_target_atr", 1.2))
    if stop_atr_mult is None:
        stop_atr_mult = float(cfg.get("stop_loss_atr", 0.8))
    if pip_size is None:
        pip_size = float(cfg.get("pip_size", 0.0001))
    if use_numba is None:
        use_numba = bool(cfg.get("tbm_numba", True))
    if parallel is None:
        parallel = bool(cfg.get("tbm_parallel", True))
    if pair and pip_size == 0.0001 and "JPY" in str(pair).upper():
        pip_size = 0.01

    # Guard: need at least vertical_bars + 2 rows to produce any label.
    if len(bars) < vertical_bars + 2:
        warnings.warn(
            f"[TBMLabeling] Only {len(bars)} bars but vertical_bars={vertical_bars}. "
            "Need at least vertical_bars + 2 rows. Returning empty DataFrame."
        )
        empty_idx = features.index[:0]
        return pd.DataFrame(
            {"reward_long": [], "reward_short": [], "reward": [], "label": []},
            index=empty_idx,
        )

    close = bars["close"].reindex(features.index).ffill().values.astype(np.float64)

    # DS-001: Extract bid/ask for realistic exit pricing
    bid = None
    ask = None
    if "bid_close" in bars.columns and "ask_close" in bars.columns:
        bid = bars["bid_close"].reindex(features.index).ffill().values.astype(np.float64)
        ask = bars["ask_close"].reindex(features.index).ffill().values.astype(np.float64)
        entry_long = ask.copy()   # enter long at ask
        entry_short = bid.copy()  # enter short at bid
    elif "spread_pips" in features.columns:
        spread_half = features["spread_pips"].values.astype(np.float64) * pip_size / 2
        entry_long = close + spread_half
        entry_short = close - spread_half
        bid = close - spread_half  # approximate bid from spread
        ask = close + spread_half  # approximate ask from spread
    else:
        entry_long = close.copy()
        entry_short = close.copy()

    atr = (
        features[atr_col].values.astype(np.float64)
        if atr_col in features.columns
        else np.full(len(close), 0.0005, dtype=np.float64)
    )

    n = len(close)
    delay = max(0, int(execution_delay_bars))
    n_valid = n - vertical_bars - delay
    reward_long = np.zeros(n, dtype=np.float32)
    reward_short = np.zeros(n, dtype=np.float32)
    label = np.zeros(n, dtype=np.int8)

    if n_valid <= 0:
        return pd.DataFrame(
            {
                "reward_long": reward_long,
                "reward_short": reward_short,
                "reward": reward_long,
                "label": label,
            },
            index=features.index,
        ).iloc[0:0]

    lo, tl, so, ts, backend = _run_barrier_scan(
        close,
        entry_long,
        entry_short,
        atr,
        float(profit_atr_mult),
        float(stop_atr_mult),
        int(vertical_bars),
        use_numba=use_numba,
        parallel=parallel,
        execution_delay_bars=delay,
        bid=bid,
        ask=ask,
    )

    comb = _combine_directional_labels(lo, tl, so, ts)
    reward_long[:n_valid] = lo.astype(np.float32)
    reward_short[:n_valid] = so.astype(np.float32)
    label[:n_valid] = comb

    reward = np.where(
        np.abs(reward_long) >= np.abs(reward_short),
        reward_long,
        reward_short,
    )

    result = pd.DataFrame(
        {
            "reward_long": reward_long,
            "reward_short": reward_short,
            "reward": reward,
            "label": label,
        },
        index=features.index,
    )
    # Drop bars that lack a full forward window (and any unused tail slots).
    result = result.iloc[:n_valid].copy()

    vc = pd.Series(result["label"]).value_counts()
    print(
        f"[TBMLabeling] {len(result):,} labels | backend={backend} | "
        f"Long+: {vc.get(1, 0):,}  Hold: {vc.get(0, 0):,}  Short+: {vc.get(-1, 0):,}"
    )
    return result
