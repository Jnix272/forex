"""
Numba-accelerated barrier scan for RL reward labeling.

Replaces the pure-Python O(n × lookahead) loops in rl_reward_labeling.py
with compiled parallel code for 10-50× speedup.
"""

import numpy as np

try:
    from numba import njit, prange
    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False
    # No-op decorators so the module still imports
    def njit(*args, **kwargs):
        def dec(f): return f
        return dec
    def prange(n): return range(n)


@njit(parallel=False, fastmath=True, cache=True)
def _scan_barriers_simple(
    close: np.ndarray,
    entry_long: np.ndarray,
    entry_short: np.ndarray,
    exit_long_path: np.ndarray,
    exit_short_path: np.ndarray,
    atr: np.ndarray,
    valid_market: np.ndarray,
    profit_atr_mult: float,
    stop_atr_mult: float,
    tx_cost_pips: float,
    pip_size: float,
    lookahead_bars: int,
    delay: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated barrier scan for compute_rl_reward_labels.

    Returns (reward_long, reward_short) arrays of shape (n,).
    """
    n = len(close)
    reward_long = np.zeros(n, dtype=np.float32)
    reward_short = np.zeros(n, dtype=np.float32)

    n_valid = n - lookahead_bars - delay
    if n_valid <= 0:
        return reward_long, reward_short

    for i in range(n_valid):
        entry_i = i + delay
        if not valid_market[i] or not valid_market[entry_i]:
            continue

        el = entry_long[entry_i]
        es = entry_short[entry_i]
        a = atr[entry_i]

        tp_l = el + profit_atr_mult * a
        sl_l = el - stop_atr_mult * a
        tp_s = es - profit_atr_mult * a
        sl_s = es + stop_atr_mult * a

        end = entry_i + 1 + lookahead_bars
        horizon_l = exit_long_path[entry_i + 1:end]
        horizon_s = exit_short_path[entry_i + 1:end]

        if len(horizon_l) == 0 or len(horizon_s) == 0:
            continue

        # Check finite and valid
        finite_ok = True
        valid_ok = True
        for j in range(len(horizon_l)):
            if not np.isfinite(horizon_l[j]) or not np.isfinite(horizon_s[j]):
                finite_ok = False
                break
            if not valid_market[entry_i + 1 + j]:
                valid_ok = False
                break

        if not finite_ok or not valid_ok:
            continue

        # Long path
        pnl_l = 0.0
        found = False
        for j in range(len(horizon_l)):
            p = horizon_l[j]
            if p >= tp_l:
                pnl_l = (tp_l - el) / pip_size
                found = True
                break
            elif p <= sl_l:
                pnl_l = (sl_l - el) / pip_size
                found = True
                break
        if not found:
            pnl_l = (horizon_l[-1] - el) / pip_size
        reward_long[i] = pnl_l - tx_cost_pips

        # Short path
        pnl_s = 0.0
        found = False
        for j in range(len(horizon_s)):
            p = horizon_s[j]
            if p <= tp_s:
                pnl_s = (es - tp_s) / pip_size
                found = True
                break
            elif p >= sl_s:
                pnl_s = (es - sl_s) / pip_size
                found = True
                break
        if not found:
            pnl_s = (es - horizon_s[-1]) / pip_size
        reward_short[i] = pnl_s - tx_cost_pips

    return reward_long, reward_short


@njit(parallel=True, fastmath=True, cache=True)
def _scan_barriers_regime(
    close: np.ndarray,
    entry_long: np.ndarray,
    entry_short: np.ndarray,
    exit_long_path: np.ndarray,
    exit_short_path: np.ndarray,
    atr: np.ndarray,
    valid_market: np.ndarray,
    # Pre-computed per-bar barrier parameters
    tp_mult: np.ndarray,      # shape (n,) — take-profit ATR multiplier
    sl_mult: np.ndarray,      # shape (n,) — stop-loss ATR multiplier
    horizon_arr: np.ndarray,  # shape (n,) — lookahead bars per entry
    tx_pips_arr: np.ndarray,   # shape (n,) — transaction cost in pips
    pip_size: float,
    delay: int,
    max_horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated barrier scan for compute_rl_reward_labels_regime.

    Returns (reward_long, reward_short, path_quality, confidence_target, tx_pips_out)
    """
    n = len(close)
    reward_long = np.zeros(n, dtype=np.float32)
    reward_short = np.zeros(n, dtype=np.float32)
    path_quality = np.zeros(n, dtype=np.float32)
    confidence_target = np.zeros(n, dtype=np.float32)
    tx_out = np.zeros(n, dtype=np.float32)

    n_valid = n - max_horizon - delay
    if n_valid <= 0:
        return reward_long, reward_short, path_quality, confidence_target, tx_out

    for i in prange(n_valid):
        entry_i = i + delay
        if not valid_market[i] or not valid_market[entry_i]:
            continue

        el = entry_long[entry_i]
        es = entry_short[entry_i]
        a = atr[entry_i]

        tp_m = tp_mult[entry_i]
        sl_m = sl_mult[entry_i]
        h = int(horizon_arr[entry_i])
        tx = tx_pips_arr[entry_i]

        if h <= 0 or entry_i + 1 + h > n:
            continue

        tp_l = el + tp_m * a
        sl_l = el - sl_m * a
        tp_s = es - tp_m * a
        sl_s = es + sl_m * a

        end = entry_i + 1 + h
        horizon_l = exit_long_path[entry_i + 1:end]
        horizon_s = exit_short_path[entry_i + 1:end]

        if len(horizon_l) == 0:
            continue

        # Check finite and valid
        finite_ok = True
        valid_ok = True
        for j in range(len(horizon_l)):
            if not np.isfinite(horizon_l[j]) or not np.isfinite(horizon_s[j]):
                finite_ok = False
                break
            if not valid_market[entry_i + 1 + j]:
                valid_ok = False
                break

        if not finite_ok or not valid_ok:
            continue

        # ── Long path ──
        mae_l = 0.0
        mfe_l = 0.0
        pnl_l = 0.0
        exit_bar_l = len(horizon_l)
        found = False

        for j, p in enumerate(horizon_l):
            mae = (el - p) / pip_size
            mfe = (p - el) / pip_size
            if mae > mae_l: mae_l = mae
            if mfe > mfe_l: mfe_l = mfe
            if p >= tp_l:
                pnl_l = (tp_l - el) / pip_size
                exit_bar_l = j + 1
                found = True
                break
            elif p <= sl_l:
                pnl_l = (sl_l - el) / pip_size
                exit_bar_l = j + 1
                found = True
                break
        if not found:
            pnl_l = (horizon_l[-1] - el) / pip_size

        # Bad-win penalty
        bad_win_l = 0.0
        if pnl_l > 0:
            is_large_mae = mae_l > abs(pnl_l) * 1.5
            is_late_exit = exit_bar_l > h * 0.8
            is_spike_win = (mfe_l > tp_m * a / pip_size * 1.5) and (exit_bar_l <= 2)
            if is_large_mae or is_late_exit or is_spike_win:
                bad_win_l = abs(pnl_l) * 0.3
        reward_long[i] = pnl_l - tx - bad_win_l

        # ── Short path ──
        mae_s = 0.0
        mfe_s = 0.0
        pnl_s = 0.0
        exit_bar_s = len(horizon_s)
        found = False

        for j, p in enumerate(horizon_s):
            mae = (p - es) / pip_size
            mfe = (es - p) / pip_size
            if mae > mae_s: mae_s = mae
            if mfe > mfe_s: mfe_s = mfe
            if p <= tp_s:
                pnl_s = (es - tp_s) / pip_size
                exit_bar_s = j + 1
                found = True
                break
            elif p >= sl_s:
                pnl_s = (es - sl_s) / pip_size
                exit_bar_s = j + 1
                found = True
                break
        if not found:
            pnl_s = (es - horizon_s[-1]) / pip_size

        # Bad-win penalty (short)
        bad_win_s = 0.0
        if pnl_s > 0:
            is_large_mae = mae_s > abs(pnl_s) * 1.5
            is_late_exit = exit_bar_s > h * 0.8
            is_spike_win = (mfe_s > tp_m * a / pip_size * 1.5) and (exit_bar_s <= 2)
            if is_large_mae or is_late_exit or is_spike_win:
                bad_win_s = abs(pnl_s) * 0.3
        reward_short[i] = pnl_s - tx - bad_win_s

        # Path quality (low when MAE is close to loss size)
        max_adverse = max(mae_l, mae_s)
        barrier_size = tp_m * a / pip_size + 1e-6
        conf_target = 1.0 - max_adverse / barrier_size
        if conf_target < 0.0:
            conf_target = 0.0
        elif conf_target > 1.0:
            conf_target = 1.0
        confidence_target[i] = conf_target

        # Path quality: 1.0 = clean, 0.0 = noisy/meandering
        # Good paths: small MAE relative to reward, timely exit
        if pnl_l > pnl_s and pnl_l > 0:
            pq_val = 1.0 - mae_l / max(abs(pnl_l), 1e-6)
            if pq_val < 0.2:
                pq_val = 0.2
            elif pq_val > 1.0:
                pq_val = 1.0
            path_quality[i] = pq_val
        elif pnl_s > 0:
            pq_val = 1.0 - mae_s / max(abs(pnl_s), 1e-6)
            if pq_val < 0.2:
                pq_val = 0.2
            elif pq_val > 1.0:
                pq_val = 1.0
            path_quality[i] = pq_val
        else:
            path_quality[i] = 0.5  # neutral for losing/zero-reward trades

        tx_out[i] = tx

    return reward_long, reward_short, path_quality, confidence_target, tx_out


def _numba_available() -> bool:
    """Check if Numba is available."""
    return _NUMBA_OK
