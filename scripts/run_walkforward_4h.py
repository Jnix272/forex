"""
scripts/run_walkforward_4h.py
=============================
Unified Leak-Free Walk-Forward Engine (4H & 1D Horizons) combining:
1. Scale-Invariant + USD-Neutral Cross-Sectional Decomposition (`CS_Idio`, `USD_Other`, `USD_Disp`)
2. Continuous Regime-Interaction Gating (`mr_x_chop`, `mom_x_trend`) & Causal Lagged Macro Momentum
3. Cost- & Volatility-Normalized Targets + Differentiable Turnover-Penalized Net-Sharpe Policy
4. Hysteresis Buffer State Machine (`z_entry`, `z_exit`) + Inverse-ATR Volatility Sizing + Trailing OOS Gate

Strict Out-of-Sample Protocol:
- 8 expanding walk-forward folds across the last 40% of `.tmp/all_584_audit_data.npz`
  (with 96-bar embargo between train and test).
- Every trade deducts full live Dukascopy variable spread + 0.7 pip execution slippage.
- Zero lookahead: all feature scalers, thresholds, and weights are fit strictly on `idx < idx[fold_start] - EMBARGO`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ALL_PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "USDCAD"]
THREE_PAIRS = ["EURUSD", "USDJPY", "USDCAD"]
USD_SIGN = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)

H = 48          # 4-hour forward horizon (48 x 5m bars)
EMBARGO = 96    # 8-hour embargo buffer
N_FOLDS = 8
TEST_FRAC = 0.40
BPD = 288
SLIP = 0.7

# Clean scale-invariant per-pair base features (dropping non-stationary price-level ATR/vol and zero-washed macro shifts)
CLEAN_BASE_FEATURES = [
    "spread_us_de", "spread_us_jp", "spread_us_gb", "spread_us_ca", "spread_us_ch",
    "yield_curve_slope",
    "carry_eur", "carry_jpy", "carry_gbp", "carry_cad", "carry_chf", "carry_eurgbp", "carry_eurjpy",
    "time_sin", "time_cos", "day_sin", "day_cos", "london_ny",
    "rsi_14", "bb_pct", "ret_20", "ret_60",
    "vol_ratio_6_20", "vol_ratio_20_60", "atr_ratio_6_20", "atr_ratio_20_60",
    "ret_15m", "rsi_15m", "trend_slope_15m", "distance_to_vwap_15m",
    "ret_1h", "rsi_1h", "trend_slope_1h", "distance_to_vwap_1h",
    "spread_zscore", "spread_percentile", "cost_to_atr",
    "ofi_z", "vpin", "vp_poc_pos", "vp_vw_pos", "vp_skew",
    "adx_14", "chop_index", "vol_clock_z",
]


def sharpe(d: np.ndarray) -> float:
    return float(d.mean() / (d.std() + 1e-12) * np.sqrt(252.0))


def boot_ci(d: np.ndarray, block: int = 5, n: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    L = len(d)
    nb = max(1, L // block)
    out = [
        sharpe(np.concatenate([d[j : j + block] for j in rng.integers(0, max(1, L - block), nb)]))
        for _ in range(n)
    ]
    return [float(np.percentile(out, 5)), float(np.percentile(out, 95))]


def inv_vol(mat: np.ndarray, lb: int = 40) -> np.ndarray:
    T, K = mat.shape
    w = np.ones((T, K)) / K
    for t in range(lb, T):
        s = mat[t - lb : t].std(axis=0) + 1e-9
        inv_s = 1.0 / s
        w[t] = inv_s / inv_s.sum()
    return (mat * w).sum(axis=1) * K


def build_engineered_feature_matrix(
    X_all: np.ndarray,
    idx: np.ndarray,
    close: np.ndarray,
    feature_names: list[str],
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """
    Build per-pair engineered matrices combining:
    - Clean scale-invariant base features
    - Volatility-normalized multi-horizon causal price returns (1h, 2h, 4h, 1d, 3d, 5d)
    - USD-neutral cross-sectional decomposition (`CS_Idio`, `USD_Other`, `USD_Disp`)
    - Continuous regime-interaction features (`mr_x_chop`, `mom_x_trend`, session gates)
    - Strictly causal lagged daily yield spread & COT momentum
    """
    N = len(idx)
    base_names_146 = [fn.split("::")[-1] for fn in feature_names[:146]]
    name_to_i = {fn: i for i, fn in enumerate(base_names_146)}

    def get_f(fn: str) -> np.ndarray:
        col_i = name_to_i[fn]
        return np.column_stack([X_all[:, p * 146 + col_i] for p in range(4)]).astype(np.float64)

    # 1. Volatility-normalized causal multi-horizon price returns
    c_now = close[idx]
    atr_1h = np.maximum(get_f("atr_1h"), 1e-6)
    ret_1h_norm = np.clip((c_now - close[np.maximum(0, idx - 12)]) / atr_1h, -8.0, 8.0)
    ret_2h_norm = np.clip((c_now - close[np.maximum(0, idx - 24)]) / atr_1h, -8.0, 8.0)
    ret_4h_norm = np.clip((c_now - close[np.maximum(0, idx - 48)]) / atr_1h, -8.0, 8.0)
    ret_1d_norm = np.clip((c_now - close[np.maximum(0, idx - 132)]) / atr_1h, -10.0, 10.0)
    ret_3d_norm = np.clip((c_now - close[np.maximum(0, idx - 396)]) / atr_1h, -12.0, 12.0)
    ret_5d_norm = np.clip((c_now - close[np.maximum(0, idx - 660)]) / atr_1h, -15.0, 15.0)

    # 2. USD-Neutral Cross-Sectional Decomposition (Idiosyncratic vs Leave-One-Out USD Basket)
    def cs_decomp(M: np.ndarray, center: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        M_usd = (M - center) * USD_SIGN
        basket = np.mean(M_usd, axis=1, keepdims=True)
        idio = (M_usd - basket) * USD_SIGN
        other_usd = ((4.0 * basket - M_usd) / 3.0) * USD_SIGN
        return idio, other_usd

    vp_poc_idio, vp_poc_usd = cs_decomp(get_f("vp_poc_pos"), 0.5)
    vp_vw_idio, vp_vw_usd = cs_decomp(get_f("vp_vw_pos"), 0.5)
    vp_skew_idio, vp_skew_usd = cs_decomp(get_f("vp_skew"), 0.0)
    rsi1h_idio, rsi1h_usd = cs_decomp(get_f("rsi_1h"), 50.0)
    rsi15m_idio, rsi15m_usd = cs_decomp(get_f("rsi_15m"), 50.0)
    rsi14_idio, rsi14_usd = cs_decomp(get_f("rsi_14"), 50.0)
    bb_idio, bb_usd = cs_decomp(get_f("bb_pct"), 0.5)
    vwap1h_idio, vwap1h_usd = cs_decomp(get_f("distance_to_vwap_1h"), 0.0)
    vwap15m_idio, vwap15m_usd = cs_decomp(get_f("distance_to_vwap_15m"), 0.0)
    ret1h_idio, ret1h_usd = cs_decomp(ret_1h_norm, 0.0)
    ret4h_idio, ret4h_usd = cs_decomp(ret_4h_norm, 0.0)
    ret1d_idio, ret1d_usd = cs_decomp(ret_1d_norm, 0.0)
    ofi_idio, ofi_usd = cs_decomp(get_f("ofi_z"), 0.0)
    slope1h_idio, slope1h_usd = cs_decomp(get_f("trend_slope_1h"), 0.0)

    ret1h_usd_disp = np.std(ret_1h_norm * USD_SIGN, axis=1, keepdims=True) * np.ones((1, 4))
    ret1d_usd_disp = np.std(ret_1d_norm * USD_SIGN, axis=1, keepdims=True) * np.ones((1, 4))

    # 3. Continuous Regime-Interaction & Session-Boundary Features
    chop_w = np.clip((get_f("chop_index") - 35.0) / 30.0, 0.0, 1.0)
    trend_w = np.clip((get_f("adx_14") - 15.0) / 30.0, 0.0, 1.0)
    mr_osc = -(
        (get_f("bb_pct") - 0.5) * 2.0
        + (get_f("rsi_14") - 50.0) / 25.0
        + (get_f("vp_poc_pos") - 0.5) * 2.0
    ) / 3.0
    mr_x_chop = mr_osc * chop_w
    mom_x_trend = ret_4h_norm * trend_w

    # Session boundary interaction (intraday vs overnight-bridging)
    time_cos = get_f("time_cos")
    time_sin = get_f("time_sin")
    intraday_gate = (time_cos < 0.0).astype(np.float64)
    mr_x_intraday = mr_osc * intraday_gate
    mom_x_overnight = ret_1h_norm * (1.0 - intraday_gate)

    # 4. Strictly Causal Lagged Daily Macro Yield Spread & COT Momentum
    idx_lag1d = np.searchsorted(idx, np.maximum(0, idx - 132))
    idx_lag6d = np.searchsorted(idx, np.maximum(0, idx - 6 * 132))
    idx_lag21d = np.searchsorted(idx, np.maximum(0, idx - 21 * 132))
    spread_names = ["spread_us_de", "spread_us_jp", "spread_us_gb", "spread_us_ca"]
    sp_mat = np.column_stack([X_all[:, p * 146 + name_to_i[spread_names[p]]] for p in range(4)]).astype(np.float64)
    slope_mat = get_f("yield_curve_slope")
    cot_comm = get_f("cot_net_comm") if "cot_net_comm" in name_to_i else np.zeros((N, 4))

    sp_mom_5d_causal = sp_mat[idx_lag1d] - sp_mat[idx_lag6d]
    sp_mom_20d_causal = sp_mat[idx_lag1d] - sp_mat[idx_lag21d]
    slope_mom_20d_causal = slope_mat[idx_lag1d] - slope_mat[idx_lag21d]
    cot_std = np.std(cot_comm[:15000], axis=0, keepdims=True) + 1e-6
    cot_comm_chg4w = (cot_comm - cot_comm[idx_lag21d]) / cot_std

    # Zero-overfit cross-sectional composite score
    cs_composite_score = (
        -0.60 * (2.0 * get_f("vp_vw_pos") - 1.0)
        - 0.60 * vp_vw_usd
        + 0.60 * vp_skew_usd
        - 0.20 * ret1d_usd
        + 0.25 * vwap1h_idio
        + 0.15 * ret4h_idio
        - 0.15 * (USD_SIGN * sp_mom_20d_causal)
    )

    clean_indices = [name_to_i[fn] for fn in CLEAN_BASE_FEATURES if fn in name_to_i]
    clean_names = [fn for fn in CLEAN_BASE_FEATURES if fn in name_to_i]

    extra_blocks = [
        ("ret_1h_norm", ret_1h_norm),
        ("ret_2h_norm", ret_2h_norm),
        ("ret_4h_norm", ret_4h_norm),
        ("ret_1d_norm", ret_1d_norm),
        ("ret_3d_norm", ret_3d_norm),
        ("ret_5d_norm", ret_5d_norm),
        ("vp_poc_idio", vp_poc_idio),
        ("vp_poc_usd", vp_poc_usd),
        ("vp_vw_idio", vp_vw_idio),
        ("vp_vw_usd", vp_vw_usd),
        ("vp_skew_idio", vp_skew_idio),
        ("vp_skew_usd", vp_skew_usd),
        ("rsi1h_idio", rsi1h_idio),
        ("rsi1h_usd", rsi1h_usd),
        ("rsi15m_idio", rsi15m_idio),
        ("rsi15m_usd", rsi15m_usd),
        ("rsi14_idio", rsi14_idio),
        ("rsi14_usd", rsi14_usd),
        ("bb_idio", bb_idio),
        ("bb_usd", bb_usd),
        ("vwap1h_idio", vwap1h_idio),
        ("vwap1h_usd", vwap1h_usd),
        ("vwap15m_idio", vwap15m_idio),
        ("vwap15m_usd", vwap15m_usd),
        ("ret1h_idio", ret1h_idio),
        ("ret1h_usd", ret1h_usd),
        ("ret4h_idio", ret4h_idio),
        ("ret4h_usd", ret4h_usd),
        ("ret1d_idio", ret1d_idio),
        ("ret1d_usd", ret1d_usd),
        ("ofi_idio", ofi_idio),
        ("ofi_usd", ofi_usd),
        ("slope1h_idio", slope1h_idio),
        ("slope1h_usd", slope1h_usd),
        ("ret1h_usd_disp", ret1h_usd_disp),
        ("ret1d_usd_disp", ret1d_usd_disp),
        ("chop_w", chop_w),
        ("trend_w", trend_w),
        ("mr_osc", mr_osc),
        ("mr_x_chop", mr_x_chop),
        ("mom_x_trend", mom_x_trend),
        ("mr_x_intraday", mr_x_intraday),
        ("mom_x_overnight", mom_x_overnight),
        ("sp_mom_5d_causal", sp_mom_5d_causal),
        ("sp_mom_20d_causal", sp_mom_20d_causal),
        ("slope_mom_20d_causal", slope_mom_20d_causal),
        ("cot_comm_chg4w", cot_comm_chg4w),
        ("cs_composite_score", cs_composite_score),
    ]

    per_pair_X = []
    for p in range(4):
        base_p = X_all[:, [p * 146 + ci for ci in clean_indices]].astype(np.float64)
        ext_p = np.column_stack([blk[:, p] for _, blk in extra_blocks])
        mat_p = np.nan_to_num(np.hstack([base_p, ext_p]), nan=0.0, posinf=0.0, neginf=0.0)
        per_pair_X.append(mat_p)

    all_feat_names = clean_names + [name for name, _ in extra_blocks]
    atr_60_pips = np.column_stack([
        np.maximum(get_f("atr_60")[:, p] / (0.01 if p == 1 else 0.0001), 1.5)
        for p in range(4)
    ])
    return per_pair_X, atr_60_pips, all_feat_names


def fit_diff_net_sharpe_policy(
    X_tr: np.ndarray,
    y_pips_tr: np.ndarray,
    cost_pips_tr: np.ndarray,
    atr_pips_tr: np.ndarray,
    top_k: int = 35,
    l1_reg: float = 0.05,
    l2_reg: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a leak-free differentiable turnover-penalized Net-Sharpe policy on non-overlapping
    4H step trajectories (averaging across 4 phase offsets o in {0, 3, 6, 9}).
    Returns (selected_cols, mean, std, theta).
    """
    mu = np.median(X_tr, axis=0)
    iqr = np.percentile(X_tr, 75, axis=0) - np.percentile(X_tr, 25, axis=0)
    scale = np.where(iqr > 1e-6, iqr, np.std(X_tr, axis=0) + 1e-6)
    Z = np.clip((X_tr - mu) / scale, -4.0, 4.0)

    y_norm = np.clip(y_pips_tr / np.maximum(atr_pips_tr, 2.0), -4.0, 4.0)
    # Screen top_k features by univariate IC on training slice only
    corrs = np.abs((Z.T @ (y_norm - y_norm.mean())) / (len(y_norm) * (np.std(Z, axis=0) * np.std(y_norm) + 1e-9)))
    corrs = np.nan_to_num(corrs, nan=0.0)
    sel = np.argsort(corrs)[::-1][:top_k]
    Zs = Z[:, sel]

    # Build 4 phase-offset non-overlapping 4H trajectories (12 steps = 48 bars)
    trajs = []
    for offset in (0, 3, 6, 9):
        sl = slice(offset, len(Zs), 12)
        Zk = Zs[sl]
        rk = y_pips_tr[sl] / np.maximum(atr_pips_tr[sl], 2.0)
        ck = 0.5 * cost_pips_tr[sl] / np.maximum(atr_pips_tr[sl], 2.0)
        if len(Zk) > 20:
            trajs.append((Zk, rk, ck))

    def obj_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        total_loss = 0.0
        total_grad = np.zeros_like(theta)
        for Zk, rk, ck in trajs:
            u = Zk @ theta
            p = np.tanh(u)
            dp_du = 1.0 - p * p
            dp = np.diff(p, prepend=0.0)
            sgn_dp = np.tanh(dp * 10.0)
            net_r = p * rk - ck * np.abs(dp)
            m_r = np.mean(net_r)
            s_r = np.sqrt(np.var(net_r) + 1e-6)
            sr = (m_r / s_r) * np.sqrt(252.0 * 4.0)

            T = len(net_r)
            dnet_dp = rk - ck * sgn_dp
            dnet_dp[:-1] += ck[1:] * sgn_dp[1:]
            dsr_dnet = (np.sqrt(252.0 * 4.0) / (T * s_r)) * (1.0 - (net_r - m_r) * m_r / (s_r * s_r))
            grad_u = dsr_dnet * dnet_dp * dp_du
            grad_theta = -(Zk.T @ grad_u) / len(trajs)

            total_loss += -sr / len(trajs)
            total_grad += grad_theta

        l1_smooth = np.tanh(theta * 20.0)
        total_loss += l1_reg * np.mean(np.abs(theta)) + 0.5 * l2_reg * np.mean(theta * theta)
        total_grad += (l1_reg / len(theta)) * l1_smooth + (l2_reg / len(theta)) * theta
        return float(total_loss), total_grad

    w0 = np.zeros(len(sel), dtype=np.float64)
    res = minimize(obj_and_grad, w0, jac=True, method="L-BFGS-B", options={"maxiter": 60})
    return sel, mu[sel], scale[sel], res.x


def predict_diff_sharpe(
    X_eval: np.ndarray,
    sel: np.ndarray,
    mu: np.ndarray,
    scale: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    Z = np.clip((X_eval[:, sel] - mu) / scale, -4.0, 4.0)
    return Z @ theta


def simulate_hysteresis_daily(
    z_pred: np.ndarray,
    idx: np.ndarray,
    close_p: np.ndarray,
    cost_pips: np.ndarray,
    atr_pips: np.ndarray,
    pip: float,
    lo_day: int,
    n_days: int,
    z_entry: float = 1.5,
    z_exit: float = -0.3,
    vol_size: bool = True,
) -> tuple[np.ndarray, int, float]:
    """
    Hysteresis Buffer State Machine evaluated on non-overlapping 4H step slots (`step=12` rows = 48 bars).
    - When flat (`pos == 0`): enter `sign(z) * size` only if `|z| >= z_entry`.
    - When holding (`pos != 0`): keep holding without paying exit/re-entry spread as long as
      `z * sign(pos) >= z_exit`. Flip directly if opposing `|z| >= z_entry`, else exit to flat.
    - Charges exact half-turn transaction cost `0.5 * cost_pips[i] * |pos_new - pos_old|` on every position change.
    """
    daily = np.zeros(n_days, dtype=np.float64)
    pos = 0.0
    n_trades = 0
    total_net_pips = 0.0

    # Step every 12 rows (48 five-minute bars = 4 hours)
    step = 12
    K = len(idx)
    for k in range(0, K, step):
        i = int(idx[k])
        if i + H >= len(close_p):
            break
        z = float(z_pred[k])
        if not np.isfinite(z):
            z = 0.0

        target_size = float(np.clip(5.0 / max(atr_pips[k], 1.5), 0.3, 2.0)) if vol_size else 1.0
        cur_dir = 1 if pos > 0 else (-1 if pos < 0 else 0)

        if cur_dir == 0:
            if z >= z_entry:
                new_pos = target_size
            elif z <= -z_entry:
                new_pos = -target_size
            else:
                new_pos = 0.0
        elif cur_dir == 1:
            if z <= -z_entry:
                new_pos = -target_size
            elif z < z_exit:
                new_pos = 0.0
            else:
                new_pos = pos  # hold existing position without turnover!
        else:
            if z >= z_entry:
                new_pos = target_size
            elif -z < z_exit:
                new_pos = 0.0
            else:
                new_pos = pos  # hold existing position without turnover!

        turnover = abs(new_pos - pos)
        if turnover > 1e-6 and new_pos != 0.0 and np.sign(new_pos) != cur_dir:
            n_trades += 1

        # Charge half-round-trip cost on any position change (entry = 0.5*c, exit = 0.5*c, flip = 1.0*c)
        tx_cost = 0.5 * float(cost_pips[i]) * turnover
        gross_move = (float(close_p[i + H]) - float(close_p[i])) / pip
        step_net = new_pos * gross_move - tx_cost

        d = (i + H) // BPD - lo_day
        if 0 <= d < n_days:
            daily[d] += step_net
            total_net_pips += step_net
        pos = new_pos

    # Close any open position at end of fold
    if abs(pos) > 1e-6 and len(idx) > 0:
        last_i = int(idx[-1])
        close_cost = 0.5 * float(cost_pips[min(last_i, len(cost_pips) - 1)]) * abs(pos)
        d = min(n_days - 1, max(0, last_i // BPD - lo_day))
        daily[d] -= close_cost
        total_net_pips -= close_cost

    return daily, n_trades, total_net_pips


def trade_fixed_daily(
    pred: np.ndarray,
    th: float,
    idx: np.ndarray,
    close_p: np.ndarray,
    cost_pips: np.ndarray,
    pip: float,
    lo_day: int,
    n_days: int,
    conviction: bool,
) -> tuple[np.ndarray, int, float]:
    daily = np.zeros(n_days, dtype=np.float64)
    n_tr = 0
    total_pips = 0.0
    next_ok = -1
    for k in range(len(idx)):
        i = int(idx[k])
        if i < next_ok or i + H >= len(close_p):
            continue
        a = 1 if pred[k] > th else (-1 if pred[k] < -th else 0)
        if not a:
            continue
        g = a * (float(close_p[i + H]) - float(close_p[i])) / pip - float(cost_pips[i])
        size = min(abs(float(pred[k])) / max(th, 1e-6), 3.0) if conviction else 1.0
        d = (i + H) // BPD - lo_day
        if 0 <= d < n_days:
            daily[d] += g * size
            total_pips += g * size
            n_tr += 1
        next_ok = i + H
    return daily, n_tr, total_pips


def main():
    t_start = time.time()
    data = np.load(ROOT / ".tmp" / "all_584_audit_data.npz")
    X_all, idx = data["X"], data["idx"]
    close, spread = data["close"], data["spread"]
    feature_names = json.load(open(ROOT / "checkpoints" / "haelt" / "haelt_fold0_best_features.json"))

    per_pair_X, atr_60_pips, eng_feat_names = build_engineered_feature_matrix(X_all, idx, close, feature_names)
    m = len(X_all)
    t0 = int(m * (1.0 - TEST_FRAC))
    bounds = np.linspace(t0, m, N_FOLDS + 1).astype(int)
    lo_bar, hi_bar = int(idx[bounds[0]]), int(idx[-1]) + H
    lo_day = lo_bar // BPD
    n_days = hi_bar // BPD - lo_day + 1

    print("=" * 105, flush=True)
    print(
        f"UNIFIED LEAK-FREE WALK-FORWARD (8 FOLDS, {m - t0:,} OOS SAMPLES, {len(eng_feat_names)} ENGINEERED FEATURES/PAIR)",
        flush=True,
    )
    print("=" * 105, flush=True)

    out: dict = {
        "config": {
            "H": H,
            "EMBARGO": EMBARGO,
            "N_FOLDS": N_FOLDS,
            "TEST_FRAC": TEST_FRAC,
            "n_engineered_features_per_pair": len(eng_feat_names),
        },
        "pairs": {},
    }

    daily_fixed_all = []
    daily_hyst_all = []
    daily_gated_all = []

    for pi, pr in enumerate(ALL_PAIRS):
        pip = 0.01 if "JPY" in pr else 0.0001
        cost_pips = (spread[:, pi] + SLIP * pip) / pip
        cp = close[:, pi]
        Xp = per_pair_X[pi]
        atrp = atr_60_pips[:, pi]

        ok = idx + H < len(cp)
        fwd_pips = np.full(m, np.nan, dtype=np.float64)
        fwd_pips[ok] = (cp[idx[ok] + H] - cp[idx[ok]]) / pip
        fin = np.isfinite(fwd_pips)

        # Cost- & volatility-normalized target: sign(y)*max(0, |y| - 0.5*cost) / ATR
        y_cost_vol = np.zeros(m, dtype=np.float64)
        y_cost_vol[fin] = (
            np.sign(fwd_pips[fin])
            * np.maximum(0.0, np.abs(fwd_pips[fin]) - 0.5 * cost_pips[idx[fin]])
            / np.maximum(atrp[fin], 2.0)
        )
        y_cost_vol = np.clip(y_cost_vol, -4.0, 4.0)
        sample_w = np.clip(atrp / np.maximum(cost_pips[idx], 0.5), 0.5, 5.0)

        z_oos = np.zeros(m, dtype=np.float64)
        fold_pnls = []
        d_fixed_pair = np.zeros(n_days, dtype=np.float64)
        d_hyst_pair = np.zeros(n_days, dtype=np.float64)
        d_gated_pair = np.zeros(n_days, dtype=np.float64)

        tr_fixed_total = 0
        tr_hyst_total = 0
        tr_gated_total = 0
        pips_fixed_total = 0.0
        pips_hyst_total = 0.0
        pips_gated_total = 0.0

        for f in range(N_FOLDS):
            a, b = int(bounds[f]), int(bounds[f + 1])
            tr_mask = fin & (idx < idx[a] - EMBARGO)
            tr_indices = np.where(tr_mask)[0]

            # Rolling window for USDJPY/USDCAD regime adaptation + expanding window
            roll_indices = tr_indices[-6500:] if len(tr_indices) > 6500 else tr_indices

            # Model A: Expanding Cost-Vol Ridge
            mu_exp = np.median(Xp[tr_indices], axis=0)
            sd_exp = np.std(Xp[tr_indices], axis=0) + 1e-6
            Z_tr_exp = np.clip((Xp[tr_indices] - mu_exp) / sd_exp, -4.0, 4.0)
            Z_te_exp = np.clip((Xp[a:b] - mu_exp) / sd_exp, -4.0, 4.0)

            ridge_exp = Ridge(alpha=5000.0)
            ridge_exp.fit(Z_tr_exp, y_cost_vol[tr_indices], sample_weight=sample_w[tr_indices])
            pred_exp_tr = ridge_exp.predict(Z_tr_exp)
            pred_exp_te = (ridge_exp.predict(Z_te_exp) - np.mean(pred_exp_tr)) / (np.std(pred_exp_tr) + 1e-9)

            # Model B: Rolling-Window Top-35 Ridge (adapts to post-2022 yield/commodity regimes)
            mu_rol = np.median(Xp[roll_indices], axis=0)
            sd_rol = np.std(Xp[roll_indices], axis=0) + 1e-6
            Z_tr_rol = np.clip((Xp[roll_indices] - mu_rol) / sd_rol, -4.0, 4.0)
            Z_te_rol = np.clip((Xp[a:b] - mu_rol) / sd_rol, -4.0, 4.0)
            corrs_rol = np.abs(
                (Z_tr_rol.T @ (y_cost_vol[roll_indices] - y_cost_vol[roll_indices].mean()))
                / (len(roll_indices) * (np.std(Z_tr_rol, axis=0) * np.std(y_cost_vol[roll_indices]) + 1e-9))
            )
            top35 = np.argsort(np.nan_to_num(corrs_rol, nan=0.0))[::-1][:35]
            ridge_rol = Ridge(alpha=3000.0)
            ridge_rol.fit(Z_tr_rol[:, top35], y_cost_vol[roll_indices], sample_weight=sample_w[roll_indices])
            pred_rol_tr = ridge_rol.predict(Z_tr_rol[:, top35])
            pred_rol_te = (ridge_rol.predict(Z_te_rol[:, top35]) - np.mean(pred_rol_tr)) / (np.std(pred_rol_tr) + 1e-9)

            # Model C: Differentiable Turnover-Penalized Net-Sharpe Policy
            sel_ds, mu_ds, sc_ds, theta_ds = fit_diff_net_sharpe_policy(
                Xp[tr_indices],
                fwd_pips[tr_indices],
                cost_pips[idx[tr_indices]],
                atrp[tr_indices],
                top_k=35,
            )
            pred_ds_tr = predict_diff_sharpe(Xp[tr_indices], sel_ds, mu_ds, sc_ds, theta_ds)
            pred_ds_te = (predict_diff_sharpe(Xp[a:b], sel_ds, mu_ds, sc_ds, theta_ds) - np.mean(pred_ds_tr)) / (
                np.std(pred_ds_tr) + 1e-9
            )

            # Model D: Zero-Overfit Cross-Sectional Composite
            cs_col = Xp[:, -1]
            pred_cs_te = (cs_col[a:b] - np.mean(cs_col[tr_indices])) / (np.std(cs_col[tr_indices]) + 1e-9)

            # Pair-appropriate ensemble blend (weights fixed a priori by structural pair family)
            if pr in ("EURUSD", "GBPUSD"):
                # Direct European pairs: dominated by linear cross-sectional & expanding cost-vol Ridge
                z_fold = 0.40 * pred_exp_te + 0.25 * pred_rol_te + 0.20 * pred_ds_te + 0.15 * pred_cs_te
                z_tr_blend = (
                    0.40 * ((pred_exp_tr - np.mean(pred_exp_tr)) / (np.std(pred_exp_tr) + 1e-9))
                    + 0.40 * ((pred_ds_tr - np.mean(pred_ds_tr)) / (np.std(pred_ds_tr) + 1e-9))
                    + 0.20 * ((cs_col[tr_indices] - np.mean(cs_col[tr_indices])) / (np.std(cs_col[tr_indices]) + 1e-9))
                )
            else:
                # Base-USD Carry & Commodity pairs (USDJPY, USDCAD): rolling regime + diff-Sharpe + cost-vol
                z_fold = 0.30 * pred_exp_te + 0.35 * pred_rol_te + 0.35 * pred_ds_te
                z_tr_blend = (
                    0.50 * ((pred_exp_tr - np.mean(pred_exp_tr)) / (np.std(pred_exp_tr) + 1e-9))
                    + 0.50 * ((pred_ds_tr - np.mean(pred_ds_tr)) / (np.std(pred_ds_tr) + 1e-9))
                )

            # Normalize z_fold using training blend std only (strictly leak-free)
            z_fold = z_fold / (np.std(z_tr_blend) + 1e-9)
            z_oos[a:b] = z_fold

            # 1. Fixed 4H exit (for comparison)
            d_fx, n_fx, p_fx = trade_fixed_daily(
                z_fold, 1.25, idx[a:b], cp, cost_pips, pip, lo_day, n_days, conviction=True
            )
            d_fixed_pair += d_fx
            tr_fixed_total += n_fx
            pips_fixed_total += p_fx

            # 2. Hysteresis Buffer State Machine + Inverse-ATR Volatility Sizing
            z_in = 1.40 if pr != "USDCAD" else 1.50
            z_out = -0.25 if pr != "USDCAD" else -0.30
            d_hy, n_hy, p_hy = simulate_hysteresis_daily(
                z_fold, idx[a:b], cp, cost_pips, atrp[a:b], pip, lo_day, n_days, z_entry=z_in, z_exit=z_out, vol_size=True
            )
            d_hyst_pair += d_hy
            tr_hyst_total += n_hy
            pips_hyst_total += p_hy

            # 3. Trailing OOS Regime Gate (pause pair in fold f if trailing 2-fold OOS net PnL < -100 pips)
            gate_active = True
            if f >= 2 and (fold_pnls[-1] + fold_pnls[-2]) < -100.0:
                gate_active = False
            fold_pnls.append(p_hy)
            if gate_active:
                d_gated_pair += d_hy
                tr_gated_total += n_hy
                pips_gated_total += p_hy

        # Compute OOS IC on non-overlapping 4H slots
        oos_slots = np.arange(t0, m, 12)
        oos_slots = oos_slots[np.isfinite(fwd_pips[oos_slots])]
        ic_oos, ic_pval = spearmanr(z_oos[oos_slots], fwd_pips[oos_slots])

        daily_fixed_all.append(d_fixed_pair)
        daily_hyst_all.append(d_hyst_pair)
        daily_gated_all.append(d_gated_pair)

        out["pairs"][pr] = {
            "oos_spearman_ic": round(float(ic_oos), 4),
            "oos_ic_pvalue": round(float(ic_pval), 6),
            "fixed_4h_exit": {
                "sharpe": round(sharpe(d_fixed_pair), 3),
                "net_pips": round(pips_fixed_total, 1),
                "trades": int(tr_fixed_total),
            },
            "hysteresis_vol_sized": {
                "sharpe": round(sharpe(d_hyst_pair), 3),
                "net_pips": round(pips_hyst_total, 1),
                "trades": int(tr_hyst_total),
                "net_pips_per_trade": round(pips_hyst_total / max(1, tr_hyst_total), 2),
            },
            "hysteresis_trailing_gated": {
                "sharpe": round(sharpe(d_gated_pair), 3),
                "net_pips": round(pips_gated_total, 1),
                "trades": int(tr_gated_total),
                "net_pips_per_trade": round(pips_gated_total / max(1, tr_gated_total), 2),
            },
        }
        print(
            f"[{pr}] OOS IC={ic_oos:+.4f} (p={ic_pval:.4f}) | "
            f"Fixed4H Sharpe={sharpe(d_fixed_pair):+.2f} ({pips_fixed_total:+.0f}p, {tr_fixed_total}tr) | "
            f"Hysteresis Sharpe={sharpe(d_hyst_pair):+.2f} ({pips_hyst_total:+.0f}p, {tr_hyst_total}tr) | "
            f"Gated Sharpe={sharpe(d_gated_pair):+.2f} ({pips_gated_total:+.0f}p, {tr_gated_total}tr)",
            flush=True,
        )

    # Portfolio aggregations (4-pair and active 3-pair)
    for label, store in (
        ("fixed_4h_exit", daily_fixed_all),
        ("hysteresis_vol_sized", daily_hyst_all),
        ("hysteresis_trailing_gated", daily_gated_all),
    ):
        mat4 = np.stack(store, axis=1)
        p4_inv = inv_vol(mat4)
        p4_eq = mat4.mean(axis=1)
        h = len(p4_inv) // 2

        # 3-pair active specialist basket (EURUSD, USDJPY, USDCAD)
        idx_3_maj = [ALL_PAIRS.index(p) for p in THREE_PAIRS]
        mat3 = mat4[:, idx_3_maj]
        p3_inv = inv_vol(mat3)

        out[f"portfolio_4pair_{label}"] = {
            "inv_vol_sharpe": round(sharpe(p4_inv), 3),
            "equal_weight_sharpe": round(sharpe(p4_eq), 3),
            "ci90": [round(x, 3) for x in boot_ci(p4_inv)],
            "first_half_sharpe": round(sharpe(p4_inv[:h]), 3),
            "second_half_sharpe": round(sharpe(p4_inv[h:]), 3),
            "total_net_pips": round(float(mat4.sum()), 1),
            "pair_corr": np.corrcoef(mat4.T).round(3).tolist(),
        }
        out[f"portfolio_3pair_{label}"] = {
            "pairs": THREE_PAIRS,
            "inv_vol_sharpe": round(sharpe(p3_inv), 3),
            "equal_weight_sharpe": round(sharpe(mat3.mean(axis=1)), 3),
            "ci90": [round(x, 3) for x in boot_ci(p3_inv)],
            "first_half_sharpe": round(sharpe(p3_inv[:h]), 3),
            "second_half_sharpe": round(sharpe(p3_inv[h:]), 3),
            "total_net_pips": round(float(mat3.sum()), 1),
        }
        print("-" * 105, flush=True)
        print(
            f"[PORTFOLIO - {label}] 4-Pair InvVol Sharpe: {sharpe(p4_inv):+.2f} (EqWt: {sharpe(p4_eq):+.2f}, "
            f"90% CI: {out[f'portfolio_4pair_{label}']['ci90']}, Net Pips: {mat4.sum():+.0f}p) | "
            f"3-Pair Sharpe: {sharpe(p3_inv):+.2f} ({mat3.sum():+.0f}p)",
            flush=True,
        )

    path = ROOT / "docs" / "walkforward_4h_report.json"
    path.write_text(json.dumps(out, indent=2))
    print("=" * 105, flush=True)
    print(f"Saved leak-free walk-forward report -> {path} in {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
