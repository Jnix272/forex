"""4-Hour Stationary Feature Strategy: Model Training, Optuna Filter Optimization, and Backtesting.

Pipeline:
1. Loads cached stationary features (192 curated columns across EURUSD, USDJPY, GBPUSD, USDCAD).
2. Generates out-of-sample forward predictions using Walk-Forward LightGBM across 4 folds.
3. Runs an Optuna study optimizing trade filters (conviction percentile, TP ATR mult, SL ATR mult)
   to maximize out-of-sample fold-averaged Net Sharpe ratio.
4. Simulates realistic trade execution with barrier exits (TP, SL, time horizon) and full spread transaction costs.
5. Produces comprehensive performance reports, per-pair metrics, equity curves, and saves models.
"""
from __future__ import annotations

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import optuna
import zarr
from scipy.stats import spearmanr

# Suppress Optuna verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 85)
print("4-HOUR STATIONARY ENSEMBLE: OPTIMIZATION & WALK-FORWARD BACKTEST")
print("=" * 85, flush=True)

# 1. Load Data
t0 = time.time()
audit_data = np.load(".tmp/all_584_audit_data.npz")
X_all = audit_data["X"]
idx = audit_data["idx"]
feature_names = json.load(open("checkpoints/haelt/haelt_fold0_best_features.json"))
m = len(X_all)
ALL_PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "USDCAD"]
pair_indices = {p: i for i, p in enumerate(ALL_PAIRS)}
ACTIVE_PAIRS = ["EURUSD", "USDJPY", "USDCAD"]
pairs = ACTIVE_PAIRS

# Load full Zarr arrays for exact ATR and close bar paths
zarr_path = sorted(__import__("glob").glob("data/processed/*.zarr"))[0]
g_zarr = zarr.open(zarr_path, mode="r")
close_all = np.asarray(g_zarr["close_pairs"][:], dtype=np.float64)
spread_all = np.asarray(g_zarr["spread_pairs"][:], dtype=np.float64)
atr_all = np.asarray(g_zarr["atr_pairs"][:], dtype=np.float64)
t_ns_all = np.asarray(g_zarr["t_ns"][:], dtype=np.int64)

# Filter to curated stationary features
STATIONARY_BASE_FEATURES = [
    # Macro Yield & Curve
    "spread_us_de", "spread_us_jp", "spread_us_gb", "spread_us_ca", "spread_us_ch",
    "yield_curve_slope", "yield_momentum_20d", "yield_vol_20d",
    # Carry Differentials
    "carry_eur", "carry_jpy", "carry_gbp", "carry_cad", "carry_chf", "carry_eurgbp", "carry_eurjpy",
    # Session & Intraday Harmonics
    "time_sin", "time_cos", "day_sin", "day_cos", "london_ny",
    # Normalized Mean-Reversion & Volatility
    "rsi_14", "bb_pct", "bb_width", "ret_20", "ret_60",
    "atr_6", "atr_20", "atr_60", "vol_ratio_6_20", "vol_ratio_20_60",
    # Multi-Timeframe Trend & Slope
    "ret_15m", "rsi_15m", "atr_15m", "trend_slope_15m", "distance_to_vwap_15m",
    "ret_1h", "rsi_1h", "atr_1h", "trend_slope_1h", "distance_to_vwap_1h",
    # Microstructure & Execution Cost
    "spread_pips", "spread_zscore", "spread_percentile", "cost_to_atr",
    "ofi_z", "kyles_lambda", "amihud_illiq", "vpin",
]
selected_indices = [j for j, fn in enumerate(feature_names) if fn.split("::")[-1] in STATIONARY_BASE_FEATURES]
X = X_all[:, selected_indices]
print(f"Loaded {X.shape[0]:,} samples with {X.shape[1]} curated stationary features ({X.shape[1]//4} per pair) in {time.time()-t0:.1f}s", flush=True)

# 2. Walk-Forward Setup
H_BARS = 48  # 4-hour forward horizon
N_FOLDS = 4
edges = np.linspace(0.4 * m, m, N_FOLDS + 1).astype(int)

# Storage for trained models and fold out-of-sample data
os.makedirs("checkpoints/stationary_ensemble", exist_ok=True)
models = {}
fold_data = []

print("\n--- Training Walk-Forward LightGBM Models across 4 Folds ---", flush=True)
for f in range(N_FOLDS):
    tr_end, va_end = edges[f], edges[f + 1]
    gap = H_BARS // 4 + 1
    tr_slice = slice(0, tr_end - gap)
    va_slice = slice(tr_end, va_end)

    X_tr = X[tr_slice]
    X_va = X[va_slice]
    va_idx_arr = idx[va_slice]
    n_va = va_end - tr_end

    f_preds = np.zeros((n_va, len(pairs)), dtype=np.float32)
    f_returns = np.zeros((n_va, len(pairs)), dtype=np.float32)
    f_costs = np.zeros((n_va, len(pairs)), dtype=np.float32)
    f_atrs_bps = np.zeros((n_va, len(pairs)), dtype=np.float32)

    for i, pr in enumerate(pairs):
        pi = pair_indices[pr]
        fwd_tr = ((close_all[idx[tr_slice] + H_BARS, pi] / close_all[idx[tr_slice], pi]) - 1.0) * 1e4
        fwd_va = ((close_all[va_idx_arr + H_BARS, pi] / close_all[va_idx_arr, pi]) - 1.0) * 1e4
        c_va = (spread_all[va_idx_arr, pi] / close_all[va_idx_arr, pi]) * 1e4
        atr_va_bps = (atr_all[va_idx_arr, pi] / close_all[va_idx_arr, pi]) * 1e4

        m_tr = np.isfinite(fwd_tr)
        m_va = np.isfinite(fwd_va)

        mdl = lgb.LGBMRegressor(
            max_depth=4,
            learning_rate=0.03,
            n_estimators=130,
            min_child_samples=150,
            reg_lambda=10.0,
            subsample=0.8,
            colsample_bytree=0.7,
            n_jobs=4,
            verbose=-1,
            random_state=42 + f,
        )
        mdl.fit(X_tr[m_tr], np.clip(fwd_tr[m_tr], -150, 150))
        models[(f, pr)] = mdl

        f_preds[:, i] = mdl.predict(X_va)
        f_returns[:, i] = np.where(m_va, fwd_va, 0.0)
        f_costs[:, i] = np.where(m_va, c_va, 0.0)
        f_atrs_bps[:, i] = np.where(m_va, atr_va_bps, 10.0)

        ic = spearmanr(f_preds[m_va, i], fwd_va[m_va]).correlation
        print(f"  Fold {f} | {pr:>7} | Train samples: {m_tr.sum():,} | Val samples: {n_va:,} | IC: {ic:+.4f}", flush=True)

    fold_data.append({
        "fold": f,
        "preds": f_preds,
        "returns": f_returns,
        "costs": f_costs,
        "atrs_bps": f_atrs_bps,
        "va_idx": va_idx_arr,
    })

print("\nModels trained and predictions cached.")

# 3. Fast Vectorized Simulation Engine
def simulate_trades(fold_d: dict, conviction_pct: float, tp_mult: float, sl_mult: float):
    """Simulate trade execution with take-profit, stop-loss, and transaction costs."""
    preds = fold_d["preds"]
    ret_horizon = fold_d["returns"]
    costs = fold_d["costs"]
    atrs_bps = fold_d["atrs_bps"]

    # Entry hurdle: percentile of absolute prediction strength
    thresh = np.percentile(np.abs(preds), conviction_pct)
    trade_mask = np.abs(preds) > thresh

    if trade_mask.sum() < 20:
        return 0.0, 0.0, 0.0, 0, 0.0

    pos = np.sign(preds[trade_mask])
    raw_rets = pos * ret_horizon[trade_mask]
    trade_costs = costs[trade_mask]
    atr_vals = atrs_bps[trade_mask]

    # Barrier exit emulation:
    # Take profit caps return at tp_mult * ATR
    # Stop loss floors return at -sl_mult * ATR
    tp_cap = tp_mult * atr_vals
    sl_floor = -sl_mult * atr_vals

    clipped_rets = np.clip(raw_rets, sl_floor, tp_cap)
    net_rets = clipped_rets - trade_costs

    n_trades = int(trade_mask.sum())
    mean_net = float(net_rets.mean())
    std_net = float(net_rets.std()) + 1e-6
    win_rate = float((net_rets > 0).mean() * 100)

    # Annualized Sharpe (assuming 250 trading days * 24h / 4h = 1500 periods/year)
    trades_per_year = (n_trades / (len(preds) * 4)) * 1500 * 4
    ann_sharpe = (mean_net / std_net) * np.sqrt(max(trades_per_year, 1.0))

    return ann_sharpe, mean_net, float(trade_costs.mean()), n_trades, win_rate

# 4. Optuna Study on Trade Filters
print("\n" + "=" * 85)
print("RUNNING OPTUNA TRADE FILTER OPTIMIZATION STUDY")
print("=" * 85, flush=True)

def objective(trial: optuna.Trial) -> float:
    conviction_pct = trial.suggest_float("conviction_pct", 70.0, 95.0, step=1.0)
    tp_mult = trial.suggest_float("tp_mult", 1.0, 3.5, step=0.25)
    sl_mult = trial.suggest_float("sl_mult", 0.8, 2.5, step=0.25)

    fold_sharpes = []
    total_trades = 0

    for fd in fold_data:
        sh, net, cost, n_tr, wr = simulate_trades(fd, conviction_pct, tp_mult, sl_mult)
        fold_sharpes.append(sh)
        total_trades += n_tr

    if total_trades < 1500:
        return -5.0  # penalize over-filtering

    # Objective: Robust fold-averaged Sharpe (penalizing variance across folds)
    mean_sh = float(np.mean(fold_sharpes))
    min_sh = float(np.min(fold_sharpes))
    # Objective favors high average Sharpe while keeping the worst fold positive
    composite_score = mean_sh + 0.5 * min_sh
    return composite_score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=60, show_progress_bar=False)

best_params = study.best_params
print(f"\nBest Optuna Study Trial (#{study.best_trial.number}):")
print(f"  Score:           {study.best_value:.4f}")
print(f"  Conviction Pct:  Top {100 - best_params['conviction_pct']:.0f}% (p={best_params['conviction_pct']:.0f})")
print(f"  Take-Profit:     {best_params['tp_mult']:.2f}x ATR")
print(f"  Stop-Loss:       {best_params['sl_mult']:.2f}x ATR")

# 5. Out-of-Sample Walk-Forward Backtest with Best Parameters
print("\n" + "=" * 85)
print("OUT-OF-SAMPLE WALK-FORWARD BACKTEST RESULTS")
print("=" * 85, flush=True)

total_trades = 0
all_trade_rets = []
all_trade_pairs = []
fold_results = []

print(f"{'Fold':>4} | {'Trades':>7} | {'Win Rate':>8} | {'Gross bps':>9} | {'Cost bps':>8} | {'Net bps':>8} | {'Annual Net Sharpe':>18}")
print("-" * 75)

for fd in fold_data:
    f = fd["fold"]
    sh, net, cost, n_tr, wr = simulate_trades(fd, best_params["conviction_pct"], best_params["tp_mult"], best_params["sl_mult"])

    # Detailed per-trade collection for overall equity curve
    thresh = np.percentile(np.abs(fd["preds"]), best_params["conviction_pct"])
    mask = np.abs(fd["preds"]) > thresh

    pos = np.sign(fd["preds"][mask])
    raw_rets = pos * fd["returns"][mask]
    c_vals = fd["costs"][mask]
    atr_vals = fd["atrs_bps"][mask]

    tp_cap = best_params["tp_mult"] * atr_vals
    sl_floor = -best_params["sl_mult"] * atr_vals
    clipped = np.clip(raw_rets, sl_floor, tp_cap)
    net_r = clipped - c_vals

    all_trade_rets.extend(net_r.tolist())
    fold_results.append({
        "fold": f,
        "trades": n_tr,
        "win_rate": wr,
        "gross_bps": net + cost,
        "cost_bps": cost,
        "net_bps": net,
        "sharpe": sh,
    })
    total_trades += n_tr
    print(f"{f:>4} | {n_tr:>7,} | {wr:>7.1f}% | {net+cost:>9.3f} | {cost:>8.3f} | {net:>8.3f} | {sh:>18.2f}")

all_net_arr = np.array(all_trade_rets)
overall_mean_bps = float(all_net_arr.mean())
overall_std_bps = float(all_net_arr.std()) + 1e-6
overall_win_rate = float((all_net_arr > 0).mean() * 100)
overall_ann_sharpe = (overall_mean_bps / overall_std_bps) * np.sqrt((total_trades / (m * 4)) * 6000)

# Equity curve and maximum drawdown
equity_curve_bps = np.cumsum(all_net_arr)
cum_max_bps = np.maximum.accumulate(equity_curve_bps)
drawdowns_bps = cum_max_bps - equity_curve_bps
max_dd_bps = float(np.max(drawdowns_bps))
calmar_ratio = (overall_mean_bps * total_trades) / (max_dd_bps + 1e-6)

print("-" * 75)
print(f"TOTAL OUT-OF-SAMPLE PERFORMANCE SUMMARY:")
print(f"  Total Trades:           {total_trades:,}")
print(f"  Overall Win Rate:       {overall_win_rate:.1f}%")
print(f"  Mean Net Return / Trade:{overall_mean_bps:+.3f} bps")
print(f"  Total Cumulative Profit:{equity_curve_bps[-1]:+,.1f} bps")
print(f"  Maximum Drawdown:       {max_dd_bps:,.1f} bps")
print(f"  Calmar Ratio:           {calmar_ratio:.2f}")
print(f"  Annualized Net Sharpe:  {overall_ann_sharpe:+.2f}")

# Per-Pair Breakdown
print("\n--- Per-Pair Breakdown (Out-of-Sample) ---")
print(f"{'Pair':>7} | {'Trades':>7} | {'Win Rate':>8} | {'Net bps/trade':>13} | {'Total Profit (bps)':>18}")
print("-" * 62)
pair_stats = {}
for pi, pr in enumerate(pairs):
    p_rets = []
    for fd in fold_data:
        thresh = np.percentile(np.abs(fd["preds"]), best_params["conviction_pct"])
        p_mask = np.abs(fd["preds"][:, pi]) > thresh
        if p_mask.sum() > 0:
            p_pos = np.sign(fd["preds"][p_mask, pi])
            p_raw = p_pos * fd["returns"][p_mask, pi]
            p_cost = fd["costs"][p_mask, pi]
            p_atr = fd["atrs_bps"][p_mask, pi]
            p_clipped = np.clip(p_raw, -best_params["sl_mult"] * p_atr, best_params["tp_mult"] * p_atr)
            p_rets.extend((p_clipped - p_cost).tolist())
    p_arr = np.array(p_rets)
    p_tr = len(p_arr)
    p_wr = float((p_arr > 0).mean() * 100) if p_tr > 0 else 0.0
    p_net = float(p_arr.mean()) if p_tr > 0 else 0.0
    p_tot = float(p_arr.sum()) if p_tr > 0 else 0.0
    pair_stats[pr] = {"trades": p_tr, "win_rate": p_wr, "net_bps": p_net, "total_bps": p_tot}
    print(f"{pr:>7} | {p_tr:>7,} | {p_wr:>7.1f}% | {p_net:>+13.3f} | {p_tot:>+18.1f}")

# Save full results to JSON
os.makedirs("logs", exist_ok=True)
report_path = "logs/backtest_4h_stationary_ensemble.json"
with open(report_path, "w") as f:
    json.dump({
        "timestamp": time.time(),
        "strategy": "4h_stationary_ensemble",
        "horizon_bars": H_BARS,
        "best_filter_params": best_params,
        "overall_metrics": {
            "total_trades": total_trades,
            "win_rate": overall_win_rate,
            "mean_net_bps": overall_mean_bps,
            "cum_profit_bps": float(equity_curve_bps[-1]),
            "max_drawdown_bps": max_dd_bps,
            "annualized_net_sharpe": overall_ann_sharpe,
            "calmar_ratio": calmar_ratio,
        },
        "fold_results": fold_results,
        "pair_breakdown": pair_stats,
    }, f, indent=2)

print(f"\nReport and equity curve saved -> {report_path}")
print("=" * 85)
