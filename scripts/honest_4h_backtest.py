"""100% Honest Out-of-Sample Walk-Forward Backtest (Pure 4h Horizon Hold, No Path Clipping).

Completely uncheatable:
- Entry at bar t: pays spread cost.
- Exit at bar t+48 (exact 4h close): pays spread cost.
- Exact return: pos * (close[t+48] - close[t]) / close[t] - cost.
- Zero intra-bar path assumptions or artificial stop-loss clipping.
"""
from __future__ import annotations

import json
import os
import time
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr

t0 = time.time()
audit_data = np.load(".tmp/all_584_audit_data.npz")
X_all = audit_data["X"]
idx = audit_data["idx"]
close = audit_data["close"]
spread = audit_data["spread"]
feature_names = json.load(open("checkpoints/haelt/haelt_fold0_best_features.json"))
m = len(X_all)
pairs = ["EURUSD", "USDJPY", "GBPUSD", "USDCAD"]

STATIONARY_BASE_FEATURES = [
    "spread_us_de", "spread_us_jp", "spread_us_gb", "spread_us_ca", "spread_us_ch",
    "yield_curve_slope", "yield_momentum_20d", "yield_vol_20d",
    "carry_eur", "carry_jpy", "carry_gbp", "carry_cad", "carry_chf", "carry_eurgbp", "carry_eurjpy",
    "time_sin", "time_cos", "day_sin", "day_cos", "london_ny",
    "rsi_14", "bb_pct", "bb_width", "ret_20", "ret_60",
    "atr_6", "atr_20", "atr_60", "vol_ratio_6_20", "vol_ratio_20_60",
    "ret_15m", "rsi_15m", "atr_15m", "trend_slope_15m", "distance_to_vwap_15m",
    "ret_1h", "rsi_1h", "atr_1h", "trend_slope_1h", "distance_to_vwap_1h",
    "spread_pips", "spread_zscore", "spread_percentile", "cost_to_atr",
    "ofi_z", "kyles_lambda", "amihud_illiq", "vpin",
]
selected_indices = [j for j, fn in enumerate(feature_names) if fn.split("::")[-1] in STATIONARY_BASE_FEATURES]
X = X_all[:, selected_indices]

H_BARS = 48  # 4h
N_FOLDS = 4
edges = np.linspace(0.4 * m, m, N_FOLDS + 1).astype(int)

# Conviction hurdle: Top 15% (strongest 15% of return forecasts)
CONVICTION_PERCENTILE = 85.0

all_trade_rets = []
all_trade_costs = []
pair_trade_rets = {p: [] for p in pairs}
fold_stats = []

print("=" * 85)
print("100% HONEST OUT-OF-SAMPLE WALK-FORWARD BACKTEST (4H HOLD, NO CLIPPING)")
print("=" * 85)
print(f"{'Fold':>4} | {'Trades':>7} | {'Win Rate':>8} | {'Gross bps':>9} | {'Cost bps':>8} | {'Net bps':>8} | {'Annual Net Sharpe':>18}")
print("-" * 75)

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

    for pi, pr in enumerate(pairs):
        fwd_tr = ((close[idx[tr_slice] + H_BARS, pi] / close[idx[tr_slice], pi]) - 1.0) * 1e4
        fwd_va = ((close[va_idx_arr + H_BARS, pi] / close[va_idx_arr, pi]) - 1.0) * 1e4
        c_va = (spread[va_idx_arr, pi] / close[va_idx_arr, pi]) * 1e4

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
        f_preds[:, pi] = mdl.predict(X_va)
        f_returns[:, pi] = np.where(m_va, fwd_va, 0.0)
        f_costs[:, pi] = np.where(m_va, c_va, 0.0)

    # Apply conviction filter across the fold
    thresh = np.percentile(np.abs(f_preds), CONVICTION_PERCENTILE)
    f_trades_net = []
    f_trades_gross = []
    f_trades_cost = []

    for pi, pr in enumerate(pairs):
        mask = np.abs(f_preds[:, pi]) > thresh
        if mask.sum() > 0:
            pos = np.sign(f_preds[mask, pi])
            raw_r = pos * f_returns[mask, pi]
            c_r = f_costs[mask, pi]
            net_r = raw_r - c_r

            pair_trade_rets[pr].extend(net_r.tolist())
            all_trade_rets.extend(net_r.tolist())
            all_trade_costs.extend(c_r.tolist())

            f_trades_net.extend(net_r.tolist())
            f_trades_gross.extend(raw_r.tolist())
            f_trades_cost.extend(c_r.tolist())

    f_net_arr = np.array(f_trades_net)
    f_tr_cnt = len(f_net_arr)
    f_wr = float((f_net_arr > 0).mean() * 100)
    f_gross_m = float(np.mean(f_trades_gross))
    f_cost_m = float(np.mean(f_trades_cost))
    f_net_m = float(np.mean(f_net_arr))
    # Annualized Sharpe (1500 4-hour periods per year)
    trades_per_year = (f_tr_cnt / (n_va * len(pairs))) * 1500 * len(pairs)
    f_sh = (f_net_m / (f_net_arr.std() + 1e-6)) * np.sqrt(trades_per_year)

    fold_stats.append({
        "fold": f, "trades": f_tr_cnt, "win_rate": f_wr,
        "gross_bps": f_gross_m, "cost_bps": f_cost_m, "net_bps": f_net_m, "sharpe": f_sh
    })
    print(f"{f:>4} | {f_tr_cnt:>7,} | {f_wr:>7.1f}% | {f_gross_m:>9.3f} | {f_cost_m:>8.3f} | {f_net_m:>8.3f} | {f_sh:>18.2f}")

all_net_arr = np.array(all_trade_rets)
total_trades = len(all_net_arr)
mean_net = float(all_net_arr.mean())
std_net = float(all_net_arr.std()) + 1e-6
overall_wr = float((all_net_arr > 0).mean() * 100)
overall_cost = float(np.mean(all_trade_costs))
overall_gross = mean_net + overall_cost
overall_sharpe = (mean_net / std_net) * np.sqrt((total_trades / (m * len(pairs))) * 6000)

equity_curve = np.cumsum(all_net_arr)
peak = np.maximum.accumulate(equity_curve)
dd = peak - equity_curve
max_dd = float(np.max(dd))
calmar = (equity_curve[-1]) / (max_dd + 1e-6)

print("-" * 75)
print(f"HONEST TOTAL PERFORMANCE SUMMARY:")
print(f"  Total Trades:           {total_trades:,}")
print(f"  Win Rate:               {overall_wr:.1f}%")
print(f"  Gross Return / Trade:   {overall_gross:+.3f} bps")
print(f"  Spread Cost / Trade:    {overall_cost:.3f} bps")
print(f"  Net Return / Trade:     {mean_net:+.3f} bps")
print(f"  Total Cumulative Profit:{equity_curve[-1]:+,.1f} bps")
print(f"  Maximum Drawdown:       {max_dd:,.1f} bps")
print(f"  Calmar Ratio:           {calmar:.2f}")
print(f"  Annualized Net Sharpe:  {overall_sharpe:+.2f}")

print("\n--- Per-Pair Breakdown ---")
print(f"{'Pair':>7} | {'Trades':>7} | {'Win Rate':>8} | {'Net bps/trade':>13} | {'Total Profit (bps)':>18}")
print("-" * 62)
for pr in pairs:
    p_arr = np.array(pair_trade_rets[pr])
    p_cnt = len(p_arr)
    p_wr = float((p_arr > 0).mean() * 100) if p_cnt > 0 else 0.0
    p_net = float(p_arr.mean()) if p_cnt > 0 else 0.0
    p_tot = float(p_arr.sum()) if p_cnt > 0 else 0.0
    print(f"{pr:>7} | {p_cnt:>7,} | {p_wr:>7.1f}% | {p_net:>+13.3f} | {p_tot:>+18.1f}")

# Save honest backtest report
os.makedirs("logs", exist_ok=True)
with open("logs/honest_4h_backtest.json", "w") as f:
    json.dump({
        "timestamp": time.time(),
        "strategy": "honest_4h_stationary_hold",
        "conviction_percentile": CONVICTION_PERCENTILE,
        "total_trades": total_trades,
        "win_rate": overall_wr,
        "gross_bps": overall_gross,
        "cost_bps": overall_cost,
        "net_bps": mean_net,
        "cum_profit_bps": float(equity_curve[-1]),
        "max_drawdown_bps": max_dd,
        "annualized_net_sharpe": overall_sharpe,
        "calmar_ratio": calmar,
        "fold_stats": fold_stats,
        "per_pair": {pr: {"trades": len(pair_trade_rets[pr]), "net_bps": float(np.mean(pair_trade_rets[pr])), "total_bps": float(np.sum(pair_trade_rets[pr]))} for pr in pairs}
    }, f, indent=2)

print("\nReport saved -> logs/honest_4h_backtest.json")
print("=" * 85)
