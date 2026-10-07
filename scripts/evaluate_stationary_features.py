"""Evaluate curated stationary feature set with walk-forward LightGBM.

Tests the new curated feature subset (Macro Yield, Carry, Time Harmonics,
Mean-Reversion Oscillators, Higher-Timeframe Volatility/Trend, Spread Cost)
against 1h (12 bars) and 4h (48 bars) forward horizons across 4 walk-forward folds.
"""
from __future__ import annotations

import json
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr

# Load cached audit data and feature names
data = np.load(".tmp/all_584_audit_data.npz")
X_all = data["X"]
idx = data["idx"]
close = data["close"]
spread = data["spread"]
feature_names = json.load(open("checkpoints/haelt/haelt_fold0_best_features.json"))

m = len(X_all)
n_features = len(feature_names)
pairs = ["EURUSD", "USDJPY", "GBPUSD", "USDCAD"]

# Define curated stationary feature patterns (base feature names)
STATIONARY_BASE_FEATURES = [
    # 1. Macro Yield & Curve
    "spread_us_de", "spread_us_jp", "spread_us_gb", "spread_us_ca", "spread_us_ch",
    "yield_curve_slope", "yield_momentum_20d", "yield_vol_20d",
    # 2. Carry Differentials
    "carry_eur", "carry_jpy", "carry_gbp", "carry_cad", "carry_chf", "carry_eurgbp", "carry_eurjpy",
    # 3. Session & Intraday Harmonics
    "time_sin", "time_cos", "day_sin", "day_cos", "london_ny",
    # 4. Normalized Mean-Reversion & Volatility
    "rsi_14", "bb_pct", "bb_width", "ret_20", "ret_60",
    "atr_6", "atr_20", "atr_60", "vol_ratio_6_20", "vol_ratio_20_60",
    # 5. Multi-Timeframe Trend & Slope
    "ret_15m", "rsi_15m", "atr_15m", "trend_slope_15m", "distance_to_vwap_15m",
    "ret_1h", "rsi_1h", "atr_1h", "trend_slope_1h", "distance_to_vwap_1h",
    # 6. Microstructure & Execution Cost
    "spread_pips", "spread_zscore", "spread_percentile", "cost_to_atr",
    "ofi_z", "kyles_lambda", "amihud_illiq", "vpin",
]

# Find indices of curated features in full 584 feature list
selected_indices = []
selected_names = []
for j, fn in enumerate(feature_names):
    base = fn.split("::")[-1]
    if base in STATIONARY_BASE_FEATURES:
        selected_indices.append(j)
        selected_names.append(fn)

selected_indices = np.array(selected_indices)
print(f"Total curated features selected: {len(selected_indices)} / {n_features}")
print(f"Per pair: {len(selected_indices) // len(pairs)} features")

X = X_all[:, selected_indices]

# Walk-forward evaluation across 4 folds
N_FOLDS = 4
HORIZONS = {"1h (12 bars)": 12, "4h (48 bars)": 48}
edges = np.linspace(0.4 * m, m, N_FOLDS + 1).astype(int)

for h_label, h_bars in HORIZONS.items():
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD LIGHTGBM EVALUATION: HORIZON {h_label}")
    print(f"{'='*80}")
    print(f"{'Fold':>4} | {'Pair':>7} | {'IC':>8} | {'Hit %':>6} | {'Gross bps':>9} | {'Cost bps':>8} | {'Net bps':>8} | {'Annualized Net Sharpe':>21}")
    print("-" * 88)

    all_fold_metrics = []

    for f in range(N_FOLDS):
        tr_end, va_end = edges[f], edges[f + 1]
        gap = max(h_bars, 30) // 4 + 1
        
        tr_slice = slice(0, tr_end - gap)
        va_slice = slice(tr_end, va_end)

        X_tr = X[tr_slice]
        X_va = X[va_slice]

        for pi, pr in enumerate(pairs):
            ok_tr = idx[tr_slice] + h_bars < data["close"].shape[0]
            ok_va = idx[va_slice] + h_bars < data["close"].shape[0]

            fwd_tr = ((close[idx[tr_slice] + h_bars, pi] / close[idx[tr_slice], pi]) - 1.0) * 1e4
            fwd_va = ((close[idx[va_slice] + h_bars, pi] / close[idx[va_slice], pi]) - 1.0) * 1e4
            c_va = (spread[idx[va_slice], pi] / close[idx[va_slice], pi]) * 1e4

            m_tr = np.isfinite(fwd_tr)
            m_va = np.isfinite(fwd_va)

            mdl = lgb.LGBMRegressor(
                max_depth=4,
                learning_rate=0.03,
                n_estimators=120,
                min_child_samples=150,
                reg_lambda=10.0,
                subsample=0.8,
                colsample_bytree=0.7,
                n_jobs=4,
                verbose=-1,
                random_state=42 + f,
            )
            mdl.fit(X_tr[m_tr], np.clip(fwd_tr[m_tr], -150, 150))
            pred = mdl.predict(X_va[m_va])

            y_true = fwd_va[m_va]
            costs = c_va[m_va]

            ic = spearmanr(pred, y_true).correlation
            hit = float((np.sign(pred) == np.sign(y_true)).mean() * 100)

            # Trade execution with conviction threshold: top/bottom 30% conviction
            threshold = np.percentile(np.abs(pred), 70)
            trade_mask = np.abs(pred) > threshold

            if trade_mask.sum() > 50:
                pos = np.sign(pred[trade_mask])
                trade_returns = pos * y_true[trade_mask]
                trade_costs = costs[trade_mask]
                net_trade_returns = trade_returns - trade_costs

                gross_bps = float(trade_returns.mean())
                cost_bps = float(trade_costs.mean())
                net_bps = float(net_trade_returns.mean())
                # Annualized Sharpe (assuming 250 days * 24 hours / horizon)
                trades_per_year = (trade_mask.sum() / len(y_true)) * (250 * 24 * 12 / h_bars)
                std_ret = float(net_trade_returns.std()) + 1e-6
                ann_sharpe = (net_bps / std_ret) * np.sqrt(trades_per_year)
            else:
                gross_bps, cost_bps, net_bps, ann_sharpe = 0.0, 0.0, 0.0, 0.0

            all_fold_metrics.append((ic, hit, gross_bps, cost_bps, net_bps, ann_sharpe))
            print(f"{f:>4} | {pr:>7} | {ic:>8.4f} | {hit:>6.1f} | {gross_bps:>9.3f} | {cost_bps:>8.3f} | {net_bps:>8.3f} | {ann_sharpe:>21.2f}")

    metrics_arr = np.array(all_fold_metrics)
    print("-" * 88)
    print(f"MEAN OVER ALL 4 FOLDS & PAIRS ({h_label}):")
    print(f"  Mean IC:          {metrics_arr[:, 0].mean():>8.4f}")
    print(f"  Mean Hit Rate:    {metrics_arr[:, 1].mean():>6.1f}%")
    print(f"  Mean Gross Bps:   {metrics_arr[:, 2].mean():>9.3f} bps")
    print(f"  Mean Cost Bps:    {metrics_arr[:, 3].mean():>8.3f} bps")
    print(f"  Mean Net Bps:     {metrics_arr[:, 4].mean():>8.3f} bps")
    print(f"  Mean Net Sharpe:  {metrics_arr[:, 5].mean():>8.2f}")
