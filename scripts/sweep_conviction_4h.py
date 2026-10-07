"""Sweep conviction thresholds on the 4h horizon using curated stationary features."""
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr

data = np.load(".tmp/all_584_audit_data.npz")
X_all = data["X"]
idx = data["idx"]
close = data["close"]
spread = data["spread"]
import json
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

h_bars = 48  # 4h
N_FOLDS = 4
edges = np.linspace(0.4 * m, m, N_FOLDS + 1).astype(int)

# Fit models and store out-of-sample predictions
preds = []
y_trues = []
costs_all = []

for f in range(N_FOLDS):
    tr_end, va_end = edges[f], edges[f + 1]
    gap = 48 // 4 + 1
    tr_slice = slice(0, tr_end - gap)
    va_slice = slice(tr_end, va_end)
    X_tr = X[tr_slice]
    X_va = X[va_slice]

    f_preds = np.zeros((va_end - tr_end, len(pairs)))
    f_trues = np.zeros((va_end - tr_end, len(pairs)))
    f_costs = np.zeros((va_end - tr_end, len(pairs)))

    for pi, pr in enumerate(pairs):
        fwd_tr = ((close[idx[tr_slice] + h_bars, pi] / close[idx[tr_slice], pi]) - 1.0) * 1e4
        fwd_va = ((close[idx[va_slice] + h_bars, pi] / close[idx[va_slice], pi]) - 1.0) * 1e4
        c_va = (spread[idx[va_slice], pi] / close[idx[va_slice], pi]) * 1e4

        m_tr = np.isfinite(fwd_tr)
        m_va = np.isfinite(fwd_va)

        mdl = lgb.LGBMRegressor(
            max_depth=4, learning_rate=0.03, n_estimators=120, min_child_samples=150,
            reg_lambda=10.0, subsample=0.8, colsample_bytree=0.7, n_jobs=4, verbose=-1, random_state=42 + f
        )
        mdl.fit(X_tr[m_tr], np.clip(fwd_tr[m_tr], -150, 150))
        p = mdl.predict(X_va)
        f_preds[:, pi] = p
        f_trues[:, pi] = fwd_va
        f_costs[:, pi] = c_va

    preds.append(f_preds)
    y_trues.append(f_trues)
    costs_all.append(f_costs)

P = np.vstack(preds)
Y = np.vstack(y_trues)
C = np.vstack(costs_all)

print(f"{'Conviction %':>12} | {'Trades':>7} | {'Win Rate':>8} | {'Gross bps':>9} | {'Cost bps':>8} | {'Net bps':>8} | {'Annual Sharpe':>13}")
print("-" * 80)
for pct in [50, 60, 70, 75, 80, 85, 90, 92, 95]:
    thresh = np.percentile(np.abs(P), pct)
    mask = np.abs(P) > thresh
    pos = np.sign(P[mask])
    tr_ret = pos * Y[mask]
    tr_cost = C[mask]
    net_ret = tr_ret - tr_cost

    win = (net_ret > 0).mean() * 100
    gross = tr_ret.mean()
    cost = tr_cost.mean()
    net = net_ret.mean()
    
    # trades per year (across 4 pairs)
    tpy = (mask.sum() / len(Y)) * (250 * 24 * 12 / h_bars)
    ann_sh = (net / (net_ret.std() + 1e-6)) * np.sqrt(tpy)
    print(f"Top {100-pct:>2}% (>{thresh:>4.2f}) | {mask.sum():>7} | {win:>7.1f}% | {gross:>9.3f} | {cost:>8.3f} | {net:>8.3f} | {ann_sh:>13.2f}")
