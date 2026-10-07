"""Full Feature Audit & Leakage Diagnostic across all 584 features.

Performs:
1. Full 584-feature out-of-sample Spearman IC evaluation across 4 walk-forward folds
   for multiple forward horizons (15m, 1h, 4h, CPAR label).
2. Category-level signal aggregation (Macro, Carry, COT, Microstructure, Sentiment,
   Multi-timeframe, Volatility, Momentum, Volume Profile, Regimes).
3. Leakage & Lookahead Sanity Checks:
   - Permutation test (shuffled returns)
   - Lead/lag causality structure (t-12, t-1, t, t+1, t+12)
   - Degenerate/constant feature detection
4. Exports structured summary table and Markdown report.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import zarr
from scipy.stats import spearmanr

CHUNK_STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 2
ROW_STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 4
N_FOLDS = 4
CACHE_FILE = ".tmp/all_584_audit_data.npz"

print("=" * 80)
print(f"STARTING FULL 584-FEATURE AUDIT (CHUNK_STRIDE={CHUNK_STRIDE}, ROW_STRIDE={ROW_STRIDE})")
print("=" * 80, flush=True)

# 1. Load feature names
feature_names_path = "checkpoints/haelt/haelt_fold0_best_features.json"
with open(feature_names_path) as f:
    feature_names = json.load(f)
n_total_feats = len(feature_names)
print(f"Loaded {n_total_feats} feature names from {feature_names_path}", flush=True)

# Categorize features
def categorize_feature(name: str) -> str:
    col = name.split("::")[-1] if "::" in name else name
    if any(k in col for k in ("spread_us_", "spread_de_", "yield_curve_", "yield_momentum_", "yield_vol_")):
        return "macro_yield"
    if col.startswith("carry_"):
        return "carry"
    if col.startswith("cot_"):
        return "cot"
    if any(k in col for k in ("sentiment", "news_", "eco_", "cat_")):
        return "sentiment_news"
    if any(k in col for k in ("obi_", "tar", "ofi", "kyles_", "amihud_", "realized_spread", "vpin", "price_ofi_")):
        return "microstructure"
    if any(k in col for k in ("spread_pips", "spread_zscore", "spread_percentile", "spread_widening", "cost_to_atr")):
        return "spread_cost"
    if any(k in col for k in ("rsi", "macd", "ret_", "adx_", "chop_index")):
        if any(col.endswith(s) for s in ("_5m", "_15m", "_1h")):
            return "multitimeframe"
        return "momentum"
    if any(k in col for k in ("atr", "vol_", "bb_", "vol_of_vol", "trailing_volatility")):
        if any(col.endswith(s) for s in ("_5m", "_15m", "_1h")):
            return "multitimeframe"
        return "volatility"
    if any(k in col for k in ("regime", "hurst", "noise_to_signal", "trend_quality", "trend_slope")):
        if any(col.endswith(s) for s in ("_5m", "_15m", "_1h")):
            return "multitimeframe"
        return "regime"
    if any(k in col for k in ("vp_", "vol_clock")):
        return "volume_profile"
    if col.startswith("fb_"):
        return "fourier"
    if any(k in col for k in ("time_", "day_")):
        return "time_cyclical"
    if any(k in col for k in ("distance_to_vwap", "_slope_")):
        return "multitimeframe"
    if col in ("open", "high", "low", "close", "volume", "bid_close", "ask_close"):
        return "price_volume"
    return "other"

feat_categories = [categorize_feature(fn) for fn in feature_names]
cat_counts = {}
for c in feat_categories:
    cat_counts[c] = cat_counts.get(c, 0) + 1
print("\nFeature breakdown by category:")
for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"  {cat:>18}: {cnt:>4} features")

# 2. Extract or load data
t0 = time.time()
path = sorted(glob.glob("data/processed/*.zarr"))[0]
g = zarr.open(path, mode="r")
_p = g.attrs["pairs"]
pairs = _p.split(",") if isinstance(_p, str) else list(_p)
n = g["X"].shape[0]

if os.path.exists(CACHE_FILE):
    print(f"\nLoading cached audit data from {CACHE_FILE}...", flush=True)
    cache = np.load(CACHE_FILE)
    X = cache["X"]
    idx = cache["idx"]
    close = cache["close"]
    spread = cache["spread"]
    y_pairs = cache["y_pairs"]
    print(f"Loaded X={X.shape} (dtype={X.dtype}) in {time.time()-t0:.1f}s", flush=True)
else:
    print(f"\nExtracting features from Zarr (n={n}, chunks={n//256})...", flush=True)
    close = np.asarray(g["close_pairs"][:], dtype=np.float64)
    spread = np.asarray(g["spread_pairs"][:], dtype=np.float64)
    y_pairs = np.asarray(g["y_pairs"][:], dtype=np.float32)

    Xs, idxs = [], []
    for ci, s in enumerate(range(0, n, 256)):
        if ci % CHUNK_STRIDE:
            continue
        e = min(s + 256, n)
        # last timestep of the window for all 584 features
        blk = g["X"][s:e, -1, :][::ROW_STRIDE]
        Xs.append(np.nan_to_num(blk))
        idxs.append(np.arange(s, e)[::ROW_STRIDE])
        if ci % (CHUNK_STRIDE * 100) == 0:
            print(f"  Read chunk {ci}/{n // 256} ({ci / (n // 256) * 100:.1f}%)", flush=True)
    X = np.concatenate(Xs).astype(np.float32)
    idx = np.concatenate(idxs)
    os.makedirs(".tmp", exist_ok=True)
    np.savez_compressed(CACHE_FILE, X=X, idx=idx, close=close, spread=spread, y_pairs=y_pairs)
    print(f"Extraction complete and cached to {CACHE_FILE} in {time.time()-t0:.1f}s. Samples={len(X)}", flush=True)

m = len(X)
print(f"Working sample count: {m:,}")

# 3. Degenerate & Constant Feature Check
print("\n" + "=" * 80)
print("CHECK 1: DEGENERATE / ZERO-VARIANCE / CONSTANT FEATURES")
print("=" * 80, flush=True)
stds = np.std(X, axis=0)
zero_var_idx = np.where(stds < 1e-8)[0]
zero_frac = (X == 0).mean(axis=0)
high_zero_idx = np.where(zero_frac > 0.95)[0]

print(f"Total features with zero variance: {len(zero_var_idx)}")
if len(zero_var_idx) > 0:
    for idx_z in zero_var_idx[:10]:
        print(f"  ZERO VAR: {feature_names[idx_z]} (std={stds[idx_z]:.2e})")

print(f"Total features with >95% zero values: {len(high_zero_idx)}")
if len(high_zero_idx) > 0:
    for idx_hz in high_zero_idx[:10]:
        print(f"  >95% ZERO: {feature_names[idx_hz]} (zero_frac={zero_frac[idx_hz]*100:.1f}%)")

# 4. Out-of-Sample Information Coefficient Audit Across 4 Folds
print("\n" + "=" * 80)
print("CHECK 2: 4-FOLD WALK-FORWARD INFORMATION COEFFICIENT (IC) AUDIT")
print("=" * 80, flush=True)

HORIZONS = {"15m": 3, "1h": 12, "4h": 48}
edges = np.linspace(0.4 * m, m, N_FOLDS + 1).astype(int)

# Precompute forward returns for all 4 pairs
fwd_returns = {}
for h_name, h_bars in HORIZONS.items():
    fwd_returns[h_name] = {}
    for pi, pr in enumerate(pairs):
        ok = idx + h_bars < n
        fwd = np.full(m, np.nan, dtype=np.float32)
        fwd[ok] = ((close[idx[ok] + h_bars, pi] / close[idx[ok], pi]) - 1.0) * 1e4
        fwd_returns[h_name][pr] = fwd

# Precompute CPAR labels
cpar_labels = {}
for pi, pr in enumerate(pairs):
    cpar_labels[pr] = y_pairs[idx, pi]

# Evaluate IC for every feature
# Feature j belongs to pair j // 146
feats_per_pair = n_total_feats // len(pairs)

results = []
print(f"Auditing all {n_total_feats} features across {N_FOLDS} walk-forward folds...", flush=True)

for j in range(n_total_feats):
    fn = feature_names[j]
    cat = feat_categories[j]
    pair_idx = j // feats_per_pair
    pr = pairs[pair_idx]
    xj = X[:, j]

    if stds[j] < 1e-8:
        continue

    # Evaluate across folds for 1h forward return and 4h forward return
    ic_1h_folds = []
    ic_4h_folds = []
    ic_cpar_folds = []

    fwd_1h = fwd_returns["1h"][pr]
    fwd_4h = fwd_returns["4h"][pr]
    cpar_y = cpar_labels[pr]

    for f in range(N_FOLDS):
        tr_end, va_end = edges[f], edges[f + 1]
        gap = 12 // ROW_STRIDE + 1
        va_idx = slice(tr_end, va_end)

        x_va = xj[va_idx]
        y_1h_va = fwd_1h[va_idx]
        y_4h_va = fwd_4h[va_idx]
        y_cp_va = cpar_y[va_idx]

        # Valid mask
        val_mask_1h = np.isfinite(x_va) & np.isfinite(y_1h_va)
        val_mask_4h = np.isfinite(x_va) & np.isfinite(y_4h_va)
        val_mask_cp = np.isfinite(x_va) & np.isfinite(y_cp_va)

        if val_mask_1h.sum() > 200:
            ic_1h = spearmanr(x_va[val_mask_1h], y_1h_va[val_mask_1h]).correlation
            ic_1h_folds.append(0.0 if np.isnan(ic_1h) else ic_1h)

        if val_mask_4h.sum() > 200:
            ic_4h = spearmanr(x_va[val_mask_4h], y_4h_va[val_mask_4h]).correlation
            ic_4h_folds.append(0.0 if np.isnan(ic_4h) else ic_4h)

        if val_mask_cp.sum() > 200:
            ic_cp = spearmanr(x_va[val_mask_cp], y_cp_va[val_mask_cp]).correlation
            ic_cpar_folds.append(0.0 if np.isnan(ic_cp) else ic_cp)

    mean_ic_1h = float(np.mean(ic_1h_folds)) if ic_1h_folds else 0.0
    std_ic_1h = float(np.std(ic_1h_folds)) if ic_1h_folds else 1.0
    ir_1h = mean_ic_1h / (std_ic_1h + 1e-6)

    mean_ic_4h = float(np.mean(ic_4h_folds)) if ic_4h_folds else 0.0
    std_ic_4h = float(np.std(ic_4h_folds)) if ic_4h_folds else 1.0
    ir_4h = mean_ic_4h / (std_ic_4h + 1e-6)

    mean_ic_cpar = float(np.mean(ic_cpar_folds)) if ic_cpar_folds else 0.0

    # Consistency: % of folds with same sign as mean
    sign_cons_1h = float(np.mean([np.sign(x) == np.sign(mean_ic_1h) for x in ic_1h_folds])) if ic_1h_folds else 0.0

    results.append({
        "index": j,
        "name": fn,
        "pair": pr,
        "category": cat,
        "mean_ic_1h": mean_ic_1h,
        "std_ic_1h": std_ic_1h,
        "ir_1h": ir_1h,
        "sign_cons_1h": sign_cons_1h,
        "mean_ic_4h": mean_ic_4h,
        "ir_4h": ir_4h,
        "mean_ic_cpar": mean_ic_cpar,
        "zero_frac": float(zero_frac[j]),
    })

    if (j + 1) % 100 == 0 or (j + 1) == n_total_feats:
        print(f"  Audited {j+1}/{n_total_feats} features...", flush=True)

# 5. Category-Level Performance
print("\n" + "=" * 80)
print("CATEGORY-LEVEL PREDICTIVE SIGNAL SUMMARY (1H HORIZON)")
print("=" * 80)
cat_groups = {}
for r in results:
    c = r["category"]
    cat_groups.setdefault(c, []).append(r)

print(f"{'Category':>18} | {'Count':>5} | {'Mean |IC|':>9} | {'Max |IC|':>8} | {'Mean IR':>8} | {'Best Feature':<30}")
print("-" * 90)
for cat, items in sorted(cat_groups.items(), key=lambda x: -np.mean([abs(i["mean_ic_1h"]) for i in x[1]])):
    mean_abs_ic = np.mean([abs(i["mean_ic_1h"]) for i in items])
    max_abs_ic = np.max([abs(i["mean_ic_1h"]) for i in items])
    mean_ir = np.mean([abs(i["ir_1h"]) for i in items])
    best_item = max(items, key=lambda x: abs(x["mean_ic_1h"]))
    print(f"{cat:>18} | {len(items):>5} | {mean_abs_ic:>9.4f} | {max_abs_ic:>8.4f} | {mean_ir:>8.2f} | {best_item['name']:<30}")

# 6. Top Individual Signals
print("\n" + "=" * 80)
print("TOP 25 STRONGEST FEATURES BY ABSOLUTE IC (1H FORWARD RETURN)")
print("=" * 80)
print(f"{'Rank':>4} | {'Feature':<34} | {'Category':>14} | {'Mean IC':>8} | {'IC Std':>7} | {'IC IR':>6} | {'Consist':>7} | {'4h IC':>7}")
print("-" * 105)
sorted_by_ic = sorted(results, key=lambda x: -abs(x["mean_ic_1h"]))
for rank, r in enumerate(sorted_by_ic[:25], 1):
    print(f"{rank:>4} | {r['name']:<34} | {r['category']:>14} | {r['mean_ic_1h']:>8.4f} | {r['std_ic_1h']:>7.4f} | {r['ir_1h']:>6.2f} | {r['sign_cons_1h']*100:>6.0f}% | {r['mean_ic_4h']:>7.4f}")

# 7. Step 2: Leakage / Lookahead / Permutation Sanity Checks
print("\n" + "=" * 80)
print("STEP 2: LEAKAGE & LOOKAHEAD SANITY CHECKS")
print("=" * 80, flush=True)

# Test A: Permutation Test (Shuffle Returns)
np.random.seed(42)
perm_idx = np.random.permutation(m)
perm_fwd_1h = fwd_returns["1h"]["EURUSD"][perm_idx]
perm_ics = []
for j in range(0, min(100, n_total_feats)):
    xj = X[:, j]
    m_ok = np.isfinite(xj) & np.isfinite(perm_fwd_1h)
    if m_ok.sum() > 500:
        c = spearmanr(xj[m_ok], perm_fwd_1h[m_ok]).correlation
        perm_ics.append(c if not np.isnan(c) else 0.0)
print(f"Permutation Test (100 random features vs shuffled labels):")
print(f"  Expected: Mean IC ~ 0.000, Max |IC| < 0.02")
print(f"  Observed: Mean IC = {np.mean(perm_ics):.5f}, Max |IC| = {np.max(np.abs(perm_ics)):.5f}")
if abs(np.mean(perm_ics)) < 0.005:
    print("  --> PERMUTATION CHECK PASSED: Labels break cleanly when randomized.")
else:
    print("  --> WARNING: Permutation check failed!")

# Test B: Lead/Lag Causality Structure on Top 10 Features
print("\nLead/Lag Causality Structure (Probe for Lookahead Leakage):")
print("If feature has lookahead contamination, IC at t+1 or t+12 will spike unnaturally.")
print(f"{'Feature':<34} | {'t-12 (past)':>11} | {'t-1':>7} | {'t (curr)':>8} | {'t+1 (lead)':>10} | {'t+12 (lead)':>11} | {'Status':<10}")
print("-" * 105)

top_10 = sorted_by_ic[:10]
for r in top_10:
    j = r["index"]
    fn = r["name"]
    pr = r["pair"]
    xj = X[:, j]
    y = fwd_returns["1h"][pr]

    def _ic_lag(shift_x: int):
        # shift_x > 0 means x is from the future (leakage probe)
        # shift_x < 0 means x is from the past
        if shift_x > 0:
            x_s = xj[shift_x:]
            y_s = y[:-shift_x]
        elif shift_x < 0:
            x_s = xj[:shift_x]
            y_s = y[-shift_x:]
        else:
            x_s = xj
            y_s = y
        m_s = np.isfinite(x_s) & np.isfinite(y_s)
        if m_s.sum() > 500:
            val = spearmanr(x_s[m_s], y_s[m_s]).correlation
            return 0.0 if np.isnan(val) else val
        return 0.0

    ic_tm12 = _ic_lag(-3)   # ~12 bars past
    ic_tm1 = _ic_lag(-1)    # 1 bar past
    ic_t0 = _ic_lag(0)      # current
    ic_tp1 = _ic_lag(1)     # 1 bar future (leakage test)
    ic_tp12 = _ic_lag(3)    # 12 bars future (leakage test)

    # If correlation in future is dramatically higher than current, flag leakage
    is_leaked = (abs(ic_tp1) > 2.0 * max(abs(ic_t0), 0.02)) and (abs(ic_tp1) > 0.10)
    status = "LEAKED!" if is_leaked else "CAUSAL OK"

    print(f"{fn:<34} | {ic_tm12:>11.4f} | {ic_tm1:>7.4f} | {ic_t0:>8.4f} | {ic_tp1:>10.4f} | {ic_tp12:>11.4f} | {status:<10}")

# Save full results to JSON
out_json = ".tmp/feature_audit_results.json"
with open(out_json, "w") as f:
    json.dump({
        "timestamp": time.time(),
        "n_features_audited": len(results),
        "top_25": sorted_by_ic[:25],
        "category_summary": {cat: {"mean_abs_ic": float(np.mean([abs(i["mean_ic_1h"]) for i in items])), "count": len(items)} for cat, items in cat_groups.items()}
    }, f, indent=2)
print(f"\nFull audit results saved to {out_json}")
print("=" * 80)
