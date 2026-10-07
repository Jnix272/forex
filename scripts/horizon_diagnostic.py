"""Multi-horizon LightGBM signal diagnostic with lag features.

Labels are rebuilt from close_pairs at horizons 3/12/48 bars (15m/1h/4h) so the
signal-to-cost ratio can be compared. Features: last step, mean of last 12 steps,
and last-minus-12-steps-ago change, on the top-80 active features. Net bps uses
spread_pairs/close as per-trade cost. Features are cached after the first read.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import zarr
from scipy.stats import spearmanr

CHUNK_STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ROW_STRIDE = 4
N_FOLDS = 4
HORIZONS = [3, 12, 48]
CACHE = ".tmp/horizon_feats.npz"

path = sorted(glob.glob("data/processed/*.zarr"))[0]
g = zarr.open(path, mode="r")
meta = json.load(open("checkpoints/haelt/haelt_fold3_best.pt.metadata.json"))
cols = np.where(np.asarray(meta["feature_ablation"]["mask"], dtype=bool))[0]
_p = g.attrs["pairs"]
pairs = _p.split(",") if isinstance(_p, str) else list(_p)
n = g["X"].shape[0]
close = np.asarray(g["close_pairs"][:], dtype=np.float64)
spread = np.asarray(g["spread_pairs"][:], dtype=np.float64)
print(f"n={n} pairs={pairs} close[0]={close[0]} spread mean={spread.mean(0)}", flush=True)

if os.path.exists(CACHE):
    z = np.load(CACHE)
    X, idx = z["X"], z["idx"]
    print(f"loaded cache {X.shape}", flush=True)
else:
    Xs, idxs = [], []
    for ci, s in enumerate(range(0, n, 256)):
        if ci % CHUNK_STRIDE:
            continue
        e = min(s + 256, n)
        w = np.nan_to_num(g["X"][s:e, -13:, :][::ROW_STRIDE][:, :, cols])
        Xs.append(np.concatenate([w[:, -1], w[:, -12:].mean(1), w[:, -1] - w[:, 0]], axis=1))
        idxs.append(np.arange(s, e)[::ROW_STRIDE])
        if ci % (CHUNK_STRIDE * 100) == 0:
            print(f"  read chunk {ci}/{n // 256}", flush=True)
    X = np.concatenate(Xs).astype(np.float32)
    idx = np.concatenate(idxs)
    os.makedirs(".tmp", exist_ok=True)
    np.savez(CACHE, X=X, idx=idx)

names = [f"{k}_{c}" for k in ("last", "mean12", "chg12") for c in cols]
m = len(X)
edges = np.linspace(0.4 * m, m, N_FOLDS + 1).astype(int)
summary = {h: [] for h in HORIZONS}
imp = np.zeros(X.shape[1])
for h in HORIZONS:
    print(f"\n=== horizon {h} bars ({h * 5} min) ===")
    print(f"{'fold':>4} {'pair':>7} {'IC':>8} {'hit%':>6} {'grossBps':>9} {'costBps':>8} {'netBps':>8}", flush=True)
    for pi, pr in enumerate(pairs):
        ok = idx + h < n
        fwd = np.full(m, np.nan)
        fwd[ok] = (close[idx[ok] + h, pi] / close[idx[ok], pi] - 1) * 1e4
        cost = spread[idx, pi] / close[idx, pi] * 1e4
        for f in range(N_FOLDS):
            tr_end, va_end = edges[f], edges[f + 1]
            gap = max(h, 30) // ROW_STRIDE + 1
            trm = np.isfinite(fwd[: tr_end - gap])
            vam = np.isfinite(fwd[tr_end:va_end])
            mdl = lgb.LGBMRegressor(
                max_depth=4, learning_rate=0.05, n_estimators=150, min_child_samples=200,
                reg_lambda=5.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=4, verbose=-1,
            )
            mdl.fit(X[: tr_end - gap][trm], np.clip(fwd[: tr_end - gap][trm], -200, 200))
            imp += mdl.feature_importances_
            p = mdl.predict(X[tr_end:va_end][vam])
            yv = fwd[tr_end:va_end][vam]
            ic = spearmanr(p, yv).correlation
            hit = float((np.sign(p) == np.sign(yv)).mean() * 100)
            gross = float((np.sign(p) * yv).mean())
            c = float(cost[tr_end:va_end][vam].mean())
            summary[h].append((ic, hit, gross, c))
            print(f"{f:>4} {pr:>7} {ic:>8.4f} {hit:>6.1f} {gross:>9.3f} {c:>8.3f} {gross - c:>8.3f}", flush=True)

print("\n=== SUMMARY (mean over folds x pairs) ===")
print(f"{'horizon':>8} {'IC':>8} {'hit%':>6} {'grossBps':>9} {'costBps':>8} {'netBps':>8}")
for h in HORIZONS:
    a = np.array(summary[h])
    print(f"{h * 5:>6}m {a[:,0].mean():>8.4f} {a[:,1].mean():>6.1f} {a[:,2].mean():>9.3f} {a[:,3].mean():>8.3f} {(a[:,2]-a[:,3]).mean():>8.3f}")
print("\nTop 15 features by split importance:")
for i in np.argsort(imp)[::-1][:15]:
    print(f"  {names[i]}  {imp[i]:.0f}")
