"""Cheap CPU signal diagnostic: does ANY gradient-boosted model find gross edge?

Uses last-timestep features of the cached zarr dataset (top-80 ablation mask),
expanding-window walk-forward folds, per-pair IC / hit-rate / gross mean return.
Reads a strided subset of chunks so it can run next to GPU training.
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
import zarr
from scipy.stats import spearmanr
import lightgbm as lgb

CHUNK_STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 3  # use every Nth chunk
ROW_STRIDE = 4  # thin overlapping bars inside a chunk
N_FOLDS = 4

path = sorted(glob.glob("data/processed/*.zarr"))[0]
g = zarr.open(path, mode="r")
meta = json.load(open("checkpoints/haelt/haelt_fold3_best.pt.metadata.json"))
mask = np.asarray(meta["feature_ablation"]["mask"], dtype=bool)
cols = np.where(mask)[0]
_p = g.attrs["pairs"]
pairs = _p.split(",") if isinstance(_p, str) else list(_p)
n = g["X"].shape[0]
ck = 256
print(f"dataset n={n} pairs={pairs} active_feats={len(cols)}", flush=True)

Xs, Ys, Ss, Ts = [], [], [], []
for ci, s in enumerate(range(0, n, ck)):
    if ci % CHUNK_STRIDE:
        continue
    e = min(s + ck, n)
    blk = g["X"][s:e, -1, :][::ROW_STRIDE][:, cols]
    Xs.append(blk)
    Ys.append(g["y_pairs"][s:e][::ROW_STRIDE])
    Ss.append(g["spread_pairs"][s:e][::ROW_STRIDE])
    Ts.append(g["t_ns"][s:e][::ROW_STRIDE])
    if ci % (CHUNK_STRIDE * 100) == 0:
        print(f"  read chunk {ci}/{n // ck}", flush=True)
X = np.nan_to_num(np.concatenate(Xs)).astype(np.float32)
Y = np.concatenate(Ys)
S = np.concatenate(Ss)
print(f"samples={len(X)} y_pairs mean={Y.mean():.5f} std={Y.std():.5f} spread mean={S.mean():.6f}", flush=True)

m = len(X)
edges = np.linspace(0.4 * m, m, N_FOLDS + 1).astype(int)
print(f"{'fold':>4} {'pair':>7} {'IC':>8} {'hit%':>6} {'grossBps':>9} {'costBps':>8} {'netBps':>8}")
for f in range(N_FOLDS):
    tr_end, va_end = edges[f], edges[f + 1]
    gap = 30  # embargo (label horizon)
    for pi, pr in enumerate(pairs):
        ytr = Y[: tr_end - gap, pi]
        yva = Y[tr_end:va_end, pi]
        mdl = lgb.LGBMRegressor(
            max_depth=4, learning_rate=0.05, n_estimators=150, min_child_samples=200,
            reg_lambda=5.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            n_jobs=4, verbose=-1,
        )
        mdl.fit(X[: tr_end - gap], ytr)
        p = mdl.predict(X[tr_end:va_end])
        ic = spearmanr(p, yva).correlation
        nz = yva != 0
        hit = float((np.sign(p[nz]) == np.sign(yva[nz])).mean() * 100)
        gross = float((np.sign(p) * yva).mean() * 1e4)
        cost = float(S[tr_end:va_end, pi].mean() * 1e4)
        print(f"{f:>4} {pr:>7} {ic:>8.4f} {hit:>6.1f} {gross:>9.3f} {cost:>8.3f} {gross - cost:>8.3f}", flush=True)
