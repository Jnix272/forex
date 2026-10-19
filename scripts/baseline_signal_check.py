"""CPU baseline: do the triple-barrier labels carry any predictable signal?

Trains simple models on the last-bar features of each window and scores them
walk-forward (expanding train, embargoed, non-overlapping test trades) with
spread + 1 pip cost. Run at low priority; does not touch the GPU.

    python scripts/baseline_signal_check.py [--stride 6] [--folds 4]
"""

from __future__ import annotations

import argparse
import glob
import math
import time

import numpy as np
import zarr

PAIRS = ["EURUSD", "USDJPY", "USDCAD"]
H = 30  # lookahead bars (matches label horizon)


def pip(name: str) -> float:
    return 0.01 if "JPY" in name else 0.0001


def auc(y: np.ndarray, s: np.ndarray) -> float:
    from scipy.stats import rankdata

    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=6, help="train row stride")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--fraction", type=float, default=0.2, help="trade most-confident fraction")
    ap.add_argument("--max-rows", type=int, default=0, help="debug: cap sampled rows")
    a = ap.parse_args()

    path = glob.glob("data/processed/*triple_barrier*lh30*.zarr")[0]
    root = zarr.open(path, mode="r")
    n = root["X"].shape[0]
    # Lockbox: never touch the reserved holdout tail (172,144:191,271).
    n_use = 172_144
    t0 = time.time()

    idx = np.arange(0, n_use, a.stride)
    if a.max_rows:
        idx = idx[: a.max_rows]
    print(f"[load] {len(idx)} rows (stride {a.stride}) of {n_use}; reading last-bar features...", flush=True)
    Xz = root["X"]
    feats = np.empty((len(idx), Xz.shape[2]), dtype=np.float32)
    B = 2000
    for s in range(0, len(idx), B):
        sl = idx[s : s + B]
        blk = Xz[int(sl[0]) : int(sl[-1]) + 1, -1, :]
        feats[s : s + len(sl)] = blk[sl - sl[0]]
        if (s // B) % 5 == 0:
            print(f"  {s}/{len(idx)}  {time.time() - t0:.0f}s", flush=True)
    ycls = np.asarray(root["ycls_pairs"][:n_use])[idx]
    close = np.asarray(root["close_pairs"][:])
    spread = np.asarray(root["spread_pairs"][:])
    t_ns = np.asarray(root["t_ns"][:n_use])
    print("[labels] unique ycls:", np.unique(ycls[~np.isnan(ycls)])[:6], flush=True)

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    keep = feats.std(0) > 1e-8
    feats = feats[:, keep]
    print(f"[feat] {feats.shape[1]} non-constant features", flush=True)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Direction: BUY vs SELL among tradable rows. Map labels to {0:SELL,1:BUY}.
    # Convention in this project: ycls in {-1,0,1}-like or {0,1,2}; HOLD is the middle value.
    vals = np.unique(ycls[~np.isnan(ycls)])
    mid = np.median(vals)
    emb = max(1, H // a.stride) + 1
    edges = np.linspace(0.4, 1.0, a.folds + 1)  # expanding: first 40% train min
    results = {}
    for k, pname in enumerate(PAIRS):
        y_raw = ycls[:, k]
        tradable = ~np.isnan(y_raw) & (y_raw != mid)
        y = (y_raw > mid).astype(int)
        for mname in ("logreg", "hgb"):
            aucs, nets, grosses, ntr = [], [], [], 0
            for f in range(a.folds):
                lo, hi = int(edges[f] * len(idx)), int(edges[f + 1] * len(idx))
                tr = np.where(tradable[: lo - emb])[0]
                te = np.arange(lo, hi)
                if len(tr) < 500 or len(te) < 50:
                    continue
                sc = StandardScaler().fit(feats[tr])
                Xtr, Xte = sc.transform(feats[tr]), sc.transform(feats[te])
                Xtr, Xte = np.clip(Xtr, -8, 8), np.clip(Xte, -8, 8)
                if mname == "logreg":
                    mdl = LogisticRegression(C=0.01, max_iter=200)
                else:
                    mdl = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0)
                mdl.fit(Xtr, y[tr])
                p = mdl.predict_proba(Xte)[:, 1]
                tt = tradable[te]
                aucs.append(auc(y[te][tt], p[tt]))
                # Honest-ish PnL: non-overlapping trades (every H bars), top-confidence fraction.
                gi = idx[te]
                order = np.arange(0, len(te))
                pick = [0]
                for j in order[1:]:
                    if gi[j] - gi[pick[-1]] >= H:
                        pick.append(j)
                pick = np.array(pick)
                conf = np.abs(p[pick] - 0.5)
                thr = np.quantile(conf, 1 - a.fraction)
                sel = pick[conf >= thr]
                gi_s = gi[sel]
                ok = gi_s + H < len(close)
                sel, gi_s = sel[ok], gi_s[ok]
                d = np.where(p[sel] > 0.5, 1.0, -1.0)
                entry, fwd = close[gi_s, k], close[gi_s + H, k]
                g = d * (fwd - entry) / entry
                c = (np.abs(spread[gi_s, k]) + 1.0 * pip(pname)) / entry
                nets.append(g - c)
                grosses.append(g)
                ntr += len(sel)
            if not aucs:
                continue
            net = np.concatenate(nets)
            gross = np.concatenate(grosses)
            span_years = (t_ns[-1] - t_ns[int(edges[0] * len(idx))]) / 1e9 / 86400 / 365.25
            ann = math.sqrt(max(len(net), 1) / max(span_years, 1e-9))
            sh = lambda r: float(r.mean() / r.std(ddof=1) * ann) if len(r) > 2 and r.std(ddof=1) > 0 else 0.0
            results[(pname, mname)] = (np.nanmean(aucs), sh(gross), sh(net), len(net), net.mean() * 1e4)
            print(
                f"[{pname} {mname:6s}] AUC={np.nanmean(aucs):.3f} (folds {[round(x, 3) for x in aucs]}) "
                f"trades={len(net)} grossSharpe={sh(gross):+.2f} netSharpe={sh(net):+.2f} net={net.mean() * 1e4:+.2f}bps",
                flush=True,
            )
    print("\n=== SUMMARY (AUC>0.52 & netSharpe>0 consistently => signal exists) ===")
    for (pname, mname), (au, gs, ns, nt, nb) in results.items():
        print(f"{pname:7s} {mname:6s} AUC={au:.3f} gross={gs:+.2f} net={ns:+.2f} trades={nt} net_bps={nb:+.2f}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
