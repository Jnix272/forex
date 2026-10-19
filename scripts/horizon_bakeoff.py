"""Empirical Horizon Bake-Off: Evaluate label predictability across horizons.

Tests lookahead horizons H in {6, 10, 12, 15, 20, 30} bars on 5m Dukascopy data
across EURUSD, USDJPY, USDCAD to find the sweet spot where market microstructure
momentum exceeds transaction costs before diffusion destroys signal.

Usage:
    python scripts/horizon_bakeoff.py [--stride 6] [--folds 4]
"""

from __future__ import annotations

import argparse
import glob
import math
import time

import numpy as np
import zarr
from numba import njit, prange
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PAIRS = ["EURUSD", "USDJPY", "USDCAD"]


def pip(name: str) -> float:
    return 0.01 if "JPY" in name else 0.0001


def auc(y: np.ndarray, s: np.ndarray) -> float:
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@njit(parallel=True, fastmath=True)
def compute_labels_for_horizon(
    close: np.ndarray,
    spread: np.ndarray,
    atr: np.ndarray,
    H: int,
    penalty: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute spread-inclusive CPAR direction label (-1, 0, +1) and continuous return."""
    N = len(close)
    labels = np.zeros(N, dtype=np.int8)
    returns = np.zeros(N, dtype=np.float32)

    for i in prange(N - H):
        sp = spread[i]
        el = close[i] + sp / 2.0  # ask entry for long
        es = close[i] - sp / 2.0  # bid entry for short
        vol = atr[i] if atr[i] > 1e-8 else 1e-4

        end_idx = i + H
        # Exit long at bid, exit short at ask
        xl_end = close[end_idx] - spread[end_idx] / 2.0
        xs_end = close[end_idx] + spread[end_idx] / 2.0

        fwd_ret_l = xl_end - el
        fwd_ret_s = es - xs_end

        # MAE
        min_xl = el
        max_xs = es
        for t in range(1, H + 1):
            cur_xl = close[i + t] - spread[i + t] / 2.0
            cur_xs = close[i + t] + spread[i + t] / 2.0
            if cur_xl < min_xl:
                min_xl = cur_xl
            if cur_xs > max_xs:
                max_xs = cur_xs

        mae_l = max(0.0, el - min_xl)
        mae_s = max(0.0, max_xs - es)

        cpar_l = (fwd_ret_l / vol) - (penalty * mae_l / vol)
        cpar_s = (fwd_ret_s / vol) - (penalty * mae_s / vol)

        if cpar_l > 0.0 and cpar_l >= cpar_s:
            labels[i] = 1
            returns[i] = fwd_ret_l / el
        elif cpar_s > 0.0 and cpar_s > cpar_l:
            labels[i] = -1
            returns[i] = fwd_ret_s / es
        else:
            labels[i] = 0
            returns[i] = 0.0

    return labels, returns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=6, help="train row stride")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--fraction", type=float, default=0.20, help="trade most-confident fraction")
    ap.add_argument("--max-rows", type=int, default=0, help="debug: cap sampled rows")
    a = ap.parse_args()

    matches = glob.glob("data/processed/*triple_barrier*.zarr")
    if not matches:
        raise FileNotFoundError("No triple_barrier zarr found in data/processed")
    path = matches[0]
    print(f"[bakeoff] Loading cache: {path}", flush=True)
    root = zarr.open(path, mode="r")

    n_use = 172_144  # strictly respect the lockbox tail
    t0 = time.time()

    idx = np.arange(0, n_use, a.stride)
    if a.max_rows:
        idx = idx[: a.max_rows]
    print(f"[load] Sampled {len(idx)} rows (stride {a.stride}) of {n_use} non-lockbox rows...", flush=True)

    Xz = root["X"]
    feats = np.empty((len(idx), Xz.shape[2]), dtype=np.float32)
    B = 4000
    for s in range(0, len(idx), B):
        sl = idx[s : s + B]
        blk = Xz[int(sl[0]) : int(sl[-1]) + 1, -1, :]
        feats[s : s + len(sl)] = blk[sl - sl[0]]

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    keep = feats.std(0) > 1e-8
    feats = feats[:, keep]
    print(f"[feat] {feats.shape[1]} non-constant features loaded in {time.time() - t0:.1f}s", flush=True)

    close_all = np.asarray(root["close_pairs"][:])
    spread_all = np.asarray(root["spread_pairs"][:])
    atr_all = np.asarray(root["atr_pairs"][:])
    t_ns = np.asarray(root["t_ns"][:n_use])

    horizons = [6, 10, 12, 15, 20, 30]
    edges = np.linspace(0.4, 1.0, a.folds + 1)
    span_years = (t_ns[-1] - t_ns[int(edges[0] * len(idx))]) / 1e9 / 86400 / 365.25

    print("\n" + "=" * 80)
    print(f"  HORIZON BAKE-OFF: Testing H in {horizons} on {len(idx)} rows ({a.folds} folds)")
    print("=" * 80 + "\n")

    leaderboard = []

    for H in horizons:
        h_t0 = time.time()
        emb = max(1, H // a.stride) + 1
        h_results = {}

        for k, pname in enumerate(PAIRS):
            # Compute spread-inclusive labels for this pair and horizon
            p_close = close_all[:, k].astype(np.float64)
            p_spread = np.abs(spread_all[:, k].astype(np.float64))
            p_atr = atr_all[:, k].astype(np.float64)

            labels_full, _ = compute_labels_for_horizon(p_close, p_spread, p_atr, H=H, penalty=0.5)
            y_sub = labels_full[idx]

            tradable = y_sub != 0
            y = (y_sub == 1).astype(int)

            for mname in ("logreg", "hgb"):
                aucs, nets, grosses, ntr = [], [], [], 0
                for f in range(a.folds):
                    lo, hi = int(edges[f] * len(idx)), int(edges[f + 1] * len(idx))
                    tr = np.where(tradable[: lo - emb])[0]
                    te = np.arange(lo, hi)
                    if len(tr) < 500 or len(te) < 50:
                        continue

                    sc = StandardScaler().fit(feats[tr])
                    Xtr = np.clip(sc.transform(feats[tr]), -8, 8)
                    Xte = np.clip(sc.transform(feats[te]), -8, 8)

                    if mname == "logreg":
                        mdl = LogisticRegression(C=0.01, max_iter=200)
                    else:
                        mdl = HistGradientBoostingClassifier(
                            max_depth=3, learning_rate=0.05, max_iter=100, l2_regularization=1.0
                        )

                    mdl.fit(Xtr, y[tr])
                    p = mdl.predict_proba(Xte)[:, 1]

                    tt = tradable[te]
                    if tt.sum() > 10:
                        aucs.append(auc(y[te][tt], p[tt]))

                    # Non-overlapping trades simulation
                    gi = idx[te]
                    pick = [0]
                    for j in range(1, len(te)):
                        if gi[j] - gi[pick[-1]] >= H:
                            pick.append(j)
                    pick = np.array(pick)
                    conf = np.abs(p[pick] - 0.5)
                    thr = np.quantile(conf, 1.0 - a.fraction)
                    sel = pick[conf >= thr]

                    gi_s = gi[sel]
                    ok = gi_s + H < len(p_close)
                    sel, gi_s = sel[ok], gi_s[ok]

                    d = np.where(p[sel] > 0.5, 1.0, -1.0)
                    entry_p = p_close[gi_s]
                    exit_p = p_close[gi_s + H]

                    # Real trade PnL with spread and 1-pip commission
                    g = d * (exit_p - entry_p) / entry_p
                    cost = (p_spread[gi_s] + 1.0 * pip(pname)) / entry_p
                    nets.append(g - cost)
                    grosses.append(g)
                    ntr += len(sel)

                if aucs and nets:
                    net_cat = np.concatenate(nets)
                    gross_cat = np.concatenate(grosses)
                    ann = math.sqrt(max(len(net_cat), 1) / max(span_years, 1e-9))
                    sh_net = float(net_cat.mean() / net_cat.std(ddof=1) * ann) if len(net_cat) > 2 and net_cat.std(ddof=1) > 0 else 0.0
                    sh_gross = float(gross_cat.mean() / gross_cat.std(ddof=1) * ann) if len(gross_cat) > 2 and gross_cat.std(ddof=1) > 0 else 0.0
                    mean_auc = float(np.nanmean(aucs))
                    win_rate = float((net_cat > 0).mean())
                    net_bps = float(net_cat.mean() * 1e4)

                    h_results[(pname, mname)] = {
                        "auc": mean_auc,
                        "gross_sharpe": sh_gross,
                        "net_sharpe": sh_net,
                        "win_rate": win_rate,
                        "net_bps": net_bps,
                        "trades": len(net_cat),
                    }

        # Average across pairs for this horizon
        avg_auc_hgb = np.mean([h_results[(p, "hgb")]["auc"] for p in PAIRS if (p, "hgb") in h_results])
        avg_auc_lr = np.mean([h_results[(p, "logreg")]["auc"] for p in PAIRS if (p, "logreg") in h_results])
        avg_net_sharpe_hgb = np.mean([h_results[(p, "hgb")]["net_sharpe"] for p in PAIRS if (p, "hgb") in h_results])
        avg_net_sharpe_lr = np.mean([h_results[(p, "logreg")]["net_sharpe"] for p in PAIRS if (p, "logreg") in h_results])
        avg_net_bps_hgb = np.mean([h_results[(p, "hgb")]["net_bps"] for p in PAIRS if (p, "hgb") in h_results])

        leaderboard.append({
            "H": H,
            "minutes": H * 5,
            "auc_hgb": avg_auc_hgb,
            "auc_lr": avg_auc_lr,
            "net_sharpe_hgb": avg_net_sharpe_hgb,
            "net_sharpe_lr": avg_net_sharpe_lr,
            "net_bps_hgb": avg_net_bps_hgb,
            "details": h_results,
            "time_s": time.time() - h_t0,
        })

        print(
            f"Horizon H={H:2d} ({H*5:3d}m): HGB AUC={avg_auc_hgb:.3f} (Net Sharpe={avg_net_sharpe_hgb:+.2f}, Net={avg_net_bps_hgb:+.2f}bps) | "
            f"LogReg AUC={avg_auc_lr:.3f} (Net Sharpe={avg_net_sharpe_lr:+.2f}) in {time.time() - h_t0:.1f}s",
            flush=True,
        )

    print("\n" + "=" * 80)
    print("  LEADERBOARD SUMMARY (Sorted by HGB Net Sharpe)")
    print("=" * 80)
    print(f"{'H':>3} {'Span':>5} {'HGB AUC':>9} {'HGB NetSh':>10} {'HGB NetBps':>11} {'LR AUC':>8} {'LR NetSh':>9}")
    print("-" * 65)

    leaderboard.sort(key=lambda x: x["net_sharpe_hgb"], reverse=True)
    for entry in leaderboard:
        print(
            f"{entry['H']:3d} {entry['minutes']:4d}m "
            f"{entry['auc_hgb']:9.3f} {entry['net_sharpe_hgb']:+10.2f} {entry['net_bps_hgb']:+10.2f}bps "
            f"{entry['auc_lr']:8.3f} {entry['net_sharpe_lr']:+9.2f}"
        )

    best = leaderboard[0]
    print("\n" + "=" * 80)
    print(f"  WINNER: H={best['H']} bars ({best['minutes']} minutes) with HGB Net Sharpe = {best['net_sharpe_hgb']:+.2f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
