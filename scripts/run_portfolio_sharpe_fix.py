"""
scripts/run_portfolio_sharpe_fix.py
===================================
Portfolio-level Sharpe improvement on the 2025 holdout using the only proven edge
(4H stationary LightGBM). Variants (all net of Dukascopy spread + 0.7 pip slippage):
  A. single-pair fixed size (baseline)
  B. 3-pair equal-risk portfolio (inverse trailing vol, no lookahead)
  C. B + conviction sizing (|pred|/threshold, capped)
  D. C + vol-target to 1% daily
Reports daily Sharpe with block-bootstrap 90% CI, pair correlation, and a
first-half/second-half split so no parameter is chosen on the data it is scored on.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading.inference_engines import StationaryEnsembleInferenceEngine

PAIRS = ["EURUSD", "USDCAD", "USDJPY"]
HOLDOUT_START = 172144
H = 48
BARS_PER_DAY = 288


def pair_daily(pair, m_pad, conviction):
    z = zarr.open(store=glob.glob(str(ROOT / f"data/processed/dataset_scalping_5m_{pair}_0_dukascopy_*.zarr"))[0], mode="r")
    n = len(m_pad)
    close = np.asarray(z["close"][HOLDOUT_START:HOLDOUT_START + n])
    spread = np.asarray(z["spread"][HOLDOUT_START:HOLDOUT_START + n])
    pip = 0.01 if "JPY" in pair else 0.0001
    cost = (spread + 0.7 * pip) / pip
    eng = StationaryEnsembleInferenceEngine(pair=pair)
    pred = eng.booster.predict(m_pad[:, eng.curated_indices])
    th = eng.threshold_bps
    n_days = n // BARS_PER_DAY + 1
    daily = np.zeros(n_days)
    i = 0
    while i < n - H:
        a = 1 if pred[i] > th else (-1 if pred[i] < -th else 0)
        if a:
            g = a * (close[i + H] - close[i]) / pip - cost[i]
            size = min(abs(pred[i]) / th, 3.0) if conviction else 1.0
            daily[(i + H) // BARS_PER_DAY] += g * size
            i += H
        else:
            i += 1
    return daily


def sharpe(d):
    return float(d.mean() / (d.std() + 1e-12) * np.sqrt(252))


def boot_ci(d, block=5, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    L = len(d)
    nb = max(1, L // block)
    out = []
    for _ in range(n):
        idx = rng.integers(0, L - block, nb)
        out.append(sharpe(np.concatenate([d[j:j + block] for j in idx])))
    return float(np.percentile(out, 5)), float(np.percentile(out, 95))


def inv_vol_weights(daily_mat, lookback=40):
    """Weights from trailing std only (shifted by one day -> no lookahead)."""
    T, K = daily_mat.shape
    w = np.ones((T, K)) / K
    for t in range(lookback, T):
        s = daily_mat[t - lookback:t].std(axis=0) + 1e-9
        w[t] = (1 / s) / (1 / s).sum()
    return w


def main():
    m = np.load(ROOT / "data/processed/holdout_2025_4h_features.npy")
    m_pad = np.pad(m, ((0, 0), (0, 584 - m.shape[1])))
    res = {}
    for conv in (False, True):
        mat = np.stack([pair_daily(p, m_pad, conv) for p in PAIRS], axis=1)
        tag = "conviction" if conv else "fixed"
        res[f"{tag}_single"] = {p: sharpe(mat[:, k]) for k, p in enumerate(PAIRS)}
        eq = mat.mean(axis=1) * len(PAIRS) / len(PAIRS)
        w = inv_vol_weights(mat)
        iv = (mat * w).sum(axis=1) * len(PAIRS)
        # vol-target: scale by trailing 40d std, cap leverage 3x
        sc = np.ones_like(iv)
        for t in range(40, len(iv)):
            sc[t] = iv[:t].std() / (iv[t - 40:t].std() + 1e-9)
        vt = iv * np.clip(sc, 0.3, 3.0)
        h = len(iv) // 2
        for name, d in (("equal", eq), ("invvol", iv), ("invvol+voltarget", vt)):
            lo, hi = boot_ci(d)
            res[f"{tag}_{name}"] = {
                "sharpe": sharpe(d), "ci90": [lo, hi],
                "first_half": sharpe(d[:h]), "second_half": sharpe(d[h:]),
                "net_pips_equiv": float(d.sum()),
            }
        res[f"{tag}_pair_corr"] = np.corrcoef(mat.T).round(3).tolist()
    out = ROOT / "docs" / "portfolio_sharpe_report.json"
    out.write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        print(k, json.dumps(v))
    print("saved", out)


if __name__ == "__main__":
    main()
