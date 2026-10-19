"""
scripts/signal_discovery_ic.py
==============================
Leak-free signal discovery. Before building any model, ask: does ANY feature have
out-of-sample predictive power, at which horizon, net of what cost?

Protocol: samples are split in time into DISCOVERY (first 50%) and CONFIRMATION (last 50%,
after a 1-day embargo). Per pair x horizon x feature we compute Spearman IC in 8 time blocks
of each half. A feature is "selected" if its discovery mean IC has a consistent sign
(>=6/8 blocks) and |t|>2.5; it is "confirmed" only if the confirmation half has the same sign,
|mean IC| >= 0.01 and >=6/8 blocks agree. We also report the chance rate (random-sign baseline
via permuting feature order in time) so a handful of confirmations is not mistaken for edge.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
ALL_PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "USDCAD"]
HORIZONS = {"1h": 12, "4h": 48, "1d": 288}
NB = 8


def ic_blocks(Xr, y, nb=NB):
    """Spearman IC per time block: X ranked within block, y ranked within block."""
    out = np.zeros((nb, Xr.shape[1]))
    edges = np.linspace(0, len(y), nb + 1).astype(int)
    for b in range(nb):
        s = slice(edges[b], edges[b + 1])
        xb = np.apply_along_axis(rankdata, 0, Xr[s])
        yb = rankdata(y[s])
        xb = (xb - xb.mean(0)) / (xb.std(0) + 1e-12)
        yb = (yb - yb.mean()) / (yb.std() + 1e-12)
        out[b] = (xb * yb[:, None]).mean(0)
    return out


def main():
    d = np.load(ROOT / ".tmp" / "all_584_audit_data.npz")
    X, idx, close = d["X"], d["idx"], d["close"]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        names = json.load(open(ROOT / "checkpoints" / "haelt" / "haelt_fold0_best_features.json"))
    except Exception:
        names = [f"f{j}" for j in range(X.shape[1])]
    names = names if len(names) == X.shape[1] else [f"f{j}" for j in range(X.shape[1])]
    # Raw price levels (and cumulative/level-like fields) have a mechanical negative in-window IC
    # with forward change (regression to the unknown window mean) -> not tradeable. Exclude.
    LEVEL = {"open", "high", "low", "close", "bid_close", "ask_close", "bid_open", "ask_open", "bid", "ask", "mid"}
    import re
    PRICE_LIKE = re.compile(r"(^|_)(bb_upper|bb_lower|vwap|ema|sma|pivot|support|resistance|kc_upper|kc_lower|donchian)")
    SLOW = re.compile(r"(cot_|carry_|yield_|spread_us_|spread_de_|eco_|sentiment|vix|gdp|cpi|rate_diff)")
    keep = np.array([(n.split("::")[-1] not in LEVEL) and not PRICE_LIKE.search(n.split("::")[-1])
                     and not SLOW.search(n.split("::")[-1]) for n in names])
    print(f"excluding {int((~keep).sum())} price-level columns; scanning {int(keep.sum())} stationary-ish features", flush=True)
    X = X[:, keep]
    names = [n for n, k in zip(names, keep) if k]
    m = len(X)
    half = m // 2
    gap = 288 // 8 + 1  # ~1 day embargo in samples
    D, C = slice(0, half), slice(half + gap, m)
    report = {}
    rng = np.random.default_rng(0)
    for pi, pr in enumerate(ALL_PAIRS):
        for hn, h in HORIZONS.items():
            ok = idx + h < len(close)
            y = np.full(m, np.nan)
            y[ok] = np.log(close[idx[ok] + h, pi] / close[idx[ok], pi])
            fin = np.isfinite(y)
            Xd, yd = X[D][fin[D]], y[D][fin[D]]
            Xc, yc = X[C][fin[C]], y[C][fin[C]]
            icd = ic_blocks(Xd, yd)
            icc = ic_blocks(Xc, yc)
            md, sd = icd.mean(0), icd.std(0) + 1e-9
            t_d = md / (sd / np.sqrt(NB))
            sign_ok = (np.sign(icd) == np.sign(md)).sum(0) >= 6
            sel = sign_ok & (np.abs(t_d) > 2.5)
            mc = icc.mean(0)
            conf = sel & (np.sign(mc) == np.sign(md)) & (np.abs(mc) >= 0.01) & ((np.sign(icc) == np.sign(md)).sum(0) >= 6)
            # chance baseline: shuffle block order of the confirmation ICs' signs across features
            perm = rng.permutation(len(md))
            conf_chance = sel & (np.sign(mc[perm]) == np.sign(md)) & (np.abs(mc[perm]) >= 0.01) & (
                (np.sign(icc[:, perm]) == np.sign(md)).sum(0) >= 6)
            top = np.argsort(-np.abs(mc * conf))[:5]
            report[f"{pr}_{hn}"] = {
                "n_selected": int(sel.sum()), "n_confirmed": int(conf.sum()), "n_confirmed_chance": int(conf_chance.sum()),
                "best_confirmed": [{"feat": names[j], "disc_ic": float(md[j]), "conf_ic": float(mc[j])} for j in top if conf[j]],
                "max_abs_conf_ic": float(np.abs(mc * conf).max()),
            }
            print(pr, hn, {k: v for k, v in report[f"{pr}_{hn}"].items() if k != "best_confirmed"}, flush=True)
            for e in report[f"{pr}_{hn}"]["best_confirmed"][:3]:
                print("    ", e, flush=True)
    (ROOT / "docs" / "signal_discovery_ic.json").write_text(json.dumps(report, indent=2))
    print("saved docs/signal_discovery_ic.json")


if __name__ == "__main__":
    main()
