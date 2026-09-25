"""Honest baseline: does a simple model beat "no trade" after costs?

Trains one XGBoost classifier per pair on the last-timestep features of the
dataset cache, with walk-forward folds and an embargo gap, and scores its
directions with the same net-of-spread, non-overlapping per-pair Sharpe and
bootstrap CI used in training (training.honest_eval.pooled_pair_metrics).

If the pooled CI lower bound is not > 0 on most folds, the label has no
tradable signal on these features and deep-model retraining will not find one.

    python scripts/baseline_honest.py                   # newest *_ba0925b cache
    python scripts/baseline_honest.py --cache <zarr> --folds 5 --model logreg
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.dataset_builder import PRICE_LEVEL_FEATURES, VENUE_SCALED_FEATURES  # noqa: E402
from training.honest_eval import fx_bars_per_year, pooled_pair_metrics  # noqa: E402


def _latest_cache() -> str:
    hits = sorted(glob.glob("data/processed/*_ba0925b*.zarr"), key=os.path.getmtime)
    if not hits:
        raise SystemExit("No *_ba0925b cache found; run training once to rebuild the dataset, or pass --cache.")
    return hits[-1]


def _load(cache: str):
    import zarr

    z = zarr.open(cache, mode="r")
    for key in ("ycls_pairs", "close_pairs"):
        if key not in z:
            raise SystemExit(f"{cache} has no {key}; it predates DATASET_BUILD_VERSION a0925b.")
    X = z["X"]
    n, _, f = X.shape
    last = np.empty((n, f), dtype=np.float32)
    step = int(X.chunks[0])
    for i in range(0, n, step):  # chunk-aligned reads: last timestep only
        last[i : i + step] = np.asarray(X[i : i + step, -1, :], dtype=np.float32)
    names = None
    schema = Path(cache + "_feature_schema.json")
    if schema.exists():
        s = json.loads(schema.read_text(encoding="utf-8"))
        if isinstance(s, list) and len(s) == f:
            names = [str(c) for c in s]
    spread = np.asarray(z["spread_pairs"][:]) if "spread_pairs" in z else None
    return last, names, np.asarray(z["ycls_pairs"][:]), np.asarray(z["close_pairs"][:]), spread, dict(z.attrs)


def _feature_mask(names: list[str] | None, X: np.ndarray) -> np.ndarray:
    keep = np.nanstd(X, axis=0) > 1e-8  # drop constant columns
    if names:
        keep &= np.array([n.split("::")[-1] not in (PRICE_LEVEL_FEATURES | VENUE_SCALED_FEATURES) for n in names])
    return keep


def _fit_predict(model: str, Xtr, ytr, Xte, seed: int) -> np.ndarray:
    """Return directions in {-1, 0, +1} for Xte (labels ytr in {-1, 0, +1})."""
    classes = np.unique(ytr)
    if len(classes) < 2:
        return np.zeros(len(Xte))
    if model == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=300, C=0.1, class_weight="balanced")
        clf.fit(np.clip(sc.transform(Xtr), -10, 10), ytr)
        return clf.predict(np.clip(sc.transform(Xte), -10, 10))
    import xgboost as xgb

    y_idx = np.searchsorted(classes, ytr)
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
        min_child_weight=50, reg_lambda=5.0, tree_method="hist", random_state=seed, n_jobs=-1,
    )
    clf.fit(Xtr, y_idx)
    return classes[clf.predict(Xte)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--folds", type=int, default=5, help="Walk-forward test blocks over the last half of the data")
    ap.add_argument("--model", choices=["xgb", "logreg"], default="xgb")
    ap.add_argument("--lookahead", type=int, default=None, help="Label horizon in bars (default: cache attrs)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    cache = args.cache or _latest_cache()
    print(f"[Baseline] cache={cache}")
    X, names, ycls, close_p, spread_p, attrs = _load(cache)
    n, n_pairs = ycls.shape
    pairs = [p.strip() for p in str(attrs.get("pairs", "")).split(",") if p.strip()]
    pairs = pairs if len(pairs) == n_pairs else [f"pair_{k}" for k in range(n_pairs)]
    h = int(args.lookahead or attrs.get("lookahead_bars", 30))
    seq_len = int(attrs.get("seq_len", 120))
    embargo = seq_len + h
    bpy = fx_bars_per_year(str(attrs.get("bar_freq", "5min")))

    keep = _feature_mask(names, X)
    X = np.nan_to_num(X[:, keep], nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[Baseline] rows={n:,} pairs={pairs} features={int(keep.sum())}/{len(keep)} horizon={h} embargo={embargo}")

    start = n // 2
    edges = np.linspace(start, n, args.folds + 1).astype(int)
    rows = []
    for k in range(args.folds):
        te = np.arange(edges[k], edges[k + 1])
        tr = np.arange(0, max(0, edges[k] - embargo))
        if len(tr) < 1000 or len(te) < 100:
            continue
        dirs = np.zeros((len(te), n_pairs))
        for p in range(n_pairs):
            ok = np.isfinite(ycls[tr, p])
            dirs[:, p] = _fit_predict(args.model, X[tr][ok], ycls[tr, p][ok].astype(int), X[te], args.seed)
        m = pooled_pair_metrics(dirs, te, close_p, spread_p, h, pair_names=pairs, bars_per_year=bpy)
        pp = " ".join(f"{name}={v['sharpe_net']:+.2f}" for name, v in m["per_pair"].items())
        print(
            f"[Fold {k}] test=[{te[0]},{te[-1]}] trades={m['n_trades']} net_sharpe={m['sharpe_net']:+.2f} "
            f"95%CI=[{m['sharpe_net_ci_low']:+.2f},{m['sharpe_net_ci_high']:+.2f}] "
            f"P(<=0)={m['sharpe_net_p_le_0']:.2f} | {pp}"
        )
        rows.append({"fold": k, **{kk: v for kk, v in m.items() if kk != "per_pair"},
                     "per_pair": {kk: v["sharpe_net"] for kk, v in m["per_pair"].items()}})

    n_pos = sum(r["sharpe_net_ci_low"] > 0 for r in rows)
    verdict = "SIGNAL" if rows and n_pos > len(rows) / 2 else "NO SIGNAL"
    print(f"\n[Baseline] {verdict}: CI lower bound > 0 on {n_pos}/{len(rows)} folds ({args.model})")
    out = Path("logs") / f"baseline_honest_{args.model}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"cache": cache, "verdict": verdict, "folds": rows}, indent=2), encoding="utf-8")
    print(f"[Baseline] report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
