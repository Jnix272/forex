"""Dataset health checks for a built cache (the checks used in the 2026-09-25 audit)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def inspect_cache(cache: str | Path, max_rows: int = 4000) -> dict:
    import zarr

    z = zarr.open(str(cache), mode="r")
    keys = list(z.array_keys())
    attrs = dict(z.attrs)
    n = int(z["X"].shape[0])
    pairs = [p.strip() for p in str(attrs.get("pairs", "")).split(",") if p.strip()]
    h = int(attrs.get("lookahead_bars", 30))
    out: dict = {"arrays": {k: list(z[k].shape) for k in keys}, "attrs": attrs, "problems": [], "pairs": {}}
    problems = out["problems"]

    for k in keys:
        if z[k].shape[0] != n:
            problems.append(f"length mismatch {k}: {z[k].shape[0]} vs X {n}")
    for k in ("y_pairs", "ycls_pairs", "close_pairs", "spread_pairs", "t_ns"):
        if k not in keys:
            problems.append(f"missing per-pair array {k} (cache predates DATASET_BUILD_VERSION a0925b)")

    if "t_ns" in keys:
        t = np.asarray(z["t_ns"][:])
        d = np.diff(t)
        valid = t[t >= 0]
        out["time"] = {
            "missing_frac": float((t < 0).mean()),
            "backwards": int((d < 0).sum()),
            "duplicates": int((d == 0).sum()),
            "start": str(np.datetime64(int(valid.min()), "ns")) if valid.size else None,
            "end": str(np.datetime64(int(valid.max()), "ns")) if valid.size else None,
        }
        if out["time"]["duplicates"]:
            problems.append(f"{out['time']['duplicates']} duplicate timestamps")

    if "ycls_pairs" in keys and "close_pairs" in keys:
        ycp = np.asarray(z["ycls_pairs"][:])
        cp = np.asarray(z["close_pairs"][:])
        sp = np.asarray(z["spread_pairs"][:]) if "spread_pairs" in keys else None
        if not np.allclose(np.asarray(z["y_cls"][:]), ycp[:, 0], equal_nan=True):
            problems.append("scalar y_cls is not the market pair's own label")
        for k, p in enumerate(pairs or [f"pair_{i}" for i in range(ycp.shape[1])]):
            c = ycp[:, k]
            pip = 0.01 if "JPY" in p else 0.0001
            fwd = np.r_[cp[h:, k] - cp[:-h, k], np.full(h, np.nan)]
            info = {
                "sell": float(np.mean(c == -1)), "hold": float(np.mean(c == 0)), "buy": float(np.mean(c == 1)),
                "nan": float(np.mean(np.isnan(c))),
                "median_close": float(np.nanmedian(cp[:, k])),
                "median_spread_pips": float(np.nanmedian(sp[:, k]) / pip) if sp is not None else None,
                "fwd_pips_buy": float(np.nanmean(fwd[c == 1]) / pip) if (c == 1).any() else None,
                "fwd_pips_sell": float(np.nanmean(fwd[c == -1]) / pip) if (c == -1).any() else None,
            }
            out["pairs"][p] = info
            if not 0.02 <= info["hold"] <= 0.9:
                problems.append(f"{p}: HOLD share {info['hold']:.1%} implausible")
            if info["median_spread_pips"] is not None and not 0.05 <= info["median_spread_pips"] <= 10:
                problems.append(f"{p}: median spread {info['median_spread_pips']:.2f} pips out of range")
            if info["fwd_pips_buy"] is not None and info["fwd_pips_sell"] is not None and not (
                info["fwd_pips_buy"] > 0 > info["fwd_pips_sell"]
            ):
                problems.append(f"{p}: labels disagree with the forward price move")

    schema_path = Path(str(cache) + "_feature_schema.json")
    names = None
    if schema_path.exists():
        s = json.loads(schema_path.read_text(encoding="utf-8"))
        names = s if isinstance(s, list) else s.get("feature_names")
    X = z["X"]
    last = np.asarray(X[:, -1, :], dtype=np.float32)
    if n > max_rows:
        last = last[np.sort(np.random.default_rng(0).choice(n, max_rows, replace=False))]
    std = np.nanstd(last, axis=0)
    names = names if names and len(names) == last.shape[1] else [f"f{i}" for i in range(last.shape[1])]
    const = [names[j] for j in np.where(std < 1e-9)[0]]
    out["features"] = {
        "n": len(names), "nan_frac": float(np.isnan(last).mean()), "inf": int(np.isinf(last).sum()),
        "max_abs": float(np.nanmax(np.abs(last))), "constant": const,
        "constant_bases": sorted({c.split("::")[-1] for c in const}),
    }
    expect_varying = ("ret_5", "rsi_14", "vol_clock_pace", "session_label", "sentiment_raw", "trend_quality")
    for base in expect_varying:
        for p in pairs:
            nm = f"{p}::{base}"
            if nm in names and std[names.index(nm)] < 1e-9:
                problems.append(f"{nm} is constant")
    return out
