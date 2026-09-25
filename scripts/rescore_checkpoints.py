"""Re-score trained checkpoints on real net-of-spread PnL (honest metric).

Runs each checkpoint over the most recent ``--holdout`` fraction of the
training cache (chronologically last = never trained on in single-split runs;
treat with care for CV folds) and reports per-trade Sharpe after spread,
win rate and bps, overall and for the most confident signals only.

Usage (Windows, from repo root):
    .venv311\\Scripts\\python scripts\\rescore_checkpoints.py ^
        --cache data\\cache\\<your_cache>.zarr --checkpoints checkpoints

Only reads files; writes a CSV summary to logs/rescore_<timestamp>.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.honest_eval import fx_bars_per_year, load_price_arrays, net_pnl_metrics  # noqa: E402


def _open_X(cache: Path):
    try:
        import zarr

        return zarr.open(str(cache), mode="r")["X"]
    except Exception:
        f = cache if cache.suffix == ".npy" else Path(str(cache) + ".npy")
        return np.load(f, mmap_mode="r")


def _batch_proba(engine, xb: np.ndarray) -> np.ndarray:
    """(B, T, F) raw -> (B, 3) [SELL, HOLD, BUY] probabilities, mirroring predict_proba."""
    import torch

    from inference.pytorch_inference import _logits_to_proba

    scaled = np.stack([engine._transform_window(w) for w in xb])
    x = torch.as_tensor(scaled, dtype=torch.float32, device=engine.device)
    with torch.no_grad():
        out = engine.model(x)
    if isinstance(out, dict):
        out = next(v for k, v in out.items() if "dir" in k or "logit" in k or "cls" in k)
    if isinstance(out, (tuple, list)):
        out = out[0]
    out = out.detach().float().cpu().numpy().reshape(len(xb), -1)
    return np.stack([_logits_to_proba(row) for row in out])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="training cache (.zarr dir or .npy)")
    ap.add_argument("--checkpoints", default="checkpoints", help="dir searched recursively for *_best.pt")
    ap.add_argument("--holdout", type=float, default=0.2, help="last fraction of samples to score")
    ap.add_argument("--horizon", type=int, default=12, help="bars held per trade (lookahead_bars)")
    ap.add_argument("--bar-freq", default="5min")
    ap.add_argument("--max-samples", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--pattern", default="*_best.pt")
    args = ap.parse_args()

    from inference.pytorch_inference import PyTorchInferenceEngine

    cache = Path(args.cache)
    X = _open_X(cache)
    close, spread = load_price_arrays(cache)
    if close is None:
        print("[Rescore] cache has no 'close' array - cannot compute honest PnL")
        return 1
    n = int(X.shape[0])
    start = int(n * (1.0 - args.holdout))
    idx = np.arange(start, n)
    if len(idx) > args.max_samples:
        idx = idx[-args.max_samples:]
    seq_len = int(X.shape[1])
    bpy = fx_bars_per_year(args.bar_freq)
    print(f"[Rescore] cache={cache.name} samples={n:,} scoring {len(idx):,} (last {args.holdout:.0%}) "
          f"horizon={args.horizon} seq={seq_len}")

    ckpts = sorted(Path(args.checkpoints).rglob(args.pattern))
    rows = []
    for ck in ckpts:
        name = ck.stem.replace("_best", "")
        model_name = ck.parent.name if ck.parent.name != Path(args.checkpoints).name else name.split("_")[0]
        t0 = time.time()
        try:
            eng = PyTorchInferenceEngine(str(ck), model_name, seq_len=seq_len, cache_path=str(cache))
            probs = []
            for s in range(0, len(idx), args.batch):
                probs.append(_batch_proba(eng, np.asarray(X[idx[s : s + args.batch]], dtype=np.float32)))
            P = np.concatenate(probs)
        except Exception as e:
            print(f"[Rescore] {ck}: FAILED ({e})")
            rows.append({"checkpoint": str(ck), "error": str(e)})
            continue

        edge = P[:, 2] - P[:, 0]  # BUY - SELL conviction
        d_all = np.sign(edge)
        res = {"checkpoint": str(ck)}
        for tag, q in (("all", 0.0), ("top20", 0.8), ("top10", 0.9)):
            thr = np.quantile(np.abs(edge), q) if q > 0 else 0.0
            d = np.where(np.abs(edge) >= thr, d_all, 0.0)
            m = net_pnl_metrics(d, idx, close, spread, args.horizon, bars_per_year=bpy)
            for k, v in m.items():
                res[f"{tag}_{k}"] = round(v, 4) if isinstance(v, float) else v
        # Live gate is max(proba) >= hold_threshold; report the equivalent cutoffs.
        pmax = P.max(axis=1)
        res["hold_thr_top20"] = round(float(np.quantile(pmax, 0.8)), 4)
        res["hold_thr_top10"] = round(float(np.quantile(pmax, 0.9)), 4)
        for tag, thr in (("live045", 0.45), ("pmax_top20", res["hold_thr_top20"])):
            d = np.where((pmax >= thr) & (P.argmax(axis=1) != 1), d_all, 0.0)
            m = net_pnl_metrics(d, idx, close, spread, args.horizon, bars_per_year=bpy)
            res[f"{tag}_sharpe_net"] = round(m["sharpe_net"], 4)
            res[f"{tag}_n_trades"] = m["n_trades"]
        res["pred_buy_frac"] = round(float((d_all > 0).mean()), 3)
        res["edge_std"] = round(float(edge.std()), 5)
        rows.append(res)
        print(
            f"[Rescore] {ck.relative_to(args.checkpoints)}: "
            f"net_sharpe all={res['all_sharpe_net']:+.2f} top20={res['top20_sharpe_net']:+.2f} "
            f"top10={res['top10_sharpe_net']:+.2f} | win={res['all_win_rate']:.1%} "
            f"net={res['all_mean_ret_bps']:+.2f}bps cost={res['all_cost_bps']:.2f}bps "
            f"trades={res['all_n_trades']} buy%={res['pred_buy_frac']:.0%} edge_sd={res['edge_std']} "
            f"| live@0.45={res['live045_sharpe_net']:+.2f} ({res['live045_n_trades']}tr) "
            f"hold_thr_top20={res['hold_thr_top20']} "
            f"({time.time() - t0:.0f}s)"
        )

    out = Path("logs") / f"rescore_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "checkpoint", k))
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[Rescore] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
