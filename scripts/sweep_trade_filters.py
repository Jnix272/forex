"""Sweep the honest-eval min-move filter per pair.

Usage (after training, with saved validation directions):
    python scripts/sweep_trade_filters.py --cache <dataset.zarr> --preds preds.npz
        preds.npz: directions (N, P) in {-1,0,1}, sample_idx (N,) cache row indices

Smoke test with random directions on the last 20% of the cache:
    python scripts/sweep_trade_filters.py --cache <dataset.zarr> --demo

Prints, per pair and multiplier k (trade only if ATR*sqrt(h) >= k * round-trip
cost): trades, honest net Sharpe, bootstrap 95% CI low, mean net bps. Then the
pooled Sharpe with all pairs and with each pair excluded. Gap-spanning horizons
are always skipped. Pick k from validation folds only, and only once gross
Sharpe is positive, otherwise you are fitting noise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.honest_eval import load_aux_arrays, load_price_arrays, pooled_pair_metrics  # noqa: E402

DEFAULT_PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "USDCAD"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--preds", help="npz with directions (N,P) and sample_idx (N,)")
    ap.add_argument("--demo", action="store_true", help="random directions on last 20%% of rows")
    ap.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--mults", nargs="+", type=float, default=[0, 3, 5, 10, 15])
    ap.add_argument("--max-spread-pips", type=float, default=None, help="optional spread cap for all pairs")
    args = ap.parse_args()

    close_p, spread_p = load_price_arrays(args.cache, pairs=True)
    t_ns, atr_p = load_aux_arrays(args.cache)
    if close_p is None or atr_p is None:
        sys.exit("cache needs close_pairs and atr_pairs")
    if args.preds:
        z = np.load(args.preds)
        dirs, idx = np.asarray(z["directions"], dtype=np.float64), np.asarray(z["sample_idx"])
    elif args.demo:
        n0 = int(len(close_p) * 0.8)
        idx = np.arange(n0, len(close_p))
        dirs = np.random.default_rng(0).choice([-1.0, 1.0], size=(len(idx), close_p.shape[1]))
    else:
        sys.exit("give --preds or --demo")
    names = args.pairs
    if len(names) != dirs.shape[1]:
        sys.exit(f"{dirs.shape[1]} direction columns but {len(names)} pair names")

    def run(cols, k):
        sel = [names[c] for c in cols]
        filt = {n: {"min_move_cost_mult": k, "max_spread_pips": args.max_spread_pips} for n in sel}
        return pooled_pair_metrics(
            dirs[:, cols], idx, close_p[:, cols], None if spread_p is None else spread_p[:, cols],
            args.horizon, pair_names=sel, t_ns=t_ns, atr_pairs=atr_p[:, cols],
            skip_gap_windows=True, trade_filters=filt,
        )

    allc = list(range(len(names)))
    print(f"{'pair':8} {'k':>4} {'trades':>7} {'net_sharpe':>10} {'ci_low':>7} {'net_bps':>8}")
    for ci, n in enumerate(names):
        for k in args.mults:
            m = run([ci], k)["per_pair"][n]
            print(f"{n:8} {k:4g} {m['n_trades']:7d} {m['sharpe_net']:10.2f} "
                  f"{m.get('sharpe_net_ci_low', 0.0):7.2f} {m['mean_ret_bps']:8.2f}")
    print("\nPooled (same k for every pair):")
    for k in args.mults:
        row = [f"k={k:g}", f"all={run(allc, k)['sharpe_net']:.2f}"]
        for ci, n in enumerate(names):
            row.append(f"-{n}={run([c for c in allc if c != ci], k)['sharpe_net']:.2f}")
        print("  " + "  ".join(row))


if __name__ == "__main__":
    main()
