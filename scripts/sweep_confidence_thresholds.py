"""Sweep decision thresholds on out-of-sample modern regime data.

Evaluates haelt_fold1_best.pt / haelt_fold0_best.pt across a range of confidence
thresholds against real prices and spreads using honest_eval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.honest_eval import load_aux_arrays, load_price_arrays, net_pnl_metrics, pooled_pair_metrics
from training.model_factory import build_model
from training.supervised_loop import ZarrStreamDataset, _load_scaler_state


def main():
    parser = argparse.ArgumentParser(description="Sweep confidence thresholds on trained HAELT checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/haelt_modern_2018_2025/haelt/haelt_fold1_best.pt")
    parser.add_argument("--config", default="checkpoints/haelt_modern_2018_2025/haelt/haelt_fold1_config.json")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    cfg_path = Path(args.config)
    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        return 1
    if not cfg_path.exists():
        print(f"[Error] Config not found: {cfg_path}")
        return 1

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cache_path = cfg["cache_path"]
    val_range = cfg["val_range"]
    val_idx = np.arange(val_range[0], val_range[1])
    n_features = cfg["n_features"]
    seq_len = cfg["seq_len"]

    print(f"Loading checkpoint: {ckpt_path}")
    print(f"Dataset cache: {cache_path}")
    print(f"Validation slice: {val_range[0]} to {val_range[1]} ({len(val_idx):,} samples)")

    # Load scaler state
    scaler_path = ckpt_path.parent / ckpt_path.name.replace(".pt", "_scaler.npz")
    scaler = _load_scaler_state(scaler_path) if scaler_path.exists() else None

    # Load arrays for honest evaluation
    close_pairs, spread_pairs = load_price_arrays(cache_path, pairs=True)
    t_ns, atr_pairs = load_aux_arrays(cache_path)
    pair_names = ["EURUSD", "USDJPY", "USDCAD"]

    # Build model and load weights
    device = torch.device(args.device)
    model = build_model(
        cfg.get("model", "haelt"),
        n_features,
        seq_len,
        n_classes=3,
        d_model=cfg.get("d_model", 256),
        nhead=cfg.get("nhead", 8),
        hidden_size=cfg.get("hidden_size", 256),
        num_layers=cfg.get("num_layers", 3),
        dropout=cfg.get("dropout", 0.35),
    ).to(device)

    raw_state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = raw_state["model_state"] if "model_state" in raw_state else raw_state
    model.load_state_dict(state_dict)
    model.eval()

    # Create validation dataset
    val_ds = ZarrStreamDataset(
        cache_path,
        val_idx,
        shuffle_chunks=False,
        multitask_targets=True,
        scaler=scaler,
    )
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("\nRunning GPU inference across validation batches...")
    logits_list = []
    with torch.no_grad():
        for batch in val_dl:
            xb = batch[0].to(device)
            out = model(xb)
            logits = out[0] if isinstance(out, tuple) else out
            logits_list.append(logits.cpu().float())

    all_logits = torch.cat(logits_list, dim=0)  # (N, n_classes) or (N, n_pairs, 3)
    probs = torch.softmax(all_logits, dim=-1).numpy()
    print(f"Inference complete. Probs shape: {probs.shape}")

    print("\n" + "=" * 80)
    print(f"{'Threshold':>10} | {'Trades':>8} | {'Win Rate':>9} | {'Gross Sharpe':>13} | {'Net Sharpe':>11} | {'95% CI Low':>11} | {'Mean Net (bps)':>14}")
    print("-" * 80)

    best_threshold = None
    best_sharpe = -999.0

    for thresh in args.thresholds:
        # Action decision: BUY if prob(BUY) - prob(SELL) > thresh, SELL if prob(SELL) - prob(BUY) > thresh, else HOLD
        if probs.ndim == 2 and probs.shape[1] == 3:
            # Classes: 0=SELL, 1=HOLD, 2=BUY
            p_sell = probs[:, 0]
            p_buy = probs[:, 2]
            diff = p_buy - p_sell
            dirs = np.where(diff > thresh, 1.0, np.where(diff < -thresh, -1.0, 0.0))
            dirs = dirs.reshape(-1, 1)
        elif probs.ndim == 3:
            # Multi-pair
            diff = probs[:, :, 2] - probs[:, :, 0]
            dirs = np.where(diff > thresh, 1.0, np.where(diff < -thresh, -1.0, 0.0))
        else:
            dirs = np.zeros((len(probs), len(pair_names)))

        res = pooled_pair_metrics(
            dirs,
            val_idx[:len(dirs)],
            close_pairs,
            spread_pairs,
            horizon=30,
            pair_names=pair_names[:dirs.shape[1]],
            t_ns=t_ns,
            atr_pairs=atr_pairs,
            skip_gap_windows=True,
        )

        n_trades = res["n_trades"]
        wr = res["win_rate"] * 100.0
        s_gross = res["sharpe_gross"]
        s_net = res["sharpe_net"]
        ci_low = res.get("sharpe_net_ci_low", 0.0)
        mean_bps = res["mean_ret_bps"]

        print(f"{thresh:>10.2f} | {n_trades:>8d} | {wr:>8.1f}% | {s_gross:>13.2f} | {s_net:>11.2f} | {ci_low:>11.2f} | {mean_bps:>14.2f}")

        if n_trades >= 30 and s_net > best_sharpe:
            best_sharpe = s_net
            best_threshold = thresh

    print("=" * 80)
    if best_threshold is not None:
        print(f"Optimal Threshold: {best_threshold:.2f} (Net Sharpe: {best_sharpe:.2f})")
    else:
        print("No threshold produced >= 30 trades with positive Sharpe.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
