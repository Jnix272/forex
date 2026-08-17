import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtesting.backtest import ForexScalingBacktest
from config.settings import BACKTEST, FEATURES, active_checkpoint_dir
from config.strategy_profiles import STRATEGY_PROFILES, strategy_profile
from data.data_ingestion import ForexDataPipeline, load_or_generate
from data.news_feed import get_latest_headlines
from features.advanced_features import AdvancedFeatureBuilder
from features.feature_engineering import FeatureEngineer
from features.finbert_sentiment import SentimentPipeline
from scripts.backtest_model import (
    PIP_SIZES,
    _bars_per_year_from_freq,
    _batched_logits,
    _checkpoint_state_dict,
    _load_checkpoint_config,
    _normalize_backtest_metrics,
    _to_pandas_bars,
    _write_df,
)
from training.train_gpu import build_model


def log(m: str) -> None:
    print(m, flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="True Walk-Forward Out-Of-Sample Backtester")
    p.add_argument("--strategy-mode", default="scalping", choices=sorted(STRATEGY_PROFILES.keys()))
    p.add_argument("--bar-freq", default=None)
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--start", default="2008-01-01", help="Must match training start to align folds")
    p.add_argument("--end", default="2025-12-31", help="Must match training end to align folds")
    p.add_argument("--pair", default="EURUSD")
    p.add_argument("--lots", type=float, default=float(BACKTEST.get("lots", 0.1)))
    p.add_argument("--equity", type=float, default=10000.0)
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--inference-batch-size", type=int, default=512)
    p.add_argument("--min-confidence", type=float, default=float(BACKTEST.get("min_confidence", 0.45)))
    p.add_argument("--stop-pips", type=float, default=15.0)
    p.add_argument("--take-pips", type=float, default=30.0)
    p.add_argument("--min-gap-bars", type=int, default=3)
    p.add_argument("--slippage-pips", type=float, default=float(BACKTEST.get("slippage_pips", 0.7)))
    p.add_argument("--commission-per-lot", type=float, default=float(BACKTEST.get("commission_per_lot", 3.5)))
    p.add_argument("--execution-delay-bars", type=int, default=int(BACKTEST.get("execution_delay_bars", 1)))
    p.add_argument("--source", default="dukascopy", choices=["dukascopy", "synthetic", "tds", "lmax_historical"])
    p.add_argument("--n-folds", type=int, default=7, help="Number of walk-forward folds (default: 7)")
    args = p.parse_args()
    prof = strategy_profile(args.strategy_mode)
    if args.bar_freq is None:
        args.bar_freq = str(prof["bar_freq"])
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(prof["checkpoint_dir"])
    if args.strategy_mode != "scalping":
        if args.seq_len == 60:
            args.seq_len = int(prof["seq_len"])
        if args.min_confidence == float(BACKTEST.get("min_confidence", 0.45)):
            args.min_confidence = float(prof["min_confidence"])
    return args


def get_fold_checkpoint(model: str, fold: int, checkpoint_dir: str) -> Path | None:
    search_dirs = [Path(checkpoint_dir), active_checkpoint_dir(), Path("checkpoints")]
    for base in search_dirs:
        if not base.exists():
            continue
        candidates = [
            base / f"{model}_fold{fold}_best.pt",
            base / model / f"{model}_fold{fold}_best.pt",
            base / f"{model}_fold{fold}_ep10.pt",
            base / f"{model}_fold{fold}_ep5.pt",
        ]
        for p in candidates:
            if p.exists():
                return p
    return None


def run_true_walkforward():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("=== True OOS Walk-Forward Backtester ===")
    log(f"Model: {args.model}")
    log(f"Folds: {args.n_folds}")
    log(f"Period: {args.start} to {args.end}")
    log(f"Device: {device}")

    try:
        sent = SentimentPipeline()
    except Exception:
        sent = None

    log(f"Loading FULL dataset {args.start} -> {args.end}...")
    pipeline = ForexDataPipeline(bar_freq=args.bar_freq)
    fe = FeatureEngineer(
        atr_window=FEATURES.get("atr_window", 14), lag_windows=FEATURES.get("lag_windows", [5, 10, 20])
    )
    afb = AdvancedFeatureBuilder()

    ticks = load_or_generate(source=args.source, pair=args.pair, start=args.start, end=args.end, n_rows=1000000)
    base_bars = _to_pandas_bars(pipeline.run(ticks))
    if base_bars is None or len(base_bars) == 0:
        log("Fatal: Failed to load dataset or no bars generated.")
        return

    # NOTE: feature engineering is deferred to inside the fold loop to prevent
    # data leakage (W1). Rolling indicators would otherwise see future bars.
    n_samples = len(base_bars)
    # FIX W2: n_folds+1 edges produce exactly n_folds fold boundaries
    # (original n_folds+2 produced an out-of-bounds final-fold index)
    edges = np.linspace(0, n_samples, args.n_folds + 1, dtype=np.int64)

    all_signals = []

    for k in range(args.n_folds):
        val_start_idx = int(edges[k])
        # FIX W2+W3: edges has n_folds+1 entries; last fold end is clamped to n_samples
        val_end_idx = min(int(edges[k + 1]), n_samples)

        inf_start_idx = max(0, val_start_idx - args.seq_len)

        val_start_date = base_bars.index[val_start_idx]
        val_end_date = base_bars.index[val_end_idx - 1]

        log(f"\n--- Fold {k} ---")
        log(f"Val slice: index {val_start_idx} to {val_end_idx}")
        log(f"Val dates: {val_start_date} to {val_end_date}")

        ckpt_path = get_fold_checkpoint(args.model, k, args.checkpoint_dir)
        if not ckpt_path:
            log(f"WARNING: Could not find checkpoint for {args.model} fold {k}. Skipping this era!")
            continue

        log(f"Loaded weights: {ckpt_path.name}")

        cfg = _load_checkpoint_config(ckpt_path)
        n_features = int(cfg.get("n_features") or 0)
        if n_features <= 0:
            log(f"WARNING: {ckpt_path.name} missing n_features; skipping fold.")
            continue

        # FIX W1: build features only up to val_end_idx to prevent data leakage.
        # Rolling indicators (ATR, lags, regime) must not see bars beyond this fold.
        fold_bars = base_bars.iloc[:val_end_idx]
        f_base_fold = fe.build(fold_bars)
        f_adv_fold = afb.build(fold_bars, base_features=f_base_fold)
        features_df = pd.concat([f_base_fold, f_adv_fold], axis=1).reindex(fold_bars.index).ffill().fillna(0.0)

        model_kwargs = {
            "d_model": cfg.get("d_model", 256),
            "nhead": cfg.get("nhead", 8),
            "num_layers": cfg.get("num_layers", 3),
            "hidden_size": cfg.get("hidden_size", 256),
            "dropout": cfg.get("dropout", 0.0),
            "multitask": cfg.get("multitask", False),
        }
        if "expert" in args.model.lower():
            model_kwargs["n_experts"] = cfg.get("n_experts", 4)
            model_kwargs["top_k"] = cfg.get("top_k", 2)

        n_pairs = int(cfg.get("_n_pairs") or cfg.get("n_pairs") or 1)
        f_per_pair = int(cfg.get("_f_per_pair") or cfg.get("f_per_pair") or max(1, n_features // max(1, n_pairs)))

        class Cfg:
            def __init__(self):
                self.model = args.model
                self.hidden_size = model_kwargs["hidden_size"]  # noqa: B023
                self.d_model = model_kwargs["d_model"]  # noqa: B023
                self.nhead = model_kwargs["nhead"]  # noqa: B023
                self.num_layers = model_kwargs["num_layers"]  # noqa: B023
                self.dropout = model_kwargs["dropout"]  # noqa: B023
                self.seq_len = args.seq_len
                self.multitask = model_kwargs["multitask"]  # noqa: B023
                self.pair_embed_dim = 0
                self.loss = str(cfg.get("loss", "cross_entropy"))  # noqa: B023
                self.corr_window = 20
                self.corr_window_long = 60
                self.momentum_window = 20
                self._n_pairs = n_pairs  # noqa: B023
                self._f_per_pair = f_per_pair  # noqa: B023

        model = build_model(args.model, n_features, Cfg()).to(device)
        state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        model.load_state_dict(_checkpoint_state_dict(state), strict=False)
        model.eval()

        X_slice = features_df.iloc[inf_start_idx:val_end_idx].copy()

        if sent is not None and "finbert_sentiment" not in X_slice.columns:
            try:
                X_slice["finbert_sentiment"] = float(
                    sent.score_headlines(get_latest_headlines(limit=12) or ["Market update"])
                )
            except Exception:
                X_slice["finbert_sentiment"] = 0.0

        if X_slice.shape[1] < n_features:
            pad_cols = pd.DataFrame(
                0.0, index=X_slice.index, columns=[f"xpad_{i}" for i in range(n_features - X_slice.shape[1])]
            )
            X_slice = pd.concat([X_slice, pad_cols], axis=1)
        elif X_slice.shape[1] > n_features:
            X_slice = X_slice.iloc[:, :n_features]

        x_t = torch.tensor(
            np.nan_to_num(X_slice.values, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32, device=device
        )

        with torch.no_grad():
            logits = _batched_logits(model, x_t, args.seq_len, args.inference_batch_size)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        cls, conf = probs.argmax(axis=1), probs.max(axis=1)

        pip_size = PIP_SIZES.get(args.pair.upper(), 0.0001)
        last_signal_i = -(10**9)
        fold_signals = []

        regime_vals = X_slice["regime_label"].values if "regime_label" in X_slice.columns else None
        close_vals = base_bars["close"].values
        ts_vals = base_bars.index

        for off, c in enumerate(cls):
            i = inf_start_idx + args.seq_len + off
            # FIX W6: guard against out-of-bounds access on the last fold
            if i >= n_samples:
                break

            adj_min_conf = args.min_confidence
            if regime_vals is not None:
                rl = float(np.asarray(regime_vals[args.seq_len + off]).item())
                if rl > 0.5:
                    adj_min_conf = max(0.5, args.min_confidence - 0.05)
                elif rl < -0.5:
                    adj_min_conf = min(0.95, args.min_confidence + 0.05)

            if conf[off] < adj_min_conf:
                continue
            if i - last_signal_i < max(1, args.min_gap_bars):
                continue

            price = float(np.asarray(close_vals[i]).item())
            # NOTE W7: class mapping is correct (not inverted):
            #   c==0 = SELL signal → act=2 (OPEN_SHORT): stop above, TP below
            #   c==2 = BUY  signal → act=1 (OPEN_LONG):  stop below, TP above
            if c == 0:
                act = 2
                stop_loss = price + args.stop_pips * pip_size
                take_profit = price - args.take_pips * pip_size
            elif c == 2:
                act = 1
                stop_loss = price - args.stop_pips * pip_size
                take_profit = price + args.take_pips * pip_size
            else:
                act = 0
                stop_loss = 0.0
                take_profit = 0.0

            if act:
                last_signal_i = i
                kelly_frac = 0.25
                win_prob = float(conf[off])
                reward_to_risk = args.take_pips / max(args.stop_pips, 1e-5)
                if reward_to_risk > 0:
                    kelly_k = win_prob - ((1.0 - win_prob) / reward_to_risk)
                else:
                    kelly_k = 0.0
                dynamic_lots = args.lots * max(0.0, kelly_k * kelly_frac)
                dynamic_lots = max(0.01, dynamic_lots)

                fold_signals.append(
                    {
                        "timestamp": ts_vals[i],
                        "action": act,
                        "lots": dynamic_lots,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "confidence": win_prob,
                        "fold": k,
                    }
                )

        all_signals.extend(fold_signals)
        log(f"Fold {k} generated {len(fold_signals)} trades.")

    if not all_signals:
        log("No trades generated across all folds.")
        return

    log(f"\nTotal trades stitched: {len(all_signals)}")
    sig_df = pd.DataFrame(all_signals).set_index("timestamp")
    sig_df = sig_df[~sig_df.index.duplicated(keep="first")].sort_index()

    # FIX W5: clamp bars lower bound so the backtest never gets an empty frame
    # when sig_df.index[0] precedes the first bar (e.g. due to execution delay)
    bars_start = max(base_bars.index[0], sig_df.index[0])
    bt = ForexScalingBacktest(
        bars=base_bars.loc[bars_start:],
        signals=sig_df,
        initial_equity=args.equity,
        commission_per_lot=args.commission_per_lot,
        slippage_pips=args.slippage_pips,
        pip_size=PIP_SIZES.get(args.pair.upper(), 0.0001),
        execution_delay_bars=max(1, int(args.execution_delay_bars)),
        bars_per_year=_bars_per_year_from_freq(args.bar_freq),
    )
    # FIX W8: assign run() result (was discarded, which is misleading)
    bt.run()
    metrics = bt.performance_metrics()
    norm_metrics = _normalize_backtest_metrics(metrics)

    out_dir = Path("backtests/true_walk_forward")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # FIX W4: use _write_df() - bt output is now pl.DataFrame after Polars migration
    _write_df(bt.get_trade_log(), out_dir / f"{args.model}_OOS_{stamp}_trades.csv")
    _write_df(bt.results_df, out_dir / f"{args.model}_OOS_{stamp}_equity.csv")
    (out_dir / f"{args.model}_OOS_{stamp}_summary.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "start": args.start,
                "end": args.end,
                "folds_stitched": args.n_folds,
                "metrics": metrics,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    log("\n=== TRUE OOS BACKTEST SUMMARY ===")
    log(f"Trades:       {norm_metrics['n_trades']}")
    log(f"Win Rate:     {norm_metrics['win_rate']:.2%}")
    log(f"Sharpe Ratio: {norm_metrics['sharpe']:.4f}")
    log(f"Max Drawdown: {norm_metrics['max_drawdown']:.2%}")
    log(f"Net PnL:      ${norm_metrics['net_pnl']:,.2f}")
    log(f"Return:       {norm_metrics['total_return_pct']:.2f}%")
    log(f"Results saved to {out_dir}")


if __name__ == "__main__":
    run_true_walkforward()
