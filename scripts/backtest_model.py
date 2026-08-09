import argparse
import json
import sys
from datetime import UTC, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtesting.backtest import ForexScalingBacktest
from backtesting.improvements import MonteCarloBacktest
from config.settings import BACKTEST, FEATURES
from config.strategy_profiles import STRATEGY_PROFILES, strategy_profile
from data.data_ingestion import ForexDataPipeline, load_or_generate
from data.news_feed import get_latest_headlines
from features.advanced_features import AdvancedFeatureBuilder
from features.feature_engineering import FeatureEngineer
from features.finbert_sentiment import SentimentPipeline
from models.ensemble import EnsembleMetaLearner
from training.train_gpu import build_model

DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "EURGBP", "NZDUSD", "EURJPY", "GBPJPY"]
PIP_SIZES = {"EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001, "USDCAD": 0.0001, "USDCHF": 0.0001, "EURGBP": 0.0001, "NZDUSD": 0.0001, "USDJPY": 0.01, "EURJPY": 0.01, "GBPJPY": 0.01}


def _pair_feature_dims(n_features: int, n_pairs: int = 1, cfg: dict | None = None) -> tuple[int, int]:
    """Resolve (n_pairs, f_per_pair) from checkpoint config — never hardcode width."""
    cfg = cfg or {}
    n_pairs = int(cfg.get("_n_pairs") or cfg.get("n_pairs") or n_pairs or 1)
    n_pairs = max(1, n_pairs)
    fpp = cfg.get("_f_per_pair") or cfg.get("f_per_pair")
    if fpp is not None:
        return n_pairs, max(1, int(fpp))
    return n_pairs, max(1, int(n_features) // n_pairs)


def _fit_feature_width(frame: pd.DataFrame, width: int) -> pd.DataFrame:
    """Pad or truncate feature columns to ``width`` (checkpoint n_features / f_per_pair)."""
    width = max(1, int(width))
    if frame.shape[1] >= width:
        return frame.iloc[:, :width]
    pad = pd.DataFrame(
        0.0,
        index=frame.index,
        columns=[f"pad_{i}" for i in range(width - frame.shape[1])],
    )
    return pd.concat([frame, pad], axis=1)


def log(m: str) -> None:
    print(m, flush=True)


def _append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _fmt_metric(value, fmt: str = ".4f", prefix: str = "", suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        return f"{prefix}{float(value):{fmt}}{suffix}"
    except Exception:
        return str(value)


def _send_discord_backtest(fields: dict, force: bool = True) -> None:
    try:
        from monitoring.discord_alerts import DiscordAlerter
        DiscordAlerter(verbose=False).send("backtest_result", fields, force=force)
    except Exception as exc:
        log(f"[Discord] backtest alert skipped: {exc}")


def _normalize_backtest_metrics(metrics: dict) -> dict:
    if not metrics or metrics.get("error"):
        return {
            "n_trades": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "net_pnl": 0.0,
            "gross_pnl": 0.0,
            "total_commission": 0.0,
            "profit_factor": 0.0,
            "total_return_pct": 0.0,
            "error": metrics.get("error", "No metrics") if isinstance(metrics, dict) else "No metrics",
        }
    if "win_rate_pct" in metrics:
        win_rate = float(metrics.get("win_rate_pct", 0.0) or 0.0) / 100.0
    else:
        win_rate = float(metrics.get("win_rate", 0.0) or 0.0)
        if win_rate > 1.0:
            win_rate /= 100.0
    if "max_drawdown_pct" in metrics:
        max_dd = float(metrics.get("max_drawdown_pct", 0.0) or 0.0) / 100.0
    else:
        max_dd = float(metrics.get("max_drawdown", 0.0) or 0.0)
        if abs(max_dd) > 1.0:
            max_dd /= 100.0
    return {
        "n_trades": float(metrics.get("n_trades", 0.0) or 0.0),
        "sharpe": float(metrics.get("sharpe", metrics.get("sharpe_ratio", 0.0)) or 0.0),
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "net_pnl": float(metrics.get("net_pnl", metrics.get("net_pnl_usd", metrics.get("total_pnl_usd", 0.0))) or 0.0),
        "gross_pnl": float(metrics.get("gross_pnl_usd", metrics.get("gross_pnl", 0.0)) or 0.0),
        "total_commission": float(metrics.get("total_commission_usd", metrics.get("transaction_costs", 0.0)) or 0.0),
        "profit_factor": float(metrics.get("profit_factor", 0.0) or 0.0),
        "total_return_pct": float(metrics.get("total_return_pct", 0.0) or 0.0),
        "error": "",
    }


def _build_windows(start: str, end: str, window_days: int, step_days: int) -> list[tuple[str, str]]:
    s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    out, cur = [], s
    while cur <= e:
        w_end = min(cur + pd.Timedelta(days=max(1, window_days) - 1), e)
        out.append((cur.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d")))
        cur += pd.Timedelta(days=max(1, step_days))
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy-mode", default="scalping", choices=sorted(STRATEGY_PROFILES.keys()))
    p.add_argument("--bar-freq", default=None, help="Backtest bar frequency, e.g. 1min, 15min, 1h")
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--checkpoint-dir", default=None,
                   help="Directory containing production_best.pt or {model}_best.pt. Defaults to the selected strategy profile.")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2025-01-07")
    p.add_argument("--pair", default="EURUSD")
    p.add_argument("--lots", type=float, default=float(BACKTEST.get("lots", 0.1)))
    p.add_argument("--equity", type=float, default=10000.0)
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--inference-batch-size", type=int, default=512)
    p.add_argument("--min-confidence", type=float, default=float(BACKTEST.get("min_confidence", 0.45)))
    p.add_argument("--stop-pips", type=float, default=float(BACKTEST.get("stop_pips", 12.0)))
    p.add_argument("--take-pips", type=float, default=float(BACKTEST.get("take_profit_pips", 18.0)))
    p.add_argument("--min-gap-bars", type=int, default=3)
    p.add_argument("--slippage-pips", type=float, default=float(BACKTEST.get("slippage_pips", 0.7)))
    p.add_argument("--commission-per-lot", type=float, default=float(BACKTEST.get("commission_per_lot", 3.5)))
    p.add_argument("--execution-delay-bars", type=int, default=int(BACKTEST.get("execution_delay_bars", 1)))
    p.add_argument("--mc-sims", type=int, default=int(BACKTEST.get("mc_sims", 500)))
    p.add_argument("--multitask", action="store_true", default=False)
    p.add_argument("--n-pairs", type=int, default=1)
    p.add_argument("--source", default="dukascopy", choices=["dukascopy", "synthetic", "tds", "lmax_historical"])
    p.add_argument("--walk-forward", action="store_true")
    p.add_argument("--wf-window-days", type=int, default=7)
    p.add_argument("--wf-step-days", type=int, default=7)
    p.add_argument("--meta-labeling", action="store_true", default=False,
                   help="Train a meta-labeler (Improvement #6) per walk-forward window and filter signals by P(profitable).")
    p.add_argument("--meta-prob-threshold", type=float, default=0.55,
                   help="Meta-labeler probability threshold for taking a trade.")
    p.add_argument("--meta-min-samples", type=int, default=50,
                   help="Minimum trade samples required before meta-labeler training.")
    p.add_argument("--execution-engine", default="legacy", choices=["legacy", "advanced"],
                   help="Execution model (Improvement #7): legacy = fixed slippage; "
                        "advanced = latency + adverse-selection overlay that adjusts "
                        "effective slippage per window.")
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
            args.min_confidence = float(prof["guard_min_confidence"])
    return args


def _resolve_checkpoint(model: str, ckpt: str | None, checkpoint_dir: str | Path | None = None) -> Path | None:
    if ckpt:
        p = Path(ckpt)
        return p if p.exists() else None
    model = str(model or "").lower().strip()
    search_dirs = []
    if checkpoint_dir:
        search_dirs.append(Path(checkpoint_dir))
    search_dirs.append(Path("checkpoints"))
    seen = set()
    candidates = []
    for base in search_dirs:
        key = str(base.resolve() if base.exists() else base)
        if key in seen:
            continue
        seen.add(key)
        candidates.extend([
            base / "production_best.pt",
            base / f"{model}_best.pt",
            base / model / f"{model}_best.pt",
            base / "normal" / model / f"{model}_best.pt",
            base / "baseline" / model / f"{model}_best.pt",
        ])
    for p in candidates:
        if p.exists():
            return p
    return None


def _checkpoint_state_dict(c):
    if isinstance(c, dict):
        for k in ("model_state", "model_state_dict", "state_dict"):
            if isinstance(c.get(k), dict):
                return c[k]
    return c


def _batched_logits(model, x: torch.Tensor, seq_len: int, bs: int) -> torch.Tensor:
    outs = []
    if len(x) < seq_len:
        return torch.empty((0, 3), device=x.device)
    
    # Use native unfold for sliding window batching
    windows = x.unfold(0, seq_len, 1).transpose(1, 2)
    for s in range(0, len(windows), max(1, bs)):
        e = min(len(windows), s + max(1, bs))
        w = windows[s:e]
        o = model(w)
        if isinstance(o, dict):
            o = o.get("direction_logits", o.get("logits", o.get("direction")))
        elif isinstance(o, (tuple, list)):
            o = o[0]
        if isinstance(o, torch.Tensor) and o.ndim == 1:
            # Scalar direction score -> synthesize 3-class logits [SELL, HOLD, BUY]
            o = torch.stack([-o, torch.zeros_like(o), o], dim=1)
        elif isinstance(o, torch.Tensor) and o.ndim == 2 and o.shape[1] == 1:
            s_val = o.squeeze(1)
            o = torch.stack([-s_val, torch.zeros_like(s_val), s_val], dim=1)
        outs.append(o.detach().cpu())
    return torch.cat(outs, dim=0)


def _build_meta_labeler_mask(args, base_bars: pd.DataFrame, X: pd.DataFrame, cls: np.ndarray, seq_len: int):
    """
    Train a meta-labeler (Improvement #6) on the current window and return
    (meta_ok_mask, meta_labeler). ``meta_ok_mask`` is aligned to ``cls``
    (rows ``seq_len..len(X)``): True = meta-model says the trade is likely
    profitable. Returns (None, None) when training is impossible.
    """
    try:
        from labeling.triple_barrier_labeling import compute_triple_barrier_labels
        from labeling.triple_barrier_meta import MetaLabelConfig, MetaLabeler

        cls = np.asarray(cls)
        n_pred = len(cls)
        # Direction per predicted bar: cls==2 buy(+1), cls==0 sell(-1), else hold(0)
        dirn = np.zeros(n_pred, dtype=float)
        dirn[cls == 2] = 1.0
        dirn[cls == 0] = -1.0

        labels_df = compute_triple_barrier_labels(
            bars=base_bars,
            features=X,
            vertical_bars=int(getattr(args, "seq_len", 60) or 20),
            profit_atr_mult=float(getattr(args, "take_pips", 18.0) or 1.8),
            stop_atr_mult=float(getattr(args, "stop_pips", 12.0) or 0.9),
            pip_size=PIP_SIZES.get(str(args.pair).upper(), 0.0001),
            execution_delay_bars=max(1, int(getattr(args, "execution_delay_bars", 1))),
        )
        if labels_df is None or labels_df.empty:
            log("[Meta] no TBM labels available; meta-labeling disabled")
            return None, None

        # Full-length primary direction (predicted rows only), aligned by index
        pred_idx = X.index[seq_len:seq_len + n_pred]
        full_dirn = pd.Series(0.0, index=X.index)
        full_dirn.loc[pred_idx] = dirn
        full_labels = labels_df["label"].reindex(X.index).fillna(0.0).to_numpy(dtype=float)

        cfg = MetaLabelConfig(
            min_meta_samples=int(getattr(args, "meta_min_samples", 50)),
            meta_train_frac=0.7,
            meta_prob_threshold=float(getattr(args, "meta_prob_threshold", 0.55)),
            meta_features=[],
        )
        meta = MetaLabeler(cfg)
        meta.fit(full_dirn.to_numpy(dtype=float), full_labels, None)
        if not getattr(meta, "_is_fitted", False):
            log("[Meta] not enough trade samples; meta-labeling disabled")
            return None, None
        mask = meta.should_trade(dirn, None)
        return mask, meta
    except Exception as e:
        log(f"[Meta] meta-labeling disabled: {e}")
        return None, None


def _load_json_sidecar(ckpt: Path) -> dict:
    p = ckpt.with_suffix(ckpt.suffix + ".json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _load_ensemble_manifest(ckpt: Path) -> dict:
    path = ckpt.parent / "ensemble_manifest.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _load_checkpoint_config(ckpt: Path) -> dict:
    candidates = [
        ckpt.with_name(ckpt.stem + "_config.json"),
        ckpt.parent / f"{ckpt.stem.replace('_best', '')}_config.json",
        ckpt.parent / f"{ckpt.stem.replace('_best', '')}_fold0_config.json",
        ckpt.parent.parent / f"{ckpt.stem.replace('_best', '')}_fold0_config.json",
        ckpt.parent / "mamba_fold0_config.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
    return {}


def _load_ensemble_model(ckpt: Path, n_features: int, seq_len: int, device: torch.device):
    manifest = _load_ensemble_manifest(ckpt)
    meta = _load_json_sidecar(ckpt).get("meta", {})
    base_models = manifest.get("base_models") or []
    base_names = [str(b.get("name")) for b in base_models if b.get("name")]
    if not base_names:
        base_names = list(meta.get("base_names", ["haelt", "mamba", "gnn"]))
    bases = []
    for name in base_names:
        manifest_ckpt = next(
            (
                Path(str(b.get("checkpoint")))
                for b in base_models
                if str(b.get("name")) == name and b.get("checkpoint")
            ),
            None,
        )
        if manifest_ckpt is not None and not manifest_ckpt.is_absolute():
            manifest_ckpt = (ckpt.parent / manifest_ckpt).resolve()
        bck = manifest_ckpt if manifest_ckpt is not None and manifest_ckpt.exists() else _resolve_checkpoint(name, None, ckpt.parent)
        if bck is None:
            raise FileNotFoundError(f"Missing base checkpoint for ensemble base '{name}'")
        class BCfg:
            def __init__(self):
                self.model = name
                self.hidden_size = 256
                self.d_model = 256
                self.nhead = 8
                self.num_layers = 3
                self.dropout = 0.1
                self.seq_len = seq_len
                self.multitask = True
                self.pair_embed_dim = 16
                self.loss = "cross_entropy"
                self.corr_window = 20
                self.corr_window_long = 60
                self.momentum_window = 20
                bcfg = _load_checkpoint_config(bck) if bck else {}
                self._n_pairs, self._f_per_pair = _pair_feature_dims(
                    n_features, n_pairs=int(bcfg.get("_n_pairs") or meta.get("_n_pairs") or 1), cfg=bcfg or meta,
                )
        bm = build_model(name, n_features, BCfg()).to(device)
        bstate = torch.load(bck, map_location=device, weights_only=True)
        bm.load_state_dict(_checkpoint_state_dict(bstate), strict=False)
        bm.eval()
        bases.append(bm)
    em = EnsembleMetaLearner(bases, context_dim=32, hidden=64, base_names=base_names).to(device)
    estate = torch.load(ckpt, map_location=device, weights_only=True)
    em.load_state_dict(_checkpoint_state_dict(estate), strict=False)
    em.eval()
    return em


def _advanced_execution_overlay(args, base_bars, signals):
    """
    Estimate effective execution slippage using the advanced execution models
    (Improvement #7 wiring). Returns (effective_slippage_pips, meta_dict).

    ``legacy`` (default) returns the baseline --slippage-pips unchanged.
    ``advanced`` overlays latency + adverse-selection (toxicity) on the baseline
    so backtests price fills closer to real-world execution conditions.
    """
    engine = str(getattr(args, "execution_engine", "legacy") or "legacy").lower()
    if engine == "legacy":
        return float(args.slippage_pips), {}

    try:
        from backtesting.execution import AdverseSelectionModel, LatencyModel
        asm = AdverseSelectionModel()
        lat = LatencyModel()
        base = float(args.slippage_pips)
        costs = []
        closes = np.asarray(base_bars["close"].values, dtype=float)
        n = len(closes)
        for s in signals:
            i = base_bars.index.get_indexer([s["timestamp"]])[0]
            lo = max(0, i - 20)
            window = closes[lo:i + 1]
            if len(window) >= 3 and np.ptp(window) > 0:
                vol = float(np.std(np.diff(window) / window[1:]))
            else:
                vol = 0.001
            spread = float(np.abs(np.diff(closes[max(0, i - 1):i + 1])).max() or 0.0002)
            spread = max(spread, 0.0001)
            tox = asm.compute_toxicity_score(
                queue_position=10, max_queue=1000, spread=spread, volatility=vol,
            )
            latency_us = lat.sample_md_to_order_latency()
            latency_pips = min(0.5, latency_us / 1e6 * 100.0)
            costs.append(base * (1.0 + tox) + latency_pips)
        mean_eff = float(np.mean(costs)) if costs else base
        meta = {
            "execution_engine": "advanced",
            "baseline_slippage_pips": round(base, 4),
            "mean_effective_slippage_pips": round(mean_eff, 4),
            "mean_toxicity": round(float(np.mean([
                asm.compute_toxicity_score(queue_position=10, max_queue=1000,
                                           spread=0.0002, volatility=0.001)
                for _ in range(16)
            ])), 4),
            "overlay_applied_to_n_signals": len(signals),
        }
        return mean_eff, meta
    except Exception as exc:
        print(f"[Exec] advanced execution overlay unavailable ({exc}); using legacy slippage.")
        return float(args.slippage_pips), {}


def run_backtest():
    args = parse_args()
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = Path("logs/backtests")
    wf_jsonl = out_dir / f"{args.model}_{run_id}_walkforward.jsonl"
    windows = [(args.start, args.end)] if not args.walk_forward else _build_windows(args.start, args.end, args.wf_window_days, args.wf_step_days)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = _resolve_checkpoint(args.model, args.checkpoint, args.checkpoint_dir)
    if not ckpt:
        raise FileNotFoundError(f"checkpoint not found for model={args.model}")

    ckpt_cfg = _load_checkpoint_config(ckpt)
    ensemble_manifest = _load_ensemble_manifest(ckpt) if args.model.lower() == "ensemble" else {}
    manifest_schema = ensemble_manifest.get("schema", {}) if isinstance(ensemble_manifest, dict) else {}
    n_pairs_arg = max(1, int(args.n_pairs))
    default_width = int(ckpt_cfg.get("n_features") or 0)
    n_features = int(
        ckpt_cfg.get("n_features")
        or (default_width if default_width > 0 else 0)
        or 0
    )
    if args.model.lower() == "ensemble":
        n_features = int(manifest_schema.get("n_features", n_features) or n_features)
    seq_len = int(ckpt_cfg.get("seq_len", args.seq_len))
    if args.model.lower() == "ensemble":
        seq_len = int(manifest_schema.get("seq_len", seq_len))
    if n_features <= 0:
        raise ValueError(
            f"Checkpoint {ckpt} missing n_features — cannot pad/truncate to a hardcoded width."
        )
    n_pairs, f_per_pair = _pair_feature_dims(n_features, n_pairs=n_pairs_arg, cfg=ckpt_cfg)

    class Cfg:
        def __init__(self):
            self.model = args.model
            self.hidden_size = int(ckpt_cfg.get("hidden_size", 256))
            self.d_model = int(ckpt_cfg.get("d_model", 256))
            self.nhead = int(ckpt_cfg.get("nhead", 8))
            self.num_layers = int(ckpt_cfg.get("num_layers", 3))
            self.dropout = float(ckpt_cfg.get("dropout", 0.1))
            self.seq_len = seq_len
            self.multitask = args.multitask
            self.pair_embed_dim = 16 if n_pairs > 1 else 0
            self.loss = str(ckpt_cfg.get("loss", "cross_entropy"))
            self.corr_window = 20
            self.corr_window_long = 60
            self.momentum_window = 20
            self._n_pairs = n_pairs
            self._f_per_pair = f_per_pair

    if args.model.lower() == "ensemble":
        model = _load_ensemble_model(ckpt, n_features, seq_len, device)
    else:
        model = build_model(args.model, n_features, Cfg()).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(_checkpoint_state_dict(state), strict=False)
        model.eval()

    fe = FeatureEngineer(atr_window=FEATURES.get("atr_window", 14), lag_windows=FEATURES.get("lag_windows", [5, 10, 20]))
    afb = AdvancedFeatureBuilder()
    sent = SentimentPipeline(prefer_backend="finbert", use_cache=True)
    pipeline = ForexDataPipeline(bar_freq=args.bar_freq)

    # Pre-cache dataset per pair
    cached_bars = {}
    cached_features = {}
    pair_list = DEFAULT_PAIRS[:n_pairs] if n_pairs > 1 else [args.pair]
    
    for p in pair_list:
        try:
            ticks = load_or_generate(source=args.source, pair=p, start=args.start, end=args.end, n_rows=2000000)
            bars = pipeline.run(ticks)
            if bars is not None and len(bars) > seq_len:
                cached_bars[p] = bars
                f_base = fe.build(bars)
                f_adv = afb.build(bars, base_features=f_base)
                f = pd.concat([f_base, f_adv], axis=1)
                f = _fit_feature_width(f, f_per_pair)
                f = f.reindex(bars.index).ffill().fillna(0.0)
                cached_features[p] = f
        except Exception as e:
            log(f"Failed to load/cache data for {p}: {e}")

    wf_rows = []
    for idx, (ws, we) in enumerate(windows, start=1):
        log(f"[WF {idx}/{len(windows)}] {ws} -> {we}")
        per_pair = []
        base_bars = None
        for p in pair_list:
            if p not in cached_bars:
                continue
            b = cached_bars[p].loc[ws:we]
            if len(b) <= seq_len:
                log(f"[WF {idx}/{len(windows)}] {p}: no usable bars after cleaning; skipping window")
                continue
            if base_bars is None:
                base_bars = b
            f_slice = cached_features[p].loc[ws:we]
            per_pair.append(f_slice)

        if not per_pair or base_bars is None or len(base_bars) <= seq_len:
            continue

        X = pd.concat(per_pair, axis=1)
        try:
            X["finbert_sentiment"] = float(sent.score_headlines(get_latest_headlines(limit=12) or ["Market update"]))
        except Exception:
            X["finbert_sentiment"] = 0.0
        X = _fit_feature_width(X, n_features)
        if len(X) <= args.seq_len:
            log(f"[WF {idx}/{len(windows)}] not enough feature rows ({len(X)}) for seq_len={args.seq_len}; skipping")
            continue
        x_t = torch.tensor(np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32, device=device)

        with torch.no_grad():
            logits = _batched_logits(model, x_t, seq_len, args.inference_batch_size)
            probs = torch.softmax(logits, dim=-1).numpy()
        cls, conf = probs.argmax(axis=1), probs.max(axis=1)
        meta_ok = None
        meta_labeler = None
        if args.meta_labeling and len(cls) >= 30:
            meta_ok, meta_labeler = _build_meta_labeler_mask(args, base_bars, X, cls, seq_len)
        if len(conf):
            log(
                f"[Signals] confidence min/median/max = "
                f"{float(np.min(conf)):.3f}/{float(np.median(conf)):.3f}/{float(np.max(conf)):.3f} "
                f"| threshold={args.min_confidence:.3f}"
            )
        signals = []
        pip_size = PIP_SIZES.get(args.pair.upper(), 0.0001)
        stop_pips = float(args.stop_pips)
        take_pips = float(args.take_pips)
        last_signal_i = -10**9
        
        # Pre-extract arrays for speed
        regime_vals = X["regime_label"].values if "regime_label" in X.columns else None
        close_vals = base_bars["close"].values
        ts_vals = base_bars.index

        for off, c in enumerate(cls):
            i = seq_len + off
            adj_min_conf = args.min_confidence
            if regime_vals is not None:
                rl = float(regime_vals[i])
                if rl > 0.5:
                    adj_min_conf = max(0.5, args.min_confidence - 0.05)
                elif rl < -0.5:
                    adj_min_conf = min(0.95, args.min_confidence + 0.05)

            if conf[off] < adj_min_conf:
                continue
            if i - last_signal_i < max(1, int(args.min_gap_bars)):
                continue
            if meta_ok is not None and not bool(meta_ok[off]):
                continue
            price = float(close_vals[i])
            if c == 0:
                act = 2
                stop_loss = price + stop_pips * pip_size
                take_profit = price - take_pips * pip_size
            elif c == 2:
                act = 1
                stop_loss = price - stop_pips * pip_size
                take_profit = price + take_pips * pip_size
            else:
                act = 0
                stop_loss = 0.0
                take_profit = 0.0
            if act:
                last_signal_i = i

                # Fractional Kelly Sizing
                try:
                    from config.settings import SIZING
                    kelly_frac = SIZING.get("kelly_fraction", 0.25)
                except ImportError:
                    kelly_frac = 0.25

                win_prob = float(conf[off])
                reward_to_risk = take_pips / max(stop_pips, 1e-5)
                if reward_to_risk > 0:
                    kelly_k = win_prob - ((1.0 - win_prob) / reward_to_risk)
                else:
                    kelly_k = 0.0

                dynamic_lots = args.lots * max(0.0, kelly_k * kelly_frac)
                # Ensure minimum lot size
                dynamic_lots = max(0.01, dynamic_lots)

                signals.append({
                    "timestamp": ts_vals[i],
                    "action": act,
                    "lots": dynamic_lots,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "confidence": win_prob,
                })
        if not signals:
            log(f"[Signals] No trades generated for {ws} -> {we}; lower --min-confidence or inspect class probabilities.")
            _send_discord_backtest({
                "Model": args.model,
                "Pair": args.pair,
                "Window": f"{ws} -> {we}",
                "Checkpoint": ckpt.name,
                "Source": args.source,
                "Seq Len": str(seq_len),
                "Min Confidence": _fmt_metric(args.min_confidence, ".2f"),
                "Stop Pips": _fmt_metric(args.stop_pips, ".1f"),
                "Take Pips": _fmt_metric(args.take_pips, ".1f"),
                "Min Gap Bars": str(args.min_gap_bars),
                "Trades": "0",
                "Status": "No trades generated",
                "Action": "Lower --min-confidence or inspect class probabilities",
            })
            continue

        sig_df = pd.DataFrame(signals).set_index("timestamp")
        eff_slippage, exec_meta = _advanced_execution_overlay(args, base_bars, signals)
        if exec_meta:
            log(f"[Exec] {exec_meta}")
        bt = ForexScalingBacktest(
            bars=base_bars.iloc[seq_len:],
            signals=sig_df,
            initial_equity=args.equity,
            commission_per_lot=args.commission_per_lot,
            slippage_pips=eff_slippage,
            pip_size=pip_size,
            execution_delay_bars=max(1, int(args.execution_delay_bars)),
        )
        bt.run()
        metrics = bt.performance_metrics()
        norm_metrics = _normalize_backtest_metrics(metrics)
        mc = MonteCarloBacktest(n_simulations=args.mc_sims, initial_equity=args.equity).run_from_backtest(bt) if args.mc_sims > 0 and len(bt.trades) >= 5 else None

        rec = {"ts": datetime.now(UTC).isoformat(), "run_id": run_id, "component": "backtest", "event_type": "walkforward_window", "model": args.model, "window_index": idx, "window_start": ws, "window_end": we, "n_trades": norm_metrics["n_trades"], "sharpe": norm_metrics["sharpe"], "win_rate": norm_metrics["win_rate"], "max_drawdown": norm_metrics["max_drawdown"], "net_pnl": norm_metrics["net_pnl"]}
        _append_jsonl(wf_jsonl, rec)
        wf_rows.append(rec)
        _send_discord_backtest({
            "Model": args.model,
            "Pair": args.pair,
            "Window": f"{ws} -> {we}",
            "Checkpoint": ckpt.name,
            "Source": args.source,
            "Seq Len": str(seq_len),
            "N Pairs": str(args.n_pairs),
            "Multitask": str(bool(args.multitask)),
            "Min Confidence": _fmt_metric(args.min_confidence, ".2f"),
            "Stop Pips": _fmt_metric(args.stop_pips, ".1f"),
            "Take Pips": _fmt_metric(args.take_pips, ".1f"),
            "Min Gap Bars": str(args.min_gap_bars),
            "Lots": _fmt_metric(args.lots, ".2f"),
            "Equity": _fmt_metric(args.equity, ",.2f", prefix="$"),
            "Trades": _fmt_metric(norm_metrics["n_trades"], ".0f"),
            "Sharpe": _fmt_metric(norm_metrics["sharpe"], ".4f"),
            "Win Rate": _fmt_metric(norm_metrics["win_rate"], ".2%"),
            "Max Drawdown": _fmt_metric(norm_metrics["max_drawdown"], ".2%"),
            "Net PnL": _fmt_metric(norm_metrics["net_pnl"], ",.2f", prefix="$"),
            "Return": _fmt_metric(norm_metrics["total_return_pct"], ".2f", suffix="%"),
            "Metric Error": norm_metrics["error"] or "none",
            "MC Sims": str(args.mc_sims),
        })

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        bt.get_trade_log().to_csv(out_dir / f"{args.model}_{stamp}_trades.csv")
        bt.results_df.to_csv(out_dir / f"{args.model}_{stamp}_equity.csv")
        (out_dir / f"{args.model}_{stamp}_summary.json").write_text(json.dumps({"model": args.model, "checkpoint": str(ckpt), "start": ws, "end": we, "metrics": metrics, "monte_carlo": mc, "execution": exec_meta}, indent=2, default=str), encoding="utf-8")

    if wf_rows:
        wf_df = pd.DataFrame(wf_rows)
        wf_df.to_csv(out_dir / f"{args.model}_{run_id}_walkforward.csv", index=False)
        medians = {"sharpe": float(wf_df["sharpe"].median()), "win_rate": float(wf_df["win_rate"].median()), "max_drawdown": float(wf_df["max_drawdown"].median()), "net_pnl": float(wf_df["net_pnl"].median())}
        (out_dir / f"{args.model}_{run_id}_walkforward_summary.json").write_text(json.dumps({"model": args.model, "run_id": run_id, "windows": len(wf_rows), "period": {"start": args.start, "end": args.end}, "medians": medians}, indent=2), encoding="utf-8")
        _send_discord_backtest({
            "Model": args.model,
            "Run ID": run_id,
            "Period": f"{args.start} -> {args.end}",
            "Windows": str(len(wf_rows)),
            "Median Sharpe": _fmt_metric(medians["sharpe"], ".4f"),
            "Median Win Rate": _fmt_metric(medians["win_rate"], ".2%"),
            "Median Max DD": _fmt_metric(medians["max_drawdown"], ".2%"),
            "Median Net PnL": _fmt_metric(medians["net_pnl"], ",.2f", prefix="$"),
            "Output": str(out_dir),
        })


def run_execution_backtest(
    model: torch.nn.Module,
    pair_list: list[str],
    start_date: str,
    end_date: str,
    seq_len: int,
    n_features: int,
    device: torch.device,
    bar_freq: str = "1min",
    data_source: str = "dukascopy",
    stop_pips: float = 12.0,
    take_pips: float = 18.0,
    inference_batch_size: int = 512,
    min_confidence: float | None = None,
    min_gap_bars: int = 3,
    lots: float = 0.1,
    equity: float = 10_000.0,
    commission_per_lot: float = 3.5,
    slippage_pips: float = 0.7,
    execution_delay_bars: int = 1,
) -> dict:
    """Programmatic execution-aware backtest for a single window.

    Mirrors `run_backtest` but takes a pre-built model and returns a metrics
    dict consumed by training/train_gpu.py::_evaluate_forward_gate:
    {n_trades, sharpe, profit_factor, max_drawdown, net_pnl,
     signals_df (with pnl_pips), equity_curve, error}
    """
    if min_confidence is None:
        min_confidence = float(BACKTEST.get("min_confidence", 0.45))

    model.eval()
    fe = FeatureEngineer(atr_window=FEATURES.get("atr_window", 14),
                         lag_windows=FEATURES.get("lag_windows", [5, 10, 20]))
    afb = AdvancedFeatureBuilder()
    pipeline = ForexDataPipeline(bar_freq=bar_freq)

    def _empty(reason: str) -> dict:
        return {
            "n_trades": 0,
            "sharpe": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "net_pnl": 0.0,
            "signals_df": pd.DataFrame(),
            "equity_curve": [float(equity)],
            "error": reason,
        }

    base_bars = None
    per_pair = []
    for p in pair_list:
        try:
            ticks = load_or_generate(source=data_source, pair=p, start=start_date,
                                     end=end_date, n_rows=100_000)
        except Exception as exc:
            log(f"[Gate] {p}: data load failed ({exc}); skipping pair")
            continue
        bars = pipeline.run(ticks)
        if bars is None or len(bars) <= seq_len:
            log(f"[Gate] {p}: no usable bars after cleaning; skipping pair")
            continue
        if base_bars is None:
            base_bars = bars
        f_base = fe.build(bars)
        f = pd.concat([f_base, afb.build(bars, base_features=f_base)], axis=1)
        f_per_pair = max(1, int(n_features) // max(1, len(pair_list)))
        f = _fit_feature_width(f, f_per_pair)
        f = f.reindex(base_bars.index).ffill().fillna(0.0)
        per_pair.append(f)

    if not per_pair or base_bars is None or len(base_bars) <= seq_len:
        return _empty("no usable data for any pair")

    X = pd.concat(per_pair, axis=1)
    try:
        X["finbert_sentiment"] = float(SentimentPipeline(prefer_backend="finbert", use_cache=True)
                                       .score_headlines(get_latest_headlines(limit=12) or ["Market update"]))
    except Exception:
        X["finbert_sentiment"] = 0.0
    if X.shape[1] != n_features:
        if X.shape[1] > n_features:
            X = X.iloc[:, :n_features]
        else:
            X = pd.concat(
                [
                    X,
                    pd.DataFrame(
                        0.0,
                        index=X.index,
                        columns=[f"xpad_{i}" for i in range(n_features - X.shape[1])],
                    ),
                ],
                axis=1,
            )
    if len(X) <= seq_len:
        return _empty("not enough feature rows for seq_len")

    x_t = torch.tensor(np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0),
                       dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = _batched_logits(model, x_t, seq_len, inference_batch_size)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    cls = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    pip_size = PIP_SIZES.get(str(pair_list[0]).upper(), 0.0001)

    signals = []
    last_signal_i = -10**9
    
    close_vals = base_bars["close"].values
    ts_vals = base_bars.index

    for off, c in enumerate(cls):
        i = seq_len + off
        if conf[off] < min_confidence:
            continue
        if i - last_signal_i < max(1, int(min_gap_bars)):
            continue
        price = float(close_vals[i])
        if c == 0:
            act = 2
            stop_loss = price + stop_pips * pip_size
            take_profit = price - take_pips * pip_size
        elif c == 2:
            act = 1
            stop_loss = price - stop_pips * pip_size
            take_profit = price + take_pips * pip_size
        else:
            act = 0
            stop_loss = 0.0
            take_profit = 0.0
        if not act:
            continue
        last_signal_i = i

        try:
            from config.settings import SIZING
            kelly_frac = float(SIZING.get("kelly_fraction", 0.25))
        except ImportError:
            kelly_frac = 0.25

        win_prob = float(conf[off])
        reward_to_risk = take_pips / max(stop_pips, 1e-5)
        kelly_k = win_prob - ((1.0 - win_prob) / reward_to_risk) if reward_to_risk > 0 else 0.0
        dynamic_lots = max(0.01, lots * max(0.0, kelly_k * kelly_frac))

        signals.append({
            "timestamp": ts_vals[i],
            "action": act,
            "lots": dynamic_lots,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence": win_prob,
        })

    if not signals:
        return _empty("no trades generated at min_confidence")

    sig_df = pd.DataFrame(signals).set_index("timestamp")
    bt = ForexScalingBacktest(
        bars=base_bars.iloc[seq_len:],
        signals=sig_df,
        initial_equity=equity,
        commission_per_lot=commission_per_lot,
        slippage_pips=slippage_pips,
        pip_size=pip_size,
        execution_delay_bars=max(1, int(execution_delay_bars)),
    )
    bt.run()
    metrics = _normalize_backtest_metrics(bt.performance_metrics())
    metrics.update({
        "profit_factor": float(bt.performance_metrics().get("profit_factor", 0.0) or 0.0),
        "signals_df": bt.get_trade_log(),
        "equity_curve": [float(v) for v in bt.equity_curve] or [float(equity)],
        "confidence_scores": [float(s["confidence"]) for s in signals],
    })
    return metrics


if __name__ == "__main__":
    run_backtest()
