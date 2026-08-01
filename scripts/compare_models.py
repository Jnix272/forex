import sys
import json
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timezone
# Add project root to path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.train_gpu import build_model
from models.ensemble import EnsembleMetaLearner
from models.xgboost_model import XGBoostForecaster
from backtesting.backtest import ForexScalingBacktest


def _fold_stability(model_name: str, ckpt_path: str) -> dict:
    ckpt_dir = Path(ckpt_path).parent
    sharpes = []
    losses = []
    train_val_gaps = []
    for cfg_path in sorted(ckpt_dir.glob(f"{model_name}_fold*_config.json")):
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if cfg.get("best_val_sharpe_proxy") is not None:
            sharpes.append(float(cfg["best_val_sharpe_proxy"]))
        if cfg.get("best_val_loss") is not None:
            losses.append(float(cfg["best_val_loss"]))
        if cfg.get("train_val_loss_gap") is not None:
            train_val_gaps.append(float(cfg["train_val_loss_gap"]))
        elif cfg.get("best_val_loss") is not None and cfg.get("best_train_loss") is not None:
            train_val_gaps.append(float(cfg["best_val_loss"]) - float(cfg["best_train_loss"]))
    out = {
        "fold_count": len(sharpes) or len(losses),
        "fold_sharpe_mean": 0.0,
        "fold_sharpe_std": 0.0,
        "fold_sharpe_min": 0.0,
        "fold_sharpe_worst": 0.0,
        "fold_pass_rate": 0.0,
        "fold_governance_threshold": 0.0,
        "fold_implausibly_smooth": False,
        "train_val_loss_gap_mean": 0.0,
        "train_val_loss_gap_max": 0.0,
        "fold_loss_mean": 0.0,
        "fold_loss_std": 0.0,
    }
    if sharpes:
        arr = np.asarray(sharpes, dtype=np.float64)
        pass_threshold = 0.0
        out.update({
            "fold_sharpe_mean": float(arr.mean()),
            "fold_sharpe_std": float(arr.std()),
            "fold_sharpe_min": float(arr.min()),
            "fold_sharpe_worst": float(arr.min()),
            "fold_pass_rate": float((arr >= pass_threshold).mean()),
            "fold_governance_threshold": pass_threshold,
            "fold_implausibly_smooth": bool(len(arr) >= 3 and float(arr.std()) < 0.02),
        })
    if losses:
        arr = np.asarray(losses, dtype=np.float64)
        out.update({
            "fold_loss_mean": float(arr.mean()),
            "fold_loss_std": float(arr.std()),
        })
    if train_val_gaps:
        arr = np.asarray(train_val_gaps, dtype=np.float64)
        out.update({
            "train_val_loss_gap_mean": float(arr.mean()),
            "train_val_loss_gap_max": float(arr.max()),
        })
    return out

def _checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            state = checkpoint.get(key)
            if isinstance(state, dict):
                return state
    return checkpoint


def _model_metadata(model_name: str, ckpt_path: str) -> dict:
    path = Path(ckpt_path)
    candidates = [
        path.with_suffix(path.suffix + ".json"),
        path.with_name(path.name.replace(".pt", "").replace("_best", "_config") + ".json"),
        path.parent / f"{model_name}_fold0_config.json",
    ]
    meta = {}
    for p in candidates:
        if p.exists():
            try:
                meta.update(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
    return meta


def _param_count(model, model_name: str) -> int:
    if str(model_name).lower() in {"xgboost", "xgb"}:
        try:
            booster = model.model.get_booster()
            return int(len(booster.get_dump()))
        except Exception:
            return 0
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        return 0

def run_evaluation(model_name, ckpt_path, bars, X_tensor, seq_len=60, batch_size=512, min_confidence=0.60, min_gap_bars=3):
    print(f"Evaluating {model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_key = str(model_name).lower()
    
    # Load config to get architecture details
    cfg_path = Path(ckpt_path).with_name(Path(ckpt_path).name.replace(".pt", "").replace("_best", "_config") + ".json")
    if not cfg_path.exists():
        cfg_path = Path(ckpt_path).parent / f"{model_name}_fold0_config.json"
    
    cfg = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        
    n_features = int(cfg.get("n_features", X_tensor.shape[-1]))
    
    class ModelConfig:
        def __init__(self):
            self.model = model_name
            self.hidden_size = int(cfg.get("hidden_size", 256))
            self.d_model = int(cfg.get("d_model", 256))
            self.nhead = int(cfg.get("nhead", 8))
            self.num_layers = int(cfg.get("num_layers", 3))
            self.dropout = float(cfg.get("dropout", 0.1))
            self.seq_len = seq_len
            self.multitask = True
            self.pair_embed_dim = 16
            self.loss = str(cfg.get("loss", "cross_entropy"))
            self.corr_window = 20
            self.corr_window_long = 60
            self.momentum_window = 20
            self._n_pairs = 10
            self._f_per_pair = 224

    try:
        if model_key in {"xgboost", "xgb"}:
            model = XGBoostForecaster(num_classes=1, flatten_sequence=False, seq_len=seq_len)
            model.load_model(str(ckpt_path))
        elif model_key == "ensemble":
            sidecar = Path(ckpt_path).with_suffix(Path(ckpt_path).suffix + ".json")
            meta = {}
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8")).get("meta", {})
                except Exception:
                    meta = {}
            base_names = list(meta.get("base_names", ["haelt", "mamba", "gnn"]))
            bases = []
            for bn in base_names:
                bck = _ROOT / "checkpoints" / bn / f"{bn}_best.pt"
                if not bck.exists():
                    raise FileNotFoundError(f"Missing base checkpoint: {bck}")
                bmodel = build_model(bn, n_features, ModelConfig()).to(device)
                bstate = torch.load(bck, map_location=device, weights_only=False)
                bmodel.load_state_dict(_checkpoint_state_dict(bstate), strict=False)
                bmodel.eval()
                bases.append(bmodel)
            model = EnsembleMetaLearner(bases, context_dim=32, hidden=64, base_names=base_names).to(device)
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(_checkpoint_state_dict(checkpoint), strict=False)
            model.eval()
        else:
            model = build_model(model_name, n_features, ModelConfig()).to(device)
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(_checkpoint_state_dict(checkpoint), strict=False)
            model.eval()
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None

    signals = []
    all_conf = []
    trade_conf = []
    n_windows = 0
    infer_t0 = time.perf_counter()
    with torch.no_grad():
        starts = range(0, len(X_tensor), max(1, batch_size))
        for start in starts:
            end = min(len(X_tensor), start + batch_size)
            windows = X_tensor[start:end]
            n_windows += int(end - start)
            out = model(windows)
            if model_key == "ensemble":
                pred = out[0] if isinstance(out, (tuple, list)) else out
                logits = torch.stack([-pred, torch.zeros_like(pred), pred], dim=1)
            elif model_key in {"xgboost", "xgb"}:
                if isinstance(out, torch.Tensor):
                    pred = out.detach().float().flatten()
                else:
                    pred = torch.tensor(np.asarray(out), dtype=torch.float32, device=device).flatten()
                logits = torch.stack([-pred, torch.zeros_like(pred), pred], dim=1)
            elif isinstance(out, dict):
                logits = out.get("direction_logits", out.get("logits", out.get("direction")))
            elif isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                logits = out
                
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            cls_idx = probs.argmax(axis=1)
            conf = probs.max(axis=1)
            all_conf.extend(conf.astype(float).tolist())
            
            last_trade_i = -10**9
            for offset, c in enumerate(cls_idx):
                i = start + offset
                price = float(bars.close.iloc[i])
                if float(conf[offset]) < float(min_confidence):
                    continue
                if (i - last_trade_i) < int(min_gap_bars):
                    continue
                action = 0
                if c == 0:
                    action = 2
                    sl = price + 0.0015
                    tp = price - 0.0030
                elif c == 2:
                    action = 1
                    sl = price - 0.0015
                    tp = price + 0.0030
                
                if action != 0:
                    last_trade_i = i
                    trade_conf.append(float(conf[offset]))
                    signals.append({
                        "timestamp": bars.index[i],
                        "action": action,
                        "lots": 0.1,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "confidence": conf[offset]
                    })
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_s = max(time.perf_counter() - infer_t0, 1e-9)

    if not signals:
        return None

    sig_df = pd.DataFrame(signals).set_index("timestamp")
    bt = ForexScalingBacktest(bars.iloc[seq_len:], sig_df, initial_equity=10000.0)
    res = bt.run()
    raw_metrics = bt.performance_metrics() or {}
    metrics = {
        "sharpe": float(raw_metrics.get("sharpe", raw_metrics.get("sharpe_ratio", 0.0)) or 0.0),
        "net_pnl": float(raw_metrics.get("net_pnl", 0.0) or 0.0),
        "win_rate": float(raw_metrics.get("win_rate", raw_metrics.get("win_rate_pct", 0.0)) or 0.0),
        "n_trades": int(raw_metrics.get("n_trades", 0) or 0),
        "max_drawdown": float(raw_metrics.get("max_drawdown", raw_metrics.get("max_drawdown_pct", 0.0)) or 0.0),
        "total_return_pct": float(raw_metrics.get("total_return_pct", 0.0) or 0.0),
    }
    metrics.update({
        "latency_ms_per_window": float(infer_s * 1000.0 / max(n_windows, 1)),
        "windows_per_second": float(n_windows / infer_s),
        "turnover_per_1000_bars": float(metrics["n_trades"] * 1000.0 / max(len(bars), 1)),
        "return_per_trade_pct": float(metrics["total_return_pct"] / max(metrics["n_trades"], 1)),
        "sharpe_per_turnover": float(metrics["sharpe"] / max(metrics["n_trades"] / max(len(bars), 1), 1e-9)),
    })
    conf_arr = np.asarray(all_conf, dtype=np.float64) if all_conf else np.asarray([0.0])
    trade_conf_arr = np.asarray(trade_conf, dtype=np.float64) if trade_conf else np.asarray([0.0])
    meta = _model_metadata(model_name, ckpt_path)
    metrics.update({
        "params": _param_count(model, model_name),
        "train_time_s": float(meta.get("train_time_s", meta.get("train_seconds", 0.0)) or 0.0),
        "validation_sharpe": float(meta.get("best_val_sharpe_proxy", meta.get("validation_sharpe", 0.0)) or 0.0),
        "validation_loss": float(meta.get("best_val_loss", meta.get("validation_loss", 0.0)) or 0.0),
        "confidence_mean": float(conf_arr.mean()),
        "confidence_std": float(conf_arr.std()),
        "confidence_p10": float(np.quantile(conf_arr, 0.10)),
        "confidence_p90": float(np.quantile(conf_arr, 0.90)),
        "trade_confidence_mean": float(trade_conf_arr.mean()),
        "trade_confidence_std": float(trade_conf_arr.std()),
    })
    metrics.update(_fold_stability(model_name, ckpt_path))

    # Sanity guards for invalid combinations.
    flags = []
    if metrics["max_drawdown"] > 100.0:
        flags.append("max_drawdown_gt_100")
    if metrics["total_return_pct"] < 0 and metrics["sharpe"] > 1.0:
        flags.append("negative_return_with_high_sharpe")
    if metrics["n_trades"] <= 0:
        flags.append("no_trades")
    if metrics["fold_count"] >= 2 and metrics["fold_sharpe_std"] > 0.5:
        flags.append("unstable_across_folds")
    if metrics.get("fold_count", 0) >= 3 and metrics.get("fold_implausibly_smooth", False):
        flags.append("implausibly_smooth_folds")
    if metrics.get("fold_count", 0) and metrics.get("fold_pass_rate", 1.0) < 0.60:
        flags.append("low_fold_pass_rate")
    if metrics.get("train_val_loss_gap_max", 0.0) > 0.15:
        flags.append("large_train_val_gap")
    if metrics["latency_ms_per_window"] > 10.0:
        flags.append("slow_live_inference")
    metrics["sanity_flags"] = flags

    return {"results": res, "metrics": metrics, "equity": bt.get_equity_curve()}

if __name__ == "__main__":
    import zarr

    ZARR_PATH = _ROOT / "data" / "processed" / \
        "dataset_AUDUSD-EURGBP-EURJPY-EURUSD-GBPJPY-GBPUSD-NZDUSD-USDCAD-USDCHF-USDJPY_20000000_dukascopy_60_rl_reward.zarr"
    SCALER_PATH = _ROOT / "data" / "processed" / \
        "dataset_AUDUSD-EURGBP-EURJPY-EURUSD-GBPJPY-GBPUSD-NZDUSD-USDCAD-USDCHF-USDJPY_20000000_dukascopy_60_rl_reward_scaler.npz"

    print(f"[Compare] Loading cached Zarr dataset from:\n  {ZARR_PATH}")
    z = zarr.open(str(ZARR_PATH), mode="r")
    
    num_samples = min(20_000, int(z["X"].shape[0]))
    print(f"[Compare] Loading last {num_samples:,} already-scaled cached windows...")
    X_arr = np.array(z["X"][-num_samples:], dtype=np.float32)
    close_prices = np.array(z["close"][-num_samples:]) if "close" in z else None

    # X in the training cache is already scaled. Do not apply *_scaler.npz again.
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Build a synthetic bars DataFrame for the backtester
    idx = pd.date_range("2023-01-01", periods=len(X_arr), freq="1min", tz="UTC")
    if close_prices is not None:
        base_bars = pd.DataFrame({"close": close_prices[:, 0] if close_prices.ndim > 1 else close_prices}, index=idx[:len(X_arr)])
    else:
        base_bars = pd.DataFrame({"close": np.random.uniform(1.08, 1.10, len(X_arr))}, index=idx[:len(X_arr)])
    base_bars["open"]   = base_bars["close"].shift(1).bfill()
    base_bars["high"]   = base_bars["close"] * 1.0002
    base_bars["low"]    = base_bars["close"] * 0.9998
    base_bars["volume"] = 1000.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X_arr, dtype=torch.float32, device=device)

    print(f"[Compare] Dataset loaded: {X_arr.shape[0]:,} bars × {X_arr.shape[1]} features | device={device}")

    models_to_test = {
        "xgboost": _ROOT / "checkpoints" / "xgboost_best.json",
        "haelt": _ROOT / "checkpoints" / "haelt" / "haelt_best.pt",
        "mamba": _ROOT / "checkpoints" / "mamba" / "mamba_best.pt",
        "transformer": _ROOT / "checkpoints" / "transformer" / "transformer_best.pt",
        "ensemble": _ROOT / "checkpoints" / "ensemble" / "ensemble_meta_best.pt",
        "tft":   _ROOT / "checkpoints" / "tft"   / "tft_fold0_best.pt",
        "gnn":   _ROOT / "checkpoints" / "gnn"   / "gnn_best.pt",
    }

    results = {}
    compare_min_conf = 0.60
    compare_min_gap = 3
    print(f"[Compare] Realism config | min_confidence={compare_min_conf:.2f} | min_gap_bars={compare_min_gap}")
    for name, path in models_to_test.items():
        if Path(path).exists():
            print(f"\n[Compare] ── {name.upper()} ──────────────────────────")
            try:
                res = run_evaluation(
                    name,
                    str(path),
                    base_bars,
                    X_tensor,
                    min_confidence=compare_min_conf,
                    min_gap_bars=compare_min_gap,
                )
                if res:
                    results[name] = res
                    m = res["metrics"]
                    print(f"  Sharpe: {m.get('sharpe', 0):.2f} | "
                          f"Return: {m.get('total_return_pct', 0):.2f}% | "
                          f"Ret/Trade: {m.get('return_per_trade_pct', 0):.4f}% | "
                          f"Latency: {m.get('latency_ms_per_window', 0):.3f} ms/window | "
                          f"Win Rate: {m.get('win_rate', 0):.1f}% | "
                          f"Trades: {m.get('n_trades', 0)}")
                    if m.get("sanity_flags"):
                        print(f"  [Sanity] flags={m['sanity_flags']}")
            except Exception as e:
                import traceback
                print(f"  ✗ Failed: {e}")
                traceback.print_exc()
        else:
            print(f"[Compare] ✗ {name.upper()} checkpoint not found: {path}")

    if not results:
        print("\n[Compare] No models produced results. Check your checkpoints.")
        sys.exit(1)

    # ── Plot equity curves ────────────────────────────────────────────────
    plt.figure(figsize=(12, 6))
    for name, data in results.items():
        eq = data["equity"]
        sharpe = data["metrics"].get("sharpe", 0)
        ret    = data["metrics"].get("total_return_pct", 0)
        plt.plot(eq.index, eq.values, label=f"{name.upper()} | Sharpe={sharpe:.2f} | Ret={ret:.1f}%")

    out_dir = _ROOT / "compare_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.title("Model Backtest Comparison (Cached Data)")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = out_dir / "model_comparison.png"
    plt.savefig(plot_path)
    print(f"\n[Compare] Saved plot → {plot_path}")

    # ── Save metrics ─────────────────────────────────────────────────────
    metrics_summary = {k: v["metrics"] for k, v in results.items()}
    baseline = metrics_summary.get("xgboost")
    comparison_report = {
        "generated_at": datetime.now(timezone.utc).isoformat() + "Z",
        "baseline": "xgboost",
        "must_beat_baseline": True,
        "promotion_rule": (
            "Deep models must beat XGBoost on backtest Sharpe without worse max drawdown, "
            "and should justify added latency/complexity with fold stability or role-specific value."
        ),
        "models": {},
    }
    for name, metrics in metrics_summary.items():
        beats = None
        if baseline and name != "xgboost":
            beats = (
                metrics.get("sharpe", 0.0) > baseline.get("sharpe", 0.0)
                and metrics.get("max_drawdown", 0.0) <= baseline.get("max_drawdown", float("inf"))
            )
        comparison_report["models"][name] = {
            "params": metrics.get("params", 0),
            "train_time_s": metrics.get("train_time_s", 0.0),
            "validation_sharpe": metrics.get("validation_sharpe", 0.0),
            "backtest_sharpe": metrics.get("sharpe", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "latency_ms_per_window": metrics.get("latency_ms_per_window", 0.0),
            "confidence_mean": metrics.get("confidence_mean", 0.0),
            "confidence_std": metrics.get("confidence_std", 0.0),
            "confidence_p10": metrics.get("confidence_p10", 0.0),
            "confidence_p90": metrics.get("confidence_p90", 0.0),
            "trade_confidence_mean": metrics.get("trade_confidence_mean", 0.0),
            "fold_sharpe_mean": metrics.get("fold_sharpe_mean", 0.0),
            "fold_sharpe_std": metrics.get("fold_sharpe_std", 0.0),
            "fold_sharpe_worst": metrics.get("fold_sharpe_worst", 0.0),
            "fold_pass_rate": metrics.get("fold_pass_rate", 0.0),
            "train_val_loss_gap_mean": metrics.get("train_val_loss_gap_mean", 0.0),
            "train_val_loss_gap_max": metrics.get("train_val_loss_gap_max", 0.0),
            "fold_implausibly_smooth": metrics.get("fold_implausibly_smooth", False),
            "beats_xgboost_baseline": beats,
            "sanity_flags": metrics.get("sanity_flags", []),
        }
    metrics_path = out_dir / "model_comparison_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[Compare] Saved metrics -> {metrics_path}")
    report_path = out_dir / "model_comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)
    print(f"[Compare] Saved report -> {report_path}")

    print("\n" + "-" * 146)
    print(f"{'Model':<12} {'Params':>10} {'ValSh':>8} {'BtSh':>8} {'MaxDD%':>8} {'Ret/Tr%':>9} {'T/1k':>7} {'Lat ms':>8} {'ConfMu':>8} {'ConfSd':>8} {'FoldStd':>8} {'Trades':>8}")
    print("-" * 146)
    for name, data in results.items():
        m = data["metrics"]
        print(f"{name.upper():<12} "
              f"{m.get('params', 0):>10} "
              f"{m.get('validation_sharpe', 0):>8.2f} "
              f"{m.get('sharpe', 0):>8.2f} "
              f"{m.get('max_drawdown', 0):>8.2f} "
              f"{m.get('return_per_trade_pct', 0):>9.4f} "
              f"{m.get('turnover_per_1000_bars', 0):>7.2f} "
              f"{m.get('latency_ms_per_window', 0):>8.3f} "
              f"{m.get('confidence_mean', 0):>8.3f} "
              f"{m.get('confidence_std', 0):>8.3f} "
              f"{m.get('fold_sharpe_std', 0):>8.3f} "
              f"{m.get('n_trades', 0):>8}")
    print("-" * 146)
    print("[Compare] Promote only models that add a distinct role: baseline, production, fast, or cross-asset specialist.")
    print("[Compare] Done!")
    sys.exit(0)
    print(f"[Compare] Saved metrics → {metrics_path}")

    # ── Print summary table ───────────────────────────────────────────────
    print("\n" + "─" * 65)
    print(f"{'Model':<10} {'Sharpe':>8} {'Return%':>9} {'WinRate%':>10} {'Trades':>8} {'MaxDD%':>8}")
    print("─" * 65)
    for name, data in results.items():
        m = data["metrics"]
        print(f"{name.upper():<10} "
              f"{m.get('sharpe', 0):>8.2f} "
              f"{m.get('total_return_pct', 0):>9.2f} "
              f"{m.get('win_rate', 0):>10.1f} "
              f"{m.get('n_trades', 0):>8} "
              f"{m.get('max_drawdown', 0):>8.2f}")
    print("─" * 65)
    print("[Compare] Done!")
