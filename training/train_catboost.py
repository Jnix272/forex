"""
training/train_catboost.py
==========================
Trains an CatBoost baseline model on the same processed data chunks
as the deep learning pipeline.

Improvements over v1
---------------------
* --task regression | classification   (3-class direction prediction)
* --folds N                            Walk-forward cross-validation
* --tune                               20-trial hyperparameter grid search
* Real annualised Sharpe written to sidecar (validation_sharpe != 0.0)
* sequence_mode="temporal" feeds 6xF summary stats instead of 1 last bar
"""

import os
import sys
import json
import argparse
import itertools
import time
from math import sqrt
from pathlib import Path
import numpy as np
import catboost as cb
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.catboost_model import CatBoostForecaster
from config.settings import PATHS

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_yaml_config(path: str | None) -> dict:
    if not path:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to use --config") from exc
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _config_get(cfg: dict, dotted: str, default=None):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data_from_store(
    data_dir: Path,
    max_samples: int = 500_000,
    cache_path: str | None = None,
    expected_start: str | None = None,
    expected_end: str | None = None,
):
    """
    Reads processed features and labels from the latest Zarr store.
    Returns (X, y_dir, y_ret) where
      X       : (N, T, F) float32  — full sequence
      y_dir   : (N,)      int8     — direction class 0/1/2
      y_ret   : (N,)      float32  — continuous return
    """
    import zarr

    if cache_path:
        zarr_path = Path(cache_path)
        if not zarr_path.exists():
            print(f"Configured Zarr cache not found: {zarr_path}")
            return None, None, None
    else:
        zarr_stores = list(data_dir.glob("*.zarr"))
        if expected_start and expected_end:
            window_tag = f"{expected_start}_{expected_end}"
            zarr_stores = [p for p in zarr_stores if window_tag in p.name]
            if not zarr_stores:
                print(f"No Zarr cache found in {data_dir} for window {window_tag}.")
                return None, None, None
        if not zarr_stores:
            print(f"No Zarr data found in {data_dir}. Run data ingestion first.")
            return None, None, None
        zarr_stores.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        zarr_path = zarr_stores[0]

    print(f"Loading data from {zarr_path}...")
    z = zarr.open(str(zarr_path), mode="r")

    total_samples = int(z["X"].shape[0])
    n_samples = min(total_samples, max_samples)

    X = np.array(z["X"][-n_samples:], dtype=np.float32)          # (N, T, F)
    y_all = np.array(z["y"][-n_samples:], dtype=np.float32)       # (N,) or (N, K)

    if y_all.ndim == 2 and y_all.shape[1] >= 2:
        y_dir = y_all[:, 0].astype(np.int8)    # direction class  (col 0)
        y_ret = y_all[:, 1].astype(np.float32) # continuous return (col 1)
    else:
        y_dir = np.zeros(n_samples, dtype=np.int8)
        y_ret = y_all.ravel().astype(np.float32)

    return X, y_dir, y_ret

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

BARS_PER_DAY = 24 * 12   # 5-minute bars; adjust if needed

def compute_sharpe(pred_dir: np.ndarray, y_ret: np.ndarray, bars_per_day: int = BARS_PER_DAY) -> float:
    """
    Annualised Sharpe from directional predictions + return targets.

    pred_dir : predicted class (0=Short, 1=Flat, 2=Long) or probabilities
    y_ret    : true continuous returns per bar
    """
    if pred_dir.ndim == 2:
        # Probability array -> argmax
        pred_dir = pred_dir.argmax(axis=1)
    # Map 0->-1, 1->0, 2->+1
    signals = np.where(pred_dir == 2, 1.0, np.where(pred_dir == 0, -1.0, 0.0))
    pnl = signals * y_ret
    if pnl.std() < 1e-12:
        return 0.0
    ann_factor = sqrt(bars_per_day * 252)
    return float(pnl.mean() / pnl.std() * ann_factor)


def compute_dir_accuracy(pred_dir: np.ndarray, y_dir: np.ndarray) -> float:
    if pred_dir.ndim == 2:
        pred_dir = pred_dir.argmax(axis=1)
    return float((pred_dir == y_dir).mean())

# ---------------------------------------------------------------------------
# Walk-forward folds
# ---------------------------------------------------------------------------

def walk_forward_splits(n: int, folds: int = 5):
    """
    Yields (train_idx, val_idx) for expanding-window walk-forward CV.
    First train window covers 60% of data; each subsequent fold adds
    roughly (40% / folds) more training data.
    """
    base = int(n * 0.60)
    step = max(1, int(n * 0.40 / folds))
    for i in range(folds):
        train_end = base + i * step
        val_end   = min(train_end + step, n)
        if train_end >= val_end:
            break
        yield np.arange(0, train_end), np.arange(train_end, val_end)

# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

TUNE_GRID = {
    "n_estimators":      [200, 500],
    "max_depth":         [4, 6, 8],
    "subsample":         [0.7, 0.9],
    "colsample_bytree":  [0.6, 0.8],
}

def tune_hyperparams(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray,   y_val: np.ndarray,
                     task: str, lr: float, n_trials: int = 20) -> dict:
    """Cheap grid search; returns best CatBoost param dict."""
    keys   = list(TUNE_GRID.keys())
    values = list(TUNE_GRID.values())
    combos = list(itertools.product(*values))
    # Randomly sample up to n_trials
    rng = np.random.default_rng(42)
    idx = rng.choice(len(combos), size=min(n_trials, len(combos)), replace=False)
    combos = [combos[i] for i in idx]

    best_score = -float("inf")
    best_params = {}

    print(f"[Tune] Searching {len(combos)} hyperparameter combinations...")
    for combo in combos:
        params = dict(zip(keys, combo))
        if task == "classification":
            m = cb.CatBoostClassifier(
                classes_count=3, loss_function="MultiClass",
                learning_rate=lr,
                eval_metric="MultiClass", verbose=0, task_type="GPU",
                **params
            )
            m.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            preds = m.predict(X_val)
            score = compute_dir_accuracy(preds, y_val)
        else:
            m = cb.CatBoostRegressor(
                loss_function="RMSE",
                learning_rate=lr, eval_metric="RMSE", verbose=0, task_type="GPU",
                **params
            )
            m.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            preds = m.predict(X_val)
            mse = float(np.mean((preds - y_val) ** 2))
            score = -mse  # higher = better

        if score > best_score:
            best_score = score
            best_params = params
            print(f"  [Tune] New best ({score:.4f}): {params}")

    print(f"[Tune] Best params: {best_params}  score={best_score:.4f}")
    return best_params

# ---------------------------------------------------------------------------
# Feature label generation
# ---------------------------------------------------------------------------

_TEMPORAL_STAT_NAMES = ["mean", "std", "min", "max", "last", "range"]

def _build_feature_labels(n_raw_features: int, sequence_mode: str, n_tab_features: int) -> list[str]:
    """
    Generate human-readable labels for tabular CatBoost features.
    When sequence_mode='temporal', each raw feature gets 6 summary stats.
    Falls back to f0, f1, ... if the count doesn't match.
    """
    if sequence_mode == "temporal":
        labels = []
        for stat in _TEMPORAL_STAT_NAMES:
            for fi in range(n_raw_features):
                labels.append(f"f{fi}_{stat}")
        if len(labels) == n_tab_features:
            return labels
    elif sequence_mode == "last_bar":
        labels = [f"f{i}" for i in range(n_raw_features)]
        if len(labels) == n_tab_features:
            return labels
    return [f"f{i}" for i in range(n_tab_features)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--cache-path", type=str, default=None)
    parser.add_argument("--samples", type=int, default=500_000)
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample", type=float, default=0.8)
    parser.add_argument(
        "--task", default="classification",
        choices=["regression", "classification"],
        help="classification: predict direction (3-class); regression: predict return",
    )
    parser.add_argument(
        "--sequence-mode", default="temporal",
        choices=["temporal", "last_bar", "flatten"],
        help="How to convert (B, T, F) sequence to tabular CatBoost input.",
    )
    parser.add_argument("--folds", type=int, default=5,
                        help="Walk-forward CV folds (0 = single 80/20 split)")
    parser.add_argument("--tune", action="store_true",
                        help="Grid-search hyperparameters before final training")
    parser.add_argument("--tune-trials", type=int, default=20)
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic data for testing")
    args = parser.parse_args()

    cfg = _load_yaml_config(args.config)
    cb_cfg = cfg.get("catboost", {}) or {}
    data_dir = Path(args.data_dir or _config_get(cfg, "paths.data_cache", PATHS["data_processed"]))
    pairs      = _config_get(cfg, "data.pairs", None) or [_config_get(cfg, "data.pair", "EURUSD")]
    data_start = _config_get(cfg, "data.start", None)
    data_end   = _config_get(cfg, "data.end",   None)
    news_file  = _config_get(cfg, "news.historical_news_file", None)

    # Merge YAML catboost: defaults into CLI args (CLI takes precedence)
    _cb_defaults = {
        "estimators": int(cb_cfg.get("n_estimators", args.estimators)),
        "depth":      int(cb_cfg.get("max_depth", args.depth)),
        "lr":         float(cb_cfg.get("learning_rate", args.lr)),
        "subsample":  float(cb_cfg.get("subsample", args.subsample)),
        "colsample":  float(cb_cfg.get("colsample_bytree", args.colsample)),
        "task":       str(cb_cfg.get("task", args.task)),
        "sequence_mode": str(cb_cfg.get("sequence_mode", args.sequence_mode)),
        "folds":      int(cb_cfg.get("folds", args.folds)),
        "samples":    int(cb_cfg.get("max_samples", args.samples)),
    }
    for _k, _v in _cb_defaults.items():
        if not any(f"--{_k}" in a or f"--{_k.replace('_', '-')}" in a for a in sys.argv[1:]):
            setattr(args, _k, _v)
    if cb_cfg.get("tune") and not args.tune:
        args.tune = True
    if cb_cfg.get("tune_trials") and "--tune-trials" not in " ".join(sys.argv):
        args.tune_trials = int(cb_cfg["tune_trials"])

    # Extra regularization params from env vars (set by train_gpu.py pipeline)
    _env_min_child_weight = float(os.environ.get("CB_MIN_CHILD_WEIGHT", cb_cfg.get("min_child_weight", 3)))
    _env_gamma = float(os.environ.get("CB_GAMMA", cb_cfg.get("gamma", 0.1)))
    _env_reg_alpha = float(os.environ.get("CB_REG_ALPHA", cb_cfg.get("reg_alpha", 0.05)))
    _env_reg_lambda = float(os.environ.get("CB_REG_LAMBDA", cb_cfg.get("reg_lambda", 1.0)))
    _do_feature_importance = os.environ.get("CB_FEATURE_IMPORTANCE", "1" if cb_cfg.get("feature_importance", True) else "0") == "1"
    _fi_top_n = int(os.environ.get("CB_FEATURE_IMPORTANCE_TOP_N", cb_cfg.get("feature_importance_top_n", 50)))

    # ── load data ─────────────────────────────────────────────────────────────
    if args.demo:
        print("Running in DEMO mode with synthetic data.")
        N, T, F = 2000, 60, 50
        X    = np.random.randn(N, T, F).astype(np.float32)
        y_dir = np.random.randint(0, 3, N).astype(np.int8)
        y_ret = np.random.randn(N).astype(np.float32) * 0.001
    else:
        X, y_dir, y_ret = load_data_from_store(
            data_dir, args.samples, args.cache_path,
            expected_start=data_start if args.config else None,
            expected_end=data_end   if args.config else None,
        )
        if X is None:
            sys.exit(1)

    N = len(X)
    y_target = y_dir if args.task == "classification" else y_ret

    # ── build tabular features ────────────────────────────────────────────────
    _tmp = CatBoostForecaster(sequence_mode=args.sequence_mode)
    X_tab = _tmp._prepare_inputs(X)
    print(f"Data: N={N:,}  raw_features={X.shape[2]}  tabular_features={X_tab.shape[1]}  task={args.task}")

    # ── single 80/20 split for tuning ─────────────────────────────────────────
    split_idx = int(N * 0.80)
    X_train_tab, X_val_tab = X_tab[:split_idx], X_tab[split_idx:]
    y_train_target, y_val_target = y_target[:split_idx], y_target[split_idx:]
    y_val_dir = y_dir[split_idx:]
    y_val_ret = y_ret[split_idx:]

    # ── hyperparameter tuning ─────────────────────────────────────────────────
    best_params: dict = {}
    if args.tune:
        best_params = tune_hyperparams(
            X_train_tab, y_train_target,
            X_val_tab,   y_val_target,
            task=args.task, lr=args.lr,
            n_trials=args.tune_trials,
        )

    cb_params = {
        "n_estimators":     best_params.get("n_estimators", args.estimators),
        "max_depth":        best_params.get("max_depth",    args.depth),
        "learning_rate":    args.lr,
        "subsample":        best_params.get("subsample",    args.subsample),
        "colsample_bytree": best_params.get("colsample_bytree", args.colsample),
        "min_child_weight": _env_min_child_weight,
        "gamma":            _env_gamma,
        "reg_alpha":        _env_reg_alpha,
        "reg_lambda":       _env_reg_lambda,
        "early_stopping_rounds": int(cb_cfg.get("early_stopping_rounds", 15)),
    }

    # ── walk-forward CV ───────────────────────────────────────────────────────
    fold_sharpes:  list[float] = []
    fold_diraccs:  list[float] = []

    if args.folds > 0:
        print(f"\n[WalkForward] {args.folds} folds...")
        for fold_i, (tr_idx, va_idx) in enumerate(walk_forward_splits(N, args.folds)):
            Xtr = X_tab[tr_idx];  ytr = y_target[tr_idx]
            Xva = X_tab[va_idx];  yva = y_target[va_idx]
            yva_dir = y_dir[va_idx];  yva_ret = y_ret[va_idx]

            if args.task == "classification":
                m = cb.CatBoostClassifier(
                    classes_count=3, loss_function="MultiClass",
                    eval_metric="MultiClass", verbose=0, task_type="GPU", **cb_params
                )
                m.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=False)
                preds = m.predict(Xva)
            else:
                m = cb.CatBoostRegressor(
                    loss_function="RMSE",
                    eval_metric="RMSE", verbose=0, task_type="GPU", **cb_params
                )
                m.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=False)
                preds = m.predict(Xva)

            sh  = compute_sharpe(preds, yva_ret)
            acc = compute_dir_accuracy(preds, yva_dir)
            fold_sharpes.append(sh)
            fold_diraccs.append(acc)
            print(f"  Fold {fold_i+1}: Sharpe={sh:+.3f}  DirAcc={acc:.3f}  n_val={len(va_idx):,}")

        cv_sharpe = float(np.mean(fold_sharpes))
        cv_sharpe_std = float(np.std(fold_sharpes))
        cv_diраcc = float(np.mean(fold_diraccs))
        print(f"\n[WalkForward] CV Sharpe={cv_sharpe:+.3f} ± {cv_sharpe_std:.3f}  DirAcc={cv_diраcc:.3f}")
    else:
        cv_sharpe = cv_sharpe_std = cv_diраcc = 0.0

    # ── final model on full 80% train ─────────────────────────────────────────
    if WANDB:
        wandb.init(project="forex-scaling-model", name="catboost_baseline", config=vars(args))

    model = CatBoostForecaster(
        num_classes=3 if args.task == "classification" else 1,
        sequence_mode=args.sequence_mode,
        objective="multi:softmax" if args.task == "classification" else "reg:squarederror",
        eval_metric="mlogloss" if args.task == "classification" else "rmse",
        **cb_params,
    )

    print("\nTraining final CatBoost model...")
    train_t0 = time.perf_counter()
    model.fit(
        X_train_tab, y_train_target,
        eval_set=[(X_val_tab, y_val_target)],
        verbose=True,
    )
    train_time_s = time.perf_counter() - train_t0

    # ── validation metrics ────────────────────────────────────────────────────
    val_preds_raw = model.model.predict(X_val_tab)
    val_sharpe  = compute_sharpe(val_preds_raw, y_val_ret)
    val_diраcc  = compute_dir_accuracy(val_preds_raw, y_val_dir)
    val_mse     = float(np.mean((val_preds_raw.ravel() - y_val_target.ravel()) ** 2))
    val_corr    = float(np.corrcoef(val_preds_raw.ravel(), y_val_target.ravel())[0, 1]) \
                  if len(y_val_target) > 1 else 0.0
    if not np.isfinite(val_corr):
        val_corr = 0.0

    print(f"\n[Val] Sharpe={val_sharpe:+.3f}  DirAcc={val_diраcc:.3f}  MSE={val_mse:.6f}  Corr={val_corr:.4f}")

    # ── save model ────────────────────────────────────────────────────────────
    out_dir = Path("checkpoints")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "catboost_best.json"
    model.save_model(str(out_path))
    print(f"Model saved to {out_path}")

    sidecar_path = out_dir / "catboost_best.json.json"
    sidecar_path.write_text(json.dumps({
        "model":              "catboost",
        "task":               args.task,
        "sequence_mode":      args.sequence_mode,
        "n_features_raw":     int(X.shape[2]),
        "n_features_tabular": int(X_tab.shape[1]),
        "seq_len":            int(X.shape[1]),
        "is_classifier":      args.task == "classification",
        "data_dir":           str(data_dir),
        "cache_path":         str(args.cache_path or ""),
        "pairs":              pairs,
        "data_start":         data_start,
        "data_end":           data_end,
        "historical_news_file": news_file,
        "train_time_s":       train_time_s,
        "validation_sharpe":  val_sharpe,
        "validation_dir_acc": val_diраcc,
        "validation_loss":    val_mse,
        "validation_corr":    val_corr,
        "cv_folds":           args.folds,
        "cv_sharpe_mean":     cv_sharpe,
        "cv_sharpe_std":      cv_sharpe_std,
        "cv_dir_acc_mean":    cv_diраcc,
        "fold_sharpes":       fold_sharpes,
        "params":             cb_params,
        "tuned":              args.tune,
    }, indent=2))

    # ── feature importance extraction ─────────────────────────────────────────
    if _do_feature_importance:
        print("\n[FeatureImportance] Extracting CatBoost feature importance...")
        try:
            booster = model.model.get_booster()
            importance_types = ["weight", "gain", "cover"]
            fi_report: dict = {}
            for imp_type in importance_types:
                raw_scores = booster.get_score(importance_type=imp_type)
                if not raw_scores:
                    continue
                sorted_feats = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
                fi_report[imp_type] = {
                    "all": {k: float(v) for k, v in sorted_feats},
                    "top_n": {k: float(v) for k, v in sorted_feats[:_fi_top_n]},
                }
                print(f"  [{imp_type}] Top-5: {', '.join(f'{k}={v:.4f}' for k, v in sorted_feats[:5])}")

            # Also extract sklearn-style feature_importances_ if available
            if hasattr(model.model, "feature_importances_"):
                importances = model.model.feature_importances_
                n_tab_feats = X_tab.shape[1]
                n_raw_feats = X.shape[2]
                feature_labels = _build_feature_labels(n_raw_feats, args.sequence_mode, n_tab_feats)
                labeled = sorted(
                    zip(feature_labels, importances.tolist()),
                    key=lambda x: x[1], reverse=True,
                )
                fi_report["sklearn_importance"] = {
                    "top_n": {k: v for k, v in labeled[:_fi_top_n]},
                    "all": {k: v for k, v in labeled},
                }
                print(f"  [sklearn] Top-5: {', '.join(f'{k}={v:.4f}' for k, v in labeled[:5])}")

            fi_path = out_dir / "catboost_feature_importance.json"
            fi_path.write_text(json.dumps(fi_report, indent=2))
            print(f"  Saved feature importance -> {fi_path}")
        except Exception as fi_err:
            print(f"  [FeatureImportance] Extraction failed (non-fatal): {fi_err}")

    if WANDB:
        wandb.log({
            "val_sharpe": val_sharpe,
            "val_dir_acc": val_diраcc,
            "val_mse": val_mse,
            "cv_sharpe": cv_sharpe,
        })
        wandb.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
