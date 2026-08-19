"""
training/train_gpu.py  (v4 ΓÇö 20M tick scale)
=============================================
Purpose-built to train on 20,000,000 ticks without running out of RAM.

Memory math
-----------
  20M ticks -> ~333k 1-min bars -> ~333k labeled samples
  All sequences in RAM at once = ~4.5 GB  (too much on most pods)
  Solution: chunk pipeline + memory-mapped Zarr/NPY sequences

Architecture
------------
  Phase 1 ΓÇö CHUNK INGESTION
    Split 20M ticks into 500k-tick chunks.
    Each chunk: ticks -> bars -> features -> RL labels -> append to Zarr store.
    Peak RAM per chunk: ~120 MB.  Total disk: ~250 MB Zarr (LZ4 compressed).

  Phase 2 ΓÇö MEMORY-MAPPED TRAINING
    MemmapSequenceDataset reads sequences directly from Zarr / NPY on disk.
    Workers pre-fetch batches asynchronously ΓÇö GPU is never waiting.
    Effective throughput: ~1,200 batches/sec on RTX 4090 with AMP.

  Phase 3 ΓÇö RL TRAINING
    ForexTradingEnv streams samples from the same memory-mapped arrays.
    DQN replay buffer stays on GPU (pinned memory).

Usage
-----
    # Full 20M tick pipeline (cloud GPU or local workstation with enough VRAM)
    python training/train_gpu.py --n-ticks 20000000 --model haelt --epochs 100

    # With real Dukascopy data
    python training/train_gpu.py --data-source dukascopy \\
        --data-start 2020-01-01 --data-end 2023-12-31 \\
        --model haelt --epochs 100

    # With TDS export
    python training/train_gpu.py --data-source tds --model haelt --epochs 100

    # All 6 architectures sequentially
    python training/train_gpu.py --n-ticks 20000000 --all-models --epochs 50

    # Resume after interruption
    python training/train_gpu.py --n-ticks 20000000 --model haelt --resume

    # RL agents on top of supervised
    python training/train_gpu.py --n-ticks 20000000 --rl-train --rl-algo dqn
"""

import argparse
import json
import os
import sys
import time
import warnings

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass

# ΓöÇΓöÇ Windows / PythonΓÇæ3.12+ asyncio handling ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Zarr v3 uses `asyncio.run_in_executor` for all storage reads. PythonΓÇ»3.12 on
# Windows switched the default event loop to `WindowsProactorEventLoop`, which
# does **not** support fileΓÇæI/O via `run_in_executor` and can raise
# `OSError [Errno 22] Invalid argument`.  Historically we forced the older
# selector loop to work around this.
#
# The selector policy (`WindowsSelectorEventLoopPolicy`) is now deprecated and
# will be removed in PythonΓÇ»3.16, which triggers the warning you see.  To keep
# the code futureΓÇæproof while preserving compatibility we:
#   1. Suppress the deprecation warning when the selector policy is used.
#   2. Apply the selector policy only on Python versions where it still
#      exists (<ΓÇ»3.16).
#   3. On newer Python releases we fall back to the default policy, which
#      works correctly for Zarr v3.
import asyncio

if sys.platform == "win32":
    import warnings

    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="asyncio",
    )
    if sys.version_info < (3, 16):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    else:
        # No special policy needed on Python ΓëÑ3.16 ΓÇô the default is fine.
        pass
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import yaml as _yaml  # noqa: F401

    _YAML = True
except ImportError:
    _YAML = False

try:
    from tqdm import tqdm as _tqdm

    def _pbar(it=None, **kw):
        return _tqdm(it, **kw)
except ImportError:

    class _DummyBar:
        """No-op progress bar used when tqdm is not installed."""

        def __init__(self, *a, **kw):
            pass

        def update(self, n=1):
            pass

        def set_postfix(self, **kw):
            pass

        def close(self):
            pass

        def __iter__(self):
            return iter([])

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    def _pbar(it=None, **kw):
        return iter(it) if it is not None else _DummyBar()


import numpy as np

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))

# -- Core project imports -------------------------------------------------------
from models.architectures import (
    MODEL_REGISTRY,
)
from monitoring.drift_gate import run_drift_gate

# Advanced Training Mechanics
from validation.mlflow_logger import MLflowModelLogger

try:
    from models.ensemble import EnsembleMetaLearner, train_meta_learner  # noqa: F401

    ENSEMBLE = True
except ImportError:
    ENSEMBLE = False
from config.settings import (
    MONITORING,
)


def _sharpe_ann_factor(args=None) -> float:
    """Annualization factor for val Sharpe - auto-detected from data.

    Priority:

      1. Explicit CLI / YAML override (``args.sharpe_annualization_factor``).
      2. Auto-detect from the active cache's bar frequency and
         lookahead horizon.  This is the textbook-correct factor for a
         stream of per-trade (per-lookahead) returns:
         ``sqrt(bars_per_year / lookahead_bars)``.
      3. Last-resort neutral factor of 1.0 (so we never silently
         inflate Sharpe when nothing is known).

    Replaces the old hard-coded fallback of 325.0 which inflated Sharpe
    by 2.3x–12.7x depending on the user's session/full-day assumption.
    """  # noqa: RUF002
    override = None
    cache_path = None
    bar_freq = None
    lookahead = 1
    full_day = False
    if args is not None:
        override = getattr(args, "sharpe_annualization_factor", None)
        cache_path = getattr(args, "cache_path", None) or getattr(args, "data_cache", None)
        bar_freq = getattr(args, "bar_freq", None)
        lookahead = int(getattr(args, "lookahead_bars", None) or getattr(args, "label_lookahead_bars", None) or 1)
        full_day = bool(getattr(args, "fx_full_day", False))
    try:
        from training.sharpe_annualization import auto_annualization_factor

        return float(
            auto_annualization_factor(
                cache_path=cache_path,
                bar_freq=bar_freq,
                lookahead_bars=lookahead,
                full_day=full_day,
                override=override,
            )
        )
    except Exception:
        # If auto-detection is unavailable for any reason, fall back to
        # the override (or 1.0) rather than a stale magic number.
        if override is not None:
            try:
                return float(override)
            except (TypeError, ValueError):
                pass
        return 1.0


try:
    import numcodecs  # noqa: F401
    import torch
    import torch.nn as nn  # noqa: F401
    import torch.nn.functional as F  # noqa: F401 - used in dynamic eval contexts
    from torch.amp import GradScaler, autocast  # noqa: F401
    from torch.utils.data import DataLoader, Dataset, IterableDataset  # noqa: F401

    TORCH = True
except ImportError:
    print("[ERROR] PyTorch not installed. pip install torch")
    sys.exit(1)

from training.gpu_cache_io import (
    ZARR,
)

if not ZARR:
    print("[WARN] zarr not installed - using NPY memmap fallback. pip install zarr numcodecs")


from training.core import WANDB, _safe_wandb_log

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter

    TENSORBOARD = True
except ImportError:
    _SummaryWriter = None  # type: ignore[misc, assignment]
    TENSORBOARD = False

try:
    from monitoring.rich_display import _RichDisplay

    RICH_DISPLAY = True
except Exception:
    _RichDisplay = None  # type: ignore[misc, assignment]
    RICH_DISPLAY = False

# -----------------------------------------------------------------------------
# DEVICE / PREFLIGHT - re-exported from training/gpu_device.py
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# TRAINING LOGGER  (delegates to monitoring/train_logger.py)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# CACHE / DATA PIPELINE HELPERS - re-exported from training/cache_integrity.py
# -----------------------------------------------------------------------------

# Keep chunk schema lock name on train_gpu for any residual refs

# -----------------------------------------------------------------------------
# DATASET BUILDER - re-exported from training/dataset_builder.py
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# DIRECTION / LABEL HELPERS - re-exported from training/direction_control.py
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# PYTORCH PROFILER  (--profile flag)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# BEST-FOLD PROMOTION
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# FEATURE ABLATION - re-exported from training/feature_ablation.py
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# CUSTOM LOSSES (TRADING-AWARE) - see training/gpu_losses.py
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# MODEL FACTORY - re-exported from training/model_factory.py
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# POST-TRAIN (ensemble / gate / auto-tune) - training/post_train.py
# -----------------------------------------------------------------------------

# supervised_train / encoder warm-start: training/supervised_loop.py
# -----------------------------------------------------------------------------
# PRETRAIN - re-exported from training/pretrain_runner.py
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# RL - re-exported from training/rl_runner.py
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# TRAINING LOOP - re-exported from training/supervised_loop.py
# -----------------------------------------------------------------------------
from training.cache_integrity import (
    _clamp_n_samples_to_disk,
    _get_pairs,
    _promotion_holdout_n,
    _warn_multitask_cache_sidecars,
)

# Explicit imports for backward compatibility with test modules
from training.config_validate import validate_run_config
from training.core import (
    _FIRST_CHUNK_COLS,
    OPTUNA,
)

# -----------------------------------------------------------------------------
# DATASETS - re-exported from training/gpu_datasets.py (imported near losses)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# SPLITS - re-exported from training/cv_splits.py
# -----------------------------------------------------------------------------
from training.cv_splits import (
    _build_cv_splits,
    _embargo_bars,
    _purge_bars,
    _validation_method,
    walk_forward_splits,
)
from training.dataset_builder import (
    _FIRST_CHUNK_COLS,  # noqa: F401, F811
    build_dataset_chunked,
)
from training.feature_ablation import (
    _atomic_copy,
)

# -----------------------------------------------------------------------------
# CLI - re-exported from training/gpu_cli.py
# -----------------------------------------------------------------------------
from training.gpu_cli import (
    _apply_auto_run_dir,
    _baseline_ablation_completion_status,
    _latest_resumable_fold,
    _load_walk_forward_resume_history,
    _member_training_args,
    _model_completion_status,
    _set_global_seed,
    _slug_part,
    _supervised_resume_status,
    apply_hardware_profile,
    parse_args,
)
from training.gpu_datasets import (
    ZarrStreamDataset,
)
from training.gpu_device import (
    setup_device,
)
from training.model_factory import (
    build_model,
)
from training.post_train import (
    _append_model_comparison_report,
    _best_epoch_from_history,
    _evaluate_forward_gate,
    _evaluate_tune_split,
    _generate_model_card,
    _history_for_auto_tune,
    _maybe_auto_tune_next_run,
    _promote_best_fold,
    _safe_save_json,
    run_ensemble_meta,
    run_profiler,
)
from training.pretrain_runner import (
    _fold_history_summary,
    _parse_pretrain_ablation_models,
    _pretrain_ablation_verdict,
    _read_json_dict,
    _update_pretrain_report,
    run_pretrain,
)
from training.rl_runner import (
    _feature_schema_payload,
    _verify_onnx_schema_deployment,
    run_rl,
)
from training.supervised_loop import (
    run_diversity_finetune,
    supervised_train,
)

# Settings aliases expected by gpu_cli host bind / older tests



# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------


class _StageTimer:
    """Accumulate wall-clock seconds + GPU util samples per pipeline stage."""

    def __init__(self) -> None:
        self.times: dict[str, float] = {}
        self.gpu_samples: list[dict[str, float]] = []
        self.gpu_by_stage: dict[str, list[dict[str, float]]] = {}

    @staticmethod
    def _sample_gpu() -> dict[str, float]:
        sample = {
            "gpu_util_pct": -1.0,
            "gpu_mem_util_pct": -1.0,
            "gpu_temp_c": -1.0,
            "gpu_mem_mb": -1.0,
        }
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            sample["gpu_util_pct"] = float(util.gpu)
            sample["gpu_mem_util_pct"] = float(util.memory)
            sample["gpu_temp_c"] = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            sample["gpu_mem_mb"] = float(mem.used / 1_000_000)
        except Exception:
            try:
                import torch

                if torch.cuda.is_available():
                    sample["gpu_mem_mb"] = float(torch.cuda.memory_allocated() / 1_000_000)
            except Exception:
                pass
        return sample

    def stage(self, name: str):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t0 = time.perf_counter()
                self.gpu0 = timer._sample_gpu()
                return self

            def __exit__(self, *exc):
                dt = time.perf_counter() - self.t0
                timer.times[name] = timer.times.get(name, 0.0) + dt
                gpu1 = timer._sample_gpu()
                samples = [self.gpu0, gpu1]
                timer.gpu_samples.extend(samples)
                timer.gpu_by_stage.setdefault(name, []).extend(samples)
                util = max(self.gpu0.get("gpu_util_pct", -1), gpu1.get("gpu_util_pct", -1))
                util_s = f" gpu_util≤{util:.0f}%" if util >= 0 else ""
                print(f"[Timing] {name}: {dt:.1f}s{util_s}")
                return False

        return _Ctx()

    def summary(self) -> dict[str, float]:
        return {k: round(v, 3) for k, v in self.times.items()}

    def gpu_summary(self) -> dict[str, Any]:
        utils = [s["gpu_util_pct"] for s in self.gpu_samples if s.get("gpu_util_pct", -1) >= 0]
        temps = [s["gpu_temp_c"] for s in self.gpu_samples if s.get("gpu_temp_c", -1) >= 0]
        mems = [s["gpu_mem_mb"] for s in self.gpu_samples if s.get("gpu_mem_mb", -1) >= 0]
        out: dict[str, Any] = {
            "n_samples": len(self.gpu_samples),
            "gpu_util_pct_max": round(max(utils), 1) if utils else None,
            "gpu_util_pct_mean": round(sum(utils) / len(utils), 1) if utils else None,
            "gpu_temp_c_max": round(max(temps), 1) if temps else None,
            "gpu_mem_mb_max": round(max(mems), 1) if mems else None,
        }
        per_stage = {}
        for name, samples in self.gpu_by_stage.items():
            su = [s["gpu_util_pct"] for s in samples if s.get("gpu_util_pct", -1) >= 0]
            if su:
                per_stage[name] = {
                    "gpu_util_pct_max": round(max(su), 1),
                    "gpu_util_pct_mean": round(sum(su) / len(su), 1),
                }
        if per_stage:
            out["by_stage"] = per_stage
        return out


def main():
    run_start_time = datetime.now(UTC).isoformat()
    _timer = _StageTimer()
    args = parse_args()
    apply_hardware_profile(args)
    _set_global_seed(getattr(args, "seed", None))

    # Ensure structured `log_data_load` records reach stdout alongside the
    # bare `print()` statements the rest of the pipeline uses. Nothing fancy:
    # one-line INFO format keeps the run log readable and grep-able.
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(message)s",
        force=False,
    )

    # Expand user-home-relative paths (e.g. ~/forex_data) from config/CLI
    args.checkpoint_dir = str(Path(args.checkpoint_dir).expanduser())
    args.data_cache = str(Path(args.data_cache).expanduser())

    if getattr(args, "validate_config", False):
        sys.exit(validate_run_config(args))

    if getattr(args, "hparam_search", False) and getattr(args, "all_models", False):
        print(f"[HPO] --hparam-search runs only --model {args.model} (config model.all_models ignored).")
        args.all_models = False

    run_name = _apply_auto_run_dir(args)
    mlflow_logger = MLflowModelLogger(verbose=False)

    # Initialize Discord alerter globally for the run
    try:
        from monitoring.discord_alerts import DiscordAlerter

        alerter = DiscordAlerter(verbose=False)
    except Exception as e:
        print(f"[Discord] Initialization failed: {e}")
        alerter = None

    # Load persistent training memory and apply conservative nudges unless this

    # run is intended to be a clean baseline/fresh start.

    if bool(getattr(args, "training_memory", True)):
        try:
            from training.training_memory import TrainingMemory

            _train_memory = TrainingMemory(path="logs/training_memory.json")

            _train_memory.apply_to_args(args)

            print(f"[TrainingMemory] {_train_memory.summary()}")

        except Exception as _tm_e:
            print(f"[TrainingMemory] Could not load/apply: {_tm_e}")

            _train_memory = None

    else:
        print("[TrainingMemory] Disabled for this run (--no-training-memory)")

        _train_memory = None

    _all_pairs = _get_pairs(args)
    _pairs_str = ", ".join(_all_pairs)
    _embed_str = f"  embed={getattr(args, 'pair_embed_dim', 0)}d" if len(_all_pairs) > 1 else ""
    if getattr(args, "all_models", False):
        _requested_models = [m.strip().lower() for m in str(getattr(args, "models", "") or "").split(",") if m.strip()]

        _display_models = _requested_models or list(MODEL_REGISTRY.keys())

        _model_display = "ALL_MODELS"

        _queue_display = ", ".join(m.upper() for m in _display_models)

    else:
        _model_display = str(args.model).upper()

        _queue_display = None

    try:
        if alerter:
            alerter.send_training_started(
                model=_model_display,
                run_name=run_name,
                pairs=getattr(args, "pairs", []),
                data_window=f"{getattr(args, 'start_date', 'unknown')} to {getattr(args, 'end_date', 'unknown')}",
            )
    except Exception as e:
        print(f"[Discord] Failed to send training_started: {e}")

    print(f"\n{'=' * 62}")
    print("  Forex Scaling Model ΓÇö 20M Tick GPU Trainer")
    print(f"  Run: {run_name}  |  Ticks: {args.n_ticks:,}  |  Mode: {_model_display}")
    if _queue_display:
        print(f"  Model queue: {_queue_display}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Pairs: {_pairs_str}{_embed_str}")
    print(f"  Strategy: {args.strategy_mode}  |  Bars: {args.bar_freq}  |  Lookahead: {args.lookahead_bars} bars")
    print(f"  Batch: {args.batch_size}  |  Epochs: {args.epochs}  |  AMP: {args.amp}")
    print(f"  Labels: {args.label_method}  |  Loss: {args.loss}  |  Early-stop: {args.early_stop_metric}")
    print(
        f"  Historical news: {getattr(args, 'historical_news_mode', 'calendar')}  |  "
        f"Cache format: {'NPY on Windows' if sys.platform == 'win32' else 'Zarr'}"
    )
    if getattr(args, "multitask", False):
        print(f"  MultiTask: ON  (w_ret={args.mt_w_ret}  w_conf={args.mt_w_conf})")
    if getattr(args, "pretrain_regime", False):
        print("  Pretrain: regime-aware TSCL (hard negatives from opposite regime)")
    if getattr(args, "train_ensemble", False):
        print(f"  Ensemble meta-learner: ON  (epochs={args.ensemble_epochs}  div_weight={args.ensemble_div_weight})")
    print(f"{'=' * 62}")

    # ΓÜá∩╕Å  Synthetic data warning ΓÇö always visible
    if getattr(args, "data_source", "dukascopy") == "synthetic":
        print(f"\n{'!' * 62}")
        print("  ΓÜá  WARNING: SYNTHETIC DATA")
        print("  Training on artificially generated price data.")
        print("  Results DO NOT reflect real market performance.")
        print("  Use --data-source dukascopy for real data.")
        print(f"{'!' * 62}\n")

    device, n_gpus, amp_dtype = setup_device(
        dtype_override=getattr(args, "dtype", "auto"),
        deterministic=bool(getattr(args, "deterministic", False)),
    )

    # A-M3: opt-in fully-deterministic mode (overrides setup_device's speed flags).
    if getattr(args, "deterministic", False):
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            print("[Deterministic] cudnn.deterministic=True, benchmark=False, deterministic algorithms ON")
        except Exception as _de:
            print(f"[Deterministic] Could not fully enable determinism: {_de}")

    # -- Phase 1: Build / load chunked dataset ---------------------------------
    _retrain_flag = Path(args.checkpoint_dir) / "needs_retrain.flag"
    if _retrain_flag.exists():
        try:
            _rf = json.loads(_retrain_flag.read_text(encoding="utf-8"))
            print(f"[Retrain] needs_retrain flag consumed: {_rf.get('reason', 'demotion')}")
        except Exception:
            print("[Retrain] needs_retrain flag consumed")
        try:
            _retrain_flag.unlink()
        except OSError:
            pass

    with _timer.stage("dataset_build"):
        cache_path, n_samples, n_features, scaler = build_dataset_chunked(args)

    if getattr(args, "build_only", False):
        print(f"\n[Pipeline] Dataset built successfully at {cache_path}. Exiting due to --build-only.")
        sys.exit(0)

    n_samples = _clamp_n_samples_to_disk(cache_path, n_samples)
    _max_n = int(getattr(args, "max_samples", 0) or 0)
    if _max_n > 0 and _max_n < n_samples:
        print(
            f"[Data] --max-samples {_max_n:,} applied (was {n_samples:,}; using the "
            f"earliest {_max_n:,} rows of the time-ordered cache)."
        )
        n_samples = _max_n
    _warn_multitask_cache_sidecars(cache_path, args)
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        _fp = int(getattr(args, "_f_per_pair", n_features) or n_features)
        args._feat_names = list(scaler.feature_names_in_)[:_fp]
    print(f"\n[Dataset] {n_samples:,} sequences ├ù {n_features} features ├ù seq_len {args.seq_len}")

    if getattr(args, "data_quality_check", False):
        print("\n[DataQuality] Running data quality check...")
        import subprocess

        script_path = Path(__file__).resolve().parent.parent / "scripts" / "data_quality_check.py"
        try:
            subprocess.run([sys.executable, str(script_path), "--cache-path", str(cache_path), "--full"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[DataQuality] Error running data quality check: {e}")
            if not getattr(args, "skip_training", False):
                sys.exit(1)

    if getattr(args, "skip_training", False):
        print("\n[Main] --skip-training provided. Exiting before training.")
        return

    # -- Optional pre-run drift gate -------------------------------------------
    if getattr(args, "drift_gate", False):
        print("\n[DriftGate] Running pre-run drift gate...")
        try:
            drift = run_drift_gate(
                cache_path=cache_path,
                baseline_samples=int(getattr(args, "drift_baseline_samples", 20_000)),
                live_samples=int(getattr(args, "drift_live_samples", 5_000)),
                psi_threshold=float(getattr(args, "drift_psi_threshold", MONITORING.get("psi_threshold", 0.2))),
                ks_pvalue_threshold=float(
                    getattr(args, "drift_ks_pvalue_threshold", MONITORING.get("ks_pvalue_threshold", 0.05))
                ),
                ks_statistic_threshold=float(getattr(args, "drift_ks_statistic_threshold", 0.05)),
            )
            if drift.get("drift_detected", False):
                reasons = "; ".join(drift.get("reasons", [])) or "drift detected"
                raise RuntimeError(f"[DriftGate] FAILED: {reasons}")
            print("[DriftGate] PASS")
        except Exception as e:
            if getattr(args, "drift_fail_open", False):
                print(f"[DriftGate] WARN: {e} ΓÇö continuing due to --drift-fail-open")
            else:
                raise

    # -- W&B ------------------------------------------------------------------
    wandb_run: Any = None
    if WANDB and not args.no_wandb and os.getenv("WANDB_API_KEY"):
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={**vars(args), "n_samples": n_samples, "n_features": n_features},
            tags=[args.model, args.data_source, "20M"],
        )
    elif not WANDB:
        print("[W&B] Skipped: wandb package not installed (pip install wandb)")
    elif args.no_wandb:
        print("[W&B] Disabled (tracking.no_wandb or --no-wandb)")
    elif not os.getenv("WANDB_API_KEY"):
        print("[W&B] Skipped: WANDB_API_KEY not set in .env")

    # ── Log build artifacts to W&B ──────────────────────────
    if wandb_run is not None:
        try:
            _manifest_p = Path(cache_path).parent / "dataset_manifest.json"
            if _manifest_p.exists():
                wandb_run.log_artifact(str(_manifest_p), type="dataset_manifest")
            _quality_p = Path(cache_path).parent / "priority4_data_feature_report.json"
            if _quality_p.exists():
                wandb_run.log_artifact(str(_quality_p), type="data_quality_report")
            _build_log_p = Path(cache_path).parent / "build_log.jsonl"
            if _build_log_p.exists():
                wandb_run.log_artifact(str(_build_log_p), type="build_log")
        except Exception:
            pass

    from training.config_validate import resolve_models_to_train as _resolve_models_to_train

    models_to_train, _ = _resolve_models_to_train(args, apply_resume_filter=False)
    _tabular = {"xgboost", "catboost"}
    _deep = [m for m in models_to_train if m not in _tabular]
    _bad = [m for m in _deep if m not in MODEL_REGISTRY]
    if _bad:
        raise ValueError(f"Unknown deep model(s) {_bad}; expected one of {list(MODEL_REGISTRY)}")
    models_to_train = _deep

    if args.all_models and getattr(args, "resume", False) and not getattr(args, "retrain_completed_models", False):
        pending_models = []
        skipped_models = []
        for _idx, _model_name in enumerate(models_to_train):
            _probe_args = _member_training_args(args, _model_name, _idx, len(models_to_train))
            _done, _reason = _model_completion_status(_model_name, _probe_args.checkpoint_dir)
            if _done:
                skipped_models.append((_model_name, _reason))
            else:
                pending_models.append(_model_name)
                print(f"[AllModels] Will train {_model_name}: {_reason}")
        for _model_name, _reason in skipped_models:
            print(f"[AllModels] Skipping completed {_model_name}: {_reason}")
        models_to_train = pending_models
        if not models_to_train:
            print("[AllModels] No unfinished models found. Use --retrain-completed-models to rerun all members.")
            return

    for _mi, model_name in enumerate(models_to_train):
        model_args = _member_training_args(args, model_name, _mi, len(models_to_train))
        if _train_memory is not None:
            try:
                _train_memory.apply_to_model_args(model_args, model_name, base_args=args)
            except Exception as _tm_apply_e:
                print(f"[TrainingMemory] Per-model apply skipped for {model_name}: {_tm_apply_e}")
        _set_global_seed(getattr(model_args, "seed", None))
        model_artifact_dir = Path(model_args.checkpoint_dir)
        model_artifact_dir.mkdir(parents=True, exist_ok=True)
        model_args.seq_len = args.seq_len
        model = build_model(model_name, n_features, model_args).to(device)


        # -- PyTorch profiler (--profile) --------------------------------------
        if getattr(model_args, "profile", False):
            _prof_ds = ZarrStreamDataset(cache_path, np.arange(min(512, n_samples)), shuffle_chunks=False)
            _prof_dl = DataLoader(_prof_ds, batch_size=min(model_args.batch_size, 64), num_workers=0)
            run_profiler(
                build_model(model_name, n_features, model_args).to(device),
                _prof_dl,
                device,
                amp_dtype,
                model_args.amp,
                log_dir=str(Path(model_args.checkpoint_dir).parent / "logs"),
                run_name=f"{model_name}_{run_name}",
                seq_len=getattr(model_args, "seq_len", None),
            )
            print("[Profiler] Done ΓÇö exiting (remove --profile to run full training).")
            return

        # Optional HPO
        if model_args.hparam_search and OPTUNA:
            print(f"\n[HPO] Optuna {model_args.n_trials} trials...")

            def objective(trial):
                ta = argparse.Namespace(**vars(model_args))  # noqa: B023
                ta.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
                ta.hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
                ta.d_model = trial.suggest_categorical("d_model", [128, 256, 512])
                ta.dropout = trial.suggest_float("dropout", 0.05, 0.3)
                ta.batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
                ta.epochs = 5
                ta.patience = 3
                ta.resume = False
                ta.all_models = False
                # Architecture search changes hidden/d_model ΓÇö existing contrastive
                # checkpoints won't transfer; skip pretrain for proxy trials.
                ta.pretrain = False
                ta.disable_pretrain_load = True
                ta.pretrain_ablation = "false"
                ta.checkpoint_dir = str(Path(model_args.checkpoint_dir) / "hpo_trials" / f"trial_{trial.number}")  # noqa: B023
                build_model(model_name, n_features, ta).to(device)  # noqa: B023
                h, bv = supervised_train(model_name, cache_path, n_samples, n_features, ta, device, n_gpus, run=None)  # noqa: B023, RUF059
                return bv

            direction = "maximize" if model_args.early_stop_metric == "sharpe" else "minimize"
            import optuna

            study = optuna.create_study(direction=direction, pruner=optuna.pruners.MedianPruner())
            study.optimize(
                objective,
                n_trials=model_args.n_trials,
                show_progress_bar=True,
                catch=(RuntimeError,),
            )
            if study.best_trial is None:
                raise RuntimeError(
                    f"[HPO] All {model_args.n_trials} trials failed ΓÇö "
                    "check logs above; try scripts/optuna_tune.py for curriculum/arch search."
                )
            for k, v in study.best_params.items():
                setattr(model_args, k.replace("-", "_"), v)
            print(f"[HPO] Best: {study.best_params}  val={study.best_value:.6f}")
            model = build_model(model_name, n_features, model_args).to(device)

        # Optional contrastive pre-training (TSCL expects dense embeddings, not class logits)
        _baseline_cv_hist = None
        _abl_arg = str(getattr(model_args, "pretrain_ablation", "auto")).lower()
        _pretrain_ablation_models = _parse_pretrain_ablation_models(getattr(model_args, "pretrain_ablation_models", ""))

        _run_ablation = (_abl_arg == "true") or (_abl_arg == "auto" and model_name in _pretrain_ablation_models)
        if _run_ablation:
            _baseline_done, _baseline_reason = _baseline_ablation_completion_status(
                model_name, model_args.checkpoint_dir, model_args
            )
            if getattr(model_args, "resume", False) and _baseline_done:
                print(
                    f"[Ablation] Baseline already complete for {model_name}; "
                    f"skipping no-pretrain proof ({_baseline_reason})."
                )
                _run_ablation = False
        if _run_ablation:
            print(f"\n[Ablation] Running baseline (NO PRETRAIN) for {model_name}...")
            base_args = argparse.Namespace(**vars(model_args))
            base_args.pretrain = False
            base_args.pretrain_ablation = False
            base_args.checkpoint_dir = str(Path(model_args.checkpoint_dir) / "baseline")

            Path(base_args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

            print(f"[Ablation] Baseline artifacts -> {base_args.checkpoint_dir}")

            _holdout_n = _promotion_holdout_n(n_samples, base_args)
            _cv_n = max(0, n_samples - _holdout_n)

            if base_args.walk_forward_cv:
                _embargo = _embargo_bars(base_args)
                _purge = _purge_bars(base_args)
                _method = _validation_method(base_args)
                splits = walk_forward_splits(_cv_n, base_args.walk_forward_folds, _embargo, _purge, _method)
                _baseline_cv_hist = []
                for fi, (tr_i, va_i) in enumerate(splits):
                    _h, _bv = supervised_train(
                        f"baseline_{model_name}",
                        cache_path,
                        n_samples,
                        n_features,
                        base_args,
                        device,
                        n_gpus,
                        run=wandb_run,
                        train_idx=tr_i,
                        val_idx=va_i,
                        fold_id=fi,
                        amp_dtype=amp_dtype,
                    )
                    _baseline_cv_hist.append({"fold": fi, "best_metric": _bv, "history": _h})
            else:
                _h, _bv = supervised_train(
                    f"baseline_{model_name}",
                    cache_path,
                    n_samples,
                    n_features,
                    base_args,
                    device,
                    n_gpus,
                    run=wandb_run,
                    amp_dtype=amp_dtype,
                )
                _baseline_cv_hist = [{"fold": 0, "best_metric": _bv, "history": _h}]
            print("[Ablation] Baseline completed.")
            _maybe_auto_tune_next_run(
                base_args,
                _history_for_auto_tune(_baseline_cv_hist),
                {"promoted": True, "reasons": ["pretrain_ablation_baseline"]},
                phase="pretrain_ablation_baseline",
                model_name=f"baseline_{model_name}",
                force_dry=True,
            )

        _supervised_started, _supervised_reason = _supervised_resume_status(
            model_name, model_args.checkpoint_dir, model_args
        )
        if model_args.pretrain and getattr(model_args, "resume", False) and _supervised_started:
            print(
                f"[Pretrain] Supervised checkpoints already exist for {model_name}; "
                f"skipping pretrain on resume ({_supervised_reason})."
            )
        elif model_args.pretrain:
            if model_args.loss == "cross_entropy":
                pt_ns = argparse.Namespace(**vars(model_args))
                pt_ns.loss = "huber"
                model = build_model(model_name, n_features, pt_ns).to(device)
            with _timer.stage(f"pretrain_{model_name}"):
                model = run_pretrain(model, cache_path, n_features, model_args, device, run=wandb_run)

        # Supervised training (single split or walk-forward CV)
        log_dir = Path(model_args.checkpoint_dir).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        _fold_metrics = None  # populated by walk-forward for the promotion gate
        _holdout_n = _promotion_holdout_n(n_samples, model_args)
        _cv_n = max(0, n_samples - _holdout_n)
        if _holdout_n > 0:
            print(f"[Holdout] Reserved last {_holdout_n:,} bars for promotion gate (CV uses 0:{_cv_n:,})")
        if model_args.walk_forward_cv:
            splits, _cv_strategy = _build_cv_splits(model_args, _cv_n)
            print(
                f"[CV] strategy={_cv_strategy} | {len(splits)} folds "
                f"| embargo={_embargo_bars(model_args)} purge={_purge_bars(model_args)}"
            )
            cv_hist: list[dict] = []
            _start_fold = 0
            _artifact_run_name = str(getattr(model_args, "run_name_slug", "") or _slug_part(run_name, max_len=140))
            _artifact_model_name = _slug_part(model_name, max_len=80)
            if getattr(model_args, "resume", False):
                _resume_fold = _latest_resumable_fold(model_name, model_args.checkpoint_dir, len(splits))
                if _resume_fold is not None and _resume_fold > 0:
                    _start_fold = _resume_fold
                    cv_hist = _load_walk_forward_resume_history(
                        model_name,
                        model_args.checkpoint_dir,
                        log_dir,
                        _artifact_run_name,
                        _artifact_model_name,
                        _start_fold,
                        str(model_args.early_stop_metric),
                    )
                    print(
                        f"[WalkForward] Resume: restored {len(cv_hist)} completed fold(s); "
                        f"continuing from fold {_start_fold}."
                    )
            for fi, (tr_i, va_i) in enumerate(splits):
                if fi < _start_fold:
                    continue
                with _timer.stage(f"supervised_{model_name}_fold{fi}"):
                    history, best_val = supervised_train(
                        model_name,
                        cache_path,
                        n_samples,
                        n_features,
                        model_args,
                        device,
                        n_gpus,
                        run=wandb_run,
                        train_idx=tr_i,
                        val_idx=va_i,
                        fold_id=fi,
                        amp_dtype=amp_dtype,
                    )
                cv_hist.append({"fold": fi, "best_metric": best_val, "history": history})
            _artifact_run_name = str(getattr(model_args, "run_name_slug", "") or _slug_part(run_name, max_len=140))

            _artifact_model_name = _slug_part(model_name, max_len=80)

            with open(log_dir / f"{_artifact_run_name}_{_artifact_model_name}_cv.json", "w", encoding="utf-8") as fp:
                json.dump(cv_hist, fp)
            _promote_best_fold(
                model_name, model_args.checkpoint_dir, cv_hist, model_args.early_stop_metric, alerter=alerter
            )
            _generate_model_card(model_name, model_args, cv_hist, model_args.checkpoint_dir, n_features)
            _fold_metrics = [e.get("best_metric") for e in cv_hist if e.get("best_metric") is not None]
        else:
            with _timer.stage(f"supervised_{model_name}"):
                history, best_val = supervised_train(
                    model_name,
                    cache_path,
                    n_samples,
                    n_features,
                    model_args,
                    device,
                    n_gpus,
                    run=wandb_run,
                    amp_dtype=amp_dtype,
                )
            with open(log_dir / f"{run_name}_{model_name}.json", "w", encoding="utf-8") as fp:
                json.dump(history, fp)
            _generate_model_card(model_name, model_args, history, model_args.checkpoint_dir, n_features)

        # SYS-002: evaluate best model on tune split (isolated from val/early-stopping)
        _tune_eval_idx = getattr(model_args, "_tune_eval_idx", None)
        _tune_eval_metrics = {}
        if _tune_eval_idx is not None and len(_tune_eval_idx) > 0:
            try:
                _best_ckpt = Path(model_args.checkpoint_dir) / f"{model_name}_best.pt"
                if _best_ckpt.exists():
                    _ckpt_data = torch.load(_best_ckpt, map_location=device, weights_only=True)
                    _eval_model = build_model(model_name, n_features, model_args).to(device)
                    _eval_model.load_state_dict(
                        _ckpt_data["model_state_dict"]
                        if isinstance(_ckpt_data, dict) and "model_state_dict" in _ckpt_data
                        else _ckpt_data.get("state_dict", _ckpt_data)
                        if isinstance(_ckpt_data, dict)
                        else _ckpt_data
                    )
                    _tune_eval_metrics = _evaluate_tune_split(
                        _eval_model, cache_path, _tune_eval_idx, model_args, device, amp_dtype
                    )
                    del _eval_model
                else:
                    print(f"[TuneEval] Best checkpoint not found at {_best_ckpt}, skipping tune eval")
            except Exception as _te:
                print(f"[TuneEval] Evaluation failed (non-fatal): {_te}")

        # Write train_summary.json ΓÇö distinct from manifest, contains only training metrics
        _history_for_tune = (
            _history_for_auto_tune(cv_hist)
            if model_args.walk_forward_cv and "cv_hist" in locals()
            else _history_for_auto_tune(history)
        )

        # SYS-002: override val metrics with tune-split metrics for auto-tune decisions
        if _tune_eval_metrics:
            _history_for_tune = dict(_history_for_tune)
            _history_for_tune["tune_loss"] = _tune_eval_metrics["tune_loss"]
            _history_for_tune["tune_sharpe"] = _tune_eval_metrics["tune_sharpe"]
            _history_for_tune["tune_samples"] = _tune_eval_metrics["tune_samples"]
            _history_for_tune["_tune_eval_isolated"] = True
        best_epoch = _best_epoch_from_history(_history_for_tune)

        _ts_path = model_artifact_dir / "train_summary.json"
        _ts_path.parent.mkdir(parents=True, exist_ok=True)
        _ts_hist = _history_for_tune
        _ts_sharpe_curve = _ts_hist.get("val_sharpe", [])
        _ts_vloss_curve = _ts_hist.get("val_loss", [])
        _ts_tloss_curve = _ts_hist.get("train_loss", [])
        _ts_summary = {
            "model_name": model_name,
            "run_name": run_name,
            "train_mode": "walk_forward_cv" if model_args.walk_forward_cv else "single_split",
            "n_folds": len(cv_hist) if model_args.walk_forward_cv and "cv_hist" in locals() else 1,
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "epochs_completed": len(_ts_tloss_curve),
            "best_val_loss": round(min(_ts_vloss_curve), 6) if _ts_vloss_curve else None,
            "best_val_sharpe": round(max(_ts_sharpe_curve), 6) if _ts_sharpe_curve else None,
            "final_val_loss": round(_ts_vloss_curve[-1], 6) if _ts_vloss_curve else None,
            "final_val_sharpe": round(_ts_sharpe_curve[-1], 6) if _ts_sharpe_curve else None,
            "gen_gap_final": round(_ts_vloss_curve[-1] - _ts_tloss_curve[-1], 6)
            if _ts_vloss_curve and _ts_tloss_curve
            else None,
            "early_stop_metric": model_args.early_stop_metric,
            "completed_at": datetime.now(UTC).isoformat(),
        }
        try:
            _safe_save_json(_ts_summary, _ts_path)
            print(f"[TrainSummary] Written -> {_ts_path}")
        except Exception as _tse:
            print(f"[TrainSummary] Write failed (non-fatal): {_tse}")

        # Pretrain Ablation & Report
        _pt_report_path = model_artifact_dir / "pretrain_report.json"
        _pt_folds = (
            cv_hist if model_args.walk_forward_cv else [{"fold": 0, "best_metric": best_val, "history": history}]
        )
        _pt_summary = _fold_history_summary(_pt_folds, model_args.early_stop_metric)
        _pt_report = _read_json_dict(_pt_report_path)
        _pt_report.update(
            {
                "model_name": model_name,
                "pretrain_enabled": bool(getattr(model_args, "pretrain", False)),
                "supervised_training_summary": _pt_summary,
                "completed_at": datetime.now(UTC).isoformat(),
            }
        )
        try:
            _safe_save_json(_pt_report, _pt_report_path)
        except Exception as _pre:
            print(f"[PretrainReport] Write failed (non-fatal): {_pre}")

        if _run_ablation and _baseline_cv_hist is not None:
            _abl_path = model_artifact_dir / "pretrain_ablation.json"
            _baseline_summary = _fold_history_summary(_baseline_cv_hist, model_args.early_stop_metric)
            _pretrained_summary = _fold_history_summary(_pt_folds, model_args.early_stop_metric)
            _verdict, _deltas = _pretrain_ablation_verdict(_baseline_summary, _pretrained_summary)
            _abl_summary = {
                "model_name": model_name,
                "early_stop_metric": model_args.early_stop_metric,
                "comparison": {
                    "verdict": _verdict,
                    "deltas_pretrain_minus_baseline": _deltas,
                    "baseline_summary": _baseline_summary,
                    "pretrained_summary": _pretrained_summary,
                },
                "baseline_folds": _baseline_cv_hist,
                "pretrained_folds": _pt_folds,
                "completed_at": datetime.now(UTC).isoformat(),
            }
            try:
                _safe_save_json(_abl_summary, _abl_path)
                _update_pretrain_report(
                    model_args,
                    {
                        "downstream_metric_delta_vs_no_pretrain": _deltas,
                        "ablation_verdict": _verdict,
                        "ablation_report_path": str(_abl_path),
                    },
                )
                print(f"[Ablation] Comparison written -> {_abl_path}")
            except Exception as _ae:
                print(f"[Ablation] Write failed: {_ae}")

        if alerter:
            _best_f = 0
            _best_v = 0.0
            if model_args.walk_forward_cv and "cv_hist" in locals():
                _m_key = "best_metric"
                valid_folds = [f for f in cv_hist if f.get(_m_key) is not None]
                if valid_folds:
                    if model_args.early_stop_metric == "sharpe":
                        best_entry = max(valid_folds, key=lambda x: x[_m_key])
                    else:
                        best_entry = min(valid_folds, key=lambda x: x[_m_key])
                    _best_f = best_entry.get("fold", 0)
                    _best_v = best_entry.get(_m_key, 0.0)
            else:
                _best_v = best_val if best_val is not None else 0.0

            try:
                alerter.send_training_completed(
                    model=model_name, fold=_best_f, metric=model_args.early_stop_metric, score=float(_best_v)
                )
            except Exception as e:
                print(f"[Discord] Failed to send training_completed: {e}")

        try:
            ckpt_base = Path(model_args.checkpoint_dir)
            ckpt_best = ckpt_base / model_name / f"{model_name}_best.pt"
            if not ckpt_best.exists():
                alt = ckpt_base / f"{model_name}_best.pt"
                ckpt_best = alt if alt.exists() else ckpt_best
            retrain_lock = Path(model_args.checkpoint_dir) / "retrain_in_progress.lock"
            retrain_reason = "manual_train"
            if retrain_lock.exists():
                try:
                    _rj = json.loads(retrain_lock.read_text(encoding="utf-8"))
                    retrain_reason = str(_rj.get("reason", "auto_retrain"))
                except Exception:
                    retrain_reason = "auto_retrain"

            # B-C1: run the REAL promotion gate on a held-out forward window
            # instead of recording a hardcoded `promoted: True`.
            if getattr(model_args, "promotion_gate", True):
                try:
                    gate_result = _evaluate_forward_gate(
                        model_name,
                        cache_path,
                        n_samples,
                        n_features,
                        model_args,
                        device,
                        fold_sharpes=_fold_metrics,
                    )
                except Exception as _ge:
                    print(f"[PromotionGate] evaluation failed: {_ge}")
                    gate_result = {
                        "promoted": False,
                        "details": {"error": str(_ge)},
                        "reasons": [f"gate error: {_ge}"],
                        "summary": "REJECT (gate error)",
                    }
            else:
                # Gate disabled or supervised skipped: record metrics only, do
                # NOT claim promotion.
                gate_result = {
                    "promoted": False,
                    "details": {
                        "best_val_loss": float(best_val) if best_val is not None else float("inf"),
                        "epochs": float(len(history.get("train_loss", []))) if isinstance(history, dict) else 0.0,
                    },
                    "reasons": ["promotion gate not run"],
                    "summary": "NOT EVALUATED",
                }
            # Persist the gate decision so the deploy step (continuous_finetune)
            # can decide whether to promote this checkpoint to production (B-C1/B-C3).
            try:
                _prom_dir = model_artifact_dir
                _prom_dir.mkdir(parents=True, exist_ok=True)
                _prom_path = _prom_dir / "promotion_gate.json"

                # Enrich with requested model identifiers
                gate_result["model"] = model_name
                gate_result["gate_on_checkpoint"] = f"{model_name}_best.pt"

                _safe_save_json(gate_result, _prom_path)
                print(f"[PromotionGate] decision written -> {_prom_path}")
            except Exception as _pe:
                print(f"[PromotionGate] could not write decision json: {_pe}")
            try:
                _append_model_comparison_report(model_args, model_name, _ts_summary, gate_result)

            except Exception as _cmp_e:
                print(f"[ModelComparison] update failed (non-fatal): {_cmp_e}")

            deploy_result = {"status": "skipped", "error": None, "failed_step": None}
            _prod = None
            _prev = None
            _onnx_final = None
            _schema_final = None

            _reload_flag = None
            if gate_result.get("promoted") and ckpt_best.exists():
                # B-C3: deploy to the SAME production path the live engine and the
                # demotion monitor's rollback read, not the per-run checkpoint dir.
                try:
                    from monitoring.demotion_monitor import (
                        PREV_CHECKPOINT as _prev,
                    )
                    from monitoring.demotion_monitor import (
                        PROD_CHECKPOINT as _prod,
                    )
                except Exception:
                    _prod = Path(model_args.checkpoint_dir) / "production_best.pt"
                    _prev = Path(model_args.checkpoint_dir) / "production_prev.pt"
                _prod.parent.mkdir(parents=True, exist_ok=True)

                try:
                    from inference.onnx_inference import export_to_onnx

                    _onnx_tmp = _prod.with_suffix(".tmp.onnx")

                    _onnx_final = _prod.with_suffix(".onnx")

                    _onnx_prev = _onnx_final.with_name("production_prev.onnx")

                    _schema_tmp = _prod.with_name(f".{_prod.stem}.schema.tmp.json")

                    _schema_final = _prod.with_suffix(".schema.json")

                    _schema_prev = _schema_final.with_name("production_prev.schema.json")

                    export_to_onnx(
                        checkpoint_path=str(ckpt_best),
                        model_name=model_name,
                        seq_len=int(getattr(model_args, "seq_len", 60)),
                        output_path=str(_onnx_tmp),
                        n_features=int(n_features),
                    )

                    _safe_save_json(_feature_schema_payload(model_args, n_features=n_features), _schema_tmp)

                    _deploy_verify = _verify_onnx_schema_deployment(
                        _onnx_tmp,
                        _schema_tmp,
                        model_args,
                        n_features=int(n_features),
                        seq_len=int(getattr(model_args, "seq_len", 60)),
                    )

                    deploy_result["verification"] = _deploy_verify

                    if _deploy_verify.get("status") != "pass":
                        deploy_result["failed_step"] = "onnx_schema_verification"

                        raise RuntimeError(f"ONNX/schema verification failed: {_deploy_verify.get('errors')}")

                    if _prod.exists():
                        _atomic_copy(_prod, _prev)  # back up current prod -> prev
                    if _onnx_final.exists():
                        _atomic_copy(_onnx_final, _onnx_prev)

                    if _schema_final.exists():
                        _atomic_copy(_schema_final, _schema_prev)

                    _atomic_copy(ckpt_best, _prod)  # challenger -> prod (atomic)

                    os.replace(str(_onnx_tmp), str(_onnx_final))

                    os.replace(str(_schema_tmp), str(_schema_final))

                    print(f"[Deploy] Atomically promoted checkpoint/ONNX/schema -> {_prod}")

                    # Signal the live engine to hot-reload the new production model.
                    try:
                        import tempfile

                        _reload_flag = _prod.parent / "reload_model.flag"
                        _fd, _tmp_flag = tempfile.mkstemp(prefix=".reload.", suffix=".tmp", dir=str(_prod.parent))
                        os.close(_fd)
                        with open(_tmp_flag, "w", encoding="utf-8") as f:
                            f.write(f"reload {datetime.now(UTC).isoformat()}\n")
                        os.replace(_tmp_flag, _reload_flag)
                        print(f"[Deploy] Reload signalled -> {_reload_flag}")
                    except Exception as _re:
                        print(f"[Deploy] could not write reload flag: {_re}")

                    deploy_result["status"] = "success"
                    _onnx_final = _onnx_final

                    if alerter:
                        try:
                            alerter.send_promotion_gate_passed(
                                model=model_name, sharpe=float(gate_result.get("details", {}).get("sharpe", 0.0))
                            )
                            alerter.send_production_deploy_completed(
                                model=model_name,
                                onnx_path=str(_prod.with_suffix(".onnx").name),
                                schema_path=str(_prod.with_suffix(".schema.json").name),
                            )
                        except Exception as e:
                            print(f"[Discord] Failed to send promotion / deploy alerts: {e}")

                except Exception as e_deploy:
                    for _tmp_art in ("_onnx_tmp", "_schema_tmp"):
                        try:
                            _tmp_path = locals().get(_tmp_art)

                            if _tmp_path is not None and Path(_tmp_path).exists():
                                Path(_tmp_path).unlink()

                        except Exception:
                            pass

                    deploy_result["status"] = "failed"
                    deploy_result["error"] = str(e_deploy)
                    deploy_result["failed_step"] = deploy_result.get("failed_step") or "checkpoint_promotion"
                    print(f"[Deploy] Critical failure during promotion: {e_deploy}")
                    if alerter:
                        try:
                            alerter.send_production_deploy_failed(model=model_name, error_msg=str(e_deploy))
                        except Exception as e:
                            print(f"[Alerting] Failed to send deploy failure alert: {e}")

            elif not gate_result.get("promoted", False) and gate_result.get("summary") != "NOT EVALUATED":
                if alerter:
                    try:
                        alerter.send_promotion_gate_failed(
                            model=model_name,
                            reasons=gate_result.get("reasons", ["Unknown reason"]),
                            profit_factor=float(gate_result.get("details", {}).get("profit_factor", 0.0)),
                            psr=float(gate_result.get("details", {}).get("psr", 0.0)),
                        )
                    except Exception as e:
                        print(f"[Discord] Failed to send promotion_gate_failed: {e}")

            # Write deployment.json ΓÇö full transaction record regardless of outcome
            _dep_dir = model_artifact_dir
            _dep_dir.mkdir(parents=True, exist_ok=True)
            _dep_doc = {
                "model_name": model_name,
                "run_name": run_name,
                "checkpoint_dir": str(_dep_dir.resolve()),
                "run_checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
                "gate_promoted": gate_result.get("promoted", False),
                "source_checkpoint": str(ckpt_best) if ckpt_best.exists() else None,
                "production_checkpoint": str(_prod)
                if _prod is not None and deploy_result.get("status") == "success"
                else None,
                "previous_checkpoint": str(_prev) if _prev is not None else None,
                "onnx_status": "exported" if _onnx_final is not None and Path(_onnx_final).exists() else "skipped",
                "onnx_path": str(_onnx_final) if _onnx_final is not None and Path(_onnx_final).exists() else None,
                "schema_path": str(_schema_final)
                if _schema_final is not None and Path(_schema_final).exists()
                else None,
                "onnx_schema_verification": deploy_result.get("verification"),
                "reload_flag_status": "written"
                if _reload_flag is not None and Path(_reload_flag).exists()
                else "skipped",
                "deploy_status": deploy_result.get("status", "skipped"),
                "deploy_error": deploy_result.get("error"),
                "failed_step": deploy_result.get("failed_step"),
                "deployed_at": datetime.now(UTC).isoformat(),
            }
            try:
                _safe_save_json(_dep_doc, _dep_dir / "deployment.json")
                print(f"[Deploy] deployment.json written -> {_dep_dir / 'deployment.json'}")
            except Exception as _de:
                print(f"[Deploy] Could not write deployment.json: {_de}")

            mlflow_logger.log_promotion(
                model_name=f"{model_name}_train",
                gate_result=gate_result,
                training_config={
                    "run_name": run_name,
                    "run_type": retrain_reason,
                    "model": model_name,
                    "epochs": int(getattr(model_args, "epochs", 0)),
                    "batch_size": int(getattr(model_args, "batch_size", 0)),
                    "seq_len": int(getattr(model_args, "seq_len", 0)),
                    "amp": bool(getattr(model_args, "amp", False)),
                    "data_source": str(getattr(model_args, "data_source", "unknown")),
                    "n_samples": int(n_samples),
                    "n_features": int(n_features),
                },
                checkpoint_path=str(ckpt_best) if ckpt_best.exists() else None,
                extra_tags={"event_type": "training", "run_type": retrain_reason},
            )
        except Exception as _ml_e:
            print(f"[MLflow] Training log skipped: {_ml_e}")

        _maybe_auto_tune_next_run(
            model_args,
            _history_for_tune,
            gate_result,
            phase="main",
            model_name=model_name,
        )

        # Write Run-level manifest for this model
        _model_dir = Path(args.checkpoint_dir) / model_name
        _model_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "model_name": model_name,
            "run_name": getattr(args, "run_name", "unknown"),
            "checkpoint_dir": str(_model_dir.resolve()),
            "run_checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
            "fold_id": getattr(model_args, "walk_forward_folds", "single"),
            "start_time": run_start_time,
            "end_time": datetime.now(UTC).isoformat(),
            "best_epoch": int(best_epoch) if "best_epoch" in locals() and best_epoch is not None else None,
            "best_metric": float(best_val) if "best_val" in locals() and best_val is not None else None,
            "checkpoint_paths": [str(ckpt_best)],
            "promotion_result": gate_result,
            "deploy_result": deploy_result if "deploy_result" in locals() else {"status": "skipped", "error": None},
        }
        if WANDB and wandb_run and "details" in gate_result:
            _deploy_logs = {}
            for k in ["profit_factor", "sharpe", "calmar", "max_drawdown"]:
                if k in gate_result["details"]:
                    _deploy_logs[f"deploy/{k}"] = gate_result["details"][k]
            if _deploy_logs:
                _safe_wandb_log(wandb_run, _deploy_logs)

        manifest.update({"config_path": getattr(args, "config", "config/run.yaml"), "warnings": [], "errors": []})
        try:
            import subprocess

            manifest["git_hash"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        except Exception:
            manifest["git_hash"] = "unknown"

        _safe_save_json(manifest, _model_dir / "manifest.json")

        # Update persistent training memory with this run's outcome
        if _train_memory is not None:
            try:
                _tm_sharpe = None
                _tm_vloss = None
                _hist_for_mem = _history_for_tune if isinstance(_history_for_tune, dict) else {}

                _mem_best_epoch = int(best_epoch) if "best_epoch" in locals() and best_epoch is not None else None

                def _mem_hist_at(key: str, default=None):

                    values = _hist_for_mem.get(key) or []  # noqa: B023

                    if _mem_best_epoch is not None and 0 <= _mem_best_epoch < len(values):  # noqa: B023
                        return values[_mem_best_epoch]  # noqa: B023

                    return default

                _mem_warnings = []

                if _hist_for_mem.get("train_loss") and _hist_for_mem.get("val_loss"):
                    _mem_final_gap = float(_hist_for_mem["val_loss"][-1]) - float(_hist_for_mem["train_loss"][-1])

                    if _mem_final_gap > 0.05:
                        _mem_warnings.append(f"High train-val gap: {_mem_final_gap:.4f}")

                else:
                    _mem_final_gap = None

                if len(_hist_for_mem.get("val_sharpe", [])) > 5:
                    _mem_max_sh = max(_hist_for_mem["val_sharpe"])

                    _mem_final_sh = _hist_for_mem["val_sharpe"][-1]

                    if _mem_max_sh - _mem_final_sh > 0.3:
                        _mem_warnings.append(f"Sharpe collapsed by {_mem_max_sh - _mem_final_sh:.3f} from peak")

                _mem_control_report = {
                    "final_train_val_gap": _mem_final_gap,
                    "early_stopped": False,
                    "curriculum_stalls": int(max(_hist_for_mem.get("curriculum_stalls", [0]) or [0])),
                    "final_seq_len": (_hist_for_mem.get("seq_len") or [None])[-1],
                    "final_difficulty_stage": (_hist_for_mem.get("difficulty_stage") or [None])[-1],
                    "overfitting_warnings": _mem_warnings,
                }

                _mem_metric_values = [
                    float(v) for v in (_fold_metrics or []) if v is not None and np.isfinite(float(v))
                ]

                if model_args.early_stop_metric == "sharpe":
                    if _mem_metric_values:
                        _tm_sharpe = max(_mem_metric_values)

                    elif best_val is not None:
                        _tm_sharpe = float(best_val)

                elif best_val is not None:
                    _tm_vloss = min(_mem_metric_values) if _mem_metric_values else float(best_val)

                _train_memory.update(
                    {
                        "model_name": model_name,
                        "run_name": run_name,
                        "phase": "main",
                        "best_sharpe": _tm_sharpe,
                        "best_val_loss": _tm_vloss,
                        "best_epoch": _mem_best_epoch,
                        "total_epochs": len(_hist_for_mem.get("train_loss", []))
                        or int(getattr(model_args, "epochs", 0)),
                        "history": _hist_for_mem,
                        "best_epoch_state": {
                            "train_loss": _mem_hist_at("train_loss"),
                            "val_loss": _mem_hist_at("val_loss"),
                            "val_sharpe": _mem_hist_at("val_sharpe"),
                            "dir_acc": _mem_hist_at("dir_acc"),
                            "lr": _mem_hist_at("lr"),
                            "seq_len": _mem_hist_at("seq_len"),
                            "difficulty_stage": _mem_hist_at("difficulty_stage"),
                            "curriculum_stalls": _mem_hist_at("curriculum_stalls"),
                        },
                        "training_control_report": _mem_control_report,
                        "gate_result": gate_result if "gate_result" in locals() else {},
                        "args_snapshot": {
                            "lr": float(getattr(model_args, "lr", 5e-5)),
                            "dropout": float(getattr(model_args, "dropout", 0.25)),
                            "patience": int(getattr(model_args, "patience", 6)),
                            "epochs": int(getattr(model_args, "epochs", 24)),
                        },
                    }
                )
                _train_memory.save()
            except Exception as _tm_err:
                print(f"[TrainingMemory] Update failed (non-fatal): {_tm_err}")

    # -- XGBoost baseline training ---------------------------------------------
    # Runs when xgboost.enabled: true in run.yaml (or --xgb-enabled CLI).
    # Shells out to training/train_xgboost.py with params from the YAML config.
    _tabular_baselines: list[str] = []
    if getattr(args, "xgb_enabled", False):
        print(f"\n{'=' * 62}")
        print("  XGBoost Baseline Training")
        print(f"{'=' * 62}")
        _xgb_cmd = [
            sys.executable,
            str(Path(__file__).parent / "train_xgboost.py"),
            "--config",
            str(getattr(args, "config", None) or "config/run.yaml"),
        ]
        _xgb_task = str(getattr(args, "xgb_task", "classification"))
        _xgb_cmd.extend(["--task", _xgb_task])
        _xgb_cmd.extend(["--sequence-mode", str(getattr(args, "xgb_sequence_mode", "temporal"))])
        _xgb_cmd.extend(["--estimators", str(getattr(args, "xgb_n_estimators", 300))])
        _xgb_cmd.extend(["--depth", str(getattr(args, "xgb_max_depth", 6))])
        _xgb_cmd.extend(["--lr", str(getattr(args, "xgb_learning_rate", 0.05))])
        _xgb_cmd.extend(["--subsample", str(getattr(args, "xgb_subsample", 0.8))])
        _xgb_cmd.extend(["--colsample", str(getattr(args, "xgb_colsample_bytree", 0.8))])
        _xgb_cmd.extend(["--folds", str(getattr(args, "xgb_folds", 5))])
        _xgb_cmd.extend(["--samples", str(getattr(args, "xgb_max_samples", 500_000))])
        if getattr(args, "xgb_tune", False):
            _xgb_cmd.append("--tune")
            _xgb_cmd.extend(["--tune-trials", str(getattr(args, "xgb_tune_trials", 20))])
        _xgb_env = os.environ.copy()
        _xgb_env["XGB_MIN_CHILD_WEIGHT"] = str(getattr(args, "xgb_min_child_weight", 3))
        _xgb_env["XGB_GAMMA"] = str(getattr(args, "xgb_gamma", 0.1))
        _xgb_env["XGB_REG_ALPHA"] = str(getattr(args, "xgb_reg_alpha", 0.05))
        _xgb_env["XGB_REG_LAMBDA"] = str(getattr(args, "xgb_reg_lambda", 1.0))
        _xgb_env["XGB_FEATURE_IMPORTANCE"] = "1" if getattr(args, "xgb_feature_importance", True) else "0"
        _xgb_env["XGB_FEATURE_IMPORTANCE_TOP_N"] = str(getattr(args, "xgb_feature_importance_top_n", 50))
        if cache_path:
            _xgb_cmd.extend(["--cache-path", str(cache_path)])
        print(f"  Command: {' '.join(_xgb_cmd)}")
        try:
            import subprocess as _sp

            with _timer.stage("xgboost"):
                _xgb_result = _sp.run(_xgb_cmd, cwd=str(Path(__file__).parent.parent), env=_xgb_env)
            if _xgb_result.returncode == 0:
                print("[XGBoost] Baseline training completed successfully.")
                _tabular_baselines.append("xgboost")
            else:
                print(f"[XGBoost] Training failed with exit code {_xgb_result.returncode}")
        except Exception as _xgb_err:
            print(f"[XGBoost] Training failed: {_xgb_err}")

    # -- CatBoost baseline training (mirror XGBoost shell) ---------------------
    if getattr(args, "cb_enabled", False):
        print(f"\n{'=' * 62}")
        print("  CatBoost Baseline Training")
        print(f"{'=' * 62}")
        _cb_cmd = [
            sys.executable,
            str(Path(__file__).parent / "train_catboost.py"),
            "--config",
            str(getattr(args, "config", None) or "config/run.yaml"),
        ]
        _cb_cmd.extend(["--task", str(getattr(args, "cb_task", "classification"))])
        _cb_cmd.extend(["--sequence-mode", str(getattr(args, "cb_sequence_mode", "temporal"))])
        _cb_cmd.extend(["--estimators", str(getattr(args, "cb_n_estimators", 300))])
        _cb_cmd.extend(["--depth", str(getattr(args, "cb_max_depth", 6))])
        _cb_cmd.extend(["--lr", str(getattr(args, "cb_learning_rate", 0.05))])
        _cb_cmd.extend(["--subsample", str(getattr(args, "cb_subsample", 0.8))])
        _cb_cmd.extend(["--colsample", str(getattr(args, "cb_colsample_bylevel", 0.8))])
        _cb_cmd.extend(["--folds", str(getattr(args, "cb_folds", 5))])
        _cb_cmd.extend(["--samples", str(getattr(args, "cb_max_samples", 500_000))])
        if getattr(args, "cb_tune", False):
            _cb_cmd.append("--tune")
            _cb_cmd.extend(["--tune-trials", str(getattr(args, "cb_tune_trials", 20))])
        _cb_env = os.environ.copy()
        _cb_env["CB_L2_LEAF_REG"] = str(getattr(args, "cb_l2_leaf_reg", 1.0))
        _cb_env["CB_FEATURE_IMPORTANCE"] = "1" if getattr(args, "cb_feature_importance", True) else "0"
        _cb_env["CB_FEATURE_IMPORTANCE_TOP_N"] = str(getattr(args, "cb_feature_importance_top_n", 50))
        if cache_path:
            _cb_cmd.extend(["--cache-path", str(cache_path)])
        print(f"  Command: {' '.join(_cb_cmd)}")
        try:
            import subprocess as _sp

            with _timer.stage("catboost"):
                _cb_result = _sp.run(_cb_cmd, cwd=str(Path(__file__).parent.parent), env=_cb_env)
            if _cb_result.returncode == 0:
                print("[CatBoost] Baseline training completed successfully.")
                _tabular_baselines.append("catboost")
            else:
                print(f"[CatBoost] Training failed with exit code {_cb_result.returncode}")
        except Exception as _cb_err:
            print(f"[CatBoost] Training failed: {_cb_err}")

    # C: Diversity fine-tuning - deep models only (tabular baselines excluded).
    # Only runs when >=2 deep models were trained in this session (--all-models).
    _div_models = [m for m in models_to_train if m in MODEL_REGISTRY]
    if args.all_models and len(_div_models) >= 2:
        try:
            from config.settings import CURRICULUM as _CURR_DIV

            _div_cfg = _CURR_DIV  # reuse config namespace for div settings
        except ImportError:
            _div_cfg = {}
        _div_w = float(getattr(args, "div_weight", 0.10))
        _same_r = float(getattr(args, "same_role_mult", 2.0))
        # With per-model subfolders, pass the base checkpoint dir so
        # run_diversity_finetune can find each model at <base>/<model>/<model>_best.pt
        _base_ckpt = Path(args.checkpoint_dir)
        run_diversity_finetune(
            checkpoint_dir=str(_base_ckpt),
            model_names=_div_models,
            cache_path=cache_path,
            n_features=n_features,
            args=args,
            device=device,
            epochs=3,
            lr=1e-5,
            div_weight=_div_w,
            same_role_mult=_same_r,
        )
    models_to_train = list(models_to_train) + _tabular_baselines

    # Ensemble meta-learner training (with diversity penalty)
    if getattr(args, "train_ensemble", False):
        with _timer.stage("ensemble_meta"):
            run_ensemble_meta(cache_path, n_features, args, device)

        print("[Deploy] Running Promotion Gate on Ensemble...")
        ensemble_args = argparse.Namespace(**vars(args))
        ensemble_args.model = "ensemble"
        try:
            ens_gate_result = _evaluate_forward_gate(
                "ensemble", cache_path, n_samples, n_features, ensemble_args, device, fold_sharpes=None
            )
            print(f"[PromotionGate] ensemble: {ens_gate_result.get('summary', '?')}")

            _prom_dir = Path(args.checkpoint_dir) / "ensemble"
            _prom_path = _prom_dir / "promotion_gate.json"
            ens_gate_result["model"] = "ensemble"
            _safe_save_json(ens_gate_result, _prom_path)

            if ens_gate_result.get("promoted"):
                _prod = _prom_dir / "ensemble_meta_best.pt"
                _dep_dir = Path(args.checkpoint_dir)
                prod_sharpe = -999.0
                dep_json = _dep_dir / "deployment.json"
                if dep_json.exists():
                    try:
                        prod_sharpe = float(
                            json.loads(dep_json.read_text())
                            .get("gate_result", {})
                            .get("details", {})
                            .get("sharpe", -999.0)
                        )
                    except Exception as e:
                        print(f"[Deploy] Corrupted deployment.json: {e}")
                        raise RuntimeError(f"Corrupted deployment.json prevents safe promotion: {e}")

                ens_sharpe = float(ens_gate_result.get("details", {}).get("sharpe", -999.0))

                if ens_sharpe > prod_sharpe:
                    print(
                        f"[Deploy] Ensemble Sharpe {ens_sharpe:.3f} > Base {prod_sharpe:.3f}. Overwriting production!"
                    )
                    _final_prod = _dep_dir / "production_best.pt"
                    _atomic_copy(_prod, _final_prod)
                    print(f"[Deploy] Atomically promoted Ensemble -> {_final_prod}")

                    try:
                        from inference.onnx_inference import export_ensemble_to_onnx

                        _onnx_tmp = _final_prod.with_suffix(".tmp.onnx")
                        _onnx_final = _final_prod.with_suffix(".onnx")
                        export_ensemble_to_onnx(
                            checkpoint_path=str(_prod),
                            checkpoint_dir=str(_dep_dir),
                            seq_len=int(getattr(args, "seq_len", 60)),
                            n_features=int(n_features),
                            output_path=str(_onnx_tmp),
                            device="cpu",
                        )
                        os.replace(str(_onnx_tmp), str(_onnx_final))
                        print(f"[Deploy] Re-exported ONNX -> {_onnx_final}")
                    except Exception as oe:
                        import traceback

                        print(f"[Deploy] Ensemble ONNX export failed: {oe}")
                        traceback.print_exc()
                else:
                    print(
                        f"[Deploy] Ensemble Sharpe {ens_sharpe:.3f} did not beat Base {prod_sharpe:.3f}. Skipping deploy."
                    )
        except Exception as e:
            import traceback

            print(f"[Deploy] Ensemble integration failed: {e}")
            traceback.print_exc()

    # RL training
    if args.rl_train:
        if getattr(args, "rl_all_models", False) and args.all_models:
            for _rl_m in models_to_train:
                _rl_args = argparse.Namespace(**vars(args))
                _rl_args.model = _rl_m
                _rl_args.checkpoint_dir = str(Path(args.checkpoint_dir) / _rl_m)
                _sup = Path(_rl_args.checkpoint_dir) / f"{_rl_m}_best.pt"
                if not _sup.exists():
                    print(f"[RL] Skipping {_rl_m} ΓÇö no {_sup.name}")
                    continue
                print(f"\n[RL] Per-model pass: {_rl_m}")
                with _timer.stage(f"rl_{_rl_m}"):
                    run_rl(cache_path, n_features, _rl_args, device, n_samples=n_samples, run=wandb_run)
        else:
            with _timer.stage("rl"):
                run_rl(cache_path, n_features, args, device, n_samples=n_samples, run=wandb_run)

    if wandb_run:
        wandb_run.finish()

    print(f"\n{'=' * 62}")
    print("  Training complete!")
    _base_ckpt_dir = Path(args.checkpoint_dir)
    print(f"  Checkpoints: {_base_ckpt_dir}/")
    for _mn in models_to_train:
        _model_dir = _base_ckpt_dir / _mn
        if _model_dir.exists():
            print(f"    {_mn:12s} -> {_model_dir}/")

    print(f"  Dataset cache: {cache_path}  (reused on --resume)")
    _stage_summary = _timer.summary()
    _gpu_summary = _timer.gpu_summary()
    if _stage_summary:
        _total_s = sum(_stage_summary.values())
        print(f"  Stage timings (total {_total_s:.1f}s):")
        for _sn, _st in sorted(_stage_summary.items(), key=lambda kv: -kv[1]):
            _pct = (100.0 * _st / _total_s) if _total_s else 0.0
            print(f"    {_sn:28s} {_st:8.1f}s  ({_pct:5.1f}%)")
    if _gpu_summary.get("gpu_util_pct_mean") is not None:
        print(
            f"  GPU util: mean={_gpu_summary['gpu_util_pct_mean']}% "
            f"max={_gpu_summary['gpu_util_pct_max']}% "
            f"| temp_max={_gpu_summary.get('gpu_temp_c_max')}°C "
            f"| mem_max={_gpu_summary.get('gpu_mem_mb_max')}MB"
        )
    print(f"{'=' * 62}")

    try:
        import subprocess

        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        git_hash = "unknown"

    run_manifest = {
        "run_start_time": run_start_time,
        "run_end_time": datetime.now(UTC).isoformat(),
        "run_name": getattr(args, "run_name", "pipeline_run"),
        "models_trained": models_to_train,
        "git_hash": git_hash,
        "stage_timings_s": _stage_summary,
        "gpu_stats": _gpu_summary,
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    if _base_ckpt_dir.exists():
        _safe_save_json(run_manifest, _base_ckpt_dir / "run_manifest.json")
        try:
            _logs = Path("logs")
            _logs.mkdir(parents=True, exist_ok=True)
            with (_logs / "stage_timings.jsonl").open("a", encoding="utf-8") as _fh:
                _fh.write(
                    json.dumps(
                        {
                            "event": "stage_timing",
                            "run_name": run_manifest["run_name"],
                            "run_end_time": run_manifest["run_end_time"],
                            "stage_timings_s": _stage_summary,
                            "gpu_stats": _gpu_summary,
                        }
                    )
                    + "\n"
                )
        except Exception as _te:
            print(f"[Timing] Could not append stage_timings.jsonl: {_te}")


if __name__ == "__main__":
    main()
