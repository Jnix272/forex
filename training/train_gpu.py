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

import os, sys, gc, json, time, argparse, warnings, shutil, threading, queue as _queue, random, math, re

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

from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml as _yaml; _YAML = True
except ImportError:
    _YAML = False

try:
    from tqdm import tqdm as _tqdm
    def _pbar(it=None, **kw): return _tqdm(it, **kw)
except ImportError:
    class _DummyBar:
        """No-op progress bar used when tqdm is not installed."""
        def __init__(self, *a, **kw): pass
        def update(self, n=1): pass
        def set_postfix(self, **kw): pass
        def close(self): pass
        def __iter__(self): return iter([])
        def __enter__(self): return self
        def __exit__(self, *_): self.close()
    def _pbar(it=None, **kw):
        return iter(it) if it is not None else _DummyBar()

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))

# -- Core project imports -------------------------------------------------------
from data.data_ingestion import generate_synthetic_tick_data, ForexDataPipeline
from data.sources import ForexDataManager
from data.cross_asset import load_cross_asset_panel
from data.historical_news import load_historical_news_bundle, collect_headlines_for_range
from features.finbert_sentiment import SentimentPipeline
from features.feature_engineering_pl import FeatureEngineer
from labeling.rl_reward_labeling import (
    compute_rl_reward_labels_regime,
    align_labels_with_features,
)
from labeling.triple_barrier_labeling import compute_triple_barrier_labels
from models.architectures import (
    TFTScalper, iTransformerScalper, HAELTHybrid,
    MambaScalper, GNNFromSequence, EXPERTEncoder,
    HuberLoss, AsymmetricDirectionalLoss, MODEL_REGISTRY,
    MultiTaskLoss, MultiTaskWrapper,
    MultiPairWrapper,
    DiversityLoss, TemperatureScaler, OverconfidencePenalty, MODEL_ROLES,
)
from models.rl_agents import ForexTradingEnv, DQNAgent, PPOAgent, train_agent, evaluate_agent
from pretrain.contrastive import (
    TimeSeriesAugmenter, TSCLTrainer, RegimeAwareTSCLTrainer,
    BYOLTrainer, MaskedReconstructionTrainer, RepresentationCollapseError,
)
from pretrain.extended_trainers import (
    VAESeqTrainer,
    ClusterContrastiveTrainer,
    ForecastPretextTrainer,
    DriftContrastiveTrainer,
)

_PRETRAIN_SINGLE_PASS = frozenset({"byol", "masked", "vae", "forecast", "drift"})
_PRETRAIN_MULTI_BLOCK = _PRETRAIN_SINGLE_PASS
_PRETRAIN_STD_QUALITY = _PRETRAIN_SINGLE_PASS
_VALID_PRETRAIN_METHODS = _PRETRAIN_SINGLE_PASS | {"tscl", "cluster"}


def _normalize_pretrain_method(method: str) -> str:
    aliases = {
        "autoencoder": "vae",
        "regime_cluster": "cluster",
        "cluster_tscl": "cluster",
        "drift_pretrain": "drift",
    }
    return aliases.get(str(method or "byol").lower(), str(method or "byol").lower())
from monitoring.drift_gate import run_drift_gate
from validation.mlflow_logger import MLflowModelLogger

try:
    from models.ensemble import EnsembleMetaLearner, train_meta_learner
    ENSEMBLE = True
except ImportError:
    ENSEMBLE = False
import config.settings as _settings
from config.settings import (
    TRAINING, PRETRAIN, RL, FEATURES, SIZING, RISK, LABELING, HARDWARE_PROFILES, PATHS, MONITORING,
    CURRICULUM as SETTINGS_CURRICULUM, EXECUTION as SETTINGS_EXECUTION,
    ENSEMBLE as SETTINGS_ENSEMBLE,
)
from config.strategy_profiles import STRATEGY_PROFILES, strategy_profile
try:
    from config.settings import GPU as _GPU_CFG
except ImportError:
    _GPU_CFG = {}


def _sharpe_ann_factor(args=None) -> float:
    """Annualization factor for val Sharpe ΓÇö YAML via args, else settings default."""
    if args is not None:
        val = getattr(args, "sharpe_annualization_factor", None)
        if val is not None:
            return float(val)
    return float(TRAINING.get("sharpe_annualization_factor", 1.0))


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: F401 — used in dynamic eval contexts
    import numcodecs
    from torch.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, Dataset, IterableDataset
    TORCH = True
except ImportError:
    print("[ERROR] PyTorch not installed. pip install torch"); sys.exit(1)

# -- Storage backends: Zarr (primary) + NPY memmap (fallback) -----------------
# Zarr advantages over legacy flat-file approaches:
#   ΓÇó Concurrent multi-process reads without locking (no retry loops)
#   ΓÇó LZ4/Blosc compression ΓÇö 3-5├ù faster decompression at similar ratios
#   ΓÇó Directory store = O(1) native append, no pre-allocated maxshape
#   ΓÇó Cloud-native (S3/GCS/Azure backends via fsspec)
# NPY memmap fallback: zero extra dependencies, fastest raw read, no compression.
try:
    import zarr
    import numcodecs
    numcodecs.blosc.use_threads = False
    if hasattr(numcodecs.blosc, 'set_nthreads'):
        numcodecs.blosc.set_nthreads(1)
    from numcodecs import Blosc as _Blosc
    import os
    ZARR = True
    try:
        _ZARR_MAJOR = int(str(getattr(zarr, "__version__", "0")).split(".")[0])
    except Exception:
        _ZARR_MAJOR = 0
    _ZARR_V3 = _ZARR_MAJOR >= 3

    def _zarr_open_group(path: str, mode: str):
        """
        Zarr v3 changed compression/codec APIs; this project uses v2-style
        numcodecs compressors. Force zarr_format=2 on v3 so create_dataset(...,
        compressor=...) keeps working and stores stay consistent.
        """
        if mode == "w" and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            
        if _ZARR_V3:
            return zarr.open_group(path, mode=mode, zarr_format=2)
        return zarr.open_group(path, mode=mode)
    def _zarr_create(group, name: str, **kwargs):
        if hasattr(group, "create_array"):
            return group.create_array(name, **kwargs)
        if hasattr(group, "array"):
            return group.array(name, **kwargs)
        if hasattr(group, "create_dataset"):
            return group.create_dataset(name, **kwargs)
        raise AttributeError("Zarr Group has no supported create method")
except ImportError:
    ZARR = False
    print("[WARN] zarr not installed ΓÇö using NPY memmap fallback. "
          "pip install zarr numcodecs")

try:
    import wandb; WANDB = True
except ImportError:
    WANDB = False

_WANDB_BROKEN = False


def _safe_wandb_log(run, payload: dict, *, step=None) -> None:
    global _WANDB_BROKEN
    if not (WANDB and run is not None) or _WANDB_BROKEN:
        return
    try:
        if step is None:
            run.log(payload)
        else:
            run.log(payload, step=step)
    except Exception as exc:
        _WANDB_BROKEN = True
        print(f"[W&B] Logging disabled after failure: {exc}")


def _safe_wandb_summary_update(run, payload: dict) -> None:
    global _WANDB_BROKEN
    if not (WANDB and run is not None) or _WANDB_BROKEN:
        return
    try:
        run.summary.update(payload)
    except Exception as exc:
        _WANDB_BROKEN = True
        print(f"[W&B] Summary updates disabled after failure: {exc}")

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
    TENSORBOARD = True
except ImportError:
    TENSORBOARD = False

try:
    from monitoring.rich_display import RichTrainingDisplay as _RichDisplay
    RICH_DISPLAY = True
except Exception:
    RICH_DISPLAY = False

try:
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING); OPTUNA = True
except ImportError:
    OPTUNA = False

from sklearn.preprocessing import StandardScaler
from infrastructure.numerics import sanitize_array

def run_preflight_sanity_checks(model, device, loader, args):
    """
    Perform pre-flight checks to catch bugs/errors before full training:
    1. Check for NaNs/Infs in a sample batch.
    2. Verify model forward/backward pass stability.
    3. Check GPU memory headroom.
    """
    print("[Preflight] Running sanity checks...")
    model.eval()
    try:
        # 1. Sample Batch Check
        batch = next(iter(loader))
        xb, yb = batch[0].to(device), batch[1].to(device)
        xb = _crop_to_seq_len(xb, getattr(args, "seq_len", None))
        
        if not torch.isfinite(xb).all():
            raise RuntimeError("Input features contain NaNs or Infs! Check your data source or normalization.")
        if not torch.isfinite(yb).all():
            raise RuntimeError("Labels contain NaNs or Infs! Check your labeling logic.")
            
        # 2. Forward Pass
        with torch.no_grad():
            out = model(xb)
            # Handle MultiTaskWrapper tuples
            out_tensor = out[0] if isinstance(out, tuple) else out
            if not torch.isfinite(out_tensor).all():
                raise RuntimeError("Model produced NaNs/Infs in forward pass! Initial weights may be too large.")
        
        # 3. Memory Check
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info(device)
            free_gb = free / 1024**3
            total_gb = total / 1024**3
            print(f"[Preflight] VRAM: {free_gb:.2f}GB free / {total_gb:.2f}GB total.")
            if free_gb < 1.0:
                print("[Preflight] WARNING: Less than 1GB VRAM free. You may hit OOM soon.")
            
        print("[Preflight] PASS: Data and model sanity checks complete.")
    except StopIteration:
        print("[Preflight] WARNING: Loader is empty. Skipping sanity checks.")
    except Exception as e:
        print(f"[Preflight] FATAL: {e}")
        raise


def _crop_to_seq_len(x, seq_len):
    """Use the most recent bars when a cached window is longer than a model profile.

    INF-001: Now logs when cropping occurs so data loss is visible in training logs.
    """
    if seq_len is None:
        return x
    target = int(seq_len)
    if target <= 0:
        return x
    if getattr(x, "ndim", 0) >= 3 and x.shape[1] > target:
        original_len = x.shape[1]
        dropped = original_len - target
        if dropped > target * 0.5:
            print(f"[WARN] _crop_to_seq_len: dropping {dropped}/{original_len} bars "
                  f"({dropped/original_len*100:.0f}%) to fit seq_len={target}. "
                  f"Consider increasing seq_len or reducing cache window.")
        return x[:, -target:, :]
    return x


# -----------------------------------------------------------------------------
# CUSTOM LOSSES (TRADING-AWARE)
# -----------------------------------------------------------------------------

class DirectionalHuberLoss(nn.Module):
    """Huber magnitude loss + extra penalty when direction is wrong."""

    def __init__(self, delta: float = 1.0, direction_weight: float = 0.5):
        super().__init__()
        self.huber = HuberLoss(delta=delta)
        self.direction_weight = float(direction_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _match_target_shape(pred, target)
        base = self.huber(pred, target)
        # Penalize opposite-sign predictions more than near-zero misses.
        wrong_sign = (pred * target) < 0
        dir_pen = wrong_sign.float() * (pred - target).abs()
        return base + self.direction_weight * dir_pen.mean()


class SharpeProxyLoss(nn.Module):
    """Minimize -Sharpe proxy while keeping pointwise stability via Huber.



    Scale balance note

    ------------------

    With ann=325 (1-min scalping) and sharpe_weight=0.6, naively computing

    `loss = Huber - sharpe_weight * (mean/std) * ann` creates a ~195Γö£├╣ gradient

    imbalance (0.6 Γö£├╣ 325) vs the Huber term. This would let the Sharpe gradient

    completely swamp the pointwise Huber stability.



    Fix: the *annualized* Sharpe is stored for logging, but the gradient term

    uses sqrt(ann) scaling ╬ô├ç├╢ which is the statistically correct annualization

    for a mean/std ratio as the number of observations grows. This keeps the

    Sharpe component within the same gradient magnitude as Huber.

    """


    def __init__(self, delta: float = 1.0, sharpe_weight: float = 0.2, eps: float = 1e-8, ann: float = 1.0):
        super().__init__()
        self.huber = HuberLoss(delta=delta)
        self.sharpe_weight = float(sharpe_weight)
        self.eps = float(eps)
        self.ann = float(ann)
        # Config supplies pre-computed sqrt (e.g. 325 = sqrt(252*420)), use directly.
        self._ann_sqrt = float(ann)


    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight=None) -> torch.Tensor:
        target = _match_target_shape(pred, target)
        try:
            base = self.huber(pred, target, weight=weight)
        except TypeError:
            base = self.huber(pred, target)
        # Use tanh as a differentiable proxy for sign so gradients flow through the Sharpe term
        direction = torch.tanh(pred)
        returns = (direction * target).flatten()
        mean = returns.mean()
        # Use sqrt(var + eps) to avoid NaN gradients when std=0 (constant predictions at start)
        var = returns.var(unbiased=False)
        std = torch.sqrt(var + self.eps)
        sharpe_gradient = (mean / std) * self._ann_sqrt

        # Minimize negative Sharpe to maximize risk-adjusted returns.
        return base - self.sharpe_weight * sharpe_gradient



# -----------------------------------------------------------------------------
# ARGUMENT PARSING
# -----------------------------------------------------------------------------

# Maps YAML keys (section.key) -> argparse dest names
_YAML_MAP = {
    "strategy.mode":             "strategy_mode",
    "strategy.bar_freq":         "bar_freq",
    "strategy.lookahead_bars":   "lookahead_bars",
    "strategy.profit_target_atr": "profit_target_atr",
    "strategy.stop_loss_atr":    "stop_loss_atr",
    "data.source":              "data_source",
    "data.pair":                "pair",
    "data.pairs":               "pairs",
    "data.pair_embed_dim":      "pair_embed_dim",
    "data.pair_align":          "pair_align",
    "data.corr_window":         "corr_window",
    "data.corr_window_long":    "corr_window_long",
    "data.momentum_window":     "momentum_window",
    "data.start":               "data_start",
    "data.end":                 "data_end",
    "data.full_day_data":       "full_day_data",
    "data.n_ticks":             "n_ticks",
    "data.chunk_size":          "chunk_size",
    "data.real_data_window_days": "real_data_window_days",
    "data.window_batch_days": "window_batch_days",
    "data.dataset_build_workers": "dataset_build_workers",
    "data.parallel_window_workers": "parallel_window_workers",
    "data.use_cache":           None,          # handled as force_rebuild inversion below
    "model.name":               "model",
    "model.all_models":         "all_models",
    "model.hidden_size":        "hidden_size",
    "model.d_model":            "d_model",
    "model.nhead":              "nhead",
    "model.num_layers":         "num_layers",
    "model.dropout":            "dropout",
    "training.epochs":          "epochs",
    "training.batch_size":      "batch_size",
    "training.lr":              "lr",
    "training.seq_len":         "seq_len",
    "training.patience":        "patience",
    "training.val_split":       "val_split",
    "training.tune_split":      "tune_split",
    "training.curriculum_gate_metric": "curriculum_gate_metric",
    "training.loss":            "loss",
    "training.label_method":    "label_method",
    "training.early_stop_metric": "early_stop_metric",
    "training.early_stop_min_delta": "early_stop_min_delta",
    "training.direction_weight": "direction_weight",
    "training.sharpe_weight": "sharpe_weight",
    "training.sharpe_annualization_factor": "sharpe_annualization_factor",
    "training.save_every":      "save_every",
    "training.grad_clip":       "grad_clip",
    "training.grad_accum_steps": "grad_accum_steps",
    "training.swa_enabled":     "swa_enabled",
    "training.swa_start_frac":  "swa_start_frac",
    "training.swa_lr":          "swa_lr",
    "training.weight_decay":    "weight_decay",
    "training.seed":            "seed",
    "training.lr_schedule":     "lr_schedule",
    "training.lr_warmup_epochs": "lr_warmup_epochs",
    "training.lr_warmup_pct":   "lr_warmup_pct",
    "training.lr_min_ratio":    "lr_min_ratio",
    "training.onecycle_pct_start": "onecycle_pct_start",
    "training.onecycle_max_lr_mult": "onecycle_max_lr_mult",
    "training.amp":             "amp",
    "training.resume":          "resume",
    "training.training_memory": "training_memory",
    "training.label_smoothing": "label_smoothing",

    "training.cross_asset_mode": "cross_asset_mode",
    "training.cross_asset_provider": "cross_asset_provider",
    "news.historical_mode":     "historical_news_mode",
    "news.historical_news_file": "historical_news_file",
    "news.economic_calendar_file": "economic_calendar_file",
    # Backward-compat aliases for older run.yaml layouts
    "data.historical_news_mode": "historical_news_mode",
    "data.historical_news_file": "historical_news_file",
    "data.economic_calendar_file": "economic_calendar_file",
    "backtest.execution_delay_bars": "execution_delay_bars",
    "walk_forward.enabled":     "walk_forward_cv",
    "walk_forward.folds":       "walk_forward_folds",
    "multitask.enabled":        "multitask",
    "multitask.w_ret":          "mt_w_ret",
    "multitask.w_conf":         "mt_w_conf",
    "multitask.class_balance_weight": "mt_class_balance_weight",

    "multitask.entropy_weight":  "mt_entropy_weight",

    "multitask.direction_weight_floor": "mt_direction_weight_floor",

    "multitask.focal_gamma":     "mt_focal_gamma",
    "direction_training.probe":  "direction_probe",
    "direction_training.probe_epochs": "direction_probe_epochs",
    "direction_training.probe_samples": "direction_probe_samples",
    "direction_training.warmup_epochs": "direction_warmup_epochs",
    "direction_training.min_true_class_share": "direction_min_true_class_share",
    "direction_training.min_pred_class_share": "direction_min_pred_class_share",
    "direction_training.max_pred_class_share": "direction_max_pred_class_share",
    "direction_training.min_recall": "direction_min_recall",

    "pretrain.enabled":         "pretrain",
    "pretrain.ablation":        "pretrain_ablation",
    "pretrain.ablation_models": "pretrain_ablation_models",

    "pretrain.method":          "pretrain_method",
    "pretrain.epochs":          "pretrain_epochs",
    "pretrain.regime_aware":    "pretrain_regime",
    "pretrain.max_epochs":      "pretrain_max_epochs",
    "pretrain.min_epochs":      "pretrain_min_epochs",
    "pretrain.handoff_patience": "pretrain_handoff_patience",
    "pretrain.handoff_min_delta": "pretrain_handoff_min_delta",
    "pretrain.handoff_loss":    "pretrain_handoff_loss",
    "pretrain.lr":              "pretrain_lr",
    "pretrain.batch":           "pretrain_batch",
    "pretrain.projection_dim":  "pretrain_projection_dim",
    "pretrain.pred_dim":        "pretrain_pred_dim",
    "pretrain.ema_decay":       "pretrain_ema_decay",
    "pretrain.sample_windows":  "pretrain_sample_windows",
    "pretrain.blocks_per_epoch": "pretrain_blocks_per_epoch",
    "pretrain.mask_prob":       "pretrain_mask_prob",
    "pretrain.recon_hidden_dim": "pretrain_recon_hidden_dim",
    "pretrain.latent_dim":      "pretrain_latent_dim",
    "pretrain.vae_beta":        "pretrain_vae_beta",
    "pretrain.n_clusters":      "pretrain_n_clusters",
    "pretrain.forecast_horizon": "pretrain_forecast_horizon",
    "pretrain.drift_margin":    "pretrain_drift_margin",
    "ensemble.enabled":         "train_ensemble",
    "ensemble.epochs":          "ensemble_epochs",
    "ensemble.div_weight":      "ensemble_div_weight",
    "ensemble.deploy":          "deploy_ensemble",
    "ensemble.explicit_diversity": "ensemble_explicit_diversity",
    "ensemble.member_seed_offset": "ensemble_member_seed_offset",
    "ensemble.member_lr_jitter": "ensemble_member_lr_jitter",
    "ensemble.member_dropout_jitter": "ensemble_member_dropout_jitter",
    "rl.enabled":               "rl_train",
    "rl.algo":                  "rl_algo",
    "rl.episodes":              "rl_episodes",
    "rl.episode_len":           "rl_episode_len",
    "rl.encoder_obs":           "rl_encoder_obs",
    "rl.val_frac":              "rl_val_frac",
    "rl.min_val_sharpe":        "rl_min_val_sharpe",
    "rl.all_models":            "rl_all_models",
    "rl.deploy":                "deploy_rl",
    "pretrain.temperature":     "pretrain_temperature",
    "calibration.overconf_penalty": "overconf_penalty",
    "calibration.overconf_weight": "overconf_weight",
    "calibration.overconf_threshold": "overconf_threshold",
    "calibration.calibrate":     "calibrate",
    "hardware.profile":         "hardware_profile",
    "hardware.num_workers":     "num_workers",
    "hardware.prefetch_factor": "prefetch_factor",
    "hardware.pin_memory":      "pin_memory",
    "hardware.persistent_workers": "persistent_workers",
    "hardware.val_num_workers": "val_num_workers",
    "hardware.val_prefetch_factor": "val_prefetch_factor",
    "hardware.thread_prefetch_batches": "thread_prefetch_batches",
    "tracking.wandb_project":   "wandb_project",
    "tracking.run_name":        "run_name",
    "tracking.no_wandb":        "no_wandb",
    "tracking.auto_tune":       "auto_tune",

    "tracking.dry_tune":        "dry_tune",

    "tracking.ollama_auto_tune": "ollama_auto_tune",
    "distillation.teacher_model": "teacher_model",
    "distillation.teacher_ckpt": "teacher_ckpt",
    "distillation.alpha":       "distill_weight",
    "distillation.temperature": "distill_temperature",
    "quick.enabled":            "quick_mode",
    "monitoring.drift_gate":    "drift_gate",
    "monitoring.drift_fail_open": "drift_fail_open",
    "monitoring.drift_baseline_samples": "drift_baseline_samples",
    "monitoring.drift_live_samples": "drift_live_samples",
    "monitoring.drift_psi_threshold": "drift_psi_threshold",
    "monitoring.drift_ks_pvalue_threshold": "drift_ks_pvalue_threshold",
    "monitoring.drift_ks_statistic_threshold": "drift_ks_statistic_threshold",
    "diversity_loss.weight":    "div_weight",
    "diversity_loss.same_role_mult": "same_role_mult",
    "data.integrity_gate":      "integrity_gate",
    "data.auto_rebuild_on_mismatch": "auto_rebuild_on_mismatch",
    "paths.checkpoint_dir":     "checkpoint_dir",
    "paths.data_cache":         "data_cache",
    # XGBoost baseline
    "xgboost.enabled":          "xgb_enabled",
    "xgboost.task":             "xgb_task",
    "xgboost.sequence_mode":    "xgb_sequence_mode",
    "xgboost.n_estimators":     "xgb_n_estimators",
    "xgboost.max_depth":        "xgb_max_depth",
    "xgboost.learning_rate":    "xgb_learning_rate",
    "xgboost.subsample":        "xgb_subsample",
    "xgboost.colsample_bytree": "xgb_colsample_bytree",
    "xgboost.min_child_weight": "xgb_min_child_weight",
    "xgboost.gamma":            "xgb_gamma",
    "xgboost.reg_alpha":        "xgb_reg_alpha",
    "xgboost.reg_lambda":       "xgb_reg_lambda",
    "xgboost.objective":        "xgb_objective",
    "xgboost.eval_metric":      "xgb_eval_metric",
    "xgboost.early_stopping_rounds": "xgb_early_stopping_rounds",
    "xgboost.folds":            "xgb_folds",
    "xgboost.tune":             "xgb_tune",
    "xgboost.tune_trials":      "xgb_tune_trials",
    "xgboost.max_samples":      "xgb_max_samples",
    "xgboost.feature_importance": "xgb_feature_importance",
    "xgboost.feature_importance_top_n": "xgb_feature_importance_top_n",
    # CatBoost baseline
    "catboost.enabled":          "cb_enabled",
    "catboost.task":             "cb_task",
    "catboost.sequence_mode":    "cb_sequence_mode",
    "catboost.n_estimators":     "cb_n_estimators",
    "catboost.max_depth":        "cb_max_depth",
    "catboost.learning_rate":    "cb_learning_rate",
    "catboost.subsample":        "cb_subsample",
    "catboost.colsample_bytree": "cb_colsample_bytree",
    "catboost.min_child_weight": "cb_min_child_weight",
    "catboost.gamma":            "cb_gamma",
    "catboost.reg_alpha":        "cb_reg_alpha",
    "catboost.reg_lambda":       "cb_reg_lambda",
    "catboost.objective":        "cb_objective",
    "catboost.eval_metric":      "cb_eval_metric",
    "catboost.early_stopping_rounds": "cb_early_stopping_rounds",
    "catboost.folds":            "cb_folds",
    "catboost.tune":             "cb_tune",
    "catboost.tune_trials":      "cb_tune_trials",
    "catboost.max_samples":      "cb_max_samples",
    "catboost.feature_importance": "cb_feature_importance",
    "catboost.feature_importance_top_n": "cb_feature_importance_top_n",
    # Validation / purged CV
    "validation.method":        "validation_method",
    "validation.n_splits":      "validation_n_splits",
    "validation.purge_bars":    "validation_purge_bars",
    "validation.embargo_bars":  "validation_embargo_bars",
    "validation.min_train_size": "validation_min_train_size",
}


def _apply_yaml_config(parser: argparse.ArgumentParser, config_path: str) -> None:
    """Load config/run.yaml and set argparse defaults from it."""
    if not _YAML:
        print("[Config] PyYAML not installed ΓÇö ignoring --config. pip install pyyaml")
        return
    path = Path(config_path)
    if not path.exists():
        print(f"[Config] WARN: config file not found: {config_path}")
        return

    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = _yaml.safe_load(fh)
    except Exception as e:
        print(
            f"[Config] YAML parse failed for {config_path}: {e}\n"
            "[Config] Continuing with argparse defaults + explicit CLI flags."
        )
        return
    if cfg is None:
        cfg = {}

    defaults: dict = {}
    distillation_enabled = bool((cfg.get("distillation") or {}).get("enabled", False))
    for yaml_key, dest in _YAML_MAP.items():
        if yaml_key.startswith("distillation.") and not distillation_enabled:
            continue
        section, key = yaml_key.split(".", 1)
        val = (cfg.get(section) or {}).get(key)
        if val is None:
            continue
        if dest is None:
            # data.use_cache=false -> force_rebuild=true
            if yaml_key == "data.use_cache":
                defaults["force_rebuild"] = not bool(val)
            continue
        # Blank strings mean "use the hardcoded default"
        if isinstance(val, str) and val.strip() == "":
            continue
        defaults[dest] = val

    if isinstance(cfg.get("curriculum"), dict):
        defaults["curriculum"] = cfg["curriculum"]
    if isinstance(cfg.get("feature_ablation"), dict):

        defaults["feature_ablation"] = cfg["feature_ablation"]

    if isinstance(cfg.get("execution"), dict):
        defaults["execution"] = cfg["execution"]

    pretrain_sec = cfg.get("pretrain") or {}
    if isinstance(pretrain_sec.get("augmentations"), dict):
        defaults["pretrain_augmentations"] = pretrain_sec["augmentations"]

    rl_sec = cfg.get("rl") or {}
    if isinstance(rl_sec.get("reward"), dict):
        defaults["rl_reward_weights"] = rl_sec["reward"]
    rl_overrides = {}
    if isinstance(rl_sec.get("dqn"), dict):
        rl_overrides["dqn"] = rl_sec["dqn"]
    if isinstance(rl_sec.get("ppo"), dict):
        rl_overrides["ppo"] = rl_sec["ppo"]
    if rl_overrides:
        defaults["rl_algo_overrides"] = rl_overrides

    parser.set_defaults(**defaults)
    print(f"[Config] Loaded {config_path}")


def _sync_runtime_config(args) -> None:
    """Apply YAML-only nested config blocks to modules that read config.settings."""
    execution = getattr(args, "execution", None)
    if isinstance(execution, dict):
        _settings.EXECUTION.update(execution)
        SETTINGS_EXECUTION.update(execution)


def parse_args():
    p = argparse.ArgumentParser(description="Forex Model ΓÇö 20M Tick GPU Trainer")
    p.set_defaults(curriculum=SETTINGS_CURRICULUM)
    p.set_defaults(execution=SETTINGS_EXECUTION)
    p.add_argument("--config", type=str, default=None,
                   help="Path to a YAML run config (e.g. config/run.yaml). "
                        "Values are used as defaults; explicit CLI flags override them.")

    # Strategy profile
    p.add_argument("--strategy-mode", type=str, default="scalping",
                   choices=sorted(STRATEGY_PROFILES.keys()),
                   help="Trading horizon profile. scalping=1min fast trades; normal=1h slower trades.")
    p.add_argument("--bar-freq", type=str, default=None,
                   help="Bar frequency for feature/label construction, e.g. 1min, 15min, 1h.")
    p.add_argument("--lookahead-bars", type=int, default=None,
                   help="Label forward horizon in bars. Defaults to the selected strategy profile.")
    p.add_argument("--profit-target-atr", type=float, default=None,
                   help="ATR profit barrier for triple-barrier/normal labels.")
    p.add_argument("--stop-loss-atr", type=float, default=None,
                   help="ATR stop barrier for triple-barrier/normal labels.")

    # Scale
    p.add_argument("--n-ticks",      type=int,   default=20_000_000,
                   help="Total tick count to train on (default: 20M)")
    p.add_argument("--chunk-size",   type=int,   default=500_000,
                   help="Ticks per processing chunk (RAM safety valve)")
    p.add_argument(
        "--real-data-window-days",
        type=int,
        default=0,
        help="Days per real-data ingestion window. 0 = auto from --chunk-size.",
    )
    p.add_argument(
        "--window-batch-days",
        type=int,
        default=1,
        help="Group N consecutive date windows into one batch. "
             "Effective window = real_data_window_days * window_batch_days. "
             "Larger batches give features more lookback context (default: 1).",
    )

    # Data source
    p.add_argument("--data-source",  type=str,   default="dukascopy",
                   choices=["synthetic","dukascopy","tds","lmax_historical","auto","databento"],
                   help="Which data source to use")
    p.add_argument("--data-start",   type=str,   default="2008-01-01")
    p.add_argument("--data-end",     type=str,
                   default=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"))
    p.add_argument("--pair",          type=str,   default="EURUSD")
    p.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Comma-separated pairs for joint multi-pair training, e.g. EURUSD,GBPUSD,USDJPY. "
             "Overrides --pair when set. Can also be a list in config/run.yaml under data.pairs.",
    )
    p.add_argument(
        "--pair-embed-dim",
        type=int,
        default=0,
        help="Learnable pair embedding size (int). Appended to each pair's features before "
             "the backbone. 0 = disabled (pairs are simply concatenated on the feature axis).",
    )
    p.add_argument(
        "--corr-window",
        type=int, default=20,
        help="Short rolling correlation window in bars for MultiPairWrapper cross-pair features. "
             "Default: 20.",
    )
    p.add_argument(
        "--corr-window-long",
        type=int, default=60,
        help="Long rolling correlation window in bars for MultiPairWrapper. Default: 60.",
    )
    p.add_argument(
        "--momentum-window",
        type=int, default=20,
        help="Windowed relative momentum lookback in bars for MultiPairWrapper. Default: 20.",
    )
    p.add_argument(
        "--pair-align",
        type=str,
        default="inner",
        choices=["inner", "outer"],
        help="Timestamp alignment across pairs: inner=common bars only (default), "
             "outer=fill missing bars with NaN.",
    )
    p.add_argument(
        "--full-day-data",
        action="store_true",
        help="Dukascopy: load all 24h (00ΓÇô23 UTC). Default is session-only (07ΓÇô17 UTC).",
    )

    # Model
    p.add_argument("--model",        type=str,   default="haelt",
                   choices=["tft","transformer","haelt","mamba","gnn","expert","catboost"])
    p.add_argument("--all-models", dest="all_models", action="store_true")

    p.add_argument("--no-all-models", dest="all_models", action="store_false",

                   help="Force a single-model run even if config model.all_models=true.")

    p.add_argument("--models", type=str, default="",

                   help="Comma-separated model list for --all-models, e.g. transformer,expert. "

                        "Empty means every registered supervised architecture.")

    p.add_argument("--div-weight",      type=float, default=0.10,
                   help="C: DiversityLoss weight during post-training diversity fine-tuning")
    p.add_argument("--same-role-mult",  type=float, default=2.0,
                   help="C: Extra diversity penalty multiplier for same-role model pairs")

    # Training
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch-size",   type=int,   default=2048,
                   help="Batch size ΓÇö 2048 optimal for 20M samples on RTX 4090")
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--lr-schedule",  type=str, default="warmup_cosine",
                   choices=["onecycle", "warmup_cosine"],
                   help="Learning-rate schedule: onecycle (legacy default) or warmup_cosine.")
    p.add_argument("--lr-warmup-epochs", type=int, default=3,
                   help="Warmup epochs used by warmup_cosine schedule.")
    p.add_argument("--lr-warmup-pct", type=float, default=0.1,
                   help="Warmup fraction fallback for warmup_cosine when warmup_epochs <= 0.")
    p.add_argument("--lr-min-ratio", type=float, default=0.05,
                   help="Final LR ratio for warmup_cosine (final_lr = lr * lr_min_ratio).")
    p.add_argument("--onecycle-pct-start", type=float, default=0.1,
                   help="OneCycleLR warmup fraction (legacy path).")
    p.add_argument("--onecycle-max-lr-mult", type=float, default=10.0,
                   help="OneCycleLR peak multiplier over base lr (legacy path).")
    p.add_argument("--seq-len",      type=int,   default=60)
    p.add_argument("--patience",     type=int,   default=10)
    p.add_argument("--seed",         type=int, default=1337,
                   help="Global random seed (A-M3: seeded by default for reproducibility). "
                        "Pass a different int to vary runs; threads into numpy/torch/augmenter RNGs.")
    p.add_argument("--deterministic", dest="deterministic",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="A-M3: force fully deterministic kernels (cudnn.deterministic=True, "
                        "benchmark=False, torch.use_deterministic_algorithms). Slower but reproducible.")
    p.add_argument("--val-split",
                    type=float,
                    default=0.1,
                    help="Validation fraction (default 0.1)")
    p.add_argument("--tune-split",
                    type=float,
                    default=0.05,
                    help="Fraction reserved for auto-tune evaluation, separate from val (default 0.05). "
                         "Set to 0 to disable three-way split (reverts to val reuse).")
    p.add_argument("--curriculum-gate-metric",
                    type=str,
                    default="train_loss",
                    choices=["train_loss", "val_sharpe"],
                    help="Metric used for curriculum progression gating. 'train_loss' (default) "
                         "prevents val set leakage into curriculum decisions (SYS-005). "
                         "'val_sharpe' restores legacy behavior.")
    p.add_argument("--amp",    action="store_true", default=False,
                   help="Enable AMP (automatic mixed precision) for faster training. Disabled by default to avoid NaNs.")
    p.add_argument("--no-amp", action="store_true", default=False,
                   dest="no_amp",
                   help="Disable AMP ΓÇö forces FP32 training. Eliminates NaN-grad skips on 2240-feature inputs at the cost of ~30% slower throughput.")
    p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help=(
            "AMP precision dtype. auto=BF16 on Ada/Ampere CC>=8.0, FP16 on older GPUs. "
            "bf16: stable, no GradScaler, preferred for RTX 40-series Tensor Cores. "
            "fp16: needs GradScaler, use if bf16 not supported. fp32: no AMP (debug)."
        ),
    )
    p.add_argument(
        "--cross-asset-mode",
        type=str,
        default="auto",
        choices=["auto", "real", "synthetic", "off"],
        help="Cross-asset features source: auto=real for real FX data, synthetic for synthetic FX; "
             "real=attempt external commodities/yields download; synthetic/off disables external fetch",
    )
    p.add_argument(
        "--cross-asset-provider",
        type=str,
        default="auto",
        choices=["auto", "stooq", "yahoo", "fred", "eodhd"],
        help="Cross-asset data provider. Env CROSS_ASSET_SOURCE overrides this when set.",
    )
    p.add_argument(
        "--sentiment-mode",
        type=str,
        default="finbert",
        choices=["off", "finbert", "auto"],
        help="Sentiment feature mode: finbert=force FinBERT, auto=best available, off=disable sentiment feature columns",
    )
    p.add_argument(
        "--historical-news-mode",
        type=str,
        default="calendar",
        choices=["off", "calendar", "full"],
        help="Offline historical news mode: off=neutral, calendar=economic no-trade events, full=calendar + headline sentiment/counts.",
    )
    p.add_argument(
        "--historical-news-file",
        type=str,
        default=None,
        help="Optional CSV/JSON/JSONL historical headlines file. Defaults to data/raw/news/historical_news_combined.parquet or HISTORICAL_NEWS_FILE.",
    )
    p.add_argument(
        "--economic-calendar-file",
        type=str,
        default=None,
        help="Optional CSV/JSON/JSONL economic calendar file. Defaults to data/raw/eco_calendar/events.csv or ECONOMIC_CALENDAR_FILE.",
    )
    p.add_argument("--grad-clip",    type=float, default=5.0)
    p.add_argument("--grad-accum-steps", type=int, default=2,
                   help="Gradient accumulation steps; effective batch = batch_size * grad_accum_steps")
    p.add_argument("--swa-enabled", dest="swa_enabled",
                   action=argparse.BooleanOptionalAction,
                   default=bool(TRAINING.get("swa_enabled", False)),
                   help="Enable Stochastic Weight Averaging over the final training phase.")
    p.add_argument("--swa-start-frac", type=float,
                   default=float(TRAINING.get("swa_start_frac", 0.75)),
                   help="Fraction of total epochs before SWA starts, e.g. 0.75.")
    p.add_argument("--swa-lr", type=float,
                   default=float(TRAINING.get("swa_lr", 1e-5)),
                   help="Constant learning rate used by the SWA scheduler.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--label-method",
        type=str,
        default="rl_reward",
        choices=["rl_reward", "triple_barrier"],
        help="Supervised targets: RL forward P&L labels vs triple-barrier (ATR barriers + vertical)",
    )
    p.add_argument(
        "--loss",
        type=str,
        default=None,
        choices=["huber", "asymmetric", "cross_entropy", "directional_huber", "sharpe_huber"],
        help="huber/asymmetric/directional_huber/sharpe_huber on scalar targets; "
             "cross_entropy=3-class {-1,0,1} with balanced weights",
    )
    p.add_argument("--direction-weight", type=float, default=0.5,
                   help="Extra wrong-direction penalty multiplier for directional_huber loss")
    p.add_argument("--sharpe-weight", type=float, default=0.2,
                   help="Sharpe proxy weight for sharpe_huber loss")
    p.add_argument("--early-stop-min-delta", type=float, default=0.0,
                   help="Minimum validation improvement required to reset patience")
    p.add_argument("--guard-min-confidence", type=float, default=0.85,
                   help="Minimum confidence required to execute a trade during validation (Disagreement Gating).")
    p.add_argument("--num-workers",  type=int,   default=8,
                   help="DataLoader workers ΓÇö 8 is sweet spot for H100/A100")
    p.add_argument("--prefetch-factor", type=int, default=4,
                   help="DataLoader prefetch (per worker); lower on 16GB RAM PCs")
    p.add_argument("--val-num-workers", type=int, default=None,
                   help="Validation DataLoader workers. Default: auto from train workers.")
    p.add_argument("--val-prefetch-factor", type=int, default=None,
                   help="Validation prefetch factor. Default: auto (lower than train).")
    p.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=None,
                   help="Force DataLoader pin_memory=True for train/val.")
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false",
                   help="Force DataLoader pin_memory=False for train/val.")
    p.add_argument("--persistent-workers", dest="persistent_workers", action="store_true", default=None,
                   help="Force DataLoader persistent_workers=True when workers > 0.")
    p.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false",
                   help="Force DataLoader persistent_workers=False.")
    p.add_argument("--thread-prefetch-batches", type=int, default=2,
                   help="Background-thread prefetch queue depth when workers=0.")
    p.add_argument("--dataset-build-workers", type=int, default=1,
                   help="Parallel threads for loading date windows during dataset build. "
                        "1 = sequential (safe default). 2-4 overlaps tick I/O across windows.")
    p.add_argument("--parallel-window-workers", type=int, default=1,
                   help="Parallel processes for date-window feature engineering + labeling. "
                        "1 = sequential (default). 2-4 parallelises CPU-heavy chunk builds "
                        "across windows using ProcessPoolExecutor.")
    p.add_argument("--hardware-profile", type=str, default=None,
                   choices=list(HARDWARE_PROFILES.keys()) if HARDWARE_PROFILES else None,
                   help="Apply tuned defaults (batch/workers/chunk/prefetch/paths). "
                        "rtx_4060_16gb_ram: RTX 4060 8GB VRAM + 16GB system RAM")

    # Architecture
    p.add_argument("--hidden-size",  type=int,   default=256)
    p.add_argument("--num-layers",   type=int,   default=3)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--d-model",      type=int,   default=256)
    p.add_argument("--nhead",        type=int,   default=8)
    p.add_argument(
        "--fair-sweep",
        action="store_true",
        help="Architecture bake-off: identical hyperparams from run.yaml for every model "
             "(alias for --no-model-profile).",
    )
    p.add_argument(
        "--model-profile",
        dest="model_profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply per-architecture tuned defaults from config/models.py (default: on). "
             "Use --no-model-profile or --fair-sweep for a fair architecture comparison.",
    )
    p.add_argument("--feature-ablation-name", type=str, default="",

                   help="Name recorded in feature_ablation_report.json for this feature ablation run.")

    p.add_argument("--feature-ablation-drop-groups", type=str, default="",

                   help="Comma-separated curriculum feature groups to zero for this run, e.g. news,cross_asset.")

    p.add_argument("--feature-ablation-keep-groups", type=str, default="",

                   help="Comma-separated curriculum feature groups to keep; all other grouped features are zeroed.")

    p.add_argument("--feature-ablation-drop-features", type=str, default="",

                   help="Comma-separated exact feature names to zero for this run.")


    # Pre-training & Ablation
    p.add_argument("--pretrain", action="store_true", help="Enable contrastive pre-training")
    p.add_argument("--ablate-pretrain", action="store_true", help="Run ablation test on pretraining vs no-pretraining")

    # Pre-training
    p.add_argument(
        "--pretrain-method",
        choices=["byol", "tscl", "masked", "vae", "autoencoder", "cluster", "forecast", "drift"],
        default=str(PRETRAIN.get("method", "byol")).lower(),
        help="Self-supervised pretrain: byol (default), tscl, masked, vae, cluster, forecast, drift",
    )
    p.add_argument("--pretrain-epochs",  type=int,   default=30)
    p.add_argument("--pretrain-max-epochs", type=int, default=0,
                   help="Hard cap for pretraining epochs. 0 keeps pretrain_epochs unchanged.")
    p.add_argument("--pretrain-min-epochs", type=int, default=0,
                   help="Minimum pretrain epochs before handoff checks can stop early.")
    p.add_argument("--pretrain-handoff-patience", type=int, default=0,
                   help="Stop pretraining early after this many plateau epochs (0 disables).")
    p.add_argument("--pretrain-handoff-min-delta", type=float, default=0.0,
                   help="Minimum pretrain loss improvement to reset handoff patience.")
    p.add_argument("--pretrain-handoff-loss", type=float, default=float("-inf"),
                   help="Stop pretraining once loss <= threshold after min epochs. Disabled by default.")
    p.add_argument(
        "--pretrain-regime",
        action="store_true",
        help="Use regime-aware TSCL: same-regime positives + cross-regime hard negatives",
    )
    p.add_argument("--pretrain-lr", type=float, default=float(PRETRAIN.get("pretrain_lr", 1e-4)),
                   help="Pretrain optimizer learning rate")
    p.add_argument("--pretrain-batch", type=int, default=int(PRETRAIN.get("pretrain_batch", 256)),
                   help="Preferred pretrain batch size before VRAM safety cap")
    p.add_argument("--pretrain-projection-dim", type=int, default=int(PRETRAIN.get("projection_dim", 256)),
                   help="Projection dimension for BYOL/TSCL heads")
    p.add_argument("--pretrain-pred-dim", type=int, default=int(PRETRAIN.get("pred_dim", 128)),
                   help="BYOL predictor hidden dimension")
    p.add_argument("--pretrain-ema-decay", type=float, default=float(PRETRAIN.get("ema_decay", 0.996)),
                   help="BYOL target-network EMA decay")
    p.add_argument("--pretrain-sample-windows", default="auto",
                   help="Windows loaded per pretrain block, or 'auto' for RAM-based sizing")
    p.add_argument("--pretrain-blocks-per-epoch", default="auto",
                   help="Fresh pretrain blocks per outer epoch, or 'auto' for effective sample volume")
    p.add_argument("--pretrain-mask-prob", type=float, default=float(PRETRAIN.get("mask_prob", 0.20)),
                   help="Masked reconstruction probability when --pretrain-method masked")
    p.add_argument("--pretrain-recon-hidden-dim", type=int, default=int(PRETRAIN.get("recon_hidden_dim", 512)),
                   help="Masked reconstruction decoder hidden size")
    p.add_argument("--pretrain-latent-dim", type=int, default=int(PRETRAIN.get("latent_dim", 64)),
                   help="VAE latent dimension when --pretrain-method vae")
    p.add_argument("--pretrain-vae-beta", type=float, default=float(PRETRAIN.get("vae_beta", 0.001)),
                   help="KL weight for VAE pretrain")
    p.add_argument("--pretrain-n-clusters", type=int, default=int(PRETRAIN.get("n_clusters", 3)),
                   help="k-means clusters for cluster contrastive pretrain")
    p.add_argument("--pretrain-forecast-horizon", type=int, default=int(PRETRAIN.get("forecast_horizon", 5)),
                   help="Future bars to predict in forecast pretext task")
    p.add_argument("--pretrain-drift-margin", type=float, default=float(PRETRAIN.get("drift_margin", 1.0)),
                   help="Target L2 distance between clean and drift-augmented embeddings")
    p.add_argument(
        "--force-pretrain",
        action="store_true",
        help="Delete existing contrastive encoder checkpoint and pretrain from scratch",
    )
    p.add_argument(
        "--multitask",
        action="store_true",
        help="Replace single prediction head with MultiTaskHead "
             "(direction CE + magnitude Huber + confidence BCE)",
    )
    p.add_argument("--mt-w-ret",  type=float, default=0.5,
                   help="Multi-task loss weight for return_hat Huber term (default 0.5)")
    p.add_argument("--mt-w-conf", type=float, default=0.3,
                   help="Multi-task loss weight for confidence BCE term (default 0.3)")
    p.add_argument("--direction-probe", dest="direction_probe",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Run a short balanced direction probe before full supervised training.")
    p.add_argument("--direction-probe-epochs", type=int, default=2,
                   help="Epochs for the pre-training direction probe.")
    p.add_argument("--direction-probe-samples", type=int, default=4096,
                   help="Total samples used by the balanced direction probe.")
    p.add_argument("--direction-warmup-epochs", type=int, default=2,
                   help="Initial epochs trained with balanced direction-only batches.")
    p.add_argument("--direction-min-true-class-share", type=float, default=0.15,
                   help="Minimum train/val true class share required before training.")
    p.add_argument("--direction-min-pred-class-share", type=float, default=0.05,
                   help="Minimum validation predicted share for each direction class.")
    p.add_argument("--direction-max-pred-class-share", type=float, default=0.80,
                   help="Maximum validation predicted share for any one direction class.")
    p.add_argument("--direction-min-recall", type=float, default=0.001,
                   help="Minimum per-class validation recall for direction readiness gates.")
    p.add_argument("--overconf-penalty", dest="overconf_penalty",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Apply training-time overconfidence penalty for regression losses.")
    p.add_argument("--overconf-weight", type=float, default=0.3,
                   help="Weight for the overconfidence penalty.")
    p.add_argument("--overconf-threshold", type=float, default=0.6,
                   help="Absolute prediction threshold that triggers overconfidence checks.")
    p.add_argument("--calibrate", dest="calibrate",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Fit post-training temperature calibration on the validation set.")
    p.add_argument(
        "--train-ensemble",
        action="store_true",
        default=bool(SETTINGS_ENSEMBLE.get("enabled", False)),
        help="After supervised training, train the EnsembleMetaLearner "
             "with diversity penalty across all trained base models",
    )
    p.add_argument("--ensemble-epochs",     type=int,   default=int(SETTINGS_ENSEMBLE.get("epochs", 10)),
                   help="Epochs to train the meta-learner (default 10)")
    p.add_argument("--ensemble-div-weight", type=float, default=float(SETTINGS_ENSEMBLE.get("div_weight", 0.1)),
                   help="Diversity penalty weight for meta-learner training (default 0.1)")
    p.add_argument("--deploy-ensemble", action="store_true",
                   default=bool(SETTINGS_ENSEMBLE.get("deploy", False)),
                   help="After ensemble ONNX export, atomically promote it to production_best.onnx for the C++ server.")
    p.add_argument("--ensemble-explicit-diversity", action="store_true",
                   default=bool(SETTINGS_ENSEMBLE.get("explicit_diversity", False)),
                   help="Apply explicit per-member diversity controls during all-model training.")
    p.add_argument("--ensemble-member-seed-offset", type=int,
                   default=int(SETTINGS_ENSEMBLE.get("member_seed_offset", 997)),
                   help="Seed offset between ensemble members when explicit diversity is enabled.")
    p.add_argument("--ensemble-member-lr-jitter", type=float,
                   default=float(SETTINGS_ENSEMBLE.get("member_lr_jitter", 0.0)),
                   help="Relative LR jitter spread across members (e.g. 0.2 => +/-10%%).")
    p.add_argument("--ensemble-member-dropout-jitter", type=float,
                   default=float(SETTINGS_ENSEMBLE.get("member_dropout_jitter", 0.0)),
                   help="Absolute dropout jitter spread across members (clamped to [0,0.8]).")

    # RL
    p.add_argument("--rl-train",     action="store_true")
    p.add_argument("--rl-algo",      type=str,   default="dqn", choices=["dqn","ppo"])
    p.add_argument("--rl-episodes",  type=int,   default=500)
    p.add_argument("--rl-episode-len", type=int, default=2048,
                   help="A-H1: bars per RL episode (sub-window sampled at a random offset "
                        "each reset). 0 = full series each episode.")
    p.add_argument("--rl-encoder-obs", dest="rl_encoder_obs",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="A-C3: use the frozen supervised encoder embedding as the RL "
                        "observation (connects supervisedΓåÆRL). --no-rl-encoder-obs falls "
                        "back to raw last-timestep features.")
    p.add_argument("--rl-val-frac", type=float, default=0.15,
                   help="Fraction of the RL window held out for validation rollouts.")
    p.add_argument("--rl-min-val-sharpe", type=float, default=-999.0,
                   help="Minimum validation Sharpe required to save rl_*_best.pt.")
    p.add_argument("--rl-all-models", action="store_true", default=False,
                   help="With --all-models, run RL once per trained architecture subfolder.")
    p.add_argument("--deploy-rl", action="store_true",
                   default=bool(RL.get("deploy", False)),
                   help="After RL ONNX export, atomically promote it to production_best.onnx for the C++ server.")
    p.add_argument("--pretrain-temperature", type=float,
                   default=float(PRETRAIN.get("temperature", 0.5)),
                   help="Initial TSCL temperature (learnable during pretrain).")

    # Fine-tune / warm-start (B-C2)
    p.add_argument("--finetune-warm-start", dest="finetune_warm_start",
                   action="store_true", default=False,
                   help="B-C2: load prior production/best weights then CONTINUE supervised "
                        "training on the new window (distinct from --resume, which skips "
                        "training when a best checkpoint already exists).")
    p.add_argument("--warm-start-from", type=str, default=None,
                   help="Explicit checkpoint to warm-start from. Default: production_best.pt "
                        "then the model's own _best.pt.")

    # Promotion gate (B-C1)
    p.add_argument("--promotion-gate", dest="promotion_gate",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="B-C1: after training, backtest the challenger on a held-out forward "
                        "window and run PromotionGate to decide deployment (writes "
                        "<model>_promotion.json). --no-promotion-gate disables.")
    p.add_argument("--force-promotion", action="store_true",
                   help="Bypass the challenger vs production gate and force the promotion.")
    p.add_argument("--promote-forward-frac", type=float, default=0.1,
                   help="Fraction of most-recent samples used as the held-out forward "
                        "window for the promotion-gate backtest (B-C1).")

    # HPO
    p.add_argument("--hparam-search",action="store_true")
    p.add_argument("--n-trials",     type=int,   default=30)
    p.add_argument(
        "--auto-optuna",
        dest="auto_optuna",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overlay config/optuna/run_optuna_best_<model>_<metric>.yaml when present "
             "(default: optuna.auto_load in run.yaml, else true).",
    )

    # Tracking
    p.add_argument("--wandb-project",type=str,   default="forex-scaling-model")
    p.add_argument("--run-name",     type=str,   default=None)
    p.add_argument("--auto-run-dir", action="store_true", default=False,
                   help="Generate a descriptive checkpoint folder under checkpoints/runs "
                        "from model, strategy, pairs, seq_len, folds, and RL/ensemble mode.")
    p.add_argument("--run-dir-root", type=str, default=None,
                   help="Base directory for --auto-run-dir. Defaults to <checkpoint-dir>/runs.")
    p.add_argument("--auto-tune", dest="auto_tune",

                   action=argparse.BooleanOptionalAction,

                   default=True,

                   help="Write auto-tune proposal artifacts after each completed training phase. "

                        "--no-auto-tune disables proposal generation and config nudges.")

    p.add_argument("--dry-tune",     action="store_true", default=False,
                   help="Write auto-tune proposals without mutating config/run.yaml.")
    p.add_argument("--no-wandb",     action="store_true")
    p.add_argument("--ollama-auto-tune", action="store_true", default=False,
                   help="Allow Ollama to edit config and restart training after a run.")
    p.add_argument("--save-every",   type=int,   default=5)

    # Paths
    p.add_argument("--checkpoint-dir", type=str, default=PATHS["checkpoints"])
    p.add_argument("--data-cache", type=str, default=PATHS["data_processed"])
    p.add_argument("--resume", dest="resume",

                   action=argparse.BooleanOptionalAction,

                   default=False,

                   help="Resume model/optimizer state from existing checkpoints. "

                        "Use --no-resume for a clean supervised run even when config/run.yaml enables resume.")

    p.add_argument("--training-memory", dest="training_memory",

                   action=argparse.BooleanOptionalAction,

                   default=True,

                   help="Apply conservative hyperparameter nudges from logs/training_memory.json. "

                        "Use --no-training-memory for a clean baseline/fresh run.")

    p.add_argument("--retrain-completed-models", action="store_true", default=False,
                   help="With --all-models --resume, retrain models that already have "
                        "completed artifacts instead of skipping to unfinished models.")
    p.add_argument("--force-rebuild", "--rebuild-cache", dest="force_rebuild", action="store_true",

                   help="Ignore cached Zarr/NPY store and rebuild from scratch")
    p.add_argument("--quick-mode", action="store_true",
                   help="Fast sanity run: fewer folds/epochs, no ensemble or RL.")
    p.add_argument("--drift-gate", dest="drift_gate",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Run a pre-training input-distribution drift gate on cached features. "
                        "Keep disabled for historical training runs.")
    p.add_argument("--drift-fail-open", dest="drift_fail_open", action="store_true", default=False,
                   help="If drift gate check errors, continue training with a warning.")
    p.add_argument("--drift-baseline-samples", type=int, default=20_000,
                   help="Baseline sample rows from cache start for drift gate.")
    p.add_argument("--drift-live-samples", type=int, default=5_000,
                   help="Recent sample rows from cache end for drift gate.")
    p.add_argument("--drift-psi-threshold", type=float, default=float(MONITORING.get("psi_threshold", 0.2)),
                   help="PSI threshold for drift gate fail condition.")
    p.add_argument("--drift-ks-pvalue-threshold", type=float, default=float(MONITORING.get("ks_pvalue_threshold", 0.05)),
                   help="KS p-value threshold for drift gate fail condition.")
    p.add_argument("--drift-ks-statistic-threshold", type=float, default=0.05,
                   help="KS D-statistic (effect-size) floor for drift gate. "
                        "KS only fails when BOTH p-value < threshold AND D-stat >= this value. "
                        "Prevents false alarms on large datasets (default 0.05).")
    p.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Run torch.profiler for 3 warm-up + 5 active batches then exit. "
            "Outputs a Chrome trace to logs/profile_<model>_<run>.json. "
            "Open in chrome://tracing or https://ui.perfetto.dev. "
            "Reveals whether you are compute-bound, memory-bound, or input-bound. "
            "Recommended before tuning batch size or enabling torch.compile."
        ),
    )
    p.add_argument("--integrity-gate", dest="integrity_gate", action="store_true", default=True,
                   help="Fail fast when cached X/y lengths are inconsistent.")
    p.add_argument("--no-integrity-gate", dest="integrity_gate", action="store_false",
                   help="Disable strict cache integrity gate (not recommended).")
    p.add_argument("--auto-rebuild-on-mismatch", action="store_true",
                   help="If cache integrity fails, delete cache/sidecars and rebuild automatically.")
    p.add_argument("--pretrain-ablation", type=str, nargs="?", const="true", choices=["true", "false", "auto"], default="auto",
                   help="If true or auto (for transformer/haelt), runs a full training baseline with NO PRETRAIN first.")
    p.add_argument("--pretrain-ablation-models", type=str, default="",

                   help="Comma-separated model list used when --pretrain-ablation auto. "

                        "Default/config recommendation: tft,transformer,haelt.")

    p.add_argument("--ignore-manifest", action="store_true",
                   help="Bypass dataset_manifest.json checks and force load existing cache.")
    p.add_argument(
        "--walk-forward-cv",
        action="store_true",
        help="Purged walk-forward CV (train past / val future, embargo=seq_len+lookahead+delay) instead of one split",
    )
    p.add_argument(
        "--walk-forward-folds",
        type=int,
        default=None,
        help="Number of walk-forward folds (default: TRAINING['walk_forward_folds'])",
    )
    p.add_argument(
        "--early-stop-metric",
        type=str,
        default=None,
        choices=["loss", "sharpe"],
        help="Checkpoint early stopping on val loss or validation Sharpe proxy (default: TRAINING)",
    )
    p.add_argument(
        "--execution-delay-bars",
        type=int,
        default=1,
        help="Bars between model signal and executable entry; used for training labels and backtests.",
    )
    p.add_argument("--data-quality-check", action="store_true",
                   help="Run the data quality check script on the Zarr cache before training.")
    p.add_argument("--skip-training", action="store_true",
                   help="Exit after data quality check (or dataset build) without training.")
    p.add_argument(
        "--validate-config",
        action="store_true",
        help="Audit run.yaml/CLI for contradictions, estimate runtime, and exit without training.",
    )
    
    p.add_argument("--teacher-model", type=str, default=None,
                   help="Name of teacher model to distill from (e.g., haelt, ensemble)")
    p.add_argument("--teacher-ckpt", type=str, default=None,
                   help="Explicit teacher checkpoint path for distillation")
    p.add_argument("--distill-weight", type=float, default=0.5,
                   help="Weight of distillation loss relative to supervised loss")
    p.add_argument("--distill-temperature", type=float, default=2.0,
                   help="Temperature for distillation (if applicable)")

    # -- Pre-parse to find --config, then apply YAML defaults before full parse --
    pre, _ = p.parse_known_args()
    if pre.config:
        _apply_yaml_config(p, pre.config)
        from training.optuna_config import apply_optuna_overlay_if_needed
        apply_optuna_overlay_if_needed(
            p, pre.config, getattr(pre, "auto_optuna", None), _apply_yaml_config
        )

    args = p.parse_args()
    # --no-amp: force FP32 regardless of --dtype or hardware profile
    if getattr(args, "no_amp", False):
        args.dtype = "fp32"
        args.amp   = False
    if args.val_split is None:
        args.val_split = float(TRAINING["val_split"])
    if args.loss is None:
        args.loss = str(TRAINING.get("loss", "huber"))
    if args.walk_forward_folds is None:
        args.walk_forward_folds = int(TRAINING.get("walk_forward_folds", 6))
    if args.early_stop_metric is None:
        args.early_stop_metric = str(TRAINING.get("early_stop_metric", "sharpe"))
    if args.grad_accum_steps is None:
        args.grad_accum_steps = int(TRAINING.get("grad_accum_steps", 1))
    prof = strategy_profile(args.strategy_mode)
    scalp = strategy_profile("scalping")
    if args.bar_freq is None:
        args.bar_freq = str(prof["bar_freq"])
    if args.strategy_mode != "scalping" and int(args.seq_len) == int(scalp["seq_len"]):
        args.seq_len = int(prof["seq_len"])
    if args.lookahead_bars is None:
        args.lookahead_bars = int(prof["lookahead_bars"])
    if args.profit_target_atr is None:
        args.profit_target_atr = float(prof["profit_target_atr"])
    if args.stop_loss_atr is None:
        args.stop_loss_atr = float(prof["stop_loss_atr"])
    if args.strategy_mode != "scalping" and int(args.execution_delay_bars) == int(scalp["execution_delay_bars"]):
        args.execution_delay_bars = int(prof["execution_delay_bars"])
    print(f"[Strategy] {args.strategy_mode} | bars={args.bar_freq} | seq_len={args.seq_len} | "
          f"lookahead={args.lookahead_bars} | TP/SL={args.profit_target_atr}/{args.stop_loss_atr} ATR")
    if args.quick_mode:
        # Synthetic/quick smokes are too small for purged walk-forward
        # (embargo ≈ seq_len+lookahead often exceeds the sample count).
        if str(getattr(args, "data_source", "")).lower() == "synthetic":
            args.walk_forward_cv = False
            args.walk_forward_folds = 1
            # Tiny synthetic caches cannot satisfy direction-probe class floors.
            args.direction_probe = False
            args.ignore_preflight = True
        else:
            args.walk_forward_cv = True
            args.walk_forward_folds = min(max(int(args.walk_forward_folds), 1), 2)
        args.epochs = min(int(args.epochs), 8)
        args.pretrain_epochs = min(int(args.pretrain_epochs), 5)
        args.patience = min(int(args.patience), 4)
        args.train_ensemble = False
        args.rl_train = False
        # Keep curriculum within the quick-run seq_len so cache integrity
        # does not demand SETTINGS schedules up to 120 bars.
        cur = getattr(args, "curriculum", None)
        if isinstance(cur, dict):
            capped = []
            for entry in (cur.get("seq_schedule") or []):
                if not isinstance(entry, dict):
                    continue
                e = dict(entry)
                if e.get("seq_len") is not None:
                    e["seq_len"] = min(int(e["seq_len"]), int(args.seq_len))
                capped.append(e)
            args.curriculum = {**cur, "seq_schedule": capped or [
                {"epoch_start": 0, "seq_len": int(args.seq_len)}
            ]}
        print(f"[Quick] ON | folds={args.walk_forward_folds} | epochs={args.epochs} | "
              f"pretrain_epochs={args.pretrain_epochs} | ensemble=off | rl=off"
              f" | wf={'on' if args.walk_forward_cv else 'off'}")
    # B-M1: a warm-start fine-tune on a short rolling window must NOT run k-fold
    # walk-forward CV (a 1-epoch fine-tune would attempt 5 folds then hit the
    # small-data fallback). Force the single-split + embargo path explicitly.
    if getattr(args, "finetune_warm_start", False) and args.walk_forward_cv:
        args.walk_forward_cv = False
        print("[FineTune] Warm-start mode: walk-forward CV disabled "
              "(single embargoed split).")
    if getattr(args, "fair_sweep", False):
        args.model_profile = False
    args._cli_profile_overrides = _collect_cli_profile_overrides()
    _sync_runtime_config(args)
    return args


def apply_hardware_profile(args):
    """Override training paths and loader settings for a known local GPU/RAM combo."""
    name = getattr(args, "hardware_profile", None)
    if not name:
        return
    prof = HARDWARE_PROFILES.get(name)
    if not prof:
        return
    for k, v in prof.items():
        if k == "local_project_paths":
            continue
        setattr(args, k, v)
    if prof.get("local_project_paths"):
        args.checkpoint_dir = PATHS["checkpoints"]
        args.data_cache = PATHS["data_processed"]
    print(f"[Hardware] profile={name} | batch={args.batch_size} | workers={args.num_workers} | "
          f"chunk={args.chunk_size} | prefetch={args.prefetch_factor}")
    print(f"             checkpoint_dir={args.checkpoint_dir} | data_cache={args.data_cache}")


def _set_global_seed(seed: Optional[int]) -> None:
    """Set all relevant RNG seeds when a seed is provided."""
    if seed is None:
        return
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _slug_part(value: object, max_len: int = 80) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    text = re.sub(r"-{2,}", "-", text)
    return (text[:max_len].strip("-") or "run")


def _build_auto_run_name(args) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    pairs = _get_pairs(args)
    pair_label = f"{len(pairs)}pairs" if len(pairs) > 3 else "-".join(pairs)
    if getattr(args, "all_models", False):
        model_label = "all-models"
    elif getattr(args, "train_ensemble", False):
        model_label = f"ensemble-{getattr(args, 'model', 'base')}"
    elif getattr(args, "rl_train", False):
        model_label = f"rl-{getattr(args, 'rl_algo', 'dqn')}-{getattr(args, 'model', 'model')}"
    else:
        model_label = getattr(args, "model", "model")
    modes = []
    if getattr(args, "quick_mode", False):
        modes.append("quick")
    if getattr(args, "train_ensemble", False) and "ensemble" not in str(model_label):
        modes.append("ensemble")
    if getattr(args, "rl_train", False) and "rl" not in str(model_label):
        modes.append(f"rl-{getattr(args, 'rl_algo', 'dqn')}")
    if getattr(args, "walk_forward_cv", False):
        modes.append(f"wf{int(getattr(args, 'walk_forward_folds', 0) or 0)}")
    if getattr(args, "pretrain_ablation", None) not in (None, "false", False):
        modes.append(f"ablate-{getattr(args, 'pretrain_ablation')}")
    if getattr(args, "deploy_ensemble", False):
        modes.append("deploy-ensemble")
    if getattr(args, "deploy_rl", False):
        modes.append("deploy-rl")
    parts = [
        ts,
        model_label,
        getattr(args, "strategy_mode", "strategy"),
        pair_label,
        f"seq{int(getattr(args, 'seq_len', 0) or 0)}",
        *modes,
    ]
    return _slug_part("_".join(str(p) for p in parts), max_len=140)


def _apply_auto_run_dir(args) -> str:
    env_dir = os.getenv("CHECKPOINT_RUN_DIR", "").strip()
    if env_dir and not getattr(args, "auto_run_dir", False):
        args.checkpoint_dir = env_dir
        run_name = args.run_name or Path(env_dir).name
        args.run_name = run_name
        args.run_name_slug = _slug_part(run_name, max_len=140)

        return run_name

    run_name = args.run_name or _build_auto_run_name(args)
    args.run_name = run_name
    args.run_name_slug = _slug_part(run_name, max_len=140)

    if getattr(args, "auto_run_dir", False):
        root = Path(args.run_dir_root).expanduser() if getattr(args, "run_dir_root", None) else Path(args.checkpoint_dir).expanduser() / "runs"
        run_dir = root / args.run_name_slug

        args.checkpoint_dir = str(run_dir)
        os.environ["CHECKPOINT_RUN_DIR"] = str(run_dir)
        run_doc = {
            "run_name": run_name,
            "checkpoint_dir": str(run_dir),
            "run_dir_root": str(root),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": getattr(args, "model", None),
            "all_models": bool(getattr(args, "all_models", False)),
            "train_ensemble": bool(getattr(args, "train_ensemble", False)),
            "rl_train": bool(getattr(args, "rl_train", False)),
            "rl_algo": getattr(args, "rl_algo", None),
            "strategy_mode": getattr(args, "strategy_mode", None),
            "pairs": _get_pairs(args),
            "seq_len": int(getattr(args, "seq_len", 0) or 0),
            "walk_forward_folds": int(getattr(args, "walk_forward_folds", 0) or 0),
        }
        try:
            _safe_save_json(run_doc, root / "latest_run.json")
            _safe_save_json(run_doc, run_dir / "run_info.json")
        except Exception as exc:
            print(f"[RunDir] could not write run metadata: {exc}")
        print(f"[RunDir] auto-run-dir enabled -> {run_dir}")
    return run_name


_PROFILE_CLI_FLAGS = {
    "--lr": "lr",
    "--dropout": "dropout",
    "--num-layers": "num_layers",
    "--hidden-size": "hidden_size",
    "--d-model": "d_model",
    "--nhead": "nhead",
    "--seq-len": "seq_len",
    "--weight-decay": "weight_decay",
    "--batch-size": "batch_size",
    "--loss": "loss",

    "--early-stop-metric": "early_stop_metric",

    "--pretrain-method": "pretrain_method",

    "--pretrain-epochs": "pretrain_epochs",

    "--pretrain-lr": "pretrain_lr",

    "--pretrain-ablation": "pretrain_ablation",

}


def _collect_cli_profile_overrides() -> frozenset:
    """Dest names explicitly set on the CLI for profile-managed hyperparameters."""
    overrides: set[str] = set()
    argv = sys.argv[1:]
    idx = 0
    while idx < len(argv):
        tok = argv[idx]
        if tok in _PROFILE_CLI_FLAGS:
            overrides.add(_PROFILE_CLI_FLAGS[tok])
        elif tok.startswith("--") and "=" in tok:
            flag = tok.split("=", 1)[0]
            if flag in _PROFILE_CLI_FLAGS:
                overrides.add(_PROFILE_CLI_FLAGS[flag])
        idx += 1
    return frozenset(overrides)


def _normalize_architecture_profile(profile: dict, model_name: str) -> dict:
    """Map config/models.py keys to train_gpu argparse dest names."""
    key = model_name.lower().strip()
    out: dict = {}
    if "learning_rate" in profile:
        out["lr"] = float(profile["learning_rate"])
    if "dropout" in profile:
        out["dropout"] = float(profile["dropout"])
    if "seq_len" in profile:
        out["seq_len"] = int(profile["seq_len"])
    if "weight_decay" in profile:
        out["weight_decay"] = float(profile["weight_decay"])
    if "batch_size" in profile:
        out["batch_size"] = int(profile["batch_size"])
    if "loss" in profile:

        out["loss"] = str(profile["loss"]).lower()

    if "early_stop_metric" in profile:

        out["early_stop_metric"] = str(profile["early_stop_metric"]).lower()

    if "pretrain_epochs" in profile:
        out["pretrain_epochs"] = int(profile["pretrain_epochs"])
    if "pretrain_lr" in profile:
        out["pretrain_lr"] = float(profile["pretrain_lr"])
    if "pretrain_method" in profile:
        out["pretrain_method"] = str(profile["pretrain_method"]).lower()
    if "pretrain_ablation" in profile:

        out["pretrain_ablation"] = str(profile["pretrain_ablation"]).lower()


    # WIRE-002: Map dim_feedforward to dim_ff for all architectures
    if "dim_feedforward" in profile:
        out["dim_ff"] = int(profile["dim_feedforward"])

    if key == "haelt":
        if "lstm_hidden" in profile:
            out["hidden_size"] = int(profile["lstm_hidden"]) * 2
        if "d_model" in profile:
            out["d_model"] = int(profile["d_model"])
        if "nhead" in profile:
            out["nhead"] = int(profile["nhead"])
        if "num_layers" in profile:
            out["num_layers"] = int(profile["num_layers"])
        elif "n_transformer_layers" in profile:
            out["num_layers"] = int(profile["n_transformer_layers"])
    elif key == "tft":
        if "hidden_size" in profile:
            out["hidden_size"] = int(profile["hidden_size"])
        if "nhead" in profile:
            out["nhead"] = int(profile["nhead"])
        elif "attention_head_size" in profile:
            out["nhead"] = int(profile["attention_head_size"])
        if "lstm_layers" in profile:
            out["num_layers"] = int(profile["lstm_layers"])
    elif key == "gnn":
        if "hidden_channels" in profile:
            out["hidden_size"] = int(profile["hidden_channels"])
        if "num_layers" in profile:
            out["num_layers"] = int(profile["num_layers"])
        if "heads" in profile:
            out["nhead"] = int(profile["heads"])
        if "node_features" in profile:
            out["node_features"] = int(profile["node_features"])
    else:
        for field in ("d_model", "nhead", "num_layers", "hidden_size"):
            if field in profile:
                out[field] = int(profile[field])
    return out


def _apply_model_profile(args, model_name: str, *, enabled: bool = True):
    """Merge architecture_config(name) onto args; explicit CLI overrides win."""
    if not enabled:
        return args
    try:
        from config.models import architecture_config
        profile = architecture_config(model_name)
    except Exception as exc:
        print(f"[Profile] Skipped for {model_name}: {exc}")
        return args

    normalized = _normalize_architecture_profile(profile, model_name)
    if not normalized:
        return args

    cli_overrides = getattr(args, "_cli_profile_overrides", None) or _collect_cli_profile_overrides()
    log_parts: list[str] = []
    recipe_name = str(profile.get("recipe_name") or profile.get("decision_role") or "").strip()

    if recipe_name:

        setattr(args, "recipe_name", recipe_name)

        log_parts.append(f"recipe={recipe_name}")

    for dest, value in normalized.items():
        if dest in cli_overrides or not hasattr(args, dest):
            continue
        setattr(args, dest, value)
        if dest == "lr":
            log_parts.append(f"lr={float(value):.3e}")
        elif dest == "dropout":
            log_parts.append(f"dropout={float(value):.3f}")
        elif dest == "weight_decay":
            log_parts.append(f"weight_decay={float(value):.3e}")
        else:
            log_parts.append(f"{dest}={value}")

    args.model = model_name
    args._profile_applied = True
    if log_parts:
        print(f"[Profile] {model_name}: " + " ".join(log_parts))
    return args


def _member_training_args(base_args, model_name: str, member_idx: int, total_members: int):
    """
    Clone args, apply per-architecture profile, and optional ensemble diversity controls.
    """
    out = argparse.Namespace(**vars(base_args))
    out.model = model_name

    # Place each model's checkpoints in its own subfolder: checkpoints/<model_name>/
    base_ckpt = Path(base_args.checkpoint_dir)
    out.checkpoint_dir = str(base_ckpt / model_name)
    Path(out.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if getattr(base_args, "model_profile", True):
        out = _apply_model_profile(out, model_name, enabled=True)

    explicit = bool(getattr(base_args, "ensemble_explicit_diversity", False))
    if not (explicit and total_members > 1):
        return out

    # Spread members across a deterministic [-0.5, +0.5] range.
    center = (total_members - 1) / 2.0
    rel = (member_idx - center) / max(1.0, float(total_members - 1))

    base_seed = getattr(base_args, "seed", None)
    if base_seed is not None:
        out.seed = int(base_seed) + int(getattr(base_args, "ensemble_member_seed_offset", 997)) * member_idx

    lr_jitter = float(getattr(base_args, "ensemble_member_lr_jitter", 0.0))
    if lr_jitter > 0:
        out.lr = float(out.lr) * max(0.25, 1.0 + rel * lr_jitter)

    drop_jitter = float(getattr(base_args, "ensemble_member_dropout_jitter", 0.0))
    if drop_jitter > 0:
        out.dropout = float(np.clip(float(out.dropout) + rel * drop_jitter, 0.0, 0.8))

    print(f"[EnsembleDiversity] {model_name}: seed={getattr(out, 'seed', None)} "
          f"lr={out.lr:.3e} dropout={out.dropout:.3f} (member {member_idx+1}/{total_members})")
    if getattr(out, "pretrain", False) or getattr(out, "ablate_pretrain", False):
        if getattr(out, "pretrain_method", "") == PRETRAIN.get("method", "byol").lower():
            if "haelt" in model_name.lower():
                out.pretrain_method = "masked"
            elif "mamba" in model_name.lower():
                out.pretrain_method = "forecast"
            elif "tft" in model_name.lower():
                out.pretrain_method = "masked"
            elif "gnn" in model_name.lower():
                out.pretrain_method = "cluster"
            elif "expert" in model_name.lower():
                out.pretrain_method = "tscl"

    return out


def _model_build_args(base_args, model_name: str) -> argparse.Namespace:
    """Per-architecture args for build_model / checkpoint load (no ensemble jitter)."""
    out = argparse.Namespace(**vars(base_args))
    out.model = str(model_name).lower().strip()
    if getattr(base_args, "model_profile", True):
        out = _apply_model_profile(out, out.model, enabled=True)
    return out


def _model_completion_status(model_name: str, checkpoint_dir: str | Path) -> tuple[bool, str]:
    """Return whether an all-model member appears fully trained.

    Crash checkpoints are deliberately ignored: they prove the model started,
    not that it produced a clean resume/best artifact.
    """
    model = str(model_name).lower().strip()
    ckpt_dir = Path(checkpoint_dir)
    best_paths = [
        ckpt_dir / f"{model}_best.pt",
        ckpt_dir / model / f"{model}_best.pt",
    ]
    has_best = any(p.exists() for p in best_paths)
    has_manifest = (ckpt_dir / "manifest.json").exists()
    has_train_summary = (ckpt_dir / "train_summary.json").exists()
    has_fold_selection = (ckpt_dir / "fold_selection.json").exists()
    has_deployment = (ckpt_dir / "deployment.json").exists()
    crash_files = sorted(ckpt_dir.glob(f"*{model}*_crash.pt"))

    if has_deployment and (has_best or has_manifest):
        return True, "deployment.json + completed checkpoint metadata"
    if has_fold_selection and has_best:
        return True, "fold_selection.json + best checkpoint"
    if has_manifest and has_best:
        return True, "manifest.json + best checkpoint"
    if has_train_summary and has_best and not has_fold_selection:
        return True, "train_summary.json + best checkpoint"
    if crash_files and not has_best:
        return False, f"crash checkpoint only ({crash_files[-1].name})"
    if has_best:
        return False, "best checkpoint exists but completion metadata is missing"
    return False, "no completed artifacts"


def _baseline_ablation_completion_status(model_name: str, checkpoint_dir: str | Path, args) -> tuple[bool, str]:
    """Return whether the no-pretrain baseline proof for a model is already complete.

    Baseline ablation artifacts live under <checkpoint_dir>/baseline and do not go
    through the full model promotion/deployment path, so the generic completion
    helper is too weak here. For walk-forward runs we require every expected fold
    best checkpoint before skipping baseline on resume.
    """
    model = str(model_name).lower().strip()
    baseline_dir = Path(checkpoint_dir) / "baseline"
    if not baseline_dir.exists():
        return False, "baseline artifact directory missing"

    walk_forward = bool(getattr(args, "walk_forward_cv", False))
    if walk_forward:
        n_folds = max(1, int(getattr(args, "walk_forward_folds", 1)))
        missing = []
        for fi in range(n_folds):
            fold_best = baseline_dir / f"baseline_{model}_fold{fi}_best.pt"
            if not fold_best.exists():
                missing.append(fold_best.name)
        if not missing:
            return True, f"all {n_folds} baseline fold checkpoints present"
        return False, f"missing baseline fold checkpoints: {', '.join(missing[:3])}" + (
            " ..." if len(missing) > 3 else ""
        )

    single_best = baseline_dir / f"baseline_{model}_best.pt"
    if single_best.exists():
        return True, "single-split baseline checkpoint present"
    return False, f"missing {single_best.name}"


def _supervised_resume_status(model_name: str, checkpoint_dir: str | Path, args) -> tuple[bool, str]:
    """Return whether supervised training already started for this model."""
    model = str(model_name).lower().strip()
    ckpt_dir = Path(checkpoint_dir)
    walk_forward = bool(getattr(args, "walk_forward_cv", False))

    if walk_forward:
        n_folds = max(1, int(getattr(args, "walk_forward_folds", 1)))
        last_paths = [ckpt_dir / f"{model}_fold{fi}_last.pt" for fi in range(n_folds)]
        best_paths = [ckpt_dir / f"{model}_fold{fi}_best.pt" for fi in range(n_folds)]
        existing_last = [p for p in last_paths if p.exists()]
        if existing_last:
            latest = max(existing_last, key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
            return True, f"supervised resume checkpoint present ({latest.name})"
        existing_best = [p for p in best_paths if p.exists()]
        if existing_best:
            latest = max(existing_best, key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
            return True, f"supervised fold checkpoint present ({latest.name})"
        return False, "no supervised fold checkpoints found"

    last_path = ckpt_dir / f"{model}_last.pt"
    best_path = ckpt_dir / f"{model}_best.pt"
    if last_path.exists():
        return True, f"supervised resume checkpoint present ({last_path.name})"
    if best_path.exists():
        return True, f"supervised best checkpoint present ({best_path.name})"
    return False, "no supervised checkpoints found"


def _latest_resumable_fold(model_name: str, checkpoint_dir: str | Path, n_folds: int) -> int | None:
    """Return the latest fold index with a resumable supervised checkpoint."""
    model = str(model_name).lower().strip()
    ckpt_dir = Path(checkpoint_dir)
    candidates: list[tuple[float, int]] = []
    for fi in range(max(1, int(n_folds))):
        for path in (
            ckpt_dir / f"{model}_fold{fi}_last.pt",
            ckpt_dir / f"{model}_fold{fi}_best.pt",
        ):
            if path.exists():
                try:
                    candidates.append((path.stat().st_mtime, fi))
                except Exception:
                    candidates.append((0.0, fi))
                break
    if not candidates:
        return None
    candidates.sort()
    return int(candidates[-1][1])


def _effective_max_seq_len(args) -> int:
    """Max sequence length required by training config + curriculum schedule."""
    seqs = [int(getattr(args, "seq_len", 60) or 60)]
    cur = getattr(args, "curriculum", None)
    if cur is None or cur is False or cur == "none" or cur == "":
        return max(seqs)
    if not isinstance(cur, dict):
        cur = SETTINGS_CURRICULUM
    for entry in (cur.get("seq_schedule") or []):
        if isinstance(entry, dict) and entry.get("seq_len") is not None:
            seqs.append(int(entry["seq_len"]))
    return max(seqs)


def _load_cv_fold_entry(
    model_name: str,
    checkpoint_dir: str | Path,
    fold_idx: int,
    early_stop_metric: str = "sharpe",
) -> dict | None:
    """Rebuild one walk-forward fold summary from its checkpoint artifacts."""
    model = str(model_name).lower().strip()
    ckpt_dir = Path(checkpoint_dir)
    fold_suffix = f"_fold{int(fold_idx)}"
    last_path = ckpt_dir / f"{model}{fold_suffix}_last.pt"
    best_path = ckpt_dir / f"{model}{fold_suffix}_best.pt"
    ckpt_path = last_path if last_path.exists() else (best_path if best_path.exists() else None)
    if ckpt_path is None:
        return None
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    history = ck.get("history")
    if not isinstance(history, dict) or not history:
        return None
    best_metric = None
    if early_stop_metric == "sharpe" and history.get("val_sharpe"):
        best_metric = float(max(history["val_sharpe"]))
    elif history.get("val_loss"):
        best_metric = float(min(history["val_loss"]))
    elif ck.get("best_sharpe") is not None:
        best_metric = float(ck["best_sharpe"])
    elif ck.get("best_val_loss") is not None:
        best_metric = float(ck["best_val_loss"])
    return {"fold": int(fold_idx), "best_metric": best_metric, "history": history}


def _load_walk_forward_resume_history(
    model_name: str,
    checkpoint_dir: str | Path,
    log_dir: Path,
    run_name_slug: str,
    model_slug: str,
    start_fold: int,
    early_stop_metric: str = "sharpe",
) -> list[dict]:
    """Load completed fold metrics for folds [0, start_fold) when resuming walk-forward CV."""
    if start_fold <= 0:
        return []
    entries: list[dict] = []
    cv_path = Path(log_dir) / f"{run_name_slug}_{model_slug}_cv.json"
    if cv_path.exists():
        try:
            data = json.loads(cv_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                entries = [
                    e for e in data
                    if isinstance(e, dict) and int(e.get("fold", -1)) < int(start_fold)
                ]
        except Exception as exc:
            _log_warn(f"[WalkForward] Could not read prior cv.json ({exc}); rebuilding from checkpoints.")
    loaded_folds = {int(e.get("fold", -1)) for e in entries}
    for fi in range(int(start_fold)):
        if fi in loaded_folds:
            continue
        entry = _load_cv_fold_entry(model_name, checkpoint_dir, fi, early_stop_metric)
        if entry is not None:
            entries.append(entry)
    entries.sort(key=lambda e: int(e.get("fold", 0)))
    return entries


# -----------------------------------------------------------------------------
# TRAINING LOGGER  (delegates to monitoring/train_logger.py)
# -----------------------------------------------------------------------------

try:
    from monitoring.train_logger import TrainingLogger as _TrainingLogger
    _TRAIN_LOGGER_AVAILABLE = True
except Exception:
    _TRAIN_LOGGER_AVAILABLE = False

_TRAIN_LOGGER: Optional[Any] = None   # TrainingLogger instance, set in supervised_train

# Convenience shims ΓÇö used throughout the file so call-sites need no changes
def _log_error(msg: str, exc: Optional[Exception] = None) -> None:
    if _TRAIN_LOGGER is not None:
        _TRAIN_LOGGER.error(msg, exc)

def _log_warn(msg: str) -> None:
    if _TRAIN_LOGGER is not None:
        _TRAIN_LOGGER.warning(msg)

def _log_info(msg: str) -> None:
    if _TRAIN_LOGGER is not None:
        _TRAIN_LOGGER.info(msg)

def _log_oom(batch_idx: int, epoch: int, oom_count: int) -> None:
    if _TRAIN_LOGGER is not None:
        _TRAIN_LOGGER.on_batch_oom(batch_idx, epoch, oom_count)

def _log_nan(batch_idx: int, epoch: int, nan_count: int) -> None:
    if _TRAIN_LOGGER is not None:
        _TRAIN_LOGGER.on_batch_nan(batch_idx, epoch, nan_count)

class _DummyCtx:
    """No-op context manager ΓÇö used when rich display is unavailable."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


# -----------------------------------------------------------------------------
# THERMAL THROTTLE  (laptop-safe GPU temperature guard)
# -----------------------------------------------------------------------------

def _gpu_temp_celsius() -> int:
    """Return current GPU 0 temperature in ┬░C, or -1 if pynvml unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    except Exception:
        return -1

def _thermal_check(limit: int = 83, pause_secs: float = 2.0) -> None:
    """
    Pause training if GPU temperature exceeds limit.
    Called every N batches in the training loop.
    limit=0 disables the check (desktop with active cooling).
    """
    if limit <= 0:
        return
    temp = _gpu_temp_celsius()
    if temp < 0:
        return  # pynvml unavailable ΓÇö skip silently
    if temp >= limit:
        msg = f"[Thermal] GPU {temp}┬░C >= limit {limit}┬░C ΓÇö pausing {pause_secs}s"
        print(msg)
        _log_warn(msg)
        import time as _t; _t.sleep(pause_secs)


# -----------------------------------------------------------------------------
# GPU SETUP
# -----------------------------------------------------------------------------

def setup_device(dtype_override: str = "auto", deterministic: bool = False) -> "tuple[torch.device, int, torch.dtype]":
    """
    Detect GPU, configure cuDNN/TF32 flags, and select the optimal AMP dtype.

    Returns
    -------
    (device, n_gpus, amp_dtype)

    AMP dtype selection (override with --dtype or GPU["amp_dtype"] in settings):
    ΓÇó BF16  ΓÇö Ada Lovelace / Ampere (CC >= 8.0): no GradScaler, no overflow,
               same Tensor Core throughput as FP16, preferred for RTX 40-series.
    ΓÇó FP16  ΓÇö Turing / Volta (CC < 8.0): needs GradScaler to prevent underflow.
    ΓÇó FP32  ΓÇö CPU fallback or --dtype fp32 for debugging.

    Tensor Cores activate automatically when:
      1. Mixed precision is enabled (FP16 or BF16 via autocast).
      2. Standard layers are used (nn.Linear, nn.Conv*).
      3. Feature/hidden dims are multiples of 8 (all arch defaults satisfy this).
    """
    if not torch.cuda.is_available():
        print("[GPU] No CUDA detected ΓÇö running on CPU (very slow for 20M ticks)")
        _log_warn("[GPU] CUDA unavailable ΓÇö CPU mode")
        return torch.device("cpu"), 1, torch.float32

    n   = torch.cuda.device_count()
    dev = torch.device("cuda:0")

    # -- Linux-specific: set multiprocessing start method to fork ---------------
    # fork() is the Linux default but setting it explicitly prevents edge cases
    # where PyTorch internally triggers spawn (e.g. nested multiprocessing calls).
    # Must be set before any DataLoader workers are created.
    if os.name != "nt":
        try:
            import torch.multiprocessing as _tmp
            _tmp.set_start_method("fork", force=True)
        except RuntimeError:
            pass   # already set elsewhere ΓÇö fine

    # -- Linux-specific: pin CPU threads so DataLoader workers don't fight ------
    # Without this, 4 workers each try to use all CPU cores via OpenMP/MKL,
    # causing cache thrashing. One thread per worker is optimal for I/O-bound
    # zarr decompression (Blosc already uses internal multi-threading per chunk).
    if os.name != "nt":
        _n_cpu = os.cpu_count() or 4
        os.environ.setdefault("OMP_NUM_THREADS",  "1")
        os.environ.setdefault("MKL_NUM_THREADS",  "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS",  str(min(4, _n_cpu)))

    # -- cuDNN / TF32 flags ----------------------------------------------------
    allow_tf32 = bool(_GPU_CFG.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32       = allow_tf32
    if deterministic:
        torch.backends.cudnn.benchmark     = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark        = bool(_GPU_CFG.get("cudnn_benchmark", True))
        torch.backends.cudnn.deterministic    = False   # max speed
    # Unlock TF32 Tensor Cores for all FP32 matmuls (~3x faster on Ada/Ampere).
    # 'high' = TF32 matmul + BF16 AMP ΓÇö recommended for RTX 40-series.
    torch.set_float32_matmul_precision("high")

    # -- CUDA memory: use 95% of VRAM, leave 5% for driver/display overhead ----
    # PyTorch default reserves ~20% for its caching allocator which causes
    # premature OOM on 8 GB cards. 0.95 gives ~8.5 GB usable on RTX 4060.
    try:
        torch.cuda.set_per_process_memory_fraction(0.95, dev)
    except Exception:
        pass   # older PyTorch versions don't support this

    # -- AMP dtype selection ---------------------------------------------------
    cfg_dtype = _GPU_CFG.get("amp_dtype", "auto")
    effective_override = dtype_override if dtype_override != "auto" else cfg_dtype

    if effective_override == "fp32":
        amp_dtype = torch.float32
    elif effective_override == "fp16":
        amp_dtype = torch.float16
    elif effective_override == "bf16":
        amp_dtype = torch.bfloat16
    else:
        # Auto: prefer BF16 on Ada/Ampere (compute capability >= 8.0)
        cc_major = torch.cuda.get_device_capability(0)[0]
        if cc_major >= 8 and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        elif torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16

    dtype_name = {torch.bfloat16: "BF16", torch.float16: "FP16", torch.float32: "FP32"}.get(amp_dtype, "?")

    for i in range(n):
        g = torch.cuda.get_device_properties(i)
        vram = g.total_memory / 1e9
        cc   = f"{g.major}.{g.minor}"
        print(f"[GPU {i}] {g.name} | {vram:.0f} GB VRAM | CC {cc} | CUDA {torch.version.cuda}")
        print(f"         AMP dtype: {dtype_name} | TF32: {allow_tf32} | "
              f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        if vram < 12:
            print(f"         NOTE: low VRAM ({vram:.0f} GB). Use --hardware-profile "
                  "rtx_4060_16gb_ram or ubuntu_rtx_laptop and --batch-size 384ΓÇô512.")
    _log_info(f"[GPU] device={dev} n_gpus={n} amp_dtype={dtype_name} CC={cc}")
    return dev, n, amp_dtype


# -----------------------------------------------------------------------------
# PHASE 1 ΓÇö CHUNKED DATA PIPELINE
# -----------------------------------------------------------------------------

def _get_pairs(args) -> List[str]:
    """Return the list of pairs to train on, from --pairs or --pair."""
    raw = getattr(args, "pairs", None)
    if not raw:
        return [args.pair.upper()]
    if isinstance(raw, list):
        return [p.strip().upper() for p in raw if p and p.strip()]
    return [p.strip().upper() for p in str(raw).split(",") if p.strip()]


def _real_data_window_days(args) -> int:
    """
    Choose a conservative date-window size for real-data ingestion.

    We derive this from ``chunk_size`` so the real-data path respects the same
    RAM safety valve as the synthetic chunked builder.
    """
    explicit = int(getattr(args, "real_data_window_days", 0) or 0)
    if explicit > 0:
        return explicit

    session_hours = 24 if getattr(args, "full_day_data", False) else 11
    # Conservative FX tick density estimate to keep per-window RAM bounded.
    est_ticks_per_hour = 10_000
    est_ticks_per_day = max(session_hours * est_ticks_per_hour, 1)
    chunk_size = max(int(getattr(args, "chunk_size", 500_000) or 500_000), 1)
    days = max(1, chunk_size // est_ticks_per_day)
    return min(max(int(days), 1), 31)


def _effective_window_days(args) -> int:
    """Return per-window day count after applying the batch multiplier.

    effective = real_data_window_days * window_batch_days

    ``window_batch_days`` groups consecutive base windows into a single
    processing batch so feature engineering sees more context at window
    boundaries.  Default is 1 (no batching, backward compatible).
    """
    base = _real_data_window_days(args)
    batch = max(1, int(getattr(args, "window_batch_days", 1) or 1))
    return base * batch


def _iter_date_windows(start: str, end: str, window_days: int) -> List[Tuple[str, str]]:
    """Split an inclusive YYYY-MM-DD range into inclusive date windows."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end, "%Y-%m-%d")
    if end_dt < start_dt:
        raise ValueError(f"data_end ({end}) is earlier than data_start ({start})")

    windows: List[Tuple[str, str]] = []
    current = start_dt
    step = max(int(window_days), 1)
    while current <= end_dt:
        win_end = min(current + timedelta(days=step - 1), end_dt)
        windows.append((current.strftime("%Y-%m-%d"), win_end.strftime("%Y-%m-%d")))
        current = win_end + timedelta(days=1)
    return windows


def _resolve_cross_asset_source(args) -> str:
    """Resolve cross-asset provider with env override matching downloader behavior."""
    return str(
        os.getenv("CROSS_ASSET_SOURCE", "").strip()
        or getattr(args, "cross_asset_provider", "auto")
        or "auto"
    ).strip().lower()


def _cache_target_col(args) -> str:

    """Keep cache y as reward/PnL; direction labels live in y_cls sidecar.



    Classification models train from y_cls, but validation Sharpe and promotion

    gates need continuous reward/PnL in y. Do not switch cache y to the class

    label just because the supervised loss is cross_entropy.

    """

    return "reward"





def _get_cache_path(args) -> Path:
    pairs    = _get_pairs(args)
    pair_tag = "-".join(sorted(pairs))
    target_col = _cache_target_col(args)

    exec_delay = int(getattr(args, "execution_delay_bars", 1))
    strategy = str(getattr(args, "strategy_mode", "scalping") or "scalping").lower()
    bar_freq = str(getattr(args, "bar_freq", "1min") or "1min").lower()
    lookahead = int(getattr(args, "lookahead_bars", LABELING.get("lookahead_bars", 15)))
    tp_atr = float(getattr(args, "profit_target_atr", LABELING.get("profit_target_atr", 1.5)))
    sl_atr = float(getattr(args, "stop_loss_atr", LABELING.get("stop_loss_atr", 0.8)))
    news_mode = str(getattr(args, "historical_news_mode", "calendar") or "calendar").lower()
    news_tag = f"news-{news_mode}"
    ca_mode = str(getattr(args, "cross_asset_mode", "auto") or "auto").lower()
    ca_source = _resolve_cross_asset_source(args)
    ca_tag = f"ca-{ca_mode}-{ca_source}"
    tag      = (
        f"{strategy}_{bar_freq}_{pair_tag}_{args.n_ticks}_{args.data_source}_{args.seq_len}_"
        f"{args.label_method}_{target_col}_lh{lookahead}_tp{tp_atr:g}_sl{sl_atr:g}_"
        f"exec{exec_delay}_{news_tag}_{ca_tag}"
    )
    if getattr(args, "data_start", None) and getattr(args, "data_end", None):
        tag += f"_{args.data_start}_{args.data_end}"
    use_zarr_cache = bool(ZARR)
    ext = ".zarr" if use_zarr_cache else ""
    return Path(args.data_cache) / f"dataset_{tag}{ext}"


def _base_path(cache_path) -> str:
    """Strip .zarr extension to get base name used for NPY/NPZ sidecars."""
    p = str(cache_path)
    if p.endswith(".zarr"):
        return p[:-5]
    return p


def _scaler_npz_path(cache_path: Path) -> Path:
    return Path(_base_path(str(cache_path)) + "_scaler.npz")


def _x_path(cache_path) -> str:
    return _base_path(str(cache_path)) + "_X.npy"


def _y_path(cache_path) -> str:
    return _base_path(str(cache_path)) + "_y.npy"


def _diff_path(cache_path) -> str:
    """NPY sidecar for per-sample difficulty scores (uint8: 0=easy,1=medium,2=hard)."""
    return _base_path(str(cache_path)) + "_diff.npy"


def _pq_path(cache_path) -> str:
    """NPY sidecar for per-sample path-quality scores (float32, range 0ΓÇô1).
    Produced only when regime labeling is used; absent when base labeling is used.
    """
    return _base_path(str(cache_path)) + "_pq.npy"


def _y_cls_path(cache_path) -> str:
    """NPY sidecar / zarr array for direction labels {-1,0,+1} when y stores reward."""
    return _base_path(str(cache_path)) + "_y_cls.npy"


def _close_path(cache_path) -> str:
    """Per-sequence mid/close price at the label bar (float32, absolute FX quote)."""
    return _base_path(str(cache_path)) + "_close.npy"


def _atr_path(cache_path) -> str:
    """Per-sequence ATR at the label bar (float32, price units ΓÇö matches env SL/TP)."""
    return _base_path(str(cache_path)) + "_atr.npy"


def _spread_path(cache_path) -> str:
    """Per-sequence bid-ask spread at the label bar (float32, price units)."""
    return _base_path(str(cache_path)) + "_spread.npy"


_RL_MARKET_ZARR_KEYS = ("close", "atr", "spread")


def _cache_has_rl_market_arrays(cache_path: str) -> bool:
    """True when close/atr/spread exist with the same row count as X."""
    p = Path(cache_path)
    n_x = _on_disk_sequence_count(cache_path)
    if n_x is None:
        return False
    if ZARR and p.is_dir() and (p / ".zgroup").exists():
        try:
            z = _zarr_open_group(cache_path, mode="r")
            if not all(k in z for k in _RL_MARKET_ZARR_KEYS):
                return False
            return all(int(z[k].shape[0]) == int(n_x) for k in _RL_MARKET_ZARR_KEYS)
        except Exception:
            return False
    for fn in (_close_path, _atr_path, _spread_path):
        fp = Path(fn(cache_path))
        if not fp.exists():
            return False
        try:
            if int(np.load(str(fp), mmap_mode="r").shape[0]) != int(n_x):
                return False
        except Exception:
            return False
    return True


def _require_rl_market_cache(cache_path: str) -> None:
    if _cache_has_rl_market_arrays(cache_path):
        return
    raise RuntimeError(
        "[RL] Cache missing real market arrays (close, atr, spread). "
        "Rebuild with: .\\.venv-gpu\\Scripts\\python.exe scripts\\train.py --rebuild-cache "
        "or training\\train_gpu.py --force-rebuild"
    )


def _load_rl_market_from_cache(cache_path: str, start: int, n_env: int
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load per-sequence close/ATR/spread aligned with X/y indices."""
    end = start + n_env
    if ZARR and str(cache_path).endswith(".zarr") and Path(cache_path).is_dir():
        z = _zarr_open_group(cache_path, mode="r")
        prices = np.asarray(z["close"][start:end], dtype=np.float32)
        atr = np.asarray(z["atr"][start:end], dtype=np.float32)
        spreads = np.asarray(z["spread"][start:end], dtype=np.float32)
    else:
        prices = np.asarray(np.load(_close_path(cache_path), mmap_mode="r")[start:end],
                            dtype=np.float32)
        atr = np.asarray(np.load(_atr_path(cache_path), mmap_mode="r")[start:end],
                         dtype=np.float32)
        spreads = np.asarray(np.load(_spread_path(cache_path), mmap_mode="r")[start:end],
                             dtype=np.float32)
    return prices, atr, spreads


def _market_bar_arrays_from_feats(
    feats,
    x_index,
    fe: "FeatureEngineer",
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bar-level close/ATR/spread aligned with feature rows (before sequence filter)."""
    close_col = "mid_close" if "mid_close" in feats.columns else "close"
    if close_col not in feats.columns:
        raise ValueError(
            f"[Data] RL market cache requires '{close_col}' or 'close' in features"
        )
    atr_w = int(getattr(fe, "atr_w", FEATURES.get("atr_window", 6)))
    atr_col = f"atr_{atr_w}"
    if atr_col not in feats.columns:
        atr_col = next(
            (c for c in (f"atr_{atr_w}", "atr_6", "atr_20", "atr") if c in feats.columns),
            None,
        )
    if atr_col is None:
        raise ValueError("[Data] RL market cache requires an ATR column (e.g. atr_6)")
    pip = float(LABELING.get("pip_size", 0.0001))
    close_bars = feats[close_col].reindex(x_index).astype(np.float64).values
    atr_bars = feats[atr_col].reindex(x_index).astype(np.float64).values
    if "spread_pips" in feats.columns:
        spread_bars = feats["spread_pips"].reindex(x_index).astype(np.float64).values * pip
    elif "spread_avg" in feats.columns:
        spread_bars = feats["spread_avg"].reindex(x_index).astype(np.float64).values
    else:
        spread_bars = np.full(len(x_index), 0.5 * pip, dtype=np.float64)
    close_seq = np.asarray(close_bars[seq_len - 1:], dtype=np.float32)
    atr_seq = np.asarray(np.maximum(atr_bars[seq_len - 1:], pip), dtype=np.float32)
    spread_seq = np.asarray(np.maximum(spread_bars[seq_len - 1:], pip * 0.1), dtype=np.float32)
    return close_seq, atr_seq, spread_seq


def _resolve_pair_feat_indices(feat_names: Optional[list], f_per_pair: int) -> tuple[int, int]:
    """Return (return_idx, atr_idx) within each pair's feature slice."""
    names = [str(c).split("::")[-1] for c in list(feat_names or [])[:f_per_pair]]

    if not names:
        return 0, min(1, f_per_pair - 1)
    ret_candidates = [f"ret_{w}" for w in (5, 20, 60, 30, 120)] + ["return", "ret"]

    atr_w = int(FEATURES.get("atr_window", 6))
    atr_candidates = [f"atr_{atr_w}", "atr_6", "atr_20", "atr"]
    ri = next((names.index(c) for c in ret_candidates if c in names), 0)
    ai = next((names.index(c) for c in atr_candidates if c in names), min(1, f_per_pair - 1))
    return ri, ai


def _promotion_holdout_n(n_samples: int, args) -> int:
    """Bars reserved for promotion gate ΓÇö never used in walk-forward CV."""
    frac = min(max(float(getattr(args, "promote_forward_frac", 0.1)), 0.01), 0.5)
    return min(200_000, max(50, int(n_samples * frac)))


def _trainable_max_index(n_total: int, args) -> int:
    """Last exclusive index usable for pretrain/RL (excludes holdout + embargo)."""
    n_total = max(0, int(n_total))
    return max(0, n_total - _promotion_holdout_n(n_total, args) - _embargo_bars(args))


def _pretrain_channel_chunk(args, n_features: int) -> Optional[int]:
    """Per-pair feature block size for channel-shuffle augmentation."""
    fpp = getattr(args, "_f_per_pair", None)
    if fpp is not None and int(fpp) > 0:
        embed = int(getattr(args, "pair_embed_dim", 0) or 0)
        n_pairs = int(getattr(args, "_n_pairs", 1) or 1)
        if n_pairs > 1 and embed > 0:
            return int(fpp) + embed
        return int(fpp)
    return None


def _make_pretrain_augmenter(args, n_features: int) -> "TimeSeriesAugmenter":
    """Build augmenter from PRETRAIN defaults + optional YAML overrides."""
    aug_cfg = getattr(args, "pretrain_augmentations", None) or PRETRAIN.get("augmentations") or {}
    scale_rng = aug_cfg.get("scaling_range", (0.8, 1.2))
    if isinstance(scale_rng, list):
        scale_rng = tuple(scale_rng)
    crop_rng = aug_cfg.get("crop_ratio", (0.7, 1.0))
    if isinstance(crop_rng, list):
        crop_rng = tuple(crop_rng)
    return TimeSeriesAugmenter(
        jitter_std=float(aug_cfg.get("jitter_std", 0.02)),
        scale_range=scale_rng,
        feature_drop_p=float(aug_cfg.get("feature_drop_p", 0.3)),
        crop_ratio=crop_rng,
        seed=getattr(args, "seed", None),
        channel_chunk=_pretrain_channel_chunk(args, n_features),
    )


def _rl_reward_weights(args) -> dict:
    """Map YAML/CLI reward weights to ForexTradingEnv keys."""
    raw = getattr(args, "rl_reward_weights", None) or RL.get("reward") or {}
    return {
        "pnl":       float(raw.get("pnl", raw.get("pnl_weight", 1.0))),
        "drawdown":  float(raw.get("drawdown", raw.get("drawdown_penalty", 0.5))),
        "tx_cost":   float(raw.get("tx_cost", raw.get("transaction_cost_penalty", 0.3))),
        "overtrade": float(raw.get("overtrade", raw.get("overtrading_penalty", 0.2))),
    }


def _rl_algo_kwargs(args, algo: str) -> dict:
    """Merge settings.RL hyperparams with optional YAML overrides."""
    algo = str(algo).lower()
    base = dict(RL.get(algo, {}))
    override = getattr(args, "rl_algo_overrides", None) or {}
    if isinstance(override, dict) and algo in override and isinstance(override[algo], dict):
        base.update(override[algo])
    return base


def _rl_train_val_slices(n_total: int, args) -> tuple[int, int, int, int]:
    """
    Return (train_start, train_n, val_start, val_n) within the trainable index range.
    Uses the earliest contiguous pool (not the promotion holdout tail).
    """
    max_end = _trainable_max_index(n_total, args)
    pool = min(100_000, max_end) if max_end > 0 else 0
    if pool < 256:
        return 0, max(0, pool), max(0, pool), 0
    val_frac = float(getattr(args, "rl_val_frac", 0.15))
    val_n = max(256, int(pool * val_frac))
    val_n = min(val_n, pool // 2)
    train_n = pool - val_n
    train_start = 0
    val_start = train_n
    return train_start, train_n, val_start, val_n


def _on_disk_sequence_count(cache_path: str) -> Optional[int]:
    """
    Rows actually readable by MemmapSequenceDataset.
    Priority: Zarr directory store > NPY memmap sidecars.
    """
    # 1. Zarr directory store
    if ZARR and str(cache_path).endswith(".zarr") and Path(cache_path).is_dir():
        try:
            z = _zarr_open_group(cache_path, mode="r")
            return int(min(z["X"].shape[0], z["y"].shape[0]))
        except Exception:
            return None
    # 2. NPY memory-map sidecars
    px, py = Path(_x_path(cache_path)), Path(_y_path(cache_path))
    if px.exists() and py.exists():
        X = np.load(str(px), mmap_mode="r")
        y = np.load(str(py), mmap_mode="r")
        return int(min(X.shape[0], y.shape[0]))
    return None


def _clamp_n_samples_to_disk(cache_path: str, n_samples: int) -> int:
    """Clamp n_samples to actual on-disk row count to prevent OOB in DataLoader workers."""
    n_disk = _on_disk_sequence_count(cache_path)
    if n_disk is None or n_disk >= n_samples:
        return n_samples
    print(f"[Data] WARN: on-disk arrays have {n_disk:,} rows but pipeline reported "
          f"{n_samples:,} ΓÇö clamping to {n_disk:,} (check X/Y export parity)")
    return n_disk


def _cache_length_snapshot(cache_path: str) -> dict:
    """
    Return cache lengths for integrity checks.
    Keys may include: zarr_X, zarr_y, npy_X, npy_y.
    """
    out: dict = {}
    p = Path(cache_path)
    # Zarr directory store
    if ZARR and p.is_dir() and (p / ".zgroup").exists():
        try:
            z = _zarr_open_group(cache_path, mode="r")
            if "X" in z: out["zarr_X"] = int(z["X"].shape[0])
            if "y" in z: out["zarr_y"] = int(z["y"].shape[0])
            if "y_cls" in z: out["zarr_y_cls"] = int(z["y_cls"].shape[0])
            if "pq" in z: out["zarr_pq"] = int(z["pq"].shape[0])
            if "diff" in z: out["zarr_diff"] = int(z["diff"].shape[0])
            for mk in _RL_MARKET_ZARR_KEYS:
                if mk in z:
                    out[f"zarr_{mk}"] = int(z[mk].shape[0])
        except Exception:
            out["zarr_unreadable"] = 1
    px, py = Path(_x_path(cache_path)), Path(_y_path(cache_path))
    try:
        import numpy.lib.format as np_fmt
        if px.exists():
            with open(px, "rb") as f:
                version = np_fmt.read_magic(f)
                shape, fortran, dtype = np_fmt._read_array_header(f, version)
            out["npy_X"] = int(shape[0])
        if py.exists():
            with open(py, "rb") as f:
                version = np_fmt.read_magic(f)
                shape, fortran, dtype = np_fmt._read_array_header(f, version)
            out["npy_y"] = int(shape[0])
        for key, path in (
            ("npy_y_cls", _y_cls_path(cache_path)),
            ("npy_pq", _pq_path(cache_path)),
            ("npy_diff", _diff_path(cache_path)),
            ("npy_close", _close_path(cache_path)),
            ("npy_atr", _atr_path(cache_path)),
            ("npy_spread", _spread_path(cache_path)),
        ):
            pp = Path(path)
            if pp.exists():
                with open(pp, "rb") as f:
                    version = np_fmt.read_magic(f)
                    shape, fortran, dtype = np_fmt._read_array_header(f, version)
                out[key] = int(shape[0])
    except Exception as e:
        print(f"[Cache] Direct NPY header read failed: {e}. Falling back to mmap.")
        if px.exists():
            out["npy_X"] = int(np.load(str(px), mmap_mode="r").shape[0])
        if py.exists():
            out["npy_y"] = int(np.load(str(py), mmap_mode="r").shape[0])
        for key, path in (
            ("npy_y_cls", _y_cls_path(cache_path)),
            ("npy_pq", _pq_path(cache_path)),
            ("npy_diff", _diff_path(cache_path)),
            ("npy_close", _close_path(cache_path)),
            ("npy_atr", _atr_path(cache_path)),
            ("npy_spread", _spread_path(cache_path)),
        ):
            pp = Path(path)
            if pp.exists():
                out[key] = int(np.load(str(pp), mmap_mode="r").shape[0])
    return out


def _validate_cache_integrity(cache_path: str, args=None) -> tuple[bool, str]:
    snap = _cache_length_snapshot(cache_path)
    problems = []
    
    # Manifest validation
    if args and not getattr(args, "ignore_manifest", False):
        manifest_path = Path(cache_path).with_name(Path(cache_path).name + "_manifest.json")
        manifest: dict = {}
        if not manifest_path.exists():
            # Support legacy _meta.json as a fallback
            legacy_meta = Path(cache_path).with_name(Path(cache_path).name + "_meta.json")
            if legacy_meta.exists():
                manifest_path = legacy_meta
            else:
                # Zarr stores may only have attrs (no sidecar) ΓÇö synthesize a manifest
                p_cache = Path(cache_path)
                if ZARR and p_cache.is_dir() and (p_cache / ".zgroup").exists():
                    try:
                        import zarr
                        z_store = zarr.open(str(p_cache), mode="r")
                        manifest = dict(getattr(z_store, "attrs", {}) or {})
                    except Exception:
                        manifest = {}
                if not manifest:
                    problems.append("dataset_manifest.json missing")
        
        if manifest_path.exists() and not manifest:
            import json
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
            except Exception as e:
                problems.append(f"Manifest read error: {e}")

        if manifest:
            try:
                expected_pairs = _get_pairs(args)
                manifest_pairs = manifest.get("pairs")
                if len(expected_pairs) > 1 and not manifest_pairs:

                    problems.append("Manifest missing pairs for multi-pair cache")

                if manifest_pairs:
                    if isinstance(manifest_pairs, str):
                        manifest_pairs = [p.strip().upper() for p in manifest_pairs.split(",") if p.strip()]
                    if manifest_pairs != expected_pairs:
                        problems.append(f"Manifest mismatch: pairs {manifest_pairs} != requested {expected_pairs}")
                    if len(expected_pairs) > 1:

                        schema_path = Path(str(cache_path) + "_feature_schema.json")

                        if not schema_path.exists():

                            problems.append("Multi-pair feature schema missing")

                        else:

                            try:

                                schema = json.loads(schema_path.read_text(encoding="utf-8"))

                                expected_n = int(manifest.get("n_features", 0) or 0)

                                if not isinstance(schema, list) or len(schema) != expected_n:

                                    problems.append(

                                        f"Multi-pair feature schema length {len(schema) if isinstance(schema, list) else 'invalid'} != n_features {expected_n}"

                                    )

                            except Exception as e:

                                problems.append(f"Multi-pair feature schema unreadable: {e}")

                if manifest.get("seq_len"):
                    requested_seq = _effective_max_seq_len(args)
                    if int(manifest.get("seq_len")) != int(requested_seq):
                        problems.append(
                            f"Manifest mismatch: seq_len {manifest.get('seq_len')} != "
                            f"required max {requested_seq} (training.seq_len / curriculum target)"
                        )
                if (

                    getattr(args, "label_method", "") == "rl_reward"

                    and manifest.get("y_cls_source") != "labels.label"

                ):

                    problems.append(

                        "Manifest y_cls_source is stale/missing; rebuild required so y_cls uses true direction labels"

                    )

                # We can also check bar_freq, strategy_mode, etc
            except Exception as e:
                problems.append(f"Manifest validation error: {e}")
                
    p = Path(cache_path)
    x_len = snap.get("zarr_X", snap.get("npy_X"))
    if args and getattr(args, "label_method", "") == "rl_reward" and x_len is not None:
        has_y_cls = "zarr_y_cls" in snap or "npy_y_cls" in snap
        if not has_y_cls:
            problems.append("RL reward cache missing y_cls direction sidecar")

    # Zarr checks
    if ZARR and p.is_dir() and (p / ".zgroup").exists():
        if snap.get("zarr_unreadable", 0) == 1:
            problems.append("Zarr store unreadable/corrupt")
        elif "zarr_X" not in snap or "zarr_y" not in snap:
            problems.append("Zarr store missing required arrays: X and/or y")
        if "zarr_y_cls" in snap and snap["zarr_y_cls"] != snap.get("zarr_X"):
            problems.append(
                f"Zarr y_cls={snap['zarr_y_cls']:,} != X={snap.get('zarr_X', 0):,}"
            )
        if "zarr_pq" in snap and snap["zarr_pq"] != snap.get("zarr_X"):
            problems.append(
                f"Zarr pq={snap['zarr_pq']:,} != X={snap.get('zarr_X', 0):,}"
            )
        if "zarr_diff" in snap and snap["zarr_diff"] != snap.get("zarr_X"):
            problems.append(
                f"Zarr diff={snap['zarr_diff']:,} != X={snap.get('zarr_X', 0):,}"
            )
        for mk in _RL_MARKET_ZARR_KEYS:
            zk = f"zarr_{mk}"
            if zk in snap and snap[zk] != snap.get("zarr_X"):
                problems.append(f"Zarr {mk}={snap[zk]:,} != X={snap.get('zarr_X', 0):,}")
    if "zarr_X" in snap and "zarr_y" in snap and snap["zarr_X"] != snap["zarr_y"]:
        problems.append(f"Zarr X={snap['zarr_X']:,} != y={snap['zarr_y']:,}")
    if "npy_X" in snap and "npy_y" in snap and snap["npy_X"] != snap["npy_y"]:
        problems.append(f"NPY X={snap['npy_X']:,} != y={snap['npy_y']:,}")
    for key in ("npy_y_cls", "npy_pq", "npy_diff", "npy_close", "npy_atr", "npy_spread"):
        if key in snap and "npy_X" in snap and snap[key] != snap["npy_X"]:
            problems.append(f"{key}={snap[key]:,} != NPY X={snap['npy_X']:,}")
    if not problems:
        return True, ""
    return False, " | ".join(problems)


def _verify_dataset(
    cache_path: str,
    args,
    n_samples: int,
    n_features: int,
    context: str = "Data",
) -> dict:
    """Comprehensive post-build verification of features, labels, and alignment.

    Returns a report dict with per-feature stats, label distribution,
    alignment checks, and anomaly flags.  All results are appended to
    build_log.jsonl and printed to stdout.
    """
    from data.dataset_manifest import DatasetManifest

    report = {
        "context": context,
        "n_samples": n_samples,
        "n_features": n_features,
        "features": {},
        "labels": {},
        "alignment": {},
        "anomalies": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        cache_p = Path(cache_path)
        is_zarr = cache_p.is_dir() and (cache_p / ".zgroup").exists()

        # ── Load a sample for verification ──────────────────────
        sample_n = min(n_samples, 5000) if n_samples > 0 else 0
        if sample_n == 0:
            report["anomalies"].append("zero_samples")
            return report

        if is_zarr:
            import zarr as _zarr
            z = _zarr.open(str(cache_path), mode="r")
            X_sample = z["X"][:sample_n]
            y_sample = z["y"][:sample_n]
            y_cls_sample = z.get("y_cls", None)
            if y_cls_sample is not None:
                y_cls_sample = y_cls_sample[:sample_n]
            time_idx_sample = None
            try:
                time_idx_sample = z["X"].attrs.get("time_idx")
            except Exception:
                pass
        else:
            x_path = Path(_x_path(cache_path))
            y_path = Path(_y_path(cache_path))
            X_sample = np.load(x_path, mmap_mode="r")[:sample_n]
            y_sample = np.load(y_path, mmap_mode="r")[:sample_n]
            y_cls_sample = None
            yc_path = Path(_y_cls_path(cache_path))
            if yc_path.exists():
                try:
                    y_cls_sample = np.load(str(yc_path), mmap_mode="r")[:sample_n]
                except Exception:
                    pass
            time_idx_sample = None

        # ── Per-feature statistics ──────────────────────────────
        if X_sample.ndim == 3:
            for feat_idx in range(min(X_sample.shape[2], 20)):
                col_data = X_sample[:, :, feat_idx].flatten()
                finite = col_data[np.isfinite(col_data)]
                nan_rate = 1.0 - (len(finite) / max(len(col_data), 1))
                feat_name = f"feature_{feat_idx}"
                try:
                    if hasattr(args, "_feat_names") and feat_idx < len(args._feat_names):
                        feat_name = args._feat_names[feat_idx]
                except Exception:
                    pass
                feat_report = {
                    "min": float(np.min(finite)) if len(finite) > 0 else None,
                    "max": float(np.max(finite)) if len(finite) > 0 else None,
                    "mean": float(np.mean(finite)) if len(finite) > 0 else None,
                    "std": float(np.std(finite)) if len(finite) > 0 else None,
                    "nan_rate": round(nan_rate, 6),
                    "n_finite": len(finite),
                }
                report["features"][feat_name] = feat_report
                if nan_rate > 0.05:
                    report["anomalies"].append(f"high_nan_{feat_name}:{nan_rate:.4f}")
                if len(finite) > 0 and np.std(finite) < 1e-12:
                    report["anomalies"].append(f"zero_variance_{feat_name}")

        # ── Label statistics ────────────────────────────────────
        y_finite = y_sample[np.isfinite(y_sample)]
        if len(y_finite) > 0:
            report["labels"] = {
                "mean_reward": round(float(np.mean(y_finite)), 6),
                "std_reward": round(float(np.std(y_finite)), 6),
                "min_reward": round(float(np.min(y_finite)), 6),
                "max_reward": round(float(np.max(y_finite)), 6),
                "n_finite": len(y_finite),
                "nan_rate": round(1.0 - len(y_finite) / len(y_sample), 6),
            }
            # Direction distribution if y_cls available
            if y_cls_sample is not None:
                y_cls_finite = y_cls_sample[np.isfinite(y_cls_sample)]
                if len(y_cls_finite) > 0:
                    unique, counts = np.unique(y_cls_finite, return_counts=True)
                    dist = {int(v): int(c) for v, c in zip(unique, counts)}
                    report["labels"]["direction_dist"] = dist
                    total = sum(dist.values())
                    for side, cnt in dist.items():
                        pct = cnt / total * 100
                        if pct < 5 and side != 0:
                            report["anomalies"].append(
                                f"rare_direction_{side}: {pct:.1f}%"
                            )
        else:
            report["anomalies"].append("all_labels_nan")

        # ── Alignment check ─────────────────────────────────────
        report["alignment"] = {
            "n_samples": n_samples,
            "n_features": n_features,
            "X_shape": list(X_sample.shape),
            "y_shape": list(y_sample.shape),
            "y_cls_available": y_cls_sample is not None,
        }

        # ── Time index monotonicity (if available) ──────────────
        if time_idx_sample is not None:
            try:
                tidx = np.asarray(time_idx_sample[:sample_n], dtype=np.int64)
                if len(tidx) > 1:
                    monotonic = bool(np.all(np.diff(tidx) >= 0))
                    report["alignment"]["time_index_monotonic"] = monotonic
                    if not monotonic:
                        report["anomalies"].append("time_index_not_monotonic")
            except Exception:
                pass

    except Exception as e:
        report["anomalies"].append(f"verification_error: {e}")

    # ── Log results ──────────────────────────────────────────
    try:
        dm = DatasetManifest(str(Path(cache_path).parent))
        dm.log_build_event(
            "verification_complete",
            n_rows=n_samples,
            n_features=n_features,
            extra={
                "anomaly_count": len(report["anomalies"]),
                "anomalies": report["anomalies"],
                "label_mean": report["labels"].get("mean_reward"),
                "label_std": report["labels"].get("std_reward"),
                "direction_dist": report["labels"].get("direction_dist"),
            },
        )
    except Exception:
        pass

    # ── Print summary ────────────────────────────────────────
    anomaly_str = ""
    if report["anomalies"]:
        anomaly_str = f" ⚠ ANOMALIES: {', '.join(report['anomalies'][:10])}"
        if len(report["anomalies"]) > 10:
            anomaly_str += f" (+{len(report['anomalies']) - 10} more)"

    feat_nan = [f for f, s in report["features"].items() if s.get("nan_rate", 0) > 0.05]
    if feat_nan:
        anomaly_str += f" | high-NaN features: {len(feat_nan)}"

    print(
        f"[{context}] Verify: {n_samples:,} samples x {n_features} features"
        f" | reward μ={report['labels'].get('mean_reward', 'N/A')}"
        f" σ={report['labels'].get('std_reward', 'N/A')}"
        f"{anomaly_str}"
    )

    return report


def _postprocess_cache_integrity_check(cache_path: str, args, *, context: str = "Data") -> None:
    """Fail immediately if a freshly processed cache is incomplete or inconsistent."""
    ok, reason = _validate_cache_integrity(cache_path, args)
    if not ok:
        raise RuntimeError(
            f"[{context}] Post-processing cache integrity failed: {reason}. "
            "Delete/rebuild the processed cache before training."
        )
    snap = _cache_length_snapshot(cache_path)
    n_rows = snap.get("zarr_X", snap.get("npy_X", 0))
    print(f"[{context}] Post-processing cache integrity PASS ({int(n_rows):,} rows)")


def _cache_has_multitask_sidecars(cache_path: str) -> bool:
    """True when y_cls sidecar exists with the same row count as X."""
    p = Path(cache_path)
    n_x = _on_disk_sequence_count(cache_path)
    if n_x is None:
        return False
    if ZARR and p.is_dir() and (p / ".zgroup").exists():
        try:
            z = _zarr_open_group(cache_path, mode="r")
            if "y_cls" not in z:
                return False
            return int(z["y_cls"].shape[0]) == int(n_x)
        except Exception:
            return False
    yp = Path(_y_cls_path(cache_path))
    if not yp.exists():
        return False
    try:
        return int(np.load(str(yp), mmap_mode="r").shape[0]) == int(n_x)
    except Exception:
        return False


def _warn_multitask_cache_sidecars(cache_path: str, args) -> None:
    """Hint to rebuild when multitask training expects y_cls/pq sidecars."""
    if not getattr(args, "multitask", False):
        return
    if _cache_has_multitask_sidecars(cache_path):
        return
    msg = (
        "[Data] Multitask is enabled but cache has no y_cls sidecar "
        f"({cache_path}). Direction/confidence heads will threshold rewards "
        "instead of true class labels.\n"
        "  Rebuild cache: .\\.venv-gpu\\Scripts\\python.exe scripts\\train.py "
        "--rebuild-cache\n"
        "  Or: training\\train_gpu.py --force-rebuild"
    )
    print(msg, flush=True)
    _log_warn(msg)


def _delete_cache_artifacts(cache_path: str) -> None:
    import shutil as _shutil
    p = Path(cache_path)
    # Zarr is a directory ΓÇö use shutil.rmtree
    if p.is_dir() and str(cache_path).endswith(".zarr"):
        _shutil.rmtree(p)
        print(f"[Data] Removed corrupt zarr store: {p}")
    elif p.exists():
        p.unlink()
        print(f"[Data] Removed corrupt cache artifact: {p}")
    for fp in (
        Path(_x_path(cache_path)), Path(_y_path(cache_path)), _scaler_npz_path(p),
        Path(_diff_path(cache_path)), Path(_pq_path(cache_path)),
        Path(_y_cls_path(cache_path)),
        Path(_close_path(cache_path)), Path(_atr_path(cache_path)),
        Path(_spread_path(cache_path)),
        Path(str(cache_path) + "_manifest.json"),

        Path(str(cache_path) + "_meta.json"),

        Path(str(cache_path) + "_resume.json"),

        Path(str(cache_path) + "_feature_schema.json"),

        Path(str(cache_path) + "_pair_readiness_report.json"),
    ):
        if fp.exists():
            fp.unlink()
            print(f"[Data] Removed corrupt cache artifact: {fp}")



def _core_model(model: "nn.Module") -> "nn.Module":
    return model.module if hasattr(model, "module") else model


def _identity_scaler(n_features: int) -> StandardScaler:
    s = StandardScaler()
    s.mean_ = np.zeros(n_features)
    s.scale_ = np.ones(n_features)
    s.var_ = np.ones(n_features)
    s.n_features_in_ = n_features
    return s


def _set_scaler_feature_names(scaler: StandardScaler, columns) -> None:

    """Attach ordered feature names when fitting scalers on numpy arrays."""

    try:

        if getattr(scaler, "feature_names_in_", None) is not None:

            return

        names = [str(c) for c in columns]

        if names:

            scaler.feature_names_in_ = np.asarray(names, dtype=str)

    except Exception:

        pass





def _scaler_feature_names(scaler: Optional[StandardScaler]) -> list[str]:

    try:

        names = getattr(scaler, "feature_names_in_", None)

        if names is not None:

            return [str(c) for c in list(names)]

    except Exception:

        pass

    return []





def _write_feature_schema_json(cache_path: Path, feature_names: list[str]) -> None:

    if not feature_names:

        return

    try:

        import json

        with open(str(cache_path) + "_feature_schema.json", "w", encoding="utf-8") as f:

            json.dump([str(c) for c in feature_names], f)

    except Exception:

        pass





def _build_multipair_feature_schema(

    scalers: dict,

    pairs: list[str],

    n_features_total: int,

) -> list[str]:

    """Return the ordered full schema for pair-concatenated caches."""

    names: list[str] = []

    for pair in pairs:

        pair_names = _scaler_feature_names(scalers.get(pair))

        if not pair_names:

            return []

        names.extend([f"{pair}::{name}" for name in pair_names])

    if len(names) != int(n_features_total):

        return []

    return names





def _save_scaler_npz(cache_path: Path, scaler: StandardScaler) -> None:
    if not hasattr(scaler, "mean_") or scaler.mean_ is None:
        return
    p = _scaler_npz_path(cache_path)
    payload = dict(
        mean=scaler.mean_,
        scale=scaler.scale_,
        var=scaler.var_,
        n_features_in_=int(scaler.n_features_in_),
        n_samples_seen_=int(getattr(scaler, "n_samples_seen_", 0) or 0),
    )
    if hasattr(scaler, "feature_names_in_") and scaler.feature_names_in_ is not None:
        payload["feature_names"] = np.asarray([str(c) for c in scaler.feature_names_in_], dtype=str)

    np.savez(p, **payload)


def _load_scaler_npz(cache_path: Path) -> Optional[StandardScaler]:
    p = _scaler_npz_path(cache_path)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    s = StandardScaler()
    s.mean_ = np.asarray(z["mean"], dtype=np.float64)
    s.scale_ = np.asarray(z["scale"], dtype=np.float64)
    s.var_ = np.asarray(z["var"], dtype=np.float64)
    s.n_features_in_ = int(z["n_features_in_"])
    if "n_samples_seen_" in z.files:
        s.n_samples_seen_ = int(z["n_samples_seen_"])
    if "feature_names" in z.files:
        s.feature_names_in_ = np.asarray(z["feature_names"], dtype=object)
    return s


def _ticks_have_usable_datetime_index(ticks: "pd.DataFrame") -> bool:
    """True when tick index is time-like enough for OHLC resampling."""
    idx = getattr(ticks, "index", None)
    if idx is None or len(idx) == 0:
        return False
    if isinstance(idx, pd.DatetimeIndex):
        return True
    return bool(pd.api.types.is_datetime64_any_dtype(idx))


def _normalize_tick_index_utc(ticks: "pd.DataFrame") -> "pd.DataFrame":
    """Ensure a proper UTC DatetimeIndex (pandas resample requires this)."""
    out = ticks.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out.index.name = "timestamp"
    return out


def _multipair_zero_samples_help(
    pair_ticks: Optional[Dict[str, "pd.DataFrame"]],
) -> str:
    lines = ["Per-pair tick load summary:"]
    if not pair_ticks:
        lines.append("  (no tick dict ΓÇö loader failed.)")
        return "\n".join(lines)
    for p, df in pair_ticks.items():
        if df is None:
            lines.append(f"  {p}: None")
            continue
        n = len(df)
        idx = getattr(df, "index", None)
        kind = type(idx).__name__ if idx is not None else "None"
        dt_ok = _ticks_have_usable_datetime_index(df)
        lines.append(
            f"  {p}: ticks={n:,} index={kind} datetime_ok={dt_ok}"
        )
        if n > 0 and idx is not None:
            try:
                lines.append(
                    f"       range: {idx.min()} -> {idx.max()}"
                )
            except Exception:
                pass
    lines.append(
        "If every pair shows ticks=0: hour files may be empty (blocked download or bad cache). "
        "Delete data/raw/dukascopy/<PAIR>/ and rerun, shorten the date range, or set "
        "data.full_day_data: true. If ticks>0 but datetime_ok=False, tick index is malformed."
    )
    return "\n".join(lines)


def _json_scalar(value):
    if value is None:
        return None
    try:
        if isinstance(value, np.generic):
            value = value.item()
    except Exception:
        pass
    try:
        import pandas as pd
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
    except Exception:
        pass
    if isinstance(value, np.datetime64):
        return str(value.astype("datetime64[ns]"))
    if isinstance(value, (float, int, str, bool)):
        return value
    return str(value)


def _pair_readiness_entry(pair: str) -> dict:
    global _PAIR_READINESS_STATS
    if "_PAIR_READINESS_STATS" not in globals():
        _PAIR_READINESS_STATS = {}
    if pair not in _PAIR_READINESS_STATS:
        _PAIR_READINESS_STATS[pair] = {
            "pair": pair,
            "raw_ticks": 0,
            "windows_seen": 0,
            "timestamp_start": None,
            "timestamp_end": None,
            "duplicate_timestamps": 0,
            "_observed_hours": [],

            "empty_windows": 0,

            "timestamp_utc_normalized": False,

            "missing_columns": [],
            "schema_errors": [],
            "bars_after_resampling": 0,
            "dropped_bars": 0,
            "dropped_bars_by_reason": {

                "weekend": 0,

                "holiday": 0,

                "dead_bar": 0,

                "spread": 0,

                "atr": 0,

                "news": 0,

                "label_filter": 0,

            },

            "dropped_sequences": 0,
            "dropped_sequences_by_reason": {

                "row_quality": 0,

                "weekend": 0,

                "holiday": 0,

                "dead_bar": 0,

                "spread": 0,

                "atr": 0,

                "news": 0,

                "label_filter": 0,

                "zero_feature_window": 0,

                "invalid_reward_label": 0,

                "invalid_direction_label": 0,

                "path_quality_filter": 0,

            },

            "label_filter_counts": {

                "invalid_label": 0,

                "invalid_reward": 0,

                "invalid_direction_label": 0,

                "invalid_path_quality": 0,

            },

            "nan_count": 0,
            "posinf_count": 0,
            "neginf_count": 0,
            "seq_count": 0,
            "label_counts": {},
            "difficulty_counts": {"0": 0, "1": 0, "2": 0},
            "spread": {},
            "atr": {},
            "valid": True,
            "reasons": [],
        }
    return _PAIR_READINESS_STATS[pair]


def _bump_reason_counts(entry: dict, field: str, counts: Optional[dict]) -> None:

    if not counts:

        return

    target = entry.setdefault(field, {})

    for key, value in counts.items():

        try:

            inc = int(value)

        except Exception:

            continue

        if inc <= 0:

            continue

        target[str(key)] = int(target.get(str(key), 0) or 0) + inc





def _update_pair_readiness_raw(pair: str, ticks: "pd.DataFrame") -> None:
    entry = _pair_readiness_entry(pair)
    entry["windows_seen"] += 1
    if ticks is None:
        entry["schema_errors"].append("ticks_none")
        entry["valid"] = False
        return
    n_ticks = int(len(ticks))

    if n_ticks == 0:

        entry["empty_windows"] = int(entry.get("empty_windows", 0) or 0) + 1

        return

    entry["raw_ticks"] += n_ticks

    cols = set(getattr(ticks, "columns", []))
    if "timestamp_utc" in cols:
        ts_raw = ticks["timestamp_utc"]
    else:
        ts_raw = getattr(ticks, "index", None)
        if ts_raw is None:
            entry["missing_columns"].append("timestamp_utc")
            entry["schema_errors"].append("timestamp_missing")
            entry["timestamp_utc_normalized"] = False
            entry["valid"] = False
            return

    if not ({"bid", "ask"}.issubset(cols) or {"bid_close", "ask_close"}.issubset(cols)):
        for col in ("bid", "ask"):
            if col not in cols:
                entry["missing_columns"].append(col)

    try:
        import pandas as pd
        # ts_raw may be a Polars Series (Datetime ns UTC), numpy array, or pandas Series/Index.

        if "polars" in str(type(ts_raw)).lower():

            # Force conversion to standard numpy datetime64 to avoid pyarrow/polars incompatibilities with pd.to_datetime

            ts_raw_pd = ts_raw.to_pandas()

            if hasattr(ts_raw_pd, 'dt') and ts_raw_pd.dt.tz is not None:

                ts_raw_pd = ts_raw_pd.dt.tz_convert(None)

            ts_raw_pd = ts_raw_pd.astype("datetime64[ns]")

        elif hasattr(ts_raw, "to_pandas"):

            ts_raw_pd = ts_raw.to_pandas()

        else:

            ts_raw_pd = ts_raw



        ts_utc = pd.to_datetime(ts_raw_pd, utc=True, errors="coerce")

        valid_ts = ts_utc[~pd.isna(ts_utc)]
        if len(valid_ts):
            entry["timestamp_utc_normalized"] = True

            start = valid_ts.min()
            end = valid_ts.max()
            if entry["timestamp_start"] is None or str(start) < str(entry["timestamp_start"]):
                entry["timestamp_start"] = _json_scalar(start)
            if entry["timestamp_end"] is None or str(end) > str(entry["timestamp_end"]):
                entry["timestamp_end"] = _json_scalar(end)
            entry["duplicate_timestamps"] += int(pd.Series(valid_ts).duplicated().sum())
            try:

                hours = pd.Series(valid_ts).dt.floor("h").dropna().astype(str).unique().tolist()

                observed = set(entry.get("_observed_hours", []) or [])

                observed.update(hours)

                entry["_observed_hours"] = sorted(observed)

            except Exception:

                pass

        else:
            entry["schema_errors"].append("timestamp_parse_failed")
            entry["valid"] = False
    except Exception as exc:
        entry["schema_errors"].append(f"timestamp_check_failed:{type(exc).__name__}")
        entry["valid"] = False

    try:
        if {"bid", "ask"}.issubset(cols):
            bad_bidask = (ticks["ask"].astype(float) <= ticks["bid"].astype(float)).sum()
            if int(bad_bidask) > 0:
                entry["schema_errors"].append(f"ask_lte_bid:{int(bad_bidask)}")
                entry["valid"] = False
    except Exception:
        pass


def _update_pair_readiness_processed(
    pair: str,
    *,
    bars_count: int = 0,
    feature_frame=None,
    labels=None,
    dropped_rows: int = 0,
    dropped_sequences: int = 0,
    diff_seq=None,
    spread_seq=None,
    atr_seq=None,
) -> None:
    entry = _pair_readiness_entry(pair)
    entry["bars_after_resampling"] += int(bars_count or 0)
    entry["dropped_bars"] += int(dropped_rows or 0)
    entry["dropped_sequences"] += int(dropped_sequences or 0)
    if feature_frame is not None:
        try:
            numeric = feature_frame.select_dtypes(include=[np.number])
            vals = numeric.to_numpy(dtype=np.float64, copy=False)
            entry["nan_count"] += int(np.isnan(vals).sum())
            entry["posinf_count"] += int(np.isposinf(vals).sum())
            entry["neginf_count"] += int(np.isneginf(vals).sum())
        except Exception:
            pass
    if labels is not None:
        try:
            label_col = "label" if "label" in labels.columns else None
            if label_col:
                counts = labels[label_col].value_counts(dropna=False).to_dict()
                for k, v in counts.items():
                    key = str(int(k)) if isinstance(k, (int, float, np.integer, np.floating)) and np.isfinite(k) else str(k)
                    entry["label_counts"][key] = int(entry["label_counts"].get(key, 0) + int(v))
        except Exception:
            pass
    if diff_seq is not None:
        try:
            vals, counts = np.unique(np.asarray(diff_seq, dtype=np.uint8), return_counts=True)
            for v, c in zip(vals, counts):
                entry["difficulty_counts"][str(int(v))] = int(entry["difficulty_counts"].get(str(int(v)), 0) + int(c))
        except Exception:
            pass
    for name, arr in (("spread", spread_seq), ("atr", atr_seq)):
        if arr is None:
            continue
        try:
            a = np.asarray(arr, dtype=np.float64)
            a = a[np.isfinite(a)]
            if a.size:
                existing = entry.get(name, {})
                samples = existing.get("_samples", [])
                samples.extend(a[::max(1, len(a) // 500)].tolist())
                samples = samples[-2000:]
                entry[name] = {
                    "median": float(np.median(samples)),
                    "p95": float(np.percentile(samples, 95)),
                    "max": float(np.max(samples)),
                    "_samples": samples,
                }
        except Exception:
            pass


def _finalize_pair_readiness_report(args, cache_path, pairs, *, alignment: Optional[dict] = None) -> dict:
    stats = globals().get("_PAIR_READINESS_STATS", {})
    pair_reports = []
    failed = False
    warnings = []
    for pair in pairs:
        entry = dict(stats.get(pair, _pair_readiness_entry(pair)))
        seq = int(entry.get("seq_count", 0) or 0)
        n_feat = int(entry.get("n_features", 0) or 0)
        seq_len = int(getattr(args, "seq_len", 60) or 60)
        total_values = max(1, seq * max(1, seq_len) * max(1, n_feat))
        nonfinite = int(entry.get("nan_count", 0) + entry.get("posinf_count", 0) + entry.get("neginf_count", 0))
        nonfinite_pct = 100.0 * nonfinite / total_values
        schema_errors = list(dict.fromkeys(entry.get("schema_errors", [])))

        reasons = list(dict.fromkeys(entry.get("reasons", [])))

        raw_ticks = int(entry.get("raw_ticks", 0) or 0)

        for err in schema_errors:

            if err == "timestamp_parse_failed":

                if raw_ticks == 0 or seq == 0:

                    reasons.append(err)

            else:

                reasons.append(err)

        if entry.get("missing_columns"):
            reasons.append("missing_required_columns")
        if seq == 0:
            reasons.append("zero_valid_sequences")
        if raw_ticks > 0 and not entry.get("timestamp_utc_normalized", False):

            reasons.append("timestamps_not_confirmed_utc")
        if nonfinite_pct > 1.0:
            reasons.append("feature_nonfinite_pct_gt_1")
        label_counts = entry.get("label_counts", {})
        non_hold = sum(int(v) for k, v in label_counts.items() if str(k) not in {"0", "0.0"})
        if label_counts and non_hold == 0:
            reasons.append("label_classes_collapsed_to_hold")
        diff_counts = entry.get("difficulty_counts", {})
        if diff_counts and int(diff_counts.get("1", 0)) == 0 and int(diff_counts.get("2", 0)) == 0 and seq > 0:
            warnings.append(f"{pair}: difficulty_all_easy")
        observed_hours = sorted(set(entry.get("_observed_hours", []) or []))

        missing_hours = []

        coverage = {}

        try:

            import pandas as pd

            if entry.get("timestamp_start") and entry.get("timestamp_end") and observed_hours:

                start_ts = pd.Timestamp(entry["timestamp_start"]).floor("h")

                end_ts = pd.Timestamp(entry["timestamp_end"]).floor("h")

                expected_hours = pd.date_range(start=start_ts, end=end_ts, freq="h", tz="UTC")

                expected_set = {str(ts) for ts in expected_hours}

                observed_set = {str(pd.Timestamp(h).tz_convert("UTC") if pd.Timestamp(h).tzinfo else pd.Timestamp(h).tz_localize("UTC")) for h in observed_hours}

                missing_hours = sorted(expected_set - observed_set)

                coverage = {

                    "expected_hours": int(len(expected_set)),

                    "observed_hours": int(len(observed_set)),

                    "missing_hours": int(len(missing_hours)),

                    "coverage_pct": round(100.0 * len(observed_set) / max(1, len(expected_set)), 6),

                    "missing_hours_sample": missing_hours[:50],

                }

                if missing_hours:

                    warnings.append(f"{pair}: missing_hours={len(missing_hours)}")

        except Exception as exc:

            coverage = {"error": f"missing_hour_check_failed:{type(exc).__name__}"}

        status = "fail" if reasons else ("warn" if nonfinite > 0 else "pass")
        if status == "fail":
            failed = True
        clean_entry = {k: v for k, v in entry.items() if k not in {"valid", "_observed_hours"}}

        for metric in ("spread", "atr"):
            if isinstance(clean_entry.get(metric), dict):
                clean_entry[metric] = {k: v for k, v in clean_entry[metric].items() if not k.startswith("_")}
        clean_entry.update({
            "n_features": n_feat,
            "nonfinite_count": nonfinite,
            "nonfinite_pct": round(nonfinite_pct, 6),
            "hour_coverage": coverage,

            "status": status,
            "reasons": list(dict.fromkeys(reasons)),
        })
        pair_reports.append(clean_entry)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_path": str(cache_path),
        "data_source": str(getattr(args, "data_source", "")),
        "start_date": str(getattr(args, "data_start", "")),
        "end_date": str(getattr(args, "data_end", "")),
        "bar_freq": str(getattr(args, "bar_freq", "1min")),
        "seq_len": int(getattr(args, "seq_len", 0) or 0),
        "pairs": pair_reports,
        "alignment": alignment or {},
        "warnings": warnings,
        "status": "fail" if failed else ("warn" if warnings else "pass"),
    }
    return report


def _write_pair_readiness_report(args, cache_path, pairs, *, alignment: Optional[dict] = None) -> dict:
    report = _finalize_pair_readiness_report(args, cache_path, pairs, alignment=alignment)
    path = Path(str(cache_path) + "_pair_readiness_report.json")
    try:
        _safe_save_json(report, path)
    except NameError:
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[Pair Readiness] report -> {path} ({report['status'].upper()})")
    return report


def _compute_difficulty_scores(feats: "pd.DataFrame", seq_len: int) -> np.ndarray:
    """
    B: Per-sample difficulty score (uint8: 0=easy, 1=medium, 2=hard).

    Difficulty is assigned per bar using four independent signals.
    The bar's score is the MAX of all signals (worst-case wins):

    Session signal (UTC hour):
      0 = easy   -> London / London-NY overlap: 07-17 UTC (peak liquidity)
      0 = easy   -> NY session: 13-21 UTC (high liquidity, strong trends)
      1 = medium -> Asian session: 00-07 UTC (thinner, range-bound)
      1 = medium -> Pre-London / post-NY: 05-07, 18-21 UTC
      2 = hard   -> Rollover / off-hours: 21-24 + 00-01 UTC (widest spreads,
                   swap charges, thin books)

    Spread signal (liquidity):
      Uses `liquidity_vacuum` if available (spread / median_spread ratio),
      else falls back to `spread_avg` / rolling median.
      1 = medium if spread > 1.5├ù median
      2 = hard   if spread > 2.0├ù median

    News window signal:
      2 = hard when `news_ok == 0` (within ┬▒15 min of high-impact release)
      1 = medium when `eco_surprise` is nonzero (data release bar itself)

    Volatility spike signal:
      2 = hard when `vol_ok == 0` (ATR > 3├ù rolling mean ΓÇö spike / flash crash)

    The difficulty of window i is that of its *last* bar (index i + seq_len - 1),
    matching how labels are assigned.
    """
    import pandas as pd

    idx      = feats.index
    n        = len(feats)
    hour_arr = np.array(idx.hour if hasattr(idx, "hour") else np.zeros(n, dtype=np.int32),
                        dtype=np.int32)

    # -- Signal 1: session ----------------------------------------------------
    # hard = rollover (21ΓÇô01 UTC): swap charges, thin books, wide spreads
    # medium = pure Asia (01ΓÇô07 UTC) or late-NY tail (18ΓÇô21 UTC)
    # easy = London peak (07ΓÇô17) + NY core (13ΓÇô21 overlap included in easy)
    is_rollover = ((hour_arr >= 21) | (hour_arr < 1))           # 21:00ΓÇô01:00 UTC
    is_asia     = ((hour_arr >= 1)  & (hour_arr < 7))           # 01:00ΓÇô07:00 UTC
    is_late_ny  = ((hour_arr >= 18) & (hour_arr < 21))          # 18:00ΓÇô21:00 UTC
    session_diff = np.where(is_rollover, 2,
                   np.where(is_asia | is_late_ny, 1, 0)).astype(np.uint8)

    base_diff = session_diff.copy()

    # -- Signal 2: spread / liquidity ----------------------------------------
    if "liquidity_vacuum" in feats.columns:
        lv = feats["liquidity_vacuum"].ffill().fillna(1.0).values
        spread_diff = np.where(lv > 2.0, 2, np.where(lv > 1.5, 1, 0)).astype(np.uint8)
        base_diff = np.maximum(base_diff, spread_diff)
    elif "spread_avg" in feats.columns:
        spr = feats["spread_avg"].ffill().fillna(0.0).values
        med_spr = pd.Series(spr).rolling(120, min_periods=10).median().ffill().fillna(0.0).values
        spr_ratio = np.where(med_spr > 0, spr / np.maximum(med_spr, 1e-10), 1.0)
        spread_diff = np.where(spr_ratio > 2.0, 2, np.where(spr_ratio > 1.5, 1, 0)).astype(np.uint8)
        base_diff = np.maximum(base_diff, spread_diff)

    # -- Signal 3: news windows -----------------------------------------------
    # news_ok=0 means within ┬▒15 min of a high-impact economic release.
    # These bars have erratic price action and artificially wide spreads.
    if "news_ok" in feats.columns:
        news_ok = feats["news_ok"].fillna(1.0).values
        news_diff = np.where(news_ok < 0.5, 2, 0).astype(np.uint8)
        base_diff = np.maximum(base_diff, news_diff)

    # eco_surprise != 0 -> the release bar itself (not the buffer) -> medium
    if "eco_surprise" in feats.columns:
        eco = feats["eco_surprise"].fillna(0.0).values
        eco_diff = np.where(eco != 0.0, np.maximum(np.ones(n, dtype=np.uint8), base_diff), base_diff).astype(np.uint8)
        base_diff = eco_diff

    # -- Signal 4: volatility spike -------------------------------------------
    # vol_ok=0 means ATR > 3├ù its rolling mean (flash crash / spike).
    # Model should not learn patterns from these bars.
    if "vol_ok" in feats.columns:
        vol_ok = feats["vol_ok"].fillna(1.0).values
        spike_diff = np.where(vol_ok < 0.5, 2, 0).astype(np.uint8)
        base_diff = np.maximum(base_diff, spike_diff)

    # -- Align to sequence windows --------------------------------------------
    # Window i uses bars [i, i+seq_len). Its difficulty = last bar's score.
    n_seq = len(base_diff) - seq_len + 1
    if n_seq <= 0:
        return np.array([], dtype=np.uint8)
    diff_seq = base_diff[seq_len - 1 : seq_len - 1 + n_seq]
    return diff_seq.astype(np.uint8)


def _robust_clip_frame(feats: "pd.DataFrame", *, q_low: float = 0.001, q_high: float = 0.999) -> "pd.DataFrame":
    numeric = feats.select_dtypes(include=[np.number])
    if numeric.empty:
        return feats
    lo = numeric.quantile(q_low)
    hi = numeric.quantile(q_high)
    clipped = numeric.clip(lower=lo, upper=hi, axis=1)
    out = feats.copy()
    out[clipped.columns] = clipped
    return out


def _compute_row_quality_mask(
    bars: "pd.DataFrame",
    feats: "pd.DataFrame",
    labels: "pd.DataFrame",
) -> "tuple[pd.Series, dict, dict, dict]":

    bars_aligned = bars.reindex(feats.index).ffill()
    mask = pd.Series(True, index=feats.index)
    reason_masks: dict[str, "pd.Series"] = {}

    label_filter_counts = {

        "invalid_label": 0,

        "invalid_reward": 0,

        "invalid_direction_label": 0,

        "invalid_path_quality": 0,

    }



    def _reason(name: str, bad) -> None:

        nonlocal mask

        bad_s = pd.Series(bad, index=feats.index).fillna(False).astype(bool)

        reason_masks[name] = reason_masks.get(name, pd.Series(False, index=feats.index)) | bad_s

        mask &= ~bad_s



    try:

        idx = pd.DatetimeIndex(feats.index)

        if len(idx):

            weekday = idx.weekday

            hour = idx.hour

            weekend_closed = (weekday == 5) | ((weekday == 6) & (hour < 21)) | ((weekday == 4) & (hour >= 22))

            _reason("weekend", weekend_closed)

    except Exception:

        pass



    for col in ("holiday", "is_holiday", "holiday_flag", "market_holiday"):

        if col not in feats.columns:

            continue

        try:

            vals = feats[col]

            if pd.api.types.is_numeric_dtype(vals):

                _reason("holiday", vals.astype(float) > 0.5)

            else:

                _reason("holiday", vals.astype(str).str.lower().isin({"1", "true", "yes", "holiday"}))

        except Exception:

            pass


    dead_bar = pd.Series(False, index=feats.index)

    for col in ("open", "high", "low", "close"):
        if col in bars_aligned.columns:
            vals = bars_aligned[col].astype(float)
            dead_bar |= ~(np.isfinite(vals) & (vals > 0))

    if {"high", "low", "close"}.issubset(bars_aligned.columns):
        high = bars_aligned["high"].astype(float)

        low = bars_aligned["low"].astype(float)

        close = bars_aligned["close"].astype(float)

        dead_bar |= ~(high >= low)

        dead_bar |= ~close.between(low, high)

        if "open" in bars_aligned.columns:

            open_ = bars_aligned["open"].astype(float)

            flat_ohlc = ((high - low).abs() <= 1e-12) & ((open_ - close).abs() <= 1e-12)

            if "volume" in bars_aligned.columns:

                vol = pd.to_numeric(bars_aligned["volume"], errors="coerce").fillna(0.0)

                dead_bar |= flat_ohlc & (vol <= 0.0)

            else:

                dead_bar |= flat_ohlc

    _reason("dead_bar", dead_bar)


    for col in ("atr_6", "atr_20", "spread_pips", "liquidity_vacuum", "vol_ok", "news_ok"):
        if col in feats.columns:
            vals = feats[col].astype(float)
            bad_finite = ~np.isfinite(vals)

            if col == "news_ok":
                # Exclude high impact windows from training entirely!
                _reason("news", bad_finite | (vals <= 0.5))

            elif col in {"spread_pips", "liquidity_vacuum"}:

                _reason("spread", bad_finite)

            elif col in {"atr_6", "atr_20", "vol_ok"}:

                _reason("atr", bad_finite)

    if "atr_6" in feats.columns:
        atr = feats["atr_6"].astype(float)
        med_atr = atr.rolling(500, min_periods=50).median().ffill().fillna(atr.median())
        _reason("atr", (atr <= 0) | (atr > (med_atr * 25.0).replace(0, np.inf)))

    if "vol_ok" in feats.columns:

        _reason("atr", feats["vol_ok"].astype(float) <= 0.5)

    if "spread_pips" in feats.columns:
        spr = feats["spread_pips"].astype(float)
        med_spr = spr.rolling(500, min_periods=50).median().ffill().fillna(spr.median())
        _reason("spread", (spr < 0) | (spr > (med_spr * 20.0 + 0.1)))

    if "liquidity_vacuum" in feats.columns:
        _reason("spread", feats["liquidity_vacuum"].astype(float) > 20.0)


    if "label" in labels.columns:
        y = labels["label"].reindex(feats.index)
        bad_label = ~y.isin([-1, 0, 1])

        label_filter_counts["invalid_label"] += int(bad_label.fillna(True).sum())

        _reason("label_filter", bad_label)

    if "reward" in labels.columns:
        r = labels["reward"].reindex(feats.index).astype(float)
        bad_reward = ~np.isfinite(r)

        label_filter_counts["invalid_reward"] += int(bad_reward.fillna(True).sum())

        _reason("label_filter", bad_reward)


    reason_counts = {k: int(v.fillna(False).sum()) for k, v in reason_masks.items()}

    return mask.fillna(False), reason_counts, label_filter_counts, reason_masks



def _sequence_quality_mask(
    X_arr: np.ndarray,
    row_ok: np.ndarray,
    seq_len: int,
    *,
    max_bad_frac: float = 0.05,
    max_zero_frac: float = 0.80,
) -> np.ndarray:
    n_seq = len(X_arr) - seq_len + 1
    if n_seq <= 0:
        return np.zeros(0, dtype=bool)
    row_bad = (~row_ok.astype(bool)).astype(np.float32)
    bad_counts = np.convolve(row_bad, np.ones(seq_len, dtype=np.float32), mode="valid")
    window_ok = bad_counts <= max(0.0, max_bad_frac) * seq_len
    zero_rows = (np.abs(X_arr).sum(axis=1) <= 1e-8).astype(np.float32)
    zero_counts = np.convolve(zero_rows, np.ones(seq_len, dtype=np.float32), mode="valid")
    window_ok &= zero_counts <= max_zero_frac * seq_len
    return window_ok.astype(bool)


def _sequence_quality_reason_masks(

    X_arr: np.ndarray,

    row_ok: np.ndarray,

    row_reason_masks: Optional[dict],

    seq_len: int,

    *,

    max_bad_frac: float = 0.05,

    max_zero_frac: float = 0.80,

) -> dict[str, np.ndarray]:

    n_seq = len(X_arr) - seq_len + 1

    if n_seq <= 0:

        return {}

    out: dict[str, np.ndarray] = {}

    row_bad = (~row_ok.astype(bool)).astype(np.float32)

    bad_counts = np.convolve(row_bad, np.ones(seq_len, dtype=np.float32), mode="valid")

    out["row_quality"] = bad_counts > max(0.0, max_bad_frac) * seq_len



    zero_rows = (np.abs(X_arr).sum(axis=1) <= 1e-8).astype(np.float32)

    zero_counts = np.convolve(zero_rows, np.ones(seq_len, dtype=np.float32), mode="valid")

    out["zero_feature_window"] = zero_counts > max_zero_frac * seq_len



    for name, values in (row_reason_masks or {}).items():

        try:

            arr = np.asarray(values, dtype=bool)

            if len(arr) != len(X_arr):

                continue

            reason_counts = np.convolve(arr.astype(np.float32), np.ones(seq_len, dtype=np.float32), mode="valid")

            out[str(name)] = reason_counts > 0

        except Exception:

            continue

    return out





class _ChunkResult(tuple):
    """Tuple-compatible chunk result with legacy ``[-1]`` n_features access."""

    def __new__(
        cls,
        X_seq,
        y_seq,
        diff_seq,
        pq_seq,
        y_cls_seq,
        close_seq,
        atr_seq,
        spread_seq,
        n_features,
        time_idx,
    ):
        return super().__new__(
            cls,
            (
                X_seq,
                y_seq,
                diff_seq,
                pq_seq,
                y_cls_seq,
                close_seq,
                atr_seq,
                spread_seq,
                n_features,
                time_idx,
            ),
        )

    def __getitem__(self, key):
        if key == -1:
            return tuple.__getitem__(self, 8)
        return tuple.__getitem__(self, key)


def _chunk_result(
    X_seq,
    y_seq,
    diff_seq,
    pq_seq,
    y_cls_seq,
    close_seq,
    atr_seq,
    spread_seq,
    n_features,
    time_idx,
) -> _ChunkResult:
    return _ChunkResult(
        X_seq,
        y_seq,
        diff_seq,
        pq_seq,
        y_cls_seq,
        close_seq,
        atr_seq,
        spread_seq,
        n_features,
        time_idx,
    )


def _build_chunk(
    ticks_chunk: "pd.DataFrame",
    fe:          FeatureEngineer,
    scaler:      StandardScaler,
    seq_len:     int,
    chunk_idx:   int,
    win_start:   str = None,
    label_method: str = "rl_reward",
    target_col: str = "label",
    execution_delay_bars: int = 1,
    bar_freq: str = "1min",
    lookahead_bars: Optional[int] = None,
    profit_target_atr: Optional[float] = None,
    stop_loss_atr: Optional[float] = None,
    cross_asset: Optional[Dict[str, "pd.Series"]] = None,
    sentiment_pipe: Optional["SentimentPipeline"] = None,
    pair: str = "EURUSD",
    historical_news_mode: str = "calendar",
    historical_news_file: Optional[str] = None,
    economic_calendar_file: Optional[str] = None,
    cot_data: Optional["pl.DataFrame"] = None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]":

    """
    Process one tick chunk -> (X, y, diff, pq, y_cls, close, atr, spread, n_features, time_idx).

    close/atr/spread are per-sequence values at the label bar (last timestep of each window):
      - close: absolute FX quote (mid_close or close from bars)
      - atr: ATR in price units (matches ForexTradingEnv SL/TP)
      - spread: bid-ask width in price units (spread_pips * pip_size)
    """
    import pandas as pd
    import polars as pl

    _empty = (
        np.array([]), np.array([]), np.array([], dtype=np.uint8),
        np.array([], dtype=np.float32), np.array([], dtype=np.float32),
        np.array([], dtype=np.float32), np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
    )
    _empty_time = np.array([], dtype="datetime64[ns]")

    # Graceful exit for empty/bad chunks (e.g., all vendor hour files missing/empty)
    if ticks_chunk is None or len(ticks_chunk) == 0:
        _update_pair_readiness_raw(pair, ticks_chunk)
        return _chunk_result(*_empty, 0, _empty_time)
    _update_pair_readiness_raw(pair, ticks_chunk)
    if "timestamp_utc" not in ticks_chunk.columns:
        return _chunk_result(*_empty, 0, _empty_time)

    pipeline = ForexDataPipeline(bar_freq=str(bar_freq or "1min"), session_filter=False,
                                  apply_frac_diff=False)
    bars_polars = pipeline.run(ticks_chunk)
    if len(bars_polars) < seq_len + 20:
        return _chunk_result(*_empty, 0, _empty_time)
        
    bars_pd = bars_polars.to_pandas()
    bars_pd.set_index("timestamp_utc", inplace=True)
    bars_pd.index.name = "timestamp"

    news_bundle = load_historical_news_bundle(
        bars_pd.index[0],
        bars_pd.index[-1],
        pair,
        mode=historical_news_mode,
        news_file=historical_news_file,
        calendar_file=economic_calendar_file,
    )
    sent_pl = news_bundle.sentiment if news_bundle.sentiment is not None else None
    sentiment_input = None
    if sent_pl is not None and sent_pl.height > 0 and sentiment_pipe is not None:
        try:
            import pandas as pd
            news_pdf = news_bundle.news_events_df
            if news_pdf is not None and news_pdf.height > 0:
                news_df = news_pdf.to_pandas()
                if "headline" in news_df.columns:
                    news_df = news_df.rename(columns={"headline": "text", "summary": "description"})
                scores = sentiment_pipe.predict(news_df)
                sentiment_input = pd.Series(scores, index=news_df.index)
        except Exception:
            pass

    # Convert to Polars
    # When resetting index in pandas, if the index is named 'timestamp', it becomes a 'timestamp' column.
    bars_df = bars_pd.reset_index()
    if "index" in bars_df.columns:
        bars_df = bars_df.rename(columns={"index": "timestamp_utc"})
    if "timestamp" in bars_df.columns:
        bars_df = bars_df.rename(columns={"timestamp": "timestamp_utc"})
    bars = pl.from_pandas(bars_df)

    cross_pl = None
    if cross_asset:
        cross_pl = {}
        for k, v in cross_asset.items():
            df_v = v.to_frame(name="value").reset_index()
            if "index" in df_v.columns: df_v = df_v.rename(columns={"index": "timestamp_utc"})
            if "timestamp" in df_v.columns: df_v = df_v.rename(columns={"timestamp": "timestamp_utc"})
            cross_pl[k] = pl.from_pandas(df_v)

    if sentiment_input is not None:
        df_sent = sentiment_input.to_frame(name="sentiment").reset_index()
        if "index" in df_sent.columns:
            df_sent = df_sent.rename(columns={"index": "timestamp_utc"})
        if "timestamp" in df_sent.columns:
            df_sent = df_sent.rename(columns={"timestamp": "timestamp_utc"})
        sent_pl = pl.from_pandas(df_sent)

    eco_act_pl = news_bundle.eco_actual
    eco_fc_pl = news_bundle.eco_forecast
    eco_prior_pl = getattr(news_bundle, "eco_prior", None)
    news_events = news_bundle.news_events or None
    art_counts_pl = news_bundle.article_counts
    news_cats_pl = news_bundle.category_flags

    F = fe.build(
        bars,
        cross_asset=cross_pl,
        sentiment=sent_pl,
        eco_act=eco_act_pl,
        eco_fc=eco_fc_pl,
        eco_prior=eco_prior_pl,
        art_counts=art_counts_pl,
        finbert_embs=news_bundle.finbert_embeddings,
        news_events=news_events,
        cot_data=cot_data,
        pair=pair,
        news_cats=news_cats_pl,
    )
    
    # Cast back to Pandas to retain downstream compatibility with labels
    F_pd = F.to_pandas().set_index("timestamp_utc")
    
    if win_start:
        import pandas as pd
        ws_dt = pd.to_datetime(win_start, utc=True)
        F_pd = F_pd[F_pd.index >= ws_dt]
        bars_pd = bars_pd[bars_pd.index >= ws_dt]
        
    feats = F_pd
    if "news_ok" in feats.columns:
        news_no_trade = (1.0 - feats["news_ok"].astype(float)).clip(0.0, 1.0)
        if "no_trade_score" in feats.columns:
            feats["no_trade_score"] = np.maximum(feats["no_trade_score"].astype(float), news_no_trade)
        else:
            feats["no_trade_score"] = news_no_trade
    if len(feats) < seq_len + 10:
        return _chunk_result(*_empty, 0, _empty_time)

    if label_method == "triple_barrier":
        labels = compute_triple_barrier_labels(
            bars_pd,
            feats,
            vertical_bars=int(lookahead_bars or LABELING["lookahead_bars"]),
            profit_atr_mult=float(profit_target_atr or LABELING["profit_target_atr"]),
            stop_atr_mult=float(stop_loss_atr or LABELING["stop_loss_atr"]),
            pip_size=LABELING["pip_size"],
            execution_delay_bars=int(execution_delay_bars),
        )
    else:
        # Use regime-conditional labeling: barrier widths, session costs, bad-win
        # penalties, no-trade zones, and multi-target outputs (path_quality,
        # confidence_target) are all derived from feature columns when present.
        labels = compute_rl_reward_labels_regime(
            bars_pd,
            feats,
            lookahead_bars=int(lookahead_bars or LABELING["lookahead_bars"]),
            pip_size=LABELING["pip_size"],
            session_col="session"       if "session"       in feats.columns else None,
            regime_col="regime"         if "regime"        in feats.columns else None,
            no_trade_col="no_trade_score" if "no_trade_score" in feats.columns else None,
        latency_col="latency_ms"    if "latency_ms"    in feats.columns else None,
            execution_delay_bars=int(execution_delay_bars),
        )
    row_quality, row_drop_reasons, label_filter_counts, row_reason_masks = _compute_row_quality_mask(bars_pd, feats, labels)

    X, y, sidecar = align_labels_with_features(labels, feats, target_col=target_col)
    
    # Track stats for Pair Readiness Gate
    stats = _pair_readiness_entry(pair)
    stats["n_features"] = int(X.shape[1]) if hasattr(X, "shape") and len(X.shape) > 1 else int(stats.get("n_features", 0) or 0)
    _update_pair_readiness_processed(
        pair,
        bars_count=len(bars_pd),
        feature_frame=X,
        labels=labels,
    )
    _bump_reason_counts(stats, "dropped_bars_by_reason", row_drop_reasons)

    _bump_reason_counts(stats, "label_filter_counts", label_filter_counts)

        
    if len(X) < seq_len:
        return _chunk_result(*_empty, 0, _empty_time)
        
    row_reason_values = {

        name: mask.reindex(X.index).fillna(False).values.astype(bool)

        for name, mask in (row_reason_masks or {}).items()

    }

    row_quality = row_quality.reindex(X.index).fillna(False).values.astype(bool)
    bad_rows = int((~row_quality).sum())
    if bad_rows:
        stats["dropped_bars"] += bad_rows
        print(f"[DataQuality] Chunk {chunk_idx}: flagged {bad_rows:,}/{len(row_quality):,} low-quality row(s)")

    X_arr = X.values
    X_arr = sanitize_array(X_arr, context="chunk features before scaling")

    global _FIRST_CHUNK_COLS
    if '_FIRST_CHUNK_COLS' not in globals():
        _FIRST_CHUNK_COLS = list(X.columns)
    else:
        if list(X.columns) != list(_FIRST_CHUNK_COLS):

            missing = set(_FIRST_CHUNK_COLS) - set(X.columns)
            extra = set(X.columns) - set(_FIRST_CHUNK_COLS)
            raise ValueError(
                "Feature schema/order changed between chunks. "
                f"Missing={sorted(missing)}, Extra={sorted(extra)}"
            )

    # Sanitize: replace ┬▒inf / NaN (can arise from log-return or cross-asset
    # derived features) with 0 so StandardScaler.partial_fit never sees inf.
    n_feat = X_arr.shape[1]

    if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != n_feat:
        import logging
        logging.getLogger('train_gpu').debug(f'Mismatch! Scaler expects {scaler.n_features_in_}, but got {n_feat}.')
        try:
            if hasattr(scaler, 'feature_names_in_'):
                logging.getLogger('train_gpu').debug(f'Scaler expected columns: {list(scaler.feature_names_in_)}')
                missing = set(scaler.feature_names_in_) - set(X.columns)
                extra = set(X.columns) - set(scaler.feature_names_in_)
                logging.getLogger('train_gpu').debug(f'Missing columns: {missing}')
                logging.getLogger('train_gpu').debug(f'Extra columns: {extra}')
        except Exception:
            pass
        raise ValueError(f'Feature count mismatch: {n_feat} vs {scaler.n_features_in_}')

    _existing_feature_names = getattr(scaler, "feature_names_in_", None)

    if _existing_feature_names is not None:

        try:

            delattr(scaler, "feature_names_in_")

        except Exception:

            pass

    scaler.partial_fit(X_arr)
    _set_scaler_feature_names(scaler, X.columns)

    X_arr = sanitize_array(
        scaler.transform(X_arr),
        context="chunk features after scaling",
    )
    X_arr = np.asarray(X_arr, dtype=np.float32)

    # Path-quality gating: bars where the winning trade had a noisy/meandering path
    # (path_quality < 0.2) are relabelled as hold (0) to suppress gradient noise.
    y_arr = y.astype(np.float32)
    y_arr = sanitize_array(y_arr, context="chunk labels")
    
    pq_arr = sidecar["path_quality"].values if "path_quality" in sidecar.columns else np.ones(len(y_arr), dtype=np.float32)
    if "no_trade_score" in feats.columns:
        no_trade_arr = feats.reindex(X.index)["no_trade_score"].fillna(0.0).values.astype(np.float32)
        pq_arr = pq_arr * np.clip(1.0 - no_trade_arr, 0.0, 1.0)

    # Build sliding window sequences
    # Window i uses rows [i, i+seq_len), label is the bar at i+seq_len-1 (last bar).
    # sliding_window_view produces (N - seq_len + 1) windows;
    # labels start at index seq_len-1 so there are (N - seq_len + 1) of them too.
    n_seq = len(X_arr) - seq_len + 1
    if n_seq <= 0:
        return _chunk_result(*_empty, n_feat, _empty_time)

    seq_ok = _sequence_quality_mask(X_arr, row_quality, seq_len)
    seq_reason_masks = _sequence_quality_reason_masks(X_arr, row_quality, row_reason_values, seq_len)


    X_seq  = np.lib.stride_tricks.sliding_window_view(
        X_arr, (seq_len, n_feat)
    ).squeeze(1)                         # (n_seq, seq_len, n_feat)
    X_seq  = np.ascontiguousarray(X_seq, dtype=np.float32)
    y_seq  = np.asarray(y_arr[seq_len - 1:], dtype=np.float32)   # label = last bar of each window
    if "label" in sidecar.columns:
        lbl_arr = sidecar["label"].reindex(X.index).values.astype(np.float32)
        y_cls_seq = np.asarray(lbl_arr[seq_len - 1:], dtype=np.float32)
    else:
        y_cls_seq = np.sign(y_seq).astype(np.float32)
    pq_seq = pq_arr[seq_len - 1:]        # path quality aligned with y_seq
    pq_seq = np.asarray(pq_seq, dtype=np.float32)
    close_seq, atr_seq, spread_seq = _market_bar_arrays_from_feats(feats, X.index, fe, seq_len)
    time_idx = X.index[seq_len - 1:]

    # B: per-sample difficulty scores aligned with X (inner-joined features)
    try:
        diff_seq = _compute_difficulty_scores(feats.reindex(X.index), seq_len)
    except Exception as e:
        _log_warn(f"[DiffCurriculum] Difficulty scoring failed ({e}); defaulting all samples to easy (stage 0).")
        diff_seq = np.zeros(len(y_seq), dtype=np.uint8)

    if len(diff_seq) != len(y_seq):
        _log_warn(
            f"[DiffCurriculum] Difficulty length mismatch ({len(diff_seq)} vs {len(y_seq)}); "
            "defaulting all samples to easy (stage 0)."
        )
        diff_seq = np.zeros(len(y_seq), dtype=np.uint8)

    label_ok = np.isfinite(y_seq)
    if target_col == "label":
        label_ok &= np.isin(np.round(y_seq), [-1.0, 0.0, 1.0])
    cls_ok = np.isfinite(y_cls_seq) & np.isin(np.round(y_cls_seq), [-1.0, 0.0, 1.0])
    pq_ok = np.isfinite(pq_seq) & (pq_seq >= 0.0) & (pq_seq <= 1.0)
    target_reason = "invalid_reward_label" if target_col != "label" else "label_filter"

    if target_reason in seq_reason_masks:

        seq_reason_masks[target_reason] = np.asarray(seq_reason_masks[target_reason], dtype=bool) | ~label_ok

    else:

        seq_reason_masks[target_reason] = ~label_ok

    seq_reason_masks["invalid_direction_label"] = ~cls_ok

    seq_reason_masks["path_quality_filter"] = ~pq_ok

    _bump_reason_counts(stats, "label_filter_counts", {

        "invalid_direction_label": int((~cls_ok).sum()),

        "invalid_path_quality": int((~pq_ok).sum()),

    })

    keep = seq_ok & label_ok & cls_ok & pq_ok
    dropped = int((~keep).sum())
    if dropped:
        print(f"[DataQuality] Chunk {chunk_idx}: dropped {dropped:,}/{len(keep):,} low-quality sequence(s)")
        stats["dropped_sequences"] += dropped
        _bump_reason_counts(

            stats,

            "dropped_sequences_by_reason",

            {name: int((np.asarray(mask, dtype=bool) & ~keep).sum()) for name, mask in seq_reason_masks.items()},

        )

    if not keep.any():
        return _chunk_result(*_empty, n_feat, _empty_time)
    X_seq = X_seq[keep]
    y_seq = y_seq[keep]
    diff_seq = diff_seq[keep]
    pq_seq = pq_seq[keep]
    y_cls_seq = y_cls_seq[keep]
    close_seq = close_seq[keep]
    atr_seq = atr_seq[keep]
    spread_seq = spread_seq[keep]
    time_idx = time_idx[keep]
    stats["seq_count"] += int(len(y_seq))
    stats["difficulty_counts"] = stats.get("difficulty_counts", {"0": 0, "1": 0, "2": 0})
    _update_pair_readiness_processed(
        pair,
        dropped_sequences=0,
        diff_seq=diff_seq,
        spread_seq=spread_seq,
        atr_seq=atr_seq,
    )

    return _chunk_result(
        X_seq, y_seq, diff_seq, pq_seq, y_cls_seq,
        close_seq, atr_seq, spread_seq, n_feat, time_idx,
    )


def _scaler_npz_path_pair(cache_path: Path, pair: str) -> Path:
    """Per-pair scaler sidecar: dataset_EURUSD-GBPUSD_..._scaler_EURUSD.npz"""
    return Path(_base_path(str(cache_path)) + f"_scaler_{pair}")


def _build_multipair_chunk(
    pair_ticks:   dict,
    fe:           "FeatureEngineer",
    scalers:      dict,
    seq_len:      int,
    chunk_idx:    int,
    win_start:    str = None,
    label_method: str = "rl_reward",
    target_col:   str = "label",
    execution_delay_bars: int = 1,
    bar_freq: str = "1min",
    lookahead_bars: Optional[int] = None,
    profit_target_atr: Optional[float] = None,
    stop_loss_atr: Optional[float] = None,
    cross_asset: Optional[Dict[str, "pd.Series"]] = None,
    sentiment_pipe: Optional["SentimentPipeline"] = None,
    historical_news_mode: str = "calendar",
    historical_news_file: Optional[str] = None,
    economic_calendar_file: Optional[str] = None,
    cot_data: Optional["pl.DataFrame"] = None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]":
    """
    Process raw ticks for P pairs into joint sequences.

    pair_ticks : {pair_name: pd.DataFrame}
    scalers    : {pair_name: StandardScaler}  ΓÇö one per pair, fitted in-place

    Returns
    -------
    X : (N, T, P * F_per_pair)   ΓÇö pairs concatenated on feature axis
    y : (N,)                      ΓÇö mean label across pairs
    y_cls : (N,)                  ΓÇö consensus direction {-1,0,+1} across pairs
    pq : (N,)                     ΓÇö mean path-quality across pairs
    diff : (N,) uint8             ΓÇö max difficulty across pairs (curriculum)
    close/atr/spread : (N,) from the first pair with valid sequences (RL market path)
    n_features_total : int
    """
    pair_Xs: dict = {}
    pair_ys: dict = {}
    pair_ycls: dict = {}
    pair_pqs: dict = {}
    pair_diffs: dict = {}
    pair_times: dict = {}
    market_close = market_atr = market_spread = None

    for pair, ticks in pair_ticks.items():
        chunk_result = _build_chunk(
            ticks, fe, scalers[pair], seq_len, chunk_idx, win_start=win_start, label_method=label_method,
            target_col=target_col,
            execution_delay_bars=execution_delay_bars,
            bar_freq=bar_freq,
            lookahead_bars=lookahead_bars,
            profit_target_atr=profit_target_atr,
            stop_loss_atr=stop_loss_atr,
            cross_asset=cross_asset,
            sentiment_pipe=sentiment_pipe,
            pair=pair,
            historical_news_mode=historical_news_mode,
            historical_news_file=historical_news_file,
            economic_calendar_file=economic_calendar_file,
            cot_data=cot_data,
        )
        if len(chunk_result) == 10:
            (
                X_seq, y_seq, diff_seq, pq_seq, y_cls_seq,
                close_seq, atr_seq, spread_seq, _, time_idx
            ) = chunk_result
        elif len(chunk_result) == 9:
            (
                X_seq, y_seq, diff_seq, pq_seq, y_cls_seq,
                close_seq, atr_seq, spread_seq, _
            ) = chunk_result
            # Older tests/mocks predate timestamp-aware alignment. Preserve their
            # positional behavior while real _build_chunk paths return time_idx.
            time_idx = np.arange(len(y_seq))
        else:
            raise ValueError(
                f"_build_chunk returned {len(chunk_result)} values; expected 9 or 10"
            )
        if X_seq.size == 0:
            continue
        pair_Xs[pair] = X_seq   # (N, T, F)
        pair_ys[pair] = y_seq   # (N,)
        pair_ycls[pair] = y_cls_seq
        pair_pqs[pair] = pq_seq
        pair_diffs[pair] = diff_seq
        pair_times[pair] = time_idx
        if market_close is None:
            market_close, market_atr, market_spread = close_seq, atr_seq, spread_seq

    _empty8 = (
        np.array([]), np.array([]), np.array([], dtype=np.float32),
        np.array([], dtype=np.float32), np.array([], dtype=np.uint8),
        np.array([], dtype=np.float32), np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
    )
    if not pair_Xs:
        return *_empty8, 0

    missing = [p for p in pair_ticks.keys() if p not in pair_Xs]
    if missing:
        print(f"Warning: Required pair(s) produced no usable sequences: {missing}. Skipping chunk.")
        return *_empty8, 0

    # PAIR READINESS GATE
    print("\n[Pair Readiness]")
    gate_failed = False
    global _PAIR_READINESS_STATS
    if '_PAIR_READINESS_STATS' in globals():
        for p in pair_ticks.keys():
            if p not in _PAIR_READINESS_STATS:
                continue
            stats = _PAIR_READINESS_STATS[p]
            n_features = pair_Xs[p].shape[2] if p in pair_Xs else 120
            seq_len_val = pair_Xs[p].shape[1] if p in pair_Xs else 60
            total_values = max(1, stats.get("seq_count", 1) * seq_len_val * n_features)
            nan_pct = (stats.get("nan_count", 0) / total_values) * 100
            if stats.get("seq_count", 0) == 0 or nan_pct > 1.0:
                status = "FAIL"
                gate_failed = True
                stats.setdefault("reasons", []).append("chunk_pair_readiness_failed")
            elif nan_pct > 0.0:
                status = "WARN"
            else:
                status = "PASS"
                
            print(f"  {p} {status}  seq={stats.get('seq_count',0):,} dropped={stats.get('dropped_bars',0):,} nan_pct={nan_pct:.2f}%")
            
    if gate_failed:
        print("[Pair Readiness] WARN: chunk-level gate failure recorded; final JSON report will fail the build.")

    # Timestamp inner join. Build explicit timestamp -> row index maps instead
    # of boolean masks so duplicate or out-of-order timestamps cannot leave
    # pairs with different row counts or mismatched sample order.
    def _time_key(value):
        if isinstance(value, np.datetime64):
            return ("dt", int(value.astype("datetime64[ns]").astype(np.int64)))
        if hasattr(value, "to_datetime64"):
            dt_value = value.to_datetime64()
            return ("dt", int(dt_value.astype("datetime64[ns]").astype(np.int64)))
        if hasattr(value, "value") and value.__class__.__name__ == "Timestamp":
            return ("dt", int(value.value))
        if isinstance(value, np.generic):
            value = value.item()
        return ("raw", value)

    def _time_key_json(key):
        try:
            kind, value = key
            if kind == "dt":
                import pandas as pd
                return pd.Timestamp(int(value), unit="ns", tz="UTC").isoformat()
        except Exception:
            pass
        return _json_scalar(key)

    time_maps = {}
    common_keys = None
    for p, times in pair_times.items():
        idx_by_key = {}
        for i, t in enumerate(np.asarray(times)):
            idx_by_key.setdefault(_time_key(t), i)
        time_maps[p] = idx_by_key
        keys = set(idx_by_key)
        common_keys = keys if common_keys is None else common_keys.intersection(keys)

    common_keys = sorted(common_keys or [])
    if len(common_keys) == 0:
        globals()["_PAIR_ALIGNMENT_STATS"] = {
            "status": "fail",
            "reason": "no_common_timestamps",
            "input_sequence_counts": {p: int(len(pair_times.get(p, []))) for p in pair_ticks},
        }
        raise RuntimeError("No common timestamps found across pairs")

    _sample = next(iter(pair_Xs.values()))
    n_feat_per_pair = _sample.shape[2]

    y_list: list = []
    ycls_list: list = []
    pq_list: list = []
    diff_list: list = []
    pair_indices: dict = {}
    for pair in pair_ticks:
        idx = np.asarray([time_maps[pair][k] for k in common_keys], dtype=np.int64)
        pair_indices[pair] = idx
        y_list.append(pair_ys[pair][idx])
        ycls_list.append(pair_ycls[pair][idx])
        pq_list.append(pair_pqs[pair][idx])
        diff_list.append(pair_diffs[pair][idx])

    first_pair = list(pair_ticks.keys())[0]
    market_idx = np.asarray([time_maps[first_pair][k] for k in common_keys], dtype=np.int64)
    market_close = market_close[market_idx]
    market_atr = market_atr[market_idx]
    market_spread = market_spread[market_idx]

    expected_rows = len(common_keys)
    row_counts = {pair: int(len(pair_indices[pair])) for pair in pair_ticks}
    input_counts = {p: int(len(pair_times[p])) for p in pair_ticks}
    dropped_by_inner_join = {p: int(max(0, input_counts[p] - expected_rows)) for p in pair_ticks}
    diff_vals, diff_counts = np.unique(np.max(np.stack(diff_list, axis=1), axis=1).astype(np.uint8), return_counts=True)
    globals()["_PAIR_ALIGNMENT_STATS"] = {
        "status": "pass",
        "pair_align": "inner",
        "common_sequence_count": int(expected_rows),
        "timestamp_start": _time_key_json(common_keys[0]) if common_keys else None,
        "timestamp_end": _time_key_json(common_keys[-1]) if common_keys else None,
        "input_sequence_counts": input_counts,
        "dropped_by_inner_join": dropped_by_inner_join,
        "drop_pct_by_pair": {
            p: round(100.0 * dropped_by_inner_join[p] / max(1, input_counts[p]), 6)
            for p in pair_ticks
        },
        "difficulty_counts_joint": {str(int(v)): int(c) for v, c in zip(diff_vals, diff_counts)},
    }
    if any(n != expected_rows for n in row_counts.values()):
        globals()["_PAIR_ALIGNMENT_STATS"]["status"] = "fail"
        globals()["_PAIR_ALIGNMENT_STATS"]["reason"] = "mismatched_rows_after_alignment"
        raise RuntimeError(
            f"Pair timestamp alignment produced mismatched rows: "
            f"expected={expected_rows}, per_pair={row_counts}"
        )

    # Build the aligned tensor directly instead of materializing per-pair
    # fancy-indexed copies and then concatenating them.  On 3-pair/full-feature
    # Windows runs this avoids multiple extra ~1 GiB peak allocations.
    pair_order = list(pair_ticks.keys())
    n_total_features = n_feat_per_pair * len(pair_order)
    try:
        X_multi = np.empty(
            (expected_rows, _sample.shape[1], n_total_features),
            dtype=np.float32,
        )
    except MemoryError as exc:
        need_gib = (
            expected_rows * _sample.shape[1] * n_total_features * np.dtype(np.float32).itemsize
        ) / (1024 ** 3)
        raise MemoryError(
            f"Unable to allocate aligned multi-pair tensor "
            f"({expected_rows:,} x {_sample.shape[1]} x {n_total_features}; "
            f"~{need_gib:.2f} GiB). Reduce real_data_window_days, seq_len, "
            "number of pairs, or disable high-cardinality feature groups."
        ) from exc

    row_batch = max(1, min(expected_rows, 256))
    for pair_pos, pair in enumerate(pair_order):
        src = pair_Xs[pair]
        idx = pair_indices[pair]
        feat_start = pair_pos * n_feat_per_pair
        feat_end = feat_start + n_feat_per_pair
        for row_start in range(0, expected_rows, row_batch):
            row_end = min(expected_rows, row_start + row_batch)
            X_multi[row_start:row_end, :, feat_start:feat_end] = src[idx[row_start:row_end]]
        pair_Xs[pair] = None
        gc.collect()

    y_multi = np.mean(np.stack(y_list, axis=1), axis=1).astype(np.float32)   # (N,)
    cls_mean = np.mean(np.stack(ycls_list, axis=1), axis=1)
    y_cls_multi = np.where(
        np.abs(cls_mean) < 0.33, 0.0, np.sign(cls_mean),
    ).astype(np.float32)
    pq_multi = np.mean(np.stack(pq_list, axis=1), axis=1).astype(np.float32)
    diff_multi = np.max(np.stack(diff_list, axis=1), axis=1).astype(np.uint8)
    n_min = X_multi.shape[0]
    return (
        X_multi, y_multi, y_cls_multi, pq_multi, diff_multi,
        market_close[:n_min], market_atr[:n_min], market_spread[:n_min],
        X_multi.shape[2],
    )


def _merge_scalers(scaler_list: "List[StandardScaler]") -> "StandardScaler":
    """Merge independently fitted StandardScalers using the parallel merge formula.

    Each scaler must have been fitted via partial_fit / fit so that
    ``n_samples_seen_``, ``mean_``, and ``var_`` are populated.
    Returns a new StandardScaler with combined statistics.
    """
    from sklearn.preprocessing import StandardScaler
    combined = StandardScaler()
    if not scaler_list:
        return combined
    valid = [s for s in scaler_list if hasattr(s, "n_samples_seen_") and s.n_samples_seen_ is not None]
    if not valid:
        return scaler_list[0] if scaler_list else combined
    if len(valid) == 1:
        return valid[0]

    total_n = sum(int(np.atleast_1d(s.n_samples_seen_)[0]) for s in valid)
    if total_n == 0:
        return valid[0]

    n_features = len(valid[0].mean_)
    combined_mean = np.zeros(n_features, dtype=np.float64)
    for s in valid:
        n = int(np.atleast_1d(s.n_samples_seen_)[0])
        combined_mean += n * s.mean_
    combined_mean /= total_n

    combined_var = np.zeros(n_features, dtype=np.float64)
    for s in valid:
        n = int(np.atleast_1d(s.n_samples_seen_)[0])
        combined_var += n * (s.var_ + (s.mean_ - combined_mean) ** 2)
    combined_var /= total_n

    combined.mean_ = combined_mean
    combined.var_ = combined_var
    combined.scale_ = np.sqrt(combined_var)
    combined.scale_[combined.scale_ == 0] = 1.0
    combined.n_samples_seen_ = np.full(n_features, total_n, dtype=np.int64)
    combined.n_features_in_ = n_features
    if hasattr(valid[0], "feature_names_in_"):
        combined.feature_names_in_ = valid[0].feature_names_in_
    return combined


def _parallel_window_worker(worker_args: dict):
    """Top-level picklable function for ProcessPoolExecutor window processing.

    Each worker creates its own FeatureEngineer, ForexDataManager,
    SentimentPipeline, and StandardScalers so that no shared mutable state
    crosses the process boundary.

    Returns
    -------
    dict with keys: window_idx, X, y, y_cls, pq, diff, close, atr, spread,
                    n_feat, scalers (dict of fitted StandardScaler per pair),
                    or error string on failure.
    """
    import traceback
    try:
        win_start = worker_args["win_start"]
        win_end = worker_args["win_end"]
        window_idx = worker_args["window_idx"]
        pairs = worker_args["pairs"]
        data_source = worker_args["data_source"]
        full_day_data = worker_args["full_day_data"]
        seq_len = worker_args["seq_len"]
        label_method = worker_args["label_method"]
        target_col = worker_args["target_col"]
        execution_delay_bars = worker_args["execution_delay_bars"]
        bar_freq = worker_args["bar_freq"]
        lookahead_bars = worker_args["lookahead_bars"]
        profit_target_atr = worker_args["profit_target_atr"]
        stop_loss_atr = worker_args["stop_loss_atr"]
        cross_asset = worker_args.get("cross_asset")
        sentiment_mode = worker_args["sentiment_mode"]
        historical_news_mode = worker_args["historical_news_mode"]
        historical_news_file = worker_args.get("historical_news_file")
        economic_calendar_file = worker_args.get("economic_calendar_file")
        cot_data_path = worker_args.get("cot_data_path")

        from data.sources import ForexDataManager
        from features.feature_engineering_pl import FeatureEngineer
        from sklearn.preprocessing import StandardScaler
        from config.settings import FEATURES

        fe = FeatureEngineer(
            atr_window=FEATURES["atr_window"],
            ofi_window=FEATURES["ofi_window"],
            tar_window=FEATURES["trade_arrival_window"],
            rsi_period=FEATURES["rsi_period"],
            macd_fast=FEATURES["macd_fast"],
            macd_slow=FEATURES["macd_slow"],
            macd_signal=FEATURES["macd_signal"],
            bb_window=FEATURES["bollinger_window"],
            bb_std=FEATURES["bollinger_std"],
            lag_windows=FEATURES["lag_windows"],
        )
        scalers = {p: StandardScaler() for p in pairs}
        mgr = ForexDataManager(verbose=False)

        pair_ticks = {}
        for p in pairs:
            pair_ticks[p] = mgr.load(
                pair=p,
                source=data_source,
                start=win_start,
                end=win_end,
                session_only=not full_day_data,
            )

        sentiment_pipe = None
        if str(sentiment_mode).lower() != "off":
            try:
                from features.finbert_sentiment import SentimentPipeline
                pref = "finbert" if str(sentiment_mode).lower() == "finbert" else "vader"
                sentiment_pipe = SentimentPipeline(prefer_backend=pref, use_cache=True)
            except Exception:
                sentiment_pipe = None

        if (
            sentiment_pipe is not None
            and str(historical_news_mode).lower() == "full"
        ):
            try:
                from data.historical_news import collect_headlines_for_range as _chr
                _headlines = _chr(
                    win_start, win_end, pairs,
                    news_file=historical_news_file,
                    calendar_file=economic_calendar_file,
                )
                if _headlines:
                    sentiment_pipe.prefetch_headlines(_headlines)
                del _headlines
            except Exception:
                pass

        cot_data = None
        if cot_data_path:
            try:
                import polars as pl
                _cp = Path(cot_data_path)
                if _cp.exists():
                    cot_data = pl.read_parquet(_cp)
            except Exception:
                cot_data = None

        result = _build_multipair_chunk(
            pair_ticks, fe, scalers, seq_len, window_idx, label_method,
            target_col=target_col,
            execution_delay_bars=execution_delay_bars,
            bar_freq=bar_freq,
            lookahead_bars=lookahead_bars,
            profit_target_atr=profit_target_atr,
            stop_loss_atr=stop_loss_atr,
            cross_asset=cross_asset,
            sentiment_pipe=sentiment_pipe,
            historical_news_mode=historical_news_mode,
            historical_news_file=historical_news_file,
            economic_calendar_file=economic_calendar_file,
            cot_data=cot_data,
        )
        X_seq, y_seq, y_cls_seq, pq_seq, diff_seq, close_seq, atr_seq, spread_seq, n_feat = result

        return {
            "window_idx": window_idx,
            "X": X_seq, "y": y_seq, "y_cls": y_cls_seq,
            "pq": pq_seq, "diff": diff_seq,
            "close": close_seq, "atr": atr_seq, "spread": spread_seq,
            "n_feat": n_feat,
            "scalers": scalers,
        }
    except Exception:
        return {"window_idx": worker_args.get("window_idx", -1), "error": traceback.format_exc()}


def _build_multipair_dataset(
    args,
    pairs:      List[str],
    cache_path: Path,
    fe:         "FeatureEngineer",
) -> "tuple[str, int, int, StandardScaler]":
    """
    Multi-pair variant of build_dataset_chunked.
    Loads ticks for all pairs in parallel (dukascopy) or sequentially (other sources),
    builds joint (N, T, P*F) sequences, and writes them to Zarr (primary) / NPY (fallback).
    Returns (cache_path, n_samples, n_features, first_pair_scaler).
    """
    print(f"\n[MultiPair] {len(pairs)} pairs: {', '.join(pairs)}")
    print(f"            Source: {args.data_source} | {args.data_start} -> {args.data_end}")
    use_real_cross = (
        args.cross_asset_mode == "real"
        or (args.cross_asset_mode == "auto" and args.data_source != "synthetic")
    )
    cross_asset_source = _resolve_cross_asset_source(args)
    # Treat empty env var the same as "not set" so the default data/processed/cross_asset
    # path is always used when CROSS_ASSET_CACHE_DIR is unset or blank.
    cross_asset_cache_dir = (
        os.getenv("CROSS_ASSET_CACHE_DIR", "").strip()
        or str(Path(args.data_cache) / "cross_asset")
    )
    cross_asset = None
    sentiment_pipe = None
    import polars as pl
    cot_path = Path("data/raw/cot/cot_financials_cleaned.parquet")
    cot_data = pl.read_parquet(cot_path) if cot_path.exists() else None
    if cot_data is not None:
        print(f"[MultiPair] Loaded COT data ({len(cot_data)} rows)")
    if str(getattr(args, "sentiment_mode", "finbert")).lower() != "off":
        try:
            pref = "finbert" if str(args.sentiment_mode).lower() == "finbert" else "vader"
            sentiment_pipe = SentimentPipeline(prefer_backend=pref, use_cache=True)
            print(f"[Sentiment] mode={args.sentiment_mode} enabled")
        except Exception as e:
            print(f"[Sentiment] WARN: init failed ({e}) ΓÇö disabling sentiment features")
            sentiment_pipe = None
    if (
        sentiment_pipe is not None
        and str(getattr(args, "historical_news_mode", "calendar")).lower() == "full"
    ):
        try:
            _headlines = collect_headlines_for_range(
                args.data_start,
                args.data_end,
                pairs,
                news_file=getattr(args, "historical_news_file", None),
                calendar_file=getattr(args, "economic_calendar_file", None),
            )
            if _headlines:
                sentiment_pipe.prefetch_headlines(_headlines)
            del _headlines
        except Exception as _pf_err:
            print(f"[Sentiment] prefetch skipped ({_pf_err})")

    if use_real_cross:
        try:
            cross_asset = load_cross_asset_panel(
                start=args.data_start,
                end=args.data_end,
                cache_dir=cross_asset_cache_dir,
                source=cross_asset_source,
            )
            print(f"[CrossAsset] Loaded external assets: {len(cross_asset)} "
                  f"(source={cross_asset_source}, cache={cross_asset_cache_dir})")
        except Exception as e:
            print(f"[CrossAsset] WARN: external load failed ({e}) ΓÇö falling back to synthetic")
            cross_asset = None

    scalers       = {p: StandardScaler() for p in pairs}
    n_features    = 0
    total_samples = 0
    z_store       = None   # zarr (primary)
    pair_ticks: Optional[Dict[str, "pd.DataFrame"]] = None
    globals()["_PAIR_READINESS_STATS"] = {}
    globals()["_PAIR_ALIGNMENT_STATS"] = {}

    use_zarr = bool(ZARR)
    if ZARR and not use_zarr:
        print("[MultiPair] Zarr writes disabled on Windows; using NPY memmap cache.")
    if use_zarr:
        _compressor = _Blosc(cname="lz4", clevel=3, shuffle=_Blosc.BITSHUFFLE)
    else:
        # Out-of-core binary file states to prevent RAM OOM
        bin_state = {
            "opened": False, "x_f": None, "y_f": None, "ycls_f": None,
            "pq_f": None, "diff_f": None, "total": 0, "x_shape": None, "y_shape": None,
        }
    store_rl_sidecars = args.label_method == "rl_reward"

    def _sidecar_or_default(pq_seq, diff_seq, n_rows: int):
        pq = pq_seq if pq_seq is not None else np.ones(n_rows, dtype=np.float32)
        diff = diff_seq if diff_seq is not None else np.zeros(n_rows, dtype=np.uint8)
        return pq, diff

    def _append_chunk(X_seq, y_seq, y_cls_seq, pq_seq, diff_seq, close_seq, atr_seq, spread_seq):
        nonlocal z_store, total_samples
        n_rows = int(len(X_seq))
        pq_arr, diff_arr = _sidecar_or_default(pq_seq, diff_seq, n_rows)
        if use_zarr:
            if z_store is None:
                if getattr(args, "_resume_zarr", False):
                    z_store = _zarr_open_group(str(cache_path), mode="a")
                    existing_samples = int(z_store["X"].shape[0]) if "X" in z_store else 0
                    total_samples += existing_samples
                    z_store["X"].append(X_seq)
                    z_store["y"].append(y_seq)
                    z_store["y_cls"].append(y_cls_seq)
                    z_store["close"].append(close_seq)
                    z_store["atr"].append(atr_seq)
                    z_store["spread"].append(spread_seq)
                    if store_rl_sidecars:
                        if "pq" in z_store:
                            z_store["pq"].append(pq_arr)
                        if "diff" in z_store:
                            z_store["diff"].append(diff_arr)
                else:
                    if __import__('pathlib').Path(cache_path).exists():
                        for _retry in range(10):
                            __import__('shutil').rmtree(cache_path, ignore_errors=True)
                            if not __import__('pathlib').Path(cache_path).exists(): break
                            __import__('time').sleep(0.5)
                    z_store = _zarr_open_group(str(cache_path), mode="w")
                    # 2048-row chunks: ~30├ù fewer decompressions per epoch vs 64-row chunks.
                    c0 = (min(2048, len(X_seq)),) + X_seq.shape[1:]
                    _zarr_create(z_store, "X", shape=X_seq.shape, chunks=c0,
                                 dtype="float32", compressor=_compressor)
                    _zarr_create(z_store, "y", shape=y_seq.shape, chunks=(c0[0],),
                                 dtype="float32", compressor=_compressor)
                    _zarr_create(z_store, "y_cls", shape=y_cls_seq.shape, chunks=(c0[0],),
                                 dtype="float32", compressor=_compressor)
                    _zarr_create(z_store, "close", shape=close_seq.shape, chunks=(c0[0],),
                                 dtype="float32", compressor=_compressor)
                    _zarr_create(z_store, "atr", shape=atr_seq.shape, chunks=(c0[0],),
                                 dtype="float32", compressor=_compressor)
                    _zarr_create(z_store, "spread", shape=spread_seq.shape, chunks=(c0[0],),
                                 dtype="float32", compressor=_compressor)
                    if store_rl_sidecars:
                        _zarr_create(z_store, "pq", shape=pq_arr.shape, chunks=(c0[0],),
                                     dtype="float32", compressor=_compressor)
                        _zarr_create(z_store, "diff", shape=diff_arr.shape, chunks=(c0[0],),
                                     dtype="uint8", compressor=_compressor)
                    z_store["X"][:] = X_seq
                    z_store["y"][:] = y_seq
                    z_store["y_cls"][:] = y_cls_seq
                    z_store["close"][:] = close_seq
                    z_store["atr"][:] = atr_seq
                    z_store["spread"][:] = spread_seq
                    if store_rl_sidecars:
                        z_store["pq"][:] = pq_arr
                        z_store["diff"][:] = diff_arr
            else:
                z_store["X"].append(X_seq)
                z_store["y"].append(y_seq)
                z_store["y_cls"].append(y_cls_seq)
                z_store["close"].append(close_seq)
                z_store["atr"].append(atr_seq)
                z_store["spread"].append(spread_seq)
                if store_rl_sidecars:
                    if "pq" in z_store:
                        z_store["pq"].append(pq_arr)
                    if "diff" in z_store:
                        z_store["diff"].append(diff_arr)
        else:
            if not bin_state["opened"]:
                bin_state["x_f"] = open(str(_x_path(cache_path)) + ".bin", "wb")
                bin_state["y_f"] = open(str(_y_path(cache_path)) + ".bin", "wb")
                bin_state["ycls_f"] = open(_y_cls_path(cache_path).replace(".npy", ".bin"), "wb")
                bin_state["close_f"] = open(str(cache_path) + "_close.bin", "wb")
                bin_state["atr_f"] = open(str(cache_path) + "_atr.bin", "wb")
                bin_state["spread_f"] = open(str(cache_path) + "_spread.bin", "wb")
                if store_rl_sidecars:
                    bin_state["pq_f"] = open(str(cache_path) + "_pq.bin", "wb")
                    bin_state["diff_f"] = open(str(cache_path) + "_diff.bin", "wb")
                bin_state["x_shape"] = list(X_seq.shape)
                bin_state["y_shape"] = list(y_seq.shape)
                bin_state["opened"] = True

            X_seq.tofile(bin_state["x_f"])
            y_seq.tofile(bin_state["y_f"])
            bin_state["ycls_f"].write(y_cls_seq.tobytes())
            bin_state["close_f"].write(close_seq.tobytes())
            bin_state["atr_f"].write(atr_seq.tobytes())
            bin_state["spread_f"].write(spread_seq.tobytes())
            if store_rl_sidecars:
                bin_state["pq_f"].write(pq_arr.tobytes())
                bin_state["diff_f"].write(diff_arr.tobytes())
            bin_state["total"] += len(X_seq)

    # -- Load ticks ----------------------------------------------------------
    real_windows_handled = False
    if args.data_source != "synthetic":
        real_windows_handled = True
        _base_days = _real_data_window_days(args)
        window_days = _effective_window_days(args)
        date_windows = _iter_date_windows(args.data_start, args.data_end, window_days)
        _batch_n = max(1, int(getattr(args, "window_batch_days", 1) or 1))
        if _batch_n > 1:
            print(f"[MultiPair] Real-data windows: {len(date_windows)} x {window_days} day(s) "
                  f"(base {_base_days}d x {_batch_n} batch)")
        else:
            print(f"[MultiPair] Real-data windows: {len(date_windows)} x {window_days} day(s)")

        mgr = ForexDataManager(verbose=True)
        _build_workers = max(1, int(getattr(args, "dataset_build_workers", 1) or 1))

        resume_idx = -1
        if getattr(args, "_resume_zarr", False):
            try:
                import json
                with open(str(cache_path) + "_resume.json", "r") as f:
                    resume_idx = json.load(f).get("last_completed_window_idx", -1)
            except Exception:
                pass

        def _load_window_ticks(win_start, win_end):
            """Load ticks for all pairs in a single date window."""
            import pandas as pd
            ws_dt = pd.to_datetime(win_start, utc=True) - pd.Timedelta(days=14)
            ws_str = ws_dt.strftime("%Y-%m-%d")
            
            ticks = {}
            for p in pairs:
                ticks[p] = mgr.load(
                    pair         = p,
                    source       = args.data_source,
                    start        = ws_str,
                    end          = win_end,
                    session_only = not getattr(args, "full_day_data", False),
                )
            return ticks

        if _build_workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            print(f"[MultiPair] Parallel tick loading: {_build_workers} threads")

        _pending = list(enumerate(date_windows))
        _pending = [(idx, ws, we) for idx, (ws, we) in _pending if idx > resume_idx]

        _pw_workers = max(1, int(getattr(args, "parallel_window_workers", 1) or 1))

        if _pw_workers > 1:
            # ── PARALLEL window processing (multi-process) ──────────────
            from concurrent.futures import ProcessPoolExecutor, as_completed
            print(f"[MultiPair] Parallel window processing: {_pw_workers} processes")

            _cot_path_str = str(Path("data/raw/cot/cot_financials_cleaned.parquet"))
            _cot_path_exists = Path(_cot_path_str).exists()

            _worker_args_all = []
            for idx, ws, we in _pending:
                _worker_args_all.append({
                    "win_start": ws,
                    "win_end": we,
                    "window_idx": idx,
                    "pairs": list(pairs),
                    "data_source": args.data_source,
                    "full_day_data": bool(getattr(args, "full_day_data", False)),
                    "seq_len": args.seq_len,
                    "label_method": args.label_method,
                    "target_col": _cache_target_col(args),
                    "execution_delay_bars": int(getattr(args, "execution_delay_bars", 1)),
                    "bar_freq": str(getattr(args, "bar_freq", "1min")),
                    "lookahead_bars": int(getattr(args, "lookahead_bars", LABELING["lookahead_bars"])),
                    "profit_target_atr": float(getattr(args, "profit_target_atr", LABELING["profit_target_atr"])),
                    "stop_loss_atr": float(getattr(args, "stop_loss_atr", LABELING["stop_loss_atr"])),
                    "cross_asset": cross_asset,
                    "sentiment_mode": str(getattr(args, "sentiment_mode", "finbert")),
                    "historical_news_mode": str(getattr(args, "historical_news_mode", "calendar")),
                    "historical_news_file": getattr(args, "historical_news_file", None),
                    "economic_calendar_file": getattr(args, "economic_calendar_file", None),
                    "cot_data_path": _cot_path_str if _cot_path_exists else None,
                })

            _all_worker_scalers: Dict[str, list] = {p: [] for p in pairs}
            _batch_sz = max(_pw_workers, 2)
            _n_errors = 0

            for _b_start in range(0, len(_worker_args_all), _batch_sz):
                _batch = _worker_args_all[_b_start:_b_start + _batch_sz]
                _batch_results: Dict[int, dict] = {}

                with ProcessPoolExecutor(max_workers=_pw_workers) as _p_pool:
                    _futures = {
                        _p_pool.submit(_parallel_window_worker, wa): wa["window_idx"]
                        for wa in _batch
                    }
                    for _fut in as_completed(_futures):
                        _res = _fut.result()
                        _widx = _res["window_idx"]
                        if "error" in _res:
                            print(f"  [Window {_widx+1}/{len(date_windows)}] "
                                  f"FAILED:\n{_res['error']}")
                            _n_errors += 1
                            continue
                        _batch_results[_widx] = _res

                for _wa in _batch:
                    _widx = _wa["window_idx"]
                    if _widx not in _batch_results:
                        continue
                    _res = _batch_results[_widx]
                    _X = _res["X"]
                    if _X is None or _X.size == 0:
                        del _batch_results[_widx]
                        continue

                    n_features = _res["n_feat"]
                    total_samples += len(_X)
                    _append_chunk(
                        _res["X"], _res["y"], _res["y_cls"], _res["pq"],
                        _res["diff"], _res["close"], _res["atr"], _res["spread"],
                    )
                    print(f"  [Window {_widx+1}/{len(date_windows)}] "
                          f"{len(_X):,} joint sequences | {total_samples:,} total")

                    for _p in pairs:
                        if _p in _res.get("scalers", {}):
                            _all_worker_scalers[_p].append(_res["scalers"][_p])

                    try:
                        import json as _json_mod
                        with open(str(cache_path) + "_resume.json", "w") as _rf:
                            _json_mod.dump({"last_completed_window_idx": _widx}, _rf)
                    except Exception:
                        pass

                    del _batch_results[_widx]

                gc.collect()
                if _TRAIN_LOGGER:
                    _TRAIN_LOGGER.heartbeat()

            for _p in pairs:
                if _all_worker_scalers[_p]:
                    scalers[_p] = _merge_scalers(_all_worker_scalers[_p])

            if _n_errors:
                print(f"[MultiPair] WARNING: {_n_errors} window(s) failed during "
                      f"parallel processing")

        else:
            # ── SEQUENTIAL window processing (original behavior) ────────
            _pool = None
            if _build_workers > 1:
                from concurrent.futures import ThreadPoolExecutor
                print(f"[MultiPair] Parallel tick loading: {_build_workers} threads")

            if _build_workers > 1:
                _pool = ThreadPoolExecutor(max_workers=_build_workers)
                _prefetch = {}
                _look_ahead = min(_build_workers, 3)
                for i in range(min(_look_ahead, len(_pending))):
                    idx, ws, we = _pending[i]
                    _prefetch[idx] = _pool.submit(_load_window_ticks, ws, we)

            try:
                for _q_pos, (window_idx, win_start, win_end) in enumerate(_pending):
                    print(f"  [Window {window_idx+1}/{len(date_windows)}] "
                          f"{win_start} -> {win_end}")

                    if _pool is not None:
                        pair_ticks = _prefetch.pop(window_idx).result()
                        _next_q = _q_pos + _look_ahead
                        if _next_q < len(_pending):
                            _ni, _ns, _ne = _pending[_next_q]
                            _prefetch[_ni] = _pool.submit(
                                _load_window_ticks, _ns, _ne)
                    else:
                        pair_ticks = _load_window_ticks(win_start, win_end)

                    X_seq, y_seq, y_cls_seq, pq_seq, diff_seq, close_seq, atr_seq, spread_seq, n_feat = (
                        _build_multipair_chunk(
                        pair_ticks, fe, scalers, args.seq_len,
                        window_idx, win_start=win_start, label_method=args.label_method,
                        target_col=_cache_target_col(args),
                        execution_delay_bars=int(getattr(args, "execution_delay_bars", 1)),
                        bar_freq=str(getattr(args, "bar_freq", "1min")),
                        lookahead_bars=int(getattr(args, "lookahead_bars", LABELING["lookahead_bars"])),
                        profit_target_atr=float(getattr(args, "profit_target_atr", LABELING["profit_target_atr"])),
                        stop_loss_atr=float(getattr(args, "stop_loss_atr", LABELING["stop_loss_atr"])),
                        cross_asset=cross_asset,
                        sentiment_pipe=sentiment_pipe,
                        historical_news_mode=str(getattr(args, "historical_news_mode", "calendar")),
                        historical_news_file=getattr(args, "historical_news_file", None),
                        economic_calendar_file=getattr(args, "economic_calendar_file", None),
                        cot_data=cot_data,
                    ))
                    if X_seq is None or X_seq.size == 0:
                        del pair_ticks
                        gc.collect()
                        continue
                    n_features = n_feat
                    total_samples += len(X_seq)
                    _append_chunk(X_seq, y_seq, y_cls_seq, pq_seq, diff_seq,
                                  close_seq, atr_seq, spread_seq)
                    print(f"    {len(X_seq):,} joint sequences | "
                          f"{total_samples:,} total")
                    if _TRAIN_LOGGER:
                        _TRAIN_LOGGER.heartbeat()
                    del pair_ticks, X_seq, y_seq, y_cls_seq, pq_seq, diff_seq
                    del close_seq, atr_seq, spread_seq
                    try:
                        import json
                        with open(str(cache_path) + "_resume.json", "w") as f:
                            json.dump(
                                {"last_completed_window_idx": window_idx}, f)
                    except Exception:
                        pass
                    gc.collect()
            finally:
                if _pool is not None:
                    _pool.shutdown(wait=False)

    if args.data_source == "synthetic":
        n_remaining = args.n_ticks
        chunk_n     = 0
        while n_remaining > 0:
            chunk_n_ticks = min(args.chunk_size, n_remaining)
            pair_ticks = {p: generate_synthetic_tick_data(n_rows=chunk_n_ticks) for p in pairs}
            X_seq, y_seq, y_cls_seq, pq_seq, diff_seq, close_seq, atr_seq, spread_seq, n_feat = (
                _build_multipair_chunk(
                pair_ticks, fe, scalers, args.seq_len, chunk_n, args.label_method,
                target_col=_cache_target_col(args),

                execution_delay_bars=int(getattr(args, "execution_delay_bars", 1)),
                bar_freq=str(getattr(args, "bar_freq", "1min")),
                lookahead_bars=int(getattr(args, "lookahead_bars", LABELING["lookahead_bars"])),
                profit_target_atr=float(getattr(args, "profit_target_atr", LABELING["profit_target_atr"])),
                stop_loss_atr=float(getattr(args, "stop_loss_atr", LABELING["stop_loss_atr"])),
                cross_asset=cross_asset,
                sentiment_pipe=sentiment_pipe,
                historical_news_mode=str(getattr(args, "historical_news_mode", "calendar")),
                historical_news_file=getattr(args, "historical_news_file", None),
                economic_calendar_file=getattr(args, "economic_calendar_file", None),
                cot_data=cot_data,
            ))
            if X_seq is not None and X_seq.size > 0:
                n_features     = n_feat
                total_samples += len(X_seq)
                _append_chunk(X_seq, y_seq, y_cls_seq, pq_seq, diff_seq, close_seq, atr_seq, spread_seq)
                pct = min((args.n_ticks - n_remaining + chunk_n_ticks) / args.n_ticks * 100, 100)
                print(f"  Chunk {chunk_n+1} | {len(X_seq):,} seqs | {pct:.0f}%")
            n_remaining -= chunk_n_ticks
            chunk_n     += 1

    elif not real_windows_handled:
        # Real data: load all pairs at once then process
        mgr        = ForexDataManager(verbose=True)
        pair_ticks = {}
        for p in pairs:
            print(f"  Loading {p}...")
            pair_ticks[p] = mgr.load(
                pair         = p,
                source       = args.data_source,
                start        = args.data_start,
                end          = args.data_end,
                session_only = not getattr(args, "full_day_data", False),
            )

        X_seq, y_seq, y_cls_seq, pq_seq, diff_seq, close_seq, atr_seq, spread_seq, n_feat = (
            _build_multipair_chunk(
            pair_ticks, fe, scalers, args.seq_len, 0, args.label_method,
            target_col=_cache_target_col(args),

            execution_delay_bars=int(getattr(args, "execution_delay_bars", 1)),
            bar_freq=str(getattr(args, "bar_freq", "1min")),
            lookahead_bars=int(getattr(args, "lookahead_bars", LABELING["lookahead_bars"])),
            profit_target_atr=float(getattr(args, "profit_target_atr", LABELING["profit_target_atr"])),
            stop_loss_atr=float(getattr(args, "stop_loss_atr", LABELING["stop_loss_atr"])),
            cross_asset=cross_asset,
            sentiment_pipe=sentiment_pipe,
            historical_news_mode=str(getattr(args, "historical_news_mode", "calendar")),
            historical_news_file=getattr(args, "historical_news_file", None),
            economic_calendar_file=getattr(args, "economic_calendar_file", None),
            cot_data=cot_data,
        ))
        if X_seq.size > 0:
            n_features    = n_feat
            total_samples = len(X_seq)
            _append_chunk(X_seq, y_seq, y_cls_seq, pq_seq, diff_seq, close_seq, atr_seq, spread_seq)
            print(f"  {total_samples:,} joint sequences ├ù {n_features} features")

    if total_samples == 0:
        err = (
            "[MultiPair] No usable samples produced. Check date range and data source.\n"
            + _multipair_zero_samples_help(None)
        )
        _log_error(err)
        raise RuntimeError(err)

    # -- Finalise cache -------------------------------------------------------
    if use_zarr and z_store is not None:
        z_store.attrs["total_samples"] = total_samples
        z_store.attrs["n_features"] = n_features
        z_store.attrs["seq_len"]    = args.seq_len
        z_store.attrs["n_pairs"]    = len(pairs)
        z_store.attrs["pairs"]      = ",".join(pairs)
        z_store.attrs["strategy_mode"] = str(getattr(args, "strategy_mode", "scalping"))
        z_store.attrs["bar_freq"] = str(getattr(args, "bar_freq", "1min"))
        z_store.attrs["lookahead_bars"] = int(getattr(args, "lookahead_bars", LABELING["lookahead_bars"]))
        import json
        meta = {
            "total_samples": int(total_samples),
            "n_features": int(n_features),
            "seq_len": int(args.seq_len),
            "n_pairs": len(pairs),
            "pairs": list(pairs),
            "strategy_mode": str(getattr(args, "strategy_mode", "scalping")),
            "bar_freq": str(getattr(args, "bar_freq", "1min")),
            "lookahead_bars": int(getattr(args, "lookahead_bars", LABELING["lookahead_bars"])),
            "has_rl_market": True,
            "target_col": _cache_target_col(args),

            "y_cls_source": "labels.label",

        }
        with open(str(cache_path) + "_manifest.json", "w") as f:
            json.dump(meta, f)
        _write_feature_schema_json(

            cache_path,

            _build_multipair_feature_schema(scalers, pairs, n_features),

        )

        _save_scaler_npz(cache_path, scalers[pairs[0]])
    else:
        if bin_state["opened"]:
            bin_state["x_f"].close()
            bin_state["y_f"].close()
            if bin_state.get("ycls_f"):
                bin_state["ycls_f"].close()
            if bin_state.get("pq_f"):
                bin_state["pq_f"].close()
            if bin_state.get("diff_f"):
                bin_state["diff_f"].close()
            for _mk in ("close_f", "atr_f", "spread_f"):
                if bin_state.get(_mk):
                    bin_state[_mk].close()

            final_x_shape = tuple([bin_state["total"]] + bin_state["x_shape"][1:])
            final_y_shape = tuple([bin_state["total"]] + bin_state["y_shape"][1:])
            final_1d = (bin_state["total"],)

            x_mmap = np.memmap(str(_x_path(cache_path)) + ".bin", dtype="float32", mode="r", shape=final_x_shape)
            y_mmap = np.memmap(str(_y_path(cache_path)) + ".bin", dtype="float32", mode="r", shape=final_y_shape)
            np.save(_x_path(cache_path), x_mmap)
            np.save(_y_path(cache_path), y_mmap)

            ycls_bin = _y_cls_path(cache_path).replace(".npy", ".bin")
            if Path(ycls_bin).exists():
                ycls_mmap = np.memmap(ycls_bin, dtype="float32", mode="r", shape=final_1d)
                np.save(_y_cls_path(cache_path), ycls_mmap)
            if store_rl_sidecars:
                pq_bin = str(cache_path) + "_pq.bin"
                diff_bin = str(cache_path) + "_diff.bin"
                if Path(pq_bin).exists():
                    pq_mmap = np.memmap(pq_bin, dtype="float32", mode="r", shape=final_1d)
                    np.save(_pq_path(cache_path), pq_mmap)
                if Path(diff_bin).exists():
                    diff_mmap = np.memmap(diff_bin, dtype="uint8", mode="r", shape=final_1d)
                    np.save(_diff_path(cache_path), diff_mmap)
            for _bin_suffix, _save_fn in (
                ("_close.bin", _close_path),
                ("_atr.bin", _atr_path),
                ("_spread.bin", _spread_path),
            ):
                _bpath = str(cache_path) + _bin_suffix
                if Path(_bpath).exists():
                    _mmap = np.memmap(_bpath, dtype="float32", mode="r", shape=final_1d)
                    np.save(_save_fn(cache_path), _mmap)

            import json
            meta = {
                "total_samples": bin_state["total"],
                "n_features": final_x_shape[2],
                "seq_len": args.seq_len,
                "n_pairs": len(pairs),
                "pairs": list(pairs),
                "strategy_mode": str(getattr(args, "strategy_mode", "scalping")),
                "bar_freq": str(getattr(args, "bar_freq", "1min")),
                "lookahead_bars": int(getattr(args, "lookahead_bars", LABELING["lookahead_bars"])),
                "has_rl_market": True,
                "target_col": _cache_target_col(args),

                "y_cls_source": "labels.label",

            }
            with open(str(cache_path) + "_manifest.json", "w") as f:
                json.dump(meta, f)
            _write_feature_schema_json(

                cache_path,

                _build_multipair_feature_schema(scalers, pairs, final_x_shape[2]),

            )


            total_samples = bin_state["total"]
            n_features = final_x_shape[2]

        _save_scaler_npz(cache_path, scalers[pairs[0]])
        
    for p, sc in scalers.items():
        _save_scaler_npz(_scaler_npz_path_pair(cache_path, p), sc)

    readiness_report = _write_pair_readiness_report(
        args,
        cache_path,
        pairs,
        alignment=globals().get("_PAIR_ALIGNMENT_STATS", {}),
    )
    if readiness_report.get("status") == "fail":
        raise RuntimeError(
            f"Pair Readiness Gate Failed. See {str(cache_path)}_pair_readiness_report.json"
        )

    _postprocess_cache_integrity_check(str(cache_path), args, context="MultiPair")

    print(f"\n[MultiPair] Dataset built: {total_samples:,} samples ├ù {n_features} features")
    _verify_dataset(str(cache_path), args, total_samples, n_features, context="MultiPair")
    print(f"            Cached: {cache_path}")

    # ── Write enriched dataset manifest + build log ──────────
    try:
        from data.dataset_manifest import DatasetManifest
        _dm = DatasetManifest(str(Path(cache_path).parent))
        _dm.log_build_event("build_complete", n_rows=total_samples, n_features=n_features)
        _dm.write_manifest(
            source=str(getattr(args, "data_source", "dukascopy")),
            pairs=pairs,
            start=str(getattr(args, "data_start", "")),
            end=str(getattr(args, "data_end", "")),
            freq=str(getattr(args, "bar_freq", "1min")),
            news_mode=str(getattr(args, "historical_news_mode", "calendar")).lower(),
            feature_count=n_features,
            label_method=str(getattr(args, "label_method", "rl_reward")),
            seq_len=int(args.seq_len),
            schema_hash="",
            feature_list=[],
            n_rows_total=total_samples,
            lookahead_bars=int(getattr(args, "lookahead_bars", LABELING.get("lookahead_bars", 15))),
            embargo_bars=int(getattr(args, "embargo_bars", LABELING.get("embargo_bars", 60))),
            purge_bars=int(getattr(args, "purge_bars", 120)),
        )
    except Exception as _m_err:
        _log_warn(f"[Manifest] write failed ({_m_err})")

    # ── Future-leak check ────────────────────────────────────
    try:
        _fwd_ret = None
        if str(cache_path).endswith(".zarr") and Path(cache_path).is_dir():
            import zarr as _zarr
            _z = _zarr.open(str(cache_path), mode="r")
            if "y" in _z:
                _fwd_ret = np.asarray(_z["y"][:], dtype=np.float32)
        if _fwd_ret is not None and len(_fwd_ret) > 0:
            _leaks = DatasetManifest.check_future_leak(
                None, _fwd_ret.tolist(), max_abs_corr=0.30,
            )
            if _leaks:
                _log_warn(
                    f"[LeakCheck] {len(_leaks)} feature(s) correlated with "
                    f"forward returns (|r| > 0.30). Review for data leakage."
                )
            else:
                _log_info("[LeakCheck] No features correlated with forward returns.")
    except Exception as _le_err:
        _log_warn(f"[LeakCheck] skipped ({_le_err})")

    # ── Lockbox reservation ──────────────────────────────────
    try:
        _lookback_days = int(getattr(args, "real_data_window_days", 7) or 7)
        _lockbox_end = str(getattr(args, "data_end", datetime.now(timezone.utc).strftime("%Y-%m-%d")))
        _lockbox_start = (datetime.fromisoformat(_lockbox_end) - timedelta(days=_lookback_days)).strftime("%Y-%m-%d")
        DatasetManifest.reserve_lockbox(str(Path(cache_path).parent), _lockbox_start, _lockbox_end)
    except Exception:
        pass

    return str(cache_path), total_samples, n_features, scalers[pairs[0]]


def build_dataset_chunked(args) -> "tuple[str, int, int, StandardScaler]":
    """
    Ingest up to 20M ticks in chunks, write sequences to Zarr (primary) / NPY memmap (fallback).

    Returns: (cache_path, n_samples, n_features, scaler)
    """

    use_zarr = bool(ZARR)
    pairs    = _get_pairs(args)
    is_multi = len(pairs) > 1
    if not is_multi:
        globals()["_PAIR_READINESS_STATS"] = {}
        globals()["_PAIR_ALIGNMENT_STATS"] = {}

    def _cache_present(cp: Path) -> bool:
        if str(cp).endswith(".zarr"):
            return cp.is_dir()
        return Path(_x_path(cp)).exists() and Path(_y_path(cp)).exists()

    # Multi-pair: delegate to dedicated function
    if is_multi:
        Path(args.data_cache).mkdir(parents=True, exist_ok=True)
        cache_path = _get_cache_path(args)
        if _cache_present(cache_path) and not args.force_rebuild:
            ok, reason = _validate_cache_integrity(str(cache_path), args)
            if not ok:
                if getattr(args, "auto_rebuild_on_mismatch", False):
                    print(f"[MultiPair] Cache mismatch ({reason}) - rebuilding.")
                    _delete_cache_artifacts(str(cache_path))
                elif getattr(args, "integrity_gate", True):
                    raise RuntimeError(f"Cache integrity failed: {reason}. Use --force-rebuild.")
        if _cache_present(cache_path) and not args.force_rebuild:
            cache_format = "zarr" if str(cache_path).endswith(".zarr") else "npy"
            if getattr(args, "_resume_zarr", False):
                pass # Fall through to rebuild
            elif cache_format == "zarr":
                import zarr
                z_store = zarr.open(cache_path, mode="r")
                if "total_samples" not in z_store.attrs:
                    args._resume_zarr = True
                else:
                    n_samples = z_store.attrs["total_samples"]
                    n_features = z_store.attrs["n_features"]
                    scaler = _load_scaler_npz(Path(cache_path)) or _identity_scaler(n_features)
                    n_samples = _clamp_n_samples_to_disk(str(cache_path), n_samples)
                    print(f"[MultiPair] {n_samples:,} samples x {n_features} features (cached)")
                    _warn_multitask_cache_sidecars(str(cache_path), args)
                    args._n_pairs    = len(pairs)
                    args._f_per_pair = n_features // len(pairs)
                    return str(cache_path), n_samples, n_features, scaler
            else:
                import json
                meta_path = str(cache_path) + "_meta.json"
                if not os.path.exists(meta_path):
                    args._resume_zarr = True
                else:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    n_samples = meta["total_samples"]
                    n_features = meta["n_features"]
                    scaler = _load_scaler_npz(Path(cache_path)) or _identity_scaler(n_features)
                    n_samples = _clamp_n_samples_to_disk(str(cache_path), n_samples)
                    print(f"[MultiPair] {n_samples:,} samples x {n_features} features (cached)")
                    _warn_multitask_cache_sidecars(str(cache_path), args)
                    args._n_pairs    = len(pairs)
                    args._f_per_pair = n_features // len(pairs)
                    return str(cache_path), n_samples, n_features, scaler
        fe = FeatureEngineer(
            atr_window=FEATURES["atr_window"], ofi_window=FEATURES["ofi_window"],
            tar_window=FEATURES["trade_arrival_window"], rsi_period=FEATURES["rsi_period"],
            macd_fast=FEATURES["macd_fast"], macd_slow=FEATURES["macd_slow"],
            macd_signal=FEATURES["macd_signal"], bb_window=FEATURES["bollinger_window"],
            bb_std=FEATURES["bollinger_std"], lag_windows=FEATURES["lag_windows"],
        )
        cache_str, n_samples, n_features, scaler = _build_multipair_dataset(
            args, pairs, cache_path, fe,
        )
        args._n_pairs    = len(pairs)
        args._f_per_pair = n_features // len(pairs)
        return cache_str, n_samples, n_features, scaler

    cache_path = _get_cache_path(args)
    Path(args.data_cache).mkdir(parents=True, exist_ok=True)
    _cache_engine = "zarr" if use_zarr and str(cache_path).endswith(".zarr") else "npy"
    _news_mode = str(getattr(args, "historical_news_mode", "calendar") or "calendar").lower()
    print(f"[Data] Cache target: {cache_path}")
    print(f"[Data] Cache engine: {_cache_engine.upper()} | historical_news_mode={_news_mode}")

    if _cache_present(cache_path) and not args.force_rebuild:
        ok, reason = _validate_cache_integrity(str(cache_path), args)
        if not ok:
            if getattr(args, "auto_rebuild_on_mismatch", False):
                print(f"[Data] WARN: cache integrity mismatch ({reason}) ΓÇö auto rebuilding.")
                _delete_cache_artifacts(str(cache_path))
            elif getattr(args, "integrity_gate", True):
                raise RuntimeError(
                    f"Cache integrity check failed: {reason}. "
                    "Run with --force-rebuild or --auto-rebuild-on-mismatch."
                )
        print(f"\n[Data] Found cached dataset: {cache_path}")
    if _cache_present(cache_path) and not args.force_rebuild:
        # -- Zarr cache ------------------------------------------------------
        if use_zarr and str(cache_path).endswith(".zarr") and cache_path.is_dir():
            z = _zarr_open_group(str(cache_path), mode="r")
            n_samples  = min(int(z["X"].shape[0]), int(z["y"].shape[0]))
            n_features = z["X"].shape[2]
            scaler     = _load_scaler_npz(Path(cache_path)) or _identity_scaler(n_features)
            n_samples  = _clamp_n_samples_to_disk(str(cache_path), n_samples)
            print(f"[Data] {n_samples:,} samples ├ù {n_features} features (zarr cache)")
            _warn_multitask_cache_sidecars(str(cache_path), args)
            return str(cache_path), n_samples, n_features, scaler
        pass
    print(f"\n[Data] Building 20M tick dataset ΓÇö chunk size: {args.chunk_size:,}")
    _pairs_display = ", ".join(_get_pairs(args))
    print(f"       Source: {args.data_source} | Pairs: {_pairs_display}")
    print(f"       News mode: {_news_mode} | Cache engine: {_cache_engine.upper()}")
    use_real_cross = (
        args.cross_asset_mode == "real"
        or (args.cross_asset_mode == "auto" and args.data_source != "synthetic")
    )
    cross_asset_source = _resolve_cross_asset_source(args)
    # Treat empty env var the same as "not set" so the default data/processed/cross_asset
    # path is always used when CROSS_ASSET_CACHE_DIR is unset or blank.
    cross_asset_cache_dir = (
        os.getenv("CROSS_ASSET_CACHE_DIR", "").strip()
        or str(Path(args.data_cache) / "cross_asset")
    )
    cross_asset = None
    if use_real_cross:
        try:
            cross_asset = load_cross_asset_panel(
                start=args.data_start,
                end=args.data_end,
                cache_dir=cross_asset_cache_dir,
                source=cross_asset_source,
            )
            print(f"[CrossAsset] Loaded external assets: {len(cross_asset)} "
                  f"(source={cross_asset_source}, cache={cross_asset_cache_dir})")
        except Exception as e:
            print(f"[CrossAsset] WARN: external load failed ({e}) ΓÇö falling back to synthetic")
            cross_asset = None

    fe     = FeatureEngineer(
        atr_window  = FEATURES["atr_window"],
        ofi_window  = FEATURES["ofi_window"],
        tar_window  = FEATURES["trade_arrival_window"],
        rsi_period  = FEATURES["rsi_period"],
        macd_fast   = FEATURES["macd_fast"],
        macd_slow   = FEATURES["macd_slow"],
        macd_signal = FEATURES["macd_signal"],
        bb_window   = FEATURES["bollinger_window"],
        bb_std      = FEATURES["bollinger_std"],
        lag_windows = FEATURES["lag_windows"],
    )
    scaler  = StandardScaler()
    chunk_n     = 0
    total_samples = 0
    n_features_total  = 0
    z_store     = None   # zarr group (primary)
    
    # Load COT data
    import polars as pl
    cot_path = Path("data/raw/cot/cot_financials_cleaned.parquet")
    cot_data = pl.read_parquet(cot_path) if cot_path.exists() else None
    if cot_data is not None:
        print(f"[Data] Loaded Smart Money COT data ({len(cot_data)} rows)")

    # Sentiment pipeline (single-pair path)
    sentiment_pipe = None
    if str(getattr(args, "sentiment_mode", "finbert")).lower() != "off":
        try:
            pref = "finbert" if str(args.sentiment_mode).lower() == "finbert" else "vader"
            sentiment_pipe = SentimentPipeline(prefer_backend=pref, use_cache=True)
            print(f"[Sentiment] mode={args.sentiment_mode} enabled")
        except Exception as e:
            print(f"[Sentiment] WARN: init failed ({e}) — disabling sentiment features")
            sentiment_pipe = None

    # Pre-score all headlines across windows so per-chunk calls are cache hits
    if (
        sentiment_pipe is not None
        and str(getattr(args, "historical_news_mode", "calendar")).lower() == "full"
    ):
        try:
            _headlines = collect_headlines_for_range(
                args.data_start,
                args.data_end,
                [str(getattr(args, "pair", "EURUSD"))],
                news_file=getattr(args, "historical_news_file", None),
                calendar_file=getattr(args, "economic_calendar_file", None),
            )
            if _headlines:
                sentiment_pipe.prefetch_headlines(_headlines)
            del _headlines
        except Exception as _pf_err:
            print(f"[Sentiment] prefetch skipped ({_pf_err})")

    if ZARR and not use_zarr:
        print("[Data] Zarr writes disabled on Windows; using NPY memmap cache.")
    if use_zarr:
        # LZ4 via Blosc: ~3├ù faster decompress than LZF, good random-access perf
        _compressor = _Blosc(cname="lz4", clevel=3, shuffle=_Blosc.BITSHUFFLE)
    else:
        # Stream raw bytes to disk to avoid 200GB RAM usage and Windows 32-bit NPY save overflows
        _x_fp = open(_x_path(cache_path).replace(".npy", ".bin"), "wb")
        _y_fp = open(_y_path(cache_path).replace(".npy", ".bin"), "wb")
        _diff_fp = open(str(cache_path) + "_diff.bin", "wb") if args.label_method == "rl_reward" else None
        _pq_fp = open(str(cache_path) + "_pq.bin", "wb") if args.label_method == "rl_reward" else None
        _ycls_fp = open(_y_cls_path(cache_path).replace(".npy", ".bin"), "wb")
        _close_fp = open(str(cache_path) + "_close.bin", "wb")
        _atr_fp = open(str(cache_path) + "_atr.bin", "wb")
        _spread_fp = open(str(cache_path) + "_spread.bin", "wb")
        _n_samples_written = 0

    # -- Load all ticks or generate in chunks ---------------------------------
    if args.data_source == "synthetic":
        print(f"[Data] Generating {args.n_ticks:,} synthetic ticks in chunks...")
        chunk_specs = [(idx, None, None, min(args.chunk_size, max(args.n_ticks - idx * args.chunk_size, 0)))
                       for idx in range((args.n_ticks + args.chunk_size - 1) // args.chunk_size)]
    else:
        _base_days = _real_data_window_days(args)
        window_days = _effective_window_days(args)
        date_windows = _iter_date_windows(args.data_start, args.data_end, window_days)
        chunk_specs = [(idx, win_start, win_end, None) for idx, (win_start, win_end) in enumerate(date_windows)]
        _batch_n = max(1, int(getattr(args, "window_batch_days", 1) or 1))
        if _batch_n > 1:
            print(f"[Data] Real-data windows: {len(date_windows)} x {window_days} day(s) "
                  f"(base {_base_days}d x {_batch_n} batch)")
        else:
            print(f"[Data] Real-data windows: {len(date_windows)} x {window_days} day(s)")

    mgr = None if args.data_source == "synthetic" else ForexDataManager(verbose=True)

    for chunk_n, win_start, win_end, chunk_ticks in chunk_specs:
        t0 = time.time()

        try:
            if args.data_source == "synthetic":
                ticks_chunk = generate_synthetic_tick_data(n_rows=chunk_ticks)
            else:
                print(
                    f"[Data] Loading {args.data_source} for {args.pair} "
                    f"({win_start} -> {win_end}). "
                    "First run downloads many hourly files; this can take tens of minutes to hours."
                )
                ticks_chunk = mgr.load(
                    pair   = args.pair,
                    source = args.data_source,
                    start  = win_start,
                    end    = win_end,
                    session_only = (not getattr(args, "full_day_data", False)),
                )

            X_seq, y_seq, diff_seq, pq_seq, y_cls_seq, close_seq, atr_seq, spread_seq, n_feat, _time_idx = (
                _build_chunk(
                ticks_chunk, fe, scaler,
                seq_len    = args.seq_len,
                chunk_idx  = chunk_n,
                label_method = args.label_method,
                target_col = _cache_target_col(args),

                execution_delay_bars = int(getattr(args, "execution_delay_bars", 1)),
                bar_freq = str(getattr(args, "bar_freq", "1min")),
                lookahead_bars = int(getattr(args, "lookahead_bars", LABELING["lookahead_bars"])),
                profit_target_atr = float(getattr(args, "profit_target_atr", LABELING["profit_target_atr"])),
                stop_loss_atr = float(getattr(args, "stop_loss_atr", LABELING["stop_loss_atr"])),
                cross_asset = cross_asset,
                sentiment_pipe = sentiment_pipe,
                pair = str(getattr(args, "pair", "EURUSD")),
                historical_news_mode = str(getattr(args, "historical_news_mode", "calendar")),
                historical_news_file = getattr(args, "historical_news_file", None),
                economic_calendar_file = getattr(args, "economic_calendar_file", None),
                cot_data = cot_data,
            ))
            del ticks_chunk
            gc.collect()

        except Exception as _exc:
            _log_error(f"[Data] chunk {chunk_n} build failed", _exc)
            raise

        if X_seq.size == 0:
            continue

        n_features_total = n_feat
        n_samples_chunk = len(X_seq)
        total_samples += n_samples_chunk

        if use_zarr:
            if z_store is None:
                if getattr(args, "_resume_zarr", False):
                    z_store = _zarr_open_group(str(cache_path), mode="a")
                    existing_samples = int(z_store["X"].shape[0]) if "X" in z_store else 0
                    total_samples += existing_samples
                else:
                    if __import__('pathlib').Path(cache_path).exists():
                        for _retry in range(10):
                            __import__('shutil').rmtree(cache_path, ignore_errors=True)
                            if not __import__('pathlib').Path(cache_path).exists(): break
                            __import__('time').sleep(0.5)
                    z_store = _zarr_open_group(str(cache_path), mode="w")
                    c0 = (min(64, n_samples_chunk),) + X_seq.shape[1:]
                    _zarr_create(z_store, "X", shape=(0,) + X_seq.shape[1:], chunks=c0, dtype="float32", compressor=_compressor)
                    _zarr_create(z_store, "y", shape=(0,), chunks=(c0[0],), dtype="float32", compressor=_compressor)
                    _zarr_create(z_store, "diff", shape=(0,), chunks=(c0[0],), dtype="uint8", compressor=_compressor)
                    _zarr_create(z_store, "pq", shape=(0,), chunks=(c0[0],), dtype="float32", compressor=_compressor)
                    _zarr_create(z_store, "y_cls", shape=(0,), chunks=(c0[0],), dtype="float32", compressor=_compressor)
                    _zarr_create(z_store, "close", shape=(0,), chunks=(c0[0],), dtype="float32", compressor=_compressor)
                    _zarr_create(z_store, "atr", shape=(0,), chunks=(c0[0],), dtype="float32", compressor=_compressor)
                    _zarr_create(z_store, "spread", shape=(0,), chunks=(c0[0],), dtype="float32", compressor=_compressor)
            
            z_store["X"].append(X_seq)
            z_store["y"].append(y_seq)
            z_store["y_cls"].append(y_cls_seq)
            z_store["close"].append(close_seq)
            if "atr" in z_store: z_store["atr"].append(atr_seq)
            if "spread" in z_store: z_store["spread"].append(spread_seq)
            # Keep sidecar lengths aligned with X even when a chunk omits them.
            if "diff" in z_store:
                z_store["diff"].append(
                    diff_seq if diff_seq is not None else np.zeros(n_samples_chunk, dtype=np.uint8)
                )
            if "pq" in z_store:
                z_store["pq"].append(
                    pq_seq if pq_seq is not None else np.ones(n_samples_chunk, dtype=np.float32)
                )
        else:
            _x_fp.write(X_seq.tobytes())
            _y_fp.write(y_seq.tobytes())
            _ycls_fp.write(y_cls_seq.tobytes())
            _close_fp.write(close_seq.tobytes())
            _atr_fp.write(atr_seq.tobytes())
            _spread_fp.write(spread_seq.tobytes())
            if args.label_method == "rl_reward":
                _diff_fp.write(diff_seq.tobytes())
                _pq_fp.write(pq_seq.tobytes())
            _n_samples_written += n_samples_chunk

        elapsed = time.time() - t0
        if args.data_source == "synthetic":
            done = min((chunk_n + 1) * args.chunk_size, args.n_ticks)
            pct = min(done / args.n_ticks * 100, 100)
            print(f"  Chunk {chunk_n+1} | {n_samples_chunk:,} seqs | "
                  f"{elapsed:.1f}s | {pct:.0f}% ({total_samples:,} total)")
        else:
            print(f"  Window {chunk_n+1}/{len(chunk_specs)} | {n_samples_chunk:,} seqs | "
                  f"{elapsed:.1f}s | {total_samples:,} total")

        # Structured build log
        try:
            from data.dataset_manifest import DatasetManifest
            _manifest_dir = Path(cache_path).parent if str(cache_path).endswith((".zarr", "")) else Path(cache_path).parent
            _dm = DatasetManifest(str(_manifest_dir))
            _dm.log_build_event(
                "chunk_built",
                pair=str(getattr(args, "pair", "EURUSD")),
                chunk_idx=chunk_n,
                n_rows=n_samples_chunk,
                n_features=n_features_total,
                duration_s=elapsed,
            )
        except Exception:
            pass

        if _TRAIN_LOGGER: _TRAIN_LOGGER.heartbeat()
        del X_seq, y_seq, diff_seq, pq_seq, y_cls_seq, close_seq, atr_seq, spread_seq
        gc.collect()

    if total_samples == 0:
        err = (
            "[Data] No usable samples were produced from the selected date range/source. "
            "Likely causes: vendor returned mostly empty hour files, wrong pair/date range, "
            "or blocked data endpoint. Try a shorter recent range first and verify raw cache."
        )
        _log_error(err)
        raise RuntimeError(err)

    # -- Finalise storage ------------------------------------------------------
    if use_zarr and z_store is not None:
        z_store.attrs["total_samples"] = total_samples
        z_store.attrs["n_features"] = n_features_total
        z_store.attrs["seq_len"]    = args.seq_len
        z_store.attrs["strategy_mode"] = str(getattr(args, "strategy_mode", "scalping"))
        z_store.attrs["bar_freq"] = str(getattr(args, "bar_freq", "1min"))
        z_store.attrs["lookahead_bars"] = int(getattr(args, "lookahead_bars", LABELING["lookahead_bars"]))
        meta = {
            "total_samples": int(total_samples),
            "seq_len": int(args.seq_len),
            "n_features": int(n_features_total),
            "label_method": args.label_method,
            "strategy_mode": str(getattr(args, "strategy_mode", "scalping")),
            "bar_freq": str(getattr(args, "bar_freq", "1min")),
            "lookahead_bars": int(getattr(args, "lookahead_bars", LABELING["lookahead_bars"])),
            "has_rl_market": True,
            "pairs": list(_get_pairs(args)),
            "target_col": _cache_target_col(args),

            "y_cls_source": "labels.label",

        }
        import json
        with open(str(cache_path) + "_manifest.json", "w") as f:
            json.dump(meta, f)
        _write_feature_schema_json(cache_path, _scaler_feature_names(scaler))

        _save_scaler_npz(cache_path, scaler)
    else:
        _x_fp.close(); _y_fp.close()
        if _diff_fp: _diff_fp.close()
        if _pq_fp: _pq_fp.close()
        _ycls_fp.close()
        _close_fp.close(); _atr_fp.close(); _spread_fp.close()

        final_x_shape = (_n_samples_written, args.seq_len, n_features_total)
        final_y_shape = (_n_samples_written,)
        final_1d = (_n_samples_written,)
        if _n_samples_written > 0:
            x_mmap = np.memmap(
                _x_path(cache_path).replace(".npy", ".bin"),
                dtype="float32",
                mode="r",
                shape=final_x_shape,
            )
            y_mmap = np.memmap(
                _y_path(cache_path).replace(".npy", ".bin"),
                dtype="float32",
                mode="r",
                shape=final_y_shape,
            )
            np.save(_x_path(cache_path), x_mmap)
            np.save(_y_path(cache_path), y_mmap)

            ycls_bin = _y_cls_path(cache_path).replace(".npy", ".bin")
            if Path(ycls_bin).exists():
                ycls_mmap = np.memmap(ycls_bin, dtype="float32", mode="r", shape=final_1d)
                np.save(_y_cls_path(cache_path), ycls_mmap)
            if args.label_method == "rl_reward":
                pq_bin = str(cache_path) + "_pq.bin"
                diff_bin = str(cache_path) + "_diff.bin"
                if Path(pq_bin).exists():
                    pq_mmap = np.memmap(pq_bin, dtype="float32", mode="r", shape=final_1d)
                    np.save(_pq_path(cache_path), pq_mmap)
                if Path(diff_bin).exists():
                    diff_mmap = np.memmap(diff_bin, dtype="uint8", mode="r", shape=final_1d)
                    np.save(_diff_path(cache_path), diff_mmap)
        for _bin_suffix, _save_fn in (
            ("_close.bin", _close_path),
            ("_atr.bin", _atr_path),
            ("_spread.bin", _spread_path),
        ):
            _bpath = str(cache_path) + _bin_suffix
            if Path(_bpath).exists() and _n_samples_written > 0:
                _mmap = np.memmap(_bpath, dtype="float32", mode="r", shape=final_1d)
                np.save(_save_fn(cache_path), _mmap)

        # Save metadata for memmap loading
        meta = {
            "total_samples": _n_samples_written,
            "seq_len": args.seq_len,
            "n_features": n_features_total,
            "label_method": args.label_method,
            "strategy_mode": str(getattr(args, "strategy_mode", "scalping")),
            "bar_freq": str(getattr(args, "bar_freq", "1min")),
            "lookahead_bars": int(getattr(args, "lookahead_bars", LABELING["lookahead_bars"])),
            "has_rl_market": True,
            "pairs": list(_get_pairs(args)),
            "target_col": _cache_target_col(args),

            "y_cls_source": "labels.label",

        }
        import json
        with open(str(cache_path) + "_manifest.json", "w") as f:
            json.dump(meta, f)
        _write_feature_schema_json(cache_path, _scaler_feature_names(scaler))

            
        _save_scaler_npz(cache_path, scaler)

    readiness_report = _write_pair_readiness_report(
        args,
        cache_path,
        list(_get_pairs(args)),
        alignment=globals().get("_PAIR_ALIGNMENT_STATS", {}),
    )
    if readiness_report.get("status") == "fail":
        raise RuntimeError(
            f"Pair Readiness Gate Failed. See {str(cache_path)}_pair_readiness_report.json"
        )

    print(f"\n[Data] Dataset built: {total_samples:,} samples ├ù "
          f"{n_features_total} features ├ù seq_len {args.seq_len}")
    _postprocess_cache_integrity_check(str(cache_path), args, context="Data")
    _verify_dataset(str(cache_path), args, total_samples, n_features_total, context="Data")
    print(f"       Cached at: {cache_path}")

    # ── Write enriched dataset manifest + build log ──────────────────
    try:
        from data.dataset_manifest import DatasetManifest
        _dm = DatasetManifest(str(Path(cache_path).parent))
        _dm.log_build_event("build_complete", n_rows=total_samples, n_features=n_features_total)
        _dm.write_manifest(
            source=str(getattr(args, "data_source", "dukascopy")),
            pairs=_get_pairs(args),
            start=str(getattr(args, "data_start", "")),
            end=str(getattr(args, "data_end", "")),
            freq=str(getattr(args, "bar_freq", "1min")),
            news_mode=str(getattr(args, "historical_news_mode", "calendar")).lower(),
            feature_count=n_features_total,
            label_method=str(getattr(args, "label_method", "rl_reward")),
            seq_len=int(args.seq_len),
            schema_hash=_scaler_feature_names(scaler) if scaler else "",
            feature_list=list(_scaler_feature_names(scaler)) if scaler else [],
            n_rows_total=total_samples,
            lookahead_bars=int(getattr(args, "lookahead_bars", LABELING.get("lookahead_bars", 15))),
            embargo_bars=int(getattr(args, "embargo_bars", LABELING.get("embargo_bars", 60))),
            purge_bars=int(getattr(args, "purge_bars", 120)),
        )
    except Exception as _m_err:
        _log_warn(f"[Manifest] write failed ({_m_err})")

    # ── Future-leak check ────────────────────────────────────────────
    try:
        _feats_col = None
        _fwd_ret = None
        if str(cache_path).endswith(".zarr") and Path(cache_path).is_dir():
            import zarr as _zarr
            _z = _zarr.open(str(cache_path), mode="r")
            if "X" in _z and "y" in _z:
                _feats_col = [str(c) for c in _z["X"].attrs.get("columns", [])]
                _fwd_ret = np.asarray(_z["y"][:], dtype=np.float32)
        if _feats_col and _fwd_ret is not None and len(_feats_col) > 0:
            _leaks = DatasetManifest.check_future_leak(
                None, _fwd_ret.tolist(), max_abs_corr=0.30,
            )
            if _leaks:
                _log_warn(
                    f"[LeakCheck] {len(_leaks)} feature(s) correlated with "
                    f"forward returns (|r| > 0.30). Review for data leakage."
                )
            else:
                _log_info("[LeakCheck] No features correlated with forward returns.")
    except Exception as _le_err:
        _log_warn(f"[LeakCheck] skipped ({_le_err})")

    # ── Lockbox reservation ──────────────────────────────────────────
    try:
        _lookback_days = int(getattr(args, "real_data_window_days", 7) or 7)
        _lockbox_end = str(getattr(args, "data_end", datetime.now(timezone.utc).strftime("%Y-%m-%d")))
        _lockbox_start = (datetime.fromisoformat(_lockbox_end) - timedelta(days=_lookback_days)).strftime("%Y-%m-%d")
        DatasetManifest.reserve_lockbox(str(Path(cache_path).parent), _lockbox_start, _lockbox_end)
    except Exception:
        pass

    return str(cache_path), total_samples, n_features_total, scaler


# -----------------------------------------------------------------------------
# THREAD PREFETCH WRAPPER
# On Windows, num_workers=0 is forced (process-spawn overhead + WDDM contention),
# so DataLoader decompresses zarr chunks synchronously ΓÇö GPU idles during CPU work.
# This wrapper runs a background *thread* (no spawn cost) that decompresses the
# next N batches while the GPU is busy with the current one, hiding I/O latency.
# -----------------------------------------------------------------------------

class _ThreadPrefetchLoader:
    """Wraps any DataLoader to prefetch batches in a daemon background thread.

    Usage:
        loader = DataLoader(ds, ...)
        loader = _ThreadPrefetchLoader(loader, prefetch=2)

    The background thread decompresses / loads the next ``prefetch`` batches
    while the training loop is busy doing a forward+backward pass.  Queue depth
    of 2 is enough to overlap one full decompression cycle with GPU compute
    without wasting extra RAM.
    """
    def __init__(self, loader, prefetch: int = 2):
        self._loader  = loader
        self._prefetch = max(1, prefetch)

    # Forward attribute access to the underlying loader (len, dataset, etc.)
    def __getattr__(self, name):
        return getattr(self._loader, name)

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        _sentinel = object()
        q = _queue.Queue(maxsize=self._prefetch)

        def _producer():
            try:
                for batch in self._loader:
                    q.put(batch)
            except Exception as exc:          # propagate exceptions to consumer
                q.put(exc)
            finally:
                q.put(_sentinel)

        t = threading.Thread(target=_producer, daemon=True)
        t.start()
        try:
            while True:
                item = q.get()
                if item is _sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            # Drain the queue so the producer thread can exit cleanly if the
            # consumer breaks early (e.g. chunk-level early stopping).
            t.join(timeout=0)
            while not q.empty():
                try:
                    q.get_nowait()
                except _queue.Empty:
                    break


# -----------------------------------------------------------------------------
# MEMORY-MAPPED SEQUENCE DATASET
# -----------------------------------------------------------------------------

class MemmapSequenceDataset(Dataset):
    """
    Reads pre-built sequences directly from Zarr / NPY memmap on disk.
    Never loads the full dataset into RAM ΓÇö workers stream batches asynchronously.

    Read priority
    -------------
    1. Zarr directory store (.zarr)  ΓÇö concurrent reads, no locking, LZ4 compressed
    2. NPY memory-maps (_X.npy / _y.npy) ΓÇö O(1) random access, zero compression overhead

    Why this is fast:
      - Zarr/NPY: OS page cache pre-fetches adjacent chunks while GPU trains.
      - num_workers=4-8 parallel DataLoader workers ΓÇö no SWMR or retry loops needed.
      - pin_memory=True eliminates CPU->GPU copy latency.
      - persistent_workers=True avoids worker restart overhead per epoch.
    """

    def __init__(self, cache_path: str, indices: np.ndarray):
        self.cache_path = cache_path
        # Contiguous copy: avoids pickling a view whose base is the full index array.
        self.indices = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))

        # -- Detect storage backend --------------------------------------------
        npy_x = Path(_x_path(cache_path))
        npy_y = Path(_y_path(cache_path))

        self.use_zarr = (
            ZARR
            and Path(cache_path).is_dir()
            and (Path(cache_path) / ".zgroup").exists()
        )

        if self.use_zarr:
            _z = _zarr_open_group(cache_path, mode="r")
            self.X_zarr = _z["X"]
            self.y_zarr = _z["y"]
        else:
            self.X_mmap = np.load(str(npy_x), mmap_mode="r")
            self.y_mmap = np.load(str(npy_y), mmap_mode="r")

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        if self.use_zarr:
            X = np.array(self.X_zarr[real_idx], dtype=np.float32)
            y = float(self.y_zarr[real_idx])
        else:
            X = np.array(self.X_mmap[real_idx], dtype=np.float32)
            y = float(self.y_mmap[real_idx])
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        y = float(np.nan_to_num(np.float32(y), nan=0.0, posinf=0.0, neginf=0.0))
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __getstate__(self):
        # Never pickle memmaps/zarr arrays into worker processes ΓÇö
        # pickling materialises the full array on Windows (MemoryError).
        return {"cache_path": self.cache_path, "indices": self.indices,
                "use_zarr": self.use_zarr}

    def __setstate__(self, state):
        self.cache_path = state["cache_path"]
        self.indices    = state["indices"]
        self.use_zarr   = state.get("use_zarr", False)
        if self.use_zarr:
            _z = _zarr_open_group(self.cache_path, mode="r")
            self.X_zarr = _z["X"]
            self.y_zarr = _z["y"]
        else:
            npy_x = Path(_x_path(self.cache_path))
            npy_y = Path(_y_path(self.cache_path))
            self.X_mmap = np.load(str(npy_x), mmap_mode="r")
            self.y_mmap = np.load(str(npy_y), mmap_mode="r")


class ZarrStreamDataset(IterableDataset):
    """
    Sequential-read IterableDataset for zarr-backed training data.

    Problem solved
    --------------
    MemmapSequenceDataset.__getitem__ reads one random row at a time.  For
    zarr with 512-row chunks (~262 MB each), every row access decompresses an
    entire 262 MB block.  With batch_size=256 and random shuffle, each batch
    triggers ~242 separate chunk decompressions ΓÇö measured at 2ΓÇô49 s each.

    This class reads zarr in blocks aligned to the zarr chunk boundary so each
    physical chunk on disk is decompressed **exactly once** per epoch visit,
    regardless of batch size.  Rows within each block are shuffled in memory;
    the order in which blocks are visited is also shuffled each epoch.

    Multi-worker safety
    -------------------
    Each DataLoader worker receives a disjoint, contiguous slice of the sorted
    index array.  Because the index is sorted, each worker naturally owns
    different zarr chunks ΓÇö no locking or coordination needed.
    """

    def __init__(self, cache_path: str, indices: np.ndarray,
                 shuffle_chunks: bool = True, multitask_targets: bool = False,
                 return_indices: bool = False):
        self.cache_path     = cache_path
        # Sort once so contiguous positions map to the same zarr chunk.
        self.sorted_idx     = np.sort(np.asarray(indices, dtype=np.int64))
        self.shuffle_chunks = shuffle_chunks
        self.multitask_targets = bool(multitask_targets)
        self.return_indices = bool(return_indices)
        self.use_zarr       = (
            ZARR
            and Path(cache_path).is_dir()
            and (Path(cache_path) / ".zgroup").exists()
        )
        # Read the zarr row-chunk size directly from metadata (no data I/O).
        self._zarr_row_chunk = 512  # safe default
        if self.use_zarr:
            import json as _json
            meta_file = Path(cache_path) / "X" / ".zarray"
            if meta_file.exists():
                self._zarr_row_chunk = int(
                    _json.loads(meta_file.read_text())["chunks"][0]
                )

    def __len__(self) -> int:
        return len(self.sorted_idx)

    def _open_arrays(self):
        import torch
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        
        if getattr(self, '_opened_arrays', None) is None:
            self._opened_arrays = {}
            
        if worker_id not in self._opened_arrays:
            if self.use_zarr:
                z = _zarr_open_group(self.cache_path, mode="r")
                y_cls = z["y_cls"] if "y_cls" in z else None
                pq = z["pq"] if "pq" in z else None
                self._opened_arrays[worker_id] = (z["X"], z["y"], y_cls, pq, True)
            else:
                X = np.load(_x_path(self.cache_path), mmap_mode="r")
                y = np.load(_y_path(self.cache_path), mmap_mode="r")
                y_cls_p, pq_p = Path(_y_cls_path(self.cache_path)), Path(_pq_path(self.cache_path))
                y_cls = np.load(str(y_cls_p), mmap_mode="r") if y_cls_p.exists() else None
                pq = np.load(str(pq_p), mmap_mode="r") if pq_p.exists() else None
                self._opened_arrays[worker_id] = (X, y, y_cls, pq, False)
                
        return self._opened_arrays[worker_id]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        idx = self.sorted_idx

        # Assign each worker a contiguous slice of the sorted index so workers
        # naturally own different zarr chunks (no duplicated decompression).
        if worker_info is not None:
            n, wid = worker_info.num_workers, worker_info.id
            per    = (len(idx) + n - 1) // n
            idx    = idx[wid * per : (wid + 1) * per]

        if len(idx) == 0:
            return

        X_arr, y_arr, y_cls_arr, pq_arr, is_zarr = self._open_arrays()
        cs = self._zarr_row_chunk

        # Split sorted index into per-zarr-chunk blocks (all indices in a block
        # belong to the same physical chunk ΓåÆ exactly 1 decompression per block).
        chunk_nums   = idx // cs
        split_pts    = np.where(np.diff(chunk_nums))[0] + 1
        blocks       = np.split(idx, split_pts)

        if self.shuffle_chunks:
            np.random.shuffle(blocks)

        for block_idx in blocks:
            if is_zarr:
                X_blk = np.array(X_arr.oindex[block_idx], dtype=np.float32)
                y_blk = np.array(y_arr.oindex[block_idx], dtype=np.float32)
                yc_blk = (np.array(y_cls_arr.oindex[block_idx], dtype=np.float32)
                          if y_cls_arr is not None else None)
                pq_blk = (np.array(pq_arr.oindex[block_idx], dtype=np.float32)
                          if pq_arr is not None else None)
            else:
                X_blk = np.array(X_arr[block_idx], dtype=np.float32)
                y_blk = np.array(y_arr[block_idx], dtype=np.float32)
                yc_blk = (np.array(y_cls_arr[block_idx], dtype=np.float32)
                          if y_cls_arr is not None else None)
                pq_blk = (np.array(pq_arr[block_idx], dtype=np.float32)
                          if pq_arr is not None else None)

            # Sanitise: replace NaN/Inf with 0 so bad chunks can't poison the model
            np.nan_to_num(X_blk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.nan_to_num(y_blk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            if yc_blk is not None:
                np.nan_to_num(yc_blk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            if pq_blk is not None:
                np.nan_to_num(pq_blk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            perm = np.random.permutation(len(X_blk))
            for j in perm:
                y_t = torch.tensor(float(y_blk[j]), dtype=torch.float32)
                if self.return_indices:
                    smp_idx = int(block_idx[j])
                    if self.multitask_targets:
                        yc = float(yc_blk[j]) if yc_blk is not None else float(y_blk[j])
                        pq = float(pq_blk[j]) if pq_blk is not None else min(1.0, abs(float(y_blk[j])))
                        yield (
                            torch.tensor(X_blk[j], dtype=torch.float32),
                            y_t,
                            torch.tensor(yc, dtype=torch.float32),
                            torch.tensor(pq, dtype=torch.float32),
                            torch.tensor(smp_idx, dtype=torch.long),
                        )
                    else:
                        yield (torch.tensor(X_blk[j], dtype=torch.float32), y_t,
                               torch.tensor(smp_idx, dtype=torch.long))
                else:
                    y_t = torch.tensor(float(y_blk[j]), dtype=torch.float32)
                    if self.multitask_targets:
                        yc = float(yc_blk[j]) if yc_blk is not None else float(y_blk[j])
                        pq = float(pq_blk[j]) if pq_blk is not None else min(1.0, abs(float(y_blk[j])))
                        yield (
                            torch.tensor(X_blk[j], dtype=torch.float32),
                            y_t,
                            torch.tensor(yc, dtype=torch.float32),
                            torch.tensor(pq, dtype=torch.float32),
                        )
                    else:
                        yield (torch.tensor(X_blk[j], dtype=torch.float32), y_t)


# -----------------------------------------------------------------------------
# SPLITS + LABEL UTILITIES
# -----------------------------------------------------------------------------

def _embargo_bars(args) -> int:
    """A-H3: embargo gap (in samples) that must separate train from val so a
    training sample's forward-looking label cannot peek into the validation set.

    If validation.embargo_bars is set in config/run.yaml, use that value directly.
    Otherwise compute dynamically: seq_len + lookahead_bars + execution_delay_bars.
    """
    cfg_embargo = getattr(args, "validation_embargo_bars", None)
    if cfg_embargo is not None:
        return max(1, int(cfg_embargo))
    seq_len   = int(getattr(args, "seq_len", 60) or 60)
    lookahead = int(getattr(args, "lookahead_bars", LABELING.get("lookahead_bars", 15)))
    delay     = int(getattr(args, "execution_delay_bars", 1) or 0)
    return max(1, seq_len + lookahead + delay)


def _purge_bars(args) -> int:
    """Purge zone: training samples within this window of validation are dropped
    to prevent feature overlap (rolling windows extending into val period)."""
    cfg_purge = getattr(args, "validation_purge_bars", None)
    if cfg_purge is not None:
        return max(0, int(cfg_purge))
    # Default: use seq_len as purge if not configured
    seq_len = int(getattr(args, "seq_len", 60) or 60)
    return max(0, seq_len)


def _validation_method(args) -> str:
    """Read validation.method from config (default: purged_embargo)."""
    method = getattr(args, "validation_method", None)
    if method:
        return str(method).lower()
    return "purged_embargo"


def _embargo_split(n_samples: int, val_split: float, embargo: int, purge: int = 0,
                   method: str = "purged_embargo") -> Tuple[np.ndarray, np.ndarray]:
    """A-H3: chronological train/val split with embargo and optional purge gap.

    Val is the most-recent `val_split` fraction; the `embargo` samples immediately
    before the val window are DROPPED from train to prevent label leakage.
    If method == "purged_embargo", an additional `purge` samples are dropped to
    prevent feature-window overlap (rolling features extending into val period).
    """
    val_split = min(max(float(val_split), 0.0), 0.9)
    val_n     = int(n_samples * val_split)
    val_start = max(0, n_samples - val_n)
    if method == "purged_embargo":
        train_end = max(0, val_start - int(embargo) - int(purge))
    else:
        train_end = max(0, val_start - int(embargo))
    train_idx = np.arange(0, train_end, dtype=np.int64)
    val_idx   = np.arange(val_start, n_samples, dtype=np.int64)
    return train_idx, val_idx


def _three_way_split(n_samples: int, val_split: float, tune_split: float,
                     embargo: int, purge: int = 0,
                     method: str = "purged_embargo") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three-way chronological split: train | embargo | val (early stop) | embargo | tune_eval.

    Prevents data leakage between:
    - val set (used ONLY for early stopping)
    - tune_eval set (used ONLY for auto-tune hyperparameter decisions)

    Returns (train_idx, val_idx, tune_idx).
    """
    val_split = min(max(float(val_split), 0.0), 0.5)
    tune_split = min(max(float(tune_split), 0.0), 0.3)

    tune_n = int(n_samples * tune_split)
    val_n = int(n_samples * val_split)

    # tune_eval is the most recent chunk
    tune_start = max(0, n_samples - tune_n)
    # val is the chunk before tune_eval (with embargo between)
    val_end = max(0, tune_start - int(embargo))
    val_start = max(0, val_end - val_n)
    # train is everything before val (with embargo+purge between)
    if method == "purged_embargo":
        train_end = max(0, val_start - int(embargo) - int(purge))
    else:
        train_end = max(0, val_start - int(embargo))

    train_idx = np.arange(0, train_end, dtype=np.int64)
    val_idx = np.arange(val_start, val_end, dtype=np.int64)
    tune_idx = np.arange(tune_start, n_samples, dtype=np.int64)
    return train_idx, val_idx, tune_idx


def walk_forward_splits(n_samples: int, n_folds: int, embargo: int, purge: int = 0,
                        method: str = "purged_embargo") -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window walk-forward: each fold trains on [0, val_start - embargo - purge),
    validates on [val_start, val_end). Prevents overlap leakage via embargo + purge.
    """
    if n_samples < max(embargo + purge + n_folds * 2, 500):
        # A-H3: small-data fallback must still embargo+purge the train/val boundary
        return [_embargo_split(n_samples, float(TRAINING.get("val_split", 0.2)), embargo, purge, method)]
    edges = np.linspace(0, n_samples, n_folds + 2, dtype=np.int64)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        va, vb = int(edges[k + 1]), int(edges[k + 2])
        if method == "purged_embargo":
            train_end = max(0, va - int(embargo) - int(purge))
        else:
            train_end = max(0, va - int(embargo))
        tr = np.arange(0, train_end, dtype=np.int64)
        va_idx = np.arange(va, vb, dtype=np.int64)
        if len(tr) < 100 or len(va_idx) < 10:
            continue
        out.append((tr, va_idx))
    if not out:
        # A-H3: embargoed+purge fallback (was a plain 80/20 split with no embargo).
        return [_embargo_split(n_samples, float(TRAINING.get("val_split", 0.2)), embargo, purge, method)]
    return out


def _load_diff_array(cache_path: str, n_samples: int) -> Optional[np.ndarray]:
    """
    B: Load the full per-sample difficulty array (uint8) from cache.
    Returns None if the sidecar does not exist (graceful fallback -> train on all).
    """
    if ZARR and str(cache_path).endswith(".zarr") and Path(cache_path).is_dir():
        try:
            z = _zarr_open_group(cache_path, mode="r")
            if "diff" in z:
                return np.asarray(z["diff"][:n_samples], dtype=np.uint8)
        except Exception:
            pass
    np_path = _diff_path(cache_path)
    if Path(np_path).exists():
        try:
            return np.load(np_path, mmap_mode="r").astype(np.uint8)[:n_samples]
        except Exception:
            pass
    return None


def _load_feature_schema(cache_path: str, n_features: int) -> Optional[list[str]]:
    """Load the ordered feature schema saved next to a processed cache."""
    schema_path = Path(str(cache_path) + "_feature_schema.json")
    if not schema_path.exists():
        return None
    try:
        import json
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        if isinstance(schema, list) and len(schema) == int(n_features):
            return [str(c) for c in schema]
    except Exception:
        pass
    return None


def _coerce_auto_int(value, auto_value: int, *, minimum: int = 1) -> int:
    """Return an integer config value, accepting 'auto' as a computed default."""
    if value is None:
        return max(minimum, int(auto_value))
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"", "auto", "none"}:
            return max(minimum, int(auto_value))
        value = raw
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return max(minimum, int(auto_value))


def _make_pretrain_span_plan(
    n_total: int,
    n_windows: int,
    *,
    diff: Optional[np.ndarray] = None,
    max_spans: int = 8,
    rng: Optional[np.random.Generator] = None,
) -> list[tuple[int, int]]:
    """
    Plan chunk-friendly pretrain reads as multiple contiguous spans.

    If difficulty labels are available, starts are selected across difficulty
    buckets before falling back to timeline-spread spans. The return value is a
    list of (start, length) slices sorted by start for sequential-ish reads.
    """
    n_total = max(0, int(n_total))
    n_windows = min(max(1, int(n_windows)), n_total) if n_total else 0
    if n_windows <= 0:
        return []
    rng = rng or np.random.default_rng()
    max_spans = max(1, min(int(max_spans), n_windows, n_total))
    span_len = max(1, int(math.ceil(n_windows / max_spans)))

    starts: list[int] = []
    if diff is not None:
        try:
            d = np.asarray(diff[:n_total], dtype=np.uint8)
            buckets = [np.flatnonzero(d == level) for level in (0, 1, 2)]
            buckets = [b for b in buckets if len(b)]
            if buckets:
                while len(starts) < max_spans:
                    bucket = buckets[len(starts) % len(buckets)]
                    anchor = int(bucket[int(rng.integers(0, len(bucket)))])
                    starts.append(max(0, min(anchor - span_len // 2, n_total - span_len)))
        except Exception:
            starts = []

    if not starts:
        if max_spans == 1:
            starts = [int(rng.integers(0, max(1, n_total - span_len + 1)))]
        else:
            edges = np.linspace(0, max(0, n_total - span_len), num=max_spans, dtype=int)
            jitter = max(1, span_len // 2)
            starts = [
                int(np.clip(edge + int(rng.integers(-jitter, jitter + 1)), 0, max(0, n_total - span_len)))
                for edge in edges
            ]

    spans: list[tuple[int, int]] = []
    remaining = n_windows
    for start in sorted(dict.fromkeys(starts)):
        if remaining <= 0:
            break
        length = min(span_len, remaining, n_total - start)
        if length > 0:
            spans.append((int(start), int(length)))
            remaining -= length

    while remaining > 0:
        length = min(span_len, remaining)
        start = int(rng.integers(0, max(1, n_total - length + 1)))
        spans.append((start, length))
        remaining -= length

    return sorted(spans, key=lambda x: x[0])


def _read_pretrain_spans(
    x_reader,
    y_reader,
    spans: list[tuple[int, int]],
    *,
    seq_len: int,
    n_features: int,
    progress_desc: str = "[Pretrain] Loading spans",
) -> tuple[np.ndarray, np.ndarray]:
    total = int(sum(length for _, length in spans))
    w_out = np.zeros((total, int(seq_len), int(n_features)), dtype=np.float32)
    y_out = np.zeros(total, dtype=np.float32)
    pos = 0
    read_step = max(1, int(PRETRAIN.get("read_windows", 64)))
    with _pbar(total=total, desc=progress_desc, unit="win") as pb:
        for start, length in spans:
            span_end = start + length
            cursor = start
            while cursor < span_end:
                end = min(span_end, cursor + read_step)
                chunk_len = end - cursor
                w_chunk = _crop_to_seq_len(np.asarray(x_reader[cursor:end]), seq_len)
                np.nan_to_num(w_chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                w_out[pos:pos + chunk_len] = w_chunk
                y_chunk = np.asarray(y_reader[cursor:end], dtype=np.float32)
                np.nan_to_num(y_chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                y_out[pos:pos + chunk_len] = y_chunk
                pos += chunk_len
                cursor = end
                pb.update(chunk_len)
                if _TRAIN_LOGGER:
                    _TRAIN_LOGGER.heartbeat()
    return w_out, y_out


def _select_pretrain_trainer_class(method: str, regime_aware: bool):
    method = _normalize_pretrain_method(method)
    if method == "masked":
        return MaskedReconstructionTrainer
    if method == "tscl":
        return RegimeAwareTSCLTrainer if regime_aware else TSCLTrainer
    if method == "vae":
        return VAESeqTrainer
    if method == "cluster":
        return ClusterContrastiveTrainer
    if method == "forecast":
        return ForecastPretextTrainer
    if method == "drift":
        return DriftContrastiveTrainer
    return BYOLTrainer


def _read_y_cls_indices(cache_path: str, indices: np.ndarray, chunk: int = 500_000) -> Optional[np.ndarray]:
    """Read direction-class sidecar; None if missing (legacy caches)."""
    if ZARR and cache_path.endswith(".zarr") and Path(cache_path).is_dir():
        z = _zarr_open_group(cache_path, mode="r")
        if "y_cls" not in z:
            return None
        y = z["y_cls"]
        parts: list[np.ndarray] = []
        for s in range(0, len(indices), chunk):
            sl = indices[s : s + chunk]
            parts.append(np.asarray(y.oindex[sl]))
        return np.concatenate(parts) if parts else np.array([])
    ym = Path(_y_cls_path(cache_path))
    if not ym.exists():
        return None
    arr = np.load(str(ym), mmap_mode="r")
    parts = [np.asarray(arr[indices[s : s + chunk]]) for s in range(0, len(indices), chunk)]
    return np.concatenate(parts) if parts else np.array([])


def _read_pq_indices(cache_path: str, indices: np.ndarray, chunk: int = 500_000) -> Optional[np.ndarray]:
    if ZARR and cache_path.endswith(".zarr") and Path(cache_path).is_dir():
        z = _zarr_open_group(cache_path, mode="r")
        if "pq" not in z:
            return None
        pq = z["pq"]
        parts: list[np.ndarray] = []
        for s in range(0, len(indices), chunk):
            sl = indices[s : s + chunk]
            parts.append(np.asarray(pq.oindex[sl]))
        return np.concatenate(parts) if parts else np.array([])
    pm = Path(_pq_path(cache_path))
    if not pm.exists():
        return None
    arr = np.load(str(pm), mmap_mode="r")
    parts = [np.asarray(arr[indices[s : s + chunk]]) for s in range(0, len(indices), chunk)]
    return np.concatenate(parts) if parts else np.array([])


def _read_y_indices(cache_path: str, indices: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    parts: list[np.ndarray] = []
    if ZARR and cache_path.endswith(".zarr") and Path(cache_path).is_dir():
        z = _zarr_open_group(cache_path, mode="r")
        y = z["y"]
        for s in range(0, len(indices), chunk):
            sl = indices[s : s + chunk]
            parts.append(np.asarray(y.oindex[sl]))   # oindex = fancy/out-of-order indexing
    else:
        ym = np.load(_y_path(cache_path), mmap_mode="r")
        for s in range(0, len(indices), chunk):
            sl = indices[s : s + chunk]
            parts.append(np.asarray(ym[sl]))
    y = np.concatenate(parts) if parts else np.array([])
    return sanitize_array(y, context="cached labels")


def _class_weights_tensor(
    cache_path: str, train_idx: np.ndarray, device: "torch.device", max_samples: int = 2_000_000,
    use_direction_sidecar: bool = False,
) -> torch.Tensor:
    prior = _class_prior_array(
        cache_path,
        train_idx,
        max_samples=max_samples,
        use_direction_sidecar=use_direction_sidecar,
    )
    inv = 1.0 / np.clip(prior, 1e-6, None)
    weights = inv / max(float(inv.mean()), 1e-6)
    weights = np.clip(weights, 0.85, 1.15).astype(np.float32)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _class_prior_array(
    cache_path: str, train_idx: np.ndarray, max_samples: int = 2_000_000,
    use_direction_sidecar: bool = False,
) -> np.ndarray:
    if len(train_idx) > max_samples:
        sub = np.random.choice(train_idx, max_samples, replace=False)
    else:
        sub = train_idx
    sub = np.sort(sub)
    if use_direction_sidecar:
        y_raw = _read_y_cls_indices(cache_path, sub)
        if y_raw is None:
            y_raw = _read_y_indices(cache_path, sub)
    else:
        y_raw = _read_y_indices(cache_path, sub)

    # Round to the nearest integer label and cast to int8 so sklearn
    # never hits float32 vs float64 comparison issues.
    y = np.round(y_raw.astype(np.float64)).astype(np.int8)
    # Keep only the three valid direction labels; drop any noise values.
    y = y[np.isin(y, [-1, 0, 1])]

    counts = np.ones(3, dtype=np.float64)  # Laplace smoothing keeps absent classes finite
    for label, count in zip(*np.unique(y, return_counts=True)):
        idx = int(label) + 1
        if 0 <= idx < 3:
            counts[idx] += float(count)
    prior = counts / max(float(counts.sum()), 1.0)
    return prior.astype(np.float32)


def _class_prior_tensor(
    cache_path: str, train_idx: np.ndarray, device: "torch.device",
    use_direction_sidecar: bool = False,
) -> torch.Tensor:
    prior = _class_prior_array(
        cache_path, train_idx, use_direction_sidecar=use_direction_sidecar,
    )
    return torch.tensor(prior, dtype=torch.float32, device=device)


def _class_counts_from_y_cls(cache_path: str, indices: np.ndarray) -> dict:
    """Return S/H/B counts and shares from authoritative y_cls sidecar."""
    idx = np.asarray(indices, dtype=np.int64)
    y_raw = _read_y_cls_indices(cache_path, np.sort(idx))
    if y_raw is None:
        raise RuntimeError("[DirectionPreflight] Cache is missing y_cls direction sidecar.")
    y = np.round(np.asarray(y_raw, dtype=np.float64)).astype(np.int8)
    invalid = y[~np.isin(y, [-1, 0, 1])]
    counts = np.zeros(3, dtype=np.int64)
    for label, count in zip(*np.unique(y[np.isin(y, [-1, 0, 1])], return_counts=True)):
        counts[int(label) + 1] = int(count)
    total = max(1, int(counts.sum()))
    return {
        "counts": [int(x) for x in counts.tolist()],
        "shares": [float(x) / total for x in counts.tolist()],
        "invalid_count": int(len(invalid)),
        "total": int(total),
    }


def _balanced_direction_indices(
    cache_path: str,
    indices: np.ndarray,
    *,
    total_samples: Optional[int] = None,
    seed: int = 1337,
) -> np.ndarray:
    """Build an approximately class-balanced index list using y_cls labels."""
    idx = np.asarray(indices, dtype=np.int64)
    if len(idx) == 0:
        return idx
    y_raw = _read_y_cls_indices(cache_path, np.sort(idx))
    if y_raw is None:
        raise RuntimeError("[DirectionBalance] Cannot balance batches without y_cls sidecar.")
    sorted_idx = np.sort(idx)
    y = np.round(np.asarray(y_raw, dtype=np.float64)).astype(np.int8)
    rng = np.random.default_rng(int(seed))
    buckets = []
    for label in (-1, 0, 1):
        bucket = sorted_idx[y == label]
        if len(bucket) == 0:
            raise RuntimeError(f"[DirectionBalance] Missing class {label} in training fold.")
        buckets.append(bucket)
    if total_samples is None:
        per_class = min(len(b) for b in buckets)
    else:
        per_class = max(1, int(total_samples) // 3)
    parts = [
        rng.choice(bucket, size=per_class, replace=(len(bucket) < per_class))
        for bucket in buckets
    ]
    out = np.concatenate(parts).astype(np.int64)
    rng.shuffle(out)
    return out


def _direction_preflight(cache_path: str, train_idx: np.ndarray, val_idx: np.ndarray, args) -> dict:
    """Hard gate before supervised direction training starts."""
    snap = _cache_length_snapshot(cache_path)
    required = ("zarr_X", "zarr_y", "zarr_y_cls", "zarr_pq", "zarr_diff", "zarr_close", "zarr_atr", "zarr_spread")
    if snap and any(k.startswith("zarr_") for k in snap):
        missing = [k for k in required if k not in snap]
        if missing:
            raise RuntimeError(f"[DirectionPreflight] Missing cache arrays: {missing}")
        x_len = int(snap["zarr_X"])
        bad = {k: int(v) for k, v in snap.items() if k.startswith("zarr_") and int(v) != x_len}
        if bad:
            raise RuntimeError(f"[DirectionPreflight] Cache array length mismatch: X={x_len}, bad={bad}")

    train_stats = _class_counts_from_y_cls(cache_path, train_idx)
    val_stats = _class_counts_from_y_cls(cache_path, val_idx)
    min_share = float(getattr(args, "direction_min_true_class_share", 0.15))
    for split, stats in (("train", train_stats), ("val", val_stats)):
        if stats["invalid_count"]:
            raise RuntimeError(f"[DirectionPreflight] {split} y_cls has {stats['invalid_count']} invalid labels.")
        if min(stats["shares"]) < min_share:
            raise RuntimeError(
                f"[DirectionPreflight] {split} class prior too thin: "
                f"S/H/B={stats['shares']} min_required={min_share}"
            )

    forced = torch.tensor([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]])
    if forced.argmax(-1).tolist() != [0, 1, 2]:
        raise RuntimeError("[DirectionPreflight] Forced logit class order is broken.")
    report = {"train": train_stats, "val": val_stats, "class_order": ["Sell", "Hold", "Buy"]}
    print(
        "[DirectionPreflight] PASS | "
        f"train S/H/B={train_stats['shares'][0]:.3f}/{train_stats['shares'][1]:.3f}/{train_stats['shares'][2]:.3f} | "
        f"val S/H/B={val_stats['shares'][0]:.3f}/{val_stats['shares'][1]:.3f}/{val_stats['shares'][2]:.3f}"
    )
    return report


def _direction_recall_from_confusion(confusion: list[list[int]]) -> list[float]:
    recalls = []
    for cls_idx in range(3):
        denom = max(1, int(sum(confusion[cls_idx])))
        recalls.append(float(confusion[cls_idx][cls_idx]) / denom)
    return recalls


def _direction_gate_failed(diag: dict, args) -> tuple[bool, str]:
    pred = [int(x) for x in diag.get("pred", [0, 0, 0])]
    total = max(1, sum(pred))
    shares = [x / total for x in pred]
    recalls = [float(x) for x in diag.get("recall", [0.0, 0.0, 0.0])]
    min_pred = float(getattr(args, "direction_min_pred_class_share", 0.05))
    max_pred = float(getattr(args, "direction_max_pred_class_share", 0.80))
    min_recall = float(getattr(args, "direction_min_recall", 0.001))
    if min(shares) < min_pred:
        return True, f"min_pred_share {min(shares):.4f} < {min_pred:.4f}"
    if max(shares) > max_pred:
        return True, f"max_pred_share {max(shares):.4f} > {max_pred:.4f}"
    if min(recalls) < min_recall:
        return True, f"min_recall {min(recalls):.4f} < {min_recall:.4f}"
    return False, "ok"


def _write_class_balance_failure(run_name: str, model_name: str, epoch: int, diag: dict, reason: str) -> Path:
    out_dir = Path("logs/tests")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"class_balance_failure_{_slug_part(run_name, 120)}_{_slug_part(model_name, 40)}_ep{epoch+1}.json"
    payload = {
        "run_name": run_name,
        "model": model_name,
        "epoch": int(epoch + 1),
        "reason": reason,
        "diagnostics": diag,
        "class_order": ["Sell", "Hold", "Buy"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _safe_save_json(payload, out)
    return out


def _direction_probe(
    model,
    cache_path: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args,
    device,
    *,
    model_name: str,
    n_features: int,
    amp_dtype: "torch.dtype",
) -> dict:
    """Short balanced probe that must pass before full supervised training."""
    if not bool(getattr(args, "direction_probe", True)):
        return {"enabled": False, "passed": True}
    epochs = max(1, int(getattr(args, "direction_probe_epochs", 2)))
    samples = max(96, int(getattr(args, "direction_probe_samples", 4096)))
    seed = int(getattr(args, "seed", 1337))
    probe_train_idx = _balanced_direction_indices(cache_path, train_idx, total_samples=samples, seed=seed)
    probe_val_idx = _balanced_direction_indices(cache_path, val_idx, total_samples=max(96, samples // 2), seed=seed + 17)
    train_ds = ZarrStreamDataset(cache_path, probe_train_idx, shuffle_chunks=True, multitask_targets=True)
    val_ds = ZarrStreamDataset(cache_path, probe_val_idx, shuffle_chunks=False, multitask_targets=True)
    bs = min(max(32, int(getattr(args, "batch_size", 128))), 256)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=max(float(getattr(args, "lr", 1e-4)), 1e-4), weight_decay=0.0)
    print(f"[DirectionProbe] START | samples={len(probe_train_idx):,} val={len(probe_val_idx):,} epochs={epochs}")
    last = {}
    classification = True
    for ep in range(epochs):
        tl = train_epoch(
            model, train_dl, opt, crit, GradScaler(enabled=False), device,
            use_amp=False, classification=classification,
            grad_clip=float(getattr(args, "grad_clip", 1.0)),
            amp_dtype=amp_dtype, seq_len=int(getattr(args, "seq_len", 80)),
            multitask=True, epoch=ep, direction_only=True,
        )
        vl, da, _ = validate_epoch(
            model, val_dl, crit, device, classification,
            amp=False, amp_dtype=amp_dtype,
            seq_len=int(getattr(args, "seq_len", 80)), multitask=True,
            direction_only=True,
        )
        diag = getattr(validate_epoch, "last_class_diag", {})
        failed, reason = _direction_gate_failed(diag, args)
        last = {"train_loss": float(tl), "val_loss": float(vl), "dir_acc": float(da), "diag": diag, "failed": failed, "reason": reason}
        print(
            f"[DirectionProbe] Epoch {ep+1}/{epochs} train={tl:.4f} val={vl:.4f} acc={da:.4f} "
            f"pred={diag.get('pred')} recall={[round(x, 4) for x in diag.get('recall', [])]} reason={reason}"
        )
    if last.get("failed", True):
        out = _write_class_balance_failure(getattr(args, "run_name", "direction_probe"), model_name, 0, last.get("diag", {}), last.get("reason", "probe_failed"))
        raise RuntimeError(f"[DirectionProbe] FAILED: {last.get('reason')} | diagnostics -> {out}")
    print("[DirectionProbe] PASS")
    return {"enabled": True, "passed": True, **last}


@torch.no_grad()
def _init_multitask_direction_bias(model: "nn.Module", class_prior: "torch.Tensor") -> None:
    """Start the direction head from the fold label prior instead of a random class bias."""
    target = class_prior.detach().float().cpu().clamp_min(1e-6)
    target = target / target.sum().clamp_min(1e-6)
    bias = target.log()
    modules = [model]
    if isinstance(model, nn.DataParallel):
        modules.append(model.module)
    for root in modules:
        mt_head = getattr(root, "mt_head", None)
        direction = getattr(mt_head, "direction", None)
        if direction is None:
            continue
        for layer in reversed(list(direction.modules())):
            if isinstance(layer, nn.Linear) and layer.out_features == 3 and layer.bias is not None:
                layer.bias.copy_(bias.to(layer.bias.device, dtype=layer.bias.dtype))
                return


def labels_to_class_index(yb: "torch.Tensor") -> "torch.Tensor":
    """Map {-1,0,+1} direction labels to CE indices {0,1,2}."""
    yb = torch.nan_to_num(yb.float(), nan=0.0, posinf=1.0, neginf=-1.0)
    return (yb + 1.0).round().long().clamp(0, 2)


def _direction_class_index(

    yb: "torch.Tensor",

    y_cls: Optional["torch.Tensor"] = None,

    *,

    classification: bool = True,

) -> "torch.Tensor":

    """Return class indices, preferring the explicit direction sidecar.



    For rl_reward labels, ``yb`` is a continuous reward. Rounding it as if it

    were {-1, 0, +1} can collapse classification training and validation. The

    y_cls sidecar is the authoritative direction target when present.

    """

    if y_cls is not None:

        return labels_to_class_index(y_cls)

    if classification:

        return labels_to_class_index(yb)

    return _reward_to_class_index(yb)





def _reward_to_class_index(y_reward: "torch.Tensor", hold_eps: float = 0.5) -> "torch.Tensor":
    """Fallback when y_cls sidecar is absent: threshold continuous rewards to classes."""
    r = torch.nan_to_num(y_reward.float(), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
    cls = torch.ones(r.shape[0], dtype=torch.long, device=r.device)
    cls[r > hold_eps] = 2
    cls[r < -hold_eps] = 0
    return cls


def _match_target_shape(pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """Return target reshaped to pred for scalar regression heads."""
    target = torch.nan_to_num(target.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if pred.shape == target.shape:
        return target
    if pred.ndim == 2 and pred.shape[-1] == 1 and target.ndim == 1:
        return target.unsqueeze(-1)
    return target


def _gradients_are_finite(model: "nn.Module") -> bool:
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


@torch.no_grad()
def _recover_nonfinite_training_state(model: "nn.Module", opt: "torch.optim.Optimizer") -> None:
    for p in model.parameters():
        if _is_uninitialized_parameter(p):
            continue
        if not torch.isfinite(p).all():
            p.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    for state in opt.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                value.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    opt.zero_grad(set_to_none=True)


# -----------------------------------------------------------------------------
# MODEL FACTORY
# -----------------------------------------------------------------------------

def _multitask_head_in(model_name: str, args, n_features: int) -> int:
    """
    Return the dimensionality of each backbone's pre-head hidden state.
    This is the input size for MultiTaskWrapper / MultiTaskHead.

    Derived from each architecture's forward() ΓÇö the tensor fed into self.head
    before the MultiTaskWrapper replaces it with nn.Identity().
    """
    m = model_name.lower()
    if m == "tft":         return args.hidden_size
    # iTransformer: out.reshape(B, F * d_model) -> large, wrapper auto-projects
    if m == "transformer": return args.d_model * n_features
    # HAELT: cat([lstm_feat, trf_feat]) of size lstm_hidden + d_model
    if m == "haelt":       return (args.hidden_size // 2) + (args.d_model // 2)
    if m == "mamba":       return args.d_model
    # GNNFromSequence: h.reshape(B, hidden * n_nodes)
    if m == "gnn":         return args.hidden_size * 6
    if m == "expert":      return args.d_model
    return args.hidden_size


def _format_param_count(model: nn.Module) -> str:
    total = 0
    skipped = 0
    for p in model.parameters():
        try:
            total += p.numel()
        except ValueError:
            skipped += 1
    suffix = "+" if skipped else ""
    return f"{total / 1e6:.2f}{suffix}M"


def _is_uninitialized_parameter(param: object) -> bool:
    try:
        from torch.nn.parameter import UninitializedParameter

        return isinstance(param, UninitializedParameter)
    except Exception:
        try:
            param.numel()
            return False
        except ValueError:
            return True


def build_model(name: str, n_features: int, args) -> nn.Module:
    # -- Multi-pair embedding expansion ----------------------------------------
    n_pairs     = getattr(args, "_n_pairs", 1)
    f_per_pair  = getattr(args, "_f_per_pair", n_features)
    embed_dim   = getattr(args, "pair_embed_dim", 0)
    use_pair_emb = n_pairs > 1 and embed_dim > 0

    # Backbone input width when MultiPairWrapper is active:
    #   pairs_flat  = n_pairs * (f_per_pair + embed_dim)
    #   n_cross     = n_pairs*(n_pairs-1)//2
    #   n_inter     = 3*n_cross (RelMom+ShortCorr+LongCorr) + n_pairs (VolShare) + 2 (Disp+Conf)
    # The wrapper concatenates pairs_flat and cross features before handing off to backbone,
    # so the backbone must be built with the combined width.
    if use_pair_emb:
        _n_cross       = n_pairs * (n_pairs - 1) // 2
        _n_interaction = 3 * _n_cross + n_pairs + 2
        backbone_input = n_pairs * (f_per_pair + embed_dim) + _n_interaction
    else:
        backbone_input = n_features

    # Multitask wrapper adds its own 3-class head; base always uses nc=1
    multitask = getattr(args, "multitask", False)
    nc = 1 if multitask else (3 if args.loss == "cross_entropy" else 1)
    core_name = name.replace("baseline_", "")
    builders = {
        "tft":         lambda: TFTScalper(
                           input_size=backbone_input, hidden=args.hidden_size,
                           heads=min(8,args.nhead), lstm_layers=args.num_layers,
                           dropout=args.dropout, num_classes=nc),
        "transformer": lambda: iTransformerScalper(
                           input_size=backbone_input, seq_len=args.seq_len,
                           d_model=args.d_model, nhead=args.nhead,
                           num_layers=args.num_layers,
                           dim_ff=getattr(args, "dim_ff", args.d_model * 2),
                           dropout=args.dropout,
                           num_classes=nc),
        "haelt":       lambda: HAELTHybrid(
                           input_size=backbone_input, seq_len=args.seq_len,
                           lstm_hidden=args.hidden_size//2, d_model=args.d_model//2,
                           nhead=max(2,args.nhead//2), n_layers=args.num_layers,
                           dropout=args.dropout, num_classes=nc),
        "mamba":       lambda: MambaScalper(
                           input_size=backbone_input, d_model=args.d_model,
                           num_layers=args.num_layers, dropout=args.dropout,
                           num_classes=nc),
        "gnn":         lambda: GNNFromSequence(
                           input_size=backbone_input, hidden=args.hidden_size,
                           num_layers=args.num_layers, dropout=args.dropout,
                           n_nodes=6, num_classes=nc, nhead=min(4, args.nhead)),
        "expert":      lambda: EXPERTEncoder(
                           input_size=backbone_input, d_model=args.d_model,
                           nhead=args.nhead,
                           num_layers=args.num_layers, dropout=args.dropout,
                           num_classes=nc),
    }

    if core_name not in builders:
        raise ValueError(f"Unknown model: {core_name}. Available: {list(builders.keys())}")

    m = builders[core_name]()

    # Pair embedding wrapper (only when embed_dim > 0 and training on multiple pairs)
    if use_pair_emb:
        _cw_short = getattr(args, "corr_window",      20)
        _cw_long  = getattr(args, "corr_window_long", 60)
        _mw       = getattr(args, "momentum_window",  20)
        _feat_names = getattr(args, "_feat_names", None)
        _ri, _ai = _resolve_pair_feat_indices(_feat_names, f_per_pair)
        m = MultiPairWrapper(
            m,
            n_pairs=n_pairs, f_per_pair=f_per_pair, embed_dim=embed_dim,
            corr_window=_cw_short, corr_window_long=_cw_long, momentum_window=_mw,
            return_idx=_ri, atr_idx=_ai,
        )
        print(f"[Model] {name.upper()} | MultiPair wrapper "
              f"({n_pairs}P ├ù {f_per_pair}F + {embed_dim}E | "
              f"corr={_cw_short}/{_cw_long}bar mom={_mw}bar) | "
              f"{_format_param_count(m)} parameters")
    elif n_pairs > 1:
        print(f"[Model] {name.upper()} | {n_pairs} pairs ├ù {f_per_pair}F concatenated | "
              f"{_format_param_count(m)} parameters")

    if multitask:
        head_in = _multitask_head_in(core_name, args, backbone_input)
        m = MultiTaskWrapper(
            m, head_in=head_in,
            hidden=64, dropout=args.dropout,
            proj_threshold=1024, proj_to=256,
            force_project=(core_name == "transformer"),
        )
        print(f"[Model] {name.upper()} | MultiTask wrapper (head_in={head_in}) | "
              f"{_format_param_count(m)} parameters")
    elif n_pairs == 1:
        print(f"[Model] {name.upper()} | {_format_param_count(m)} parameters")
    return m


# -----------------------------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------------------------

_OVERCONF_PENALTY: Optional["OverconfidencePenalty"] = None  # D: set in supervised_train


# -----------------------------------------------------------------------------
# A: FEATURE STABILITY MONITORING
#
# Most training failures don't show up as loss spikes ΓÇö they show up as
# features whose distributions silently shift, causing gradients to chase
# moving targets.  The model "learns" the shift artefact rather than the
# underlying signal, producing good training metrics but poor live performance.
#
# FeatureStabilityMonitor tracks per-feature mean and std using an exponential
# moving average (EMA).  Each epoch:
#   1. Sample a batch from the training data
#   2. Compute per-feature mean / std across (B, T) positions
#   3. Compute shift score = |Deltamean| / (ema_std + eps) ΓÇö standard deviations shifted
#   4. Compute var  score  = |Deltastd|  / (ema_std + eps) ΓÇö variance change magnitude
#   5. Mark features as noisy (score > soft_threshold) or frozen (score > hard_threshold
#      for freeze_after consecutive epochs)
#   6. Output a float32 mask (1.0=stable, damping_factor=noisy, 0.0=frozen)
#   7. Apply mask to xb in train_epoch: xb = xb * mask ΓÇö unstable dims zeroed out
#
# Effect: gradients for frozen features are effectively zeroed (input = 0).
# Noisy features receive a reduced signal (input ├ù damping_factor).
# Once the distribution stabilises, the feature is automatically re-enabled.
# -----------------------------------------------------------------------------

class FeatureStabilityMonitor:
    """
    A: Training-time feature stability monitor.

    Tracks per-feature distribution drift via EMA mean/std.
    Outputs a stability mask applied to input batches in train_epoch.

    Args:
        n_features      : Number of input feature dimensions.
        ema_alpha       : EMA decay (0.9 = slow decay, 0.5 = fast adaptation).
        soft_threshold  : Shift score above which feature is marked noisy.
                          mask value = damping_factor (default 0.5).
        hard_threshold  : Shift score above which feature is immediately frozen.
                          mask value = 0.0.
        freeze_after    : Consecutive epochs above soft_threshold -> frozen.
        damping_factor  : Mask value for noisy (not yet frozen) features.
        warmup_epochs   : Epochs before monitoring starts (EMA warm-up).
        min_active_pct  : Never freeze more than (1 - min_active_pct) of features.
                          Prevents catastrophic feature collapse.
    """

    def __init__(
        self,
        n_features:     int,
        ema_alpha:      float = 0.90,
        soft_threshold: float = 2.0,
        hard_threshold: float = 4.0,
        freeze_after:   int   = 3,
        damping_factor: float = 0.50,
        warmup_epochs:  int   = 3,
        min_active_pct: float = 0.50,
    ):
        self.n          = n_features
        self.alpha      = ema_alpha
        self.soft_t     = soft_threshold
        self.hard_t     = hard_threshold
        self.freeze_af  = freeze_after
        self.damp       = damping_factor
        self.warmup     = warmup_epochs
        self.min_active = min_active_pct

        self._ema_mean  = np.zeros(n_features, dtype=np.float64)
        self._ema_std   = np.ones(n_features,  dtype=np.float64)
        self._initialized = False

        # Per-feature counters
        self._soft_streak = np.zeros(n_features, dtype=np.int32)   # epochs above soft_t
        self._frozen      = np.zeros(n_features, dtype=bool)       # permanently frozen

        # Per-epoch stats (for logging)
        self._last_shift_score = np.zeros(n_features, dtype=np.float64)
        self._epoch            = 0

    def update(self, xb: np.ndarray) -> None:
        """
        Update EMA stats with a batch sample.

        Args:
            xb: float32 array of shape (B, seq_len, n_features) or (B, n_features).
        """
        if xb.ndim == 3:
            flat = xb.reshape(-1, xb.shape[-1])   # (B*T, F)
        else:
            flat = xb

        batch_mean = flat.mean(axis=0).astype(np.float64)  # (F,)
        batch_std  = flat.std(axis=0).astype(np.float64)   # (F,)
        batch_std  = np.maximum(batch_std, 1e-8)

        if not self._initialized:
            self._ema_mean  = batch_mean.copy()
            self._ema_std   = batch_std.copy()
            self._initialized = True
            return

        # Shift scores before EMA update (compare new batch vs current EMA)
        mean_shift = np.abs(batch_mean - self._ema_mean) / (self._ema_std + 1e-8)
        std_shift  = np.abs(batch_std  - self._ema_std)  / (self._ema_std + 1e-8)
        shift_score = np.maximum(mean_shift, std_shift)
        self._last_shift_score = shift_score

        # Update EMA
        self._ema_mean = self.alpha * self._ema_mean + (1 - self.alpha) * batch_mean
        self._ema_std  = self.alpha * self._ema_std  + (1 - self.alpha) * batch_std

        self._epoch += 1
        if self._epoch <= self.warmup:
            return   # don't penalise during warm-up

        # Update instability streaks and frozen flags
        above_soft = shift_score > self.soft_t
        above_hard = shift_score > self.hard_t
        self._soft_streak[above_soft]  += 1
        self._soft_streak[~above_soft]  = 0

        # Freeze: hard threshold OR soft streak exceeded
        newly_frozen = above_hard | (self._soft_streak >= self.freeze_af)

        # Enforce min_active_pct ΓÇö never freeze too many features
        n_frozen = int(newly_frozen.sum())
        max_freeze = int(self.n * (1.0 - self.min_active))
        if n_frozen > max_freeze:
            # Keep only the worst max_freeze features frozen
            top_frozen = np.argsort(shift_score)[::-1][:max_freeze]
            mask_limit = np.zeros(self.n, dtype=bool)
            mask_limit[top_frozen] = True
            newly_frozen = newly_frozen & mask_limit

        self._frozen = newly_frozen

    def get_mask(self, device=None) -> "torch.Tensor":
        """
        Returns a float32 tensor of shape (n_features,):
          1.0 = stable      -> full signal
          damping_factor    -> noisy (above soft threshold but not frozen)
          0.0               -> frozen (consistently unstable)
        """
        mask = np.ones(self.n, dtype=np.float32)
        noisy = (self._last_shift_score > self.soft_t) & (~self._frozen)
        mask[noisy]       = self.damp
        mask[self._frozen] = 0.0
        t = torch.from_numpy(mask)
        return t.to(device) if device is not None else t

    def report(self) -> dict:
        """Return a summary dict for logging."""
        n_frozen = int(self._frozen.sum())
        n_noisy  = int(((self._last_shift_score > self.soft_t) & ~self._frozen).sum())
        top5_idx = np.argsort(self._last_shift_score)[::-1][:5].tolist()
        return {
            "feat_frozen":       n_frozen,
            "feat_noisy":        n_noisy,
            "feat_active":       self.n - n_frozen,
            "feat_max_shift":    float(self._last_shift_score.max()),
            "feat_mean_shift":   float(self._last_shift_score.mean()),
            "feat_top5_unstable": top5_idx,
        }

    def reset_frozen(self) -> None:
        """Unfreeze all features (e.g. when curriculum difficulty increases).

        Also resets shift scores so previously-frozen features return to mask=1.0
        rather than remaining soft-masked (0.5) due to stale high shift scores.
        """
        self._frozen[:] = False
        self._soft_streak[:] = 0
        self._last_shift_score[:] = 0.0

    def get_state(self) -> dict:
        """Return serializable state for checkpoint saving."""
        return {
            "ema_mean":    self._ema_mean.tolist(),
            "ema_std":     self._ema_std.tolist(),
            "soft_streak": self._soft_streak.tolist(),
            "frozen":      self._frozen.tolist(),
            "last_shift":  self._last_shift_score.tolist(),
            "initialized": self._initialized,
            "epoch":       self._epoch,
        }

    def load_state(self, state: dict) -> None:
        """Restore state from a checkpoint dict (backward-compatible)."""
        self._ema_mean         = np.array(state["ema_mean"],    dtype=np.float64)
        self._ema_std          = np.array(state["ema_std"],     dtype=np.float64)
        self._soft_streak      = np.array(state["soft_streak"], dtype=np.int32)
        self._frozen           = np.array(state["frozen"],      dtype=bool)
        self._last_shift_score = np.array(state.get("last_shift",
                                           [0.0] * self.n),    dtype=np.float64)
        self._initialized      = bool(state["initialized"])
        self._epoch            = int(state["epoch"])


# -----------------------------------------------------------------------------
# C: DIVERSITY FINE-TUNING
# After all models are individually trained, run a short joint pass that
# penalises correlated predictions across models with the same role.
#
# Why post-training (not during individual training):
#   During individual training each model only has its own output ΓÇö there are
#   no peer outputs to compare against.  A diversity fine-tuning phase loads
#   ALL trained checkpoints simultaneously, runs the same batch through every
#   model, and uses DiversityLoss to push same-role models' predictions apart
#   while keeping their task performance stable.
#
# Role assignments (from MODEL_ROLES):
#   mamba       -> fast_reaction   (two same-role pairs with transformer/tft -> context)
#   tft         -> context
#   haelt       -> confirmation
#   gnn         -> risk_modulation
#   transformer -> context
#   expert      -> confirmation
#
# Same-role pairs (tft+transformer, haelt+expert) receive a 2├ù diversity
# penalty to ensure they specialise rather than duplicate.
# -----------------------------------------------------------------------------

def run_diversity_finetune(
    checkpoint_dir: str,
    model_names:    list,
    cache_path:     str,
    n_features:     int,
    args,
    device:         "torch.device",
    epochs:         int  = 3,
    lr:             float = 1e-5,
    div_weight:     float = 0.10,
    same_role_mult: float = 2.0,
    batch_size:     int  = 512,
    max_batches:    int  = 200,
) -> None:
    """
    C: Joint diversity fine-tuning across all trained models.

    Loads each model's *_best.pt checkpoint, then for each batch:
      loss_i = task_loss_i  +  DiversityLoss(all model outputs)
    All models are updated simultaneously so the diversity gradient flows
    into each model while maintaining their individual task performance.

    Args:
        checkpoint_dir : Directory containing *_best.pt files.
        model_names    : List of model name strings to include.
        cache_path     : Zarr cache path (same as used during training).
        n_features     : Feature dimension.
        args           : Training args namespace (loss, seq_len, etc.).
        device         : torch.device.
        epochs         : Fine-tuning epochs (3 is usually sufficient).
        lr             : Learning rate (much lower than training ΓÇö fine-tuning).
        div_weight     : DiversityLoss weight multiplier.
        same_role_mult : Extra multiplier for same-role pairs.
        batch_size     : Batch size for joint forward pass.
        max_batches    : Max batches per epoch (limits GPU time).
    """
    ckpt_dir = Path(checkpoint_dir)
    classification = getattr(args, "loss", "cross_entropy") in ("cross_entropy", "multi_task", "asymmetric_directional")

    # -- Load checkpoints ------------------------------------------------------
    loaded_models  = {}
    loaded_roles   = []
    loaded_names   = []
    loaded_seq_lens = []
    for name in model_names:
        # Support both per-model subfolder layout (<base>/<model>/<model>_best.pt)
        # and the legacy flat layout (<base>/<model>_best.pt).
        ckpt_path = ckpt_dir / name / f"{name}_best.pt"
        if not ckpt_path.exists():
            ckpt_path = ckpt_dir / f"{name}_best.pt"
        if not ckpt_path.exists():
            print(f"  [DivFT] Skipping {name} ΓÇö checkpoint not found at {ckpt_path}")
            continue
        try:
            model_args = _model_build_args(args, name)
            m = build_model(name, n_features, model_args).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            
            try:
                m.load_state_dict(state, strict=True)
            except Exception:
                # If strict loading failed, load into backbone with an asserting
                # report so a near-empty load fails loudly (A-H2).
                core = m.backbone if hasattr(m, "backbone") else m
                _strict_load_report(core, state, f"DivFT:{name}", min_frac_loaded=0.6)

            m.train()
            loaded_models[name] = m
            loaded_roles.append(MODEL_ROLES.get(name, "unknown"))
            loaded_names.append(name)
            loaded_seq_lens.append(int(getattr(model_args, "seq_len", getattr(args, "seq_len", 0)) or 0))
        except Exception as e:
            print(f"  [DivFT] Could not load {name}: {e}")

    if len(loaded_models) < 2:
        print("  [DivFT] Need >=2 models for diversity fine-tuning ΓÇö skipping.")
        return

    print(f"\n[DivFT] Diversity fine-tuning: {loaded_names}")
    print(f"        Roles: {loaded_roles}")
    print(f"        epochs={epochs}  lr={lr}  div_weight={div_weight}  same_role_mult={same_role_mult}")

    # -- Shared optimizer across all models -----------------------------------
    all_params = []
    for m in loaded_models.values():
        all_params += list(m.parameters())
    opt = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-4)

    # -- Criterion (same as supervised training) -------------------------------
    if classification:
        crit = torch.nn.CrossEntropyLoss()
    else:
        crit = torch.nn.HuberLoss(delta=1.0)

    # -- Diversity loss -----------------------------------------------------
    div_loss_fn = DiversityLoss(
        weight=div_weight,
        same_role_mult=same_role_mult,
        roles=loaded_roles,
    ).to(device)

    # -- Data loader (val split ΓÇö fine-tune on held-out data only) ------------
    n_samples = int(_on_disk_sequence_count(cache_path) or 0)
    if n_samples <= 0:
        print("  [DivFT] Could not determine dataset size ΓÇö skipping.")
        return
    val_start = int(n_samples * 0.80)
    val_idx   = np.arange(val_start, n_samples)
    ds        = ZarrStreamDataset(cache_path, val_idx, shuffle_chunks=True)
    loader    = DataLoader(ds, batch_size=batch_size, shuffle=False,
                           num_workers=0, drop_last=True)

    model_list = list(loaded_models.values())
    list(loaded_models.keys())

    for ep in range(epochs):
        ep_task_loss = 0.0
        ep_div_loss  = 0.0
        n_batches    = 0

        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            xb, yb, y_cls_b, y_conf_b, _ = _unpack_batch(batch, device)
            xb, yb, y_cls_b, y_conf_b = _sanitize_batch_tensors(xb, yb, y_cls_b, y_conf_b)
            y_cls_idx = _direction_class_index(yb, y_cls_b, classification=classification)

            opt.zero_grad(set_to_none=True)

            # Forward all models on the same batch
            outputs = []
            task_loss = torch.tensor(0.0, device=device)
            for m, m_seq_len in zip(model_list, loaded_seq_lens):
                xb_m = _crop_to_seq_len(xb, m_seq_len)
                out = m(xb_m)
                outputs.append(out)
                # Per-model task loss
                if isinstance(out, tuple):
                    logits, ret_hat, conf = out
                    task_loss = task_loss + crit(logits, y_cls_idx)
                elif classification:
                    task_loss = task_loss + crit(out, y_cls_idx)
                else:
                    task_loss = task_loss + crit(out, _match_target_shape(out, yb))
            task_loss = task_loss / len(model_list)

            # Scalar outputs for diversity loss ΓÇö extract scalar per model per sample
            scalar_outs = []
            for out in outputs:
                if isinstance(out, tuple):
                    # regression head for diversity comparison
                    scalar_outs.append(out[1].reshape(-1))   # return_hat
                elif out.ndim == 2 and out.shape[-1] > 1:
                    # classification logits -> use argmax-weighted scalar
                    scalar_outs.append(out.softmax(-1)[:, -1] - out.softmax(-1)[:, 0])
                else:
                    scalar_outs.append(out.reshape(-1))

            diversity = div_loss_fn(scalar_outs)
            loss = task_loss + diversity
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            
            # Gradient Centralization
            for p in all_params:
                if p.grad is not None and p.grad.dim() > 1:
                    p.grad.sub_(p.grad.mean(dim=tuple(range(1, p.grad.dim())), keepdim=True))
                    
            opt.step()

            ep_task_loss += task_loss.item()
            ep_div_loss  += diversity.item()
            n_batches    += 1

        avg_t = ep_task_loss / max(n_batches, 1)
        avg_d = ep_div_loss  / max(n_batches, 1)
        print(f"  [DivFT] Epoch {ep+1}/{epochs} | task={avg_t:.5f}  diversity={avg_d:.5f}")

    # -- Save updated checkpoints ---------------------------------------------
    for name, m in loaded_models.items():
        ckpt_path = ckpt_dir / name / f"{name}_best.pt"
        if not ckpt_path.exists():
            ckpt_path = ckpt_dir / f"{name}_best.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt["model_state_dict"] = m.cpu().state_dict()
        ckpt["diversity_finetuned"] = True
        _safe_save(ckpt, ckpt_path)
        print(f"  [DivFT] Saved updated checkpoint: {ckpt_path.name}")


def _compute_loss(
    model_out, crit, yb, classification: bool,
    y_cls: Optional["torch.Tensor"] = None,
    y_conf: Optional["torch.Tensor"] = None,
    multitask: bool = False,
    direction_only: bool = False,
) -> "torch.Tensor":
    """
    Unified loss computation for both single-head and MultiTaskWrapper outputs.
    MultiTaskWrapper returns (direction_logits, return_hat, confidence) tuple.
    Single-head models return a scalar / 3-class logit tensor.

    D: OverconfidencePenalty is added when _OVERCONF_PENALTY is set, for
       regression heads only (classification confidence is handled by MultiTaskLoss).
    """
    if isinstance(model_out, tuple):
        logits, ret_hat, conf = model_out
        if direction_only:
            return crit(logits, _direction_class_index(yb, y_cls, classification=True))
        if multitask or not classification:
            y_cls_idx = _direction_class_index(

                yb, y_cls, classification=classification,

            )

            y_cont = _match_target_shape(ret_hat, yb)
            conf_tgt = y_conf if y_conf is not None else None
            return crit(logits, ret_hat, conf, y_cls_idx, y_cont, conf_tgt)
        return crit(logits, ret_hat, conf, _direction_class_index(yb, y_cls), yb)

    elif classification:
        return crit(model_out, _direction_class_index(yb, y_cls))

    else:
        yb = _match_target_shape(model_out, yb)
        try:
            base = crit(model_out, yb, weight=y_conf)
        except TypeError:
            base = crit(model_out, yb)
        if _OVERCONF_PENALTY is not None:
            return base + _OVERCONF_PENALTY(model_out, yb)
        return base


def _unpack_batch(batch, device):
    """Return (xb, yb, y_cls, y_conf, sample_idx) from 2- to 5-tuple batches.

    Always returns a 5-element tuple.  ``sample_idx`` is ``None`` when the
    loader does not carry index information.
    """
    n = len(batch) if isinstance(batch, (tuple, list)) else 1
    if n >= 5:
        xb, yb, y_cls, y_conf, sample_idx = batch[0], batch[1], batch[2], batch[3], batch[4]
        return (xb.to(device, non_blocking=True), yb.to(device, non_blocking=True),
                y_cls.to(device, non_blocking=True), y_conf.to(device, non_blocking=True),
                sample_idx.to(device, non_blocking=True))
    if n == 4:
        xb, yb, y_cls, y_conf = batch[0], batch[1], batch[2], batch[3]
        return (xb.to(device, non_blocking=True), yb.to(device, non_blocking=True),
                y_cls.to(device, non_blocking=True), y_conf.to(device, non_blocking=True),
                None)
    if n == 3:
        xb, yb, sample_idx = batch[0], batch[1], batch[2]
        return (xb.to(device, non_blocking=True), yb.to(device, non_blocking=True),
                None, None, sample_idx.to(device, non_blocking=True))
    xb, yb = batch[0], batch[1]
    return (xb.to(device, non_blocking=True), yb.to(device, non_blocking=True),
            None, None, None)


def _sanitize_batch_tensors(
    xb: "torch.Tensor",
    yb: "torch.Tensor",
    y_cls: Optional["torch.Tensor"],
    y_conf: Optional["torch.Tensor"],
) -> "tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]":
    """Clamp tensor batches before loss computation so one dirty sample cannot poison an epoch."""
    xb = torch.nan_to_num(xb.float(), nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
    yb = torch.nan_to_num(yb.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if y_cls is not None:
        y_cls = torch.nan_to_num(y_cls.float(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    if y_conf is not None:
        y_conf = torch.nan_to_num(y_conf.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return xb, yb, y_cls, y_conf


def train_epoch(
    model, loader, opt, crit, scaler_amp, device, use_amp, classification: bool,
    grad_clip: float = 1.0, pbar=None,
    amp_dtype: "torch.dtype" = torch.float32,
    thermal_limit: int = 83,
    feature_mask: Optional["torch.Tensor"] = None,
    scheduler=None,        # pass OneCycleLR here — stepped per optimizer update
    accum_steps: int = 1,  # gradient accumulation: effective_batch = batch × accum_steps
    seq_len: Optional[int] = None,
    multitask: bool = False,
    epoch: int = 0,
    teacher_model = None,
    distill_weight: float = 0.5,
    direction_only: bool = False,
    online_miner=None,     # Optional[OnlineHardExampleMiner] for per-sample tracking
):
    """
    One training epoch.
    ΓÇó BF16/FP16 autocast ΓÇö Tensor Cores on Ada RTX 40-series
    ΓÇó GradScaler only for FP16 (BF16 has full float32 range)
    ΓÇó Gradient accumulation ΓÇö averages gradients over accum_steps batches before stepping
    ΓÇó Per-step LR scheduling ΓÇö scheduler.step() called after each optimizer update
    ΓÇó Thermal throttle ΓÇö pauses 2 s if GPU > thermal_limit ┬░C
    ΓÇó OOM/NaN recovery ΓÇö skips batch, clears cache, flushes accumulated gradients
    """
    model.train()
    total = 0.0; n = 0; oom_skips = 0; nan_skips = 0
    use_fp16_scaler = scaler_amp.is_enabled()
    _thermal_check_freq = 50
    _n_batches = len(loader)   # works because ZarrStreamDataset has __len__

    _mask = feature_mask.to(device) if feature_mask is not None else None

    # Zero grad once at the start of the accumulation window
    opt.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        if batch_idx % _thermal_check_freq == 0:
            if _TRAIN_LOGGER: _TRAIN_LOGGER.heartbeat()
            _thermal_check(limit=thermal_limit)

        # True on the last batch of each accumulation window
        _do_step = ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == _n_batches)

        try:
            xb, yb, y_cls_b, y_conf_b, _batch_idx = _unpack_batch(batch, device)
            if seq_len is not None and xb.shape[1] > seq_len:
                xb = xb[:, -seq_len:, :]
            xb, yb, y_cls_b, y_conf_b = _sanitize_batch_tensors(xb, yb, y_cls_b, y_conf_b)

            if _mask is not None:
                xb = xb * _mask

            if use_amp and device.type == "cuda":
                with autocast("cuda", dtype=amp_dtype):
                    pred = model(xb)
                    loss = _compute_loss(
                        pred, crit, yb, classification,
                        y_cls=y_cls_b, y_conf=y_conf_b, multitask=multitask,
                        direction_only=direction_only,
                    )

                    # ── Online per-sample difficulty tracking ──────────────
                    if online_miner is not None and _batch_idx is not None:
                        try:
                            with torch.no_grad():
                                if isinstance(pred, tuple):
                                    pred_flat = pred[0]
                                else:
                                    pred_flat = pred
                                # Per-sample directional accuracy proxy
                                if classification or multitask:
                                    y_cls_idx = _direction_class_index(yb, y_cls_b, classification=True)
                                    pred_class = pred_flat.argmax(-1)
                                    per_sample = (pred_class != y_cls_idx).float()
                                else:
                                    per_sample = torch.abs(pred_flat.ravel() - yb.ravel())
                                online_miner.update_batch(
                                    _batch_idx.detach().cpu().numpy(),
                                    per_sample.detach().cpu().numpy(),
                                )
                        except Exception:
                            pass
                        finally:
                            try:
                                del _batch_idx, per_sample
                            except Exception:
                                pass
                    # ────────────────────────────────────────────────────────
                    if teacher_model is not None:
                        with torch.no_grad():
                            t_pred = teacher_model(xb)
                        if isinstance(pred, tuple):
                            p_out = pred[0]
                            t_out = t_pred[0] if isinstance(t_pred, tuple) else t_pred
                        else:
                            p_out = pred
                            t_out = t_pred if not isinstance(t_pred, tuple) else t_pred[0]
                        kd_loss = torch.nn.functional.mse_loss(p_out, t_out)
                        loss = (1.0 - distill_weight) * loss + distill_weight * kd_loss
                        
                    loss = loss / accum_steps   # scale for accumulation

                if not torch.isfinite(loss):
                    nan_skips += 1
                    _log_nan(batch_idx, epoch, nan_skips)
                    _recover_nonfinite_training_state(model, opt)
                    if nan_skips <= 3 or nan_skips % 10 == 0:
                        print(f"[Train] NaN/Inf loss at batch {batch_idx} (skip {nan_skips})")
                    opt.zero_grad(set_to_none=True)  # always flush partial grads (even mid-accum)
                    if pbar is not None:
                        pbar.update(1); pbar.set_postfix(loss="NaN-skip")
                    continue

                if use_fp16_scaler:
                    scaler_amp.scale(loss).backward()
                    if _do_step:
                        scaler_amp.unscale_(opt)
                        if not _gradients_are_finite(model):
                            nan_skips += 1
                            _log_nan(batch_idx, epoch, nan_skips)
                            _recover_nonfinite_training_state(model, opt)
                            scaler_amp.update()
                            if nan_skips <= 3 or nan_skips % 10 == 0:
                                print(f"[Train] NaN/Inf gradients at batch {batch_idx} (skip {nan_skips})")
                            if pbar is not None:
                                pbar.update(1); pbar.set_postfix(loss="NaN-grad-skip")
                            continue
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        
                        # Stability: monitor gradient norm
                        if batch_idx % 100 == 0:
                            total_norm = 0.0
                            for p in model.parameters():
                                if p.grad is not None:
                                    param_norm = p.grad.detach().data.norm(2)
                                    total_norm += param_norm.item() ** 2
                            total_norm = total_norm ** 0.5
                            if total_norm > 50.0:
                                print(f"[Stability] WARNING: High grad norm ({total_norm:.2f}) at batch {batch_idx}")

                        scaler_amp.step(opt)
                        scaler_amp.update()
                        opt.zero_grad(set_to_none=True)
                        if scheduler is not None: scheduler.step()
                else:
                    loss.backward()
                    if _do_step:
                        if not _gradients_are_finite(model):
                            nan_skips += 1
                            _log_nan(batch_idx, epoch, nan_skips)
                            _recover_nonfinite_training_state(model, opt)
                            if nan_skips <= 3 or nan_skips % 10 == 0:
                                print(f"[Train] NaN/Inf gradients at batch {batch_idx} (skip {nan_skips})")
                            if pbar is not None:
                                pbar.update(1); pbar.set_postfix(loss="NaN-grad-skip")
                            continue
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        
                        # Gradient Centralization
                        for p in model.parameters():
                            if p.grad is not None and p.grad.dim() > 1:
                                p.grad.sub_(p.grad.mean(dim=tuple(range(1, p.grad.dim())), keepdim=True))
                                
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        if scheduler is not None: scheduler.step()
            else:
                pred = model(xb)
                loss = _compute_loss(
                    pred, crit, yb, classification,
                    y_cls=y_cls_b, y_conf=y_conf_b, multitask=multitask,
                    direction_only=direction_only,
                )
                if teacher_model is not None:
                    with torch.no_grad():
                        t_pred = teacher_model(xb)
                    if isinstance(pred, tuple):
                        p_out = pred[0]
                        t_out = t_pred[0] if isinstance(t_pred, tuple) else t_pred
                    else:
                        p_out = pred
                        t_out = t_pred if not isinstance(t_pred, tuple) else t_pred[0]
                    kd_loss = torch.nn.functional.mse_loss(p_out, t_out)
                    loss = (1.0 - distill_weight) * loss + distill_weight * kd_loss

                loss = loss / accum_steps

                if not torch.isfinite(loss):
                    nan_skips += 1
                    _log_nan(batch_idx, epoch, nan_skips)
                    _recover_nonfinite_training_state(model, opt)
                    if nan_skips <= 3 or nan_skips % 10 == 0:
                        print(f"[Train] NaN/Inf loss at batch {batch_idx} (skip {nan_skips})")
                    if _do_step:
                        opt.zero_grad(set_to_none=True)
                    if pbar is not None:
                        pbar.update(1); pbar.set_postfix(loss="NaN-skip")
                    continue

                loss.backward()
                if _do_step:
                    if not _gradients_are_finite(model):
                        nan_skips += 1
                        _log_nan(batch_idx, epoch, nan_skips)
                        _recover_nonfinite_training_state(model, opt)
                        if nan_skips <= 3 or nan_skips % 10 == 0:
                            print(f"[Train] NaN/Inf gradients at batch {batch_idx} (skip {nan_skips})")
                        if pbar is not None:
                            pbar.update(1); pbar.set_postfix(loss="NaN-grad-skip")
                        continue
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    
                    # Gradient Centralization
                    for p in model.parameters():
                        if p.grad is not None and p.grad.dim() > 1:
                            p.grad.sub_(p.grad.mean(dim=tuple(range(1, p.grad.dim())), keepdim=True))
                            
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    if scheduler is not None: scheduler.step()

            loss_val = loss.item() * accum_steps   # un-scale for display
            total += loss_val; n += 1
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss_val:.5f}")

        except RuntimeError as e:
            is_oom = ("out of memory" in str(e).lower()) or ("cuda error" in str(e).lower())
            if not (device.type == "cuda" and is_oom):
                _log_error(f"[Train] Unexpected RuntimeError at batch {batch_idx}", e)
                raise
            oom_skips += 1
            opt.zero_grad(set_to_none=True)   # flush partial accum grads on OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            if pbar is not None:
                pbar.update(1); pbar.set_postfix(loss="OOM-skip")
            _log_oom(batch_idx, 0, oom_skips)
            if oom_skips <= 3 or oom_skips % 10 == 0:
                print(f"[Train] CUDA OOM batch {batch_idx}, skipped ({oom_skips} total). "
                      "Reduce --batch-size or --grad-accum-steps.")
            continue

        except Exception as e:
            _log_error(f"[Train] Unexpected error at batch {batch_idx}", e)
            raise

    if oom_skips:
        print(f"[Train] OOM summary: {oom_skips} batch(es) skipped this epoch.")
    if nan_skips:
        print(f"[Train] NaN summary: {nan_skips} batch(es) skipped this epoch.")
    return total / max(n, 1)


def build_criterion(
    args,
    device: "torch.device",
    cache_path: Optional[str] = None,
    train_idx: Optional[np.ndarray] = None,
):
    """
    Huber / asymmetric / directional_huber / sharpe_huber regression,
    weighted CE on {-1,0,+1}, or MultiTaskLoss.
    MultiTaskLoss is selected when --multitask is passed and combines:
      w_dir*CE(direction) + w_ret*Huber(return_hat) + w_conf*BCE(confidence)
    """
    multitask = getattr(args, "multitask", False)
    d = float(TRAINING.get("huber_delta", 1.0))

    if multitask:
        cw = None
        cp = None
        if cache_path is not None and train_idx is not None:
            cw = _class_weights_tensor(
                cache_path, train_idx, device, use_direction_sidecar=True,
            )
            cp = _class_prior_tensor(
                cache_path, train_idx, device, use_direction_sidecar=True,
            )
        loss_str = getattr(args, "loss", "huber").lower()
        w_sharpe = float(getattr(args, "sharpe_weight", 0.0)) if loss_str == "sharpe_huber" else 0.0
        sharpe_ann = _sharpe_ann_factor(args) if w_sharpe > 0 else 1.0
        return MultiTaskLoss(
            class_weights=cw,
            w_dir=1.0,
            w_ret=float(getattr(args, "mt_w_ret",  0.5)),
            w_conf=float(getattr(args, "mt_w_conf", 0.3)),
            huber_delta=d,
            class_balance_weight=float(getattr(args, "mt_class_balance_weight", 0.0)),

            entropy_weight=float(getattr(args, "mt_entropy_weight", 0.0)),

            direction_weight_floor=float(getattr(args, "mt_direction_weight_floor", 0.0)),

            focal_gamma=float(getattr(args, "mt_focal_gamma", 0.0)),
            class_prior=cp,
            w_sharpe=w_sharpe,
            sharpe_ann=sharpe_ann,

        ).to(device)

    if args.loss == "cross_entropy":
        if cache_path is None or train_idx is None:
            raise ValueError("cross_entropy requires cache_path and train_idx")
        w = _class_weights_tensor(

            cache_path, train_idx, device,

            use_direction_sidecar=(getattr(args, "label_method", "") == "rl_reward"),

        )

        return nn.CrossEntropyLoss(
            weight=w,
            label_smoothing=float(getattr(args, "label_smoothing", TRAINING.get("label_smoothing", 0.1))),
        )
    if args.loss == "asymmetric":
        sw = float(TRAINING.get("asymmetric_sign_weight", 2.0))
        return AsymmetricDirectionalLoss(delta=d, sign_weight=sw).to(device)
    if args.loss == "directional_huber":
        return DirectionalHuberLoss(
            delta=d,
            direction_weight=float(getattr(args, "direction_weight", 0.5)),
        ).to(device)
    if args.loss == "sharpe_huber":
        ann = _sharpe_ann_factor(args)
        return SharpeProxyLoss(
            delta=d,
            sharpe_weight=float(getattr(args, "sharpe_weight", 0.2)),
            ann=ann
        ).to(device)
    return HuberLoss(delta=d).to(device)


def _validation_class_diag(
    pred_counts: "torch.Tensor",
    true_counts: "torch.Tensor",
    confusion: "torch.Tensor",
    logits_sum: "torch.Tensor",
    probs_sum: "torch.Tensor",
    diag_true_counts: "torch.Tensor",
) -> dict:
    pred = [int(x) for x in pred_counts.detach().cpu().tolist()]
    true = [int(x) for x in true_counts.detach().cpu().tolist()]
    conf = [[int(v) for v in row] for row in confusion.detach().cpu().tolist()]
    recalls = _direction_recall_from_confusion(conf)
    denom = diag_true_counts.detach().float().clamp_min(1.0).view(3, 1)
    mean_logits = (logits_sum.detach().float() / denom).cpu().tolist()
    mean_probs = (probs_sum.detach().float() / denom).cpu().tolist()
    pred_total = max(1, sum(pred))
    true_total = max(1, sum(true))
    return {
        "pred": pred,
        "true": true,
        "pred_shares": [float(x) / pred_total for x in pred],
        "true_shares": [float(x) / true_total for x in true],
        "recall": recalls,
        "confusion": conf,
        "mean_logits_by_true_class": mean_logits,
        "mean_probs_by_true_class": mean_probs,
    }


@torch.no_grad()
def validate_epoch(model, loader, crit, device, classification: bool, pbar=None,
                   amp: bool = False, amp_dtype: torch.dtype = torch.float32,
                   seq_len: Optional[int] = None, multitask: bool = False,
                   feature_mask: Optional["torch.Tensor"] = None,
                   sharpe_ann_factor: Optional[float] = None,
                   direction_only: bool = False,
                   rl_mode: bool = False):

    model.eval()
    total = torch.zeros(1, device=device)
    correct = torch.zeros(1, device=device)
    n_acc = 0

    n_ret = 0

    oom_skips = 0
    nan_skips = 0
    valid_batches = 0
    r_sum = torch.zeros(1, device=device)
    r_sq_sum = torch.zeros(1, device=device)
    pred_counts = torch.zeros(3, device=device, dtype=torch.long)

    true_counts = torch.zeros(3, device=device, dtype=torch.long)
    confusion = torch.zeros((3, 3), device=device, dtype=torch.long)
    logits_sum = torch.zeros((3, 3), device=device)
    probs_sum = torch.zeros((3, 3), device=device)
    diag_true_counts = torch.zeros(3, device=device).clamp_min(0)

    
    # Heartbeat every N batches to keep watchdog alive during huge validation sets
    heartbeat_interval = 50 
    _mask = feature_mask.to(device) if feature_mask is not None else None

    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            try:
                xb, yb, y_cls_b, y_conf_b, _ = _unpack_batch(batch, device)
                if seq_len is not None and xb.shape[1] > seq_len:
                    xb = xb[:, -seq_len:, :]
                xb, yb, y_cls_b, y_conf_b = _sanitize_batch_tensors(xb, yb, y_cls_b, y_conf_b)
                if _mask is not None:

                    xb = xb * _mask

                if not torch.isfinite(xb).all() or not torch.isfinite(yb).all():
                    nan_skips += 1
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(loss="NaN-skip")
                    continue
                
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                    pred = model(xb)
                    y_cls_idx = _direction_class_index(

                        yb, y_cls_b, classification=classification,

                    )


                    if isinstance(pred, tuple):
                        logits, ret_hat, conf = pred
                        loss = _compute_loss(
                            pred, crit, yb, classification,
                            y_cls=y_cls_b, y_conf=y_conf_b, multitask=multitask,
                            direction_only=direction_only,
                        )
                        if not (torch.isfinite(loss) and torch.isfinite(logits).all() and torch.isfinite(ret_hat).all()):
                            nan_skips += 1
                            if pbar is not None:
                                pbar.update(1)
                                pbar.set_postfix(loss="NaN-skip")
                            continue
                        total += loss
                        try:
                            _miner = globals().get("_HardMiner")
                            if _miner is not None and callable(_miner):
                                pass  # hard-example mining is wired in the outer training loop
                        except Exception:
                            pass
                        pred_cls = logits.argmax(-1)
                        correct += (pred_cls == y_cls_idx).sum()
                        n_acc += int(y_cls_idx.numel())
                        pred_counts += torch.bincount(pred_cls.reshape(-1).clamp(0, 2), minlength=3)[:3]
                        true_counts += torch.bincount(y_cls_idx.reshape(-1).clamp(0, 2), minlength=3)[:3]
                        probs = torch.softmax(logits.float(), dim=-1)
                        for _t, _p in zip(y_cls_idx.reshape(-1).clamp(0, 2), pred_cls.reshape(-1).clamp(0, 2)):
                            confusion[int(_t), int(_p)] += 1
                        for _cls in range(3):
                            _mask_cls = y_cls_idx.reshape(-1).clamp(0, 2) == _cls
                            if bool(_mask_cls.any()):
                                diag_true_counts[_cls] += int(_mask_cls.sum())
                                logits_sum[_cls] += logits.float()[_mask_cls].sum(dim=0)
                                probs_sum[_cls] += probs[_mask_cls].sum(dim=0)
                        d = pred_cls.float() - 1.0
                    elif classification:
                        loss = crit(pred, y_cls_idx)
                        if not (torch.isfinite(loss) and torch.isfinite(pred).all()):
                            nan_skips += 1
                            if pbar is not None:
                                pbar.update(1)
                                pbar.set_postfix(loss="NaN-skip")
                            continue
                        total += loss
                        pred_cls = pred.argmax(-1)

                        correct += (pred_cls == y_cls_idx).sum()

                        n_acc += int(y_cls_idx.numel())

                        pred_counts += torch.bincount(pred_cls.reshape(-1).clamp(0, 2), minlength=3)[:3]

                        true_counts += torch.bincount(y_cls_idx.reshape(-1).clamp(0, 2), minlength=3)[:3]
                        probs = torch.softmax(pred.float(), dim=-1)
                        for _t, _p in zip(y_cls_idx.reshape(-1).clamp(0, 2), pred_cls.reshape(-1).clamp(0, 2)):
                            confusion[int(_t), int(_p)] += 1
                        for _cls in range(3):
                            _mask_cls = y_cls_idx.reshape(-1).clamp(0, 2) == _cls
                            if bool(_mask_cls.any()):
                                diag_true_counts[_cls] += int(_mask_cls.sum())
                                logits_sum[_cls] += pred.float()[_mask_cls].sum(dim=0)
                                probs_sum[_cls] += probs[_mask_cls].sum(dim=0)

                        d = pred_cls.float() - 1.0

                    else:
                        yb_reg = _match_target_shape(pred, yb)
                        loss = crit(pred, yb_reg)
                        if not (torch.isfinite(loss) and torch.isfinite(pred).all()):
                            nan_skips += 1
                            if pbar is not None:
                                pbar.update(1)
                                pbar.set_postfix(loss="NaN-skip")
                            continue
                        total += loss
                        correct += (torch.sign(pred) == torch.sign(yb_reg)).sum()
                        n_acc += int(yb_reg.numel())

                        d = torch.sign(pred)

                yb_for_returns = _match_target_shape(d, yb.float())
                # RL reward caches store unsigned max(reward_long, reward_short).
                # When rl_mode is on (or a direction sidecar is present), re-sign
                # by the optimal-side label so correct shorts contribute +PnL.
                if rl_mode or y_cls_b is not None:
                    if y_cls_b is not None:
                        side = _match_target_shape(d, y_cls_b.float()).sign()
                        yb_for_returns = yb_for_returns.abs() * side
                    # else: leave yb as provided signed returns
                r = (d * yb_for_returns).flatten()
                if r.numel() > 0:
                    r_sum += r.sum()
                    r_sq_sum += (r * r).sum()
                    n_ret += int(r.numel())

                valid_batches += 1

                if pbar is not None:
                    pbar.update(1)
                
                if (i + 1) % heartbeat_interval == 0 and _TRAIN_LOGGER:
                    _TRAIN_LOGGER.heartbeat()
                    
            except RuntimeError as e:
                is_oom = ("out of memory" in str(e).lower()) or ("cuda error" in str(e).lower())
                if not (device.type == "cuda" and is_oom):
                    raise
                oom_skips += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(loss="OOM-skip")
                continue

    if oom_skips:
        print(f"[Val] OOM summary: skipped {oom_skips} batch(es) this epoch.")
    if nan_skips:
        print(f"[Val] NaN summary: skipped {nan_skips} batch(es) this epoch.")

    ann = float(sharpe_ann_factor if sharpe_ann_factor is not None
                else TRAINING.get("sharpe_annualization_factor", 1.0))
    val_loss = total.item() / max(valid_batches, 1)

    dir_acc = correct.item() / max(n_acc, 1)

    if n_ret == 0 or valid_batches == 0:

        print(

            "[Val] WARNING: no validation return samples contributed to Sharpe "

            f"(valid_batches={valid_batches}, n_ret={n_ret}). Returning Sharpe=0."

        )

        _diag = _validation_class_diag(pred_counts, true_counts, confusion, logits_sum, probs_sum, diag_true_counts)
        validate_epoch.last_class_counts = {"pred": _diag["pred"], "true": _diag["true"]}
        validate_epoch.last_class_diag = _diag

        return val_loss, dir_acc, 0.0

    
    r_mean = r_sum / n_ret

    r_var = torch.clamp(r_sq_sum / n_ret - r_mean ** 2, min=0.0)

    sharpe = (r_mean / (r_var.sqrt() + 1e-8)).item() * ann
    _diag = _validation_class_diag(pred_counts, true_counts, confusion, logits_sum, probs_sum, diag_true_counts)
    validate_epoch.last_class_counts = {"pred": _diag["pred"], "true": _diag["true"]}
    validate_epoch.last_class_diag = _diag

    
    return val_loss, dir_acc, sharpe



def _strict_load_report(target: "nn.Module", state: dict, label: str,
                        min_frac_loaded: float = 0.5) -> dict:
    """Load `state` into `target` (strict=False) but capture and LOG the
    missing/unexpected/shape-mismatched keys, and FAIL LOUDLY when the load is
    effectively a no-op (A-H2 / A-C1).

    Only tensors whose name AND shape match are loaded; the rest are reported.
    Raises RuntimeError when fewer than `min_frac_loaded` of the target tensors
    were populated (i.e. the checkpoint didn't actually transfer).
    """
    target_sd = target.state_dict()
    # Tolerate a leading "backbone." prefix mismatch in either direction.
    if not any(k in target_sd for k in state) and state:
        if any(k.startswith("backbone.") for k in state):
            state = {k.replace("backbone.", "", 1): v for k, v in state.items()}
    filtered, mismatched = {}, []
    for k, v in state.items():
        if k in target_sd and hasattr(v, "shape") and target_sd[k].shape == v.shape:
            filtered[k] = v
        elif k in target_sd:
            mismatched.append(k)
    result = target.load_state_dict(filtered, strict=False)
    missing    = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    n_target = max(1, len(target_sd))
    n_loaded = n_target - len(missing)
    frac     = n_loaded / n_target
    print(f"[Load:{label}] {n_loaded}/{n_target} tensors ({frac:.1%}) | "
          f"missing={len(missing)} unexpected={len(unexpected)} "
          f"shape_mismatch={len(mismatched)}")
    if missing:
        print(f"  missing[:6]={missing[:6]}")
    if mismatched:
        print(f"  shape_mismatch[:6]={mismatched[:6]}")
    if frac < min_frac_loaded:
        raise RuntimeError(
            f"[Load:{label}] Only {frac:.1%} of target tensors were loaded "
            f"(< {min_frac_loaded:.0%}). The checkpoint did not transfer ΓÇö "
            "check architecture/seq_len/feature-count consistency."
        )
    return {"frac_loaded": frac, "missing": missing,
            "unexpected": unexpected, "shape_mismatch": mismatched}


def _load_pretrained_encoder(model: "nn.Module", args, device) -> bool:
    """A-C1: load the contrastive-pretrained encoder into the supervised model's
    backbone so pretraining is not wasted.

    The contrastive trainers checkpoint the backbone with the prediction head
    stripped (Identity), so the supervised head legitimately shows up as
    "missing"; everything else (the backbone) must transfer. We assert a large
    fraction loads, failing loudly if the transfer is effectively a no-op.
    """
    if getattr(args, "disable_pretrain_load", False):
        return False

    method     = str(getattr(args, "pretrain_method", PRETRAIN.get("method", "byol"))).lower()
    use_regime = getattr(args, "pretrain_regime", False) and method == "tscl"
    ckpt_dir   = Path(args.checkpoint_dir)
    candidates = []
    if use_regime:
        candidates.append(ckpt_dir / "contrastive_encoder_regime.pt")
    candidates += [ckpt_dir / "contrastive_encoder.pt",
                   ckpt_dir / "contrastive_encoder_regime.pt"]
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        print(f"[PretrainΓåÆSup] No contrastive encoder checkpoint in {ckpt_dir} ΓÇö "
              "skipping transfer (was pretraining run?).")
        return False
    encoder = model.backbone if hasattr(model, "backbone") else model
    target  = encoder.module if hasattr(encoder, "module") else encoder
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    # The head is expected to be missing ΓåÆ allow up to ~40% missing for wide heads.
    load_report = _strict_load_report(target, state, f"PretrainΓåÆ{args.model}", min_frac_loaded=0.6)
    try:
        _update_pretrain_report(args, {
            "loaded_into_supervised_training": True,
            "supervised_transfer": {
                "checkpoint_path": str(ckpt_path),
                "frac_loaded": float(load_report.get("frac_loaded", 0.0)),
                "missing_count": len(load_report.get("missing", [])),
                "unexpected_count": len(load_report.get("unexpected", [])),
                "shape_mismatch_count": len(load_report.get("shape_mismatch", [])),
            },
        })
    except Exception:
        pass
    print(f"[PretrainΓåÆSup] Loaded contrastive encoder from {ckpt_path.name} into backbone.")
    return True


def _warm_start_from_checkpoint(model: "nn.Module", args, device, model_name: str) -> bool:
    """B-C2: load prior production / best weights into the model so a fine-tune
    run CONTINUES from the deployed model instead of training from scratch.

    Distinct from --resume (which restores optimizer/epoch state and may skip
    training). Warm-start only seeds weights; training proceeds normally on the
    new window.
    """
    if not getattr(args, "finetune_warm_start", False):
        return False
    explicit = getattr(args, "warm_start_from", None)
    base = Path(args.checkpoint_dir)
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [
        base.parent / "production_best.pt",   # checkpoints/<run>/production_best.pt
        base / "production_best.pt",
        base / f"{model_name}_best.pt",
        base / model_name / f"{model_name}_best.pt",
    ]
    ckpt_path = next((p for p in candidates if p and p.exists()), None)
    if ckpt_path is None:
        print(f"[WarmStart] No prior checkpoint found for {model_name} "
              f"(looked in {base}); training from scratch.")
        return False
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ck.get("model_state", ck.get("model_state_dict", ck.get("state_dict", ck))) \
        if isinstance(ck, dict) else ck
    core = _core_model(model)
    _strict_load_report(core, state, f"WarmStart:{model_name}", min_frac_loaded=0.6)
    print(f"[WarmStart] Continuing training from {ckpt_path}")
    return True


def supervised_train(
    model_name: str,
    cache_path: str,
    n_samples:  int,
    n_features: int,
    args,
    device:     "torch.device",
    n_gpus:     int,
    run: Any = None,
    train_idx:  Optional[np.ndarray] = None,
    val_idx:    Optional[np.ndarray] = None,
    fold_id:    Optional[int] = None,
    amp_dtype:  "torch.dtype" = torch.float32,
):
    global _TRAIN_LOGGER
    _artifact_run_name = str(getattr(args, "run_name_slug", "") or _slug_part(getattr(args, "run_name", "pipeline-run"), max_len=140))

    if _TRAIN_LOGGER_AVAILABLE:
        if _TRAIN_LOGGER is None:
            _TRAIN_LOGGER = _TrainingLogger(
                log_dir    = PATHS.get("logs", "logs"),
                run_name   = f"{_artifact_run_name}_{datetime.now().strftime('%m%d_%H%M')}",

                model_name = model_name,
            )
            _TRAIN_LOGGER.setup()
        else:
            _TRAIN_LOGGER.model_name = model_name

    _log_dir  = PATHS.get("logs", "logs")
    _run_name = (
        f"{_slug_part(model_name, max_len=80)}"

        f"{'_fold' + str(fold_id) if fold_id is not None else ''}"
        f"_{datetime.now().strftime('%m%d_%H%M')}"
    )
    

    # -- TensorBoard writer ----------------------------------------------------
    _tb_writer = None
    if TENSORBOARD and not getattr(args, "no_tensorboard", False):
        _tb_dir = str(Path(_log_dir) / "tensorboard" / _run_name)
        _tb_writer = _SummaryWriter(log_dir=_tb_dir)
        print(f"[TensorBoard] Logging -> {_tb_dir}  "
              f"(tensorboard --logdir {_tb_dir} --port 6006)")

    # -- Rich live display -----------------------------------------------------
    _stop_on_sharpe_local = getattr(args, "early_stop_metric", "sharpe") == "sharpe"
    _rich_display = None
    if RICH_DISPLAY and not getattr(args, "no_rich", False):
        _rich_display = _RichDisplay(
            model_name       = model_name,
            total_epochs     = args.epochs,
            patience         = args.patience,
            metric_name      = "val_sharpe" if _stop_on_sharpe_local else "val_loss",
            higher_is_better = _stop_on_sharpe_local,
        )

    if getattr(args, "model_profile", True) and not getattr(args, "_profile_applied", False):
        args = _apply_model_profile(args, model_name, enabled=True)

    _t_start = time.time()
    classification = args.loss in ("cross_entropy", "multi_task", "asymmetric_directional")
    multitask = bool(getattr(args, "multitask", False))
    fold_suffix = f"_fold{fold_id}" if fold_id is not None else ""

    print(f"\n{'-'*60}")
    print(f"  Training: {model_name.upper()} | {n_samples:,} samples | "
          f"batch={args.batch_size} | AMP={args.amp} | loss={args.loss}")
    print(f"{'-'*60}")

    tune_idx = None
    if train_idx is None or val_idx is None:
        # A-H3: single-split path now inserts an embargo gap (seq_len + lookahead
        # + execution_delay) between train and val so forward-looking labels in
        # the last train samples can't leak into validation.
        # H6: reserve the promotion holdout tail (same as walk-forward CV).
        _holdout_n = _promotion_holdout_n(n_samples, args)
        _split_n = max(0, n_samples - _holdout_n)
        _embargo = _embargo_bars(args)
        _purge = _purge_bars(args)
        _method = _validation_method(args)
        # SYS-002: three-way split to isolate auto-tune evaluation from val (early stopping)
        _tune_split = float(getattr(args, "tune_split", 0.0) or 0.0)
        if _tune_split > 0:
            train_idx, val_idx, tune_idx = _three_way_split(
                _split_n, args.val_split, _tune_split, _embargo, _purge, _method
            )
            print(f"[Split] Three-way split: embargo={_embargo} purge={_purge} method={_method} "
                  f"tune_split={_tune_split} "
                  f"{f'| holdout tail = {_holdout_n:,} bars' if _holdout_n else ''}")
        else:
            train_idx, val_idx = _embargo_split(_split_n, args.val_split, _embargo, _purge, _method)
            print(f"[Split] Single split with embargo={_embargo} purge={_purge} method={_method} "
                  f"{f'| holdout tail = {_holdout_n:,} bars' if _holdout_n else ''}")
    print(f"[Split] Train: {len(train_idx):,} | Val: {len(val_idx):,}"
          f"{f' | Tune: {len(tune_idx):,}' if tune_idx is not None else ''}"
          f"{fold_suffix if fold_id is not None else ''}")

    # SYS-002: store tune_idx on args for post-training evaluation
    args._tune_eval_idx = tune_idx

    # ZarrStreamDataset reads each zarr chunk exactly once per epoch (sequential
    # block reads + in-block shuffle) instead of one random decompression per
    # sample.  Val indices are sorted so val reads are also sequential.
    use_direction_targets = bool(multitask or classification)
    if use_direction_targets:
        try:
            _direction_preflight(cache_path, train_idx, val_idx, args)
        except RuntimeError as exc:
            # Tiny synthetic/quick caches often cannot satisfy class-prior floors.
            if getattr(args, "ignore_preflight", False) or (
                getattr(args, "quick_mode", False)
                and str(getattr(args, "data_source", "")).lower() == "synthetic"
            ):
                print(f"[DirectionPreflight] WARN (continuing): {exc}")
            else:
                raise

    train_ds = ZarrStreamDataset(
        cache_path, train_idx, shuffle_chunks=True,

        multitask_targets=use_direction_targets,
        return_indices=True,

    )
    val_ds   = ZarrStreamDataset(
        cache_path, np.sort(val_idx), shuffle_chunks=False,

        multitask_targets=use_direction_targets,

    )

    # Windows DataLoader workers use spawned processes plus shared file mappings.
    # Large zarr batches can exhaust the OS paging/shared-memory budget and fail
    # with WinError 1455, so keep loading in-process and use thread prefetch below.
    nw = 0 if os.name == "nt" else min(max(0, int(args.num_workers)), os.cpu_count() or 4)
    pf = int(args.prefetch_factor) if nw > 0 else None
    # Validation: keep fewer batches in flight than training. Pinning large batches
    # in worker threads can trip CUDA OOM on Windows (WDDM + driver) even when
    # train fits ΓÇö the error often surfaces as pin_memory(..., device=0).
    val_nw = max(2, nw // 2)
    val_pf = min(int(pf), 2) if pf is not None else None

    # B: On Windows, persistent_workers=True can sometimes cause I/O hangs 
    # when combined with certain storage backends or external drives. 
    # Default to False on Windows for stability.
    use_persistent = (nw > 0 and os.name != "nt")
    if getattr(args, "persistent_workers", None) is not None:
        use_persistent = bool(args.persistent_workers) and nw > 0

    if getattr(args, "val_num_workers", None) is not None:
        val_nw_safe = max(0, min(int(args.val_num_workers), os.cpu_count() or 4))
    else:
        val_nw_safe = 0 if os.name == "nt" else val_nw
    if getattr(args, "val_prefetch_factor", None) is not None and val_nw_safe > 0:
        val_pf = max(1, int(args.val_prefetch_factor))

    default_pin = (os.name != "nt")
    pin_mem = default_pin if getattr(args, "pin_memory", None) is None else bool(args.pin_memory)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=nw, pin_memory=pin_mem, persistent_workers=use_persistent,
                          prefetch_factor=pf)
    _bn_train_dl = train_dl   # full-distribution loader for SWA BN update (never filtered)
    # On Windows, DataLoader worker processes crash unexpectedly during validation
    # after many training epochs (memory pressure kills subprocesses silently).
    # num_workers=0 runs loading in the main process ΓÇö safe, and fast enough for val
    # since there's no backward pass.
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=val_nw_safe, pin_memory=pin_mem,
                          persistent_workers=(use_persistent and val_nw_safe > 0),
                          prefetch_factor=None if val_nw_safe == 0 else val_pf)

    # On Windows num_workers=0 is forced, so prefetch_factor has no effect and
    # zarr decompression blocks the main thread between GPU steps.
    # Wrap with a background-thread prefetch to overlap CPU decompression with
    # GPU compute ΓÇö effectively hides one full batch worth of I/O latency.
    if nw == 0:
        _tp = max(1, int(getattr(args, "thread_prefetch_batches", 2)))
        train_dl = _ThreadPrefetchLoader(train_dl, prefetch=_tp)
        val_dl   = _ThreadPrefetchLoader(val_dl,   prefetch=_tp)
    print(f"[Loader] {len(train_dl)} train batches | {len(val_dl)} val batches | "
          f"{nw} workers | prefetch={pf if pf is not None else 0} | "
          f"val_workers={val_nw_safe} val_prefetch={val_pf if val_pf is not None else 0} "
          f"pin_mem={pin_mem} persistent={use_persistent}")

    model = build_model(model_name, n_features, args).to(device)
    if multitask:
        try:
            _fold_prior = _class_prior_tensor(
                cache_path, train_idx, device, use_direction_sidecar=True,
            )
            _init_multitask_direction_bias(model, _fold_prior)
            _p = [float(x) for x in _fold_prior.detach().cpu().tolist()]
            print(f"[ClassPrior] Fold train S/H/B prior={_p[0]:.3f}/{_p[1]:.3f}/{_p[2]:.3f} "
                  "(direction head bias initialized)")
        except Exception as _prior_exc:
            print(f"[ClassPrior] WARN: prior init skipped ({_prior_exc})")
    # A-C1: transfer contrastive-pretrained encoder weights into the backbone.
    if getattr(args, "pretrain", False):
        try:
            _loaded_pretrain = _load_pretrained_encoder(model, args, device)
            if not _loaded_pretrain:
                _update_pretrain_report(args, {
                    "loaded_into_supervised_training": False,
                    "supervised_transfer": {
                        "status": "skipped_no_checkpoint",
                    },
                })
        except Exception as _pe:
            # Fail loudly: a silent no-op here means pretraining was wasted.
            raise RuntimeError(f"[PretrainΓåÆSup] encoder transfer failed: {_pe}") from _pe
    # B-C2: warm-start from prior production/best weights (fine-tune mode).
    if getattr(args, "finetune_warm_start", False):
        _warm_start_from_checkpoint(model, args, device, model_name)

    # -- Teacher Model for Distillation --
    teacher_model = None
    distill_weight = getattr(args, "distill_weight", 0.5)
    if getattr(args, "teacher_model", None):
        print(f"\n[Distillation] Loading teacher model: {args.teacher_model}...")
        try:
            from inference.pytorch_inference import load_pytorch_model
            _t_ckpt = Path(args.teacher_ckpt) if getattr(args, "teacher_ckpt", None) else Path(args.checkpoint_dir) / args.teacher_model / f"{args.teacher_model}_best.pt"
            if not _t_ckpt.is_absolute():
                _t_ckpt = Path.cwd() / _t_ckpt
            if not _t_ckpt.exists():
                _t_ckpt = Path(args.checkpoint_dir) / "production_best.pt" # Fallback
            if not _t_ckpt.exists():
                raise FileNotFoundError(f"Teacher checkpoint not found: {_t_ckpt}")
                
            teacher_model, _, _, _ = load_pytorch_model(
                checkpoint_path=str(_t_ckpt),
                model_name=args.teacher_model,
                seq_len=getattr(args, "lookahead_bars", 60),
                n_features=n_features,
                device=device
            )
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            if n_gpus > 1:
                teacher_model = nn.DataParallel(teacher_model)
            print(f"[Distillation] Teacher loaded successfully! (weight={distill_weight})")
        except Exception as e:
            print(f"[Distillation] Warning: Failed to load teacher model: {e}")
            teacher_model = None

    if n_gpus > 1:
        model = nn.DataParallel(model); print(f"[Model] DataParallel ├ù {n_gpus} GPUs")

    # -- torch.compile (PyTorch >= 2.0) ΓÇö ~20-30 % extra throughput on Ada ----
    # Requires Triton (Linux only). The cudagraphs backend is NOT safe for
    # LSTM + attention models (dynamic internal shapes ΓåÆ silent NaN).
    # On Windows: skip compile entirely and run in eager mode.
    _triton_ok = False
    try:
        import triton  # noqa: F401
        _triton_ok = True
    except ImportError:
        pass
    if _triton_ok and device.type == "cuda" and hasattr(torch, "compile") \
            and bool(_GPU_CFG.get("torch_compile", True)):
        try:
            _compile_mode = str(_GPU_CFG.get("torch_compile_mode", "reduce-overhead"))
            # LSTM models have dynamic internal shapes incompatible with CUDA graphs
            _has_lstm = any(isinstance(m, nn.LSTM) for m in model.modules())
            if _has_lstm and _compile_mode == "reduce-overhead":
                _compile_mode = "default"
                print("[Model] LSTM detected — downgrading torch.compile to mode='default' (no CUDA graphs)")
            model = torch.compile(model, mode=_compile_mode)
            print(f"[Model] torch.compile ON (backend=inductor, mode={_compile_mode})")
            _log_info(f"[Model] torch.compile inductor mode={_compile_mode}")
        except Exception as _ce:
            print(f"[Model] torch.compile skipped: {_ce}")
            _log_warn(f"[Model] torch.compile skipped: {_ce}")
    elif device.type == "cuda":
        print("[Model] torch.compile skipped (Triton not available ΓÇö running eager mode)")
        _log_info("[Model] torch.compile skipped ΓÇö eager mode")

    crit = build_criterion(
        args, device,
        cache_path=cache_path if (classification or multitask) else None,
        train_idx=train_idx if (classification or multitask) else None,
    )
    direction_crit = nn.CrossEntropyLoss().to(device)
    # D: OverconfidencePenalty ΓÇö active for regression modes only
    global _OVERCONF_PENALTY
    if not classification and getattr(args, "overconf_penalty", True):
        _oc_w = float(getattr(args, "overconf_weight", 0.3))
        _oc_t = float(getattr(args, "overconf_threshold", 0.6))
        _OVERCONF_PENALTY = OverconfidencePenalty(conf_threshold=_oc_t, weight=_oc_w).to(device)
    else:
        _OVERCONF_PENALTY = None
    opt       = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    # Gradient accumulation: effective batch = batch_size ├ù accum_steps
    _accum = max(1, int(getattr(args, "grad_accum_steps", 1)))
    # OneCycleLR must be stepped once per OPTIMIZER UPDATE (not per batch).
    # steps_per_epoch = ceil(batches / accum_steps) so the total cycle length
    # equals epochs ├ù optimizer-updates-per-epoch.
    _eff_steps = max(1, -(-len(train_dl) // _accum))   # ceiling div
    _sched_kind = str(getattr(args, "lr_schedule", "onecycle")).strip().lower()
    _total_steps = max(1, args.epochs * _eff_steps)
    if _sched_kind == "warmup_cosine":
        _warmup_ep = max(0, int(getattr(args, "lr_warmup_epochs", 3)))
        _warmup_steps = _warmup_ep * _eff_steps
        if _warmup_steps <= 0:
            _warmup_steps = max(1, int(_total_steps * float(getattr(args, "lr_warmup_pct", 0.1))))
        _min_ratio = float(getattr(args, "lr_min_ratio", 0.05))
        _min_ratio = min(max(_min_ratio, 0.0), 1.0)
        _decay_steps = max(1, _total_steps - _warmup_steps)

        def _warmup_cosine(step: int) -> float:
            if step < _warmup_steps:
                return (step + 1) / max(1, _warmup_steps)
            progress = min(1.0, (step - _warmup_steps) / _decay_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return _min_ratio + (1.0 - _min_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_warmup_cosine)
        print(f"[Scheduler] WarmupCosine | warmup_steps={_warmup_steps} "
              f"| min_ratio={_min_ratio:.3f} | total={_total_steps:,}")
    else:
        max_lr = float(getattr(args, "onecycle_max_lr_mult", 10.0)) * args.lr
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=max_lr,
            epochs=args.epochs,
            steps_per_epoch=_eff_steps,
            pct_start=float(getattr(args, "onecycle_pct_start", 0.1)),
            anneal_strategy="cos",
        )
        print(f"[Scheduler] OneCycleLR | max_lr={max_lr:.2e} | "
              f"steps/ep={_eff_steps} (accum={_accum}) | total={_total_steps:,}")
    # GradScaler is only useful for FP16 (prevents underflow).
    # BF16 covers the same exponent range as FP32 -> scaling would be a no-op.
    _fp16_scaler_needed = (
        args.amp and device.type == "cuda" and amp_dtype == torch.float16
    )
    amp_sc = GradScaler(enabled=_fp16_scaler_needed)
    dtype_name = {torch.bfloat16: "BF16", torch.float16: "FP16", torch.float32: "FP32"}.get(amp_dtype, "?")
    print(f"[AMP] dtype={dtype_name} | GradScaler={'ON' if _fp16_scaler_needed else 'OFF (BF16/FP32)'}")

    # -- Online Hard-Example Miner ---------------------------------------------
    _online_miner = None
    _use_online_miner = bool(getattr(args, "online_hard_mining", True))
    if _use_online_miner and (classification or multitask):
        try:
            from training.hard_example_miner import OnlineHardExampleMiner
            _online_miner = OnlineHardExampleMiner(
                n_samples=len(train_idx),
                window_size=5,
                hard_quantile=0.85,
                forget_window=3,
                easy_quantile=0.30,
                boost_factor=2.0,
                decay_factor=0.90,
            )
            print(f"[OnlineMiner] Created for {model_name} ({len(train_idx):,} samples)")
        except Exception as _om_e:
            print(f"[OnlineMiner] Init failed (disabled): {_om_e}")
            _online_miner = None
    # -- SWA (Stochastic Weight Averaging) -------------------------------------
    # Averages model weights over the last (1 - swa_start_frac) fraction of training.
    # Typically gives +2-5% generalization improvement with zero extra VRAM cost.
    _swa_enabled    = bool(getattr(args, "swa_enabled", TRAINING.get("swa_enabled", False)))
    _swa_start_frac = float(getattr(args, "swa_start_frac", TRAINING.get("swa_start_frac", 0.75)))
    _swa_start_frac = min(max(_swa_start_frac, 0.0), 1.0)
    _swa_start_ep   = max(1, int(args.epochs * _swa_start_frac))
    _swa_lr         = float(getattr(args, "swa_lr", TRAINING.get("swa_lr", 1e-5)))
    _swa_model      = None
    _swa_scheduler  = None
    _swa_started    = False
    if _swa_enabled:
        try:
            from torch.optim.swa_utils import AveragedModel, SWALR, update_bn as _swa_update_bn
            # Defer instantiation until the training loop to ensure lazy modules are fully initialized
            print(f"[SWA] Enabled | start_ep={_swa_start_ep} | swa_lr={_swa_lr:.2e}")
        except Exception as _swa_e:
            _swa_enabled = False
            print(f"[SWA] Disabled (import failed): {_swa_e}")

    # -- Flash Attention / SDPA (critical combo with AMP) ---------------------
    # PyTorch >= 2.0: nn.MultiheadAttention and TransformerEncoderLayer internally
    # call F.scaled_dot_product_attention, which dispatches to the Flash Attention
    # CUDA kernel when in FP16/BF16 autocast context -> full Tensor Core utilisation.
    # Keep math SDP enabled as a numerical fallback. Forcing only flash/mem-efficient
    # kernels is faster, but long multi-pair sequences can hit non-finite attention
    # intermediates under AMP on some Windows/CUDA stacks.
    if device.type == "cuda" and args.amp and hasattr(torch.backends.cuda, "enable_flash_sdp"):
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            print("[SDPA] Flash/mem-efficient SDP enabled with math fallback")
            _log_info("[SDPA] Flash Attention enabled with math fallback")
        except Exception as _sdpa_e:
            print(f"[SDPA] Could not configure SDPA backends: {_sdpa_e}")
            _log_warn(f"[SDPA] SDPA backend config failed: {_sdpa_e}")
    elif device.type == "cuda" and not args.amp:
        # FP32 path: enable math SDP (still uses SDPA dispatch, just no Tensor Cores)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)

    ckpt_dir  = Path(args.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{model_name}{fold_suffix}_best.pt"
    cfg_path  = ckpt_dir / f"{model_name}{fold_suffix}_config.json"

    stop_on_sharpe = args.early_stop_metric == "sharpe"

    # Resume (single-split only; skip per-fold resume id)
    start_ep = 0
    last_path = ckpt_dir / f"{model_name}{fold_suffix}_last.pt"
    # Temporary holders for auxiliary resume state applied after their init blocks
    _resume_chunk_history: Optional[list]  = None
    _resume_chunk_streak:  Optional[int]   = None
    _resume_feat_state:    Optional[dict]  = None
    _resume_curriculum_state: Optional[dict] = None

    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        core = _core_model(model)
        try:
            core.load_state_dict(ck["model_state"])
            opt.load_state_dict(ck["opt_state"])
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"\n[Resume Error] Feature dimension mismatch detected! The dataset has changed since this model was last trained. Disabling resume and starting fresh...\n  Details: {e}")
                args.resume = False
                ck = {}  # Clear checkpoint dict so the below .get() calls return defaults
            else:
                raise e
        _recover_nonfinite_training_state(model, opt)
        _sched_state = ck.get("scheduler_state")
        if _sched_state is not None:
            try:
                scheduler.load_state_dict(_sched_state)
            except Exception as _se:
                print(f"[Resume] Scheduler state restore skipped: {_se}")
        if "scaler_state" in ck and ck["scaler_state"] is not None:
            amp_sc.load_state_dict(ck["scaler_state"])
        start_ep = int(ck.get("epoch", -1)) + 1
        best_val_loss = float(ck.get("best_val_loss", float("inf")))
        best_sharpe = float(ck.get("best_sharpe", float("-inf")))
        no_improve = int(ck.get("no_improve", 0))
        improved = False   # Initialize to avoid UnboundLocalError
        history = ck.get("history", {"train_loss":[], "val_loss":[], "dir_acc":[], "val_sharpe":[], "lr":[]})
        for _hist_key in ("seq_len", "difficulty_stage", "curriculum_stalls"):

            history.setdefault(_hist_key, [])

        # Restore auxiliary state (new keys; fall back gracefully for old checkpoints)
        _resume_chunk_history = list(ck.get("chunk_sharpe_history",
                                             history.get("val_sharpe", [])))
        _resume_chunk_streak  = int(ck.get("chunk_worse_streak", 0))
        _resume_feat_state    = ck.get("feat_stability_state", None)
        _resume_curriculum_state = ck.get("curriculum_state")
        print(f"[Resume] Loaded exact state from {last_path} (epoch {start_ep})")
    elif args.resume and best_path.exists() and fold_id is None:
        core = _core_model(model)
        try:
            core.load_state_dict(torch.load(best_path, map_location=device))
            _recover_nonfinite_training_state(model, opt)
            print(f"[Resume] Loaded weights from {best_path}")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"\n[Resume Error] Feature dimension mismatch detected! The dataset has changed since this model was last trained. Disabling resume and starting fresh...\n  Details: {e}")
                args.resume = False
            else:
                raise e

    if start_ep == 0:
        best_val_loss = float("inf")
        best_sharpe = float("-inf")
        no_improve = 0
        improved = False   # Initialize to avoid UnboundLocalError
        history  = {

            "train_loss": [], "val_loss": [], "dir_acc": [], "val_sharpe": [], "lr": [],

            "seq_len": [], "difficulty_stage": [], "curriculum_stalls": [],

        }

    if use_direction_targets and start_ep == 0:
        _direction_probe(
            model, cache_path, train_idx, val_idx, args, device,
            model_name=model_name, n_features=n_features, amp_dtype=amp_dtype,
        )


    # -- A3: Variable-length sequence curriculum -------------------------------
    _CURR = getattr(args, "curriculum", None)
    if not isinstance(_CURR, dict):
        _CURR = SETTINGS_CURRICULUM
    _seq_schedule = _CURR.get("seq_schedule", [])
    _chunk_patience = int(_CURR.get("chunk_early_stop_patience", 3))
    _chunk_min_batches = int(_CURR.get("chunk_early_stop_min_batches", 50))
    _feat_groups = _CURR.get("feature_groups", {})
    _adapt_cfg = _CURR.get("adaptation", {}) if isinstance(_CURR.get("adaptation", {}), dict) else {}

    _adapt_stable_window = max(2, int(_adapt_cfg.get("stable_window", 3)))

    _adapt_min_stable_sharpe = float(_adapt_cfg.get("min_stable_sharpe", 0.05))

    _adapt_collapse_min_peak = float(_adapt_cfg.get("collapse_min_peak", 0.25))

    _adapt_collapse_drop = float(_adapt_cfg.get("collapse_drop", 0.15))

    _adapt_advance_lr_mult = float(_adapt_cfg.get("advance_lr_mult", 0.85))

    _adapt_collapse_lr_mult = float(_adapt_cfg.get("collapse_lr_mult", 0.80))

    # P0/P1: new curriculum adaptation params
    # collapse_reversal_threshold: single-epoch drop below this triggers stall (replaces hardcoded -0.1)
    _adapt_reversal_threshold = float(_adapt_cfg.get("collapse_reversal_threshold", -0.10))
    # recovery_window: consecutive stable epochs post-stall before unfreezing curriculum
    _adapt_recovery_window = max(2, int(_adapt_cfg.get("recovery_window", 4)))
    # min_epochs_per_stage: cooldown epochs required between any two advances
    _adapt_min_epochs_per_stage = max(1, int(_adapt_cfg.get("min_epochs_per_stage", 3)))
    # P2: EMA alpha for smoothed Sharpe signal used by the advance gate
    # Raw val_sharpe remains for collapse detection (fast); EMA used for stability window
    _adapt_ema_alpha = float(_adapt_cfg.get("sharpe_ema_alpha", 0.30))

    # --- Schedule-floor helpers (P0: epoch_start as guaranteed minimum) ---
    def _sched_floor_seq(ep: int) -> int:
        """Return the minimum seq_len the schedule mandates at this epoch."""
        if not _seq_schedule:
            return args.seq_len
        floor = int(_seq_schedule[0]["seq_len"])
        for entry in _seq_schedule:
            if ep >= int(entry.get("epoch_start", 0)):
                floor = int(entry["seq_len"])
        return floor

    def _sched_floor_diff(ep: int) -> int:
        """Return the minimum difficulty stage the schedule mandates at this epoch."""
        if not _diff_schedule:
            return 0
        floor = int(_diff_schedule[0]["max_difficulty"])
        for entry in _diff_schedule:
            if ep >= int(entry.get("epoch_start", 0)):
                floor = int(entry["max_difficulty"])
        return floor


    # -- Adaptive Curriculum State --
    _active_seq_len = args.seq_len
    if _seq_schedule:
        _active_seq_len = int(_seq_schedule[0]["seq_len"])
    if history.get("seq_len"):
        _active_seq_len = history["seq_len"][-1]

    _rolling_sharpes = []
    _seq_frozen = False
    _last_logged_seq_len = None
    _last_logged_diff_stage = None

    # P0/P1 state ΓÇö cooldown, recovery, stall tracking
    _epochs_since_advance: int = 0          # epochs elapsed since last advance (cooldown)
    _post_stall_stable_count: int = 0       # consecutive stable epochs after last stall (recovery)
    # P2: EMA Sharpe state ΓÇö seeded from history tail on resume so continuity is preserved
    _ema_hist = history.get("sharpe_ema", [])
    _sharpe_ema: Optional[float] = float(_ema_hist[-1]) if _ema_hist else None


    def _seq_len_for_epoch(ep: int) -> int:
        return _active_seq_len

    def _unfreeze_features_for_epoch(model_ref, ep: int) -> None:
        """A4: Unfreeze parameter groups that correspond to slow feature layers."""
        # We can't freeze individual feature-group neurons post-hoc, but we CAN
        # freeze the first N encoder layers and gradually unfreeze them.
        # Here we use a pragmatic approach: freeze all but the last layer for
        # early epochs, then progressively unfreeze.
        core = _core_model(model_ref)
        # Collect named parameters to freeze/unfreeze
        earliest_unfreeze = min(
            (g["epoch_unfreeze"] for g in _feat_groups.values() if not g.get("always_on", True)),
            default=10,
        )
        if ep < earliest_unfreeze:
            # Freeze all non-essential layers (keep head + last encoder layer trainable)
            for name, param in core.named_parameters():
                if _is_uninitialized_parameter(param):
                    continue
                # Keep output head and final normalisation always trainable
                if any(k in name for k in ("head", "norm", "out_proj")):
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(ep >= 0)   # always requires_grad; gradient zeroed via scheduler
            return
        # After earliest_unfreeze: all parameters are trainable
        for param in core.parameters():
            if _is_uninitialized_parameter(param):
                continue
            param.requires_grad_(True)

    # -- B: Difficulty curriculum ΓÇö load diff sidecar once, filter each epoch -
    _diff_arr = _load_diff_array(cache_path, n_samples)
    _diff_schedule = _CURR.get("difficulty_schedule", [])
    _active_diff_stage = 0
    if _diff_schedule:
        _active_diff_stage = int(_diff_schedule[0]["max_difficulty"])
    if history.get("difficulty_stage"):
        _active_diff_stage = history["difficulty_stage"][-1]
    _feature_schema = _load_feature_schema(cache_path, n_features)
    if _feature_schema is None:
        _log_warn("[Curriculum] Ordered feature schema sidecar missing; feature-group mask will use FEATURE_MASK only if lengths match.")
    _schema_for_masks = _feature_schema or list(getattr(args, "_feat_names", []) or [])

    _feature_ablation_cfg = _feature_ablation_config(args)

    _feature_ablation_mask_np, _feature_ablation_report = _build_feature_ablation_mask(

        _schema_for_masks,

        _feat_groups,

        _feature_ablation_cfg,

        n_features,

    )

    _feature_ablation_mask = (

        torch.from_numpy(_feature_ablation_mask_np).to(device)

        if _feature_ablation_mask_np is not None

        else None

    )

    if _feature_ablation_report.get("enabled"):

        _log_info(

            f"[FeatureAblation] {model_name}: { _feature_ablation_report.get('name') } "

            f"masked={_feature_ablation_report.get('masked_count')}/{n_features}"

        )

    try:

        _fa_dir = Path(args.checkpoint_dir) / model_name

        _fa_dir.mkdir(parents=True, exist_ok=True)

        _feature_ablation_report.update({

            "model_name": model_name,

            "schema_available": bool(_schema_for_masks),

            "written_at": datetime.now(timezone.utc).isoformat(),

        })

        _safe_save_json(_feature_ablation_report, _fa_dir / f"{model_name}_feature_ablation_report.json")

    except Exception as _fa_e:

        _log_warn(f"[FeatureAblation] Report write failed: {_fa_e}")


    def _difficulty_stage_for_epoch(ep: int) -> int:
        return _active_diff_stage

    _fold_class_prior_np = _class_prior_array(
        cache_path, train_idx, use_direction_sidecar=True,
    )
    _log_info(
        "[CurriculumCalibration] Full-fold S/H/B prior="
        f"{_fold_class_prior_np[0]:.3f}/{_fold_class_prior_np[1]:.3f}/{_fold_class_prior_np[2]:.3f}"
    )

    def _calibrated_curriculum_subset(idx: np.ndarray, candidate: np.ndarray, ep: int, stage: int) -> np.ndarray:
        """Reject curriculum subsets that distort the fold direction distribution."""
        if len(candidate) >= len(idx):
            return candidate
        if len(candidate) < 50:
            _log_warn(
                f"[CurriculumCalibration] Epoch {ep+1}: stage={stage} has only "
                f"{len(candidate):,}/{len(idx):,} samples; using full fold."
            )
            return idx
        try:
            cand_prior = _class_prior_array(
                cache_path, candidate, use_direction_sidecar=True,
            )
            max_abs_delta = float(np.max(np.abs(cand_prior - _fold_class_prior_np)))
            min_share = float(np.min(cand_prior))
            max_delta_allowed = float((_CURR.get("calibration") or {}).get("max_class_prior_delta", 0.05))
            min_share_allowed = float((_CURR.get("calibration") or {}).get("min_class_share", 0.05))
            if max_abs_delta > max_delta_allowed or min_share < min_share_allowed:
                _log_warn(
                    "[CurriculumCalibration] Epoch "
                    f"{ep+1}: rejecting stage={stage} subset; "
                    f"subset prior S/H/B={cand_prior[0]:.3f}/{cand_prior[1]:.3f}/{cand_prior[2]:.3f}, "
                    f"fold prior S/H/B={_fold_class_prior_np[0]:.3f}/{_fold_class_prior_np[1]:.3f}/{_fold_class_prior_np[2]:.3f}, "
                    f"max_delta={max_abs_delta:.3f}. Using full fold."
                )
                return idx
            _log_info(
                "[CurriculumCalibration] Epoch "
                f"{ep+1}: stage={stage} subset prior S/H/B="
                f"{cand_prior[0]:.3f}/{cand_prior[1]:.3f}/{cand_prior[2]:.3f} "
                f"({len(candidate):,}/{len(idx):,} samples)"
            )
        except Exception as _cal_exc:
            _log_warn(f"[CurriculumCalibration] Epoch {ep+1}: failed ({_cal_exc}); using full fold.")
            return idx
        return candidate

    def _apply_difficulty_filter(idx: np.ndarray, ep: int) -> np.ndarray:
        """Filter training indices to only include samples at/below the epoch's difficulty level."""
        if _diff_arr is None or not _diff_schedule:
            return idx
        max_diff = _difficulty_stage_for_epoch(ep)
        if max_diff >= 2:
            return idx   # all difficulties ΓÇö no filter needed
        mask = _diff_arr[idx] <= max_diff
        filtered = idx[mask]
        if len(filtered) < 50:
            return idx   # safety: never starve the dataloader
        return _calibrated_curriculum_subset(idx, filtered, ep, max_diff)

    # -- A: Feature stability monitor -----------------------------------------
    _feat_stability = FeatureStabilityMonitor(
        n_features     = n_features,
        ema_alpha      = 0.90,
        soft_threshold = 2.0,    # sigma shift -> noisy (dampen to 0.5├ù)
        hard_threshold = 4.0,    # sigma shift -> immediately freeze
        freeze_after   = 3,      # consecutive soft epochs -> freeze
        damping_factor = 0.50,
        warmup_epochs  = 3,      # don't penalise in first 3 epochs (EMA cold start)
        min_active_pct = 0.50,   # always keep >=50% of features active
    )
    if _resume_feat_state is not None:
        try:
            _feat_stability.load_state(_resume_feat_state)
            print(f"[Resume] Restored FeatureStabilityMonitor state "
                  f"(epoch={_feat_stability._epoch}, "
                  f"frozen={int(_feat_stability._frozen.sum())})")
        except Exception as _fst_e:
            print(f"[Resume] FeatureStabilityMonitor state restore failed (cold-start): {_fst_e}")
    _feat_mask: Optional[torch.Tensor] = None   # updated each epoch; None = all active

    # -- A2: Chunk early stopping state ---------------------------------------
    _chunk_sharpe_history: list = []
    _chunk_worse_streak: int = 0
    if _resume_chunk_history is not None:
        _chunk_sharpe_history = _resume_chunk_history
        _chunk_worse_streak   = _resume_chunk_streak or 0

    # Rich display replaces the plain print table; plain fallback when unavailable
    if _rich_display is not None:
        _rich_display.__enter__()
    else:
        print(f"\n{'Ep':>5} {'Train':>11} {'Val':>11} {'DirAcc':>8} {'vSharpe':>9} "
              f"{'LR':>10} {'Time':>7} {'GPU MB':>8}")
        print("-" * 72)

    # -- Crash checkpoint helper -----------------------------------------------
    def _save_crash_ckpt(failed_ep: int, exc: Exception) -> None:
        """Save a crash checkpoint so the error can be inspected and training resumed."""
        import traceback as _tb
        crash_path = ckpt_dir / f"{model_name}{fold_suffix}_crash.pt"
        try:
            _core = _core_model(model)
            _safe_save({
                "epoch":                failed_ep,
                "model_state":          _core.state_dict(),
                "opt_state":            opt.state_dict(),
                "scheduler_state":      scheduler.state_dict(),
                "scaler_state":         amp_sc.state_dict() if args.amp and device.type == "cuda" else None,
                "best_val_loss":        best_val_loss,
                "best_sharpe":          best_sharpe,
                "no_improve":           no_improve,
                "history":              history,
                "fold_id":              fold_id,
                "chunk_sharpe_history": _chunk_sharpe_history,
                "chunk_worse_streak":   _chunk_worse_streak,
                "feat_stability_state": _feat_stability.get_state(),
                "error_msg":            str(exc),
                "error_type":           type(exc).__name__,
                "error_traceback":      _tb.format_exc(),
            }, crash_path)
            print(f"\n[Train] Crash checkpoint saved ΓåÆ {crash_path}")
            print(f"[Train] Resume from last clean epoch with: --resume")
            if _rich_display is not None:
                _rich_display.__exit__(None, None, None)
        except Exception as _cs_exc:
            print(f"[Train] Warning: could not save crash checkpoint: {_cs_exc}")

    epoch_bar = _pbar(range(start_ep, args.epochs), desc=f"Train {model_name.upper()}", unit="ep") if _rich_display is None else range(start_ep, args.epochs)

    # Restore curriculum runtime state from resume checkpoint (preferred) or history tail.
    _v_sh_history: list = list(history.get("val_sharpe", []))
    _curriculum_stalls: int = (history.get("curriculum_stalls", [0]) or [0])[-1]
    _curriculum_events: list = []
    if _resume_curriculum_state:
        _seq_frozen = bool(_resume_curriculum_state.get("seq_frozen", False))
        _rolling_sharpes = list(_resume_curriculum_state.get("rolling_sharpes", []))
        _epochs_since_advance = int(_resume_curriculum_state.get("epochs_since_advance", 0))
        _post_stall_stable_count = int(_resume_curriculum_state.get("post_stall_stable_count", 0))
        if _resume_curriculum_state.get("sharpe_ema") is not None:
            _sharpe_ema = float(_resume_curriculum_state["sharpe_ema"])
        if _resume_curriculum_state.get("active_seq_len") is not None:
            _active_seq_len = int(_resume_curriculum_state["active_seq_len"])
        if _resume_curriculum_state.get("active_diff_stage") is not None:
            _active_diff_stage = int(_resume_curriculum_state["active_diff_stage"])
        _curriculum_stalls = int(_resume_curriculum_state.get("curriculum_stalls", _curriculum_stalls))
        _curriculum_events = list(_resume_curriculum_state.get("curriculum_events", []))
        _v_sh_history = list(_resume_curriculum_state.get("v_sh_history", _v_sh_history))
        print(f"[Resume] Restored curriculum state: seq_len={_active_seq_len} "
              f"diff_stage={_active_diff_stage} stalls={_curriculum_stalls} frozen={_seq_frozen}")
    elif _curriculum_stalls > 0:
        _seq_frozen = True

    for ep in epoch_bar:
        if _TRAIN_LOGGER is not None:
            _TRAIN_LOGGER.on_epoch_start(ep, total_epochs=args.epochs,
                                         seq_len=_seq_len_for_epoch(ep))
        # -- A3: Apply adaptive curriculum seq_len -----------------------------
        # P0: enforce epoch_start as a guaranteed floor ΓÇö schedule milestones are
        # honoured even when the performance gate hasn't fired yet.
        _floor_seq = _sched_floor_seq(ep)
        if _floor_seq > _active_seq_len and not _seq_frozen:
            _log_info(f"[Curriculum] Epoch {ep+1}: schedule floor raised seq_len "
                      f"{_active_seq_len} ΓåÆ {_floor_seq} (epoch_start milestone)")
            _curriculum_events.append({
                "epoch": ep + 1, "type": "seq_len_schedule_floor",
                "from_seq_len": int(_active_seq_len), "to_seq_len": int(_floor_seq),
            })
            _active_seq_len = _floor_seq
            _epochs_since_advance = 0

        _floor_diff = _sched_floor_diff(ep)
        if _floor_diff > _active_diff_stage:
            _log_info(f"[Curriculum] Epoch {ep+1}: schedule floor raised difficulty "
                      f"{_active_diff_stage} ΓåÆ {_floor_diff} (epoch_start milestone)")
            _curriculum_events.append({
                "epoch": ep + 1, "type": "difficulty_schedule_floor",
                "from_stage": int(_active_diff_stage), "to_stage": int(_floor_diff),
            })
            _active_diff_stage = _floor_diff
            _epochs_since_advance = 0

        curr_seq_len = _active_seq_len

        if _last_logged_seq_len != curr_seq_len:

            _log_info(f"[Curriculum] Epoch {ep+1}: active seq_len={curr_seq_len}")
            try:
                from data.dataset_manifest import DatasetManifest
                _dm2 = DatasetManifest(str(Path(cache_path).parent))
                _unfrozen = [g for g, v in _feat_groups.items()
                             if not v.get("always_on", True)
                             and ep >= v.get("epoch_unfreeze", 999)]
                _dm2.log_curriculum_stage(
                    ep + 1, "seq_len_advance", curr_seq_len,
                    _unfrozen, int(_active_diff_stage),
                )
            except Exception:
                pass

            _last_logged_seq_len = curr_seq_len

        # -- A4: Feature freeze schedule ---------------------------------------
        _unfreeze_features_for_epoch(model, ep)

        # -- A: Feature stability monitoring -----------------------------------
        # Sample one batch from the training data to update the EMA stats.
        # We use a fixed random subset (512 samples) for efficiency ΓÇö no need
        # to scan the whole dataset; EMA smooths over batch-to-batch noise.
        try:
            _stab_pool  = locals().get("ep_train_idx", train_idx)
            _sample_idx = np.random.choice(_stab_pool,
                                           size=min(512, len(_stab_pool)), replace=False)
            _samp_ds = ZarrStreamDataset(cache_path, _sample_idx, shuffle_chunks=False)
            _samp_dl = DataLoader(_samp_ds, batch_size=512, shuffle=False, num_workers=0)
            _samp_xb, _ = next(iter(_samp_dl))
            _feat_stability.update(_samp_xb.numpy())
            _feat_mask = _feat_stability.get_mask(device=device)
            _stab_report = _feat_stability.report()

            if _stab_report["feat_frozen"] > 0 or _stab_report["feat_noisy"] > 0:
                _log_info(
                    f"[FeatStab] Ep {ep+1}: frozen={_stab_report['feat_frozen']} "
                    f"noisy={_stab_report['feat_noisy']} "
                    f"active={_stab_report['feat_active']}/{n_features} "
                    f"max_shift={_stab_report['feat_max_shift']:.2f}sigma"
                )
            _safe_wandb_log(run, {
                "feat/frozen": _stab_report["feat_frozen"],
                "feat/noisy": _stab_report["feat_noisy"],
                "feat/max_shift": _stab_report["feat_max_shift"],
                "epoch": ep,
            })
        except Exception as _stab_exc:
            _log_warn(f"[FeatStab] Monitor update failed (skipped): {_stab_exc}")
            _feat_mask = None   # fall back to no masking

        # -- Feature Curriculum Mask -------------------------------------------
        try:
            from config.feature_mask import FEATURE_MASK
            _schema = _feature_schema or [k for k, v in FEATURE_MASK.items() if v]
            if len(_schema) == n_features:
                _curr_mask = torch.ones(n_features, device=device)
                _zeroed_cnt = 0
                for g_name, g_cfg in _feat_groups.items():
                    if not g_cfg.get("always_on", True) and ep < g_cfg.get("epoch_unfreeze", 0):
                        for f_name in g_cfg.get("features", []):
                            if f_name in _schema:
                                _curr_mask[_schema.index(f_name)] = 0.0
                                _zeroed_cnt += 1
                if _zeroed_cnt > 0:
                    _log_info(f"[Curriculum] Epoch {ep+1}: Zeroed out {_zeroed_cnt} features by schema")
                    if _feat_mask is None:
                        _feat_mask = _curr_mask
                    else:
                        _feat_mask = _feat_mask * _curr_mask
            else:
                _log_warn(f"[Curriculum] Feature schema length {len(_schema)} does not match n_features={n_features}; skipping feature-group mask.")
        except Exception as _curr_exc:
            _log_warn(f"[Curriculum] Mask generation failed: {_curr_exc}")

        # Reset frozen features if difficulty stage increased
        # -- B: Difficulty curriculum ΓÇö rebuild dataloader with filtered indices --
        ep_train_idx = _apply_difficulty_filter(train_idx, ep)
        _direction_warmup_active = bool(
            use_direction_targets
            and multitask
            and ep < max(0, int(getattr(args, "direction_warmup_epochs", 2)))
        )
        if _direction_warmup_active:
            ep_train_idx = _balanced_direction_indices(
                cache_path,
                ep_train_idx,
                total_samples=len(ep_train_idx),
                seed=int(getattr(args, "seed", 1337)) + ep,
            )
            _log_info(
                f"[DirectionWarmup] Epoch {ep+1}: balanced direction-only batches "
                f"({len(ep_train_idx):,} samples)"
            )

        epoch_train_dl = train_dl
        # ── Online hard-example oversampling ──────────────────────────────
        if _online_miner is not None and ep > 0:
            ep_train_idx = _online_miner.get_oversampled_indices(
                ep_train_idx,
                hard_factor=1.5,
                forgotten_factor=2.0,
                easy_downsample=True,
            )
        # ──────────────────────────────────────────────────────────────────

        if len(ep_train_idx) != len(train_idx) or _direction_warmup_active:
            _diff_stage = _difficulty_stage_for_epoch(ep)
            if len(ep_train_idx) != len(train_idx) and not _direction_warmup_active:
                _log_info(f"[DiffCurriculum] Epoch {ep+1}: stage={_diff_stage} "
                          f"({len(ep_train_idx):,}/{len(train_idx):,} samples)")
            _ep_ds = ZarrStreamDataset(
                cache_path, ep_train_idx, shuffle_chunks=True, multitask_targets=use_direction_targets,
            )
            epoch_train_dl = DataLoader(
                _ep_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=nw, pin_memory=pin_mem,
                persistent_workers=(use_persistent and nw > 0),
                prefetch_factor=pf if nw > 0 else None,
            )
            if nw == 0:
                epoch_train_dl = _ThreadPrefetchLoader(
                    epoch_train_dl, prefetch=max(1, int(getattr(args, "thread_prefetch_batches", 2)))
                )

        # Batch-level progress bars
        train_pbar = _pbar(total=len(epoch_train_dl), desc=f"  Ep {ep+1:3d} [Tr]", unit="batch", leave=False)
        val_pbar   = _pbar(total=len(val_dl),   desc=f"  Ep {ep+1:3d} [Va]", unit="batch", leave=False)

        t0 = time.time()
        _thermal_lim = int(_GPU_CFG.get("thermal_limit_celsius", 83))

        # ── Online miner: begin epoch (roll buffer) ──────────────────────
        if _online_miner is not None:
            _online_miner.begin_epoch()
        # ──────────────────────────────────────────────────────────────────

        try:
            tl = train_epoch(
                model, epoch_train_dl, opt,
                direction_crit if _direction_warmup_active else crit,
                amp_sc, device, args.amp, classification,
                grad_clip=args.grad_clip, pbar=train_pbar,
                amp_dtype=amp_dtype,
                thermal_limit=_thermal_lim,
                feature_mask=_feat_mask,
                scheduler=scheduler if not (_swa_enabled and ep >= _swa_start_ep) else None,
                accum_steps=_accum,
                seq_len=curr_seq_len,
                multitask=multitask,
                epoch=ep,
                teacher_model=teacher_model,
                distill_weight=distill_weight,
                direction_only=_direction_warmup_active,
                online_miner=_online_miner,
            )
        except Exception as _epoch_exc:
            _log_error(f"[Train] Epoch {ep+1} failed for {model_name}", _epoch_exc)
            if _TRAIN_LOGGER is not None:
                _TRAIN_LOGGER.on_epoch_failure(ep, _epoch_exc)
            _save_crash_ckpt(ep, _epoch_exc)
            raise
        train_pbar.close()

        # Finish async CUDA work and free cached blocks before validation. Val uses
        # pin_memory=False, but sync+gc still reduces fragmentation after train_epoch.
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

        try:
            vl, da, v_sh = validate_epoch(
                model, val_dl, crit, device, classification, pbar=val_pbar,
                amp=args.amp, amp_dtype=amp_dtype,
                seq_len=curr_seq_len, multitask=multitask,
                feature_mask=_feat_mask,
                sharpe_ann_factor=_sharpe_ann_factor(args),
            )
            _class_counts = getattr(validate_epoch, "last_class_counts", {"pred": [0, 0, 0], "true": [0, 0, 0]})

        except Exception as _val_exc:
            val_pbar.close()
            _log_error(f"[Train] Epoch {ep+1} validation failed for {model_name}", _val_exc)
            _save_crash_ckpt(ep, _val_exc)
            raise
        val_pbar.close()

        # ── Online miner: end epoch (update forgetting tracker) ──────────
        if _online_miner is not None:
            _online_miner.end_epoch()
        # ──────────────────────────────────────────────────────────────────

        # SWA: accumulate averaged weights; use constant SWA LR instead of OneCycleLR
        if _swa_enabled and ep >= _swa_start_ep:
            if not _swa_started:
                try:
                    from torch.optim.swa_utils import AveragedModel, SWALR
                    _swa_model = AveragedModel(_core_model(model)).to(device)
                    _swa_scheduler = SWALR(opt, swa_lr=_swa_lr,
                                           anneal_epochs=max(1, int(args.epochs * 0.05)),
                                           anneal_strategy="cos")
                    _swa_started = True
                    print(f"\n[SWA] Weight averaging started at epoch {ep+1}")
                except Exception as _e:
                    print(f"\n[SWA] Failed to initialize at epoch {ep+1}: {_e}")
                    _swa_enabled = False

            if _swa_enabled:
                _swa_model.update_parameters(model)
                _swa_scheduler.step()

        lr = opt.param_groups[0]["lr"]
        el = time.time() - t0
        gm = torch.cuda.max_memory_allocated(device) // 1_000_000 if device.type=="cuda" else 0
        torch.cuda.reset_peak_memory_stats(device) if device.type=="cuda" else None

        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["dir_acc"].append(da);    history["lr"].append(lr)
        history["val_sharpe"].append(v_sh)
        history.setdefault("sharpe_ema", []).append(float(_sharpe_ema) if _sharpe_ema is not None else v_sh)
        history.setdefault("seq_len", []).append(int(_active_seq_len))
        history.setdefault("difficulty_stage", []).append(int(_active_diff_stage))
        history.setdefault("curriculum_stalls", []).append(int(_curriculum_stalls))
        history.setdefault("val_pred_counts", []).append(
            [int(x) for x in _class_counts.get("pred", [0, 0, 0])]
        )
        history.setdefault("val_true_counts", []).append(
            [int(x) for x in _class_counts.get("true", [0, 0, 0])]
        )

        
        # Adaptive Curriculum Update (P1: unified collapse detector)
        _v_sh_history.append(v_sh)
        if len(_v_sh_history) >= 2:
            _prev_peak_sharpe = max(_v_sh_history[:-1])
            _meaningful_peak  = _prev_peak_sharpe >= _adapt_collapse_min_peak

            # Condition A ΓÇö sustained drop from peak (Optuna-tunable threshold)
            _peak_drop_collapse = v_sh < (_prev_peak_sharpe - _adapt_collapse_drop)

            # Condition B ΓÇö sharp single-epoch reversal from positive to deeply negative
            # (was hardcoded to -0.1; now driven by collapse_reversal_threshold in YAML)
            _sharp_reversal = (
                len(_rolling_sharpes) >= 1
                and v_sh < _adapt_reversal_threshold
                and _rolling_sharpes[-1] > _adapt_min_stable_sharpe
            )

            _should_stall = _meaningful_peak and (_peak_drop_collapse or _sharp_reversal)
            _collapse_reason = (
                "peak_drop_and_reversal" if (_peak_drop_collapse and _sharp_reversal)
                else "peak_drop" if _peak_drop_collapse
                else "sharp_reversal" if _sharp_reversal
                else None
            )

            if _should_stall:
                _log_warn(
                    f"[Curriculum] Sharpe collapsed to {v_sh:.3f} "
                    f"(peak={_prev_peak_sharpe:.3f}, reason={_collapse_reason}). "
                    "Stalling curriculum progression."
                )
                _curriculum_stalls += 1
                _post_stall_stable_count = 0
                _curriculum_events.append({
                    "epoch": ep + 1,
                    "type": "stall",
                    "reason": _collapse_reason,
                    "val_sharpe": float(v_sh),
                    "previous_peak_sharpe": float(_prev_peak_sharpe),
                    "collapse_min_peak": float(_adapt_collapse_min_peak),
                    "collapse_drop": float(_adapt_collapse_drop),
                    "reversal_threshold": float(_adapt_reversal_threshold),
                    "seq_len": int(_active_seq_len),
                    "difficulty_stage": int(_active_diff_stage),
                    "lr_mult": float(_adapt_collapse_lr_mult),
                })
                _rolling_sharpes.clear()
                _seq_frozen = True
                for param_group in opt.param_groups:
                    param_group['lr'] *= _adapt_collapse_lr_mult
                if scheduler is not None:
                    for attr in ["base_lrs", "initial_lrs", "max_lrs", "min_lrs"]:
                        if hasattr(scheduler, attr):
                            setattr(scheduler, attr, [v * _adapt_collapse_lr_mult
                                                       for v in getattr(scheduler, attr)])

            elif v_sh < (_prev_peak_sharpe - 0.05):
                _log_info(
                    f"[Curriculum] Minor Sharpe dip to {v_sh:.3f} "
                    f"(peak={_prev_peak_sharpe:.3f}); no stall (below collapse thresholds)."
                )



        # GPU temp for display
        try:
            import pynvml as _pnvml
            _pnvml.nvmlInit()
            _h = _pnvml.nvmlDeviceGetHandleByIndex(0)
            _gpu_temp = int(_pnvml.nvmlDeviceGetTemperature(_h, _pnvml.NVML_TEMPERATURE_GPU))
        except Exception:
            _gpu_temp = -1

        _ep_metrics = {
            "train_loss": tl, "val_loss": vl, "dir_acc": da,
            "val_sharpe": v_sh, "lr": lr, "gpu_mb": gm,
            "val_pred_counts": _class_counts.get("pred", [0, 0, 0]),

            "val_true_counts": _class_counts.get("true", [0, 0, 0]),

            "gpu_temp_c": _gpu_temp,
            "oom_skips":  getattr(_TRAIN_LOGGER, "_ep_oom_count", 0) if _TRAIN_LOGGER else 0,
            "nan_skips":  getattr(_TRAIN_LOGGER, "_ep_nan_count", 0) if _TRAIN_LOGGER else 0,
            "elapsed_s":  el,
        }

        if _TRAIN_LOGGER is not None:
            _TRAIN_LOGGER.on_epoch_end(ep, _ep_metrics)

        _gate_ep = start_ep + max(0, int(getattr(args, "direction_warmup_epochs", 2))) - 1
        if multitask and ep == max(start_ep, _gate_ep):
            _class_diag = getattr(validate_epoch, "last_class_diag", {
                "pred": _class_counts.get("pred", [0, 0, 0]),
                "true": _class_counts.get("true", [0, 0, 0]),
            })
            _failed, _reason = _direction_gate_failed(_class_diag, args)
            if _failed:
                _diag_path = _write_class_balance_failure(_run_name, model_name, ep, _class_diag, _reason)
                raise RuntimeError(
                    "[ClassBalance] Direction warmup gate failed: "
                    f"{_reason}. pred S/H/B={_class_diag.get('pred')}, "
                    f"pred_shares={[round(float(s), 4) for s in _class_diag.get('pred_shares', [])]}, "
                    f"true S/H/B={_class_diag.get('true')}, "
                    f"recall={[round(float(r), 4) for r in _class_diag.get('recall', [])]}. "
                    f"Diagnostics -> {_diag_path}"
                )
            
        try:
            from infrastructure.ollama_helper import ollama
            ollama.monitor_training(ep + 1, float(tl), float(vl), _ep_metrics)
        except Exception as _oe:
            pass

        # -- TensorBoard -------------------------------------------------------
        if _tb_writer is not None:
            _tb_writer.add_scalar("Loss/train",       tl,   ep)
            _tb_writer.add_scalar("Loss/val",         vl,   ep)
            _tb_writer.add_scalar("Metrics/dir_acc",  da,   ep)
            _tb_writer.add_scalar("Metrics/sharpe",     v_sh, ep)
            if _sharpe_ema is not None:
                _tb_writer.add_scalar("Metrics/sharpe_ema", float(_sharpe_ema), ep)
            _tb_writer.add_scalar("Curriculum/epochs_since_advance", int(_epochs_since_advance), ep)
            _tb_writer.add_scalar("ValPred/sell",     _class_counts.get("pred", [0, 0, 0])[0], ep)

            _tb_writer.add_scalar("ValPred/hold",     _class_counts.get("pred", [0, 0, 0])[1], ep)

            _tb_writer.add_scalar("ValPred/buy",      _class_counts.get("pred", [0, 0, 0])[2], ep)

            _tb_writer.add_scalar("Train/lr",         lr,   ep)
            _tb_writer.add_scalar("Curriculum/seq_len", int(_active_seq_len), ep)

            _tb_writer.add_scalar("Curriculum/difficulty_stage", int(_active_diff_stage), ep)

            _tb_writer.add_scalar("Curriculum/stalls", int(_curriculum_stalls), ep)

            _tb_writer.add_scalar("GPU/mem_mb",       gm,   ep)
            if _gpu_temp > 0:
                _tb_writer.add_scalar("GPU/temp_c",   _gpu_temp, ep)
            if fold_id is not None:
                _tb_writer.add_scalar(f"Fold{fold_id}/val_sharpe", v_sh, ep)

        # -- W&B --------------------------------------------------------------
        if WANDB and run:
            _pred_counts = _class_counts.get("pred", [0, 0, 0])
            _true_counts = _class_counts.get("true", [0, 0, 0])
            _safe_wandb_log(run, {
                "train/loss": tl, "val/loss": vl, "val/dir_acc": da,
                "val/sharpe_proxy": v_sh, "train/lr": lr, "gpu_mb": gm, "epoch": ep,
                "val_pred/sell": _pred_counts[0], "val_pred/hold": _pred_counts[1],
                "val_pred/buy": _pred_counts[2],
                "val_true/sell": _true_counts[0], "val_true/hold": _true_counts[1],
                "val_true/buy": _true_counts[2],
                "curriculum/seq_len": int(_active_seq_len),
                "curriculum/difficulty_stage": int(_active_diff_stage),
                "curriculum/stalls": int(_curriculum_stalls),
                "curriculum/sharpe_ema": float(_sharpe_ema) if _sharpe_ema is not None else v_sh,
                "curriculum/epochs_since_advance": int(_epochs_since_advance),
                **({"fold": fold_id} if fold_id is not None else {}),
            })

        # -- Early stopping / is best? (Moved up to fix UnboundLocalError) ------
        min_delta = float(getattr(args, "early_stop_min_delta", 0.0))
        improved = (v_sh > (best_sharpe + min_delta)) if stop_on_sharpe else (vl < (best_val_loss - min_delta))
        
        if improved:
            if stop_on_sharpe:
                best_sharpe = v_sh
            else:
                best_val_loss = vl
            no_improve = 0
        else:
            # Suppress patience counter during LR warmup: the LR is artificially
            # suppressed so Sharpe cannot improve consistently. Only start counting
            # after warmup_epochs have fully elapsed.
            _warmup_done = ep >= int(getattr(args, "lr_warmup_epochs", 0))
            if _warmup_done:
                no_improve += 1

        # -- Rich display or plain print ---------------------------------------
        if _rich_display is not None:
            _rich_display.end_epoch(ep, _ep_metrics,
                                    is_best=improved, no_improve=no_improve)
        else:
            print(f"{ep+1:>5} {tl:>11.6f} {vl:>11.6f} {da:>8.4f} {v_sh:>9.4f} "
                  f"{lr:>10.2e} {el:>6.1f}s {gm:>7.0f}M")

        # -- Adaptive Curriculum Update --
        # SYS-005: Use train loss plateau as the primary curriculum progression signal
        # instead of val Sharpe. This prevents the validation set from influencing
        # training decisions (curriculum stage selection).
        # Val Sharpe is still used for collapse detection (safety mechanism) but NOT
        # for advancement gating.
        _curriculum_metric_source = str(getattr(args, "curriculum_gate_metric", "train_loss")).lower()
        if _curriculum_metric_source == "train_loss":
            # Gate on train loss EMA plateau: loss hasn't improved by >threshold for N epochs
            if _sharpe_ema is None:
                _sharpe_ema = -tl  # negate so "higher is better" logic still works
            else:
                _sharpe_ema = _adapt_ema_alpha * (-tl) + (1.0 - _adapt_ema_alpha) * _sharpe_ema
        else:
            # Legacy behavior: use val Sharpe (for backward compat if explicitly requested)
            if _sharpe_ema is None:
                _sharpe_ema = v_sh
            else:
                _sharpe_ema = _adapt_ema_alpha * v_sh + (1.0 - _adapt_ema_alpha) * _sharpe_ema

        # Rolling window uses the EMA value for stability judgement; raw v_sh feeds collapse detector.
        _rolling_sharpes.append(_sharpe_ema)
        if len(_rolling_sharpes) > _adapt_stable_window:
            _rolling_sharpes.pop(0)

        # P1: track cooldown
        _epochs_since_advance += 1

        # P1: Recovery — after recovery_window consecutive stable epochs post-stall, unfreeze.
        # SYS-005: when gating on train_loss, "stable" means loss has plateaued
        # (range of recent values is small relative to mean).
        if _curriculum_metric_source == "train_loss" and len(_rolling_sharpes) == _adapt_stable_window:
            _window_range = max(_rolling_sharpes) - min(_rolling_sharpes)
            _window_mean = abs(sum(_rolling_sharpes) / len(_rolling_sharpes))
            _plateau_threshold = max(1e-4, _window_mean * 0.02)
            _window_stable = _window_range < _plateau_threshold
        else:
            _window_stable = (
                len(_rolling_sharpes) == _adapt_stable_window
                and all(s > _adapt_min_stable_sharpe for s in _rolling_sharpes)
            )
        if _seq_frozen and _window_stable:
            _post_stall_stable_count += 1
            if _post_stall_stable_count >= _adapt_recovery_window:
                _seq_frozen = False
                _post_stall_stable_count = 0
                _log_info(
                    f"[Curriculum] Recovery: {_adapt_recovery_window} consecutive stable "
                    f"epochs after stall ΓÇö curriculum unfrozen at ep {ep+1}."
                )
                _curriculum_events.append({
                    "epoch": ep + 1, "type": "recovery",
                    "recovery_window": int(_adapt_recovery_window),
                    "rolling_sharpe": [float(s) for s in _rolling_sharpes],
                    "seq_len": int(_active_seq_len),
                    "difficulty_stage": int(_active_diff_stage),
                })
        elif not _seq_frozen and not _window_stable:
            _post_stall_stable_count = 0  # reset when window breaks outside a freeze

        # P1: Performance gating with cooldown guard
        _cooldown_ok = _epochs_since_advance >= _adapt_min_epochs_per_stage
        if _window_stable and _cooldown_ok and not _seq_frozen:

            # Evaluate difficulty advance (only up to schedule's next floor ΓÇö never skip a stage)
            next_diff = _active_diff_stage
            for entry in _diff_schedule:
                if int(entry["max_difficulty"]) > _active_diff_stage:
                    next_diff = int(entry["max_difficulty"])
                    break

            # Evaluate seq_len advance (next target above current; schedule floors already applied)
            next_seq = _active_seq_len
            for entry in _seq_schedule:
                if int(entry["seq_len"]) > _active_seq_len:
                    next_seq = int(entry["seq_len"])
                    break

            if next_diff > _active_diff_stage:
                _prev_diff_stage = _active_diff_stage
                _stable_window_values = [float(s) for s in _rolling_sharpes]
                _active_diff_stage = next_diff
                _epochs_since_advance = 0
                _rolling_sharpes.clear()
                _log_info(f"[DiffCurriculum] Rolling Sharpe stable. Advancing difficulty to stage {next_diff}.")
                _curriculum_events.append({
                    "epoch": ep + 1,
                    "type": "difficulty_increase",
                    "from_stage": int(_prev_diff_stage),
                    "to_stage": int(next_diff),
                    "rolling_sharpe": _stable_window_values,
                    "stable_window": int(_adapt_stable_window),
                    "min_stable_sharpe": float(_adapt_min_stable_sharpe),
                    "lr_mult": float(_adapt_advance_lr_mult),
                })
                for param_group in opt.param_groups:
                    param_group['lr'] *= _adapt_advance_lr_mult
                if scheduler is not None:
                    for attr in ["base_lrs", "initial_lrs", "max_lrs", "min_lrs"]:
                        if hasattr(scheduler, attr):
                            setattr(scheduler, attr, [v * _adapt_advance_lr_mult
                                                       for v in getattr(scheduler, attr)])
                _log_info(f"[DiffCurriculum] Regime changed: LR multiplied by {_adapt_advance_lr_mult:.3f}.")

            elif next_seq > _active_seq_len:
                _prev_seq_len = _active_seq_len
                _stable_window_values = [float(s) for s in _rolling_sharpes]
                _active_seq_len = next_seq
                _epochs_since_advance = 0
                _rolling_sharpes.clear()
                _log_info(f"[DiffCurriculum] Rolling Sharpe stable. Advancing seq_len to {next_seq}.")
                _curriculum_events.append({
                    "epoch": ep + 1,
                    "type": "seq_len_increase",
                    "from_seq_len": int(_prev_seq_len),
                    "to_seq_len": int(next_seq),
                    "rolling_sharpe": _stable_window_values,
                    "stable_window": int(_adapt_stable_window),
                    "min_stable_sharpe": float(_adapt_min_stable_sharpe),
                })



        if improved:
            core = _core_model(model)
            ckpt_meta = {
                "model_name": model_name,
                "n_features": n_features,
                "seq_len": int(curr_seq_len),
                "schema_hash": getattr(args, "feature_schema_hash", "unknown"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "fold_id": fold_id
            }
            _safe_save(core.state_dict(), best_path, metadata=ckpt_meta)
            with open(cfg_path, "w", encoding="utf-8") as _cfg_fp:
                json.dump({
                    "model": model_name, "n_features": n_features,
                    "seq_len": args.seq_len, "d_model": args.d_model,
                    "nhead": args.nhead, "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers, "dropout": args.dropout,
                    "best_val_loss": vl, "best_val_sharpe_proxy": v_sh,
                    "best_train_loss": tl,
                    "train_val_loss_gap": float(vl - tl),
                    "early_stop_metric": args.early_stop_metric,
                    "epoch": ep, "n_samples": n_samples, "loss": args.loss,
                    "fold_id": fold_id,
                }, _cfg_fp, indent=2)
            _safe_wandb_summary_update(run, {
                "best_val_loss": vl, "best_val_sharpe_proxy": v_sh, "best_epoch": ep,
            })

        if (ep+1) % args.save_every == 0:
            core = _core_model(model)
            ep_tag = f"{fold_suffix}_ep{ep+1}" if fold_suffix else f"_ep{ep+1}"
            _safe_save(core.state_dict(), ckpt_dir / f"{model_name}{ep_tag}.pt", metadata={
                "model_name": model_name,
                "n_features": n_features,
                "seq_len": int(curr_seq_len),
                "schema_hash": getattr(args, "feature_schema_hash", "unknown"),
                "epoch": ep + 1,
                "fold_id": fold_id,
            })

        # Exact resume checkpoint (model + optimizer + scheduler + AMP scaler + history)
        core = _core_model(model)
        _safe_save({
            "epoch": ep,
            "model_state": core.state_dict(),
            "opt_state": opt.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": amp_sc.state_dict() if args.amp and device.type == "cuda" else None,
            "best_val_loss": best_val_loss,
            "best_sharpe": best_sharpe,
            "no_improve": no_improve,
            "history": history,
            "fold_id": fold_id,
            "chunk_sharpe_history": _chunk_sharpe_history,
            "chunk_worse_streak":   _chunk_worse_streak,
            "feat_stability_state": _feat_stability.get_state(),
            "curriculum_state": {
                "seq_frozen": _seq_frozen,
                "rolling_sharpes": [float(s) for s in _rolling_sharpes],
                "epochs_since_advance": int(_epochs_since_advance),
                "post_stall_stable_count": int(_post_stall_stable_count),
                "sharpe_ema": float(_sharpe_ema) if _sharpe_ema is not None else None,
                "active_seq_len": int(_active_seq_len),
                "active_diff_stage": int(_active_diff_stage),
                "curriculum_stalls": int(_curriculum_stalls),
                "curriculum_events": _curriculum_events,
                "v_sh_history": [float(s) for s in _v_sh_history],
            },
        }, last_path, metadata={
            "model_name": model_name,
            "n_features": n_features,
            "seq_len": int(curr_seq_len),
            "schema_hash": getattr(args, "feature_schema_hash", "unknown"),
            "epoch": ep + 1,
            "fold_id": fold_id,
            "checkpoint_type": "resume",
        })

        if no_improve >= args.patience:
            print(f"\n[Train] Early stop (patience={args.patience}, "
                  f"metric={args.early_stop_metric})")
            break

        # -- A2: Chunk-level early stopping ------------------------------------
        # Track val Sharpe across epochs; abort if it has dropped for K consecutive epochs.
        _chunk_sharpe_history.append(v_sh)
        if len(_chunk_sharpe_history) >= _chunk_patience + 1:
            recent  = _chunk_sharpe_history[-_chunk_patience:]
            prior   = _chunk_sharpe_history[-(  _chunk_patience + 1)]
            if all(s < prior for s in recent):
                _chunk_worse_streak += 1
            else:
                _chunk_worse_streak = 0
            if _chunk_worse_streak >= _chunk_patience:
                msg = (f"[ChunkEarlyStop] Val Sharpe dropped {_chunk_patience} consecutive "
                       f"epochs (last={v_sh:.4f}). Aborting epoch loop.")
                print(f"\n{msg}"); _log_warn(msg)
                break

    # -- SWA: fix batch-norm running stats and save final averaged model -------
    if _swa_enabled and _swa_started:
        print("\n[SWA] Updating batch-norm running statistics...")
        try:
            _swa_update_bn(_bn_train_dl, _swa_model, device=device)
            print("[SWA] BN update complete.")
        except Exception as _swa_bn_e:
            print(f"[SWA] BN update warning (non-fatal): {_swa_bn_e}")
        _swa_path = ckpt_dir / f"{model_name}{fold_suffix}_swa.pt"
        _safe_save(_swa_model.module.state_dict(), _swa_path)
        print(f"[SWA] Averaged model saved ΓåÆ {_swa_path}")

    # -- D: Post-training temperature calibration ------------------------------
    if getattr(args, "calibrate", False):
        print("\n[Calibration] Fitting temperature scaler on val set...")
        try:
            # Reload best weights before calibrating
            core = _core_model(model)
            core.load_state_dict(torch.load(best_path, map_location=device))
            cal_model = TemperatureScaler(core).to(device)
            calibrate_as_classification = bool(classification or multitask)
        # Use tune_idx to prevent calibration leakage if available
        if getattr(args, "_tune_eval_idx", None) is not None:
            cal_ds = ZarrStreamDataset(cache_path, args._tune_eval_idx, shuffle_chunks=False, multitask_targets=multitask)
            cal_dl = DataLoader(cal_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
            cal_model.calibrate(cal_dl, device, classification=calibrate_as_classification)
        else:
            print("[Warning] No tune_idx found. Calibrating on val_dl (Data Leakage Risk!).")
            cal_model.calibrate(val_dl, device, classification=calibrate_as_classification)

        cal_path = ckpt_dir / f"{model_name}{fold_suffix}_calibrated.pt"
            _safe_save({"model_state": core.state_dict(),
                        "temperature": cal_model.temperature.item(),
                        "classification": calibrate_as_classification}, cal_path)
            print(f"[Calibration] Saved calibrated model -> {cal_path}")

            # INF-004: Emit calibration metadata sidecar
            try:
                _cal_temp = float(cal_model.temperature.item())
                _cal_report = {
                    "temperature": _cal_temp,
                    "classification": calibrate_as_classification,
                    "calibration_set_size": len(val_dl.dataset) if hasattr(val_dl, "dataset") else "unknown",
                    "model_name": model_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                _cal_report_path = ckpt_dir / "calibration_report.json"
                _safe_save_json(_cal_report, _cal_report_path)
                print(f"[Calibration] Report -> {_cal_report_path} (T={_cal_temp:.4f})")
            except Exception as _cr_e:
                print(f"[Calibration] Report write failed (non-fatal): {_cr_e}")

        except Exception as _cal_e:
            _log_warn(f"[Calibration] Failed: {_cal_e}")

    if history["val_sharpe"]:
        _best_ep = int(history["val_sharpe"].index(max(history["val_sharpe"]))) if stop_on_sharpe else int(history["val_loss"].index(min(history["val_loss"])))
    else:
        _best_ep = 0
    _best_met = best_sharpe if stop_on_sharpe else best_val_loss

    if _TRAIN_LOGGER is not None:
        _TRAIN_LOGGER.on_training_complete(
            best_epoch  = _best_ep,
            best_metric = _best_met,
            total_s     = time.time() - _t_start,
            metric_name = "val_sharpe" if stop_on_sharpe else "val_loss",
        )

    if _rich_display is not None:
        _rich_display.finish(best_epoch=_best_ep, best_metric=_best_met)
        _rich_display.__exit__(None, None, None)

    if _tb_writer is not None:
        _tb_writer.add_hparams(
            hparam_dict={
                "lr": args.lr, "batch_size": args.batch_size,
                "d_model": args.d_model, "num_layers": args.num_layers,
                "dropout": args.dropout, "loss": args.loss,
            },
            metric_dict={
                "hparam/best_val_sharpe": best_sharpe,
                "hparam/best_val_loss":   best_val_loss,
                "hparam/best_epoch":      float(_best_ep),
            },
        )
        _tb_writer.flush()
        _tb_writer.close()

    # -- Generate Training Control Report ---------------------------------------
    _control_report_path = ckpt_dir / f"{model_name}{fold_suffix}_training_control_report.json"
    _final_train_val_gap = history["val_loss"][-1] - history["train_loss"][-1] if history.get("val_loss") and history.get("train_loss") else 0.0
    _early_stopped = (no_improve >= args.patience)
    _best_idx = int(_best_ep) if _best_ep is not None and _best_ep >= 0 else 0

    def _hist_at(key: str, default=None):

        values = history.get(key) or []

        return values[_best_idx] if 0 <= _best_idx < len(values) else default

    _control_report = {
        "model_name": model_name,
        "fold": fold_id,
        "epochs_run": len(history.get("val_loss", [])),
        "best_epoch": _best_ep,
        "best_epoch_state": {

            "val_sharpe": float(_hist_at("val_sharpe", 0.0)),

            "dir_acc": float(_hist_at("dir_acc", 0.0)),

            "val_loss": float(_hist_at("val_loss", 0.0)),

            "train_loss": float(_hist_at("train_loss", 0.0)),

            "lr": float(_hist_at("lr", getattr(args, "lr", 0.0))),

            "seq_len": int(_hist_at("seq_len", _active_seq_len)),

            "difficulty_stage": int(_hist_at("difficulty_stage", _active_diff_stage)),

            "curriculum_stalls": int(_hist_at("curriculum_stalls", _curriculum_stalls)),

        },

        "final_train_val_gap": _final_train_val_gap,
        "early_stopped": _early_stopped,
        "overfitting_warnings": [],
        "final_seq_len": int(_active_seq_len),

        "final_difficulty_stage": int(_active_diff_stage),

        "adaptation_config": {

            "stable_window": int(_adapt_stable_window),

            "min_stable_sharpe": float(_adapt_min_stable_sharpe),

            "collapse_min_peak": float(_adapt_collapse_min_peak),

            "collapse_drop": float(_adapt_collapse_drop),

            "advance_lr_mult": float(_adapt_advance_lr_mult),

            "collapse_lr_mult": float(_adapt_collapse_lr_mult),

        },

        "curriculum_stalls": _curriculum_stalls,
        "curriculum_events": _curriculum_events
    }
    if _final_train_val_gap > 0.05:
        _control_report["overfitting_warnings"].append(f"High train-val gap: {_final_train_val_gap:.4f}")
    if len(history.get("val_sharpe", [])) > 5:
        max_sh = max(history["val_sharpe"])
        final_sh = history["val_sharpe"][-1]
        if max_sh - final_sh > 0.3:
            _control_report["overfitting_warnings"].append(f"Sharpe collapsed by {max_sh - final_sh:.3f} from peak")
    
    try:
        _safe_save_json(_control_report, _control_report_path)
        print(f"[TrainingControl] Saved report -> {_control_report_path}")
    except Exception as e:
        print(f"[TrainingControl] Failed to save report: {e}")

    if stop_on_sharpe:
        print(f"\n[Train] Best val Sharpe (proxy): {best_sharpe:.4f}  ->  {best_path}")
        
        if getattr(args, "ollama_auto_tune", False):
            try:
                from infrastructure.ollama_helper import ollama
                final_metrics = {"best_sharpe": float(best_sharpe), "best_val_loss": float(best_val_loss), "best_epoch": int(_best_ep)}
                ollama.auto_tune_model(model_name, final_metrics)
            except Exception:
                pass
            
        return history, best_sharpe
    print(f"\n[Train] Best val loss: {best_val_loss:.6f}  ->  {best_path}")
    return history, best_val_loss


# -----------------------------------------------------------------------------
# CONTRASTIVE PRE-TRAINING
# -----------------------------------------------------------------------------

def _pretrain_report_path(args) -> Path:
    return Path(getattr(args, "checkpoint_dir", ".")) / "pretrain_report.json"


def _read_json_dict(path: Path) -> dict:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _update_pretrain_report(args, updates: dict) -> None:
    path = _pretrain_report_path(args)
    report = _read_json_dict(path)
    report.update(updates or {})
    report["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        _safe_save_json(report, path)
    except NameError:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def _recommended_pretrain_method(model_name: str) -> str:
    name = str(model_name or "").lower()
    if "haelt" in name:
        return "masked_or_byol"
    if "mamba" in name:
        return "forecast_or_drift"
    if "tft" in name:
        return "masked"
    if "gnn" in name:
        return "cluster"
    if "expert" in name:
        return "tscl"
    if "transformer" in name:
        return "byol_or_tscl"
    return "masked"


def _fold_history_summary(folds: Optional[list], metric_name: str = "sharpe") -> dict:
    folds = folds or []
    best_metrics = []
    best_sharpes = []
    best_losses = []
    final_sharpes = []
    final_losses = []
    final_dir_acc = []
    final_gaps = []

    for entry in folds:
        if not isinstance(entry, dict):
            continue
        metric = entry.get("best_metric")
        if metric is not None:
            best_metrics.append(float(metric))
        hist = entry.get("history", {})
        if not isinstance(hist, dict):
            continue
        sharpe = hist.get("val_sharpe") or []
        loss = hist.get("val_loss") or []
        train_loss = hist.get("train_loss") or []
        dir_acc = hist.get("dir_acc") or []
        if sharpe:
            best_sharpes.append(float(max(sharpe)))
            final_sharpes.append(float(sharpe[-1]))
        if loss:
            best_losses.append(float(min(loss)))
            final_losses.append(float(loss[-1]))
        if dir_acc:
            final_dir_acc.append(float(dir_acc[-1]))
        if loss and train_loss:
            final_gaps.append(float(loss[-1]) - float(train_loss[-1]))

    maximize_metric = str(metric_name).lower() == "sharpe"
    best_metric = None
    if best_metrics:
        best_metric = max(best_metrics) if maximize_metric else min(best_metrics)
    return {
        "fold_count": len(folds),
        "best_metric": best_metric,
        "mean_best_metric": float(np.mean(best_metrics)) if best_metrics else None,
        "best_val_sharpe": float(max(best_sharpes)) if best_sharpes else None,
        "mean_best_val_sharpe": float(np.mean(best_sharpes)) if best_sharpes else None,
        "best_val_loss": float(min(best_losses)) if best_losses else None,
        "mean_best_val_loss": float(np.mean(best_losses)) if best_losses else None,
        "final_val_sharpe": float(np.mean(final_sharpes)) if final_sharpes else None,
        "final_val_loss": float(np.mean(final_losses)) if final_losses else None,
        "final_dir_acc": float(np.mean(final_dir_acc)) if final_dir_acc else None,
        "final_train_val_gap": float(np.mean(final_gaps)) if final_gaps else None,
    }


def _pretrain_ablation_verdict(baseline: dict, pretrained: dict) -> tuple[str, dict]:
    def _delta(key):
        a = pretrained.get(key)
        b = baseline.get(key)
        return (float(a) - float(b)) if a is not None and b is not None else None

    deltas = {
        "best_metric": _delta("best_metric"),
        "best_val_sharpe": _delta("best_val_sharpe"),
        "mean_best_val_sharpe": _delta("mean_best_val_sharpe"),
        "best_val_loss": _delta("best_val_loss"),
        "mean_best_val_loss": _delta("mean_best_val_loss"),
        "final_dir_acc": _delta("final_dir_acc"),
        "final_train_val_gap": _delta("final_train_val_gap"),
    }
    sharpe_delta = deltas.get("best_val_sharpe")
    loss_delta = deltas.get("best_val_loss")
    gap_delta = deltas.get("final_train_val_gap")
    if sharpe_delta is None and loss_delta is None:
        verdict = "unknown"
    elif (sharpe_delta is not None and sharpe_delta > 0.0) and (
        loss_delta is None or loss_delta <= 0.0
    ) and (gap_delta is None or gap_delta <= 0.02):
        verdict = "pretrain_helped"
    elif (sharpe_delta is not None and sharpe_delta < 0.0) or (
        loss_delta is not None and loss_delta > 0.0 and (sharpe_delta is None or sharpe_delta <= 0.0)
    ):
        verdict = "pretrain_hurt"
    else:
        verdict = "mixed"
    return verdict, deltas


def run_pretrain(model, cache_path, n_features, args, device, run=None):
    _method = _normalize_pretrain_method(
        str(getattr(args, "pretrain_method", PRETRAIN.get("method", "byol"))).lower()
    )
    if _method not in _VALID_PRETRAIN_METHODS:
        print(f"[Pretrain] WARN: unknown method={_method!r}; falling back to byol")
        _method = "byol"
    use_regime = getattr(args, "pretrain_regime", False) and _method == "tscl"
    _mode_labels = {
        "byol": "BYOL",
        "masked": "MaskedRecon",
        "vae": "VAE",
        "cluster": "ClusterTSCL",
        "forecast": "ForecastPretext",
        "drift": "DriftContrastive",
        "tscl": "RegimeAware-TSCL" if use_regime else "TSCL",
    }
    mode_str = _mode_labels.get(_method, "BYOL")
    target_epochs = max(1, int(getattr(args, "pretrain_epochs", 1)))
    max_epochs = int(getattr(args, "pretrain_max_epochs", 0) or 0)
    if max_epochs > 0:
        target_epochs = min(target_epochs, max_epochs)
    min_epochs = max(0, int(getattr(args, "pretrain_min_epochs", 0)))
    handoff_patience = max(0, int(getattr(args, "pretrain_handoff_patience", 0)))
    handoff_min_delta = float(getattr(args, "pretrain_handoff_min_delta", 0.0))
    handoff_loss = float(getattr(args, "pretrain_handoff_loss", float("-inf")))
    handoff_enabled = handoff_patience > 0 or handoff_loss > float("-inf")
    print(f"\n[Pretrain] {mode_str} | target_epochs={target_epochs}"
          f"{' (handoff enabled)' if handoff_enabled else ''}")

    # Reduce VRAM fragmentation (recommended by PyTorch OOM diagnostics)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    encoder = model.backbone if hasattr(model, "backbone") else model

    # --force-pretrain: wipe previous encoder checkpoint so we start fresh
    if getattr(args, "force_pretrain", False):
        for suffix in ("", "_regime"):
            old = Path(args.checkpoint_dir) / f"contrastive_encoder{suffix}.pt"
            if old.exists():
                old.unlink()
                print(f"[Pretrain] Deleted old checkpoint: {old}")

    # Cap windows-per-block to available RAM unless explicitly configured.
    # Reads are planned as several contiguous spans so zarr/memmap I/O remains
    # chunk-friendly while each epoch sees more of the timeline than one slice.
    _bytes_per_window = int(n_features) * args.seq_len * 4
    try:
        import psutil as _psutil
        _free = _psutil.virtual_memory().available
    except Exception:
        _free = 4 * 1024 ** 3                  # conservative fallback: 4 GiB
    _ram_budget = max(1 * 1024 ** 3, int(_free * 0.75))  # use 75% of free RAM
    auto_windows = min(100_000, max(512, _ram_budget // _bytes_per_window))
    n_windows = _coerce_auto_int(
        getattr(args, "pretrain_sample_windows", "auto"),
        auto_windows,
        minimum=128,
    )
    _seed = getattr(args, "seed", None)
    _rng = np.random.default_rng(_seed)

    if ZARR and cache_path.endswith(".zarr") and Path(cache_path).is_dir():
        _z      = _zarr_open_group(cache_path, mode="r")
        n_total = min(int(_z["X"].shape[0]), int(_z["y"].shape[0]))
        X_reader, y_reader = _z["X"], _z["y"]
    else:
        X_mmap   = np.load(_x_path(cache_path), mmap_mode="r")
        y_mmap   = np.load(_y_path(cache_path), mmap_mode="r")
        n_total = min(len(X_mmap), len(y_mmap))
        X_reader, y_reader = X_mmap, y_mmap

    _source_n_total = int(n_total)
    _pretrain_cap = _trainable_max_index(n_total, args)
    if 0 < _pretrain_cap < n_total:
        n_total = _pretrain_cap
        print(f"[Pretrain] Holdout-safe index cap: {n_total:,} trainable windows")

    n_windows = min(int(n_windows), int(n_total))
    diff_for_sampling = _load_diff_array(str(cache_path), n_total)
    _hard_examples_injected = 0

    def _sample_pretrain_block(desc: str = "[Pretrain] Loading spans"):
        nonlocal _hard_examples_injected
        spans = _make_pretrain_span_plan(
            n_total,
            n_windows,
            diff=diff_for_sampling,
            max_spans=8 if n_windows >= 1024 else 4,
            rng=_rng,
        )
        
        # Hard-example mining integration
        try:
            import json
            _he_path = Path("logs/hard_examples.json")
            if _he_path.exists():
                _he_data = json.loads(_he_path.read_text(encoding="utf-8"))
                _he_indices = _he_data.get("indices", [])
                if _he_indices:
                    # convert indices into single-item spans to inject
                    _valid_he = [i for i in _he_indices if 0 <= i < n_total]
                    _he_spans = [(i, i+1) for i in _valid_he[:int(n_windows * 0.2)]] # max 20% hard examples
                    if _he_spans:
                        print(f"[Pretrain] Injected {len(_he_spans):,} hard examples from {len(_he_indices):,} total.")
                        _hard_examples_injected += len(_he_spans)
                        spans.extend(_he_spans)
        except Exception as _he_e:
            print(f"[Pretrain] Hard example reuse failed: {_he_e}")

        return _read_pretrain_spans(
            X_reader, y_reader, spans,
            seq_len=args.seq_len,
            n_features=n_features,
            progress_desc=desc,
        )

    windows, y_sample = _sample_pretrain_block()

    print(f"[Pretrain] Sampled {len(windows):,} windows | shape {windows.shape[1:]}")

    ckpt   = str(Path(args.checkpoint_dir) / f"contrastive_encoder{'_regime' if use_regime else ''}.pt")
    _holdout_n = _promotion_holdout_n(_source_n_total, args)
    _embargo_n = _embargo_bars(args)
    _update_pretrain_report(args, {
        "model_name": getattr(args, "model", None),
        "pretrain_enabled": True,
        "status": "started",
        "method": _method,
        "recommended_method_for_model": _recommended_pretrain_method(getattr(args, "model", "")),
        "regime_aware": bool(use_regime),
        "cache_path": str(cache_path),
        "source_windows": int(_source_n_total),
        "trainable_windows_used_by_pretrain": int(n_total),
        "pretrain_window": {"start_index": 0, "end_index_exclusive": int(n_total)},
        "supervised_window": {
            "start_index": 0,
            "end_index_exclusive": int(max(0, _source_n_total - _holdout_n)),
            "embargo_bars": int(_embargo_n),
        },
        "promotion_holdout_window": {
            "start_index": int(max(0, _source_n_total - _holdout_n)),
            "end_index_exclusive": int(_source_n_total),
            "reserved_windows": int(_holdout_n),
        },
        "holdout_safe": bool(n_total <= max(0, _source_n_total - _holdout_n)),
        "sample_windows_per_block": int(n_windows),
        "loaded_into_supervised_training": False,
    })
    if getattr(args, "resume", False) and os.path.exists(ckpt):
        print(f"[Pretrain] Resume: skipping, loading existing checkpoint {Path(ckpt).name}")
        # A-H2: load with a report + assertion instead of silently swallowing
        # the exception (a failed load here would leave a random encoder).
        _enc = encoder.module if hasattr(encoder, "module") else encoder
        try:
            _state = torch.load(ckpt, map_location=device, weights_only=True)
        except Exception:
            _state = torch.load(ckpt, map_location=device)
        _strict_load_report(_enc, _state, "PretrainResume", min_frac_loaded=0.6)
        _update_pretrain_report(args, {
            "status": "resume_loaded_existing",
            "checkpoint_path": str(ckpt),
            "quality_gate_result": "loaded_existing_checkpoint",
        })
        return model
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # Compute a VRAM-safe batch size for the contrastive trainer.
    # With no_grad on pos/neg views, only 1 encoder pass retains activation graphs.
    # But the LSTM still allocates large hidden-state buffers proportional to
    # input_size ├ù hidden_size ├ù seq_len, so we budget conservatively.
    try:
        import torch as _torch
        _torch.cuda.synchronize(device)
        torch.cuda.empty_cache()   # reclaim any fragmented VRAM before pretraining
        _vram_total = _torch.cuda.get_device_properties(device).total_memory
        _vram_free = _vram_total - _torch.cuda.memory_allocated(device)
    except Exception:
        _vram_total = 8 * 1024 ** 3
        _vram_free = 4 * 1024 ** 3
    _bytes_per_sample = args.seq_len * n_features * 4     # FP32 input tensors
    # Single-pass methods: 1 grad pass. Cluster: 2. TSCL: 2ΓÇô3.
    if _method in _PRETRAIN_SINGLE_PASS:
        _n_passes = 1
        _safety_factor = 15
    elif _method == "cluster":
        _n_passes = 2
        _safety_factor = 20
    else:
        _n_passes = 3 if use_regime else 2
        _safety_factor = 30 if use_regime else 20
    _pt_bs_vram = max(16, int(_vram_free * 0.25) // (_bytes_per_sample * _n_passes * _safety_factor))
    _pt_bs_vram = (_pt_bs_vram // 8) * 8
    _vram_gb = _vram_total / (1024 ** 3)
    if _vram_gb <= 10:
        _max_bs = 256 if _method in _PRETRAIN_SINGLE_PASS else (128 if not use_regime else 64)
    elif _vram_gb <= 16:
        _max_bs = 512 if _method in _PRETRAIN_SINGLE_PASS else (256 if not use_regime else 128)
    else:
        _max_bs = 1024 if _method in _PRETRAIN_SINGLE_PASS else (512 if not use_regime else 256)
    _cfg_bs = int(getattr(args, "pretrain_batch", PRETRAIN.get("pretrain_batch", 256)))
    pt_bs = max(4, min(_cfg_bs, _pt_bs_vram, _max_bs))
    if _method == "tscl" and pt_bs < 32:
        print(
            f"[Pretrain] WARN: TSCL estimated batch_size={pt_bs} is too small for useful "
            "contrastive negatives; falling back to BYOL for this run."
        )
        _method = "byol"
        use_regime = False
        mode_str = "BYOL"
        _n_passes = 1
        _safety_factor = 15
        _pt_bs_vram = max(16, int(_vram_free * 0.25) // (_bytes_per_sample * _n_passes * _safety_factor))
        _pt_bs_vram = (_pt_bs_vram // 8) * 8
        _max_bs = 256 if _vram_gb <= 10 else (512 if _vram_gb <= 16 else 1024)
        pt_bs = max(4, min(_cfg_bs, _pt_bs_vram, _max_bs))
        ckpt = str(Path(args.checkpoint_dir) / "contrastive_encoder.pt")
    print(
        f"[Pretrain] method={_method} | VRAM {_vram_gb:.1f} GB | batch_size={pt_bs} "
        f"(configured={_cfg_bs}, budget={_pt_bs_vram}, cap={_max_bs})"
    )

    # Infer encoder output dim via dummy forward. Strip the prediction head first ΓÇö
    # same as BYOLTrainer / TSCLTrainer ΓÇö or backbones like GNN return (B,) logits
    # and we would mis-read encoder_dim as batch size (e.g. 1) instead of hidden dim.
    _saved_head = None
    if hasattr(encoder, "head"):
        _saved_head = encoder.head
        encoder.head = nn.Identity()
    try:
        with torch.no_grad():
            _dummy = torch.zeros(2, args.seq_len, n_features, device=device)
            _out   = encoder(_dummy)
            if _out.ndim == 2:
                encoder_dim = int(_out.shape[-1])
            elif _out.ndim >= 3:
                encoder_dim = int(_out[:, -1, :].shape[-1])
            elif _out.ndim == 1:
                # Multitask + GNN used to return (B,) when inner head was Identity but forward
                # still squeezed; fixed in GNNCrossAsset. Fallback: derive dim from multitask head_in.
                if getattr(args, "multitask", False) and getattr(args, "model", None):
                    n_pairs = getattr(args, "_n_pairs", 1)
                    f_per_pair = getattr(args, "_f_per_pair", n_features)
                    embed_dim = getattr(args, "pair_embed_dim", 0)
                    use_pair_emb = n_pairs > 1 and embed_dim > 0
                    if use_pair_emb:
                        _n_cross = n_pairs * (n_pairs - 1) // 2
                        _n_interaction = 3 * _n_cross + n_pairs + 2
                        backbone_input = n_pairs * (f_per_pair + embed_dim) + _n_interaction
                    else:
                        backbone_input = n_features
                    encoder_dim = _multitask_head_in(
                        str(args.model).lower(), args, backbone_input
                    )
                else:
                    raise RuntimeError(
                        f"[Pretrain] Encoder output is 1D {_out.shape} with head=Identity; "
                        "cannot infer representation width."
                    )
            else:
                raise RuntimeError(
                    f"[Pretrain] Unexpected encoder output ndim={_out.ndim} "
                    f"shape={tuple(_out.shape)}"
                )
    finally:
        if _saved_head is not None:
            encoder.head = _saved_head

    _pt_aug = _make_pretrain_augmenter(args, n_features)
    common = dict(
        d_model     = encoder_dim,
        proj_dim    = int(getattr(args, "pretrain_projection_dim", PRETRAIN.get("projection_dim", 256))),
        temperature = float(getattr(args, "pretrain_temperature", PRETRAIN.get("temperature", 0.5))),
        lr          = float(getattr(args, "pretrain_lr", PRETRAIN.get("pretrain_lr", 1e-4))),
        device      = str(device),
        seed        = _seed,
        aug         = _pt_aug,
    )

    def _fresh_windows():
        """Sample a fresh multi-span block for pretraining."""
        return _sample_pretrain_block()

    def _regime_labels(y, w=None):
        _fn = getattr(args, "_feat_names", None)
        if _fn is not None and w is not None:
            try:
                idx = _fn.index("regime_label")
                scores = w[:, -1, idx]
                return np.where(scores > 0.5, 1, np.where(scores < -0.5, -1, 0)).astype(np.int8)
            except ValueError:
                pass
        return np.where(y > 0.1, 1, np.where(y < -0.1, -1, 0)).astype(np.int8)

    # Build trainer
    trainer_cls = _select_pretrain_trainer_class(_method, use_regime)
    if trainer_cls is BYOLTrainer:
        trainer = BYOLTrainer(
            encoder    = encoder,
            d_model    = encoder_dim,
            proj_dim   = int(getattr(args, "pretrain_projection_dim", PRETRAIN.get("projection_dim", 256))),
            pred_dim   = int(getattr(args, "pretrain_pred_dim", 128)),
            ema_decay  = float(getattr(args, "pretrain_ema_decay", PRETRAIN.get("ema_decay", 0.996))),
            lr         = float(getattr(args, "pretrain_lr", PRETRAIN.get("pretrain_lr", 1e-4))),
            device     = str(device),
            seed       = _seed,
            aug        = _pt_aug,
        )
    elif trainer_cls is MaskedReconstructionTrainer:
        trainer = MaskedReconstructionTrainer(
            encoder=encoder,
            d_model=encoder_dim,
            seq_len=args.seq_len,
            n_features=n_features,
            hidden_dim=int(getattr(args, "pretrain_recon_hidden_dim", PRETRAIN.get("recon_hidden_dim", 512))),
            mask_prob=float(getattr(args, "pretrain_mask_prob", PRETRAIN.get("mask_prob", 0.20))),
            lr=float(getattr(args, "pretrain_lr", PRETRAIN.get("pretrain_lr", 1e-4))),
            device=str(device),
            seed=_seed,
        )
    elif trainer_cls is RegimeAwareTSCLTrainer:
        trainer = RegimeAwareTSCLTrainer(
            encoder=encoder,
            regime_labels=_regime_labels(y_sample, windows),
            hard_negative_weight=1.0,
            **common,
        )
    elif trainer_cls is VAESeqTrainer:
        trainer = VAESeqTrainer(
            encoder=encoder,
            d_model=encoder_dim,
            seq_len=args.seq_len,
            n_features=n_features,
            latent_dim=int(getattr(args, "pretrain_latent_dim", PRETRAIN.get("latent_dim", 64))),
            hidden_dim=int(getattr(args, "pretrain_recon_hidden_dim", PRETRAIN.get("recon_hidden_dim", 512))),
            beta=float(getattr(args, "pretrain_vae_beta", PRETRAIN.get("vae_beta", 0.001))),
            lr=float(getattr(args, "pretrain_lr", PRETRAIN.get("pretrain_lr", 1e-4))),
            device=str(device),
            seed=_seed,
        )
    elif trainer_cls is ClusterContrastiveTrainer:
        trainer = ClusterContrastiveTrainer(
            encoder=encoder,
            d_model=encoder_dim,
            proj_dim=int(getattr(args, "pretrain_projection_dim", PRETRAIN.get("projection_dim", 256))),
            n_clusters=int(getattr(args, "pretrain_n_clusters", PRETRAIN.get("n_clusters", 3))),
            temperature=float(getattr(args, "pretrain_temperature", PRETRAIN.get("temperature", 0.5))),
            lr=float(getattr(args, "pretrain_lr", PRETRAIN.get("pretrain_lr", 1e-4))),
            device=str(device),
            seed=_seed,
            aug=_pt_aug,
        )
    elif trainer_cls is ForecastPretextTrainer:
        trainer = ForecastPretextTrainer(
            encoder=encoder,
            d_model=encoder_dim,
            seq_len=args.seq_len,
            n_features=n_features,
            horizon=int(getattr(args, "pretrain_forecast_horizon", PRETRAIN.get("forecast_horizon", 5))),
            hidden_dim=int(getattr(args, "pretrain_recon_hidden_dim", PRETRAIN.get("recon_hidden_dim", 512))),
            lr=float(getattr(args, "pretrain_lr", PRETRAIN.get("pretrain_lr", 1e-4))),
            device=str(device),
            seed=_seed,
        )
    elif trainer_cls is DriftContrastiveTrainer:
        trainer = DriftContrastiveTrainer(
            encoder=encoder,
            d_model=encoder_dim,
            margin=float(getattr(args, "pretrain_drift_margin", PRETRAIN.get("drift_margin", 1.0))),
            lr=float(getattr(args, "pretrain_lr", PRETRAIN.get("pretrain_lr", 1e-4))),
            device=str(device),
            seed=_seed,
        )
    else:
        trainer = TSCLTrainer(encoder=encoder, **common)

    # Train epoch-by-epoch with fresh windows each time so the encoder sees
    # n_windows ├ù n_epochs unique samples instead of repeating the same n_windows.
    #
    # BYOL multi-block: each outer epoch runs _n_blocks independent blocks so the
    # encoder sees _n_blocks ├ù n_windows windows per epoch instead of just n_windows.
    # With n_windowsΓëê2270 (RAM limit on 16 GB), 3 blocks gives ~6,800 windows and
    # ~26 batches per epoch ΓÇö enough gradient signal to actually move the loss.
    auto_blocks = max(1, 6_000 // max(1, n_windows)) if _method in _PRETRAIN_MULTI_BLOCK else 1
    _n_blocks = _coerce_auto_int(
        getattr(args, "pretrain_blocks_per_epoch", "auto"),
        auto_blocks,
        minimum=1,
    ) if _method in _PRETRAIN_MULTI_BLOCK else 1
    effective_windows = int(_n_blocks * n_windows)
    print(
        f"[Pretrain] windows_per_block={n_windows:,} | blocks_per_epoch={_n_blocks} "
        f"| effective_windows_per_epoch={effective_windows:,}"
    )

    all_losses = []
    final_align = None
    final_unif = None
    final_embed_std = None
    _latest_diag = {}

    _best_loss = float("inf")
    _stale_epochs = 0
    _stopped_early = False
    try:
        _last_w = windows
        for _ep in range(target_epochs):
            if _method in _PRETRAIN_MULTI_BLOCK:
                # Multi-block: run _n_blocks fresh slices per epoch, silent except last
                _block_losses = []
                for _blk in range(_n_blocks):
                    _w, _y = _fresh_windows()
                    _last_w = _w
                    _is_last_block = (_blk == _n_blocks - 1)
                    _h = trainer.pretrain(
                        _w, epochs=1, batch_size=pt_bs,
                        checkpoint_path=ckpt,
                        silent=not _is_last_block,
                    )
                    _block_losses.append(_h["loss"][0])
                if hasattr(trainer, "save_encoder"):
                    trainer.save_encoder(ckpt)
                ls = sum(_block_losses) / len(_block_losses)
            else:
                _w, _y = _fresh_windows()
                _last_w = _w
                if use_regime:
                    rl = _regime_labels(_y, _w)
                    extreme_mask = rl != 0
                    if extreme_mask.any():
                        _ext_idx = np.flatnonzero(extreme_mask)
                        _max_extra = min(len(_ext_idx), max(256, min(len(_w) // 10, 4096)))
                        if _max_extra > 0:
                            _ext_idx = _ext_idx[:_max_extra]
                            _w = np.concatenate([_w, _w[_ext_idx]], axis=0)
                            _y = np.concatenate([_y, _y[_ext_idx]], axis=0)
                            rl = np.concatenate([rl, rl[_ext_idx]], axis=0)
                            print(f"[Pretrain] Regime oversample capped at {_max_extra:,} windows "
                                  f"(base={len(extreme_mask):,}, extreme={int(extreme_mask.sum()):,})")
                    trainer.regime_labels = rl
                _h = trainer.pretrain(_w, epochs=1, batch_size=pt_bs, checkpoint_path=ckpt)
                ls = _h["loss"][0]

            all_losses.append(ls)

            if _method in _PRETRAIN_MULTI_BLOCK:
                _diag = {}
                if hasattr(trainer, "diagnostics"):
                    _diag = trainer.diagnostics(_last_w)
                _latest_diag = dict(_diag or {})
                if "align" in _latest_diag:
                    final_align = float(_latest_diag.get("align", 0.0))
                if "unif" in _latest_diag:
                    final_unif = float(_latest_diag.get("unif", 0.0))
                if "embed_std" in _latest_diag:
                    final_embed_std = float(_latest_diag.get("embed_std", 0.0))
                _extra = ""
                if _diag:
                    if "align" in _diag:
                        _extra = (
                            f" | align={_diag.get('align', 0):.3f} "
                            f"unif={_diag.get('unif', 0):.3f}"
                        )
                    elif "masked_mse" in _diag:
                        _extra = f" | masked_mse={_diag.get('masked_mse', 0):.4f}"
                    elif "recon_loss" in _diag:
                        _extra = (
                            f" | recon={_diag.get('recon_loss', 0):.4f} "
                            f"kl={_diag.get('kl', 0):.4f}"
                        )
                    elif "forecast_mse" in _diag:
                        _extra = f" | forecast_mse={_diag.get('forecast_mse', 0):.4f}"
                    elif "drift_margin" in _diag:
                        _extra = f" | drift_dist={_diag.get('drift_margin', 0):.4f}"
                print(f"[Pretrain] Ep {_ep+1:2d}/{target_epochs} | loss={ls:.4f}"
                      f"  ({_n_blocks} blocks ├ù {n_windows:,} windows = "
                      f"{_n_blocks * n_windows:,} total){_extra}")
                if run and hasattr(run, "log"):
                    _log = {
                        "pt_loss": ls,
                        "pt_effective_windows": effective_windows,
                        "pt_windows_per_block": n_windows,
                        "pt_blocks": _n_blocks,
                        "pt_lr": float(getattr(args, "pretrain_lr", PRETRAIN.get("pretrain_lr", 1e-4))),
                    }
                    if _diag:
                        _log.update({f"pt_{k}": v for k, v in _diag.items()
                                     if k in (
                                         "align", "unif", "embed_std", "masked_mse",
                                         "recon_loss", "kl", "forecast_mse", "drift_margin",
                                     )})
                    _safe_wandb_log(run, _log)
            else:
                al = _h.get("align", [0.0])[-1]
                un = _h.get("unif",  [0.0])[-1]
                std = _h.get("embed_std", [0.0])[-1] if "embed_std" in _h else 0.0
                final_align = float(al)
                final_unif = float(un)
                final_embed_std = float(std)
                _latest_diag = {"align": final_align, "unif": final_unif, "embed_std": final_embed_std}
                print(f"[Pretrain] Ep {_ep+1:2d}/{target_epochs} | loss={ls:.3f} | align={al:.3f} | unif={un:.3f}")
                _safe_wandb_log(run, {
                    "pt_loss": ls, "pt_align": al, "pt_unif": un,
                    "pt_temp": trainer.temp.item(),
                })

            _improved = ls < (_best_loss - handoff_min_delta)
            if _improved:
                _best_loss = ls
                _stale_epochs = 0
                if hasattr(trainer, "save_encoder"):
                    chk_path = Path(ckpt)
                    ep_ckpt = chk_path.with_name(f"{chk_path.stem}_ep{_ep+1}{chk_path.suffix}")
                    trainer.save_encoder(str(ep_ckpt))
            else:
                _stale_epochs += 1

            _epochs_done = _ep + 1
            if _epochs_done >= max(1, min_epochs):
                if handoff_loss > float("-inf") and ls <= handoff_loss:
                    print(f"[Pretrain] Handoff: reached loss threshold {handoff_loss:.4f} at epoch {_epochs_done}.")
                    _stopped_early = True
                    break
                
                if handoff_patience > 0 and _stale_epochs >= handoff_patience:
                    # Resolve metrics for handoff logic
                    cur_std = _diag.get("embed_std", 0.0) if _method in _PRETRAIN_MULTI_BLOCK and _diag else (std if not _method in _PRETRAIN_MULTI_BLOCK else 0.0)
                    cur_unif = _diag.get("unif", 0.0) if _method in _PRETRAIN_MULTI_BLOCK and _diag else (un if not _method in _PRETRAIN_MULTI_BLOCK else 0.0)
                    
                    good_std = cur_std > 0.015 or cur_std == 0.0
                    good_unif = cur_unif < -0.5 or cur_unif == 0.0
                    
                    if good_std and good_unif:
                        print(f"[Pretrain] Handoff: plateau ({_stale_epochs} stale epochs). Quality met (std={cur_std:.4f}, unif={cur_unif:.2f}).")
                        _stopped_early = True
                        break
                    else:
                        print(f"[Pretrain] Handoff skipped: plateau reached but quality not met (std={cur_std:.4f}, unif={cur_unif:.2f}).")

    except RepresentationCollapseError as e:
        print(f"\n[Pretrain] ABORTED: {e}")
        print("[Pretrain] Falling back to random initialization.")
        _update_pretrain_report(args, {
            "status": "aborted",
            "quality_gate_result": "representation_collapse",
            "error": str(e),
            "epochs_completed": int(len(all_losses)),
            "loss_history": [float(x) for x in all_losses],
            "checkpoint_path": str(ckpt),
        })
        model = build_model(args.model, n_features, args).to(device)
        return model

    # Quality gate ΓÇö embedding spread for reconstruction-style methods; uniformity for contrastive
    _quality_gate = "passed"
    if _method in _PRETRAIN_STD_QUALITY:
        _final_diag = trainer.diagnostics(_last_w) if hasattr(trainer, "diagnostics") else {}
        _std = float(_final_diag.get("embed_std", 0.0))
        _latest_diag.update(_final_diag or {})
        final_embed_std = _std
        if bool(_final_diag.get("collapsed", False)) or not np.isfinite(_std):
            print(
                f"\n[Pretrain] Quality Gate Failed: {_method.upper()} embeddings collapsed "
                f"(std={_std:.6f}). Discarding pretrain weights."
            )
            _update_pretrain_report(args, {
                "status": "discarded",
                "quality_gate_result": "failed_embedding_collapse",
                "epochs_completed": int(len(all_losses)),
                "average_pretrain_loss": float(sum(all_losses) / max(len(all_losses), 1)),
                "final_embedding_std": final_embed_std,
                "diagnostics": _latest_diag,
                "checkpoint_path": str(ckpt),
            })
            model = build_model(args.model, n_features, args).to(device)
            return model
        if _std < 0.015:
            _quality_gate = "warning_low_embedding_std"
            print(
                f"\n[Pretrain] WARNING: Low {_method.upper()} embedding spread (std={_std:.6f}); "
                "continuing because it is above the collapse threshold."
            )
    elif _method in {"tscl", "cluster"}:
        _unif = float(final_unif or 0.0)
        if _unif > -0.5:
            _quality_gate = "warning_low_uniformity"
            print(f"\n[Pretrain] WARNING: Low uniformity ({_unif:.2f}). "
                  "Embeddings are clustered.")
            if _unif > -0.1:
                print("[Pretrain] Quality Gate Failed. Discarding pretrain weights.")
                _update_pretrain_report(args, {
                    "status": "discarded",
                    "quality_gate_result": "failed_low_uniformity",
                    "epochs_completed": int(len(all_losses)),
                    "average_pretrain_loss": float(sum(all_losses) / max(len(all_losses), 1)),
                    "alignment": final_align,
                    "uniformity": final_unif,
                    "final_embedding_std": final_embed_std,
                    "diagnostics": _latest_diag,
                    "checkpoint_path": str(ckpt),
                })
                model = build_model(args.model, n_features, args).to(device)
                return model

    avg_loss = sum(all_losses) / max(len(all_losses), 1)
    if _stopped_early:
        print(f"[Pretrain] Done (early handoff). avg_loss={avg_loss:.4f}")
    else:
        print(f"[Pretrain] Done. avg_loss={avg_loss:.4f}")
    _update_pretrain_report(args, {
        "status": "completed",
        "method": _method,
        "regime_aware": bool(use_regime),
        "epochs_requested": int(target_epochs),
        "epochs_completed": int(len(all_losses)),
        "stopped_early_for_handoff": bool(_stopped_early),
        "handoff": {
            "enabled": bool(handoff_enabled),
            "patience": int(handoff_patience),
            "min_delta": float(handoff_min_delta),
            "loss_threshold": None if handoff_loss == float("-inf") else float(handoff_loss),
        },
        "average_pretrain_loss": float(avg_loss),
        "final_pretrain_loss": float(all_losses[-1]) if all_losses else None,
        "loss_history": [float(x) for x in all_losses],
        "alignment": final_align,
        "uniformity": final_unif,
        "final_embedding_std": final_embed_std,
        "diagnostics": _latest_diag,
        "quality_gate_result": _quality_gate,
        "checkpoint_path": str(ckpt),
        "batch_size": int(pt_bs),
        "blocks_per_epoch": int(_n_blocks),
        "effective_windows_per_epoch": int(effective_windows),
        "hard_examples_injected": int(_hard_examples_injected),
    })
    return model


# -----------------------------------------------------------------------------
# RL TRAINING
# -----------------------------------------------------------------------------

def _build_rl_market_arrays(y_labels, base_price: float = 1.085,
                            base_spread: float = 0.00008):
    """A-C2: derive per-bar close prices, ATR and spreads for the RL environment.

    Priority
    --------
    1. ``_try_rl_market_from_features`` ΓÇö denormalized ret/atr/spread from cached
       feature windows (same bars as supervised training; preferred when scaler exists).
    2. Label integration (this function) ΓÇö treat forward-reward labels as a signed
       return walk when OHLC/features are unavailable.

    The feature cache stores scaled windows, not raw OHLC. When no scaler/feature
    columns are available, ``per_bar_ret`` is inferred from reward labels and
    integrated into a synthetic price path so PnL / SL / TP are non-zero.

    Prefer ``_load_rl_market_from_cache`` (A-C2 full). This path is last-resort only.
    """
    y = np.nan_to_num(np.asarray(y_labels, dtype=np.float64),
                      nan=0.0, posinf=0.0, neginf=0.0)
    s = float(y.std()) or 1.0
    # Scale to ~4-pip (0.0004) one-sigma 1-min move, clipped to a sane band.
    per_bar_ret = np.clip(y / (s + 1e-9) * 0.0004, -0.01, 0.01)
    prices = (base_price * np.cumprod(1.0 + per_bar_ret)).astype(np.float32)
    abs_ret = np.abs(np.diff(prices, prepend=prices[0]))
    win = 14
    kernel = np.ones(win) / win
    atr = np.convolve(abs_ret, kernel, mode="same")
    atr = np.maximum(atr, 1e-4).astype(np.float32)            # floor at 1 pip
    med_atr = float(np.median(atr)) or 1e-4
    spreads = np.clip(base_spread * (atr / med_atr),
                      base_spread, 5 * base_spread).astype(np.float32)
    return prices, atr, spreads


def _try_rl_market_from_features(
    X_last: np.ndarray,
    scaler: Optional[StandardScaler],
    feat_names: Optional[list],
    f_per_pair: int,
    base_price: float = 1.085,
    base_spread: float = 0.00008,
) -> Optional[tuple]:
    """Build RL price/ATR/spread arrays from denormalized feature columns."""
    if scaler is None or not hasattr(scaler, "mean_") or scaler.mean_ is None:
        return None
    X = np.asarray(X_last, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        return None
    sl = X[:, : min(f_per_pair, X.shape[1])]
    mean = np.asarray(scaler.mean_, dtype=np.float64)[: sl.shape[1]]
    scale = np.asarray(scaler.scale_, dtype=np.float64)[: sl.shape[1]]
    scale = np.where(scale > 1e-12, scale, 1.0)
    raw = sl * scale + mean

    names = list(feat_names or [])[: sl.shape[1]]
    ri, ai = _resolve_pair_feat_indices(names, sl.shape[1])
    spread_i = next((names.index(c) for c in ("spread_pips", "spread_avg") if c in names), None)

    ret = np.nan_to_num(raw[:, ri], nan=0.0)
    if np.std(ret) < 1e-12:
        return None
    per_bar_ret = np.clip(ret, -0.01, 0.01)
    prices = (base_price * np.cumprod(1.0 + per_bar_ret)).astype(np.float32)

    atr_raw = np.abs(np.nan_to_num(raw[:, ai], nan=0.0))
    if float(np.median(atr_raw)) <= 0:
        abs_ret = np.abs(np.diff(prices, prepend=prices[0]))
        win = 14
        atr_raw = np.convolve(abs_ret, np.ones(win) / win, mode="same")
    atr = np.maximum(atr_raw.astype(np.float32), 1e-4)
    med_atr = float(np.median(atr)) or 1e-4

    if spread_i is not None:
        spr_pips = np.maximum(np.nan_to_num(raw[:, spread_i], nan=0.0), 0.0)
        spreads = np.clip(spr_pips * 0.0001, base_spread, 5 * base_spread).astype(np.float32)
    else:
        spreads = np.clip(base_spread * (atr / med_atr),
                          base_spread, 5 * base_spread).astype(np.float32)
    return prices, atr, spreads


def _encode_rl_observations(cache_path, start: int, n_env: int, n_features: int,
                            args, device, batch: int = 4096):
    """A-C3: run the frozen supervised encoder over the RL window's full
    sequences and return its pre-head embedding per bar as the RL observation.

    This is the connective tissue between the supervised/pretrained stage and RL:
    the policy observes the supervised representation instead of raw features,
    so RL fine-tunes ON TOP of the learned encoder. The encoder is frozen.
    """
    ckpt_dir = Path(args.checkpoint_dir)
    candidates = [ckpt_dir / args.model / f"{args.model}_best.pt",
                  ckpt_dir / f"{args.model}_best.pt"]
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"no supervised checkpoint for {args.model} under {ckpt_dir}")
    model = build_model(args.model, n_features, args).to(device)
    core  = _core_model(model)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    _strict_load_report(core, state, f"RLObsEncoder:{args.model}", min_frac_loaded=0.6)
    encoder = core.backbone if hasattr(core, "backbone") else core
    saved_head = None
    if hasattr(encoder, "head"):
        saved_head = encoder.head
        encoder.head = nn.Identity()
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    if ZARR and cache_path.endswith(".zarr") and Path(cache_path).is_dir():
        _z = _zarr_open_group(cache_path, mode="r"); _Xs = _z["X"]
        def _read(a, b): return np.asarray(_Xs[start + a:start + b], dtype=np.float32)
    else:
        _Xm = np.load(_x_path(cache_path), mmap_mode="r")
        def _read(a, b): return np.asarray(_Xm[start + a:start + b], dtype=np.float32)

    embs = []
    with torch.no_grad():
        for a in range(0, n_env, batch):
            b = min(a + batch, n_env)
            xb = torch.as_tensor(_read(a, b), dtype=torch.float32, device=device)
            xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
            h = encoder(xb)
            if h.ndim == 3:
                h = h[:, -1, :]
            embs.append(h.float().cpu().numpy())
    if saved_head is not None:
        encoder.head = saved_head
    return np.concatenate(embs, axis=0).astype(np.float32)


def _load_rl_slice(cache_path: str, start: int, n_bars: int) -> tuple[np.ndarray, np.ndarray]:
    """Load y and last-timestep features for an RL window."""
    if ZARR and cache_path.endswith(".zarr") and Path(cache_path).is_dir():
        _z = _zarr_open_group(cache_path, mode="r")
        y_env = np.asarray(_z["y"][start:start + n_bars], dtype=np.float32)
        X_last = np.asarray(_z["X"][start:start + n_bars, -1, :], dtype=np.float32)
    else:
        y_env = np.asarray(
            np.load(_y_path(cache_path), mmap_mode="r")[start:start + n_bars], dtype=np.float32
        )
        X_last = np.asarray(
            np.load(_x_path(cache_path), mmap_mode="r")[start:start + n_bars, -1, :],
            dtype=np.float32,
        )
    return y_env, X_last


def _build_rl_env(
    cache_path: str,
    start: int,
    n_bars: int,
    n_features: int,
    args,
    device,
) -> ForexTradingEnv:
    """Construct ForexTradingEnv for train or validation slice."""
    y_env, X_last = _load_rl_slice(cache_path, start, n_bars)
    n_bars = len(y_env)

    prices, atr, spreads = _load_rl_market_from_cache(cache_path, start, n_bars)
    _market_source = "cache"
    if float(np.std(prices)) < 1e-12:
        _rl_scaler = _load_scaler_npz(Path(cache_path))
        _fpp = int(getattr(args, "_f_per_pair", X_last.shape[1]) or X_last.shape[1])
        _fnames = getattr(args, "_feat_names", None)
        if _fnames is None and _rl_scaler is not None and hasattr(_rl_scaler, "feature_names_in_"):
            _fnames = list(_rl_scaler.feature_names_in_)
        _market = _try_rl_market_from_features(X_last, _rl_scaler, _fnames, _fpp)
        if _market is not None:
            prices, atr, spreads = _market
            _market_source = "features"
        else:
            prices, atr, spreads = _build_rl_market_arrays(y_env)
            _market_source = "synthetic"
    if _market_source != "cache":
        print(f"[RL] WARN: market source={_market_source} ΓÇö rebuild cache for real OHLC")

    obs_feats = None
    if bool(getattr(args, "rl_encoder_obs", True)):
        try:
            obs_feats = _encode_rl_observations(cache_path, start, n_bars, n_features, args, device)
        except Exception as _ee:
            print(f"[RL] Encoder-obs unavailable ({_ee}); falling back to raw features.")
    if obs_feats is None:
        obs_feats = X_last

    _ep_len = int(getattr(args, "rl_episode_len", 0) or 0) or None
    return ForexTradingEnv(
        features=obs_feats,
        prices=prices,
        atr=atr,
        spreads=spreads,
        reward_weights=_rl_reward_weights(args),
        atr_sl_mult=RISK["atr_multiplier"],
        trail_activation_r=RISK["trail_activation_r"],
        breakeven_at_r=RISK["breakeven_at_r"],
        pyramid_pct=SIZING["pyramid_add_pct"],
        martingale_pct=SIZING["martingale_add_pct"],
        max_lots=SIZING["max_total_lots"],
        random_reset=True,
        episode_len=_ep_len,
    )


def _save_rl_checkpoint(agent, ckpt_dir: Path, algo: str, tag: str) -> Path:
    path = ckpt_dir / f"rl_{algo}_{tag}.pt"
    if hasattr(agent, "policy_net"):
        _safe_save(agent.policy_net.state_dict(), path)
    elif hasattr(agent, "net"):
        _safe_save(agent.net.state_dict(), path)
    else:
        raise RuntimeError("[RL] Agent has no saveable policy weights")
    return path


def _production_onnx_paths(args) -> tuple[Path, Path]:
    try:
        from monitoring.demotion_monitor import PROD_CHECKPOINT as _prod
        prod_onnx = Path(_prod).with_suffix(".onnx")
    except Exception:
        prod_onnx = Path(args.checkpoint_dir) / "production_best.onnx"
    prev_onnx = prod_onnx.with_name("production_prev.onnx")
    return prod_onnx, prev_onnx


def _write_feature_schema_for_onnx(schema_path: Path, args) -> None:
    _safe_save_json(_feature_schema_payload(args), schema_path)





def _feature_schema_payload(args, n_features: Optional[int] = None) -> dict:

    import hashlib

    feature_names = list(getattr(args, "_feat_names", []) or [])
    n_feat = int(n_features or getattr(args, "_n_features", 0) or len(feature_names) or 0)

    schema_hash = hashlib.md5(json.dumps(feature_names, sort_keys=True).encode()).hexdigest()
    return {

        "feature_names": feature_names,

        "hash": schema_hash,

        "n_features": n_feat,

        "seq_len": int(getattr(args, "seq_len", 60)),

        "created_at": datetime.now(timezone.utc).isoformat(),

    }





def _verify_onnx_schema_deployment(onnx_path: Path, schema_path: Path, args, *, n_features: int, seq_len: int) -> dict:

    """Verify ONNX artifact exists, schema matches training, and optional runtimes can load it."""

    result = {

        "onnx_path": str(onnx_path),

        "schema_path": str(schema_path),

        "status": "pass",

        "checks": {},

        "warnings": [],

        "errors": [],

    }

    try:

        if not onnx_path.exists() or onnx_path.stat().st_size <= 0:

            raise RuntimeError("onnx file missing or empty")

        result["checks"]["onnx_file"] = {"exists": True, "bytes": int(onnx_path.stat().st_size)}



        if not schema_path.exists():

            raise RuntimeError("schema json missing")

        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        expected = _feature_schema_payload(args, n_features=n_features)

        if int(schema.get("n_features", -1)) != int(n_features):

            raise RuntimeError(f"schema n_features mismatch: {schema.get('n_features')} != {n_features}")

        if int(schema.get("seq_len", -1)) != int(seq_len):

            raise RuntimeError(f"schema seq_len mismatch: {schema.get('seq_len')} != {seq_len}")

        if schema.get("hash") != expected.get("hash"):

            raise RuntimeError("schema feature hash mismatch")

        result["checks"]["schema"] = {

            "n_features": int(schema.get("n_features")),

            "seq_len": int(schema.get("seq_len")),

            "hash": schema.get("hash"),

        }



        try:

            import onnx

            model = onnx.load(str(onnx_path))

            onnx.checker.check_model(model)

            result["checks"]["onnx_checker"] = "pass"

        except ImportError:

            result["warnings"].append("onnx package not installed; checker skipped")

        except Exception as exc:

            raise RuntimeError(f"onnx checker failed: {exc}") from exc



        try:

            import onnxruntime as ort

            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

            input_name = sess.get_inputs()[0].name

            dummy = np.zeros((1, int(seq_len), int(n_features)), dtype=np.float32)

            outputs = sess.run(None, {input_name: dummy})

            if not outputs or not np.isfinite(np.asarray(outputs[0])).all():

                raise RuntimeError("onnxruntime smoke output missing or nonfinite")

            result["checks"]["onnxruntime_cpu"] = {

                "output_shape": list(np.asarray(outputs[0]).shape),

            }

        except ImportError:

            result["warnings"].append("onnxruntime not installed; CPU smoke test skipped")



    except Exception as exc:

        result["status"] = "fail"

        result["errors"].append(str(exc))

    return result



def _signal_cpp_server_reload(prod_onnx: Path) -> Optional[Path]:
    import tempfile

    reload_flag = prod_onnx.parent / "reload_model.flag"
    fd, tmp_flag = tempfile.mkstemp(prefix=".reload.", suffix=".tmp", dir=str(prod_onnx.parent))
    os.close(fd)
    with open(tmp_flag, "w", encoding="utf-8") as f:
        f.write(f"reload {datetime.now(timezone.utc).isoformat()}\n")
    os.replace(tmp_flag, reload_flag)
    return reload_flag


def _deploy_onnx_to_cpp_server(
    onnx_path: Path,
    args,
    model_name: str,
    artifact_dir: Path,
    source_checkpoint: Optional[Path] = None,
) -> dict:
    """Atomically promote an exported ONNX graph to the C++ server path."""
    result = {
        "model_name": model_name,
        "source_checkpoint": str(source_checkpoint) if source_checkpoint else None,
        "source_onnx": str(onnx_path),
        "production_onnx": None,
        "schema_path": None,
        "reload_flag": None,
        "status": "skipped",
        "error": None,
    }
    try:
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX artifact does not exist: {onnx_path}")
        artifact_dir.mkdir(parents=True, exist_ok=True)

        source_schema = artifact_dir / f"{model_name}_onnx_schema.json"

        _safe_save_json(_feature_schema_payload(args), source_schema)

        verify = _verify_onnx_schema_deployment(

            onnx_path,

            source_schema,

            args,

            n_features=int(getattr(args, "_n_features", 0) or len(getattr(args, "_feat_names", []) or [])),

            seq_len=int(getattr(args, "seq_len", 60)),

        )

        result["verification"] = verify

        if verify.get("status") != "pass":

            raise RuntimeError(f"ONNX/schema verification failed: {verify.get('errors')}")

        prod_onnx, prev_onnx = _production_onnx_paths(args)
        prod_onnx.parent.mkdir(parents=True, exist_ok=True)
        if prod_onnx.exists():
            _atomic_copy(prod_onnx, prev_onnx)
        _atomic_copy(onnx_path, prod_onnx)
        schema_path = prod_onnx.with_suffix(".schema.json")
        _atomic_copy(source_schema, schema_path)

        reload_flag = _signal_cpp_server_reload(prod_onnx)
        result.update({
            "production_onnx": str(prod_onnx),
            "schema_path": str(schema_path),
            "reload_flag": str(reload_flag),
            "status": "success",
        })
        print(f"[Deploy] {model_name} ONNX promoted for C++ -> {prod_onnx}")
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        print(f"[Deploy] {model_name} ONNX promotion failed: {exc}")
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _safe_save_json(result, artifact_dir / "cpp_deployment.json")
    except Exception as exc:
        print(f"[Deploy] Could not write {model_name} cpp_deployment.json: {exc}")
    return result


def run_rl(cache_path, n_features, args, device, n_samples=None, run=None):
    print(f"\n[RL] {args.rl_algo.upper()} | {args.rl_episodes} episodes | model={args.model}")
    _require_rl_market_cache(cache_path)

    total = int(n_samples or (_on_disk_sequence_count(cache_path) or 0))
    train_start, train_n, val_start, val_n = _rl_train_val_slices(total, args)
    if train_n < 256:
        raise RuntimeError(
            f"[RL] Insufficient trainable bars ({train_n}). Rebuild cache or reduce holdout."
        )

    print(
        f"[RL] Holdout-safe window | train [{train_start}:{train_start + train_n}) "
        f"| val [{val_start}:{val_start + val_n}) | trainable_end={_trainable_max_index(total, args):,}"
    )

    train_env = _build_rl_env(cache_path, train_start, train_n, n_features, args, device)
    print(
        f"[RL] Train env | obs={train_env.obs_size} | market std={float(np.std(train_env.prices)):.6f}"
    )

    dev = str(device)
    _algo = str(args.rl_algo).lower()
    _algo_kw = _rl_algo_kwargs(args, _algo)
    if _algo == "dqn":
        agent = DQNAgent(obs_size=train_env.obs_size, n_actions=train_env.n_actions, device=dev, **_algo_kw)
    else:
        agent = PPOAgent(obs_size=train_env.obs_size, n_actions=train_env.n_actions, device=dev, **_algo_kw)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _val_episodes = max(3, min(20, int(args.rl_episodes) // 25))
    _min_val_sharpe = float(getattr(args, "rl_min_val_sharpe", -999.0))
    _best_val_sharpe = float("-inf")
    _best_saved = False

    # ── RL curriculum scheduler (graduated volatility exposure) ──────────
    _rl_curriculum = None
    if bool(getattr(args, "rl_curriculum", True)):
        try:
            from models.rl_advanced import CurriculumScheduler as _CS
            _rl_curriculum = _CS(total_episodes=args.rl_episodes)
            print(f"[RL] Curriculum: graduated volatility over {args.rl_episodes} episodes")
        except Exception as _rce:
            print(f"[RL] Curriculum unavailable: {_rce}")

    returns = train_agent(
        agent, train_env, n_episodes=args.rl_episodes, agent_type=_algo,
        curriculum=_rl_curriculum,
    )

    if val_n > 0:
        val_env = _build_rl_env(cache_path, train_start + val_start, val_n, n_features, args, device)
        _, val_summary = evaluate_agent(
            agent, val_env, n_episodes=_val_episodes, agent_type=_algo, greedy=True
        )
        _val_sharpe = float(val_summary.get("sharpe", 0.0))
        print(
            f"[RL] Val | Sharpe={_val_sharpe:.3f} | Return={val_summary['total_return_pct']:+.2f}% "
            f"| Trades={val_summary['n_trades']}"
        )
        if _val_sharpe >= _best_val_sharpe:
            _best_val_sharpe = _val_sharpe
        if _val_sharpe >= _min_val_sharpe:
            _save_rl_checkpoint(agent, ckpt_dir, _algo, "best")
            meta = {
                "model": args.model,
                "algo": _algo,
                "val_sharpe": _val_sharpe,
                "val_return_pct": float(val_summary["total_return_pct"]),
                "obs_size": int(train_env.obs_size),
                "n_actions": int(train_env.n_actions),
                "encoder_obs": bool(getattr(args, "rl_encoder_obs", True)),
            }
            with (ckpt_dir / f"rl_{_algo}_best.json").open("w", encoding="utf-8") as _rl_meta_fp:
                json.dump(meta, _rl_meta_fp, indent=2)
            _best_saved = True
            try:
                from inference.onnx_inference import export_rl_execution_to_onnx, export_rl_to_onnx

                rl_best = ckpt_dir / f"rl_{_algo}_best.pt"
                sup_ckpt = ckpt_dir / args.model / f"{args.model}_best.pt"
                if not sup_ckpt.is_file():
                    sup_ckpt = ckpt_dir / f"{args.model}_best.pt"
                if not sup_ckpt.is_file():
                    sup_ckpt = ckpt_dir.parent / args.model / f"{args.model}_best.pt"
                rl_onnx = ckpt_dir / f"rl_{_algo}_best.onnx"
                export_rl_to_onnx(
                    rl_checkpoint=str(rl_best),
                    supervised_checkpoint=str(sup_ckpt),
                    model_name=str(args.model),
                    seq_len=int(getattr(args, "seq_len", 60)),
                    n_features=int(n_features),
                    output_path=str(rl_onnx),
                    algo=_algo,
                    device="cpu",
                )
                print(f"[RL] Exported ONNX -> {rl_onnx}")
                rl_exec_onnx = ckpt_dir / f"rl_{_algo}_execution.onnx"
                export_rl_execution_to_onnx(
                    rl_checkpoint=str(rl_best),
                    supervised_checkpoint=str(sup_ckpt),
                    model_name=str(args.model),
                    seq_len=int(getattr(args, "seq_len", 60)),
                    n_features=int(n_features),
                    output_path=str(rl_exec_onnx),
                    algo=_algo,
                    device="cpu",
                )
                _safe_save_json(
                    {
                        "model_name": f"rl_{_algo}_execution",
                        "source_checkpoint": str(rl_best),
                        "source_onnx": str(rl_exec_onnx),
                        "direction_model": "Use MODEL_PATH for the ensemble/supervised 3-logit direction model.",
                        "runtime_env": "Set EXECUTION_MODEL_PATH to this ONNX in the C++ server.",
                        "inputs": {
                            "features": [1, int(getattr(args, "seq_len", 60)), int(n_features)],
                            "agent_state": [1, 5],
                        },
                        "outputs": {"action_logits": [1, int(train_env.n_actions)]},
                        "actions": {
                            "0": "HOLD",
                            "1": "OPEN_LONG",
                            "2": "OPEN_SHORT",
                            "3": "SCALE_IN_25",
                            "4": "SCALE_IN_50",
                            "5": "SCALE_IN_100",
                            "6": "SCALE_OUT_25",
                            "7": "SCALE_OUT_50",
                            "8": "SCALE_OUT_100",
                            "9": "CLOSE_ALL",
                        },
                    },
                    ckpt_dir / f"rl_{_algo}_execution.json",
                )
                print(f"[RL] Exported execution ONNX -> {rl_exec_onnx}")
                if bool(getattr(args, "deploy_rl", False)):
                    args._n_features = int(n_features)
                    _deploy_onnx_to_cpp_server(
                        rl_onnx,
                        args,
                        model_name=f"rl_{_algo}",
                        artifact_dir=ckpt_dir,
                        source_checkpoint=rl_best,
                    )
            except Exception as exc:
                print(f"[RL] ONNX export/deploy skipped: {exc}")
            print(f"[RL] Saved best policy ΓåÆ {ckpt_dir / f'rl_{_algo}_best.pt'}")
        else:
            print(
                f"[RL] Val Sharpe {_val_sharpe:.3f} below min_val_sharpe {_min_val_sharpe:.3f} "
                "ΓÇö rl_*_best not updated"
            )

    _save_rl_checkpoint(agent, ckpt_dir, _algo, "last")

    s = train_env.summary()
    ret_arr = np.asarray(returns, dtype=np.float64)
    rl_stats = {
        "rl/total_return_pct": float(s["total_return_pct"]),
        "rl/sharpe": float(s["sharpe"]),
        "rl/n_trades": int(s["n_trades"]),
        "rl/episodes": int(args.rl_episodes),
        "rl/return_mean": float(ret_arr.mean()) if ret_arr.size else 0.0,
        "rl/return_std": float(ret_arr.std()) if ret_arr.size else 0.0,
        "rl/return_min": float(ret_arr.min()) if ret_arr.size else 0.0,
        "rl/return_max": float(ret_arr.max()) if ret_arr.size else 0.0,
        "rl/val_sharpe": float(_best_val_sharpe) if val_n > 0 else 0.0,
        "rl/best_saved": int(_best_saved),
    }
    print(f"[RL] Done | Train return: {s['total_return_pct']:+.2f}% | "
          f"Sharpe: {s['sharpe']:.3f} | Trades: {s['n_trades']} | "
          f"Ep mean: {rl_stats['rl/return_mean']:+.2f}%")
    if _TRAIN_LOGGER is not None:
        _TRAIN_LOGGER.info(
            f"[RL] {args.rl_algo.upper()} complete ΓÇö "
            f"return={s['total_return_pct']:+.2f}% sharpe={s['sharpe']:.3f} "
            f"val_sharpe={rl_stats['rl/val_sharpe']:.3f}"
        )
        if hasattr(_TRAIN_LOGGER, "on_rl_complete"):
            _TRAIN_LOGGER.on_rl_complete(rl_stats)
    if WANDB and run is not None:
        _safe_wandb_log(run, rl_stats)
        _safe_wandb_summary_update(
            run,
            {k.replace("rl/", "best_rl_"): v for k, v in rl_stats.items()
             if k.startswith("rl/") and k != "rl/episodes"},
        )
    return returns


# -----------------------------------------------------------------------------
# ENSEMBLE META-LEARNER TRAINING
# -----------------------------------------------------------------------------

def run_ensemble_meta(
    cache_path:  str,
    n_features:  int,
    args,
    device:      "torch.device",
) -> None:
    """
    Load all trained base model checkpoints, build an EnsembleMetaLearner,
    then train the meta-network with a diversity penalty.

    The diversity penalty has two components:
      1. Weight entropy maximisation ΓÇö prevents the meta from collapsing to
         a single model (all weight on the best base model).
      2. Base-output correlation penalty ΓÇö rewards the meta for up-weighting
         models whose predictions disagree with each other.

    Only runs when --train-ensemble is passed and at least 2 base checkpoints
    are found in the checkpoint directory.
    """
    if not ENSEMBLE:
        print("[EnsembleMeta] models.ensemble not available ΓÇö skipping.")
        return

    ckpt_dir = Path(args.checkpoint_dir)
    loaded_bases: list = []
    loaded_names: list = []
    loaded_ckpts: list[Path] = []
    loaded_seq_lens: list[int] = []

    for model_name in MODEL_REGISTRY:
        # Per-model subfolder layout first, then legacy flat layout
        ckpt = ckpt_dir / model_name / f"{model_name}_best.pt"
        if not ckpt.exists():
            ckpt = ckpt_dir / f"{model_name}_best.pt"
        if not ckpt.exists():
            continue
        try:
            model_args = _model_build_args(args, model_name)
            base = build_model(model_name, n_features, model_args).to(device)
            
            # Load checkpoint with flexibility for wrapper nesting
            ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
            state = ckpt_data.get("model_state_dict", ckpt_data.get("state_dict", ckpt_data))
            
            # Attempt to load into base directly (if checkpoint was a MultiTaskWrapper)
            # or into core backbone (if checkpoint was only the base architecture).
            try:
                base.load_state_dict(state, strict=True)
            except Exception:
                # If strict loading failed, load into backbone with an asserting
                # report so a near-empty load fails loudly (A-H2).
                core = base.backbone if hasattr(base, "backbone") else base
                _strict_load_report(core, state, f"EnsembleMeta:{model_name}", min_frac_loaded=0.6)

            base.eval()

            loaded_bases.append(base)
            loaded_names.append(model_name)
            loaded_ckpts.append(ckpt)
            loaded_seq_lens.append(int(getattr(model_args, "seq_len", getattr(args, "seq_len", 0)) or 0))
            print(f"  [EnsembleMeta] Loaded {model_name} from {ckpt.name}")
        except Exception as e:
            print(f"  [EnsembleMeta] Could not load {model_name}: {e}")

    if len(loaded_bases) < 2:
        print("[EnsembleMeta] Need >= 2 trained base models ΓÇö skipping "
              f"(found {len(loaded_bases)}: {loaded_names}). "
              "Train with --all-models first.")
        return

    print(f"\n[EnsembleMeta] Training meta-learner on {len(loaded_bases)} bases: "
          f"{loaded_names}")

    meta = EnsembleMetaLearner(
        loaded_bases,
        context_dim=32,
        hidden=64,
        base_names=loaded_names,
        base_seq_lens=loaded_seq_lens,
    ).to(device)

    # Use a random 10 % subset of the dataset for meta training
    _total = _on_disk_sequence_count(cache_path) or 10_000
    n_meta   = min(200_000, int(0.1 * _total))
    meta_idx = np.random.choice(_total, n_meta, replace=False)
    meta_ds = ZarrStreamDataset(cache_path, meta_idx, shuffle_chunks=True)
    meta_dl = DataLoader(
        meta_ds, batch_size=min(args.batch_size, 512),
        shuffle=False, num_workers=min(4, args.num_workers),
        pin_memory=(os.name != "nt") if getattr(args, "pin_memory", None) is None else bool(args.pin_memory),
    )

    ensemble_dir = ckpt_dir / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    out = ensemble_dir / "ensemble_meta_best.pt"
    history = train_meta_learner(
        meta,
        meta_dl,
        epochs=getattr(args, "ensemble_epochs", 10),
        lr=1e-3,
        diversity_weight=getattr(args, "ensemble_div_weight", 0.1),
        device=str(device),
        verbose=True,
        checkpoint_path=str(out),
        checkpoint_meta={
            "base_names": loaded_names,
            "base_seq_lens": loaded_seq_lens,
            "n_features": int(n_features),
            "seq_len": int(getattr(args, "seq_len", 60)),
            "cache_path": str(cache_path),
        },
    )

    final = ensemble_dir / "ensemble_meta_final.pt"
    _safe_save(meta.state_dict(), final)
    _meta_payload = {
        "epoch": len(history),
        "loss": float(history[-1]) if history else None,
        "history": history,
        "meta": {
            "base_names": loaded_names,
            "base_seq_lens": loaded_seq_lens,
            "n_features": int(n_features),
            "seq_len": int(getattr(args, "seq_len", 60)),
            "cache_path": str(cache_path),
        },
    }
    import json as _json
    final.with_suffix(final.suffix + ".json").write_text(
        _json.dumps(_meta_payload, indent=2), encoding="utf-8"
    )
    ensemble_manifest = {
        "kind": "ensemble_meta_learner",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_models": [
            {"name": name, "checkpoint": str(path), "seq_len": int(seq_len)}
            for name, path, seq_len in zip(loaded_names, loaded_ckpts, loaded_seq_lens)
        ],
        "artifacts": {
            "best_checkpoint": str(out),
            "final_checkpoint": str(final),
            "best_onnx": str(ensemble_dir / "ensemble_meta_best.onnx"),
            "promotion_gate": str(ensemble_dir / "promotion_gate.json"),
        },
        "schema": {
            "n_features": int(n_features),
            "seq_len": int(getattr(args, "seq_len", 60)),
            "feature_names": list(getattr(args, "_feat_names", []) or []),
        },
        "training": {
            "cache_path": str(cache_path),
            "samples": int(n_meta),
            "epochs": int(getattr(args, "ensemble_epochs", 10)),
            "diversity_weight": float(getattr(args, "ensemble_div_weight", 0.1)),
            "history": history,
            "final_loss": float(history[-1]) if history else None,
        },
    }
    _safe_save_json(ensemble_manifest, ensemble_dir / "ensemble_manifest.json")
    try:
        from inference.onnx_inference import _wrap_ensemble_logits, core_onnx_export

        ensemble_onnx = ensemble_dir / "ensemble_meta_best.onnx"
        wrapped = _wrap_ensemble_logits(meta).to(device)
        wrapped.eval()
        core_onnx_export(
            model=wrapped,
            n_features=int(n_features),
            seq_len=int(getattr(args, "seq_len", 60)),
            output_path=str(ensemble_onnx),
            output_name="logits",
        )
        print(f"[EnsembleMeta] Exported ONNX -> {ensemble_onnx}")
        if bool(getattr(args, "deploy_ensemble", False)):
            args._n_features = int(n_features)
            _deploy_onnx_to_cpp_server(
                ensemble_onnx,
                args,
                model_name="ensemble",
                artifact_dir=ensemble_dir,
                source_checkpoint=out if out.exists() else final,
            )
    except Exception as exc:
        print(f"[EnsembleMeta] ONNX export/deploy skipped: {exc}")
    best_msg = f"best={out}" if out.exists() else "best=not written"
    print(f"[EnsembleMeta] Saved -> {best_msg} | final={final} | Final loss: {history[-1]:.6f}")


# -----------------------------------------------------------------------------
# PYTORCH PROFILER  (--profile flag)
# -----------------------------------------------------------------------------

def run_profiler(model, loader, device, amp_dtype, use_amp, log_dir: str, run_name: str,
                 seq_len: Optional[int] = None) -> None:
    """
    Run torch.profiler for a short burst and write a Chrome / Perfetto trace.

    Usage:
        python training/train_gpu.py --profile --model haelt --quick-mode

    Opens trace in:
        ΓÇó Chrome: chrome://tracing -> Load -> select logs/profile_*.json
        ΓÇó Perfetto: https://ui.perfetto.dev  (richer, recommended)

    What the trace shows:
        ΓÇó CPU-GPU overlap (are kernels launching without gaps?)
        ΓÇó DataLoader stall time (input-bound vs compute-bound)
        ΓÇó torch.compile kernel fusion vs interpreted ops
        ΓÇó Memory copies and fragmentation
    """
    from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    trace_path = str(Path(log_dir) / f"profile_{run_name}")

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    print(f"\n[Profiler] Warmup 3 batches, active 5 batches ΓÇö trace -> {trace_path}/")
    _log_info(f"[Profiler] trace path: {trace_path}")

    model.train()
    data_iter = iter(loader)

    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=0, warmup=3, active=5, repeat=1),
        on_trace_ready=tensorboard_trace_handler(trace_path),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,        # stack traces slow things down; enable for deep dives
    ) as prof:
        for step in range(8):
            try:
                xb, yb = next(data_iter)
            except StopIteration:
                break
            xb = xb.to(device, non_blocking=True)
            xb = _crop_to_seq_len(xb, seq_len)
            yb = yb.to(device, non_blocking=True)
            with record_function("forward"):
                if use_amp and device.type == "cuda":
                    with autocast("cuda", dtype=amp_dtype):
                        model(xb)
                else:
                    model(xb)
            prof.step()

    # Print a short table to stdout so you don't have to open the trace to see the hottest ops
    print(prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=15))
    print(f"[Profiler] Full trace written to {trace_path}/  (open in Perfetto or chrome://tracing)")
    _log_info("[Profiler] Complete")


# -----------------------------------------------------------------------------
# BEST-FOLD PROMOTION
# -----------------------------------------------------------------------------

def _safe_save(obj, path, metadata=None) -> None:
    """Safe wrapper for torch.save that immediately verifies integrity via atomic tempfile."""
    import json, os, tempfile, torch
    from pathlib import Path
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(obj, tmp)
        if os.path.getsize(tmp) <= 0:
            raise ValueError(f"[SafeSave] Temporary checkpoint has 0 bytes: {tmp}")
        _ = torch.load(tmp, map_location="cpu", weights_only=False)
        os.replace(tmp, path)
        if metadata is not None:
            meta = dict(metadata)
            meta.update({
                "artifact_path": str(path),
                "artifact_bytes": int(path.stat().st_size),
                "verified_loadable": True,
                "verified_at": datetime.now(timezone.utc).isoformat(),
            })
            meta_path = path.with_suffix(path.suffix + ".metadata.json")
            fd_meta, tmp_meta = tempfile.mkstemp(prefix=f".{meta_path.name}.", suffix=".tmp", dir=str(meta_path.parent))
            os.close(fd_meta)
            try:
                with open(tmp_meta, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, default=str)
                with open(tmp_meta, "r", encoding="utf-8") as f:
                    loaded_meta = json.load(f)
                for key, expected in metadata.items():
                    if str(loaded_meta.get(key)) != str(expected):
                        raise ValueError(f"[SafeSave] metadata mismatch for {key}: {loaded_meta.get(key)!r} != {expected!r}")
                os.replace(tmp_meta, meta_path)
            finally:
                if os.path.exists(tmp_meta):
                    try:
                        os.remove(tmp_meta)
                    except OSError:
                        pass
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _atomic_copy(src, dst) -> None:
    """B-M2: copy `src` onto `dst` atomically (temp file in dst's dir + os.replace)
    so a concurrent reader (live inference) never sees a half-written checkpoint."""
    import tempfile
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=str(dst.parent))
    os.close(fd)
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _safe_save_json(data, path) -> None:
    """Safely write JSON to `path` using atomic tempfile replacement."""
    import json, os, tempfile
    from pathlib import Path
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _generate_model_card(model_name: str, args, history_or_cv, ckpt_dir: str, n_features: int) -> None:
    """Generates a standard Model Card JSON documenting the architecture, features, and performance."""
    card = {
        "model_name": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "architecture": getattr(args, "model", "unknown"),
        "data_window": f"{getattr(args, 'start_date', 'unknown')} to {getattr(args, 'end_date', 'unknown')}",
        "pairs": getattr(args, "pairs", []),
        "features_count": n_features,
        "label_method": getattr(args, "label_method", "unknown"),
        "promotion_status": "candidate",
        "known_weaknesses": ["Needs stress evaluation"],
    }

    # Extract best validation stats
    if isinstance(history_or_cv, list):  # CV run
        best_fold = max(history_or_cv, key=lambda x: x.get('best_metric', -999) if x.get('best_metric') is not None else -999)
        card["validation_results"] = {
            "best_val_sharpe_proxy": best_fold.get('best_metric'),
            "fold": best_fold.get('fold')
        }
        card["forward_holdout_results"] = "See fold validation metrics"
    elif isinstance(history_or_cv, dict):
        best_val = max(history_or_cv.get('val_sharpe', [0.0])) if history_or_cv.get('val_sharpe') else None
        best_loss = min(history_or_cv.get('val_loss', [999.0])) if history_or_cv.get('val_loss') else None
        card["validation_results"] = {
            "best_val_sharpe_proxy": best_val,
            "best_val_loss": best_loss
        }
        card["forward_holdout_results"] = "Pending"

    out_path = Path(ckpt_dir) / f"{model_name}_model_card.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2)
        print(f"\n[ModelCard] Generated -> {out_path}")
    except Exception as e:
        print(f"\n[ModelCard] Warning: Failed to generate model card: {e}")

def _stability_adjusted_score(score, sharpe_curve, gen_gap):
    """Apply stability penalty to a sharpe-based fold score.

    Penalises folds where the best-seen Sharpe is far above the final value
    (peak-drop), where the validation Sharpe was volatile across epochs, or
    where the generalization gap (val_loss - train_loss) is large.

    Returns (adjusted_score, volatility, gap_penalty).
    """
    import numpy as np
    volatility = 0.0
    gap_penalty = 0.0
    if len(sharpe_curve) >= 4 and score is not None:
        volatility = float(np.std(sharpe_curve, ddof=1))
        final_val = sharpe_curve[-1]
        gap_penalty = max(0.0, score - final_val)
        train_val_gap_penalty = (gen_gap * 0.1) if gen_gap is not None and gen_gap > 0 else 0.0
        score = score - (gap_penalty * 0.5) - (volatility * 0.5) - train_val_gap_penalty
    return score, volatility, gap_penalty


def _promote_best_fold(
    model_name: str,
    checkpoint_dir: str,
    cv_hist: list,
    early_stop_metric: str = "sharpe",
    alerter = None,
    force_promotion: bool = False
) -> None:
    """
    After walk-forward CV, scan all fold configs in checkpoint_dir, pick the
    fold with the best metric, and copy its checkpoint to
    <checkpoint_dir>/<model_name>_best.pt.
    """
    ckpt_dir = Path(checkpoint_dir)
    use_sharpe = early_stop_metric == "sharpe"
    best_fold = None
    best_score = None
    best_tie_breaker = None
    best_tie_breaker2 = None
    best_metrics = {}

    candidate_folds = []

    # Read metrics from the saved config JSON files
    for entry in cv_hist:
        fi = entry["fold"]
        cfg_candidates = [
            ckpt_dir / f"{model_name}_fold{fi}_config.json",
            ckpt_dir / model_name / f"{model_name}_fold{fi}_config.json"
        ]
        cfg_path = next((p for p in cfg_candidates if p.exists()), None)
        
        score = None
        tie_breaker = None
        tie_breaker2 = None
        
        history = entry.get("history", {})
        gen_gap = None
        if history and "train_loss" in history and "val_loss" in history:
            try:
                # Calculate generalization gap at the last epoch
                gen_gap = history["val_loss"][-1] - history["train_loss"][-1]
            except Exception:
                pass

        sharpe_val = None
        loss_val = None
        volatility = 0.0
        gap_penalty = 0.0

        if cfg_path:
            try:
                with open(cfg_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                sharpe_val = cfg.get("best_val_sharpe_proxy")
                loss_val = cfg.get("best_val_loss")
                
                if use_sharpe:
                    score = sharpe_val
                    sharpe_curve = history.get("val_sharpe", [])
                    score, volatility, gap_penalty = _stability_adjusted_score(
                        score, sharpe_curve, gen_gap)
                    
                    tie_breaker = -loss_val if loss_val is not None else None
                else:
                    score = -loss_val if loss_val is not None else None
                    tie_breaker = sharpe_val
            except Exception:
                pass

        if score is None:
            raw = entry.get("best_metric")
            if raw is not None:
                if use_sharpe:
                    score = raw
                    sharpe_curve = history.get("val_sharpe", [])
                    score, volatility, gap_penalty = _stability_adjusted_score(
                        score, sharpe_curve, gen_gap)
                else:
                    score = -raw
                tie_breaker = 0.0

        if score is None:
            continue

        tie_breaker2 = -gen_gap if gen_gap is not None else 0.0

        candidate_folds.append({
            "fold": fi,
            "score": score,
            "tie_breaker": tie_breaker,
            "tie_breaker2": tie_breaker2,
            "gen_gap": gen_gap,
            "volatility": volatility if use_sharpe else 0.0,
            "gap_penalty": gap_penalty if use_sharpe else 0.0
        })

        is_better = False
        if best_score is None:
            is_better = True
        elif score > best_score:
            is_better = True
        elif score == best_score:
            if tie_breaker is not None and best_tie_breaker is not None:
                if tie_breaker > best_tie_breaker:
                    is_better = True
                elif tie_breaker == best_tie_breaker:
                    if tie_breaker2 is not None and best_tie_breaker2 is not None:
                        if tie_breaker2 > best_tie_breaker2:
                            is_better = True

        if is_better:
            best_score = score
            best_tie_breaker = tie_breaker
            best_tie_breaker2 = tie_breaker2
            best_fold = fi
            best_metrics = {"sharpe": sharpe_val or 0.0, "val_loss": loss_val or 0.0, "gen_gap": gen_gap}

    if best_fold is None:
        print(f"[BestFold] {model_name}: could not determine best fold ΓÇö skipping promotion.")
        return

    src_flat = ckpt_dir / f"{model_name}_fold{best_fold}_best.pt"
    src_nested = ckpt_dir / model_name / f"{model_name}_fold{best_fold}_best.pt"
    
    src = src_nested if src_nested.exists() else src_flat

    if not src.exists():
        print(f"[BestFold] {model_name}: fold {best_fold} checkpoint not found at {src_flat} or {src_nested}")
        return
        
    metric_label = "sharpe" if use_sharpe else "val_loss"
    metric_val   = best_score if use_sharpe else -best_score
    
    # Challenger vs Production Gate
    deployment_json = ckpt_dir / "deployment.json"
    if deployment_json.exists() and not force_promotion:
        try:
            with open(deployment_json, "r") as f:
                prod_data = json.load(f)
            # If there's an existing metric value
            if "metric_value" in prod_data and prod_data.get("metric", "") == metric_label:
                prod_metric = prod_data["metric_value"]
                min_delta = 0.001 if use_sharpe else -0.001
                if use_sharpe and metric_val < prod_metric + min_delta:
                    print(f"[ChallengerGate] Rejected: new score {metric_val:.4f} is not significantly better than deployed score {prod_metric:.4f}")
                    return
                elif not use_sharpe and metric_val > prod_metric + min_delta:
                    print(f"[ChallengerGate] Rejected: new score {metric_val:.4f} is not significantly better than deployed score {prod_metric:.4f}")
                    return
                print(f"[ChallengerGate] Accepted: new score {metric_val:.4f} vs deployed score {prod_metric:.4f}")
        except Exception as e:
            print(f"[ChallengerGate] Warning: failed to parse existing deployment.json: {e}")
    
    dst_flat = ckpt_dir / f"{model_name}_best.pt"
    dst_nested = ckpt_dir / model_name / f"{model_name}_best.pt"
    
    dst_nested.parent.mkdir(parents=True, exist_ok=True)
    _atomic_copy(src, dst_flat)
    if src != dst_nested:
        _atomic_copy(src, dst_nested)
    print(f"[BestFold] {model_name}: fold {best_fold} is best ({metric_label}={metric_val:.4f}) -> promoted to {dst_flat.name} & {dst_nested.name}")

    if alerter:
        try:
            alerter.send_fold_selected(model_name, best_fold, best_metrics)
        except Exception as e:
            print(f"[Discord] Failed to send fold_selected: {e}")

    summary = {
        "model": model_name,
        "selected_fold": best_fold,
        "metric": metric_label,
        "metric_value": round(metric_val, 6),
        "secondary_metric": "val_loss" if use_sharpe else "val_sharpe",
        "secondary_value": round(best_metrics.get("val_loss", 0.0), 6),
        "gen_gap": round(best_metrics.get("gen_gap") or 0.0, 6),
        "n_candidates": len(candidate_folds),
        "source_checkpoint": str(src.name),
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "candidates": [
            {
                "fold": c["fold"],
                "score": round(c["score"], 6),
                "tie_breaker": round(c["tie_breaker"], 6) if c.get("tie_breaker") is not None else None,
                "gen_gap": round(c["gen_gap"], 6) if c.get("gen_gap") is not None else None,
            }
            for c in candidate_folds
        ],
    }
    try:
        _safe_save_json(summary, ckpt_dir / "fold_selection.json")
        print(f"[BestFold] fold_selection.json written -> {ckpt_dir / 'fold_selection.json'}")
    except Exception:
        try:
            with open(ckpt_dir / "fold_selection.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass

def _evaluate_forward_gate(model_name, cache_path, n_samples, n_features, args, device,
                           fold_sharpes=None) -> dict:
    """B-C1: backtest the freshly-trained challenger on a HELD-OUT FORWARD window
    and run the real PromotionGate (replacing the hardcoded `promoted: True`).

    The feature cache stores reward labels (a forward-P&L proxy) but NOT raw
    OHLC, so this is a label-based forward backtest: the model emits a directional
    signal per bar over the most-recent `promote_forward_frac` of samples (held
    out, chronologically after the train/val split), and per-trade P&L =
    signal ┬╖ reward_label. Sharpe, profit factor, max-DD, n_trades and PSR are
    computed from that trade series and fed to PromotionGate.evaluate_from_history.

    Returns the full PromotionGate result dict.

    NOTE: to use the full execution-aware backtester (real SL/TP, slippage,
    commissions), swap the signalΓåÆPnL block below for scripts.backtest_model run
    on this same forward window ΓÇö the gate-call plumbing stays identical.
    """
    try:
        from validation.promotion_gate import PromotionGate, GateConfig
    except Exception as e:
        return {"promoted": False, "details": {}, "reasons": [f"gate import failed: {e}"],
                "summary": "REJECT (gate unavailable)"}

    fwd_n = _promotion_holdout_n(n_samples, args)
    start = max(0, n_samples - fwd_n)
    n_fwd = n_samples - start
    if n_fwd < 50:
        return {"promoted": False, "details": {"n_trades": float(n_fwd)},
                "reasons": ["forward window too small"],
                "summary": "REJECT (insufficient forward data)"}

    ckpt_dir   = Path(args.checkpoint_dir)
    getattr(args, "loss", "") in ("cross_entropy", "multi_task", "asymmetric_directional")

    if model_name == "ensemble":
        ckpt_path = ckpt_dir / "ensemble" / "ensemble_meta_best.pt"
        if not ckpt_path.exists():
            return {"promoted": False, "details": {}, "reasons": ["no ensemble checkpoint to gate"], "summary": "REJECT (no checkpoint)"}
        
        meta_json_path = ckpt_dir / "ensemble" / "ensemble_meta_final.json"
        base_names = []
        if meta_json_path.exists():
            import json as _json
            meta_data = _json.loads(meta_json_path.read_text())
            base_names = meta_data.get("meta", {}).get("base_names", base_names)
            
        from models.ensemble import EnsembleMetaLearner
        loaded_bases = []
        for b_name in base_names:
            b_ckpt = next((p for p in [ckpt_dir / b_name / f"{b_name}_best.pt", ckpt_dir / f"{b_name}_best.pt"] if p.exists()), None)
            if b_ckpt:
                b_model = build_model(b_name, n_features, _model_build_args(args, b_name)).to(device)
                _dummy = torch.zeros(2, getattr(args, "seq_len", 60), n_features, device=device)
                _ = b_model(_dummy)
                b_state = torch.load(b_ckpt, map_location=device, weights_only=False)
                if isinstance(b_state, dict) and "model_state_dict" in b_state:
                    b_state = b_state["model_state_dict"]
                elif isinstance(b_state, dict) and "state_dict" in b_state:
                    b_state = b_state["state_dict"]
                b_model.load_state_dict(b_state, strict=False)
                loaded_bases.append(b_model)
                
        if not loaded_bases:
            return {"promoted": False, "details": {}, "reasons": ["no ensemble base models loaded"], "summary": "REJECT (no bases)"}

        model = EnsembleMetaLearner(
            loaded_bases,
            context_dim=32,
            hidden=64,
            base_names=base_names,
        ).to(device)
        
        # Initialize LazyLinear modules before loading state dict
        _dummy = torch.zeros(2, getattr(args, "seq_len", 60), n_features, device=device)
        _ = model(_dummy)
        
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False), strict=False)
        model.eval()
        core = model
        state = {} # dummy state to pass the strict load report
    else:
        candidates = [ckpt_dir / model_name / f"{model_name}_best.pt",
                      ckpt_dir / f"{model_name}_best.pt"]
        ckpt_path  = next((p for p in candidates if p.exists()), None)
        if ckpt_path is None:
            return {"promoted": False, "details": {}, "reasons": ["no checkpoint to gate"],
                    "summary": "REJECT (no checkpoint)"}

        model = build_model(model_name, n_features, _model_build_args(args, model_name)).to(device)
        core  = _core_model(model)
        
        # Initialize LazyLinear modules before loading state dict
        _dummy = torch.zeros(2, getattr(args, "seq_len", 60), n_features, device=device)
        _ = model(_dummy)
        
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
    try:
        if model_name != "ensemble":
            _strict_load_report(core, state, f"Gate:{model_name}", min_frac_loaded=0.6)
    except Exception as e:
        return {"promoted": False, "details": {}, "reasons": [f"checkpoint load failed: {e}"],
                "summary": "REJECT (load failed)"}
    model.eval()

    if ZARR and cache_path.endswith(".zarr") and Path(cache_path).is_dir():
        _z = _zarr_open_group(cache_path, mode="r"); _Xs = _z["X"]
        def _rd(a, b): return np.asarray(_Xs[start + a:start + b], dtype=np.float32)
    else:
        _Xm = np.load(_x_path(cache_path), mmap_mode="r")
        def _rd(a, b): return np.asarray(_Xm[start + a:start + b], dtype=np.float32)

    # Calculate holdout dates from the TRAINING DATA window, not wall-clock run time.
    try:
        import pandas as pd

        _start_raw = (
            getattr(args, "start_date", None)
            or getattr(args, "data_start", None)
            or getattr(args, "start", None)
        )
        _end_raw = (
            getattr(args, "end_date", None)
            or getattr(args, "data_end", None)
            or getattr(args, "end", None)
        )
        if not _start_raw or not _end_raw:
            raise ValueError(f"missing training data window start/end ({_start_raw!r}, {_end_raw!r})")

        s_dt = pd.Timestamp(_start_raw)
        e_dt = pd.Timestamp(_end_raw)
        total_days = max(1, (e_dt - s_dt).days)
        holdout_frac = getattr(args, "promote_forward_frac", 0.1)
        holdout_days = max(1, int(total_days * holdout_frac))
        holdout_start = (e_dt - pd.Timedelta(days=holdout_days)).strftime("%Y-%m-%d")
        holdout_end = e_dt.strftime("%Y-%m-%d")

    except Exception as e:

        print(f"[PromotionGate] Date calculation failed ({e}), using default fallback")

        holdout_start = "2024-01-01"

        holdout_end = "2025-01-01"



    pairs = list(_get_pairs(args))

    if not pairs:

        pairs = ["EURUSD"]



    print(f"\n[PromotionGate] Running EXECUTION-AWARE Backtest for {model_name} on {holdout_start} -> {holdout_end}")



    try:

        import sys

        _ROOT = Path(__file__).resolve().parent.parent

        if str(_ROOT) not in sys.path:

            sys.path.insert(0, str(_ROOT))

        from scripts.backtest_model import run_execution_backtest



        bt_metrics = run_execution_backtest(
            model=model,
            pair_list=pairs,
            start_date=holdout_start,
            end_date=holdout_end,
            seq_len=getattr(args, "seq_len", 60),
            n_features=n_features,
            device=device,
            bar_freq=getattr(args, "bar_freq", "1Min"),
            data_source=getattr(args, "data_source", "dukascopy"),
            stop_pips=15.0, # Will be overridden by ATR tracking ideally, using defaults for gate
            take_pips=20.0,
            inference_batch_size=getattr(args, "batch_size", 4096),
        )

    except Exception as e:

        print(f"[PromotionGate] Execution backtest failed: {e}")

        bt_metrics = {"error": str(e)}



    if bt_metrics.get("error"):

        return {"promoted": False, "details": {"n_trades": 0.0, "error": bt_metrics["error"]},

                "reasons": [f"Execution backtest failed: {bt_metrics['error']}"],

                "summary": "REJECT (backtest error)"}



    # Extract metrics for PromotionGate

    pnls = bt_metrics.pop("signals_df", pd.DataFrame())["pnl_pips"].tolist() if "signals_df" in bt_metrics and "pnl_pips" in bt_metrics["signals_df"] else []

    bt_metrics.pop("equity_curve", [10000.0])



    if bt_metrics["n_trades"] < 1:

        return {"promoted": False, "details": {"n_trades": 0.0},
                "reasons": ["challenger took no trades on forward window"],
                "summary": "REJECT (no trades)"}

    folds = list(fold_sharpes) if fold_sharpes else []
    n_trials = max(1, len(folds))
    sharpe_std = float(np.std(folds)) if len(folds) > 1 else 0.0
    gate = PromotionGate(GateConfig(strict_psr=True))   # B-M4: deflated Sharpe for retrain selection


    # Overwrite the gate evaluate call with the execution-aware metrics directly

    result = gate.evaluate(

        sharpe=bt_metrics["sharpe"],

        profit_factor=bt_metrics.get("profit_factor", 1.0),

        max_drawdown=bt_metrics["max_drawdown"],

        n_trades=bt_metrics["n_trades"],

        regime_pnl={}, # Not tracking regime pnl in backtest_model return yet

        gross_pnl=bt_metrics["net_pnl"],

        transaction_costs=0.0, # Already accounted for in net_pnl by backtester

        n_backtest_trials=n_trials,
        backtest_sharpe_std=sharpe_std,
        emergency_retrain=bool(getattr(args, "finetune_warm_start", False)),
    )
    result.setdefault("details", {}).update({

        "forward_window": float(n_fwd),

        "sharpe": float(bt_metrics.get("sharpe", 0.0)),

        "profit_factor": float(bt_metrics.get("profit_factor", 0.0)),

        "max_drawdown": float(bt_metrics.get("max_drawdown", 0.0)),

        "n_trades": int(bt_metrics.get("n_trades", 0)),

        "net_pnl": float(bt_metrics.get("net_pnl", 0.0)),

    })

    print(f"[PromotionGate] {model_name}: {result.get('summary','?')} "
          f"| trades={len(pnls)} | forward_n={n_fwd}")


    # ╬ô├╢├ç╬ô├╢├ç Confidence threshold sweep ╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç╬ô├╢├ç

    # Sweep [0.35 ╬ô├ç┬¬ 0.60] and write {model_name}_threshold_tuning.json next to

    # the checkpoint. Only meaningful when confidence has real variance.

    try:

        from validation.promotion_gate import write_threshold_tuning_json

        import numpy as np

        conf_scores = bt_metrics.get("confidence_scores", []) or []

        conf_traded = np.array(conf_scores, dtype=float) if len(conf_scores) >= 30 else np.array([])

        if len(conf_traded) > 0 and conf_traded.std() > 1e-6:

            sweep = gate.sweep_confidence_threshold(

                trade_pnls=pnls,

                confidence_scores=conf_traded.tolist(),

                annualization=252.0,

                min_trades=max(30, len(pnls) // 20),

            )

            thr_path = (ckpt_dir / model_name / f"{model_name}_threshold_tuning.json"

                        if (ckpt_dir / model_name).is_dir()

                        else ckpt_dir / f"{model_name}_threshold_tuning.json")

            written = write_threshold_tuning_json(

                sweep,

                str(thr_path),

                model_name=model_name,

                extra_meta={"forward_n": int(n_fwd), "n_traded": int(len(pnls))},

            )

            opt_thr = sweep.get("optimal_threshold")

            opt_sr  = sweep.get("optimal_sharpe")

            result.setdefault("details", {})["optimal_confidence_threshold"] = opt_thr

            print(f"[ThresholdSweep] {model_name}: optimal={opt_thr} "

                  f"(Sharpe={opt_sr}) ╬ô├Ñ├å {written}")

        else:

            print(f"[ThresholdSweep] {model_name}: skipped ╬ô├ç├╢ regression confidence proxy "

                  f"has uniform variance; use a softmax head for meaningful tuning.")

    except Exception as _thr_exc:

        print(f"[ThresholdSweep] {model_name}: sweep failed ({_thr_exc}); continuing.")



    return result


def _auto_tune_next_run(
    config_path: str,
    history: dict,
    gate_result: dict,
    best_epoch: int,
    total_epochs: int,
    run_name: str = "unknown_run",
    dry_tune: bool = False,
) -> None:
    """Audit every hyperparameter proposal and (optionally) apply it.

    Always writes ``logs/auto_tune/<run_name>_proposal.json`` with structured
    records ΓÇö whether or not dry_tune is set and whether or not any changes
    are made.  High-risk fields (data_range, label_method, checkpoint_dir,
    production thresholds) are never mutated.
    """
    import json
    import shutil
    from datetime import datetime, timezone
    from pathlib import Path

    # ΓöÇΓöÇ constants ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    _HIGH_RISK = {
        "start_date", "end_date", "data_source", "label_method",
        "checkpoint_dir", "data_cache", "production",
    }

    def _proposal(issue, section, key, prev, new, reason, confidence):
        return {
            "issue":        issue,
            "section":      section,
            "key":          key,
            "prev_value":   prev,
            "new_value":    new,
            "reason":       reason,
            "confidence":   confidence,   # "low" | "medium" | "high"
        }

    cfg_path = Path(config_path) if config_path else Path("config/run.yaml")
    log_dir  = Path("logs/auto_tune")
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_run_name = _slug_part(run_name, max_len=180)

    # Use versioned config path instead of overwriting run.yaml
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    versioned_cfg_path = cfg_path.parent / f"run_{ts_str}.yaml"

    prop_file = log_dir / f"{safe_run_name}_proposal.json"


    proposals: list[dict] = []

    # ΓöÇΓöÇ try to load config ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    data = None
    if cfg_path.exists():
        try:
            from ruamel.yaml import YAML
            yaml_io = YAML()
            yaml_io.preserve_quotes = True
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml_io.load(f)
            backup = cfg_path.with_name(cfg_path.name + ".bak")
            shutil.copy2(cfg_path, backup)
        except ImportError:
            print("[Auto-Tune] ruamel.yaml not installed ΓÇö proposal-only mode.")
        except Exception as e:
            print(f"[Auto-Tune] Could not read config: {e}")
    else:
        print(f"[Auto-Tune] Config not found at {cfg_path} ΓÇö proposal-only mode.")

    # P3: detect whether this config was written by Optuna; if so skip heuristics
    # that would clobber the curriculum schedule Optuna already optimised.
    _optuna_applied = bool((data or {}).get("optuna", {}).get("applied", False)) if data else False
    if _optuna_applied:
        print("[Auto-Tune] Optuna-applied config detected ΓÇö all config mutations skipped "
              "(proposals still recorded for audit).")

    def _get(section, key, fallback):
        if data is None:
            return fallback
        return data.get(section, {}).get(key, fallback)

    def _class_balance_diagnostics(hist: dict) -> dict:
        pred_curve = hist.get("val_pred_counts") if isinstance(hist, dict) else None
        true_curve = hist.get("val_true_counts") if isinstance(hist, dict) else None
        pred = pred_curve[-1] if pred_curve else None
        true = true_curve[-1] if true_curve else None
        out = {
            "available": bool(pred),
            "quarantined": False,
            "pred_counts": pred,
            "true_counts": true,
            "pred_shares": None,
            "true_shares": None,
            "reason": None,
        }
        if not pred:
            return out
        pred_total = max(1, sum(int(x) for x in pred))
        pred_shares = [float(x) / pred_total for x in pred]
        out["pred_shares"] = pred_shares
        if true:
            true_total = max(1, sum(int(x) for x in true))
            out["true_shares"] = [float(x) / true_total for x in true]
        if min(pred_shares) < 0.02:
            out["quarantined"] = True
            out["reason"] = "missing_or_near_missing_prediction_class"
        elif max(pred_shares) > 0.85:
            out["quarantined"] = True
            out["reason"] = "dominant_prediction_class"
        return out

    _class_diag = _class_balance_diagnostics(history)
    _auto_tune_quarantined = bool(_class_diag.get("quarantined"))
    if _auto_tune_quarantined:
        proposals.append(_proposal(
            issue="class_balance_quarantine",
            section="tracking",
            key="auto_tune",
            prev="normal",
            new="proposal_only",
            reason=(
                f"validation prediction distribution unhealthy: "
                f"pred_counts={_class_diag.get('pred_counts')} "
                f"reason={_class_diag.get('reason')}; block config mutations"
            ),
            confidence="high",
        ))

    def _set(section, key, value):
        if _optuna_applied or _auto_tune_quarantined:
            return
        if data is None or section in _HIGH_RISK or key in _HIGH_RISK:
            return
        if section not in data:
            data[section] = {}
        data[section][key] = value

    # SYS-002: when tune_eval metrics are available, prefer them over val metrics
    # to prevent validation data from leaking into hyperparameter decisions.
    _use_tune_eval = bool(history.get("_tune_eval_isolated")) if isinstance(history, dict) else False
    if _use_tune_eval:
        print("[Auto-Tune] Using isolated tune-eval metrics (SYS-002 three-way split active)")

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # HEURISTIC 1 — Overfitting: val_loss >> train_loss at final epoch
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    if (
        history and isinstance(history, dict)
        and history.get("train_loss") and (history.get("tune_loss") if _use_tune_eval else history.get("val_loss"))
    ):
        t_loss = history["train_loss"][-1]
        v_loss = history["tune_loss"] if _use_tune_eval else history["val_loss"][-1]
        if t_loss and v_loss and v_loss > t_loss * 1.15:
            gen_gap = v_loss - t_loss
            old_do = float(_get("model", "dropout", 0.25))
            new_do = min(0.50, old_do + 0.05)
            proposals.append(_proposal(
                issue       = "overfitting",
                section     = "model",
                key         = "dropout",
                prev        = old_do,
                new         = round(new_do, 2),
                reason      = f"val_loss ({v_loss:.4f}) > train_loss ({t_loss:.4f}) * 1.15 "
                              f"(gen_gap={gen_gap:.4f})",
                confidence  = "high" if gen_gap > t_loss * 0.30 else "medium",
            ))
            _set("model", "dropout", float(f"{new_do:.2f}"))

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # HEURISTIC 2 ΓÇö Premature early stopping
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    if total_epochs > 0 and best_epoch is not None and best_epoch < total_epochs * 0.25:
        old_lr = float(_get("training", "lr", 5e-5))
        new_lr = old_lr * 0.5
        proposals.append(_proposal(
            issue      = "premature_early_stop",
            section    = "training",
            key        = "lr",
            prev       = old_lr,
            new        = float(f"{new_lr:.2e}"),
            reason     = f"best_epoch={best_epoch} < 25% of total={total_epochs}",
            confidence = "medium",
        ))
        _set("training", "lr", float(f"{new_lr:.2e}"))

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # HEURISTIC 3 ΓÇö Sharpe collapse: peaked early then degraded
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    val_sharpe_curve = history.get("val_sharpe", []) if isinstance(history, dict) else []
    if val_sharpe_curve and len(val_sharpe_curve) >= 4:
        peak_val   = max(val_sharpe_curve)
        peak_epoch = val_sharpe_curve.index(peak_val)
        final_val  = val_sharpe_curve[-1]
        collapse   = (peak_val - final_val) / max(abs(peak_val), 1e-9)
        if collapse > 0.20 and peak_epoch < len(val_sharpe_curve) * 0.60:
            # LR was likely too high ΓåÆ reduce warmup peak
            old_lr = float(_get("training", "lr", 5e-5))
            new_lr = max(1e-6, old_lr * 0.70)
            proposals.append(_proposal(
                issue      = "sharpe_collapse",
                section    = "training",
                key        = "lr",
                prev       = old_lr,
                new        = float(f"{new_lr:.2e}"),
                reason     = (f"val_sharpe peaked at epoch {peak_epoch} ({peak_val:.4f}) "
                              f"then collapsed to {final_val:.4f} "
                              f"(drop={collapse:.1%})"),
                confidence = "high" if collapse > 0.40 else "medium",
            ))
            _set("training", "lr", float(f"{new_lr:.2e}"))

            # Also nudge patience down so we stop before the collapse
            old_pat = int(_get("training", "patience", 6))
            new_pat = max(3, min(old_pat, peak_epoch + 2))
            if new_pat != old_pat:
                proposals.append(_proposal(
                    issue      = "sharpe_collapse",
                    section    = "training",
                    key        = "patience",
                    prev       = old_pat,
                    new        = new_pat,
                    reason     = f"stop before collapse; peak at epoch {peak_epoch}",
                    confidence = "medium",
                ))
                _set("training", "patience", new_pat)

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # HEURISTIC 4 ΓÇö Gate failure on drawdown
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    reasons = gate_result.get("reasons", []) if gate_result else []
    if any("drawdown" in str(r).lower() for r in reasons):
        if data and "rl" in data and "reward" in data.get("rl", {}):
            old_pen = float(data["rl"]["reward"].get("drawdown", 0.5))
            new_pen = min(2.0, old_pen + 0.25)
            proposals.append(_proposal(
                issue      = "gate_fail_drawdown",
                section    = "rl.reward",
                key        = "drawdown",
                prev       = old_pen,
                new        = round(new_pen, 2),
                reason     = "promotion gate rejected on drawdown criterion",
                confidence = "medium",
            ))
            if data is not None and not _optuna_applied and not _auto_tune_quarantined:
                data["rl"]["reward"]["drawdown"] = float(f"{new_pen:.2f}")

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # HEURISTIC 5 ΓÇö Gate failure on profit factor
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    if any("profit factor" in str(r).lower() for r in reasons):
        old_tx = float(_get("execution", "slippage_vol_alpha", 0.5))
        new_tx = min(1.0, old_tx + 0.10)
        proposals.append(_proposal(
            issue      = "gate_fail_profit_factor",
            section    = "execution",
            key        = "slippage_vol_alpha",
            prev       = old_tx,
            new        = round(new_tx, 2),
            reason     = "promotion gate rejected on profit factor; tighten slippage model",
            confidence = "low",
        ))
        _set("execution", "slippage_vol_alpha", float(f"{new_tx:.2f}"))

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # HEURISTIC 6 ΓÇö Frequent Curriculum Stalls (Noisy Gradients / Hard Data)
    # P3: skipped when Optuna already optimised the curriculum schedule.
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    stalls_curve = history.get("curriculum_stalls", []) if isinstance(history, dict) else []
    if stalls_curve and stalls_curve[-1] >= 5 and not _optuna_applied and not _auto_tune_quarantined:
        # 1. Increase batch_size to smooth gradients
        old_bs = int(_get("training", "batch_size", 256))
        new_bs = min(512, int(old_bs * 1.25))
        if new_bs != old_bs:
            proposals.append(_proposal(
                issue      = "frequent_stalls",
                section    = "training",
                key        = "batch_size",
                prev       = old_bs,
                new        = new_bs,
                reason     = f"Total stalls reached {stalls_curve[-1]}; increasing batch size to smooth gradients",
                confidence = "medium",
            ))
            _set("training", "batch_size", new_bs)

        # 2. Decrease seq_len to simplify the learning task
        old_seq = int(_get("training", "seq_len", 60))
        new_seq = max(15, int(old_seq * 0.75))
        if new_seq != old_seq:
            proposals.append(_proposal(
                issue      = "frequent_stalls",
                section    = "training",
                key        = "seq_len",
                prev       = old_seq,
                new        = new_seq,
                reason     = f"Total stalls reached {stalls_curve[-1]}; reducing seq_len to simplify the task",
                confidence = "medium",
            ))
            _set("training", "seq_len", new_seq)

            # Disable the seq_schedule to avoid conflicts
            old_sched = _get("curriculum", "seq_schedule", None)
            if old_sched:
                proposals.append(_proposal(
                    issue      = "frequent_stalls",
                    section    = "curriculum",
                    key        = "seq_schedule",
                    prev       = "active",
                    new        = "disabled",
                    reason     = "Disabled sequence schedule because base seq_len was dynamically reduced",
                    confidence = "medium",
                ))
                _set("curriculum", "seq_schedule", [])
    elif stalls_curve and stalls_curve[-1] >= 5 and (_optuna_applied or _auto_tune_quarantined):
        pass  # all mutations blocked when optuna.applied or class balance is quarantined.

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # HEURISTIC 7 ΓÇö Perfect Stability (Too Easy / Underfitting)
    # P3: skipped when Optuna already optimised the curriculum schedule.
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    if stalls_curve and stalls_curve[-1] == 0 and not _optuna_applied and not _auto_tune_quarantined:
        if isinstance(history, dict) and history.get("train_loss") and history.get("val_loss"):
            t_loss = history["train_loss"][-1]
            v_loss = history["val_loss"][-1]
            if t_loss and v_loss and v_loss < t_loss * 1.05:
                old_seq = int(_get("training", "seq_len", 60))
                new_seq = min(120, int(old_seq * 1.25))
                if new_seq != old_seq:
                    proposals.append(_proposal(
                        issue      = "perfect_stability",
                        section    = "training",
                        key        = "seq_len",
                        prev       = old_seq,
                        new        = new_seq,
                        reason     = "Zero stalls and tight gen_gap; increasing seq_len to challenge the model",
                        confidence = "low",
                    ))
                    _set("training", "seq_len", new_seq)

    # ΓöÇΓöÇ always write proposal JSON ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    proposal_doc = {
        "run_name":       run_name,
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "dry_tune":       dry_tune,
        "optuna_applied": _optuna_applied,   # P3: flag recorded in proposal for auditability
        "class_balance":  _class_diag,
        "auto_tune_quarantined": _auto_tune_quarantined,
        "applied":        (not dry_tune and not _auto_tune_quarantined
                           and data is not None and len(proposals) > 0),
        "n_proposals":    len(proposals),
        "proposals":      proposals,
    }
    try:
        _safe_save_json(proposal_doc, prop_file)
    except Exception:
        try:
            with open(prop_file, "w", encoding="utf-8") as f:
                json.dump(proposal_doc, f, indent=2, default=str)
        except Exception:
            pass

    # ΓöÇΓöÇ print summary ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    if proposals:
        action = "(Dry-Run) Would modify" if dry_tune else "Modified"
        if _auto_tune_quarantined:
            action = "Quarantined; proposal-only"
        print(f"\n[Auto-Tune] {action} {cfg_path.name} for the next run "
              f"({len(proposals)} proposal(s)):")
        for p in proposals:
            print(f"  [{p['confidence'].upper()}] {p['issue']}: "
                  f"{p['section']}.{p['key']} "
                  f"{p['prev_value']} -> {p['new_value']}  ({p['reason']})")
        print(f"  Proposal written -> {prop_file}")
    else:
        print("\n[Auto-Tune] No hyperparameter changes recommended. "
              f"Proposal written -> {prop_file}")

    # ── write back to config (live mode only) ──────────────────────────
    if not dry_tune and not _auto_tune_quarantined and data is not None and proposals:
        try:
            yaml_io = YAML()
            yaml_io.preserve_quotes = True
            with open(versioned_cfg_path, "w", encoding="utf-8") as f:
                yaml_io.dump(data, f)
            print(f"[Auto-Tune] Wrote tuned config -> {versioned_cfg_path}")
        except Exception as e:
            print(f"[Auto-Tune] Failed to write back config: {e}")



def _best_epoch_from_history(history: dict) -> int:

    """Pick the epoch index used by auto-tune without depending on outer locals."""

    if isinstance(history, dict) and history.get("val_sharpe"):

        return int(history["val_sharpe"].index(max(history["val_sharpe"])))

    if isinstance(history, dict) and history.get("val_loss"):

        return int(history["val_loss"].index(min(history["val_loss"])))

    return 0





def _history_for_auto_tune(history_or_folds) -> dict:

    """Normalize single-split or walk-forward histories into one curve dict.

    Walk-forward: use the **last completed fold** only. Concatenating all fold
    curves creates a fake multi-hundred-epoch run and breaks auto-tune heuristics.
    """

    if isinstance(history_or_folds, dict):

        return history_or_folds

    if not isinstance(history_or_folds, list) or not history_or_folds:

        return {}

    last_entry = history_or_folds[-1]

    if isinstance(last_entry, dict) and isinstance(last_entry.get("history"), dict):

        return dict(last_entry["history"])

    return {}


def _evaluate_tune_split(model, cache_path: str, tune_idx: np.ndarray, args,
                         device, amp_dtype) -> dict:
    """SYS-002: Evaluate best model on the held-out tune split.

    Returns a dict with 'tune_loss' and 'tune_sharpe' that can be injected into
    the auto-tune history, preventing val-set reuse for hyperparameter decisions.
    """
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    tune_idx_sorted = np.sort(tune_idx)
    seq_len = int(getattr(args, "seq_len", 64) or 64)
    batch_size = int(getattr(args, "batch_size", 256) or 256)

    try:
        tune_ds = ZarrStreamDataset(
            cache_path, tune_idx_sorted, seq_len,
            targets="direction" if getattr(args, "classification", False) or getattr(args, "multitask", False) else "returns",
            augment=False,
        )
        tune_loader = DataLoader(tune_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=0, pin_memory=False)
    except Exception as e:
        print(f"[TuneEval] Could not create tune dataloader: {e}")
        return {}

    criterion = torch.nn.CrossEntropyLoss() if getattr(args, "classification", False) else torch.nn.MSELoss()
    total_loss = 0.0
    total_samples = 0
    all_returns = []

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
        for batch in tune_loader:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)
            try:
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                loss = criterion(out.squeeze(-1) if out.dim() > y.dim() else out, y)
                bs = x.size(0)
                total_loss += loss.item() * bs
                total_samples += bs
                if not getattr(args, "classification", False):
                    all_returns.append(out.squeeze(-1).cpu())
            except Exception:
                continue

    if total_samples == 0:
        return {}

    tune_loss = total_loss / total_samples
    tune_sharpe = 0.0
    if all_returns:
        rets = torch.cat(all_returns)
        std = rets.std()
        if std > 1e-8:
            tune_sharpe = float((rets.mean() / std) * (252 ** 0.5))

    print(f"[TuneEval] tune_loss={tune_loss:.6f}  tune_sharpe={tune_sharpe:.4f}  "
          f"samples={total_samples:,}")
    return {"tune_loss": tune_loss, "tune_sharpe": tune_sharpe, "tune_samples": total_samples}





def _parse_pretrain_ablation_models(value) -> set[str]:

    """Return model names that should run no-pretrain proof when ablation=auto."""

    default = {"tft", "transformer", "haelt"}

    if value is None:

        return default

    if isinstance(value, str):

        raw = value.strip()

        if not raw:

            return default

        if raw.lower() in {"none", "false", "off", "disabled"}:

            return set()

        return {part.strip().lower() for part in raw.split(",") if part.strip()}

    if isinstance(value, (list, tuple, set)):

        return {str(part).strip().lower() for part in value if str(part).strip()}

    return default





def _csv_set(value) -> set[str]:

    if value is None:

        return set()

    if isinstance(value, str):

        return {part.strip().lower() for part in value.split(",") if part.strip()}

    if isinstance(value, (list, tuple, set)):

        return {str(part).strip().lower() for part in value if str(part).strip()}

    return set()





def _feature_base_name(name: str) -> str:

    return str(name).split("::")[-1].split(":", 1)[-1]





def _feature_ablation_config(args) -> dict:

    cfg = getattr(args, "feature_ablation", None)

    cfg = dict(cfg) if isinstance(cfg, dict) else {}



    cli_name = str(getattr(args, "feature_ablation_name", "") or "").strip()

    cli_drop_groups = _csv_set(getattr(args, "feature_ablation_drop_groups", ""))

    cli_keep_groups = _csv_set(getattr(args, "feature_ablation_keep_groups", ""))

    cli_drop_features = _csv_set(getattr(args, "feature_ablation_drop_features", ""))



    if cli_name:

        cfg["name"] = cli_name

        cfg["enabled"] = True

    if cli_drop_groups:

        cfg["drop_groups"] = sorted(cli_drop_groups)

        cfg["enabled"] = True

    if cli_keep_groups:

        cfg["keep_groups"] = sorted(cli_keep_groups)

        cfg["enabled"] = True

    if cli_drop_features:

        cfg["drop_features"] = sorted(cli_drop_features)

        cfg["enabled"] = True



    cfg.setdefault("enabled", False)

    cfg.setdefault("name", "full_features")

    cfg.setdefault("drop_groups", [])

    cfg.setdefault("keep_groups", [])

    cfg.setdefault("drop_features", [])

    return cfg





def _build_feature_ablation_mask(schema: list, feature_groups: dict, cfg: dict, n_features: int) -> tuple[Optional[np.ndarray], dict]:

    """Build a static feature mask and a JSON-ready report for feature ablation runs."""

    enabled = bool(cfg.get("enabled", False))

    report = {

        "enabled": enabled,

        "name": str(cfg.get("name", "full_features") or "full_features"),

        "drop_groups": sorted(_csv_set(cfg.get("drop_groups", []))),

        "keep_groups": sorted(_csv_set(cfg.get("keep_groups", []))),

        "drop_features": sorted(_csv_set(cfg.get("drop_features", []))),

        "n_features": int(n_features),

        "masked_count": 0,

        "active_count": int(n_features),

        "masked_by_group": {},

        "masked_features_sample": [],

    }

    if not enabled:

        return None, report

    if len(schema) != n_features:

        report["warning"] = f"schema length {len(schema)} != n_features {n_features}; ablation mask skipped"

        return None, report



    drop_groups = set(report["drop_groups"])

    keep_groups = set(report["keep_groups"])

    drop_features = set(report["drop_features"])

    group_feature_map: dict[str, set[str]] = {}

    for g_name, g_cfg in (feature_groups or {}).items():

        group_feature_map[str(g_name).lower()] = {

            str(f).lower() for f in (g_cfg or {}).get("features", []) if str(f).strip()

        }

    if keep_groups:

        drop_groups |= {g for g in group_feature_map if g not in keep_groups}



    mask = np.ones(n_features, dtype=np.float32)

    masked_names: list[str] = []

    for idx, raw_name in enumerate(schema):

        base = _feature_base_name(str(raw_name)).lower()

        reason_group = None

        if base in drop_features:

            reason_group = "__explicit_features__"

        else:

            for g_name in drop_groups:

                if base in group_feature_map.get(g_name, set()):

                    reason_group = g_name

                    break

        if reason_group is not None:

            mask[idx] = 0.0

            report["masked_by_group"][reason_group] = int(report["masked_by_group"].get(reason_group, 0)) + 1

            if len(masked_names) < 40:

                masked_names.append(str(raw_name))



    report["masked_count"] = int((mask == 0.0).sum())

    report["active_count"] = int((mask != 0.0).sum())

    report["masked_features_sample"] = masked_names

    return mask, report





def _metric_from_gate(details: dict, key: str, default=None):

    for candidate in (key, f"forward_{key}", f"holdout_{key}"):

        if isinstance(details, dict) and candidate in details:

            return details.get(candidate)

    return default





def _append_model_comparison_report(args, model_name: str, train_summary: dict, gate_result: dict) -> None:

    """Write a shared same-forward-holdout comparison file across model runs."""

    root = Path(args.checkpoint_dir)

    path = root / "model_comparison.json"

    existing = _read_json_dict(path)

    details = gate_result.get("details", {}) if isinstance(gate_result, dict) else {}

    row = {

        "model_name": model_name,

        "recipe_name": getattr(args, "recipe_name", None),

        "loss": getattr(args, "loss", None),

        "seq_len": int(getattr(args, "seq_len", 0) or 0),

        "pretrain_enabled": bool(getattr(args, "pretrain", False)),

        "feature_ablation": (_feature_ablation_config(args).get("name") or "full_features"),

        "validation": {

            "best_val_sharpe": train_summary.get("best_val_sharpe"),

            "best_val_loss": train_summary.get("best_val_loss"),

            "final_val_sharpe": train_summary.get("final_val_sharpe"),

            "gen_gap_final": train_summary.get("gen_gap_final"),

        },

        "forward_holdout": {

            "promoted": bool(gate_result.get("promoted", False)) if isinstance(gate_result, dict) else False,

            "summary": gate_result.get("summary") if isinstance(gate_result, dict) else None,

            "sharpe": _metric_from_gate(details, "sharpe"),

            "profit_factor": _metric_from_gate(details, "profit_factor"),

            "max_drawdown": _metric_from_gate(details, "max_drawdown"),

            "n_trades": _metric_from_gate(details, "n_trades"),

            "forward_window": details.get("forward_window") if isinstance(details, dict) else None,

            "reasons": gate_result.get("reasons", []) if isinstance(gate_result, dict) else [],

        },

        "updated_at": datetime.now(timezone.utc).isoformat(),

    }



    rows = [r for r in existing.get("models", []) if r.get("model_name") != model_name]

    rows.append(row)



    def _score(r):

        fwd = r.get("forward_holdout", {})

        val = r.get("validation", {})

        sharpe = fwd.get("sharpe")

        if sharpe is None:

            sharpe = val.get("best_val_sharpe")

        try:

            return float(sharpe)

        except Exception:

            return float("-inf")



    rows = sorted(rows, key=_score, reverse=True)

    report = {

        "run_name": getattr(args, "run_name", None),

        "checkpoint_dir": str(root),

        "comparison_basis": "same promotion forward holdout window from config/promote_forward_frac",

        "models": rows,

        "leader": rows[0].get("model_name") if rows else None,

        "updated_at": datetime.now(timezone.utc).isoformat(),

    }

    _safe_save_json(report, path)

    print(f"[ModelComparison] Updated -> {path}")





def _maybe_auto_tune_next_run(

    args,

    history: dict,

    gate_result: Optional[dict] = None,

    *,

    phase: str = "main",

    model_name: str = "model",

    force_dry: bool = False,

) -> None:

    """Centralized auto-tune entrypoint for every training phase.



    Baseline/pretrain-ablation phases write proposal artifacts too, but force

    dry-run mode so comparison runs do not mutate config before the main model

    has trained and passed through promotion.

    """

    if not bool(getattr(args, "auto_tune", True)):

        print(f"[Auto-Tune] Disabled for {model_name}/{phase} (--no-auto-tune or tracking.auto_tune=false)")

        return

    hist = history if isinstance(history, dict) else {}

    best_epoch = _best_epoch_from_history(hist)

    total_epochs = len(hist.get("train_loss", [])) or int(getattr(args, "epochs", 0) or 0)

    run_base = str(getattr(args, "run_name_slug", "") or getattr(args, "run_name", "unknown_run") or "unknown_run")

    phase_run_name = _slug_part(f"{run_base}_{model_name}_{phase}", max_len=180)

    _auto_tune_next_run(

        getattr(args, "config", "config/run.yaml"),

        hist,

        gate_result or {"promoted": True, "reasons": []},

        best_epoch=best_epoch,

        total_epochs=total_epochs,

        run_name=phase_run_name,

        dry_tune=bool(
            force_dry
            or getattr(args, "dry_tune", False)
            or getattr(args, "all_models", False)
        ),

    )



from training.config_validate import validate_run_config


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    run_start_time = datetime.now(timezone.utc).isoformat()
    args     = parse_args()
    apply_hardware_profile(args)
    _set_global_seed(getattr(args, "seed", None))

    # Expand user-home-relative paths (e.g. ~/forex_data) from config/CLI
    args.checkpoint_dir = str(Path(args.checkpoint_dir).expanduser())
    args.data_cache     = str(Path(args.data_cache).expanduser())

    if getattr(args, "validate_config", False):
        sys.exit(validate_run_config(args))

    if getattr(args, "hparam_search", False) and getattr(args, "all_models", False):
        print(
            f"[HPO] --hparam-search runs only --model {args.model} "
            "(config model.all_models ignored)."
        )
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

    # Import HardExampleMiner for post-validation hard-sample collection
    try:
        from training.hard_example_miner import HardExampleMiner as _HardMiner
    except Exception:
        _HardMiner = None

    _all_pairs   = _get_pairs(args)
    _pairs_str   = ", ".join(_all_pairs)
    _embed_str   = (f"  embed={getattr(args,'pair_embed_dim',0)}d"
                    if len(_all_pairs) > 1 else "")
    if getattr(args, "all_models", False):

        _requested_models = [

            m.strip().lower()

            for m in str(getattr(args, "models", "") or "").split(",")

            if m.strip()

        ]

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
                pairs=getattr(args, 'pairs', []),
                data_window=f"{getattr(args, 'start_date', 'unknown')} to {getattr(args, 'end_date', 'unknown')}"
            )
    except Exception as e:
        print(f"[Discord] Failed to send training_started: {e}")
            
    print(f"\n{'='*62}")
    print(f"  Forex Scaling Model ΓÇö 20M Tick GPU Trainer")
    print(f"  Run: {run_name}  |  Ticks: {args.n_ticks:,}  |  Mode: {_model_display}")
    if _queue_display:
        print(f"  Model queue: {_queue_display}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Pairs: {_pairs_str}{_embed_str}")
    print(f"  Strategy: {args.strategy_mode}  |  Bars: {args.bar_freq}  |  "
          f"Lookahead: {args.lookahead_bars} bars")
    print(f"  Batch: {args.batch_size}  |  Epochs: {args.epochs}  |  AMP: {args.amp}")
    print(f"  Labels: {args.label_method}  |  Loss: {args.loss}  |  "
          f"Early-stop: {args.early_stop_metric}")
    print(f"  Historical news: {getattr(args, 'historical_news_mode', 'calendar')}  |  "
          f"Cache format: {'NPY on Windows' if sys.platform == 'win32' else 'Zarr'}")
    if getattr(args, "multitask", False):
        print(f"  MultiTask: ON  (w_ret={args.mt_w_ret}  w_conf={args.mt_w_conf})")
    if getattr(args, "pretrain_regime", False):
        print(f"  Pretrain: regime-aware TSCL (hard negatives from opposite regime)")
    if getattr(args, "train_ensemble", False):
        print(f"  Ensemble meta-learner: ON  (epochs={args.ensemble_epochs}  "
              f"div_weight={args.ensemble_div_weight})")
    print(f"{'='*62}")

    # ΓÜá∩╕Å  Synthetic data warning ΓÇö always visible
    if getattr(args, "data_source", "dukascopy") == "synthetic":
        print(f"\n{'!'*62}")
        print(f"  ΓÜá  WARNING: SYNTHETIC DATA")
        print(f"  Training on artificially generated price data.")
        print(f"  Results DO NOT reflect real market performance.")
        print(f"  Use --data-source dukascopy for real data.")
        print(f"{'!'*62}\n")

    device, n_gpus, amp_dtype = setup_device(
        dtype_override=getattr(args, "dtype", "auto"),
        deterministic=bool(getattr(args, "deterministic", False)),
    )

    # A-M3: opt-in fully-deterministic mode (overrides setup_device's speed flags).
    if getattr(args, "deterministic", False):
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            print("[Deterministic] cudnn.deterministic=True, benchmark=False, "
                  "deterministic algorithms ON")
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

    cache_path, n_samples, n_features, scaler = build_dataset_chunked(args)
    n_samples = _clamp_n_samples_to_disk(cache_path, n_samples)
    _warn_multitask_cache_sidecars(cache_path, args)
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        _fp = int(getattr(args, "_f_per_pair", n_features) or n_features)
        args._feat_names = list(scaler.feature_names_in_)[:_fp]
    print(f"\n[Dataset] {n_samples:,} sequences ├ù {n_features} features ├ù "
          f"seq_len {args.seq_len}")

    if getattr(args, "data_quality_check", False):
        print("\n[DataQuality] Running data quality check...")
        import subprocess
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "data_quality_check.py"
        try:
            subprocess.run(
                [sys.executable, str(script_path), "--cache-path", str(cache_path), "--full"],
                check=True
            )
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
                ks_pvalue_threshold=float(getattr(args, "drift_ks_pvalue_threshold", MONITORING.get("ks_pvalue_threshold", 0.05))),
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
        wandb_run = wandb.init(
            project = args.wandb_project,
            name    = run_name,
            config  = {**vars(args), "n_samples": n_samples,
                       "n_features": n_features},
            tags    = [args.model, args.data_source, "20M"],
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

    if args.all_models:
        requested_models = [

            m.strip().lower()

            for m in str(getattr(args, "models", "") or "").split(",")

            if m.strip()

        ]

        if requested_models:

            unknown = [m for m in requested_models if m not in MODEL_REGISTRY]

            if unknown:

                raise ValueError(f"Unknown --models entries {unknown}; expected one of {list(MODEL_REGISTRY)}")

            models_to_train = requested_models

        else:

            models_to_train = ["haelt", "mamba", "catboost"]

    else:

        models_to_train = [args.model]

    if args.all_models:

        if getattr(args, "resume", False) and not getattr(args, "retrain_completed_models", False):
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
        model = build_model(model_name, n_features, model_args).to(device)

        # -- Preflight Checks --------------------------------------------------
        # Run a small sanity check using a subset of the dataset
        if not getattr(model_args, "resume", False):
            try:
                temp_ds = ZarrStreamDataset(cache_path, np.arange(min(1000, n_samples)), shuffle_chunks=False)
                temp_dl = DataLoader(temp_ds, batch_size=min(model_args.batch_size, 32), num_workers=0)
                run_preflight_sanity_checks(model, device, temp_dl, model_args)
                del temp_dl, temp_ds
            except Exception as e:
                print(f"[Main] Preflight failed for {model_name}: {e}")
                if not getattr(model_args, "ignore_preflight", False):
                    sys.exit(1)

        # -- PyTorch profiler (--profile) --------------------------------------
        if getattr(model_args, "profile", False):
            _prof_ds = ZarrStreamDataset(cache_path, np.arange(min(512, n_samples)), shuffle_chunks=False)
            _prof_dl = DataLoader(_prof_ds, batch_size=min(model_args.batch_size, 64), num_workers=0)
            run_profiler(
                build_model(model_name, n_features, model_args).to(device),
                _prof_dl, device, amp_dtype, model_args.amp,
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
                ta = argparse.Namespace(**vars(model_args))
                ta.lr          = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
                ta.hidden_size = trial.suggest_categorical("hidden_size", [128,256,512])
                ta.d_model     = trial.suggest_categorical("d_model", [128,256,512])
                ta.dropout     = trial.suggest_float("dropout", 0.05, 0.3)
                ta.batch_size  = trial.suggest_categorical("batch_size", [128,256,512])
                ta.epochs      = 5
                ta.patience    = 3
                ta.resume      = False
                ta.all_models  = False
                # Architecture search changes hidden/d_model ΓÇö existing contrastive
                # checkpoints won't transfer; skip pretrain for proxy trials.
                ta.pretrain    = False
                ta.disable_pretrain_load = True
                ta.pretrain_ablation = "false"
                ta.checkpoint_dir = str(
                    Path(model_args.checkpoint_dir) / "hpo_trials" / f"trial_{trial.number}"
                )
                build_model(model_name, n_features, ta).to(device)
                h, bv = supervised_train(model_name, cache_path, n_samples,
                                          n_features, ta, device, n_gpus, run=None)
                return bv
            direction = "maximize" if model_args.early_stop_metric == "sharpe" else "minimize"
            study = optuna.create_study(direction=direction,
                                        pruner=optuna.pruners.MedianPruner())
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
            for k,v in study.best_params.items():
                setattr(model_args, k.replace("-","_"), v)
            print(f"[HPO] Best: {study.best_params}  val={study.best_value:.6f}")
            model = build_model(model_name, n_features, model_args).to(device)

        # Optional contrastive pre-training (TSCL expects dense embeddings, not class logits)
        _baseline_cv_hist = None
        _abl_arg = str(getattr(model_args, "pretrain_ablation", "auto")).lower()
        _pretrain_ablation_models = _parse_pretrain_ablation_models(

            getattr(model_args, "pretrain_ablation_models", "")

        )

        _run_ablation = (_abl_arg == "true") or (_abl_arg == "auto" and model_name in _pretrain_ablation_models)
        if _run_ablation:
            _baseline_done, _baseline_reason = _baseline_ablation_completion_status(
                model_name, model_args.checkpoint_dir, model_args
            )
            if getattr(model_args, "resume", False) and _baseline_done:
                print(f"[Ablation] Baseline already complete for {model_name}; "
                      f"skipping no-pretrain proof ({_baseline_reason}).")
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
                        f"baseline_{model_name}", cache_path, n_samples, n_features,
                        base_args, device, n_gpus, run=wandb_run,
                        train_idx=tr_i, val_idx=va_i, fold_id=fi,
                        amp_dtype=amp_dtype,
                    )
                    _baseline_cv_hist.append({"fold": fi, "best_metric": _bv, "history": _h})
            else:
                _h, _bv = supervised_train(
                    f"baseline_{model_name}", cache_path, n_samples, n_features,
                    base_args, device, n_gpus, run=wandb_run,
                    amp_dtype=amp_dtype,
                )
                _baseline_cv_hist = [{"fold": 0, "best_metric": _bv, "history": _h}]
            print(f"[Ablation] Baseline completed.")
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
            print(f"[Pretrain] Supervised checkpoints already exist for {model_name}; "
                  f"skipping pretrain on resume ({_supervised_reason}).")
        elif model_args.pretrain:
            if model_args.loss == "cross_entropy":
                pt_ns = argparse.Namespace(**vars(model_args))
                pt_ns.loss = "huber"
                model = build_model(model_name, n_features, pt_ns).to(device)
            model = run_pretrain(model, cache_path, n_features, model_args, device, run=wandb_run)

        # Supervised training (single split or walk-forward CV)
        log_dir = Path(model_args.checkpoint_dir).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        _fold_metrics = None   # populated by walk-forward for the promotion gate
        _holdout_n = _promotion_holdout_n(n_samples, model_args)
        _cv_n = max(0, n_samples - _holdout_n)
        if _holdout_n > 0:
            print(f"[Holdout] Reserved last {_holdout_n:,} bars for promotion gate "
                  f"(CV uses 0:{_cv_n:,})")
        if model_args.walk_forward_cv:
            _embargo = _embargo_bars(model_args)   # A-H3: seq_len + lookahead + delay
            _purge = _purge_bars(model_args)
            _method = _validation_method(model_args)
            splits = walk_forward_splits(
                _cv_n, model_args.walk_forward_folds, _embargo, _purge, _method,
            )
            print(f"[WalkForward] {len(splits)} folds | embargo={_embargo} purge={_purge} method={_method}")
            cv_hist: list[dict] = []
            _start_fold = 0
            _artifact_run_name = str(getattr(model_args, "run_name_slug", "") or _slug_part(run_name, max_len=140))
            _artifact_model_name = _slug_part(model_name, max_len=80)
            if getattr(model_args, "resume", False):
                _resume_fold = _latest_resumable_fold(
                    model_name, model_args.checkpoint_dir, len(splits)
                )
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
                history, best_val = supervised_train(
                    model_name, cache_path, n_samples, n_features,
                    model_args, device, n_gpus, run=wandb_run,
                    train_idx=tr_i, val_idx=va_i, fold_id=fi,
                    amp_dtype=amp_dtype,
                )
                cv_hist.append({"fold": fi, "best_metric": best_val, "history": history})
            _artifact_run_name = str(getattr(model_args, "run_name_slug", "") or _slug_part(run_name, max_len=140))

            _artifact_model_name = _slug_part(model_name, max_len=80)

            with open(log_dir / f"{_artifact_run_name}_{_artifact_model_name}_cv.json", "w", encoding="utf-8") as fp:
                import json
                json.dump(cv_hist, fp)
            _promote_best_fold(model_name, model_args.checkpoint_dir, cv_hist,
                               model_args.early_stop_metric, alerter=alerter)
            _generate_model_card(model_name, model_args, cv_hist, model_args.checkpoint_dir, n_features)
            _fold_metrics = [e.get("best_metric") for e in cv_hist
                             if e.get("best_metric") is not None]
        else:
            history, best_val = supervised_train(
                model_name, cache_path, n_samples, n_features,
                model_args, device, n_gpus, run=wandb_run,
                amp_dtype=amp_dtype,
            )
            with open(log_dir / f"{run_name}_{model_name}.json", "w", encoding="utf-8") as fp:
                import json
                json.dump(history, fp)
            _generate_model_card(model_name, model_args, history, model_args.checkpoint_dir, n_features)

        # SYS-002: evaluate best model on tune split (isolated from val/early-stopping)
        _tune_eval_idx = getattr(model_args, "_tune_eval_idx", None)
        _tune_eval_metrics = {}
        if _tune_eval_idx is not None and len(_tune_eval_idx) > 0:
            try:
                _best_ckpt = Path(model_args.checkpoint_dir) / f"{model_name}_best.pt"
                if _best_ckpt.exists():
                    _ckpt_data = torch.load(_best_ckpt, map_location=device, weights_only=False)
                    _eval_model = build_model(model_name, n_features, model_args).to(device)
                    _eval_model.load_state_dict(
                        _ckpt_data["model_state_dict"] if isinstance(_ckpt_data, dict) and "model_state_dict" in _ckpt_data
                        else _ckpt_data.get("state_dict", _ckpt_data) if isinstance(_ckpt_data, dict) else _ckpt_data
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
        _ts_vloss_curve  = _ts_hist.get("val_loss", [])
        _ts_tloss_curve  = _ts_hist.get("train_loss", [])
        _ts_summary = {
            "model_name":       model_name,
            "run_name":         run_name,
            "train_mode":       "walk_forward_cv" if model_args.walk_forward_cv else "single_split",
            "n_folds":          len(cv_hist) if model_args.walk_forward_cv and 'cv_hist' in locals() else 1,
            "n_samples":        int(n_samples),
            "n_features":       int(n_features),
            "epochs_completed": len(_ts_tloss_curve),
            "best_val_loss":    round(min(_ts_vloss_curve), 6) if _ts_vloss_curve else None,
            "best_val_sharpe":  round(max(_ts_sharpe_curve), 6) if _ts_sharpe_curve else None,
            "final_val_loss":   round(_ts_vloss_curve[-1], 6) if _ts_vloss_curve else None,
            "final_val_sharpe": round(_ts_sharpe_curve[-1], 6) if _ts_sharpe_curve else None,
            "gen_gap_final":    round(_ts_vloss_curve[-1] - _ts_tloss_curve[-1], 6)
                                if _ts_vloss_curve and _ts_tloss_curve else None,
            "early_stop_metric": model_args.early_stop_metric,
            "completed_at":     datetime.now(timezone.utc).isoformat(),
        }
        try:
            _safe_save_json(_ts_summary, _ts_path)
            print(f"[TrainSummary] Written -> {_ts_path}")
        except Exception as _tse:
            print(f"[TrainSummary] Write failed (non-fatal): {_tse}")

        # Pretrain Ablation & Report
        _pt_report_path = model_artifact_dir / "pretrain_report.json"
        _pt_folds = cv_hist if model_args.walk_forward_cv else [{"fold": 0, "best_metric": best_val, "history": history}]
        _pt_summary = _fold_history_summary(_pt_folds, model_args.early_stop_metric)
        _pt_report = _read_json_dict(_pt_report_path)
        _pt_report.update({
            "model_name": model_name,
            "pretrain_enabled": bool(getattr(model_args, "pretrain", False)),
            "supervised_training_summary": _pt_summary,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        try:
            _safe_save_json(_pt_report, _pt_report_path)
        except Exception:
            pass

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
                "completed_at": datetime.now(timezone.utc).isoformat()
            }
            try:
                _safe_save_json(_abl_summary, _abl_path)
                _update_pretrain_report(model_args, {
                    "downstream_metric_delta_vs_no_pretrain": _deltas,
                    "ablation_verdict": _verdict,
                    "ablation_report_path": str(_abl_path),
                })
                print(f"[Ablation] Comparison written -> {_abl_path}")
            except Exception as _ae:
                print(f"[Ablation] Write failed: {_ae}")

        if alerter:
            _best_f = 0
            _best_v = 0.0
            if model_args.walk_forward_cv and 'cv_hist' in locals():
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
                    model=model_name,
                    fold=_best_f,
                    metric=model_args.early_stop_metric,
                    score=float(_best_v)
                )
            except Exception as e:
                print(f"[Discord] Failed to send training_completed: {e}")

        # ΓöÇΓöÇ Hard-Example Mining ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        # Collect samples where the model was confidently wrong during the last
        # validation pass, so they can be lightly oversampled on the next run.
        if _HardMiner is not None:
            try:
                _hem = _HardMiner(run_name=run_name, model_name=model_name)
                # Reconstruct final val predictions for the single-split case.
                # Walk-forward: use the last fold's history as a proxy (indices
                # are not easily recoverable here without a re-pass, so we skip).
                _do_mine = False
                _mine_preds = _mine_labels = _mine_rewards = _mine_idx = None

                if not model_args.walk_forward_cv and isinstance(history, dict):
                    # Try loading cached val predictions if supervised_train wrote them
                    _val_pred_path = model_artifact_dir / f"{model_name}_val_preds.npz"
                    if _val_pred_path.exists():
                        try:
                            _vdata = np.load(_val_pred_path)
                            _mine_preds   = _vdata["predictions"]
                            _mine_labels  = _vdata["labels"]
                            _mine_idx     = _vdata.get("indices", np.arange(len(_mine_preds)))
                            _mine_rewards = _vdata.get("rewards", None)
                            _do_mine = True
                        except Exception as _lv:
                            print(f"[HardMiner] Could not load val preds: {_lv}")

                if _do_mine and _mine_preds is not None:
                    _hem.collect(
                        val_indices = _mine_idx,
                        predictions = _mine_preds,
                        labels      = _mine_labels,
                        rewards     = _mine_rewards,
                    )
                    _hem.save()
                else:
                    print(f"[HardMiner] {model_name}: no val preds available for mining "
                          f"(walk-forward={model_args.walk_forward_cv}); skipping.")
            except Exception as _hm_e:
                print(f"[HardMiner] Mining failed (non-fatal): {_hm_e}")


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
                        model_name, cache_path, n_samples, n_features,
                        model_args, device, fold_sharpes=_fold_metrics,
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
                        PROD_CHECKPOINT as _prod, PREV_CHECKPOINT as _prev,
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
                        _atomic_copy(_prod, _prev)        # back up current prod -> prev
                    if _onnx_final.exists():

                        _atomic_copy(_onnx_final, _onnx_prev)

                    if _schema_final.exists():

                        _atomic_copy(_schema_final, _schema_prev)


                    _atomic_copy(ckpt_best, _prod)        # challenger -> prod (atomic)

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
                            f.write(f"reload {datetime.now(timezone.utc).isoformat()}\n")
                        os.replace(_tmp_flag, _reload_flag)
                        print(f"[Deploy] Reload signalled -> {_reload_flag}")
                    except Exception as _re:
                        print(f"[Deploy] could not write reload flag: {_re}")

                    deploy_result["status"] = "success"
                    _onnx_final = _onnx_final


                    if alerter:
                        try:
                            alerter.send_promotion_gate_passed(
                                model=model_name,
                                sharpe=float(gate_result.get("details", {}).get("sharpe", 0.0))
                            )
                            alerter.send_production_deploy_completed(
                                model=model_name,
                                onnx_path=str(_prod.with_suffix('.onnx').name),
                                schema_path=str(_prod.with_suffix('.schema.json').name)
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
                "model_name":            model_name,
                "run_name":              run_name,
                "checkpoint_dir":         str(_dep_dir.resolve()),
                "run_checkpoint_dir":     str(Path(args.checkpoint_dir).resolve()),
                "gate_promoted":         gate_result.get("promoted", False),
                "source_checkpoint":     str(ckpt_best) if ckpt_best.exists() else None,
                "production_checkpoint": str(_prod) if _prod is not None and deploy_result.get("status") == "success" else None,
                "previous_checkpoint":   str(_prev) if _prev is not None else None,
                "onnx_status":           "exported" if _onnx_final is not None and Path(_onnx_final).exists() else "skipped",
                "onnx_path":             str(_onnx_final) if _onnx_final is not None and Path(_onnx_final).exists() else None,
                "schema_path":           str(_schema_final) if _schema_final is not None and Path(_schema_final).exists() else None,

                "onnx_schema_verification": deploy_result.get("verification"),

                "reload_flag_status":    "written" if _reload_flag is not None and Path(_reload_flag).exists() else "skipped",
                "deploy_status":         deploy_result.get("status", "skipped"),
                "deploy_error":          deploy_result.get("error"),
                "failed_step":           deploy_result.get("failed_step"),
                "deployed_at":           datetime.now(timezone.utc).isoformat(),
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
            "end_time": datetime.now(timezone.utc).isoformat(),
            "best_epoch": int(best_epoch) if 'best_epoch' in locals() and best_epoch is not None else None,
            "best_metric": float(best_val) if 'best_val' in locals() and best_val is not None else None,
            "checkpoint_paths": [str(ckpt_best)],
            "promotion_result": gate_result,
            "deploy_result": deploy_result if 'deploy_result' in locals() else {"status": "skipped", "error": None},
        }
        if WANDB and wandb_run and "details" in gate_result:
            _deploy_logs = {}
            for k in ["profit_factor", "sharpe", "calmar", "max_drawdown"]:
                if k in gate_result["details"]:
                    _deploy_logs[f"deploy/{k}"] = gate_result["details"][k]
            if _deploy_logs:
                _safe_wandb_log(wandb_run, _deploy_logs)
        
        manifest.update({
            "config_path": getattr(args, "config", "config/run.yaml"),
            "warnings": [],
            "errors": []
        })
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
                _tm_vloss  = None
                _hist_for_mem = _history_for_tune if isinstance(_history_for_tune, dict) else {}

                _mem_best_epoch = int(best_epoch) if 'best_epoch' in locals() and best_epoch is not None else None

                def _mem_hist_at(key: str, default=None):

                    values = _hist_for_mem.get(key) or []

                    if _mem_best_epoch is not None and 0 <= _mem_best_epoch < len(values):

                        return values[_mem_best_epoch]

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

                    float(v) for v in (_fold_metrics or [])

                    if v is not None and np.isfinite(float(v))

                ]

                if model_args.early_stop_metric == "sharpe":

                    if _mem_metric_values:

                        _tm_sharpe = max(_mem_metric_values)

                    elif best_val is not None:

                        _tm_sharpe = float(best_val)

                elif best_val is not None:
                    _tm_vloss = min(_mem_metric_values) if _mem_metric_values else float(best_val)

                _train_memory.update({
                    "model_name":   model_name,
                    "run_name":     run_name,
                    "phase":        "main",
                    "best_sharpe":  _tm_sharpe,
                    "best_val_loss": _tm_vloss,
                    "best_epoch":   _mem_best_epoch,

                    "total_epochs": len(_hist_for_mem.get("train_loss", [])) or int(getattr(model_args, "epochs", 0)),
                    "history":      _hist_for_mem,
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

                    "gate_result":  gate_result if 'gate_result' in locals() else {},
                    "args_snapshot": {
                        "lr":       float(getattr(model_args, "lr", 5e-5)),
                        "dropout":  float(getattr(model_args, "dropout", 0.25)),
                        "patience": int(getattr(model_args, "patience", 6)),
                        "epochs":   int(getattr(model_args, "epochs", 24)),
                    },
                })
                _train_memory.save()
            except Exception as _tm_err:
                print(f"[TrainingMemory] Update failed (non-fatal): {_tm_err}")


    # -- XGBoost baseline training ---------------------------------------------
    # Runs when xgboost.enabled: true in run.yaml (or --xgb-enabled CLI).
    # Shells out to training/train_xgboost.py with params from the YAML config.
    if getattr(args, "xgb_enabled", False):
        print(f"\n{'='*62}")
        print("  XGBoost Baseline Training")
        print(f"{'='*62}")
        _xgb_cmd = [
            sys.executable, str(Path(__file__).parent / "train_xgboost.py"),
            "--config", str(getattr(args, "config", None) or "config/run.yaml"),
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
            _xgb_result = _sp.run(_xgb_cmd, cwd=str(Path(__file__).parent.parent), env=_xgb_env)
            if _xgb_result.returncode == 0:
                print("[XGBoost] Baseline training completed successfully.")
                models_to_train.append("xgboost")
            else:
                print(f"[XGBoost] Training failed with exit code {_xgb_result.returncode}")
        except Exception as _xgb_err:
            print(f"[XGBoost] Training failed: {_xgb_err}")

    # C: Diversity fine-tuning ΓÇö push same-role models apart after individual training.
    # Only runs when >=2 models were trained in this session (--all-models).
    if args.all_models and len(models_to_train) >= 2:
        try:
            from config.settings import CURRICULUM as _CURR_DIV
            _div_cfg = _CURR_DIV  # reuse config namespace for div settings
        except ImportError:
            _div_cfg = {}
        _div_w    = float(getattr(args, "div_weight",     0.10))
        _same_r   = float(getattr(args, "same_role_mult", 2.0))
        # With per-model subfolders, pass the base checkpoint dir so
        # run_diversity_finetune can find each model at <base>/<model>/<model>_best.pt
        _base_ckpt = Path(args.checkpoint_dir)
        run_diversity_finetune(
            checkpoint_dir = str(_base_ckpt),
            model_names    = models_to_train,
            cache_path     = cache_path,
            n_features     = n_features,
            args           = args,
            device         = device,
            epochs         = 3,
            lr             = 1e-5,
            div_weight     = _div_w,
            same_role_mult = _same_r,
        )

    # Ensemble meta-learner training (with diversity penalty)
    if getattr(args, "train_ensemble", False):
        run_ensemble_meta(cache_path, n_features, args, device)

        print("[Deploy] Running Promotion Gate on Ensemble...")
        ensemble_args = argparse.Namespace(**vars(args))
        ensemble_args.model = "ensemble"
        try:
            ens_gate_result = _evaluate_forward_gate(
                "ensemble", cache_path, n_samples, n_features, ensemble_args, device,
                fold_sharpes=None
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
                        import json
                        prod_sharpe = float(json.loads(dep_json.read_text()).get("gate_result", {}).get("details", {}).get("sharpe", -999.0))
                    except Exception as e:
                        print(f"[Deploy] Corrupted deployment.json: {e}")
                        raise RuntimeError(f"Corrupted deployment.json prevents safe promotion: {e}")
                
                ens_sharpe = float(ens_gate_result.get("details", {}).get("sharpe", -999.0))
                
                if ens_sharpe > prod_sharpe:
                    print(f"[Deploy] Ensemble Sharpe {ens_sharpe:.3f} > Base {prod_sharpe:.3f}. Overwriting production!")
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
                    print(f"[Deploy] Ensemble Sharpe {ens_sharpe:.3f} did not beat Base {prod_sharpe:.3f}. Skipping deploy.")
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
                run_rl(cache_path, n_features, _rl_args, device,
                       n_samples=n_samples, run=wandb_run)
        else:
            run_rl(cache_path, n_features, args, device, n_samples=n_samples, run=wandb_run)

    if wandb_run: wandb_run.finish()

    print(f"\n{'='*62}")
    print(f"  Training complete!")
    _base_ckpt_dir = Path(args.checkpoint_dir)
    print(f"  Checkpoints: {_base_ckpt_dir}/")
    for _mn in models_to_train:
        _model_dir = _base_ckpt_dir / _mn
        if _model_dir.exists():
            print(f"    {_mn:12s} -> {_model_dir}/")

    print(f"  Dataset cache: {cache_path}  (reused on --resume)")
    print(f"{'='*62}")

    try:
        import subprocess
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        git_hash = "unknown"

    run_manifest = {
        "run_start_time": run_start_time,
        "run_end_time": datetime.now(timezone.utc).isoformat(),
        "run_name": getattr(args, "run_name", "pipeline_run"),
        "models_trained": models_to_train,
        "git_hash": git_hash,
        "args": {k: str(v) for k, v in vars(args).items()}
    }
    if _base_ckpt_dir.exists():
        _safe_save_json(run_manifest, _base_ckpt_dir / "run_manifest.json")
if __name__ == "__main__":
    main()
