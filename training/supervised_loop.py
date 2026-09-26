"""Supervised training loop extracted from ``training.train_gpu``.

See ``docs/CONTINUE.md``."""

from __future__ import annotations

import gc
import math
import copy
import os
import time
from contextlib import nullcontext
from training.ema import ExponentialMovingAverage
from training.honest_eval import period_balance_weights
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import hashlib
import json as _json_hash

from config.settings import (
    LABELING,
    PATHS,
)


def _resolve_schema_hash(args) -> str:
    h = getattr(args, "feature_schema_hash", None)
    if h and h != "unknown":
        return str(h)
    # Fallback: compute from feature names if available, else n_features
    try:
        names = getattr(args, "_feat_names", None) or getattr(args, "feature_names", None) or []
        if names and len(names) > 0:
            return hashlib.md5(_json_hash.dumps(list(map(str, names)), sort_keys=True).encode()).hexdigest()[:12]
    except Exception:
        pass
    try:
        n = getattr(args, "_n_features", None) or getattr(args, "n_features", None)
        if n:
            return hashlib.md5(str(int(n)).encode()).hexdigest()[:12]
    except Exception:
        pass
    return "unknown"
from models.architectures import (
    AsymmetricDirectionalLoss,
    DiversityLoss,
    HuberLoss,
    MultiTaskLoss,
    OverconfidencePenalty,
    TemperatureScaler,
)
from monitoring.rich_display import (
    _RichDisplay,
)
from monitoring.sidecar import (
    Sidecar,
)
from training.cache_integrity import (
    _on_disk_sequence_count,
    _promotion_holdout_n,
)
from training.core import (
    _GPU_CFG,
    _TRAIN_LOGGER,
    _TRAIN_LOGGER_AVAILABLE,
    WANDB,
    _crop_to_seq_len,
    _log_error,
    _log_info,
    _log_nan,
    _log_oom,
    _log_warn,
    _safe_wandb_log,
    _safe_wandb_summary_update,
)
from training.cv_splits import (
    _embargo_bars,
    _embargo_split,
    _purge_bars,
    _three_way_split,
    _validation_method,
)
from training.dataset_builder import _safe_save_json
from training.direction_control import (
    _balanced_direction_indices,
    _class_prior_tensor,
    _class_weights_tensor,
    _direction_class_index,
    _direction_preflight,
    _direction_probe,
    _direction_recall_from_confusion,
    _gradients_are_finite,
    _init_multitask_direction_bias,
    _is_uninitialized_parameter,
    _load_diff_array,
    _load_feature_schema,
    _recover_nonfinite_training_state,
    _write_class_balance_failure,
)
from training.ewc import (
    ElasticWeightConsolidation,
    apply_ewc_loss,
)
from training.feature_ablation import (
    _build_feature_ablation_mask,
    _feature_ablation_config,
)
from training.gpu_cli import (
    _apply_model_profile,
    _model_build_args,
    _slug_part,
)
from training.gpu_datasets import (
    ZarrStreamDataset,
    wrap_loader_prefetch,
)
from training.gpu_device import (
    _thermal_check,
    maybe_torch_compile,
)
from training.gpu_losses import (
    DirectionalHuberLoss,
    SharpeProxyLoss,
    _match_target_shape,
)
from training.memory_management import (
    PrioritizedDataLoader,
)
from training.model_factory import (
    _core_model,
    _strict_load_report,
    build_model,
)
from training.post_train import (
    _safe_save,
)
from training.pretrain_runner import (
    _update_pretrain_report,
)
from training.synaptic_intelligence import (
    SynapticIntelligence,
    apply_si_loss,
)

try:
    from monitoring.train_logger import TrainingLogger as _TrainingLogger
except ImportError:
    _TrainingLogger = None

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter

    TENSORBOARD = True
except ImportError:
    TENSORBOARD = False

try:
    from tqdm.auto import tqdm as _pbar  # type: ignore
except ImportError:

    def _pbar(it=None, **kw):
        return it

# -----------------------------------------------------------------------------
# R5 SPLIT: cohesive submodules (verbatim moves); re-exported here so every
# existing ``from training.supervised_loop import X`` path keeps working.
# Single source of truth for each symbol lives in its submodule.
# -----------------------------------------------------------------------------
from training.diversity_finetune import run_diversity_finetune  # noqa: E402
from training.loop_batches import (  # noqa: E402
    _SANITIZE_STATS,
    _prepare_train_batch,
    _sanitize_batch_tensors,
    _unpack_batch,
    reset_sanitize_stats,
    sanitize_stats,
)
from training.loop_epochs import (  # noqa: E402
    _non_overlapping_sharpe,
    _train_batch,
    _validation_class_diag,
    train_epoch,
    validate_epoch,
)
import training.loop_losses as _loop_losses  # noqa: E402
from training.loop_losses import (  # noqa: E402
    _OVERCONF_PENALTY,
    _CurriculumProvider,
    _CurriculumProviderConfig,
    _apply_curriculum_weights,
    _apply_kd_loss,
    _apply_online_miner,
    _build_train_loss,
    _compute_loss,
    build_criterion,
)
from training.loop_optim import (  # noqa: E402
    _centralize_gradients,
    _maybe_warn_grad_norm,
    _optimizer_step,
)


from config.settings import (
    CURRICULUM as SETTINGS_CURRICULUM,
)
from config.settings import (
    PATHS,  # noqa: F811
    PRETRAIN,
    TRAINING,
)
from models.architectures import (
    MODEL_ROLES,
    AsymmetricDirectionalLoss,  # noqa: F811
    DiversityLoss,  # noqa: F811
    HuberLoss,  # noqa: F811
    MultiTaskLoss,  # noqa: F811
    OverconfidencePenalty,  # noqa: F811
    TemperatureScaler,  # noqa: F811
)
from training.ewc import ElasticWeightConsolidation, apply_ewc_loss  # noqa: F811
from training.gpu_datasets import (
    ZarrStreamDataset,  # noqa: F811
)
from training.gpu_device import build_adamw, maybe_torch_compile  # noqa: F811
from training.gpu_losses import (
    DirectionalHuberLoss,  # noqa: F811
    SharpeProxyLoss,  # noqa: F811
    _match_target_shape,  # noqa: F811
)


# -----------------------------------------------------------------------------
# A: FEATURE STABILITY + TRAIN/VALIDATE + SUPERVISED_TRAIN
# -----------------------------------------------------------------------------

# TRAINING LOOP
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# A: FEATURE STABILITY MONITORING
#
# Most training failures don't show up as loss spikes -- they show up as
# features whose distributions silently shift, causing gradients to chase
# moving targets.  The model "learns" the shift artefact rather than the
# underlying signal, producing good training metrics but poor live performance.
#
# FeatureStabilityMonitor tracks per-feature mean and std using an exponential
# moving average (EMA).  Each epoch:
#   1. Sample a batch from the training data
#   2. Compute per-feature mean / std across (B, T) positions
#   3. Compute shift score = |Deltamean| / (ema_std + eps) -- standard deviations shifted
#   4. Compute var  score  = |Deltastd|  / (ema_std + eps) -- variance change magnitude
#   5. Mark features as noisy (score > soft_threshold) or frozen (score > hard_threshold
#      for freeze_after consecutive epochs)
#   6. Output a float32 mask (1.0=stable, damping_factor=noisy, 0.0=frozen)
#   7. Apply mask to xb in train_epoch: xb = xb * mask -- unstable dims zeroed out
#
# Effect: gradients for frozen features are effectively zeroed (input = 0).
# Noisy features receive a reduced signal (input x damping_factor).
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
        n_features: int,
        ema_alpha: float = 0.90,
        soft_threshold: float = 2.0,
        hard_threshold: float = 4.0,
        freeze_after: int = 3,
        damping_factor: float = 0.50,
        warmup_epochs: int = 3,
        min_active_pct: float = 0.50,
    ):
        self.n = n_features
        self.alpha = ema_alpha
        self.soft_t = soft_threshold
        self.hard_t = hard_threshold
        self.freeze_af = freeze_after
        self.damp = damping_factor
        self.warmup = warmup_epochs
        self.min_active = min_active_pct

        self._ema_mean = np.zeros(n_features, dtype=np.float64)
        self._ema_std = np.ones(n_features, dtype=np.float64)
        self._initialized = False

        # Per-feature counters
        self._soft_streak = np.zeros(n_features, dtype=np.int32)  # epochs above soft_t
        self._frozen = np.zeros(n_features, dtype=bool)  # permanently frozen

        # Per-epoch stats (for logging)
        self._last_shift_score = np.zeros(n_features, dtype=np.float64)
        self._epoch = 0

    def update(self, xb: np.ndarray) -> None:
        """
        Update EMA stats with a batch sample.

        Args:
            xb: float32 array of shape (B, seq_len, n_features) or (B, n_features).
        """
        if xb.ndim == 3:
            flat = xb.reshape(-1, xb.shape[-1])  # (B*T, F)
        else:
            flat = xb

        batch_mean = flat.mean(axis=0).astype(np.float64)  # (F,)
        batch_std = flat.std(axis=0).astype(np.float64)  # (F,)
        batch_std = np.maximum(batch_std, 1e-8)

        if not self._initialized:
            self._ema_mean = batch_mean.copy()
            self._ema_std = batch_std.copy()
            self._initialized = True
            return

        # Shift scores before EMA update (compare new batch vs current EMA)
        mean_shift = np.abs(batch_mean - self._ema_mean) / (self._ema_std + 1e-8)
        std_shift = np.abs(batch_std - self._ema_std) / (self._ema_std + 1e-8)
        shift_score = np.maximum(mean_shift, std_shift)
        self._last_shift_score = shift_score

        # Update EMA
        self._ema_mean = self.alpha * self._ema_mean + (1 - self.alpha) * batch_mean
        self._ema_std = self.alpha * self._ema_std + (1 - self.alpha) * batch_std

        self._epoch += 1
        if self._epoch <= self.warmup:
            return  # don't penalise during warm-up

        # Update instability streaks and frozen flags
        above_soft = shift_score > self.soft_t
        above_hard = shift_score > self.hard_t
        self._soft_streak[above_soft] += 1
        self._soft_streak[~above_soft] = 0

        # Freeze: hard threshold OR soft streak exceeded
        newly_frozen = above_hard | (self._soft_streak >= self.freeze_af)

        # Enforce min_active_pct -- never freeze too many features
        n_frozen = int(newly_frozen.sum())
        max_freeze = int(self.n * (1.0 - self.min_active))
        if n_frozen > max_freeze:
            # Keep only the worst max_freeze features frozen
            top_frozen = np.argsort(shift_score)[::-1][:max_freeze]
            mask_limit = np.zeros(self.n, dtype=bool)
            mask_limit[top_frozen] = True
            newly_frozen = newly_frozen & mask_limit

        self._frozen = newly_frozen

    def get_mask(self, device=None) -> torch.Tensor:
        """
        Returns a float32 tensor of shape (n_features,):
          1.0 = stable      -> full signal
          damping_factor    -> noisy (above soft threshold but not frozen)
          0.0               -> frozen (consistently unstable)
        """
        mask = np.ones(self.n, dtype=np.float32)
        noisy = (self._last_shift_score > self.soft_t) & (~self._frozen)
        mask[noisy] = self.damp
        mask[self._frozen] = 0.0
        t = torch.from_numpy(mask)
        return t.to(device) if device is not None else t

    def report(self) -> dict:
        """Return a summary dict for logging."""
        n_frozen = int(self._frozen.sum())
        n_noisy = int(((self._last_shift_score > self.soft_t) & ~self._frozen).sum())
        top5_idx = np.argsort(self._last_shift_score)[::-1][:5].tolist()
        return {
            "feat_frozen": n_frozen,
            "feat_noisy": n_noisy,
            "feat_active": self.n - n_frozen,
            "feat_max_shift": float(self._last_shift_score.max()),
            "feat_mean_shift": float(self._last_shift_score.mean()),
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
            "ema_mean": self._ema_mean.tolist(),
            "ema_std": self._ema_std.tolist(),
            "soft_streak": self._soft_streak.tolist(),
            "frozen": self._frozen.tolist(),
            "last_shift": self._last_shift_score.tolist(),
            "initialized": self._initialized,
            "epoch": self._epoch,
        }

    def load_state(self, state: dict) -> None:
        """Restore state from a checkpoint dict (backward-compatible)."""
        self._ema_mean = np.array(state["ema_mean"], dtype=np.float64)
        self._ema_std = np.array(state["ema_std"], dtype=np.float64)
        self._soft_streak = np.array(state["soft_streak"], dtype=np.int32)
        self._frozen = np.array(state["frozen"], dtype=bool)
        self._last_shift_score = np.array(state.get("last_shift", [0.0] * self.n), dtype=np.float64)
        self._initialized = bool(state["initialized"])
        self._epoch = int(state["epoch"])


# -----------------------------------------------------------------------------
# C: DIVERSITY FINE-TUNING
# After all models are individually trained, run a short joint pass that
# penalises correlated predictions across models with the same role.
#
# Why post-training (not during individual training):
#   During individual training each model only has its own output -- there are
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
# Same-role pairs (tft+transformer, haelt+expert) receive a 2x diversity
# penalty to ensure they specialise rather than duplicate.
# -----------------------------------------------------------------------------


def _load_pretrained_encoder(model: nn.Module, args, device) -> bool:
    """A-C1: load the contrastive-pretrained encoder into the supervised model's
    backbone so pretraining is not wasted.

    The contrastive trainers checkpoint the backbone with the prediction head
    stripped (Identity), so the supervised head legitimately shows up as
    "missing"; everything else (the backbone) must transfer. We assert a large
    fraction loads, failing loudly if the transfer is effectively a no-op.
    """
    if getattr(args, "disable_pretrain_load", False):
        return False

    method = str(getattr(args, "pretrain_method", PRETRAIN.get("method", "byol"))).lower()
    use_regime = getattr(args, "pretrain_regime", False) and method == "tscl"
    framework = str(getattr(args, "pretrain_framework", "custom") or "custom").lower()
    ckpt_dir = Path(args.checkpoint_dir)
    candidates = []
    if use_regime:
        candidates.append(ckpt_dir / "contrastive_encoder_regime.pt")
    candidates += [ckpt_dir / "contrastive_encoder.pt", ckpt_dir / "contrastive_encoder_regime.pt"]
    # Adapter-based frameworks (tsai, etc.) save under a different name.
    if framework != "custom":
        candidates.append(ckpt_dir / f"pretrain_{framework}_encoder.pt")
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        print(
            f"[PretrainΓåÆSup] No contrastive encoder checkpoint in {ckpt_dir} -- "
            "skipping transfer (was pretraining run?)."
        )
        return False
    # Only load an encoder its own report accepted: completed, gate passed, not
    # shown to hurt in an ablation, and the same file that was trained (hash).
    _rep_path = ckpt_dir / "pretrain_report.json"
    try:
        import json as _json_pt

        _rep = _json_pt.loads(_rep_path.read_text(encoding="utf-8")) if _rep_path.is_file() else {}
    except Exception:
        _rep = {}
    _why_not = None
    if not _rep:
        _why_not = "no pretrain_report.json"
    elif _rep.get("status") != "completed":
        _why_not = f"status={_rep.get('status')}"
    elif _rep.get("quality_gate_result") != "passed":
        _why_not = f"quality_gate_result={_rep.get('quality_gate_result')}"
    elif _rep.get("ablation_verdict") == "pretrain_hurt":
        _why_not = "ablation verdict pretrain_hurt"
    elif not _rep.get("scaled_inputs"):
        _why_not = "encoder trained on unscaled inputs (pre-2026-09-25)"
    else:
        import hashlib as _hl

        _h = _hl.sha256(Path(ckpt_path).read_bytes()).hexdigest()
        if _rep.get("checkpoint_sha256") and _h != _rep.get("checkpoint_sha256"):
            _why_not = "checkpoint hash differs from the one the report describes"
    if _why_not:
        print(f"[Pretrain->Sup] Not loading {ckpt_path.name}: {_why_not}")
        return False
    encoder = model.backbone if hasattr(model, "backbone") else model
    target = _core_model(encoder)
    # Load on CPU: load_state_dict copies onto the model's device anyway, and a
    # CUDA-saved encoder must still load when the run has no visible GPU.
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    # The head is expected to be missing ΓåÆ allow up to ~40% missing for wide heads.
    load_report = _strict_load_report(target, state, f"PretrainΓåÆ{args.model}", min_frac_loaded=0.6)
    try:
        _update_pretrain_report(
            args,
            {
                "loaded_into_supervised_training": True,
                "supervised_transfer": {
                    "checkpoint_path": str(ckpt_path),
                    "frac_loaded": float(load_report.get("frac_loaded", 0.0)),
                    "missing_count": len(load_report.get("missing", [])),
                    "unexpected_count": len(load_report.get("unexpected", [])),
                    "shape_mismatch_count": len(load_report.get("shape_mismatch", [])),
                },
            },
        )
    except Exception:
        pass
    print(f"[PretrainSup] Loaded contrastive encoder from {ckpt_path.name} into backbone.")
    return True


def _warm_start_from_checkpoint(model: nn.Module, args, device, model_name: str) -> bool:
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
        base.parent / "production_best.pt",  # checkpoints/<run>/production_best.pt
        base / "production_best.pt",
        base / f"{model_name}_best.pt",
        base / model_name / f"{model_name}_best.pt",
    ]
    ckpt_path = next((p for p in candidates if p and p.exists()), None)
    if ckpt_path is None:
        print(f"[WarmStart] No prior checkpoint found for {model_name} (looked in {base}); training from scratch.")
        return False
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ck.get("model_state", ck.get("model_state_dict", ck.get("state_dict", ck))) if isinstance(ck, dict) else ck
    core = _core_model(model)
    _strict_load_report(core, state, f"WarmStart:{model_name}", min_frac_loaded=0.6)
    print(f"[WarmStart] Continuing training from {ckpt_path}")
    return True


def _fold_provenance(cache_path, train_idx, val_idx, args) -> dict:
    """Train/val index ranges, their timestamps, cache and dataset version, seed."""
    out: dict = {"cache_path": str(cache_path), "seed": getattr(args, "seed", None)}
    try:
        from training.cache_integrity import DATASET_BUILD_VERSION

        out["dataset_build_version"] = DATASET_BUILD_VERSION
    except Exception:
        pass
    t_ns = None
    try:
        import zarr as _zr

        _root = _zr.open(str(cache_path), mode="r")
        if "t_ns" in _root:
            t_ns = _root["t_ns"]
    except Exception:
        t_ns = None
    for name, idx in (("train", train_idx), ("val", val_idx)):
        if idx is None or len(idx) == 0:
            continue
        lo, hi = int(np.min(idx)), int(np.max(idx))
        out[f"{name}_range"] = [lo, hi + 1]
        out[f"n_{name}"] = int(len(idx))
        if t_ns is not None:
            try:
                out[f"{name}_start_ns"], out[f"{name}_end_ns"] = int(t_ns[lo]), int(t_ns[hi])
            except Exception:
                pass
    return out


def supervised_train(
    model_name: str,
    cache_path: str,
    n_samples: int,
    n_features: int,
    args,
    device: torch.device,
    n_gpus: int,
    run: Any = None,
    train_idx: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
    fold_id: int | None = None,
    amp_dtype: torch.dtype = torch.float32,
):
    reset_sanitize_stats()
    
    # ── IRT Orchestration ──
    if getattr(args, "enable_irt", False):
        print("[IRT] Infinite Robust Training ENABLED. Forcing SI, Miner, Adversarial, and Curriculum ON.")
        args.enable_si = True
        args.enable_adversarial = True
        args.online_hard_mining = True
        args.curriculum_manager = True
        args.curriculum_miner_feedback = True


    # ── Lightning training path (opt-in via --training-framework lightning) ──
    _train_framework = str(getattr(args, "training_framework", "custom") or "custom").lower()
    if _train_framework == "lightning":
        try:
            from training.lightning_trainer import is_lightning_available, run_lightning_training

            if is_lightning_available():
                print(f"\n[Training] Using PyTorch Lightning framework for {model_name.upper()}")
                history, metrics = run_lightning_training(
                    model_name=model_name,
                    cache_path=cache_path,
                    n_samples=n_samples,
                    n_features=n_features,
                    args=args,
                    device=device,
                    n_gpus=n_gpus,
                    run=run,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    fold_id=fold_id,
                    amp_dtype=amp_dtype,
                )
                return history, metrics.get("best_sharpe", 0.0)
            else:
                print("[Lightning] Not available, falling back to custom training loop")
        except Exception as _lt_e:
            print(f"[Lightning] Training failed ({_lt_e}), falling back to custom loop")
    elif _train_framework == "composer":
        print("[Composer] Mosaic Composer framework is not fully implemented/installed, falling back to custom loop.")
    elif _train_framework != "custom":
        print(f"[Training] Unknown framework '{_train_framework}', falling back to custom loop.")

    global _TRAIN_LOGGER
    _artifact_run_name = str(
        getattr(args, "run_name_slug", "") or _slug_part(getattr(args, "run_name", "pipeline-run"), max_len=140)
    )
    sidecar = None  # always defined; reused logger path skips Sidecar creation

    if _TRAIN_LOGGER_AVAILABLE:
        if _TRAIN_LOGGER is None:
            _sidecar_cfg = getattr(args, "sidecar", None) or {}
            if _sidecar_cfg.get("enabled", False):
                try:
                    sidecar = Sidecar(
                        log_dir=PATHS.get("logs", "logs"),
                        run_name=_artifact_run_name,
                        model_name=model_name,
                        enabled=True,
                        mode=str(_sidecar_cfg.get("mode", "process")),
                        max_queue_size=int(_sidecar_cfg.get("max_queue_size", 10000)),
                        flush_interval_s=float(_sidecar_cfg.get("flush_interval_s", 2.0)),
                        retention_days=int(_sidecar_cfg.get("retention_days", 30)),
                        enable_discord=bool(_sidecar_cfg.get("enable_discord", False)),
                    )
                    sidecar.start()
                except Exception as e:
                    print(f"[Sidecar] Failed to start: {e}")
                    sidecar = None

            import training.core
            if _TrainingLogger is not None:
                from typing import Any, cast
                logger_cls = cast(Any, _TrainingLogger)
                _TRAIN_LOGGER = logger_cls(
                    log_dir=PATHS.get("logs", "logs"),
                    run_name=f"{_artifact_run_name}_{datetime.now().strftime('%m%d_%H%M')}",
                    model_name=model_name,
                    sidecar=sidecar,
                )
                training.core._TRAIN_LOGGER = _TRAIN_LOGGER
                _TRAIN_LOGGER.setup()
        else:
            if _TRAIN_LOGGER is not None:
                _TRAIN_LOGGER.model_name = model_name
                import training.core
                training.core._TRAIN_LOGGER = _TRAIN_LOGGER
            # Prefer the logger's existing sidecar on subsequent folds
            sidecar = getattr(_TRAIN_LOGGER, "sidecar", None)

    _log_dir = PATHS.get("logs", "logs")
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
        print(f"[TensorBoard] Logging -> {_tb_dir}  (tensorboard --logdir {_tb_dir} --port 6006)")

    # -- Rich live display -----------------------------------------------------
    _rich_display = None
    try:
        from monitoring.rich_display import _RichDisplay
        RICH_DISPLAY = True
    except ImportError:
        RICH_DISPLAY = False

    if RICH_DISPLAY and not getattr(args, "no_rich", False):
        _rich_display = _RichDisplay(
            model_name=model_name,
            total_epochs=args.epochs,
            metric_name="val_loss",
            higher_is_better=False,
        )

    if getattr(args, "model_profile", True) and not getattr(args, "_profile_applied", False):
        args = _apply_model_profile(args, model_name, enabled=True)

    _t_start = time.time()
    classification = args.loss in ("multi_task", "asymmetric_directional")
    multitask = bool(getattr(args, "multitask", False))
    fold_suffix = f"_fold{fold_id}" if fold_id is not None else ""

    print(f"\n{'-' * 60}")
    print(
        f"  Training: {model_name.upper()} | {n_samples:,} samples | "
        f"batch={args.batch_size} | AMP={args.amp} | loss={args.loss}"
    )
    print(f"{'-' * 60}")

    if hasattr(args, "_training_features_report"):
        report = args._training_features_report
        print("\n  [Training-Features Report]")
        for k, v in report.items():
            print(f"    {k:20s}: {v['mode']!s:<6s} (source: {v['source']})")
        print()

        try:
            import json
            from pathlib import Path as _Path

            if getattr(args, "checkpoint_dir", None):
                rep_path = _Path(args.checkpoint_dir) / "training_features_report.json"
                with open(rep_path, "w") as f:
                    json.dump(report, f, indent=2)
            if run is not None:  # noqa: SIM102
                # Safely update run config if W&B is active
                if hasattr(run, "config"):
                    run.config.update({"features_report": report}, allow_val_change=True)
        except Exception as e:
            print(f"  [Warning] Failed to write training_features_report: {e}")

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
        # Tiny caches: shrink embargo/purge so a usable train/val split remains.
        if n_samples < max(200, _embargo + _purge + _holdout_n + 20):
            _embargo = min(_embargo, max(0, n_samples // 10))
            _purge = min(_purge, max(0, n_samples // 10))
            _holdout_n = min(_holdout_n, max(1, n_samples // 5))
            _split_n = max(0, n_samples - _holdout_n)
        _method = _validation_method(args)
        # SYS-002: three-way split to isolate auto-tune evaluation from val (early stopping)
        _tune_split = float(getattr(args, "tune_split", 0.0) or 0.0)
        if _tune_split > 0:
            train_idx, val_idx, tune_idx = _three_way_split(
                _split_n, args.val_split, _tune_split, _embargo, _purge, _method
            )
            print(
                f"[Split] Three-way split: embargo={_embargo} purge={_purge} method={_method} "
                f"tune_split={_tune_split} "
                f"{f'| holdout tail = {_holdout_n:,} bars' if _holdout_n else ''}"
            )
        else:
            train_idx, val_idx = _embargo_split(_split_n, args.val_split, _embargo, _purge, _method)
            print(
                f"[Split] Single split with embargo={_embargo} purge={_purge} method={_method} "
                f"{f'| holdout tail = {_holdout_n:,} bars' if _holdout_n else ''}"
            )
    print(
        f"[Split] Train: {len(train_idx):,} | Val: {len(val_idx):,}"
        f"{f' | Tune: {len(tune_idx):,}' if tune_idx is not None else ''}"
        f"{fold_suffix if fold_id is not None else ''}"
    )
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError(
            f"[Split] Empty train/val after holdout/embargo "
            f"(n_samples={n_samples}, train={len(train_idx)}, val={len(val_idx)}). "
            f"Increase --n-ticks / use real data, or reduce seq_len/embargo for quick runs."
        )

    # ── Fold isolation check ──────────────────────────────────────────────
    # Verifies val samples are strictly after train + embargo gap at the index
    # level. The cache stores sequences in chronological order so the global
    # row index *is* the temporal position; this guards against index-space
    # leakage that the embargo split logic might have missed (e.g. tiny caches
    # where embargo got shrunk, or walk-forward CV edge cases). Soft-fail by
    # default so quick runs are not blocked; raise when --strict-fold-isolation.
    if getattr(args, "enable_fold_isolation_check", True):
        try:
            from features.lookahead_guard import (
                LookaheadViolation,
                assert_fold_isolation,
            )

            train_ts = np.asarray(train_idx, dtype=np.int64)
            val_ts = np.asarray(val_idx, dtype=np.int64)
            _iso_emb = int(_embargo) if "_embargo" in locals() else 0
            assert_fold_isolation(train_ts, val_ts, embargo_bars=_iso_emb)
            print(
                f"[FoldIsolation] OK: val_min={int(val_ts.min())} > "
                f"train_max={int(train_ts.max())} (embargo={_iso_emb})."
            )
            if tune_idx is not None and len(tune_idx) > 0:
                tune_ts = np.asarray(tune_idx, dtype=np.int64)
                # tune_eval is the most-recent segment; it must be after val.
                if len(val_ts) > 0 and tune_ts.min() <= val_ts.max():
                    _violation = (
                        f"Fold isolation violated: tune_min_ts={int(tune_ts.min())} "
                        f"<= val_max_ts={int(val_ts.max())}. "
                        f"Auto-tune evaluation data overlaps with early-stopping val set."
                    )
                    if getattr(args, "strict_fold_isolation", False):
                        raise LookaheadViolation(_violation)
                    print(f"[FoldIsolation] WARN (tune/val overlap): {_violation}")
        except LookaheadViolation as _iso_exc:
            _strict = bool(getattr(args, "strict_fold_isolation", False))
            _ignore = bool(getattr(args, "ignore_preflight", False)) or bool(getattr(args, "quick_mode", False))
            if _strict and not _ignore:
                raise
            print(f"[FoldIsolation] WARN (continuing): {_iso_exc}")
        except Exception as _iso_err:
            print(f"[FoldIsolation] check skipped ({_iso_err})")

    # SYS-002: store tune_idx on args for post-training evaluation
    args._tune_eval_idx = tune_idx

    # ZarrStreamDataset reads each zarr chunk exactly once per epoch (sequential
    # block reads + in-block shuffle) instead of one random decompression per
    # sample.  Val indices are sorted so val reads are also sequential.
    use_direction_targets = bool(multitask or classification)

    # Load scaler for feature normalization (RobustScaler or StandardScaler)
    from training.dataset_builder import _load_scaler_npz
    _global_scaler = _load_scaler_npz(Path(cache_path))
    _scaler = None
    if _global_scaler is None:
        # No cache scaler: still fit a train-only RobustScaler rather than training unscaled.
        from training.dataset_builder import _make_scaler

        _global_scaler = _make_scaler()
        print("[Data] No cache scaler.npz; fitting a fresh train-only scaler")
    if _global_scaler is not None:
        try:
            from sklearn.base import clone
            import zarr as _zarr
            _scaler = clone(_global_scaler)
            _z = _zarr.open(str(cache_path), mode="r")
            if "X" in _z and train_idx is not None and len(train_idx) > 0:
                _x_arr = _z["X"]
                _max_sample = min(50000, len(train_idx))
                _subset = np.random.choice(train_idx, _max_sample, replace=False)
                _subset.sort()
                # Last timestep only: windows overlap by seq_len-1 bars, so these rows
                # cover the same bars as full windows at 1/seq_len of the memory
                # (full windows were 50k x 120 x 584 floats, ~14 GB).
                _x_data = np.asarray(_x_arr.get_orthogonal_selection((_subset, -1, slice(None))))
                if _x_data.ndim == 3:
                    _x_data = _x_data.reshape(-1, _x_data.shape[-1])
                _x_finite = _x_data[np.isfinite(_x_data).all(axis=1)]
                if len(_x_finite) > 0:
                    _scaler.fit(_x_finite)
                    print(f"[Data] Refitted {_scaler.__class__.__name__} on {_max_sample} train samples (no data leakage)")
                else:
                    _scaler = _global_scaler
            else:
                _scaler = _global_scaler
        except Exception as _se:
            # The cache-wide scaler was fit on validation/holdout rows too; using it
            # would leak evaluation statistics into training. Fail instead.
            raise RuntimeError(f"[Data] train-only scaler refit failed: {_se}") from _se
    if _scaler is not None:
        from training.dataset_builder import neutralize_price_level_columns
        from training.direction_control import _load_feature_schema as _lfs

        _neutral = neutralize_price_level_columns(_scaler, _lfs(cache_path, n_features))
        if _neutral:
            print(f"[Data] Neutralised {len(_neutral)} raw price-level columns in the scaler (e.g. {_neutral[:3]})")
    if use_direction_targets:
        try:
            _direction_preflight(cache_path, train_idx, val_idx, args)
        except RuntimeError as exc:
            # Tiny synthetic/quick caches often cannot satisfy class-prior floors.
            if getattr(args, "ignore_preflight", False) or getattr(args, "quick_mode", False):
                print(f"[DirectionPreflight] WARN (continuing): {exc}")
            else:
                raise

    _pair_targets = bool(getattr(args, "per_pair_heads", False))
    train_ds = ZarrStreamDataset(
        cache_path,
        train_idx,
        shuffle_chunks=True,
        multitask_targets=use_direction_targets,
        return_indices=True,
        scaler=_scaler,
        pair_targets=_pair_targets,
    )
    # Period balance: weight training rows so each calendar year contributes
    # equally (news coverage and label mix differ sharply by era).
    _period_wl = None
    if bool(getattr(args, "period_balance", False)):
        _period_wl = period_balance_weights(cache_path, train_idx, n_samples)
        if _period_wl is None:
            print("[PeriodBalance] WARN: cache has no t_ns row timestamps; rebuild to enable")
    # Honest validation context: real close/spread for net-of-cost PnL selection.
    _honest_ctx = None
    try:
        from training.honest_eval import fx_bars_per_year, load_price_arrays

        _hc_close, _hc_spread = load_price_arrays(cache_path)
        if _hc_close is not None:
            _honest_ctx = {
                "sample_idx": np.sort(val_idx),
                "close": _hc_close,
                "spread": _hc_spread,
                "horizon": int(getattr(args, "lookahead_bars", None) or LABELING.get("lookahead_bars", 30)),
                "bars_per_year": fx_bars_per_year(str(getattr(args, "bar_freq", None) or "5min")),
            }
            if _pair_targets:
                _cp, _sp = load_price_arrays(cache_path, pairs=True)
                if _cp is None:
                    raise RuntimeError("per_pair_heads needs close_pairs in the cache; rebuild it")
                _pnames = list(getattr(args, "pairs", None) or [])
                _honest_ctx.update(
                    close_pairs=_cp, spread_pairs=_sp, n_pairs=int(_cp.shape[1]),
                    pair_names=_pnames if len(_pnames) == _cp.shape[1] else None,
                )
            print(f"[Val][honest] enabled: {len(_hc_close):,} cached prices, spread={'yes' if _hc_spread is not None else 'no'}")
        else:
            print("[Val][honest] WARN: cache has no 'close' array; selection falls back to label-based cost_sharpe")
    except Exception as _hc_e:
        print(f"[Val][honest] WARN: price arrays unavailable ({_hc_e})")
    val_ds = ZarrStreamDataset(
        cache_path,
        np.sort(val_idx),
        shuffle_chunks=False,
        multitask_targets=use_direction_targets,
        scaler=_scaler,
        pair_targets=_pair_targets,
    )

    # Windows DataLoader workers use spawned processes plus shared file mappings.
    # Large zarr batches can exhaust the OS paging/shared-memory budget and fail
    # with WinError 1455, so keep loading in-process and use thread prefetch below.
    nw = 0 if os.name == "nt" else min(max(0, int(args.num_workers)), os.cpu_count() or 4)
    pf = int(args.prefetch_factor) if nw > 0 else None
    # Validation: keep fewer batches in flight than training. Pinning large batches
    # in worker threads can trip CUDA OOM on Windows (WDDM + driver) even when
    # train fits -- the error often surfaces as pin_memory(..., device=0).
    val_nw = max(2, nw // 2)
    val_pf = min(int(pf), 2) if pf is not None else None

    # B: On Windows, persistent_workers=True can sometimes cause I/O hangs
    # when combined with certain storage backends or external drives.
    # Default to False on Windows for stability.
    use_persistent = nw > 0 and os.name != "nt"
    if getattr(args, "persistent_workers", None) is not None:
        use_persistent = bool(args.persistent_workers) and nw > 0

    if getattr(args, "val_num_workers", None) is not None:
        val_nw_safe = max(0, min(int(args.val_num_workers), os.cpu_count() or 4))
    else:
        val_nw_safe = 0 if os.name == "nt" else val_nw
    if getattr(args, "val_prefetch_factor", None) is not None and val_nw_safe > 0:
        val_pf = max(1, int(args.val_prefetch_factor))

    default_pin = os.name != "nt"
    pin_mem = default_pin if getattr(args, "pin_memory", None) is None else bool(args.pin_memory)

    if getattr(args, "enable_per", False):
        train_dl = PrioritizedDataLoader(
            train_ds,
            batch_size=args.batch_size,
            num_workers=nw,
            pin_memory=pin_mem,
            persistent_workers=use_persistent,
            prefetch_factor=pf,
        )
    else:
        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=pin_mem,
            persistent_workers=use_persistent,
            prefetch_factor=pf,
        )

    _bn_train_dl = train_dl  # full-distribution loader for SWA BN update (never filtered)
    # On Windows, DataLoader worker processes crash unexpectedly during validation
    # after many training epochs (memory pressure kills subprocesses silently).
    # num_workers=0 runs loading in the main process -- safe, and fast enough for val
    # since there's no backward pass.
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=val_nw_safe,
        pin_memory=pin_mem,
        persistent_workers=(use_persistent and val_nw_safe > 0),
        prefetch_factor=None if val_nw_safe == 0 else val_pf,
    )

    # Always overlap CPU Zarr decompress / H2D with GPU compute via a
    # background-thread prefetch queue (helps even when num_workers > 0).
    train_dl = wrap_loader_prefetch(train_dl, args)
    val_dl = wrap_loader_prefetch(val_dl, args)
    print(
        f"[Loader] {len(train_dl)} train batches | {len(val_dl)} val batches | "
        f"{nw} workers | prefetch={pf if pf is not None else 0} | "
        f"thread_prefetch={int(getattr(args, 'thread_prefetch_batches', 8) or 8)} | "
        f"val_workers={val_nw_safe} val_prefetch={val_pf if val_pf is not None else 0} "
        f"pin_mem={pin_mem} persistent={use_persistent}"
    )

    model = build_model(model_name, n_features, args).to(device)  # type: ignore
    if multitask:
        try:
            _fold_prior = _class_prior_tensor(
                cache_path,
                train_idx,
                device,
                use_direction_sidecar=True,
            )
            _init_multitask_direction_bias(model, _fold_prior)
            _p = [float(x) for x in _fold_prior.detach().cpu().tolist()]
            print(
                f"[ClassPrior] Fold train S/H/B prior={_p[0]:.3f}/{_p[1]:.3f}/{_p[2]:.3f} "
                "(direction head bias initialized)"
            )
        except Exception as _prior_exc:
            print(f"[ClassPrior] WARN: prior init skipped ({_prior_exc})")
    # A-C1: transfer contrastive-pretrained encoder weights into the backbone.
    if getattr(args, "pretrain", False):
        try:
            _loaded_pretrain = _load_pretrained_encoder(model, args, device)
            if not _loaded_pretrain:
                _update_pretrain_report(
                    args,
                    {
                        "loaded_into_supervised_training": False,
                        "supervised_transfer": {
                            "status": "skipped_no_checkpoint",
                        },
                    },
                )
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

            _t_ckpt = (
                Path(args.teacher_ckpt)
                if getattr(args, "teacher_ckpt", None)
                else Path(args.checkpoint_dir) / args.teacher_model / f"{args.teacher_model}_best.pt"
            )
            if not _t_ckpt.is_absolute():
                _t_ckpt = Path.cwd() / _t_ckpt
            if not _t_ckpt.exists():
                _t_ckpt = Path(args.checkpoint_dir) / "production_best.pt"  # Fallback
            if not _t_ckpt.exists():
                raise FileNotFoundError(f"Teacher checkpoint not found: {_t_ckpt}")

            teacher_model, _, _, _ = load_pytorch_model(
                checkpoint_path=str(_t_ckpt),
                model_name=args.teacher_model,
                seq_len=getattr(args, "lookahead_bars", 60),
                n_features=n_features,
                device=device,
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
        model = nn.DataParallel(model)
        print(f"[Model] DataParallel x {n_gpus} GPUs")

    # -- torch.compile (PyTorch >= 2.0) - ~20-30 % extra throughput on Ada ----
    # Enabled by default (GPU.torch_compile=True). LSTM/GRU/RNN cells stay
    # eager via torch.compiler.disable so the rest of the graph can use
    # inductor (incl. reduce-overhead); requires Triton (Linux).
    model = maybe_torch_compile(model, device, _GPU_CFG if isinstance(_GPU_CFG, dict) else None)

    crit = build_criterion(
        args,
        device,
        cache_path=cache_path if (classification or multitask) else None,
        train_idx=train_idx if (classification or multitask) else None,
    )
    direction_crit = nn.HuberLoss(delta=float(getattr(args, "huber_delta", 1.5))).to(device)  # type: ignore
    # D: OverconfidencePenalty -- active for regression modes only.
    # R5: state lives in training.loop_losses (read by _compute_loss there).
    if not classification and getattr(args, "overconf_penalty", True):
        _oc_w = float(getattr(args, "overconf_weight", 0.3))
        _oc_t = float(getattr(args, "overconf_threshold", 0.6))
        _loop_losses._OVERCONF_PENALTY = OverconfidencePenalty(conf_threshold=_oc_t, weight=_oc_w).to(device)  # type: ignore
    else:
        _loop_losses._OVERCONF_PENALTY = None
    opt = build_adamw(model.named_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Gradient accumulation: effective batch = batch_size x accum_steps
    _accum = max(1, int(getattr(args, "grad_accum_steps", 1)))
    # OneCycleLR must be stepped once per OPTIMIZER UPDATE (not per batch).
    # steps_per_epoch = ceil(batches / accum_steps) so the total cycle length
    # equals epochs x optimizer-updates-per-epoch.
    _eff_steps = max(1, -(-len(train_dl) // _accum))  # ceiling div
    _sched_kind = (
        str(getattr(args, "lr_schedule", "warmup_cosine")).strip().lower()
    )  # Fix Item 4: default to warmup_cosine
    _total_steps = max(1, args.epochs * _eff_steps)
    if _sched_kind == "warmup_cosine":
        _warmup_ep = max(0, int(getattr(args, "lr_warmup_epochs", 3)))
        _warmup_steps = _warmup_ep * _eff_steps
        if _warmup_steps <= 0:
            _warmup_steps = max(1, int(_total_steps * float(getattr(args, "lr_warmup_pct", 0.1))))
        # Guard: warmup must never exceed total steps (breaks short --epochs N runs)
        _warmup_steps = min(_warmup_steps, max(1, int(_total_steps * 0.3)))
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
        print(
            f"[Scheduler] WarmupCosine | warmup_steps={_warmup_steps} "
            f"| min_ratio={_min_ratio:.3f} | total={_total_steps:,}"
        )
    else:
        max_lr = float(getattr(args, "onecycle_max_lr_mult", 10.0)) * args.lr
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=max_lr,
            total_steps=_total_steps,
            pct_start=float(getattr(args, "onecycle_pct_start", 0.1)),
            anneal_strategy="cos",
        )
        print(
            f"[Scheduler] OneCycleLR | max_lr={max_lr:.2e} | "
            f"steps/ep={_eff_steps} (accum={_accum}) | total={_total_steps:,}"
        )
    # GradScaler is only useful for FP16 (prevents underflow).
    # BF16 covers the same exponent range as FP32 -> scaling would be a no-op.
    _fp16_scaler_needed = args.amp and device.type == "cuda" and amp_dtype == torch.float16
    amp_sc = GradScaler(enabled=_fp16_scaler_needed)
    dtype_name = {torch.bfloat16: "BF16", torch.float16: "FP16", torch.float32: "FP32"}.get(amp_dtype, "?")
    print(f"[AMP] dtype={dtype_name} | GradScaler={'ON' if _fp16_scaler_needed else 'OFF (BF16/FP32)'}")

    # -- Online Hard-Example Miner ---------------------------------------------
    _online_miner = None
    _use_online_miner = bool(getattr(args, "online_hard_mining", True))
    # Per-model miner gating: only enable for models that benefit from it
    _miner_models = str(getattr(args, "curriculum_miner_models", "") or "").strip()
    _miner_allowed = True
    if _miner_models:
        _miner_allowed = model_name.lower() in [m.strip().lower() for m in _miner_models.split(",")]
    if _use_online_miner and (classification or multitask) and _miner_allowed:
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
            # -- Seed miner with live-feedback hard examples from orchestrator ---
            _fb_path = getattr(args, "live_feedback_path", None)
            if _fb_path and _online_miner is not None:
                try:
                    import json as _json
                    from retraining.live_feedback import LiveFeedbackStore as _LFS
                    _fb_store = _LFS.__new__(_LFS)
                    _fb_store._hard_examples = _json.loads(
                        open(_fb_path, encoding="utf-8").read()
                    )
                    _fb_store._metrics = []
                    # Build priority weights aligned to train_idx timestamps
                    # (timestamps are bar indices here; exact alignment happens via
                    #  nearest-neighbor matching in get_priority_weights)
                    _base_ts = train_idx.astype(np.int64)
                    _fw = _fb_store.get_priority_weights(len(train_idx), _base_ts, hard_boost=3.0)
                    # Inject into miner EMA so hard live examples are oversampled
                    # from the very first epoch without waiting for loss accumulation.
                    _online_miner._ema_score = (_fw - 1.0).clip(0.0).astype(np.float32)
                    print(f"[OnlineMiner] Seeded with live-feedback weights from {_fb_path} "
                          f"({int((_fw > 1.0).sum())} boosted samples)")
                except Exception as _fb_exc:
                    print(f"[OnlineMiner] Live-feedback seed failed (ignored): {_fb_exc}")
        except Exception as _om_e:
            print(f"[OnlineMiner] Init failed (disabled): {_om_e}")
            _online_miner = None
    # -- SWA (Stochastic Weight Averaging) -------------------------------------
    # Averages model weights over the last (1 - swa_start_frac) fraction of training.
    # Typically gives +2-5% generalization improvement with zero extra VRAM cost.
    _swa_enabled = bool(getattr(args, "swa_enabled", TRAINING.get("swa_enabled", False)))
    _swa_start_frac = float(getattr(args, "swa_start_frac", TRAINING.get("swa_start_frac", 0.75)))
    _swa_start_frac = min(max(_swa_start_frac, 0.0), 1.0)
    _swa_start_ep = max(1, int(args.epochs * _swa_start_frac))
    _swa_lr = float(getattr(args, "swa_lr", TRAINING.get("swa_lr", 1e-5)))
    _swa_model = None
    _swa_scheduler = None
    _swa_started = False
    if _swa_enabled:
        try:
            from torch.optim.swa_utils import SWALR, AveragedModel
            from torch.optim.swa_utils import update_bn as _swa_update_bn

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
    elif device.type == "cuda" and not args.amp:  # noqa: SIM102
        # FP32 path: enable math SDP (still uses SDPA dispatch, just no Tensor Cores)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{model_name}{fold_suffix}_best.pt"
    cfg_path = ckpt_dir / f"{model_name}{fold_suffix}_config.json"
    # Persist the exact train-only scaler next to the checkpoint so inference
    # transforms live features identically (it prefers this sidecar).
    if _scaler is not None:
        try:
            from training.dataset_builder import _save_scaler_npz

            _save_scaler_npz(Path(cache_path), _scaler, path=best_path.with_name(best_path.stem + "_scaler.npz"))
            # Ordered feature names ("PAIR::feature"): live asserts its column and
            # pair order against this before trading.
            _fnames = _lfs(cache_path, n_features)
            if _fnames:
                best_path.with_name(best_path.stem + "_features.json").write_text(
                    _json_hash.dumps(_fnames), encoding="utf-8"
                )
        except Exception as _ss_e:
            print(f"[Data] WARN: could not save checkpoint scaler sidecar: {_ss_e}")

    # Resume (single-split only; skip per-fold resume id)
    start_ep = 0
    last_path = ckpt_dir / f"{model_name}{fold_suffix}_last.pt"
    # Temporary holders for auxiliary resume state applied after their init blocks
    _resume_chunk_history: list | None = None
    _resume_chunk_streak: int | None = None
    _resume_feat_state: dict | None = None

    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        core = _core_model(model)
        try:
            core.load_state_dict(ck["model_state"])
            opt.load_state_dict(ck["opt_state"])
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(
                    f"\n[Resume Error] Feature dimension mismatch detected! The dataset has changed since this model was last trained. Disabling resume and starting fresh...\n  Details: {e}"
                )
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
        best_cost_sharpe = float(ck.get("best_cost_sharpe", float("-inf")))
        improved = False
        history = ck.get("history", {"train_loss": [], "val_loss": [], "dir_acc": [], "val_sharpe": [], "lr": []})
        for _hist_key in ("seq_len", "difficulty_stage", "curriculum_stalls"):
            history.setdefault(_hist_key, [])

        # Restore auxiliary state (new keys; fall back gracefully for old checkpoints)
        _resume_chunk_history = list(ck.get("chunk_sharpe_history", history.get("val_sharpe", [])))
        _resume_chunk_streak = int(ck.get("chunk_worse_streak", 0))
        _resume_feat_state = ck.get("feat_stability_state")
        # Restore dynamic early-stop state
        _des_no_improve = int(ck.get("no_improve", 0))
        _des_ema = ck.get("des_ema", None)
        _des_best_ema = float(ck.get("des_best_ema", float("inf")))
        _des_lr_halved = bool(ck.get("des_lr_halved", False))
        _des_prev_difficulty = int(ck.get("des_prev_difficulty", -1))
        print(f"[Resume] Loaded exact state from {last_path} (epoch {start_ep})")
    elif args.resume and best_path.exists() and fold_id is None:
        core = _core_model(model)
        try:
            core.load_state_dict(torch.load(best_path, map_location=device))
            _recover_nonfinite_training_state(model, opt)
            print(f"[Resume] Loaded weights from {best_path}")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(
                    f"\n[Resume Error] Feature dimension mismatch detected! The dataset has changed since this model was last trained. Disabling resume and starting fresh...\n  Details: {e}"
                )
                args.resume = False
            else:
                raise e

    # SACS config (read once; defaults are safe no-ops when sacs_enabled=False)
    _sacs_enabled = bool(getattr(args, "sacs_enabled", False))
    _sacs_eps = float(getattr(args, "sacs_eps", 0.005))
    _sacs_n = max(1, int(getattr(args, "sacs_n_samples", 5)))
    _sacs_lam = float(getattr(args, "sacs_sharpness_weight", 1.0))
    _best_sacs_score: float = float("inf")  # lower = better (sharpness-penalized val loss)
    _best_from_warmup = False  # current best came from a direction-only warmup epoch

    if start_ep == 0:
        best_val_loss = float("inf")
        best_sharpe = float("-inf")
        best_cost_sharpe = float("-inf")
        _best_sacs_score = float("inf")
        improved = False

        history = {
            "train_loss": [],
            "val_loss": [],
            "dir_acc": [],
            "val_sharpe": [],
            "lr": [],
        }

    if use_direction_targets and start_ep == 0:
        _direction_probe(
            model,
            cache_path,
            train_idx,
            val_idx,
            args,
            device,
            model_name=model_name,
            n_features=n_features,
            amp_dtype=amp_dtype,
        )

    # -- A3: Variable-length sequence curriculum -------------------------------
    _CURR = getattr(args, "curriculum", None)
    if not isinstance(_CURR, dict):
        _CURR = SETTINGS_CURRICULUM
    _chunk_min_batches = int(_CURR.get("chunk_early_stop_min_batches", 50))
    _feat_groups = _CURR.get("feature_groups", {})

    # One-shot curriculum ↔ FEATURE_MASK ↔ schema consistency audit (mismatch C/D)
    try:
        from config.curriculum_audit import (
            audit_curriculum_feature_groups,
            audit_required_market_columns,
            audit_settings_yaml_curriculum_drift,
            format_audit_warnings,
        )
        from config.feature_mask import FEATURE_MASK as _FM_AUDIT

        _schema_for_audit = None
        try:
            _schema_for_audit = _load_feature_schema(cache_path, n_features)
        except Exception:
            _schema_for_audit = None
        if _schema_for_audit is None:
            _schema_for_audit = [k for k, v in _FM_AUDIT.items() if v]
        _audit = audit_curriculum_feature_groups(
            schema=_schema_for_audit,
            feature_groups=_feat_groups,
            feature_mask=_FM_AUDIT,
        )
        for _line in format_audit_warnings(_audit):
            _log_warn(_line)
        _drift = audit_settings_yaml_curriculum_drift(
            SETTINGS_CURRICULUM,
            _CURR,
            yaml_path=str(getattr(args, "config", "config/run.yaml") or "config/run.yaml"),
        )
        for _err in _drift.get("errors") or []:
            _log_warn(f"[CurriculumDrift] {_err}")
        for _line in format_audit_warnings(_drift, prefix="[CurriculumDrift]"):
            _log_warn(_line)
        _mkt = audit_required_market_columns(feature_mask=_FM_AUDIT)
        for _line in format_audit_warnings(_mkt, prefix="[MarketSchema]"):
            _log_warn(_line)
        for _err in _mkt.get("errors") or []:
            _log_warn(f"[MarketSchema] {_err}")
    except Exception as _audit_exc:
        _log_warn(f"[CurriculumAudit] skipped ({_audit_exc})")

    def _seq_len_for_epoch(ep: int) -> int:
        _sched = _CURR.get("seq_schedule") if isinstance(_CURR, dict) else None
        if not _sched:
            return args.seq_len
        active = args.seq_len
        for entry in sorted(_sched, key=lambda e: int(e.get("epoch_start", 0))):
            if ep >= int(entry.get("epoch_start", 0)):
                active = int(entry["seq_len"])
        return active

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
                    param.requires_grad_(ep >= 0)  # always requires_grad; gradient zeroed via scheduler
            return
        # After earliest_unfreeze: all parameters are trainable
        for param in core.parameters():
            if _is_uninitialized_parameter(param):
                continue
            param.requires_grad_(True)

    # -- B: Difficulty curriculum -- load diff sidecar once, filter each epoch -
    _diff_arr = _load_diff_array(cache_path, n_samples)
    # If no pre-built sidecar exists, synthesise difficulty from label magnitude
    # and HMM regime (if available):
    #   0=easy  → |y| > p75 AND regime=trending
    #   1=medium → |y| > p40
    #   2=hard  → |y| ≤ p40 (near-zero / ambiguous direction)
    # This makes curriculum advancement meaningful: the model must first learn
    # unambiguous directional moves before being exposed to noisy near-zero bars.
    if _diff_arr is None and str(cache_path).endswith(".zarr"):
        try:
            import zarr as _zarr_diff
            _zd = _zarr_diff.open(str(cache_path), mode="r")
            if "y" in _zd:
                _y_full = np.asarray(_zd["y"][:n_samples], dtype=np.float32)
                _abs_y = np.abs(_y_full)
                _p40 = float(np.percentile(_abs_y, 40))
                _p75 = float(np.percentile(_abs_y, 75))
                _diff_synth = np.where(_abs_y <= _p40, 2,
                              np.where(_abs_y >= _p75, 0, 1)).astype(np.uint8)
                # Regime overlay: if HMM regime available, promote trending bars to easy
                if "regime" in _zd:
                    _reg = np.asarray(_zd["regime"][:n_samples], dtype=np.int8)
                    # regime=0 assumed trending (lowest-vol HMM state)
                    _diff_synth = np.where((_reg == 0) & (_abs_y >= _p40), 0, _diff_synth)
                _diff_arr = _diff_synth
                _pct_easy = int((_diff_arr == 0).mean() * 100)
                _pct_hard = int((_diff_arr == 2).mean() * 100)
                print(f"[Curriculum] Built difficulty from |y|+regime: easy={_pct_easy}% hard={_pct_hard}%")
        except Exception as _de:
            print(f"[Curriculum] Difficulty synthesis failed (no sidecar): {_de}")
    _feature_schema = _load_feature_schema(cache_path, n_features)
    if _feature_schema is None:
        _log_warn(
            "[Curriculum] Ordered feature schema sidecar missing; feature-group mask will use FEATURE_MASK only if lengths match."
        )
    _schema_for_masks = _feature_schema or list(getattr(args, "_feat_names", []) or [])

    _feature_ablation_cfg = _feature_ablation_config(args)

    _feature_ablation_mask_np, _feature_ablation_report = _build_feature_ablation_mask(
        _schema_for_masks,
        _feat_groups,
        _feature_ablation_cfg,
        n_features,
    )

    _feature_ablation_mask = (
        torch.from_numpy(_feature_ablation_mask_np).to(device) if _feature_ablation_mask_np is not None else None
    )

    if _feature_ablation_report.get("enabled"):
        _log_info(
            f"[FeatureAblation] {model_name}: {_feature_ablation_report.get('name')} "
            f"masked={_feature_ablation_report.get('masked_count')}/{n_features}"
        )

    try:
        _fa_dir = Path(args.checkpoint_dir) / model_name

        _fa_dir.mkdir(parents=True, exist_ok=True)

        _feature_ablation_report.update(
            {
                "model_name": model_name,
                "schema_available": bool(_schema_for_masks),
                "written_at": datetime.now(UTC).isoformat(),
            }
        )

        _safe_save_json(_feature_ablation_report, _fa_dir / f"{model_name}_feature_ablation_report.json")

    except Exception as _fa_e:
        _log_warn(f"[FeatureAblation] Report write failed: {_fa_e}")

    # -- A: Feature stability monitor -----------------------------------------
    _feat_stability = FeatureStabilityMonitor(
        n_features=n_features,
        ema_alpha=0.90,
        soft_threshold=2.0,  # sigma shift -> noisy (dampen to 0.5x)
        hard_threshold=4.0,  # sigma shift -> immediately freeze
        freeze_after=3,  # consecutive soft epochs -> freeze
        damping_factor=0.50,
        warmup_epochs=3,  # don't penalise in first 3 epochs (EMA cold start)
        min_active_pct=0.50,  # always keep >=50% of features active
    )
    if _resume_feat_state is not None:
        try:
            _feat_stability.load_state(_resume_feat_state)
            print(
                f"[Resume] Restored FeatureStabilityMonitor state "
                f"(epoch={_feat_stability._epoch}, "
                f"frozen={int(_feat_stability._frozen.sum())})"
            )
        except Exception as _fst_e:
            print(f"[Resume] FeatureStabilityMonitor state restore failed (cold-start): {_fst_e}")
    _feat_mask: torch.Tensor | None = None  # updated each epoch; None = all active

    # -- A2: Chunk early stopping state ---------------------------------------
    _chunk_sharpe_history: list = []
    _chunk_worse_streak: int = 0
    if _resume_chunk_history is not None:
        _chunk_sharpe_history = _resume_chunk_history
        _chunk_worse_streak = _resume_chunk_streak or 0

    # Rich display replaces the plain print table; plain fallback when unavailable
    if _rich_display is not None:
        _rich_display.__enter__()
    else:
        print(
            f"\n{'Ep':>5} {'Train':>11} {'Val':>11} {'DirAcc':>8} {'vSharpe':>9} {'LR':>10} {'Time':>7} {'GPU MB':>8}"
        )
        print("-" * 72)

    # -- Crash checkpoint helper -----------------------------------------------
    def _save_crash_ckpt(failed_ep: int, exc: Exception) -> None:
        """Save a crash checkpoint so the error can be inspected and training resumed."""
        import traceback as _tb

        crash_path = ckpt_dir / f"{model_name}{fold_suffix}_crash.pt"
        try:
            _core = _core_model(model)
            _safe_save(
                {
                    "epoch": failed_ep,
                    "model_state": _core.state_dict(),
                    "opt_state": opt.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": amp_sc.state_dict() if args.amp and device.type == "cuda" else None,
                    "best_val_loss": best_val_loss,
                    "best_sharpe": best_sharpe,
                    "no_improve": _des_no_improve,
                    "des_ema": _des_ema,
                    "des_best_ema": _des_best_ema,
                    "des_lr_halved": _des_lr_halved,
                    "des_prev_difficulty": _des_prev_difficulty,
                    "history": history,
                    "fold_id": fold_id,
                    "chunk_sharpe_history": _chunk_sharpe_history,
                    "chunk_worse_streak": _chunk_worse_streak,
                    "feat_stability_state": _feat_stability.get_state(),
                    "error_msg": str(exc),
                    "error_type": type(exc).__name__,
                    "error_traceback": _tb.format_exc(),
                },
                crash_path,
            )
            print(f"\n[Train] Crash checkpoint saved  {crash_path}")
            print("[Train] Resume from last clean epoch with: --resume")
            if _rich_display is not None:
                _rich_display.__exit__(None, None, None)
        except Exception as _cs_exc:
            print(f"[Train] Warning: could not save crash checkpoint: {_cs_exc}")

    # Initialize curriculum variables for reporting/fallback
    _active_seq_len = args.seq_len
    _active_diff_stage = 0
    _seq_frozen = False
    _last_logged_seq_len = -1

    epoch_bar = (
        _pbar(range(start_ep, args.epochs), desc=f"Train {model_name.upper()}", unit="ep")
        if _rich_display is None
        else range(start_ep, args.epochs)
    )

    # -- Advanced Training Mechanics: EWC & Adversarial --
    _ewc = None
    if getattr(args, "enable_ewc", False) and start_ep > 0:
        # We only compute EWC if we are resuming from a previous trained state
        try:
            print("[EWC] Computing Fisher Information Matrix (max 1000 samples)...")

            def _ewc_loss_fn(outputs, labels):
                # Mirror the active criterion when possible
                if multitask or classification:
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    if labels.dtype.is_floating_point:
                        y = (labels.reshape(-1).clamp(-1, 1) + 1).round().long()
                    else:
                        y = labels.reshape(-1).long()
                    y = y.clamp(0, max(1, outputs.shape[-1] - 1))
                    return nn.functional.huber_loss(outputs.squeeze(), y.float())
                pred = outputs[0] if isinstance(outputs, tuple) else outputs
                return nn.functional.mse_loss(pred.reshape(-1), labels.reshape(-1).float()[: pred.numel()])

            _ewc = ElasticWeightConsolidation(
                model,
                train_ds,
                device,
                max_samples=1000,
                loss_fn=_ewc_loss_fn,
                classification=bool(multitask or classification),
            )
            print("[EWC] Initialized successfully. Fisher diagonal locked.")
        except Exception as e:
            _ewc = None
            print(f"[EWC] Failed to initialize EWC (continuing without EWC): {e}")
    elif getattr(args, "enable_ewc", False):
        print(
            "[EWC] --enable-ewc set but start_ep == 0 (fresh run): no prior trained "
            "state exists to protect, so EWC is deferred. It will engage on a resume "
            "run (start_ep > 0)."
        )
    _si = None
    if getattr(args, "enable_si", False):
        try:
            print("[SI] Initializing Synaptic Intelligence tracking...")
            if hasattr(model, "initialize_parameters"):
                try:
                    dummy_in = torch.zeros(2, int(getattr(args, "seq_len", 90) or 90), int(n_features), device=device)
                    model.initialize_parameters(dummy_in)
                except Exception:
                    pass
            _si = SynapticIntelligence(model, epsilon=1e-3)
            print("[SI] Initialized successfully. Tracking path integral.")
        except Exception as e:
            _si = None
            print(f"[SI] Failed to initialize SI (continuing without SI): {e}")

    # ── Dynamic SI λ (regime drift / volatility scaling) ─────────────────
    # Computed per-epoch from the FeatureStabilityMonitor's max feature shift
    # (see the Feature Stability Monitor block inside the epoch loop):
    #     λ_epoch = si_lambda * 1 / (1 + max_shift²)
    # so the SI penalty relaxes during regime shocks and re-locks after they
    # stabilize. No per-batch state is needed.
    _adversarial = None
    _adv_feature_names = _feature_schema or list(getattr(args, "_feat_names", []) or [])
    # Check per-model adversarial gating
    _adv_models = str(getattr(args, "adversarial_models", "") or "").strip()
    _adv_allowed = True
    if _adv_models:
        _adv_allowed = model_name.lower() in [m.strip().lower() for m in _adv_models.split(",")]

    if getattr(args, "enable_adversarial", False) and _adv_allowed:
        try:
            from pretrain.hard_example_mining import PretrainHardExampleMiner
            from training.adversarial_generator import create_adversarial_attack

            _adv_method = str(getattr(args, "adversarial_method", "pgd") or "pgd").lower()
            if model_name == "gnn" and _adv_method == "pgd":
                _adv_method = "graph_pgd"

            # Load feature vulnerability scores from pretraining (Task 2)
            _feature_eps_multipliers = None
            try:
                _vuln = PretrainHardExampleMiner.load_vulnerability_scores()
                if _vuln is not None and len(_vuln) > 0:
                    _feature_eps_multipliers = _vuln
                    print(f"[Adversarial] Loaded feature vulnerability scores ({len(_vuln)} dims)")
            except Exception as _vuln_e:
                print(f"[Adversarial] Could not load vulnerability scores: {_vuln_e}")

            _adversarial = create_adversarial_attack(
                method=_adv_method,
                eps=float(getattr(args, "adversarial_eps", 0.3)),
                alpha=float(getattr(args, "adversarial_alpha", 0.01)),
                steps=int(getattr(args, "adversarial_steps", 7)),
                probability=float(getattr(args, "adversarial_prob", 0.01)),
                feature_names=_adv_feature_names or None,
                normalize_grad=bool(getattr(args, "adversarial_normalize_grad", False)),
                warmup_steps=int(getattr(args, "adversarial_warmup_steps", 0)),
                feature_eps_multipliers=_feature_eps_multipliers,
            )
            _adversarial.train()
            print(
                f"[Adversarial] Initialized method={_adv_method} "
                f"(eps={getattr(args, 'adversarial_eps', 0.3)}, "
                f"prob={getattr(args, 'adversarial_prob', 0.01)}, "
                f"n_features_named={len(_adv_feature_names)})."
            )
        except Exception as e:
            _adversarial = None
            print(f"[Adversarial] Failed to initialize (continuing without adversarial): {e}")

    # Overfitting controller - flags from evaluate_epoch are applied each epoch
    from training.training_controller import TrainingController

    _adaptation_cfg = _CURR.get("adaptation") if isinstance(_CURR, dict) else None
    _train_ctrl = TrainingController(report_dir=str(ckpt_dir), adaptation=_adaptation_cfg or {})
    _train_ctrl.set_recipe(str(model_name))
    _ctrl_stop_early = False
    _ctrl_stop_counter = 0  # consecutive stop_early epochs before hard early-stop

    # -- Unified CurriculumManager (Improvement #4) ---------------------------
    # Opt-in extra curriculum layer. Mirrors n_samples on the *train* fold so
    # difficulty scores line up with the samples this fold actually trains on.
    # P3-2: when --curriculum-callback is set, the provider is built through the
    # create_curriculum_callback() factory (CustomCurriculumAdapter) instead of
    # create_curriculum_manager(); both expose the same provider surface used below.
    _curriculum_mgr = None
    _cm_mode = str(getattr(args, "curriculum_manager_mode", "combined") or "combined")
    if getattr(args, "curriculum_manager", False):
        try:
            _cm_n = len(train_idx)
            _cm_diff = None
            if _diff_arr is not None and len(_diff_arr) == n_samples:
                _cm_diff = np.asarray(_diff_arr[train_idx], dtype=float)
            if bool(getattr(args, "curriculum_callback", False)):
                from training.curriculum_callbacks import create_curriculum_callback

                _cb_kw = {
                    "n_levels": int(getattr(args, "curriculum_n_levels", 10) or 10),
                    "start_level": int(getattr(args, "curriculum_start_level", 1) or 1),
                    "advance_rate": float(getattr(args, "curriculum_advance_rate", 0.1)),
                    "max_level": int(getattr(args, "curriculum_start_level", 1) or 1)
                    + int(getattr(args, "curriculum_n_levels", 9) or 9),
                    "total_epochs": max(1, int(args.epochs)),
                    "use_loss_weighting": bool(getattr(args, "use_loss_weighting", False)),
                }
                _curriculum_provider = create_curriculum_callback(
                    "custom",
                    difficulty_scores=_cm_diff,
                    mode=_cm_mode,
                    **_cb_kw,
                )
                _curriculum_mgr = _CurriculumProvider(_curriculum_provider)
                print(f"[CurriculumCallback] Enabled (mode={_cm_mode}, factory=create_curriculum_callback)")
            else:
                from training.curriculum import create_curriculum_manager

                _curriculum_mgr = create_curriculum_manager(
                    mode=_cm_mode,
                    n_samples=_cm_n,
                    difficulty_scores=_cm_diff,
                    total_epochs=max(1, int(args.epochs)),
                    seed=int(getattr(args, "seed", 1337)),
                    # Self-paced config
                    sp_pace=str(getattr(args, "self_paced_pace", "linear")),
                    sp_lambda=float(getattr(args, "self_paced_lambda", 1.0)),
                    use_self_paced=bool(getattr(args, "use_self_paced", False)),
                    # Loss weighting config
                    lw_scheme=str(getattr(args, "loss_weighting_scheme", "focal")),
                    focal_gamma=float(getattr(args, "loss_weighting_focal_gamma", 2.0)),
                    use_loss_weighting=bool(getattr(args, "use_loss_weighting", False)),
                    # Miner feedback config
                    forgetting_threshold=float(getattr(args, "curriculum_forgetting_threshold", 0.15)),
                    easy_threshold=float(getattr(args, "curriculum_easy_threshold", 0.60)),
                    freeze_patience=int(getattr(args, "curriculum_freeze_patience", 1)),
                    min_stable_sharpe=float((_adaptation_cfg or {}).get("min_stable_sharpe", 0.50)),
                )
                _curriculum_mgr = _CurriculumProvider(_curriculum_mgr)
                print(f"[CurriculumManager] Enabled (mode={_cm_mode}) over {_cm_n:,} train samples")
        except Exception as _cm_exc:
            _curriculum_mgr = None
            print(f"[CurriculumManager] Disabled ({_cm_exc})")

    # Miner feedback gating
    _miner_models = str(getattr(args, "curriculum_miner_models", "") or "").strip()
    _miner_allowed = True
    if _miner_models:
        _miner_allowed = model_name.lower() in [m.strip().lower() for m in _miner_models.split(",")]
    _miner_feedback_enabled = (
        bool(getattr(args, "curriculum_miner_feedback", False))
        and _miner_allowed
        and _online_miner is not None
        and _curriculum_mgr is not None
    )

    # Self-paced / loss weighting model gating (computed after curriculum build;
    # used only to guard the _cm_wl application at the sample-weight lookup site).
    _sp_models = str(getattr(args, "self_paced_models", "") or "").strip()
    _sp_allowed = True
    if _sp_models:
        _sp_allowed = model_name.lower() in [m.strip().lower() for m in _sp_models.split(",")]

    _lw_models = str(getattr(args, "loss_weighting_models", "") or "").strip()
    _lw_allowed = True
    if _lw_models:
        _lw_allowed = model_name.lower() in [m.strip().lower() for m in _lw_models.split(",")]

    _ema_model = None

    # ── Dynamic early-stop state ──────────────────────────────────────────────
    # Composite EMA score: lower is better (val_loss dominates, sharpe subtracts)
    # These defaults are overwritten by resume-checkpoint restore when resuming.
    _des_ema_alpha: float = 0.3            # EMA smoothing (higher = more reactive)
    _des_base_patience: int = int(getattr(args, "early_stop_patience", 10))
    _des_lr_min: float = float(getattr(args, "lr", 5e-5)) * 1e-3  # floor: 0.1% of initial LR
    # Mutable state — overwritten by resume restore below if args.resume
    _des_ema: float | None = None
    _des_no_improve: int = 0
    _des_lr_halved: bool = False
    _des_best_ema: float = float("inf")
    _des_prev_difficulty: int = -1         # curriculum stage tracker (reset counter on advance)

    # Early-stop metric flags — defined here so they're in scope inside the epoch loop.
    stop_on_sharpe = getattr(args, "early_stop_metric", "val_loss") == "sharpe"
    stop_on_cost_sharpe = getattr(args, "early_stop_metric", "val_loss") == "cost_sharpe"

    for ep in epoch_bar:
        if _TRAIN_LOGGER is not None:
            _TRAIN_LOGGER.on_epoch_start(ep, total_epochs=args.epochs, seq_len=_seq_len_for_epoch(ep))
        curr_seq_len = _seq_len_for_epoch(ep)
        _active_seq_len = curr_seq_len

        # Advance difficulty stage from schedule
        _diff_sched = _CURR.get("difficulty_schedule") if isinstance(_CURR, dict) else None
        if _diff_sched:
            for _ds_entry in sorted(_diff_sched, key=lambda e: int(e.get("epoch_start", 0))):
                if ep >= int(_ds_entry.get("epoch_start", 0)):
                    _active_diff_stage = int(_ds_entry.get("max_difficulty", _active_diff_stage))

        if _last_logged_seq_len != curr_seq_len:
            _log_info(f"[Curriculum] Epoch {ep + 1}: active seq_len={curr_seq_len}")
            try:
                from data.dataset_manifest import DatasetManifest

                _dm2 = DatasetManifest(str(Path(cache_path).parent))
                _unfrozen = [
                    g
                    for g, v in _feat_groups.items()
                    if not v.get("always_on", True) and ep >= v.get("epoch_unfreeze", 999)
                ]
                _dm2.log_curriculum_stage(
                    ep + 1,
                    "seq_len_advance",
                    curr_seq_len,
                    _unfrozen,
                    int(_active_diff_stage),
                )
            except Exception:
                pass

            _last_logged_seq_len = curr_seq_len

        # -- A4: Feature freeze schedule ---------------------------------------
        _unfreeze_features_for_epoch(model, ep)

        # -- Feature Stability Monitor -----------------------------------------
        epoch_si_lambda = float(getattr(args, "si_lambda", 1.0))
        try:
            _stab_pool = locals().get("ep_train_idx", train_idx)
            _sample_idx = np.random.choice(_stab_pool, size=min(512, len(_stab_pool)), replace=False)
            _samp_ds = ZarrStreamDataset(cache_path, _sample_idx, shuffle_chunks=False, scaler=_scaler)
            _samp_dl = DataLoader(_samp_ds, batch_size=512, shuffle=False, num_workers=0)
            _samp_xb, _ = next(iter(_samp_dl))
            _feat_stability.update(_samp_xb.numpy())
            _feat_mask = _feat_stability.get_mask(device=device)
            _stab_report = _feat_stability.report()

            # Dynamic SI scaling based on regime drift / volatility
            _max_shift = float(_stab_report["feat_max_shift"])
            _dyn_w = 1.0 / (1.0 + (_max_shift**2))
            epoch_si_lambda = epoch_si_lambda * _dyn_w
            # Clamp to per-model bounds from profile (si_lambda_min / si_lambda_max)
            _si_lmin = float(getattr(args, "si_lambda_min", 0.0))
            _si_lmax = float(getattr(args, "si_lambda_max", epoch_si_lambda))
            epoch_si_lambda = max(_si_lmin, min(_si_lmax, epoch_si_lambda))

            if _stab_report["feat_frozen"] > 0 or _stab_report["feat_noisy"] > 0:
                _log_info(
                    f"[FeatStab] Ep {ep + 1}: frozen={_stab_report['feat_frozen']} "
                    f"noisy={_stab_report['feat_noisy']} "
                    f"active={_stab_report['feat_active']}/{n_features} "
                    f"max_shift={_max_shift:.2f}sigma (si_lambda={epoch_si_lambda:.2f})"
                )
            _safe_wandb_log(
                run,
                {
                    "feat/frozen": _stab_report["feat_frozen"],
                    "feat/noisy": _stab_report["feat_noisy"],
                    "feat/max_shift": _max_shift,
                    "feat/si_dynamic_lambda": epoch_si_lambda,
                    "epoch": ep,
                },
            )
        except Exception as _stab_exc:
            _log_warn(f"[FeatStab] Monitor update failed (skipped): {_stab_exc}")
            _feat_mask = None  # fall back to no masking

        # -- Feature Curriculum Mask -------------------------------------------
        try:
            from config.feature_mask import FEATURE_MASK

            _schema = _feature_schema or [k for k, v in FEATURE_MASK.items() if v]
            if len(_schema) == n_features:
                _curr_mask = torch.ones(n_features, device=device)
                _zeroed_cnt = 0
                _missing_cnt = 0
                _missing_sample: list[str] = []
                for g_name, g_cfg in _feat_groups.items():
                    if not g_cfg.get("always_on", True) and ep < g_cfg.get("epoch_unfreeze", 0):
                        for f_name in g_cfg.get("features", []):
                            if f_name in _schema:
                                _curr_mask[_schema.index(f_name)] = 0.0
                                _zeroed_cnt += 1
                            else:
                                _missing_cnt += 1
                                if len(_missing_sample) < 8:
                                    _missing_sample.append(f"{g_name}:{f_name}")
                if _missing_cnt and not getattr(args, "_curriculum_missing_logged", False):
                    args._curriculum_missing_logged = True
                    _log_warn(
                        f"[Curriculum] {_missing_cnt} group feature(s) not in schema "
                        f"(silently skipped when freezing); e.g. {', '.join(_missing_sample)}"
                    )
                if _zeroed_cnt > 0:
                    _log_info(f"[Curriculum] Epoch {ep + 1}: Zeroed out {_zeroed_cnt} features by schema")
                    if _feat_mask is None:
                        _feat_mask = _curr_mask
                    else:
                        _feat_mask = _feat_mask * _curr_mask
            else:
                _log_warn(
                    f"[Curriculum] Feature schema length {len(_schema)} does not match n_features={n_features}; skipping feature-group mask."
                )
        except Exception as _curr_exc:
            _log_warn(f"[Curriculum] Mask generation failed: {_curr_exc}")

        # Reset frozen features if difficulty stage increased
        # -- B: Difficulty curriculum -- rebuild dataloader with filtered indices --
        ep_train_idx = train_idx
        # Label-magnitude gate: early epochs only train on unambiguous bars.
        # Threshold decays from p75 → 0 (all samples unlocked) over half of
        # total_epochs so the model first learns clear directional moves.
        # Only active when _diff_arr was built from |y| (or loaded from sidecar)
        # AND no CurriculumManager is running (avoids double-gating).
        if _curriculum_mgr is None and _diff_arr is not None:
            _total_eps = max(1, int(args.epochs))
            _gate_epochs = max(1, _total_eps // 2)
            if ep < _gate_epochs:
                # Unlock easy (0) first, add medium (1) at epoch _gate_epochs//2
                _max_tier = 0 if ep < _gate_epochs // 2 else 1
                _tier_mask = _diff_arr[train_idx] <= _max_tier
                _gated = train_idx[_tier_mask]
                if len(_gated) >= 50:
                    ep_train_idx = _gated
                    if ep == 0 or ep == _gate_epochs // 2:
                        _log_info(
                            f"[LabelMagnitudeGate] Epoch {ep + 1}: "
                            f"tier≤{_max_tier} → {len(ep_train_idx):,}/{len(train_idx):,} samples"
                        )
        if _curriculum_mgr is not None:
            _cm_mask = _curriculum_mgr.get_inclusion_mask(ep)
            # Apply mask to ep_train_idx by intersecting with allowed indices
            _allowed = np.where(_cm_mask)[0]
            ep_train_idx = np.intersect1d(ep_train_idx, _allowed)
            if len(ep_train_idx) < 50:
                ep_train_idx = train_idx
        # -- Unified CurriculumManager (Improvement #4): apply inclusion mask --
        if _curriculum_mgr is not None:
            try:
                # Get per-sample losses from online miner for self-paced/loss weighting
                epoch_losses = None
                if _miner_feedback_enabled and _online_miner is not None:
                    epoch_losses = _online_miner._loss_buffer[-1].copy()
                    # Also get forgetting/easy ratios for curriculum pace control
                    _forgetting_rate = float(_online_miner.get_forgotten_mask().mean())
                    _easy_ratio = float(_online_miner.get_easy_mask().mean())
                else:
                    _forgetting_rate = 0.0
                    _easy_ratio = 0.0

                _cm_info = _curriculum_mgr.update(
                    ep,
                    losses=epoch_losses,
                    forgetting_rate=_forgetting_rate,
                    easy_ratio=_easy_ratio,
                )
                _cm_mask = _curriculum_mgr.get_inclusion_mask()
                if len(_cm_mask) == len(ep_train_idx) and float(_cm_mask.mean()) < 0.999:
                    _ep_cm_idx = ep_train_idx[_cm_mask]
                    if len(_ep_cm_idx) >= 50:
                        ep_train_idx = _ep_cm_idx
                        _log_info(
                            f"[CurriculumManager] Epoch {ep + 1}: included "
                            f"{len(ep_train_idx):,}/{len(_cm_mask):,} samples "
                            f"({float(_cm_mask.mean()):.0%})"
                        )
                history.setdefault("curriculum_manager_state", []).append(
                    {
                        "epoch": ep,
                        "mode": getattr(args, "curriculum_manager_mode", "combined"),
                        "inclusion_rate": float(_cm_mask.mean()) if len(_cm_mask) else 1.0,
                        "weights_mean": float(np.asarray(_cm_info.get("weights", np.ones(1))).mean()),
                    }
                )
                # Adversarial + Curriculum Coordination: scale eps with difficulty level
                if _adversarial is not None and bool(getattr(args, "adversarial_eps_curriculum_scale", False)):
                    _diff_level = _cm_info.get("difficulty_level", 1)
                    _n_levels = (
                        getattr(_curriculum_mgr.config.difficulty, "max_level", 10)
                        if _curriculum_mgr.config.difficulty
                        else 10
                    )
                    _level_ratio = _diff_level / max(1, _n_levels)
                    _base_eps = float(getattr(args, "adversarial_eps", 0.3))
                    _scaled_eps = _base_eps * _level_ratio
                    _adversarial.set_eps(_scaled_eps)
                    if hasattr(_adversarial, "set_warmup_step"):
                        _adversarial.set_warmup_step(ep)
                    _log_info(
                        f"[Adversarial+Curriculum] Epoch {ep + 1}: eps scaled to {_scaled_eps:.4f} (level {_diff_level}/{_n_levels})"
                    )

            except Exception as _cm_exc:
                _log_warn(f"[CurriculumManager] Epoch {ep + 1} update failed: {_cm_exc}")
        _cm_wl = None if _period_wl is None else _period_wl.copy()
        if _curriculum_mgr is not None and (_sp_allowed or _lw_allowed):
            try:
                if _cm_wl is None:
                    _cm_wl = np.ones(n_samples, dtype=np.float64)
                _cm_wl[train_idx] *= np.asarray(
                    _curriculum_mgr.get_sample_weights(),
                    dtype=np.float64,
                )
            except Exception as _cm_wl_exc:
                _log_warn(f"[CurriculumManager] Sample-weight lookup failed: {_cm_wl_exc}")
                _cm_wl = None
        _direction_warmup_active = bool(
            use_direction_targets and multitask and ep < max(0, int(getattr(args, "direction_warmup_epochs", 2)))
        )
        if _direction_warmup_active:
            ep_train_idx = _balanced_direction_indices(
                cache_path,
                ep_train_idx,
                total_samples=len(ep_train_idx),
                seed=int(getattr(args, "seed", 1337)) + ep,
            )
            _log_info(
                f"[DirectionWarmup] Epoch {ep + 1}: balanced direction-only batches ({len(ep_train_idx):,} samples)"
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
            if len(ep_train_idx) != len(train_idx) and not _direction_warmup_active:
                _log_info(
                    f"[Curriculum] Epoch {ep + 1}: training on ({len(ep_train_idx):,}/{len(train_idx):,} samples)"
                )
            _ep_ds = ZarrStreamDataset(
                cache_path,
                ep_train_idx,
                shuffle_chunks=True,
                multitask_targets=use_direction_targets,
                return_indices=True,
                scaler=_scaler,  # was missing: warmup/curriculum epochs saw unscaled inputs
                pair_targets=_pair_targets,
            )
            epoch_train_dl = DataLoader(
                _ep_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=nw,
                pin_memory=pin_mem,
                persistent_workers=(use_persistent and nw > 0),
                prefetch_factor=pf if nw > 0 else None,
            )
            epoch_train_dl = wrap_loader_prefetch(epoch_train_dl, args)

        # Batch-level progress bars
        train_pbar = _pbar(total=len(epoch_train_dl), desc=f"  Ep {ep + 1:3d} [Tr]", unit="batch", leave=False)
        val_pbar = _pbar(total=len(val_dl), desc=f"  Ep {ep + 1:3d} [Va]", unit="batch", leave=False)

        t0 = time.time()
        _thermal_lim = int(_GPU_CFG.get("thermal_limit_celsius", 83))

        # ── Online miner: begin epoch (roll buffer) ──────────────────────
        if _online_miner is not None:
            _online_miner.begin_epoch()
        # ─────────────────────────────────────────────────────────────────

        try:
            tl = train_epoch(
                model,
                epoch_train_dl,
                opt,
                direction_crit if _direction_warmup_active else crit,
                amp_sc,
                device,
                args.amp,
                classification,
                grad_clip=args.grad_clip,
                pbar=train_pbar,
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
                adversarial_gen=_adversarial,
                adversarial_feature_names=_adv_feature_names or None,
                ewc_module=_ewc,
                ewc_lambda=float(getattr(args, "ewc_lambda", 400.0)) * (1.0 + float(ep) / max(1, int(getattr(args, "epochs", 40)))),
                si_module=_si,
                si_lambda=epoch_si_lambda,
                sample_weight_lookup=_cm_wl,
            )
        except Exception as _epoch_exc:
            _log_error(f"[Train] Epoch {ep + 1} failed for {model_name}", _epoch_exc)
            if _TRAIN_LOGGER is not None:
                _TRAIN_LOGGER.on_epoch_failure(ep, _epoch_exc)
            _save_crash_ckpt(ep, _epoch_exc)
            raise
        train_pbar.close()

        # ── Online miner: end epoch (update EMA scores + forgetting tracker) ──
        if _online_miner is not None:
            _online_miner.end_epoch()
            # Feed per-sample losses back into the curriculum so self-paced /
            # loss-weighting difficulty updates use real training losses.
            if _curriculum_mgr is not None:
                try:
                    _epoch_losses_for_curriculum = _online_miner._loss_buffer[-1].copy()
                    _curriculum_mgr.set_losses(_epoch_losses_for_curriculum)
                except Exception as _cl_exc:
                    _log_warn(f"[Curriculum] set_losses failed: {_cl_exc}")
        # ─────────────────────────────────────────────────────────────────────

        # Finish async CUDA work and free cached blocks before validation. Val uses
        # pin_memory=False, but sync+gc still reduces fragmentation after train_epoch.
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

        try:
            from training.train_gpu import _sharpe_ann_factor
            vl, da, v_sh = validate_epoch(
                model,
                val_dl,
                crit,
                device,
                classification,
                pbar=val_pbar,
                amp=False,
                amp_dtype=torch.float32,
                seq_len=curr_seq_len,
                multitask=multitask,
                feature_mask=_feat_mask,
                sharpe_ann_factor=_sharpe_ann_factor(args),
                lookahead_bars=int(getattr(args, "lookahead_bars", None) or LABELING.get("lookahead_bars", 30)),
                sharpe_non_overlapping=bool(getattr(args, "sharpe_non_overlapping", True)),
                return_per_trade_sharpe=bool(getattr(args, "sharpe_per_trade", True)),
                direction_only=_direction_warmup_active,
                tx_cost_bps=float(LABELING.get("transaction_cost_pips", 1.5)) * 4.0,  # ~6 bps round-trip for 1.5-pip spread
                pip_size=float(LABELING.get("pip_size", 0.0001)),
                honest_ctx=_honest_ctx,
            )
            _class_counts = getattr(validate_epoch, "last_class_counts", {"pred": [0, 0, 0], "true": [0, 0, 0]})
            _cost_sharpe_val = getattr(validate_epoch, "last_cost_sharpe", None)
            if _cost_sharpe_val is not None and _tb_writer is not None:
                _tb_writer.add_scalar("Metrics/cost_aware_sharpe", _cost_sharpe_val, ep)
            if _cost_sharpe_val is not None and WANDB and run:
                _safe_wandb_log(run, {"val/cost_aware_sharpe": _cost_sharpe_val})

        except Exception as _val_exc:
            val_pbar.close()
            _log_error(f"[Train] Epoch {ep + 1} validation failed for {model_name}", _val_exc)
            _save_crash_ckpt(ep, _val_exc)
            raise
        val_pbar.close()

        # TrainingController: detect overfit / Sharpe collapse and act
        _ctrl_resp = _train_ctrl.evaluate_epoch(ep + 1, float(tl), float(vl), float(v_sh), dir_acc=float(da))
        _ctrl_curriculum = {"seq_frozen": _seq_frozen}
        _ctrl_applied = _train_ctrl.apply_responses(
            _ctrl_resp,
            model=model,
            optimizer=opt,
            scheduler=scheduler,
            curriculum_state=_ctrl_curriculum,
            epoch=ep + 1,
        )
        if _ctrl_applied.get("hold_curriculum"):
            _seq_frozen = True
        if _ctrl_applied.get("stop_early"):
            _ctrl_stop_early = True
            _ctrl_stop_counter += 1
            print(f"[TrainingController] Early-stop flag set at epoch {ep + 1} (streak {_ctrl_stop_counter})")
            if _ctrl_stop_counter >= 3:
                print(f"[TrainingController] Hard early-stop after 3 consecutive collapses at epoch {ep + 1}")
                break
        else:
            _ctrl_stop_counter = 0

        # ── Online miner: end epoch (update forgetting tracker) ──────────
        if _online_miner is not None:
            _online_miner.end_epoch()
        # ──────────────────────────────────────────────────────────────────

        # ── SI: end epoch (update parameter importance) ───────────────────
        if _si is not None:
            _si.update_omega()
        # ──────────────────────────────────────────────────────────────────

        # SWA: accumulate averaged weights; use constant SWA LR instead of OneCycleLR
        if _swa_enabled and ep >= _swa_start_ep:
            if not _swa_started:
                try:
                    from torch.optim.swa_utils import SWALR, AveragedModel

                    _swa_model = AveragedModel(_core_model(model)).to(device)  # type: ignore
                    _swa_scheduler = SWALR(
                        opt, swa_lr=_swa_lr, anneal_epochs=max(1, int(args.epochs * 0.05)), anneal_strategy="cos"
                    )
                    _swa_started = True
                    print(f"\n[SWA] Weight averaging started at epoch {ep + 1}")
                except Exception as _e:
                    print(f"\n[SWA] Failed to initialize at epoch {ep + 1}: {_e}")
                    _swa_enabled = False

            if _swa_enabled:
                _swa_model.update_parameters(model)
                _swa_scheduler.step()

        if _ema_model is None:
            _ema_model = copy.deepcopy(_core_model(model)).to(device)
        else:
            ExponentialMovingAverage.update_module(_core_model(model), _ema_model, alpha=0.99)

        lr = opt.param_groups[0]["lr"]
        el = time.time() - t0
        gm = torch.cuda.max_memory_allocated(device) // 1_000_000 if device.type == "cuda" else 0
        torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["dir_acc"].append(da)
        history["lr"].append(lr)
        history["val_sharpe"].append(v_sh)
        history.setdefault("cost_aware_sharpe", []).append(float(_cost_sharpe_val) if _cost_sharpe_val is not None else 0.0)
        _hm_ci = getattr(validate_epoch, "last_honest", None) or {}
        history.setdefault("honest_sharpe_ci_low", []).append(float(_hm_ci.get("sharpe_net_ci_low", 0.0)))
        history.setdefault("honest_sharpe_ci_high", []).append(float(_hm_ci.get("sharpe_net_ci_high", 0.0)))
        history.setdefault("honest_n_trades", []).append(int(_hm_ci.get("n_trades", 0)))
        history.setdefault("val_pred_counts", []).append([int(x) for x in _class_counts.get("pred", [0, 0, 0])])
        history.setdefault("val_true_counts", []).append([int(x) for x in _class_counts.get("true", [0, 0, 0])])

        # GPU temp for display
        try:
            import pynvml as _pnvml

            _pnvml.nvmlInit()
            _h = _pnvml.nvmlDeviceGetHandleByIndex(0)
            _gpu_temp = int(_pnvml.nvmlDeviceGetTemperature(_h, _pnvml.NVML_TEMPERATURE_GPU))
        except Exception:
            _gpu_temp = -1

        _ep_metrics = {
            "train_loss": tl,
            "val_loss": vl,
            "dir_acc": da,
            "val_sharpe": v_sh,
            "cost_aware_sharpe": float(_cost_sharpe_val) if _cost_sharpe_val is not None else 0.0,
            "lr": lr,
            "gpu_mb": gm,
            "val_pred_counts": _class_counts.get("pred", [0, 0, 0]),
            "val_true_counts": _class_counts.get("true", [0, 0, 0]),
            "gpu_temp_c": _gpu_temp,
            "oom_skips": getattr(_TRAIN_LOGGER, "_ep_oom_count", 0) if _TRAIN_LOGGER else 0,
            "nan_skips": getattr(_TRAIN_LOGGER, "_ep_nan_count", 0) if _TRAIN_LOGGER else 0,
            "elapsed_s": el,
        }

        if _TRAIN_LOGGER is not None:
            _TRAIN_LOGGER.on_epoch_end(ep, _ep_metrics)

        _gate_ep = start_ep + max(0, int(getattr(args, "direction_warmup_epochs", 2))) - 1
        if multitask and ep == max(start_ep, _gate_ep):
            _class_diag = getattr(
                validate_epoch,
                "last_class_diag",
                {
                    "pred": _class_counts.get("pred", [0, 0, 0]),
                    "true": _class_counts.get("true", [0, 0, 0]),
                },
            )
            from training.direction_control import _direction_gate_failed

            _failed, _reason = _direction_gate_failed(_class_diag, args)
            if _failed:
                _diag_path = _write_class_balance_failure(_run_name, model_name, ep, _class_diag, _reason)
                _msg = (
                    "[ClassBalance] Direction warmup gate failed: "
                    f"{_reason}. pred S/H/B={_class_diag.get('pred')}, "
                    f"pred_shares={[round(float(s), 4) for s in _class_diag.get('pred_shares', [])]}, "
                    f"true S/H/B={_class_diag.get('true')}, "
                    f"recall={[round(float(r), 4) for r in _class_diag.get('recall', [])]}. "
                    f"Diagnostics -> {_diag_path}"
                )
                if getattr(args, "ignore_preflight", False) or getattr(args, "quick_mode", False) or getattr(args, "direction_warmup_soft_gate", True):
                    print(f"{_msg} (continuing under warmup adaptation)")
                else:
                    raise RuntimeError(_msg)

        try:
            from infrastructure.ollama_helper import ollama

            ollama.monitor_training(ep + 1, float(tl), float(vl), _ep_metrics)
        except Exception as _oe:
            pass

        # -- TensorBoard -------------------------------------------------------
        if _tb_writer is not None:
            _tb_writer.add_scalar("Loss/train", tl, ep)
            _tb_writer.add_scalar("Loss/val", vl, ep)
            _tb_writer.add_scalar("Metrics/dir_acc", da, ep)
            _tb_writer.add_scalar("Metrics/sharpe", v_sh, ep)
            _tb_writer.add_scalar("ValPred/sell", _class_counts.get("pred", [0, 0, 0])[0], ep)

            _tb_writer.add_scalar("ValPred/hold", _class_counts.get("pred", [0, 0, 0])[1], ep)

            _tb_writer.add_scalar("ValPred/buy", _class_counts.get("pred", [0, 0, 0])[2], ep)

            _tb_writer.add_scalar("Train/lr", lr, ep)
            _tb_writer.add_scalar("GPU/mem_mb", gm, ep)
            if _gpu_temp > 0:
                _tb_writer.add_scalar("GPU/temp_c", _gpu_temp, ep)
            if fold_id is not None:
                _tb_writer.add_scalar(f"Fold{fold_id}/val_sharpe", v_sh, ep)

        # -- W&B --------------------------------------------------------------
        if WANDB and run:
            _pred_counts = _class_counts.get("pred", [0, 0, 0])
            _true_counts = _class_counts.get("true", [0, 0, 0])
            _safe_wandb_log(
                run,
                {
                    "train/loss": tl,
                    "val/loss": vl,
                    "val/dir_acc": da,
                    "val/sharpe_proxy": v_sh,
                    "val/cost_aware_sharpe": float(_cost_sharpe_val) if _cost_sharpe_val is not None else 0.0,
                    "train/lr": lr,
                    "gpu_mb": gm,
                    "epoch": ep,
                    "val_pred/sell": _pred_counts[0],
                    "val_pred/hold": _pred_counts[1],
                    "val_pred/buy": _pred_counts[2],
                    "val_true/sell": _true_counts[0],
                    "val_true/hold": _true_counts[1],
                    "val_true/buy": _true_counts[2],
                    **({"fold": fold_id} if fold_id is not None else {}),
                },
            )

        # -- Is best? (SACS-aware) ------------------------------------------------
        # Compute sharpness = mean val-loss increase over N random ε-ball
        # perturbations.  Robust score = val_loss + λ * sharpness.
        # Lower robust score → flatter minimum → prefer this checkpoint.
        # Falls back to plain val_loss when SACS is disabled or fails.
        _sacs_score = vl  # default: no sharpness penalty
        # S7: when the honest metric exists, select on its 95% CI lower bound
        # (lower score = better, so negate). Validation loss is a poor proxy
        # for trade quality with heavy-tailed targets.
        _hm_sel = getattr(validate_epoch, "last_honest", None) or {}
        _select_on_honest = bool(_hm_sel) and int(_hm_sel.get("n_trades", 0)) >= 30
        if _sacs_enabled:
            try:
                _core_now = _core_model(model)
                _sharpness_acc = 0.0
                for _ in range(_sacs_n):
                    _perturbed = copy.deepcopy(_core_now)
                    with torch.no_grad():
                        for _p in _perturbed.parameters():
                            if _p.requires_grad:
                                _p.add_(torch.randn_like(_p) * _sacs_eps)
                    _vl_p, _, _ = validate_epoch(
                        _perturbed, val_dl, crit, device, classification,
                        pbar=None, amp=False, amp_dtype=torch.float32,
                        seq_len=curr_seq_len, multitask=multitask,
                        feature_mask=_feat_mask,
                    )
                    del _perturbed
                    _sharpness_acc += abs(_vl_p - vl)
                _sharpness = _sharpness_acc / _sacs_n
                _sacs_score = vl + _sacs_lam * _sharpness
            except Exception as _sacs_ep_e:
                _sacs_score = vl
                print(f"[SACS] epoch {ep+1} sharpness eval failed (non-fatal): {_sacs_ep_e}")

        # Direction-warmup epochs score a direction-only loss on a class-balanced
        # subset, which is lower than and not comparable with the full multitask
        # loss. Before this guard every fold's "best" was a warmup epoch (0 or 1).
        # Keep a warmup best only as a placeholder; the first full epoch replaces it.
        if _best_from_warmup and not _direction_warmup_active:
            _best_sacs_score = float("inf")
            _best_from_warmup = False
        if _select_on_honest and not _sacs_enabled:
            _sacs_score = -float(_hm_sel.get("sharpe_net_ci_low", 0.0))
        improved = _sacs_score < _best_sacs_score
        if getattr(args, "_refit_fixed_epochs", False):
            # Final refit on all pre-holdout data: epoch count fixed from CV, the
            # "validation" rows are inside training, so keep the last epoch.
            improved = True
        if improved:
            _best_from_warmup = _direction_warmup_active

        if improved:
            best_sharpe = v_sh
            best_val_loss = vl
            _best_sacs_score = _sacs_score
            if _cost_sharpe_val is not None:
                best_cost_sharpe = _cost_sharpe_val

        # Suppress patience counter during LR warmup: the LR is artificially
        # suppressed so Sharpe cannot improve consistently. Only start counting
        # after warmup_epochs have fully elapsed.
        _warmup_done = ep >= int(getattr(args, "lr_warmup_epochs", 0))

        # ── Dynamic Early Stop ────────────────────────────────────────────────
        if _des_base_patience > 0 and _warmup_done and not getattr(args, "_refit_fixed_epochs", False):
            # Fix-1: Curriculum stage advance resets counter — a difficulty jump
            # causes a temporary val_loss spike that shouldn't trigger early stop.
            _cur_difficulty = int(
                history.get("difficulty_stage", [-1])[-1]
                if history.get("difficulty_stage") else -1
            )
            if _cur_difficulty > _des_prev_difficulty and _des_prev_difficulty >= 0:
                _des_no_improve = 0
                _des_ema = None  # reset EMA so new difficulty baseline is clean
                print(f"[DynStop] Curriculum advanced to level {_cur_difficulty} — resetting patience counter")
            _des_prev_difficulty = _cur_difficulty

            _sharpe_signal = v_sh if v_sh is not None and not (v_sh != v_sh) else 0.0
            # Use SACS score as the base (already sharpness-penalised) so dynamic stop
            # and SACS checkpoint selection agree on what "better" means. Fall back to
            # vl when SACS is disabled.
            _des_base_score = _sacs_score if (_sacs_enabled or _select_on_honest) else vl
            # No SI/EWC credit: subtracting a term that grows with the epoch count made
            # the composite "improve" on its own, so patience never ran out. When
            # selecting on the honest CI bound the score already is the Sharpe signal.
            _composite = _des_base_score if _select_on_honest else _des_base_score - 0.1 * _sharpe_signal

            # EMA-smooth the composite (reduces single-epoch noise)
            if _des_ema is None:
                _des_ema = _composite
            else:
                _des_ema = _des_ema_alpha * _composite + (1.0 - _des_ema_alpha) * _des_ema

            # Adaptive patience: grows linearly with training progress
            _progress = (ep + 1) / max(1, args.epochs)
            _adaptive_patience = int(_des_base_patience * (0.5 + _progress))  # 50%→150% of base

            # Check EMA improvement
            if _des_ema < _des_best_ema - 1e-6:
                _des_best_ema = _des_ema
                _des_no_improve = 0
            else:
                _des_no_improve += 1

            # Fix-3: SWA guard — don't stop before SWA has had a chance to run.
            # Don't hold a plateaued run open until SWA starts (epoch 30 of 40 by
            # default): that guard meant early stopping never fired. A run that
            # stops before SWA simply keeps its best checkpoint.
            _swa_guard_ok = True

            if _des_no_improve >= _adaptive_patience and _swa_guard_ok:
                _cur_lr = opt.param_groups[0]["lr"]
                if not _des_lr_halved and _cur_lr > _des_lr_min * 2:
                    # First plateau: halve LR, reset counter, give another chance
                    _new_lr = max(_cur_lr * 0.5, _des_lr_min)
                    for _pg in opt.param_groups:
                        _pg["lr"] = _new_lr
                    _des_lr_halved = True
                    _des_no_improve = 0
                    print(
                        f"\n[DynStop] Plateau at epoch {ep+1} | "
                        f"LR {_cur_lr:.2e} → {_new_lr:.2e} | "
                        f"patience={_adaptive_patience} | giving {_adaptive_patience} more epochs"
                    )
                else:
                    # Second plateau (or LR already at floor): stop
                    print(
                        f"\n[DynStop] Stopping at epoch {ep+1}/{args.epochs} | "
                        f"no EMA improvement for {_des_no_improve} epochs | "
                        f"composite_ema={_des_ema:.6f} | best={_des_best_ema:.6f}"
                    )
                    break
            elif _des_no_improve >= _adaptive_patience and not _swa_guard_ok:
                print(
                    f"[DynStop] Plateau at epoch {ep+1} but deferring stop — "
                    f"waiting for SWA to start at epoch {_swa_start_ep}"
                )

        # -- Rich display or plain print ---------------------------------------
        if _rich_display is not None:
            _rich_display.end_epoch(ep, _ep_metrics, is_best=improved)
        else:
            _cs_str = f"{_cost_sharpe_val:>9.4f}" if _cost_sharpe_val is not None else "  N/A     "
            print(f"{ep + 1:>5} {tl:>11.6f} {vl:>11.6f} {da:>8.4f} {v_sh:>9.4f} {_cs_str} {lr:>10.2e} {el:>6.1f}s {gm:>7.0f}M")

        if improved:
            core = _core_model(model)
            ckpt_meta = {
                "model_name": model_name,
                "n_features": n_features,
                "seq_len": int(curr_seq_len),
                "schema_hash": _resolve_schema_hash(args),
                "timestamp": datetime.now(UTC).isoformat(),
                "fold_id": fold_id,
            }
            _safe_save(core.state_dict(), best_path, metadata=ckpt_meta)
            with open(cfg_path, "w", encoding="utf-8") as _cfg_fp:
                json.dump(
                    {
                        "model": model_name,
                        "n_features": n_features,
                        "seq_len": args.seq_len,
                        "d_model": args.d_model,
                        "nhead": args.nhead,
                        "hidden_size": args.hidden_size,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "best_val_loss": vl,
                        "best_val_sharpe_proxy": v_sh,
                        "best_cost_aware_sharpe": float(_cost_sharpe_val) if _cost_sharpe_val is not None else 0.0,
                        "best_metric": float(_cost_sharpe_val) if stop_on_cost_sharpe and _cost_sharpe_val is not None else (float(v_sh) if stop_on_sharpe else float(vl)),
                        "best_metric_name": "cost_sharpe" if stop_on_cost_sharpe else ("val_sharpe" if stop_on_sharpe else "val_loss"),
                        "best_train_loss": tl,
                        "train_val_loss_gap": float(vl - tl),
                        "epoch": ep,
                        "n_samples": n_samples,
                        "loss": args.loss,
                        "fold_id": fold_id,
                        # T4 provenance: what this checkpoint was trained/validated on.
                        **_fold_provenance(cache_path, train_idx, val_idx, args),
                        # T5: v_sh is the honest net Sharpe when price arrays exist
                        # (S9); its CI lower bound is the selection statistic.
                        "val_sharpe_is_honest": bool(getattr(validate_epoch, "last_honest", None)),
                        "honest_sharpe_ci_low": float(
                            (getattr(validate_epoch, "last_honest", None) or {}).get("sharpe_net_ci_low", 0.0)
                        ),
                        "refit": bool(getattr(args, "_refit_fixed_epochs", False)),
                    },
                    _cfg_fp,
                    indent=2,
                )
            _safe_wandb_summary_update(
                run,
                {
                    "best_val_loss": vl,
                    "best_val_sharpe_proxy": v_sh,
                    "best_epoch": ep,
                },
            )

        if (ep + 1) % args.save_every == 0:
            core = _core_model(model)
            ep_tag = f"{fold_suffix}_ep{ep + 1}" if fold_suffix else f"_ep{ep + 1}"
            _safe_save(
                core.state_dict(),
                ckpt_dir / f"{model_name}{ep_tag}.pt",
                metadata={
                    "model_name": model_name,
                    "n_features": n_features,
                    "seq_len": int(curr_seq_len),
                    "schema_hash": _resolve_schema_hash(args),
                    "epoch": ep + 1,
                    "fold_id": fold_id,
                },
            )

        # Exact resume checkpoint (model + optimizer + scheduler + AMP scaler + history)
        core = _core_model(model)
        _safe_save(
            {
                "epoch": ep,
                "model_state": core.state_dict(),
                "opt_state": opt.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": amp_sc.state_dict() if args.amp and device.type == "cuda" else None,
                "best_val_loss": best_val_loss,
                "best_sharpe": best_sharpe,
                "no_improve": _des_no_improve,
                "des_ema": _des_ema,
                "des_best_ema": _des_best_ema,
                "des_lr_halved": _des_lr_halved,
                "des_prev_difficulty": _des_prev_difficulty,
                "history": history,
                "fold_id": fold_id,
                "chunk_sharpe_history": _chunk_sharpe_history,
                "chunk_worse_streak": _chunk_worse_streak,
                "feat_stability_state": _feat_stability.get_state(),
            },
            last_path,
            metadata={
                "model_name": model_name,
                "n_features": n_features,
                "seq_len": int(curr_seq_len),
                "schema_hash": _resolve_schema_hash(args),
                "epoch": ep + 1,
                "fold_id": fold_id,
                "checkpoint_type": "resume",
            },
        )



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
        print(f"[SWA] Averaged model saved  {_swa_path}")

    # -- D: Post-training temperature calibration ------------------------------
    if getattr(args, "calibrate", False):
        print("\n[Calibration] Fitting temperature scaler on val set...")
        try:
            # Reload best weights before calibrating
            core = _core_model(model)
            core.load_state_dict(torch.load(best_path, map_location=device))
            cal_model = TemperatureScaler(core).to(device)  # type: ignore
            calibrate_as_classification = bool(classification)
            # Use tune_idx to prevent calibration leakage if available
            if getattr(args, "_tune_eval_idx", None) is not None:
                cal_ds = ZarrStreamDataset(
                    cache_path, args._tune_eval_idx, shuffle_chunks=False, multitask_targets=multitask,
                    scaler=_scaler, pair_targets=_pair_targets,  # calibrate on the training transform
                )
                cal_dl = DataLoader(cal_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
                cal_model.calibrate(cal_dl, device, classification=calibrate_as_classification)
            else:
                print("[Warning] No tune_idx found. Calibrating on val_dl (Data Leakage Risk!).")
                cal_model.calibrate(val_dl, device, classification=calibrate_as_classification)

            cal_path = ckpt_dir / f"{model_name}{fold_suffix}_calibrated.pt"
            _safe_save(
                {
                    "model_state": core.state_dict(),
                    "temperature": cal_model.temperature.item(),
                    "classification": calibrate_as_classification,
                },
                cal_path,
            )
            print(f"[Calibration] Saved calibrated model -> {cal_path}")

            # INF-004: Emit calibration metadata sidecar
            try:
                _cal_temp = float(cal_model.temperature.item())
                _cal_report = {
                    "temperature": _cal_temp,
                    "classification": calibrate_as_classification,
                    "calibration_set_size": len(val_dl.dataset) if hasattr(val_dl, "dataset") else "unknown",
                    "model_name": model_name,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                _cal_report_path = ckpt_dir / "calibration_report.json"
                _safe_save_json(_cal_report, _cal_report_path)
                print(f"[Calibration] Report -> {_cal_report_path} (T={_cal_temp:.4f})")
            except Exception as _cr_e:
                print(f"[Calibration] Report write failed (non-fatal): {_cr_e}")

        except Exception as _cal_e:
            _log_warn(f"[Calibration] Failed: {_cal_e}")

    if history.get("cost_aware_sharpe"):
        _has_cost_hist = True
    else:
        _has_cost_hist = False

    if history["val_sharpe"]:
        if stop_on_cost_sharpe and _has_cost_hist:
            _best_ep = int(history["cost_aware_sharpe"].index(max(history["cost_aware_sharpe"])))
        elif stop_on_sharpe:
            _best_ep = int(history["val_sharpe"].index(max(history["val_sharpe"])))
        else:
            _best_ep = int(history["val_loss"].index(min(history["val_loss"])))
    else:
        _best_ep = 0
    _best_met = best_cost_sharpe if stop_on_cost_sharpe else (best_sharpe if stop_on_sharpe else best_val_loss)

    if _TRAIN_LOGGER is not None:
        _TRAIN_LOGGER.on_training_complete(
            best_epoch=_best_ep,
            best_metric=_best_met,
            total_s=time.time() - _t_start,
            metric_name="cost_sharpe" if stop_on_cost_sharpe else ("val_sharpe" if stop_on_sharpe else "val_loss"),
        )

    if _rich_display is not None:
        _rich_display.finish(best_epoch=_best_ep, best_metric=_best_met)
        _rich_display.__exit__(None, None, None)

    if _tb_writer is not None:
        _tb_writer.add_hparams(
            hparam_dict={
                "lr": args.lr,
                "batch_size": args.batch_size,
                "d_model": args.d_model,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "loss": args.loss,
            },
            metric_dict={
                "hparam/best_val_sharpe": best_sharpe,
                "hparam/best_val_loss": best_val_loss,
                "hparam/best_epoch": float(_best_ep),
            },
        )
        _tb_writer.flush()
        _tb_writer.close()

    # -- Generate Training Control Report ---------------------------------------
    _control_report_path = ckpt_dir / f"{model_name}{fold_suffix}_training_control_report.json"
    _final_train_val_gap = (
        history["val_loss"][-1] - history["train_loss"][-1]
        if history.get("val_loss") and history.get("train_loss")
        else 0.0
    )
    _early_stopped = bool(_ctrl_stop_early and _ctrl_stop_counter >= 3)
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
        },
        "final_train_val_gap": _final_train_val_gap,
        "early_stopped": _early_stopped,
        "overfitting_warnings": [],
        "final_seq_len": int(_active_seq_len),
    }
    if _final_train_val_gap > 0.05:
        _control_report["overfitting_warnings"].append(f"High train-val gap: {_final_train_val_gap:.4f}")
    if len(history.get("val_sharpe", [])) > 5:
        max_sh = max(history["val_sharpe"])
        final_sh = history["val_sharpe"][-1]
        if max_sh - final_sh > 0.3:
            _control_report["overfitting_warnings"].append(f"Sharpe collapsed by {max_sh - final_sh:.3f} from peak")
    _control_report["controller_signals"] = list(_train_ctrl.report_data.get("overfitting_signals_detected") or [])
    _control_report["controller_actions"] = list(_train_ctrl.report_data.get("actions_applied") or [])

    # Restore best-epoch weights into the live model before finalize / return
    _restored_best = False
    if best_path.exists():
        try:
            core = _core_model(model)
            core.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
            _restored_best = True
            print(f"[Train] Restored best-epoch weights from {best_path}")
        except Exception as _rb_exc:
            try:
                core = _core_model(model)
                core.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
                _restored_best = True
                print(f"[Train] Restored best-epoch weights from {best_path}")
            except Exception as _rb2:
                print(f"[Train] WARN: could not restore best weights: {_rb_exc}; {_rb2}")

    _control_report["restore_decision"] = bool(_restored_best)

    try:
        _safe_save_json(_control_report, _control_report_path)
        print(f"[TrainingControl] Saved report -> {_control_report_path}")
    except Exception as e:
        print(f"[TrainingControl] Failed to save report: {e}")

    try:
        _train_ctrl.finalize_training(
            best_epoch=int(_best_ep),
            promoted=False,
            restored_best=_restored_best,
        )
    except Exception as _tc_fin:
        print(f"[TrainingController] finalize skipped: {_tc_fin}")

    # -- SACS: Sharpness-Aware Checkpoint Selection -----------------------------
    print("\n[SACS] Running Sharpness-Aware Checkpoint Selection...")
    sacs_candidates = {"Active": model}
    if _swa_enabled and _swa_started and _swa_model is not None:
        sacs_candidates["SWA"] = _swa_model.module
    if _ema_model is not None:
        sacs_candidates["EMA"] = _ema_model

    best_sacs_score = -float("inf") if (stop_on_sharpe or stop_on_cost_sharpe) else float("inf")
    best_sacs_name = None
    best_sacs_state = None

    try:
        from training.train_gpu import _sharpe_ann_factor
        _sacs_ann = _sharpe_ann_factor(args)
    except Exception:
        _sacs_ann = 1.0

    _pt_sacs_eps = float(getattr(args, "sacs_eps", 0.005))
    _pt_sacs_n = max(1, int(getattr(args, "sacs_n_samples", 5)))
    _pt_sacs_lam = float(getattr(args, "sacs_sharpness_weight", 1.0))
    _lookahead = int(getattr(args, "lookahead_bars", None) or LABELING.get("lookahead_bars", 30))
    _tx_cost = float(LABELING.get("transaction_cost_pips", 1.5)) * 4.0
    _pip = float(LABELING.get("pip_size", 0.0001))

    for cand_name, cand_model in sacs_candidates.items():
        try:
            cand_core = _core_model(cand_model)

            vl_c, da_c, v_sh_c = validate_epoch(
                cand_core, val_dl, crit, device, classification,
                pbar=None, amp=False, amp_dtype=torch.float32, seq_len=curr_seq_len, multitask=multitask,
                feature_mask=_feat_mask, sharpe_ann_factor=_sacs_ann,
                lookahead_bars=_lookahead,
                sharpe_non_overlapping=bool(getattr(args, "sharpe_non_overlapping", True)),
                return_per_trade_sharpe=bool(getattr(args, "sharpe_per_trade", True)),
                direction_only=False, tx_cost_bps=_tx_cost, pip_size=_pip, honest_ctx=_honest_ctx,
            )
            c_cost = getattr(validate_epoch, "last_cost_sharpe", None)

            # Sharpness = mean metric change over N ε-ball perturbations.
            # For loss-based selection: sharpness is how much loss rises.
            # For Sharpe-based selection: sharpness is how much Sharpe drops.
            sharpness_acc = 0.0
            for _ in range(_pt_sacs_n):
                _perturbed = copy.deepcopy(cand_core).to(device)
                with torch.no_grad():
                    for _pm in _perturbed.parameters():
                        if _pm.requires_grad:
                            _pm.add_(torch.randn_like(_pm) * _pt_sacs_eps)
                vl_p, _, v_sh_p = validate_epoch(
                    _perturbed, val_dl, crit, device, classification,
                    pbar=None, amp=False, amp_dtype=torch.float32, seq_len=curr_seq_len, multitask=multitask,
                    feature_mask=_feat_mask, sharpe_ann_factor=_sacs_ann,
                    lookahead_bars=_lookahead,
                    sharpe_non_overlapping=bool(getattr(args, "sharpe_non_overlapping", True)),
                    return_per_trade_sharpe=bool(getattr(args, "sharpe_per_trade", True)),
                    direction_only=False, tx_cost_bps=_tx_cost, pip_size=_pip, honest_ctx=_honest_ctx,
                )
                p_cost_i = getattr(validate_epoch, "last_cost_sharpe", None)
                if stop_on_cost_sharpe and c_cost is not None and p_cost_i is not None:
                    sharpness_acc += abs(float(p_cost_i) - float(c_cost))
                elif stop_on_sharpe:
                    sharpness_acc += abs(float(v_sh_p) - float(v_sh_c))
                else:
                    sharpness_acc += abs(float(vl_p) - float(vl_c))
                del _perturbed
            sharpness = sharpness_acc / _pt_sacs_n

            # Robust score: lower = better for loss; higher = better for Sharpe.
            # Sharpness penalty always reduces quality → subtract for Sharpe, add for loss.
            if stop_on_cost_sharpe and c_cost is not None:
                c_val = float(c_cost)
                robust_score = c_val - _pt_sacs_lam * sharpness
                is_better = robust_score > best_sacs_score
            elif stop_on_sharpe:
                c_val = float(v_sh_c)
                robust_score = c_val - _pt_sacs_lam * sharpness
                is_better = robust_score > best_sacs_score
            else:
                c_val = float(vl_c)
                robust_score = c_val + _pt_sacs_lam * sharpness
                is_better = robust_score < best_sacs_score

            print(
                f"[SACS] {cand_name}: clean={c_val:.4f}  sharpness={sharpness:.4f}"
                f"  robust={robust_score:.4f}  (λ={_pt_sacs_lam}, n={_pt_sacs_n}, ε={_pt_sacs_eps})"
            )

            if is_better:
                best_sacs_score = robust_score
                best_sacs_name = cand_name
                best_sacs_state = copy.deepcopy(cand_core.state_dict())

        except Exception as _sacs_e:
            print(f"[SACS] Evaluation failed for {cand_name}: {_sacs_e}")
            
    if best_sacs_state is not None:
        print(f"[SACS] Best model: {best_sacs_name} with robust score {best_sacs_score:.4f}. Saving to {best_path}.")
        _safe_save(best_sacs_state, best_path)
        _core_model(model).load_state_dict(best_sacs_state)
    # ---------------------------------------------------------------------------

    if stop_on_cost_sharpe:
        print(f"\n[Train] Best cost-aware Sharpe (after tx costs): {best_cost_sharpe:.4f}  ->  {best_path}")
    elif stop_on_sharpe:
        print(f"\n[Train] Best val Sharpe (proxy): {best_sharpe:.4f}  ->  {best_path}")

        if getattr(args, "ollama_auto_tune", False):
            try:
                from infrastructure.ollama_helper import ollama

                final_metrics = {
                    "best_sharpe": float(best_sharpe),
                    "best_val_loss": float(best_val_loss),
                    "best_epoch": int(_best_ep),
                }
                ollama.auto_tune_model(model_name, final_metrics)
            except Exception:
                pass

        if sidecar is not None:
            try:
                sidecar.stop()
            except Exception:
                pass

        return history, best_sharpe
    print(f"\n[Train] Best val loss: {best_val_loss:.6f}  ->  {best_path}")
    if sidecar is not None:
        try:
            sidecar.stop()
        except Exception:
            pass
    return history, best_val_loss
