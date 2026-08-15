"""Supervised training loop extracted from ``training.train_gpu``.

See ``docs/CONTINUE.md``."""
from __future__ import annotations

import gc
import json
import math
import os
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

try:
    from monitoring.train_logger import TrainingLogger as _TrainingLogger
except Exception:
    _TrainingLogger = None

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD = True
except ImportError:
    TENSORBOARD = False

try:
    from tqdm.auto import tqdm as _pbar
except ImportError:
    def _pbar(it=None, **kw): return it

from config.settings import (
    CURRICULUM as SETTINGS_CURRICULUM,
)
from config.settings import (
    PATHS,
    PRETRAIN,
    TRAINING,
)
from models.architectures import (
    MODEL_ROLES,
    AsymmetricDirectionalLoss,
    DiversityLoss,
    HuberLoss,
    MultiTaskLoss,
    OverconfidencePenalty,
    TemperatureScaler,
)
from training.gpu_datasets import (
    ZarrStreamDataset,
)
from training.gpu_device import build_adamw, maybe_torch_compile
from training.ewc import ElasticWeightConsolidation, apply_ewc_loss
from training.synaptic_intelligence import (
    SynapticIntelligence,
    apply_si_loss,
)
from training.model_factory import get_model_training_profile
from training.gpu_losses import (
    DirectionalHuberLoss,
    SharpeProxyLoss,
    _match_target_shape,
)

_HOST = None
_BOUND = False
_OWNED_GLOBALS = frozenset({
    "_TRAIN_LOGGER",
    "_OVERCONF_PENALTY",
})
_HOST_DEPS = (
    '_log_error',
    '_log_warn',
    '_log_info',
    '_log_oom',
    '_log_nan',
    '_crop_to_seq_len',
    '_thermal_check',
    '_direction_class_index',
    '_direction_recall_from_confusion',
    '_gradients_are_finite',
    '_recover_nonfinite_training_state',
    '_class_prior_tensor',
    '_class_prior_array',
    '_class_weights_tensor',
    '_sharpe_ann_factor',
    '_strict_load_report',
    '_core_model',
    '_apply_model_profile',
    '_balanced_direction_indices',
    '_direction_preflight',
    '_direction_probe',
    '_embargo_bars',
    '_embargo_split',
    '_purge_bars',
    '_three_way_split',
    '_validation_method',
    '_promotion_holdout_n',
    '_load_diff_array',
    '_load_feature_schema',
    '_init_multitask_direction_bias',
    '_is_uninitialized_parameter',
    '_write_class_balance_failure',
    '_slug_part',
    '_safe_save',
    '_safe_save_json',
    '_safe_wandb_log',
    '_safe_wandb_summary_update',
    '_update_pretrain_report',
    '_feature_ablation_config',
    '_build_feature_ablation_mask',
    '_model_build_args',
    '_on_disk_sequence_count',
    'build_model',
    'DirectionalHuberLoss',
    'SharpeProxyLoss',
    '_match_target_shape',
    'ZarrStreamDataset',
    '_ThreadPrefetchLoader',
    'wrap_loader_prefetch',
    'MemmapSequenceDataset',
    'HuberLoss',
    'AsymmetricDirectionalLoss',
    'MultiTaskLoss',
    'OverconfidencePenalty',
    'TemperatureScaler',
    'DiversityLoss',
    'PATHS',
    'LABELING',
    '_TRAIN_LOGGER',
    '_TRAIN_LOGGER_AVAILABLE',
    'Sidecar',
    'ElasticWeightConsolidation',
    'apply_ewc_loss',
    'SynapticIntelligence',
    'apply_si_loss',
    'PrioritizedDataLoader',
    'run_preflight_sanity_checks',
    'CurriculumController',
    'create_curriculum_manager',
    'WANDB',
    'OPTUNA',
    'RICH_DISPLAY',
    '_GPU_CFG',
    'maybe_torch_compile',
)


def bind_host(host_mod) -> None:
    """Copy still-monolith helpers into this module's globals (cycle-safe)."""
    global _HOST, _BOUND
    _HOST = host_mod
    g = globals()
    for name in _HOST_DEPS:
        if name in _OWNED_GLOBALS:
            continue
        if hasattr(host_mod, name):
            g[name] = getattr(host_mod, name)
    # Pull owned singletons from host only when local is still unset
    for name in _OWNED_GLOBALS:
        if g.get(name) is None and hasattr(host_mod, name):
            g[name] = getattr(host_mod, name)
    _BOUND = True


def _sync_owned_to_host(*names: str) -> None:
    if _HOST is None:
        return
    g = globals()
    for name in names:
        setattr(_HOST, name, g.get(name))


def _ensure_bound() -> None:
    """Refresh host bindings without re-importing if train_gpu is mid-init."""
    import sys
    tg = sys.modules.get("training.train_gpu")
    if tg is None:
        import training.train_gpu as tg
    bind_host(tg)



_OVERCONF_PENALTY: OverconfidencePenalty | None = None  # D: set in supervised_train

# -----------------------------------------------------------------------------
# A: FEATURE STABILITY + TRAIN/VALIDATE + SUPERVISED_TRAIN
# -----------------------------------------------------------------------------

# TRAINING LOOP
# -----------------------------------------------------------------------------

_OVERCONF_PENALTY: OverconfidencePenalty | None = None  # D: set in supervised_train


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

    def get_mask(self, device=None) -> torch.Tensor:
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
    device:         torch.device,
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
    _ensure_bound()
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
    opt = build_adamw(all_params, lr=lr, weight_decay=1e-4)

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

    for ep in range(epochs):
        ep_task_loss = 0.0
        ep_div_loss  = 0.0
        n_batches    = 0

        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            xb, yb, y_cls_b, y_conf_b, _ = _unpack_batch(batch, device)
            xb, yb, y_cls_b, y_conf_b, keep = _sanitize_batch_tensors(xb, yb, y_cls_b, y_conf_b)
            if keep is not None and not bool(keep.all()):
                if not bool(keep.any()):
                    continue
                xb, yb = xb[keep], yb[keep]
                if y_cls_b is not None:
                    y_cls_b = y_cls_b[keep]
                if y_conf_b is not None:
                    y_conf_b = y_conf_b[keep]
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
            _centralize_gradients(all_params)
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
    y_cls: torch.Tensor | None = None,
    y_conf: torch.Tensor | None = None,
    multitask: bool = False,
    direction_only: bool = False,
) -> torch.Tensor:
    """
    Unified loss for single-head and MultiTaskWrapper outputs.

    ``direction_only=True`` (direction warmup / probe) always uses class-index
    CE against the direction head — never MultiTaskLoss's full (dir+ret+conf)
    signature, even when ``multitask=True``.
    """
    if isinstance(model_out, tuple):
        logits, ret_hat, conf = model_out
        y_cls_idx = _direction_class_index(
            yb, y_cls, classification=classification or direction_only,
        )
        if direction_only:
            # Warmup/probe: CE on direction logits only.
            # MultiTaskLoss.forward expects (logits, ret, conf, y_cls, y_cont, ...);
            # calling it with 2 args would TypeError or mis-bind.
            if isinstance(crit, MultiTaskLoss):
                # MultiTaskLoss.ce uses reduction="none" — mean for a scalar loss.
                return crit.ce(logits, y_cls_idx.reshape(-1).clamp(0, 2)).mean()
            try:
                return crit(logits, y_cls_idx)
            except TypeError:
                # Fallback: treat as multitask-shaped criterion
                y_cont = _match_target_shape(ret_hat, yb)
                return crit(logits, ret_hat, conf, y_cls_idx, y_cont, None)

        if multitask or isinstance(crit, MultiTaskLoss):
            y_cont = _match_target_shape(ret_hat, yb)
            return crit(logits, ret_hat, conf, y_cls_idx, y_cont, y_conf)

        # Tuple output but non-multitask criterion (rare) — direction CE
        if classification:
            return crit(logits, y_cls_idx)
        y_cont = _match_target_shape(ret_hat, yb)
        return crit(ret_hat, y_cont)

    if classification:
        return crit(model_out, _direction_class_index(yb, y_cls))

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


_SANITIZE_STATS: dict[str, int] = {
    "feature_nonfinite": 0,
    "batches_with_feature_clamp": 0,
    "target_rows_dropped": 0,
    "batches_with_target_drops": 0,
}


def _sanitize_batch_tensors(
    xb: torch.Tensor,
    yb: torch.Tensor,
    y_cls: torch.Tensor | None,
    y_conf: torch.Tensor | None,
    *,
    skip_bad_targets: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Sanitize features; surface non-finite targets instead of silently zeroing them.

    Returns ``(xb, yb, y_cls, y_conf, keep_mask)``. ``keep_mask`` is a bool
    vector over the batch dim. Features may be clamped (counted + WARN);
    targets are **never** replaced with 0 — bad rows are flagged for the
    caller to drop (or raise when ``skip_bad_targets=False``).
    """
    global _SANITIZE_STATS
    xb_f = xb.float()
    feat_bad = ~torch.isfinite(xb_f)
    n_feat_bad = int(feat_bad.sum().item())
    if n_feat_bad:
        _SANITIZE_STATS["feature_nonfinite"] += n_feat_bad
        _SANITIZE_STATS["batches_with_feature_clamp"] += 1
        n_clamp_batches = _SANITIZE_STATS["batches_with_feature_clamp"]
        if n_clamp_batches <= 3 or n_clamp_batches % 50 == 0:
            msg = (
                f"[Sanitize] WARN: clamped {n_feat_bad} non-finite feature value(s) "
                f"(batches={n_clamp_batches}, total_vals={_SANITIZE_STATS['feature_nonfinite']})"
            )
            print(msg)
            try:
                _log_warn(msg)
            except Exception:
                pass
    xb = torch.nan_to_num(xb_f, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

    # Targets: detect non-finite rows — never nan_to_num → 0.
    yb_f = yb.float()
    bsz = int(yb_f.shape[0])
    keep = torch.isfinite(yb_f).reshape(bsz, -1).all(dim=-1)

    if y_cls is not None:
        y_cls_f = y_cls.float()
        keep = keep & torch.isfinite(y_cls_f).reshape(bsz, -1).all(dim=-1)
        y_cls = y_cls_f.clamp(-1.0, 1.0)
    if y_conf is not None:
        y_conf_f = y_conf.float()
        keep = keep & torch.isfinite(y_conf_f).reshape(bsz, -1).all(dim=-1)
        y_conf = y_conf_f.clamp(0.0, 1.0)

    if not bool(keep.all()):
        n_bad = int((~keep).sum().item())
        _SANITIZE_STATS["target_rows_dropped"] += n_bad
        _SANITIZE_STATS["batches_with_target_drops"] += 1
        if not skip_bad_targets:
            raise ValueError(
                f"[Sanitize] {n_bad}/{bsz} non-finite target rows (fail-closed; "
                f"not zeroed — fix data or set skip_bad_targets=True to drop)"
            )
        n_drop_batches = _SANITIZE_STATS["batches_with_target_drops"]
        if n_drop_batches <= 3 or n_drop_batches % 50 == 0:
            msg = (
                f"[Sanitize] WARN: dropping {n_bad}/{bsz} rows with non-finite targets "
                f"(not zeroed; batches={n_drop_batches}, "
                f"total_rows={_SANITIZE_STATS['target_rows_dropped']})"
            )
            print(msg)
            try:
                _log_warn(msg)
            except Exception:
                pass
    return xb, yb_f, y_cls, y_conf, keep


def sanitize_stats() -> dict[str, int]:
    """Return a copy of running sanitize counters (features clamped / targets dropped)."""
    return dict(_SANITIZE_STATS)


def reset_sanitize_stats() -> None:
    for key in _SANITIZE_STATS:
        _SANITIZE_STATS[key] = 0


def _apply_online_miner(online_miner, pred, yb, y_cls_b, batch_idx_t, classification, multitask):
    if online_miner is None or batch_idx_t is None:
        return
    try:
        with torch.no_grad():
            pred_flat = pred[0] if isinstance(pred, tuple) else pred
            if classification or multitask:
                y_cls_idx = _direction_class_index(yb, y_cls_b, classification=True)
                per_sample = (pred_flat.argmax(-1) != y_cls_idx).float()
            else:
                per_sample = torch.abs(pred_flat.ravel() - yb.ravel())
            online_miner.update_batch(
                batch_idx_t.detach().cpu().numpy(),
                per_sample.detach().cpu().numpy(),
            )
    except Exception as exc:
        print(f"[Train] online_miner.update_batch failed: {exc}")


def _apply_kd_loss(pred, teacher_model, xb, loss, distill_weight: float):
    if teacher_model is None:
        return loss
    with torch.no_grad():
        t_pred = teacher_model(xb)
    if isinstance(pred, tuple):
        p_out = pred[0]
        t_out = t_pred[0] if isinstance(t_pred, tuple) else t_pred
    else:
        p_out = pred
        t_out = t_pred if not isinstance(t_pred, tuple) else t_pred[0]
    kd_loss = torch.nn.functional.mse_loss(p_out, t_out)
    return (1.0 - distill_weight) * loss + distill_weight * kd_loss


def _centralize_gradients(params) -> None:
    """Apply Gradient Centralization (Yong et al.) to weight grads in-place.

    Subtracts the mean over non-batch dims for every ``dim > 1`` gradient so
    weight updates stay mean-zero. Accepts an ``nn.Module`` or a parameter
    iterable so AMP / non-AMP / diversity-finetune share one hot path.
    """
    iterable = params.parameters() if isinstance(params, nn.Module) else params
    for p in iterable:
        g = p.grad
        if g is None or g.dim() <= 1:
            continue
        g.sub_(g.mean(dim=tuple(range(1, g.dim())), keepdim=True))


def _maybe_warn_grad_norm(model, batch_idx: int) -> None:
    if batch_idx % 100 != 0:
        return
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.detach().data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    if total_norm > 50.0:
        print(f"[Stability] WARNING: High grad norm ({total_norm:.2f}) at batch {batch_idx}")


class _CurriculumProviderConfig:
    """Minimal config facade for providers without a dataclass ``config``.

    The consumer inspects ``provider.config.difficulty`` for the difficulty
    level count; the CustomCurriculumAdapter exposes ``mode``/``kwargs``
    instead, so this shim reports no difficulty sub-config (falls back to
    a default level count).
    """

    def __init__(self, provider):
        self._provider = provider

    @property
    def difficulty(self):
        return None


class _CurriculumProvider:
    """Unified surface over CurriculumManager / CustomCurriculumAdapter.

    Both providers expose update()/get_sample_weights()/get_inclusion_mask(),
    but with different signatures. This shim lets the training loop call either
    through one interface (used for the P3-2 factory wiring).
    """

    def __init__(self, provider):
        self._provider = provider

    @property
    def config(self):
        cfg = getattr(self._provider, "config", None)
        if cfg is not None and hasattr(cfg, "difficulty"):
            return cfg
        return _CurriculumProviderConfig(self._provider)

    def get_inclusion_mask(self, epoch=None):
        return self._provider.get_inclusion_mask()

    def get_sample_weights(self):
        return self._provider.get_sample_weights()

    def update(self, epoch, **kwargs):
        try:
            return self._provider.update(epoch, **kwargs)
        except TypeError:
            allowed = {}
            if "losses" in kwargs:
                allowed["losses"] = kwargs["losses"]
            return self._provider.update(epoch, **allowed)


def _apply_curriculum_weights(loss, pred, yb, crit, classification, batch_idx_t, sample_weight_lookup):
    """Fold per-sample curriculum weights into the batch loss as a weighted mean.

    Only supported for single-output regression criteria that accept a
    ``weight=`` kwarg (HuberLoss, AsymmetricDirectionalLoss, SharpeProxyLoss).
    CE / multitask / tuple-output paths return the plain loss unchanged.
    """
    if classification or isinstance(pred, tuple):
        return loss
    try:
        idx_np = batch_idx_t.detach().cpu().numpy()
    except (AttributeError, RuntimeError):
        return loss
    try:
        sw = torch.as_tensor(
            np.asarray(sample_weight_lookup)[idx_np],
            dtype=torch.float32,
            device=batch_idx_t.device,
        )
    except (IndexError, ValueError, TypeError):
        return loss
    if sw.numel() == 0:
        return loss
    try:
        weighted = crit(pred, yb, weight=sw)
    except TypeError:
        return loss
    if not torch.isfinite(weighted):
        return loss
    return weighted


def _build_train_loss(
    model, xb, yb, y_cls_b, y_conf_b, crit, classification, multitask, direction_only,
    teacher_model, distill_weight, ewc_module, ewc_lambda,
    si_module, si_lambda,
    loader, batch_idx_t,
    online_miner, accum_steps, sample_weight_lookup=None,
):
    """Shared forward + loss assembly used by both AMP and non-AMP paths.

    ``si_lambda`` is the effective λ for this epoch, computed by the caller
    from the FeatureStabilityMonitor's regime-drift estimate
    (``epoch_si_lambda``): ``λ = base_λ / (1 + max_shift²)`` so the SI penalty
    relaxes under severe distribution shift and re-locks when the regime
    stabilizes.
    """
    pred = model(xb)
    loss = _compute_loss(
        pred, crit, yb, classification,
        y_cls=y_cls_b, y_conf=y_conf_b, multitask=multitask,
        direction_only=direction_only,
    )
    if sample_weight_lookup is not None and batch_idx_t is not None:
        loss = _apply_curriculum_weights(
            loss, pred, yb, crit, classification, batch_idx_t, sample_weight_lookup,
        )
    _apply_online_miner(online_miner, pred, yb, y_cls_b, batch_idx_t, classification, multitask)
    if hasattr(loader, "update_priorities") and batch_idx_t is not None:
        try:
            loader.update_priorities(batch_idx_t, loss.detach())
        except Exception as exc:
            print(f"[Train] update_priorities failed: {exc}")
    loss = _apply_kd_loss(pred, teacher_model, xb, loss, distill_weight)
    if ewc_module is not None:
        loss = apply_ewc_loss(loss, ewc_module, ewc_lambda)
    if si_module is not None:
        loss = apply_si_loss(loss, si_module, si_lambda)
    return pred, loss / accum_steps


def _optimizer_step(
    model, opt, scaler_amp, use_fp16_scaler, do_step, grad_clip, scheduler, batch_idx,
    *,
    nan_skips: int,
    epoch: int,
    pbar,
    si_module=None,
) -> tuple[bool, int]:
    """Backward is assumed done. Returns (stepped_ok, updated_nan_skips). """
    if not do_step:
        return True, nan_skips
    if use_fp16_scaler:
        scaler_amp.unscale_(opt)
    if not _gradients_are_finite(model):
        nan_skips += 1
        _log_nan(batch_idx, epoch, nan_skips)
        _recover_nonfinite_training_state(model, opt)
        if use_fp16_scaler:
            scaler_amp.update()
        if nan_skips <= 3 or nan_skips % 10 == 0:
            print(f"[Train] NaN/Inf gradients at batch {batch_idx} (skip {nan_skips})")
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(loss="NaN-grad-skip")
        return False, nan_skips
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    _maybe_warn_grad_norm(model, batch_idx)
    # Snapshot params AND raw gradients before GC mutates p.grad in place, so
    # SI's path integral is computed from the true (unclipped/raw) gradients.
    if si_module is not None:
        si_module.pre_step()
    # Shared GC for both AMP (after unscale) and non-AMP paths.
    _centralize_gradients(model)
    if use_fp16_scaler:
        scaler_amp.step(opt)
        scaler_amp.update()
    else:
        opt.step()
    if si_module is not None:
        si_module.post_step()
    opt.zero_grad(set_to_none=True)
    if scheduler is not None:
        scheduler.step()
    return True, nan_skips


def _prepare_train_batch(
    batch,
    device,
    *,
    seq_len: int | None,
    feature_mask: torch.Tensor | None,
    adversarial_gen,
    adversarial_feature_names: list[str] | None,
    model=None,
    crit=None,
    classification: bool = False,
    multitask: bool = False,
):
    """Unpack, sanitize, mask, and optionally adversarially perturb one batch.

    Returns ``(xb, yb, y_cls, y_conf, sample_idx)`` or ``None`` when every
    row has non-finite targets and the batch must be skipped.
    """
    xb, yb, y_cls_b, y_conf_b, batch_idx_t = _unpack_batch(batch, device)
    if seq_len is not None and xb.shape[1] > seq_len:
        xb = xb[:, -seq_len:, :]
    xb, yb, y_cls_b, y_conf_b, keep = _sanitize_batch_tensors(
        xb, yb, y_cls_b, y_conf_b, skip_bad_targets=True,
    )
    if keep is not None and not bool(keep.all()):
        if not bool(keep.any()):
            return None
        xb = xb[keep]
        yb = yb[keep]
        if y_cls_b is not None:
            y_cls_b = y_cls_b[keep]
        if y_conf_b is not None:
            y_conf_b = y_conf_b[keep]
        if batch_idx_t is not None:
            batch_idx_t = batch_idx_t[keep]

    if feature_mask is not None:
        xb = xb * feature_mask

    if adversarial_gen is not None:
        try:
            y_adv = y_cls_b if (classification or multitask) else yb
            res = adversarial_gen(model, xb, y_adv, crit)
            xb = res[0] if isinstance(res, (tuple, list)) else res
        except TypeError:
            with torch.no_grad():
                xb = adversarial_gen(xb, adversarial_feature_names)

    return xb, yb, y_cls_b, y_conf_b, batch_idx_t


def _train_batch(
    model,
    xb,
    yb,
    y_cls_b,
    y_conf_b,
    batch_idx_t,
    *,
    crit,
    classification: bool,
    multitask: bool,
    direction_only: bool,
    teacher_model,
    distill_weight: float,
    ewc_module,
    ewc_lambda: float,
    loader,
    online_miner,
    accum_steps: int,
    opt,
    scaler_amp,
    use_fp16_scaler: bool,
    amp_on: bool,
    amp_dtype: torch.dtype,
    do_step: bool,
    grad_clip: float,
    scheduler,
    batch_idx: int,
    epoch: int,
    pbar,
    nan_skips: int,
    si_module=None,
    si_lambda: float = 1.0,
    sample_weight_lookup=None,
) -> tuple[str, float | None, int]:
    """Shared AMP/non-AMP train step: forward → backward → optional opt step.

    Returns ``(status, loss_val, nan_skips)`` where status is
    ``"ok"``, ``"nan_loss"``, or ``"nan_grad"``.
    """
    amp_ctx = autocast("cuda", dtype=amp_dtype) if amp_on else nullcontext()
    with amp_ctx:
        _, loss = _build_train_loss(
            model, xb, yb, y_cls_b, y_conf_b, crit, classification,
            multitask, direction_only, teacher_model, distill_weight,
            ewc_module, ewc_lambda, si_module, si_lambda, loader, batch_idx_t, online_miner,
            accum_steps, sample_weight_lookup=sample_weight_lookup,
        )

    if not torch.isfinite(loss):
        nan_skips += 1
        _log_nan(batch_idx, epoch, nan_skips)
        _recover_nonfinite_training_state(model, opt)
        if nan_skips <= 3 or nan_skips % 10 == 0:
            print(f"[Train] NaN/Inf loss at batch {batch_idx} (skip {nan_skips})")
        opt.zero_grad(set_to_none=True)
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(loss="NaN-skip")
        return "nan_loss", None, nan_skips

    scale = use_fp16_scaler and amp_on
    if scale:
        scaler_amp.scale(loss).backward()
    else:
        loss.backward()

    ok, nan_skips = _optimizer_step(
        model, opt, scaler_amp, scale, do_step,
        grad_clip, scheduler, batch_idx,
        nan_skips=nan_skips, epoch=epoch, pbar=pbar,
        si_module=si_module,
    )
    if not ok:
        return "nan_grad", None, nan_skips

    return "ok", float(loss.item() * accum_steps), nan_skips


def train_epoch(
    model, loader, opt, crit, scaler_amp, device, use_amp, classification: bool,
    grad_clip: float = 1.0, pbar=None,
    amp_dtype: torch.dtype = torch.float32,
    thermal_limit: int = 83,
    feature_mask: torch.Tensor | None = None,
    scheduler=None,
    accum_steps: int = 1,
    seq_len: int | None = None,
    multitask: bool = False,
    epoch: int = 0,
    teacher_model=None,
    distill_weight: float = 0.5,
    direction_only: bool = False,
    online_miner=None,
    adversarial_gen=None,
    adversarial_feature_names: list[str] | None = None,
    ewc_module=None,
    ewc_lambda: float = 1000.0,
    si_module=None,
    si_lambda: float = 1.0,
    sample_weight_lookup=None,
):
    """One training epoch via shared ``_prepare_train_batch`` / ``_train_batch``."""
    _ensure_bound()
    model.train()
    total = 0.0
    n = 0
    oom_skips = 0
    nan_skips = 0
    use_fp16_scaler = scaler_amp.is_enabled()
    amp_on = bool(use_amp and device.type == "cuda")
    _n_batches = len(loader)
    _mask = feature_mask.to(device) if feature_mask is not None else None
    opt.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        if batch_idx % 50 == 0:
            if _TRAIN_LOGGER:
                _TRAIN_LOGGER.heartbeat()
            _thermal_check(limit=thermal_limit)

        do_step = ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == _n_batches)

        try:
            prepared = _prepare_train_batch(
                batch, device,
                seq_len=seq_len,
                feature_mask=_mask,
                adversarial_gen=adversarial_gen,
                adversarial_feature_names=adversarial_feature_names,
                model=model,
                crit=crit,
                classification=classification,
                multitask=multitask,
            )
            if prepared is None:
                nan_skips += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(loss="bad-tgt-skip")
                continue
            xb, yb, y_cls_b, y_conf_b, batch_idx_t = prepared

            status, loss_val, nan_skips = _train_batch(
                model, xb, yb, y_cls_b, y_conf_b, batch_idx_t,
                crit=crit,
                classification=classification,
                multitask=multitask,
                direction_only=direction_only,
                teacher_model=teacher_model,
                distill_weight=distill_weight,
                ewc_module=ewc_module,
                ewc_lambda=ewc_lambda,
                si_module=si_module,
                si_lambda=si_lambda,
                loader=loader,
                online_miner=online_miner,
                accum_steps=accum_steps,
                opt=opt,
                scaler_amp=scaler_amp,
                use_fp16_scaler=use_fp16_scaler,
                amp_on=amp_on,
                amp_dtype=amp_dtype,
                do_step=do_step,
                grad_clip=grad_clip,
                scheduler=scheduler,
                batch_idx=batch_idx,
                epoch=epoch,
                pbar=pbar,
                nan_skips=nan_skips,
                sample_weight_lookup=sample_weight_lookup,
            )
            if status != "ok" or loss_val is None:
                continue

            total += loss_val
            n += 1
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss_val:.5f}")

        except RuntimeError as e:
            is_oom = ("out of memory" in str(e).lower()) or ("cuda error" in str(e).lower())
            if not (device.type == "cuda" and is_oom):
                _log_error(f"[Train] Unexpected RuntimeError at batch {batch_idx}", e)
                raise
            oom_skips += 1
            opt.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss="OOM-skip")
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
    device: torch.device,
    cache_path: str | None = None,
    train_idx: np.ndarray | None = None,
):
    """
    Huber / asymmetric / directional_huber / sharpe_huber regression,
    weighted CE on {-1,0,+1}, or MultiTaskLoss.
    MultiTaskLoss is selected when --multitask is passed and combines:
      w_dir*CE(direction) + w_ret*Huber(return_hat) + w_conf*BCE(confidence)
    """
    _ensure_bound()
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
            label_smoothing=float(
                getattr(args, "label_smoothing", TRAINING.get("label_smoothing", 0.05))
            ),

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
    pred_counts: torch.Tensor,
    true_counts: torch.Tensor,
    confusion: torch.Tensor,
    logits_sum: torch.Tensor,
    probs_sum: torch.Tensor,
    diag_true_counts: torch.Tensor,
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
                   seq_len: int | None = None, multitask: bool = False,
                   feature_mask: torch.Tensor | None = None,
                   sharpe_ann_factor: float | None = None,
                   direction_only: bool = False,
                   rl_mode: bool = False):
    """Run one validation epoch in eager FP32 by default.

    Autocast is off unless ``amp=True`` is passed explicitly — validation is
    usually not compute-bound, and AMP adds cast overhead while hurting
    Sharpe / CE numeric stability.
    """

    _ensure_bound()
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
    diag_true_counts = torch.zeros(3, device=device)

    heartbeat_interval = 50
    _mask = feature_mask.to(device) if feature_mask is not None else None

    def _accumulate_class_diag(logits: torch.Tensor, y_cls_idx: torch.Tensor) -> torch.Tensor:
        """Update class diagnostics; return pred_cls."""
        nonlocal correct, n_acc, pred_counts, true_counts, confusion, logits_sum, probs_sum, diag_true_counts
        pred_cls = logits.argmax(-1)
        correct += (pred_cls == y_cls_idx).sum()
        n_acc += int(y_cls_idx.numel())
        pred_counts += torch.bincount(pred_cls.reshape(-1).clamp(0, 2), minlength=3)[:3]
        true_counts += torch.bincount(y_cls_idx.reshape(-1).clamp(0, 2), minlength=3)[:3]
        probs = torch.softmax(logits.float(), dim=-1)
        t_flat = y_cls_idx.reshape(-1).clamp(0, 2)
        p_flat = pred_cls.reshape(-1).clamp(0, 2)
        for _t, _p in zip(t_flat, p_flat):
            confusion[int(_t), int(_p)] += 1
        for _cls in range(3):
            _mask_cls = t_flat == _cls
            if bool(_mask_cls.any()):
                diag_true_counts[_cls] += int(_mask_cls.sum())
                logits_sum[_cls] += logits.float()[_mask_cls].sum(dim=0)
                probs_sum[_cls] += probs[_mask_cls].sum(dim=0)
        return pred_cls

    with torch.no_grad():
        for i, batch in enumerate(loader):
            try:
                xb, yb, y_cls_b, y_conf_b, _ = _unpack_batch(batch, device)
                if seq_len is not None and xb.shape[1] > seq_len:
                    xb = xb[:, -seq_len:, :]
                xb, yb, y_cls_b, y_conf_b, keep = _sanitize_batch_tensors(
                    xb, yb, y_cls_b, y_conf_b, skip_bad_targets=True,
                )
                if keep is not None and not bool(keep.all()):
                    if not bool(keep.any()):
                        nan_skips += 1
                        if pbar is not None:
                            pbar.update(1)
                            pbar.set_postfix(loss="bad-tgt-skip")
                        continue
                    xb, yb = xb[keep], yb[keep]
                    if y_cls_b is not None:
                        y_cls_b = y_cls_b[keep]
                    if y_conf_b is not None:
                        y_conf_b = y_conf_b[keep]
                if _mask is not None:
                    xb = xb * _mask

                if not torch.isfinite(xb).all() or not torch.isfinite(yb).all():
                    nan_skips += 1
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(loss="NaN-skip")
                    continue

                # Eager FP32 unless caller opts into amp (rare).
                amp_ctx = (
                    autocast(device_type=device.type, dtype=amp_dtype, enabled=True)
                    if amp and device.type == "cuda"
                    else nullcontext()
                )
                with amp_ctx:
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
                        if not (torch.isfinite(loss) and torch.isfinite(logits).all()
                                and torch.isfinite(ret_hat).all()):
                            nan_skips += 1
                            if pbar is not None:
                                pbar.update(1)
                                pbar.set_postfix(loss="NaN-skip")
                            continue
                        total += loss
                        pred_cls = _accumulate_class_diag(logits, y_cls_idx)
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
                        pred_cls = _accumulate_class_diag(pred, y_cls_idx)
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
                if rl_mode or y_cls_b is not None:
                    if y_cls_b is not None:
                        side = _match_target_shape(d, y_cls_b.float()).sign()
                        yb_for_returns = yb_for_returns.abs() * side
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
        _diag = _validation_class_diag(
            pred_counts, true_counts, confusion, logits_sum, probs_sum, diag_true_counts,
        )
        validate_epoch.last_class_counts = {"pred": _diag["pred"], "true": _diag["true"]}
        validate_epoch.last_class_diag = _diag
        return val_loss, dir_acc, 0.0

    r_mean = r_sum / n_ret
    r_var = torch.clamp(r_sq_sum / n_ret - r_mean ** 2, min=0.0)
    sharpe = (r_mean / (r_var.sqrt() + 1e-8)).item() * ann
    _diag = _validation_class_diag(
        pred_counts, true_counts, confusion, logits_sum, probs_sum, diag_true_counts,
    )
    validate_epoch.last_class_counts = {"pred": _diag["pred"], "true": _diag["true"]}
    validate_epoch.last_class_diag = _diag
    return val_loss, dir_acc, sharpe

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
    device:     torch.device,
    n_gpus:     int,
    run: Any = None,
    train_idx:  np.ndarray | None = None,
    val_idx:    np.ndarray | None = None,
    fold_id:    int | None = None,
    amp_dtype:  torch.dtype = torch.float32,
):
    _ensure_bound()

    # ── Lightning training path (opt-in via --training-framework lightning) ──
    _train_framework = str(getattr(args, "training_framework", "custom") or "custom").lower()
    if _train_framework == "lightning":
        try:
            from training.lightning_trainer import run_lightning_training, is_lightning_available
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
    _artifact_run_name = str(getattr(args, "run_name_slug", "") or _slug_part(getattr(args, "run_name", "pipeline-run"), max_len=140))
    sidecar = None  # always defined; reused logger path skips Sidecar creation

    if _TRAIN_LOGGER_AVAILABLE:
        if _TRAIN_LOGGER is None:
            _sidecar_cfg = getattr(args, "sidecar", None) or {}
            if _sidecar_cfg.get("enabled", False):
                try:
                    sidecar = Sidecar(
                        log_dir    = PATHS.get("logs", "logs"),
                        run_name   = _artifact_run_name,
                        model_name = model_name,
                        enabled    = True,
                        mode       = str(_sidecar_cfg.get("mode", "process")),
                        max_queue_size = int(_sidecar_cfg.get("max_queue_size", 10000)),
                        flush_interval_s = float(_sidecar_cfg.get("flush_interval_s", 2.0)),
                        retention_days   = int(_sidecar_cfg.get("retention_days", 30)),
                        enable_discord   = bool(_sidecar_cfg.get("enable_discord", False)),
                    )
                    sidecar.start()
                except Exception as e:
                    print(f"[Sidecar] Failed to start: {e}")
                    sidecar = None

            _TRAIN_LOGGER = _TrainingLogger(
                log_dir    = PATHS.get("logs", "logs"),
                run_name   = f"{_artifact_run_name}_{datetime.now().strftime('%m%d_%H%M')}",
                model_name = model_name,
                sidecar    = sidecar,
            )
            _TRAIN_LOGGER.setup()
            _sync_owned_to_host("_TRAIN_LOGGER")
        else:
            _TRAIN_LOGGER.model_name = model_name
            # Prefer the logger's existing sidecar on subsequent folds
            sidecar = getattr(_TRAIN_LOGGER, "sidecar", None)
            _sync_owned_to_host("_TRAIN_LOGGER")

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

    if hasattr(args, "_training_features_report"):
        report = args._training_features_report
        print("\n  [Training-Features Report]")
        for k, v in report.items():
            print(f"    {k:20s}: {str(v['mode']):<6s} (source: {v['source']})")
        print()
        
        try:
            import json
            from pathlib import Path
            if getattr(args, "checkpoint_dir", None):
                rep_path = Path(args.checkpoint_dir) / "training_features_report.json"
                with open(rep_path, "w") as f:
                    json.dump(report, f, indent=2)
            if run is not None:
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
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError(
            f"[Split] Empty train/val after holdout/embargo "
            f"(n_samples={n_samples}, train={len(train_idx)}, val={len(val_idx)}). "
            f"Increase --n-ticks / use real data, or reduce seq_len/embargo for quick runs."
        )

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
            if getattr(args, "ignore_preflight", False) or getattr(args, "quick_mode", False):
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

    if getattr(args, "enable_per", False):
        train_dl = PrioritizedDataLoader(train_ds, batch_size=args.batch_size,
                                         num_workers=nw, pin_memory=pin_mem,
                                         persistent_workers=use_persistent, prefetch_factor=pf)
    else:
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

    # Always overlap CPU Zarr decompress / H2D with GPU compute via a
    # background-thread prefetch queue (helps even when num_workers > 0).
    train_dl = wrap_loader_prefetch(train_dl, args)
    val_dl = wrap_loader_prefetch(val_dl, args)
    print(f"[Loader] {len(train_dl)} train batches | {len(val_dl)} val batches | "
          f"{nw} workers | prefetch={pf if pf is not None else 0} | "
          f"thread_prefetch={int(getattr(args, 'thread_prefetch_batches', 8) or 8)} | "
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

    # -- torch.compile (PyTorch >= 2.0) — ~20-30 % extra throughput on Ada ----
    # Enabled by default (GPU.torch_compile=True). LSTM/GRU/RNN cells stay
    # eager via torch.compiler.disable so the rest of the graph can use
    # inductor (incl. reduce-overhead); requires Triton (Linux).
    model = maybe_torch_compile(model, device, _GPU_CFG if isinstance(_GPU_CFG, dict) else None)

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
    _sync_owned_to_host("_OVERCONF_PENALTY")
    opt = build_adamw(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Gradient accumulation: effective batch = batch_size ├ù accum_steps
    _accum = max(1, int(getattr(args, "grad_accum_steps", 1)))
    # OneCycleLR must be stepped once per OPTIMIZER UPDATE (not per batch).
    # steps_per_epoch = ceil(batches / accum_steps) so the total cycle length
    # equals epochs ├ù optimizer-updates-per-epoch.
    _eff_steps = max(1, -(-len(train_dl) // _accum))   # ceiling div
    _sched_kind = str(getattr(args, "lr_schedule", "warmup_cosine")).strip().lower()  # Fix Item 4: default to warmup_cosine
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
            total_steps=_total_steps,
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
    _resume_chunk_history: list | None  = None
    _resume_chunk_streak:  int | None   = None
    _resume_feat_state:    dict | None  = None

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
        _resume_feat_state    = ck.get("feat_stability_state")
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
    _chunk_patience = int(_CURR.get("chunk_early_stop_patience", 3))
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
        for _err in (_drift.get("errors") or []):
            _log_warn(f"[CurriculumDrift] {_err}")
        for _line in format_audit_warnings(_drift, prefix="[CurriculumDrift]"):
            _log_warn(_line)
        _mkt = audit_required_market_columns(feature_mask=_FM_AUDIT)
        for _line in format_audit_warnings(_mkt, prefix="[MarketSchema]"):
            _log_warn(_line)
        for _err in (_mkt.get("errors") or []):
            _log_warn(f"[MarketSchema] {_err}")
    except Exception as _audit_exc:
        _log_warn(f"[CurriculumAudit] skipped ({_audit_exc})")

    def _seq_len_for_epoch(ep: int) -> int:
        return args.seq_len

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

            "written_at": datetime.now(UTC).isoformat(),

        })

        _safe_save_json(_feature_ablation_report, _fa_dir / f"{model_name}_feature_ablation_report.json")

    except Exception as _fa_e:

        _log_warn(f"[FeatureAblation] Report write failed: {_fa_e}")


    

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
    _feat_mask: torch.Tensor | None = None   # updated each epoch; None = all active

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
    
    epoch_bar = _pbar(range(start_ep, args.epochs), desc=f"Train {model_name.upper()}", unit="ep") if _rich_display is None else range(start_ep, args.epochs)


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
                    return nn.functional.cross_entropy(outputs, y)
                pred = outputs[0] if isinstance(outputs, tuple) else outputs
                return nn.functional.mse_loss(pred.reshape(-1), labels.reshape(-1).float()[: pred.numel()])
            _ewc = ElasticWeightConsolidation(
                model, train_ds, device, max_samples=1000,
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
            from training.adversarial_generator import create_adversarial_attack
            from pretrain.hard_example_mining import PretrainHardExampleMiner
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
            print(f"[Adversarial] Initialized method={_adv_method} "
                  f"(eps={getattr(args, 'adversarial_eps', 0.3)}, "
                  f"prob={getattr(args, 'adversarial_prob', 0.01)}, "
                  f"n_features_named={len(_adv_feature_names)}).")
        except Exception as e:
            _adversarial = None
            print(f"[Adversarial] Failed to initialize (continuing without adversarial): {e}")

    # Overfitting controller — flags from evaluate_epoch are applied each epoch
    from training.training_controller import TrainingController
    _train_ctrl = TrainingController(report_dir=str(ckpt_dir))
    _train_ctrl.set_recipe(str(model_name))
    _ctrl_stop_early = False

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
                _cb_kw = dict(
                    n_levels=int(getattr(args, "curriculum_n_levels", 10) or 10),
                    start_level=int(getattr(args, "curriculum_start_level", 1) or 1),
                    advance_rate=float(getattr(args, "curriculum_advance_rate", 0.1)),
                    max_level=int(getattr(args, "curriculum_freeze_patience", 1) or 1) + int(getattr(args, "curriculum_n_levels", 9) or 9),
                    total_epochs=max(1, int(args.epochs)),
                    use_loss_weighting=bool(getattr(args, "use_loss_weighting", False)),
                )
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
                )
                _curriculum_mgr = _CurriculumProvider(_curriculum_mgr)
                print(f"[CurriculumManager] Enabled (mode={_cm_mode}) "
                      f"over {_cm_n:,} train samples")
        except Exception as _cm_exc:
            _curriculum_mgr = None
            print(f"[CurriculumManager] Disabled ({_cm_exc})")

    # Miner feedback gating
    _miner_models = str(getattr(args, "curriculum_miner_models", "") or "").strip()
    _miner_allowed = True
    if _miner_models:
        _miner_allowed = model_name.lower() in [m.strip().lower() for m in _miner_models.split(",")]
    _miner_feedback_enabled = (bool(getattr(args, "curriculum_miner_feedback", False))
                               and _miner_allowed
                               and _online_miner is not None
                               and _curriculum_mgr is not None)

    # Self-paced / loss weighting model gating
    _sp_models = str(getattr(args, "self_paced_models", "") or "").strip()
    _sp_allowed = True
    if _sp_models:
        _sp_allowed = model_name.lower() in [m.strip().lower() for m in _sp_models.split(",")]

    _lw_models = str(getattr(args, "loss_weighting_models", "") or "").strip()
    _lw_allowed = True
    if _lw_models:
        _lw_allowed = model_name.lower() in [m.strip().lower() for m in _lw_models.split(",")]

    for ep in epoch_bar:
        if _TRAIN_LOGGER is not None:
            _TRAIN_LOGGER.on_epoch_start(ep, total_epochs=args.epochs,
                                         seq_len=_seq_len_for_epoch(ep))
        curr_seq_len = _seq_len_for_epoch(ep)
        _active_seq_len = curr_seq_len

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

        # -- Feature Stability Monitor -----------------------------------------
        epoch_si_lambda = float(getattr(args, "si_lambda", 1.0))
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

            # Dynamic SI scaling based on regime drift / volatility
            _max_shift = float(_stab_report["feat_max_shift"])
            _dyn_w = 1.0 / (1.0 + (_max_shift ** 2))
            epoch_si_lambda = epoch_si_lambda * _dyn_w

            if _stab_report["feat_frozen"] > 0 or _stab_report["feat_noisy"] > 0:
                _log_info(
                    f"[FeatStab] Ep {ep+1}: frozen={_stab_report['feat_frozen']} "
                    f"noisy={_stab_report['feat_noisy']} "
                    f"active={_stab_report['feat_active']}/{n_features} "
                    f"max_shift={_max_shift:.2f}sigma (si_lambda={epoch_si_lambda:.2f})"
                )
            _safe_wandb_log(run, {
                "feat/frozen": _stab_report["feat_frozen"],
                "feat/noisy": _stab_report["feat_noisy"],
                "feat/max_shift": _max_shift,
                "feat/si_dynamic_lambda": epoch_si_lambda,
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
        ep_train_idx = train_idx
        if _curriculum_mgr is not None:
            _cm_mask = _curriculum_mgr.get_inclusion_mask(ep)
            # Apply mask to ep_train_idx by intersecting with allowed indices
            _allowed = np.where(_cm_mask)[0]
            ep_train_idx = np.intersect1d(ep_train_idx, _allowed)
            if len(ep_train_idx) < 50: ep_train_idx = train_idx
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
                            f"[CurriculumManager] Epoch {ep+1}: included "
                            f"{len(ep_train_idx):,}/{len(_cm_mask):,} samples "
                            f"({float(_cm_mask.mean()):.0%})"
                        )
                history.setdefault("curriculum_manager_state", []).append({
                    "epoch": ep,
                    "mode": getattr(args, "curriculum_manager_mode", "combined"),
                    "inclusion_rate": float(_cm_mask.mean()) if len(_cm_mask) else 1.0,
                    "weights_mean": float(np.asarray(_cm_info.get("weights", np.ones(1))).mean()),
                })
                # Adversarial + Curriculum Coordination: scale eps with difficulty level
                if _adversarial is not None and bool(getattr(args, "adversarial_eps_curriculum_scale", False)):
                    _diff_level = _cm_info.get("difficulty_level", 1)
                    _n_levels = getattr(_curriculum_mgr.config.difficulty, "max_level", 10) if _curriculum_mgr.config.difficulty else 10
                    _level_ratio = _diff_level / max(1, _n_levels)
                    _base_eps = float(getattr(args, "adversarial_eps", 0.3))
                    _scaled_eps = _base_eps * _level_ratio
                    _adversarial.set_eps(_scaled_eps)
                    if hasattr(_adversarial, "set_warmup_step"):
                        _adversarial.set_warmup_step(ep)
                    _log_info(f"[Adversarial+Curriculum] Epoch {ep+1}: eps scaled to {_scaled_eps:.4f} (level {_diff_level}/{_n_levels})")
            
            except Exception as _cm_exc:
                _log_warn(f"[CurriculumManager] Epoch {ep+1} update failed: {_cm_exc}")
        _cm_wl = None
        if _curriculum_mgr is not None:
            try:
                _cm_wl = np.ones(n_samples, dtype=np.float64)
                _cm_wl[train_idx] = np.asarray(
                    _curriculum_mgr.get_sample_weights(), dtype=np.float64,
                )
            except Exception as _cm_wl_exc:
                _log_warn(f"[CurriculumManager] Sample-weight lookup failed: {_cm_wl_exc}")
                _cm_wl = None
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
            if len(ep_train_idx) != len(train_idx) and not _direction_warmup_active:
                _log_info(f"[Curriculum] Epoch {ep+1}: training on "
                          f"({len(ep_train_idx):,}/{len(train_idx):,} samples)")
            _ep_ds = ZarrStreamDataset(
                cache_path, ep_train_idx, shuffle_chunks=True,
                multitask_targets=use_direction_targets,
                return_indices=True,
            )
            epoch_train_dl = DataLoader(
                _ep_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=nw, pin_memory=pin_mem,
                persistent_workers=(use_persistent and nw > 0),
                prefetch_factor=pf if nw > 0 else None,
            )
            epoch_train_dl = wrap_loader_prefetch(epoch_train_dl, args)

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
                adversarial_gen=_adversarial,
                adversarial_feature_names=_adv_feature_names or None,
                ewc_module=_ewc,
                ewc_lambda=float(getattr(args, "ewc_lambda", 1000.0)),
                si_module=_si,
                si_lambda=epoch_si_lambda,
                sample_weight_lookup=_cm_wl,
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
                amp=False, amp_dtype=torch.float32,
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

        # TrainingController: detect overfit / Sharpe collapse and act
        _ctrl_resp = _train_ctrl.evaluate_epoch(ep + 1, float(tl), float(vl), float(v_sh))
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
            print(f"[TrainingController] Early-stop flag set at epoch {ep+1}")

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
        history.setdefault("val_pred_counts", []).append(
            [int(x) for x in _class_counts.get("pred", [0, 0, 0])]
        )
        history.setdefault("val_true_counts", []).append(
            [int(x) for x in _class_counts.get("true", [0, 0, 0])]
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
                _msg = (
                    "[ClassBalance] Direction warmup gate failed: "
                    f"{_reason}. pred S/H/B={_class_diag.get('pred')}, "
                    f"pred_shares={[round(float(s), 4) for s in _class_diag.get('pred_shares', [])]}, "
                    f"true S/H/B={_class_diag.get('true')}, "
                    f"recall={[round(float(r), 4) for r in _class_diag.get('recall', [])]}. "
                    f"Diagnostics -> {_diag_path}"
                )
                if getattr(args, "ignore_preflight", False) or getattr(args, "quick_mode", False):
                    print(f"{_msg} (continuing under quick/ignore-preflight)")
                else:
                    raise RuntimeError(_msg)

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
            _tb_writer.add_scalar("ValPred/sell",     _class_counts.get("pred", [0, 0, 0])[0], ep)

            _tb_writer.add_scalar("ValPred/hold",     _class_counts.get("pred", [0, 0, 0])[1], ep)

            _tb_writer.add_scalar("ValPred/buy",      _class_counts.get("pred", [0, 0, 0])[2], ep)

            _tb_writer.add_scalar("Train/lr",         lr,   ep)
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
                **({"fold": fold_id} if fold_id is not None else {}),
            })

        # -- Early stopping / is best? (Moved up to fix UnboundLocalError) ------
        min_delta = float(getattr(args, "early_stop_min_delta", 0.0))
        improved = (v_sh > (best_sharpe + min_delta)) if stop_on_sharpe else (vl < (best_val_loss - min_delta))

        if improved:
            # Always record both metrics at the selected best epoch (TM-013).
            best_sharpe = v_sh
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

        if improved:
            core = _core_model(model)
            ckpt_meta = {
                "model_name": model_name,
                "n_features": n_features,
                "seq_len": int(curr_seq_len),
                "schema_hash": getattr(args, "feature_schema_hash", "unknown"),
                "timestamp": datetime.now(UTC).isoformat(),
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
                    "best_metric": float(v_sh if stop_on_sharpe else vl),
                    "best_metric_name": "val_sharpe" if stop_on_sharpe else "val_loss",
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
        if _ctrl_stop_early:
            print("\n[Train] Early stop (TrainingController Sharpe-collapse signal)")
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
                    "timestamp": datetime.now(UTC).isoformat(),
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
    _early_stopped = (no_improve >= args.patience) or bool(_ctrl_stop_early)
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
    _control_report["controller_signals"] = list(
        _train_ctrl.report_data.get("overfitting_signals_detected") or []
    )
    _control_report["controller_actions"] = list(
        _train_ctrl.report_data.get("actions_applied") or []
    )

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

    if stop_on_sharpe:
        print(f"\n[Train] Best val Sharpe (proxy): {best_sharpe:.4f}  ->  {best_path}")

        if getattr(args, "ollama_auto_tune", False):
            try:
                from infrastructure.ollama_helper import ollama
                final_metrics = {"best_sharpe": float(best_sharpe), "best_val_loss": float(best_val_loss), "best_epoch": int(_best_ep)}
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

