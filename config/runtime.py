"""
config/runtime.py
=================
Immutable, validated runtime configuration.

Provides ``RuntimeConfig`` — a frozen snapshot of all settings sections built
*after* ``_sync_runtime_config`` has applied CLI/YAML overrides.  Callers that
need isolation (HPO workers, unit tests) should use ``build_runtime_config``
instead of reading the mutable globals in ``config.settings`` directly.

Usage
-----
    from config.runtime import build_runtime_config
    cfg = build_runtime_config(args)          # validated, frozen
    print(cfg.training.batch_size)            # typed attribute access
    print(cfg.risk.max_drawdown_halt)

Backward compat
---------------
The mutable dicts in ``config.settings`` still exist and still work.
``RuntimeConfig`` is opt-in — nothing breaks if you ignore this module.
The ``--legacy-config-globals`` CLI flag (in training/gpu_cli.py) disables
the validation step entirely, falling back to the old dict-mutation path.
"""

from __future__ import annotations

import copy
import types
from dataclasses import dataclass, field
from typing import Any

# ── helpers ──────────────────────────────────────────────────────────────────


def _freeze(obj: Any) -> Any:
    """Recursively convert dicts/lists to MappingProxyType/tuple for immutability."""
    if isinstance(obj, dict):
        return types.MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_freeze(v) for v in obj)
    return obj


# ── section schemas (lightweight dataclasses, no external deps) ───────────────


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    loss: str
    huber_delta: float
    asymmetric_sign_weight: float
    grad_clip: float
    weight_decay: float
    amp: bool
    val_split: float
    seq_len: int
    checkpoint_dir: str
    walk_forward_folds: int
    sharpe_annualization_factor: float
    grad_accum_steps: int
    swa_enabled: bool
    swa_start_frac: float
    swa_lr: float
    lr_warmup_epochs: int = 2
    lr_schedule: str = "warmup_cosine"
    onecycle_pct_start: float = 0.1
    onecycle_max_lr_mult: float = 10.0

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if not (0 <= self.val_split < 1.0):
            raise ValueError(f"val_split must be in [0, 1), got {self.val_split}")


@dataclass(frozen=True)
class RiskConfig:
    stop_type: str
    atr_multiplier: float
    trail_activation_r: float
    breakeven_at_r: float
    max_drawdown_halt: float
    daily_loss_limit: float
    max_notional_usd: float
    max_order_freq_per_min: int
    max_instrument_concentration: float
    var_confidence: float
    var_window: int
    cvar_multiplier: float
    gap_move_threshold: float
    require_approval_on_flatten: bool

    def __post_init__(self):
        if not (0 < self.max_drawdown_halt < 1.0):
            raise ValueError(f"max_drawdown_halt must be in (0,1), got {self.max_drawdown_halt}")
        if not (0 < self.daily_loss_limit < 1.0):
            raise ValueError(f"daily_loss_limit must be in (0,1), got {self.daily_loss_limit}")


@dataclass(frozen=True)
class LabelingConfig:
    method: str
    lookahead_bars: int
    profit_target_atr: float
    stop_loss_atr: float
    transaction_cost_pips: float
    pip_size: float
    tbm_numba: bool
    tbm_parallel: bool
    consensus_threshold: float

    def __post_init__(self):
        if self.lookahead_bars <= 0:
            raise ValueError(f"lookahead_bars must be > 0, got {self.lookahead_bars}")


@dataclass(frozen=True)
class GPUConfig:
    amp_dtype: str
    thermal_limit_celsius: int
    torch_compile: bool
    torch_compile_mode: str
    allow_tf32: bool
    cudnn_benchmark: bool

    def __post_init__(self):
        valid_dtypes = {"auto", "bf16", "fp16", "fp32"}
        if self.amp_dtype not in valid_dtypes:
            raise ValueError(f"amp_dtype must be one of {valid_dtypes}, got {self.amp_dtype!r}")


@dataclass(frozen=True)
class MonitoringConfig:
    enabled: bool
    drift_window: int
    psi_threshold: float
    ks_pvalue_threshold: float
    sharpe_drop_threshold: float
    check_freq_bars: int
    retrain_trigger: str
    retrain_cooldown_sec: int
    wandb_project: str


@dataclass(frozen=True)
class MaturityConfig:
    stage: str

    def __post_init__(self):
        valid = {"paper", "shadow", "live_small", "scale"}
        if self.stage not in valid:
            raise ValueError(f"maturity stage must be one of {valid}, got {self.stage!r}")


@dataclass(frozen=True)
class PretrainConfig:
    enabled: bool
    method: str
    temperature: float
    projection_dim: int
    pred_dim: int
    ema_decay: float
    read_windows: int
    epochs: int
    min_epochs: int
    pretrain_lr: float
    pretrain_batch: int
    checkpoint: str


# ── root config ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RuntimeConfig:
    """Immutable snapshot of all validated settings, built once after CLI/YAML override."""

    training: TrainingConfig
    risk: RiskConfig
    labeling: LabelingConfig
    gpu: GPUConfig
    monitoring: MonitoringConfig
    maturity: MaturityConfig
    pretrain: PretrainConfig
    # raw dicts kept for sections not yet schema-typed (read-only proxy)
    data: types.MappingProxyType = field(default_factory=lambda: types.MappingProxyType({}))
    sizing: types.MappingProxyType = field(default_factory=lambda: types.MappingProxyType({}))
    governance: types.MappingProxyType = field(default_factory=lambda: types.MappingProxyType({}))
    extra: types.MappingProxyType = field(default_factory=lambda: types.MappingProxyType({}))


# ── factory ───────────────────────────────────────────────────────────────────


def build_runtime_config(args: Any = None) -> RuntimeConfig:
    """
    Build an immutable ``RuntimeConfig`` from the current (post-override) settings dicts.

    If ``args`` is supplied, any recognised attribute overrides are applied
    *before* building the frozen object — so each call produces an independent
    snapshot, safe to use in a multi-run HPO worker or isolated test.

    Args:
        args: argparse.Namespace (or any object with attribute access) with
              CLI overrides.  ``None`` = use raw settings dicts as-is.

    Returns:
        RuntimeConfig: frozen, validated snapshot.
    """
    from config import settings as S

    # Deep-copy every mutable section so the snapshot is independent.
    training_d = copy.deepcopy(S.TRAINING)
    risk_d = copy.deepcopy(S.RISK)
    labeling_d = copy.deepcopy(S.LABELING)
    gpu_d = copy.deepcopy(S.GPU)
    monitoring_d = copy.deepcopy(S.MONITORING)
    maturity_d = copy.deepcopy(S.MATURITY)
    pretrain_d = copy.deepcopy(S.PRETRAIN)
    data_d = copy.deepcopy(S.DATA)
    sizing_d = copy.deepcopy(S.SIZING)
    governance_d = copy.deepcopy(S.GOVERNANCE)

    # Apply args overrides (same keys as _sync_runtime_config).
    if args is not None:
        _apply_args(args, training_d, risk_d, labeling_d, gpu_d, maturity_d, pretrain_d)

    return RuntimeConfig(
        training=TrainingConfig(**{k: training_d[k] for k in TrainingConfig.__dataclass_fields__ if k in training_d}),
        risk=RiskConfig(**{k: risk_d[k] for k in RiskConfig.__dataclass_fields__ if k in risk_d}),
        labeling=LabelingConfig(**{k: labeling_d[k] for k in LabelingConfig.__dataclass_fields__ if k in labeling_d}),
        gpu=GPUConfig(**{k: gpu_d[k] for k in GPUConfig.__dataclass_fields__ if k in gpu_d}),
        monitoring=MonitoringConfig(**{k: monitoring_d[k] for k in MonitoringConfig.__dataclass_fields__ if k in monitoring_d}),
        maturity=MaturityConfig(**{k: maturity_d[k] for k in MaturityConfig.__dataclass_fields__ if k in maturity_d}),
        pretrain=PretrainConfig(**{k: pretrain_d[k] for k in PretrainConfig.__dataclass_fields__ if k in pretrain_d}),
        data=_freeze(data_d),
        sizing=_freeze(sizing_d),
        governance=_freeze(governance_d),
    )


def _apply_args(args, training_d, risk_d, labeling_d, gpu_d, maturity_d, pretrain_d) -> None:
    """Apply argparse Namespace overrides to the mutable section dicts."""
    # training
    for attr, key in [
        ("batch_size", "batch_size"),
        ("epochs", "epochs"),
        ("lr", None),
        ("seq_len", "seq_len"),
        ("val_split", "val_split"),
        ("grad_accum_steps", "grad_accum_steps"),
    ]:
        val = getattr(args, attr, None)
        if val is not None and key is not None:
            training_d[key] = val

    # labeling
    for attr, key in [
        ("profit_target_atr", "profit_target_atr"),
        ("stop_loss_atr", "stop_loss_atr"),
        ("lookahead_bars", "lookahead_bars"),
    ]:
        val = getattr(args, attr, None)
        if val is not None:
            labeling_d[key] = val

    # pretrain
    if getattr(args, "pretrain_read_windows", None) is not None:
        pretrain_d["read_windows"] = int(args.pretrain_read_windows)

    # maturity
    if getattr(args, "maturity_stage", None):
        maturity_d["stage"] = str(args.maturity_stage)

    # gpu
    if getattr(args, "torch_compile", None) is not None:
        gpu_d["torch_compile"] = bool(args.torch_compile)
    if getattr(args, "torch_compile_mode", None):
        gpu_d["torch_compile_mode"] = str(args.torch_compile_mode)
