from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import time

@dataclass
class SessionLimit:
    max_lots: float
    max_open_trades: int
    hours_local: Optional[Tuple[time, time]] = None
    tz: Optional[Any] = None

@dataclass
class RegimeScale:
    crisis: float
    trending: float
    mean_rev: float
    normal: float

@dataclass
class LiveRiskSchema:
    kelly_fraction: float
    max_position_pct: float
    max_total_lots: float
    target_annual_vol: float
    pip_risk_default: float
    max_drawdown_halt: float
    soft_drawdown_reduce: float
    daily_loss_limit: float
    max_consecutive_losses: int
    recovery_bars: int
    session_limits: Dict[str, SessionLimit]
    atr_multiplier: float
    trail_activation_r: float
    breakeven_at_r: float
    regime_scale: RegimeScale
    corr_crisis_threshold: float
    hurst_trending: float
    hurst_mean_rev: float

    def __post_init__(self):
        if not (0 < self.kelly_fraction < 1.0):
            raise ValueError(f"kelly_fraction must be between 0 and 1, got {self.kelly_fraction}")
        if not (0 < self.max_position_pct < 1.0):
            raise ValueError(f"max_position_pct must be between 0 and 1, got {self.max_position_pct}")
        if self.max_total_lots <= 0:
            raise ValueError("max_total_lots must be > 0")
        if self.max_drawdown_halt <= 0 or self.max_drawdown_halt >= 1.0:
            raise ValueError("max_drawdown_halt must be > 0 and < 1")
        
        # Convert nested dicts to objects
        if isinstance(self.session_limits, dict):
            new_limits = {}
            for k, v in self.session_limits.items():
                if isinstance(v, dict):
                    new_limits[k] = SessionLimit(**v)
                else:
                    new_limits[k] = v
            self.session_limits = new_limits
            
        if isinstance(self.regime_scale, dict):
            self.regime_scale = RegimeScale(**self.regime_scale)


@dataclass
class TrainingSchema:
    batch_size: int
    epochs: int
    patience: int
    loss: str
    huber_delta: float
    asymmetric_sign_weight: float
    grad_clip: float
    weight_decay: float
    amp: bool
    val_split: float
    seq_len: int
    checkpoint_dir: Any
    walk_forward_folds: int
    early_stop_metric: str
    sharpe_annualization_factor: float
    onecycle_pct_start: float
    onecycle_max_lr_mult: float
    grad_accum_steps: int
    swa_enabled: bool
    swa_start_frac: float
    swa_lr: float

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if not (0 <= self.val_split < 1.0):
            raise ValueError("val_split must be between 0 and 1")


@dataclass
class SizingSchema:
    method: str
    kelly_fraction: float
    max_position_pct: float
    target_annual_vol: float
    pip_risk: float
    scaling_strategy: str
    pyramid_add_pct: float
    martingale_add_pct: float
    max_total_lots: float
    scale_out_targets: List[float]

    def __post_init__(self):
        if not (0 < self.kelly_fraction < 1.0):
            raise ValueError(f"kelly_fraction must be between 0 and 1, got {self.kelly_fraction}")
        if not (0 < self.max_position_pct < 1.0):
            raise ValueError(f"max_position_pct must be between 0 and 1, got {self.max_position_pct}")
