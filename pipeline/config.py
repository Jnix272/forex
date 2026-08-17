"""
Pipeline Configuration
======================
Configuration models for the data pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml


def dataclass_from_dict(cls, data: dict[str, Any]) -> Any:
    """Create a dataclass instance from a dictionary, handling nested dataclasses and enums."""
    if not is_dataclass(cls):
        return data

    # Resolve forward references
    try:
        type_hints = get_type_hints(cls)
    except Exception:
        type_hints = {}

    field_values = {}
    for field in fields(cls):  # noqa: F402
        field_name = field.name
        field_type = type_hints.get(field_name, field.type)

        if field_name not in data:
            # Use default
            continue

        value = data[field_name]

        # Handle Optional types
        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin is None:
            # Simple type or dataclass
            if is_dataclass(field_type):
                if isinstance(value, dict):
                    field_values[field_name] = dataclass_from_dict(field_type, value)
                else:
                    field_values[field_name] = field_type()
            elif isinstance(field_type, type) and issubclass(field_type, Enum):
                field_values[field_name] = field_type(value)
            else:
                field_values[field_name] = value
        elif origin is list:
            # List type
            if args and is_dataclass(args[0]):
                field_values[field_name] = [
                    dataclass_from_dict(args[0], v) if isinstance(v, dict) else v for v in value
                ]
            elif args and isinstance(args[0], type) and issubclass(args[0], Enum):
                field_values[field_name] = [args[0](v) for v in value]
            else:
                field_values[field_name] = value
        elif origin is dict:
            field_values[field_name] = value
        elif origin is Union:
            # Handle Optional[X] which is Union[X, None]
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1 and is_dataclass(non_none_args[0]):
                if isinstance(value, dict):
                    field_values[field_name] = dataclass_from_dict(non_none_args[0], value)
                else:
                    field_values[field_name] = value
            else:
                field_values[field_name] = value
        else:
            field_values[field_name] = value

    return cls(**field_values)


class PipelineStageName(StrEnum):
    """Pipeline stage names"""

    INGESTION = "ingestion"
    RESAMPLING = "resampling"
    FEATURE_ENGINEERING = "feature_engineering"
    LABELING = "labeling"
    DATASET_BUILD = "dataset_build"
    MATERIALIZATION = "materialization"
    VALIDATION = "validation"
    TRAINING = "training"


class DataSourceType(StrEnum):
    """Data source types"""

    DUKASCOPY = "dukascopy"
    DATABENTO = "databento"
    TDS = "tds"
    SYNTHETIC = "synthetic"
    CUSTOM = "custom"


class BarType(StrEnum):
    """Bar types"""

    TIME = "time"
    TICK = "tick"
    VOLUME = "volume"
    DOLLAR = "dollar"


class LabelingMethod(StrEnum):
    """Labeling methods"""

    RL_REWARD = "rl_reward"
    TRIPLE_BARRIER = "triple_barrier"
    BOTH = "both"


@dataclass
class DataSourceConfig:
    """Data source configuration"""

    type: DataSourceType = DataSourceType.DUKASCOPY
    pairs: list[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY"])
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    path: str | None = None
    # Dukascopy specific
    data_root: str = "data/raw/dukascopy"
    # Synthetic specific
    synthetic_rows: int = 100000
    base_price: float = 1.0850
    spread_pips: float = 0.5


@dataclass
class BarConfig:
    """Bar/resampling configuration"""

    bar_type: BarType = BarType.TIME
    freq: str = "1min"
    # Time bars
    session_filter: bool = True
    session_mode: str = "dst"  # "fixed" or "dst"
    session_start_utc: str = "07:00"
    session_end_utc: str = "21:00"
    # Information bars
    tick_threshold: int = 500
    volume_threshold: float = 10000.0
    dollar_threshold: float = 1000000.0
    # Gap handling
    gap_policy: str = "drop"  # "drop", "ffill", "interpolate"
    gap_max_minutes: int = 5
    # Spread capping
    spread_cap_multiplier: float = 3.0
    # Fractional differentiation
    apply_frac_diff: bool = True
    frac_diff_order: float = 0.4


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""

    # Feature groups to enable
    groups: list[str] = field(
        default_factory=lambda: [
            "core",
            "microstructure",
            "momentum",
            "regime",
            "cross_asset",
            "macro",
            "sentiment",
            "candlestick",
            "volume_profile",
            "volatility_clock",
            "risk_control",
        ]
    )
    # Feature cache
    cache_enabled: bool = True
    ofi_z_threshold: float = 2.0
    slow_cols: list[str] = field(
        default_factory=lambda: ["sentiment_decayed", "eco_surprise", "hurst_exponent", "cot_net_hf"]
    )
    # Regime gating
    enable_regime_gate: bool = True
    # Quality gate
    enable_quality_gate: bool = False
    # No-trade zones
    enable_no_trade_zones: bool = False
    # FinBERT
    finbert_dim: int = 8
    # Cross-asset
    ca_corr_window: int = 60
    ca_regime_window: int = 240
    ca_lags: tuple[int, ...] = (1, 5, 15)


@dataclass
class LabelingConfig:
    """Labeling configuration"""

    method: LabelingMethod = LabelingMethod.RL_REWARD
    # RL reward
    lookahead_bars: int = 30
    profit_target_atr: float = 1.2
    stop_loss_atr: float = 0.8
    transaction_cost_pips: float = 1.5
    # Triple barrier
    tb_profit_atr: float = 2.0
    tb_stop_atr: float = 1.0
    tb_horizon_bars: int = 60
    # Regime-aware labeling
    regime_aware: bool = True
    # Execution delay
    execution_delay_bars: int = 1


@dataclass
class DatasetConfig:
    """Dataset building configuration"""

    seq_len: int = 80
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    # Chunking
    chunk_size: int = 500000
    chunk_overlap: int = 200
    # Output format
    output_format: str = "zarr"  # "zarr", "npy", "parquet"
    output_dir: str = "data/processed"
    compression: str = "lz4"
    compression_level: int = 1
    # Multi-pair
    pairs: list[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY"])
    # Label column
    label_col: str = "label"
    # Difficulty scoring
    difficulty_enabled: bool = True


@dataclass
class ValidationConfig:
    """Validation configuration"""

    # Purged embargo CV
    method: str = "purged_embargo"
    n_splits: int = 7
    purge_bars: int = 120
    embargo_bars: int = 60
    min_train_size: int = 50000
    # Quality gates
    quality_gates_enabled: bool = True
    null_threshold: float = 0.01
    inf_threshold: float = 0.0
    psi_threshold: float = 0.2
    correlation_threshold: float = 0.99
    # Drift detection
    drift_detection_enabled: bool = True
    drift_window: int = 1000
    # Schema validation
    schema_validation: bool = True
    feature_schema_gate: bool = True


@dataclass
class QualityGatesConfig:
    """Quality gates configuration"""

    enabled: bool = True
    auto_remediate: bool = True
    log_dir: str = "logs/quality_gates"
    # Per-stage config
    stages: dict[str, dict] = field(default_factory=dict)


@dataclass
class LineageConfig:
    """Lineage tracking configuration"""

    enabled: bool = True
    store_type: str = "file"  # "file", "sqlite"
    path: str = "logs/lineage"
    track_git: bool = True
    track_config: bool = True


@dataclass
class FeatureStoreConfig:
    """Feature store configuration"""

    enabled: bool = True
    store_type: str = "parquet"  # "parquet", "delta"
    path: str = "./feature_store"
    partition_cols: list[str] = field(default_factory=lambda: ["pair", "year", "month", "day"])
    compression: str = "zstd"


@dataclass
class IncrementalConfig:
    """Incremental processing configuration"""

    enabled: bool = False
    state_dir: str = "./feature_state"
    warmup_bars: int = 200
    max_buffer_size: int = 10000


@dataclass
class PipelineConfig:
    """Main pipeline configuration"""

    # Metadata
    name: str = "forex_pipeline"
    version: str = "1.0.0"
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Components
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    bars: BarConfig = field(default_factory=BarConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    quality_gates: QualityGatesConfig = field(default_factory=QualityGatesConfig)
    lineage: LineageConfig = field(default_factory=LineageConfig)
    feature_store: FeatureStoreConfig = field(default_factory=FeatureStoreConfig)
    incremental: IncrementalConfig = field(default_factory=IncrementalConfig)

    # Runtime
    output_dir: str = "logs/pipeline"
    log_level: str = "INFO"
    random_seed: int = 42

    # Hardware
    hardware_profile: str = "ubuntu_rtx_laptop"
    n_workers: int = 4
    use_gpu: bool = True

    # Tags for tracking
    tags: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load configuration from YAML file"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Create config from dictionary"""
        return dataclass_from_dict(cls, data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        result = {}
        for field_name, _field_def in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            if hasattr(value, "to_dict"):
                result[field_name] = value.to_dict()
            elif hasattr(value, "__dataclass_fields__"):
                result[field_name] = value.to_dict() if hasattr(value, "to_dict") else value.__dict__
            elif isinstance(value, Enum):
                result[field_name] = value.value
            else:
                result[field_name] = value
        return result

    def to_yaml(self, path: str | Path):
        """Save configuration to YAML file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, indent=2, default_flow_style=False)

    def get_stage_config(self, stage: PipelineStageName) -> dict:
        """Get configuration for a specific stage"""
        stage_map = {
            PipelineStageName.INGESTION: ["data_source", "bars"],
            PipelineStageName.RESAMPLING: ["bars"],
            PipelineStageName.FEATURE_ENGINEERING: ["features"],
            PipelineStageName.LABELING: ["labeling"],
            PipelineStageName.DATASET_BUILD: ["dataset"],
            PipelineStageName.VALIDATION: ["validation"],
        }

        config = {}
        for attr in stage_map.get(stage, []):
            config[attr] = (
                getattr(self, attr).to_dict()
                if hasattr(getattr(self, attr), "to_dict")
                else getattr(self, attr).__dict__
            )

        return config


def load_pipeline_config(path: str | Path | None = None) -> PipelineConfig:
    """Load pipeline configuration from file or environment"""
    if path:
        return PipelineConfig.from_yaml(path)

    # Check environment variable
    env_path = os.getenv("FOREX_PIPELINE_CONFIG")
    if env_path:
        return PipelineConfig.from_yaml(env_path)

    # Default config
    return PipelineConfig()
