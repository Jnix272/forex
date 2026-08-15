"""
Pipeline Integration
====================
Integration helpers for using the new pipeline components
with existing code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from contracts import (
    TickContract, BarContract, FeatureContract, LabelContract, DatasetContract,
    ContractRegistry, Stage
)
from contracts.validation.gates import PipelineStageValidator
from contracts.validation.drift import SchemaDriftDetector
from contracts.validation.reporter import ValidationReporter
from pipeline.config import PipelineConfig, PipelineStageName, load_pipeline_config
from pipeline.orchestrator import PipelineOrchestrator, create_orchestrator
from pipeline.quality_gates import DataQualityGates, create_quality_gates
from feature_store.store import FeatureStore, ParquetFeatureStore, create_feature_store
from feature_store.materializer import FeatureMaterializer, MaterializationJob, create_materializer
from feature_store.registry import FeatureRegistry, get_registry
from lineage.tracker import LineageTracker, LineageEventType, LineageEvent
from lineage.store import FileLineageStore, SQLiteLineageStore
from features.incremental import StreamingFeatureProcessor, IncrementalFeatureEngine, create_incremental_processor


@dataclass
class PipelineComponents:
    """Container for all pipeline components"""
    config: PipelineConfig
    orchestrator: PipelineOrchestrator
    feature_store: FeatureStore
    materializer: FeatureMaterializer
    registry: FeatureRegistry
    lineage_tracker: LineageTracker | None
    drift_detector: SchemaDriftDetector
    reporter: ValidationReporter
    quality_gates: dict[str, DataQualityGates]
    validators: dict[PipelineStageName, PipelineStageValidator]
    incremental_processor: StreamingFeatureProcessor | None


def create_full_pipeline(
    config_path: str | Path | None = None,
    config: PipelineConfig | None = None,
    enable_lineage: bool = True,
    enable_feature_store: bool = True,
    enable_quality_gates: bool = True,
    enable_incremental: bool = False,
    output_dir: str | Path = "logs/pipeline",
) -> PipelineComponents:
    """
    Create a fully configured pipeline with all components.
    
    Args:
        config_path: Path to pipeline config YAML
        config: PipelineConfig object (alternative to config_path)
        enable_lineage: Enable lineage tracking
        enable_feature_store: Enable feature store
        enable_quality_gates: Enable quality gates
        enable_incremental: Enable incremental processing
        output_dir: Base output directory
        
    Returns:
        PipelineComponents with all initialized components
    """
    # Load or create config
    if config is None:
        config = load_pipeline_config(config_path)
    
    # Override config based on flags
    config.lineage.enabled = enable_lineage
    config.feature_store.enabled = enable_feature_store
    config.quality_gates.enabled = enable_quality_gates
    config.incremental.enabled = enable_incremental
    config.output_dir = str(output_dir)
    
    # Create orchestrator
    orchestrator = create_orchestrator(config=config)
    
    # Create feature store
    feature_store = create_feature_store(
        store_type=config.feature_store.store_type,
        base_path=config.feature_store.path,
    )
    
    # Create materializer
    materializer = create_materializer(
        feature_store_path=config.feature_store.path,
        quality_gates_dir=config.quality_gates.log_dir if enable_quality_gates else None,
        lineage_dir=config.lineage.path if enable_lineage else None,
    )
    
    # Get/create registry
    registry = get_registry()
    
    # Create lineage tracker
    lineage_tracker = None
    if enable_lineage:
        lineage_tracker = LineageTracker(
            store=FileLineageStore(Path(config.lineage.path) / "lineage.jsonl")
        )
    
    # Create drift detector
    drift_detector = SchemaDriftDetector(
        reference_dir=Path(output_dir) / "drift_reference"
    )
    
    # Create reporter
    reporter = ValidationReporter(Path(output_dir) / "validation")
    
    # Create quality gates
    quality_gates = {}
    if enable_quality_gates:
        quality_gates = {
            "ingestion": create_quality_gates("ingestion", remediation_log_dir=config.quality_gates.log_dir),
            "resampling": create_quality_gates("resampling", remediation_log_dir=config.quality_gates.log_dir),
            "feature_engineering": create_quality_gates("feature_engineering", remediation_log_dir=config.quality_gates.log_dir),
        }
    
    # Create validators
    validators = {
        PipelineStageName.INGESTION: PipelineStageValidator(
            stage=PipelineStageName.INGESTION,
            output_dir=Path(output_dir) / "validation",
        ),
        PipelineStageName.RESAMPLING: PipelineStageValidator(
            stage=PipelineStageName.RESAMPLING,
            output_dir=Path(output_dir) / "validation",
        ),
        PipelineStageName.FEATURE_ENGINEERING: PipelineStageValidator(
            stage=PipelineStageName.FEATURE_ENGINEERING,
            output_dir=Path(output_dir) / "validation",
        ),
        PipelineStageName.LABELING: PipelineStageValidator(
            stage=PipelineStageName.LABELING,
            output_dir=Path(output_dir) / "validation",
        ),
        PipelineStageName.DATASET_BUILD: PipelineStageValidator(
            stage=PipelineStageName.DATASET_BUILD,
            output_dir=Path(output_dir) / "validation",
        ),
    }
    
    # Create incremental processor
    incremental_processor = None
    if enable_incremental:
        incremental_processor = create_incremental_processor(
            state_dir=config.incremental.state_dir,
            warmup_bars=config.incremental.warmup_bars,
        )
    
    return PipelineComponents(
        config=config,
        orchestrator=orchestrator,
        feature_store=feature_store,
        materializer=materializer,
        registry=registry,
        lineage_tracker=lineage_tracker,
        drift_detector=drift_detector,
        reporter=reporter,
        quality_gates=quality_gates,
        validators=validators,
        incremental_processor=incremental_processor,
    )


def validate_dataframe(
    df: pl.DataFrame,
    stage: Stage,
    pair: str | None = None,
    strict: bool = True,
) -> tuple[pl.DataFrame, Any]:
    """
    Validate a DataFrame against the contract for a stage.
    
    Args:
        df: DataFrame to validate
        stage: Pipeline stage
        pair: Currency pair
        strict: If True, raise on validation errors
        
    Returns:
        Tuple of (validated DataFrame, metadata)
    """
    contract_map = {
        Stage.INGESTION: TickContract,
        Stage.RESAMPLING: BarContract,
        Stage.FEATURE_ENGINEERING: FeatureContract,
        Stage.LABELING: LabelContract,
        Stage.DATASET_BUILD: DatasetContract,
    }
    
    contract_class = contract_map.get(stage)
    if not contract_class:
        raise ValueError(f"No contract for stage: {stage}")
    
    return contract_class.validate(df, pair=pair, strict=strict)


def run_quality_gates(
    df: pl.DataFrame,
    stage: str,
    pair: str | None = None,
    auto_remediate: bool = True,
    log_dir: str | Path | None = None,
) -> tuple[pl.DataFrame, Any]:
    """
    Run quality gates on a DataFrame.
    
    Args:
        df: DataFrame to check
        stage: Stage name
        pair: Currency pair
        auto_remediate: Whether to auto-remediate issues
        log_dir: Directory for remediation logs
        
    Returns:
        Tuple of (remediated DataFrame, report)
    """
    gates = create_quality_gates(
        stage=stage,
        auto_remediate=auto_remediate,
        remediation_log_dir=log_dir,
    )
    
    return gates.run(df, pair=pair)


def track_lineage(
    event_type: LineageEventType,
    stage: str,
    pair: str | None = None,
    input_data: pl.DataFrame | list[pl.DataFrame] | None = None,
    output_data: pl.DataFrame | None = None,
    output_path: str | None = None,
    metadata: dict | None = None,
    tracker: LineageTracker | None = None,
) -> Any:
    """
    Track a lineage event.
    
    Args:
        event_type: Type of lineage event
        stage: Pipeline stage
        pair: Currency pair
        input_data: Input DataFrame(s)
        output_data: Output DataFrame
        output_path: Output file path
        metadata: Additional metadata
        tracker: LineageTracker instance (uses global if None)
        
    Returns:
        LineageRecord
    """
    if tracker is None:
        from lineage.tracker import get_tracker
        tracker = get_tracker()
    
    if tracker is None:
        return None
    
    return tracker.record_event(LineageEvent(
        event_type=event_type,
        stage=stage,
        pair=pair,
        metadata=metadata or {},
    ), input_data, output_data, output_path)


def materialize_features(
    pairs: list[str],
    start_date: str,
    end_date: str,
    version: str,
    feature_store_path: str | Path = "./feature_store",
    bar_freq: str = "1min",
    quality_gates: bool = True,
    lineage_tracking: bool = True,
) -> Any:
    """
    Convenience function to materialize features for pairs.
    
    Args:
        pairs: Currency pairs
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        version: Version string
        feature_store_path: Feature store path
        bar_freq: Bar frequency
        quality_gates: Enable quality gates
        lineage_tracking: Enable lineage tracking
        
    Returns:
        MaterializationResult
    """
    from datetime import datetime
    
    materializer = create_materializer(
        feature_store_path=feature_store_path,
        quality_gates_dir="logs/quality_gates" if quality_gates else None,
        lineage_dir="logs/lineage" if lineage_tracking else None,
    )
    
    job = MaterializationJob(
        version=version,
        pairs=pairs,
        start_date=datetime.fromisoformat(start_date),
        end_date=datetime.fromisoformat(end_date),
        bar_freq=bar_freq,
        description=f"Materialization for {', '.join(pairs)}",
        quality_gates=quality_gates,
        lineage_tracking=lineage_tracking,
    )
    
    return materializer.materialize(job)


def process_streaming(
    bars: pl.DataFrame,
    pair: str,
    state_dir: str | Path = "./feature_state",
    warmup_bars: int = 200,
) -> pl.DataFrame | None:
    """
    Process bars incrementally for streaming.
    
    Args:
        bars: New bars to process
        pair: Currency pair
        state_dir: State directory
        warmup_bars: Warmup bars needed
        
    Returns:
        Features DataFrame or None if still warming up
    """
    processor = create_incremental_processor(
        state_dir=state_dir,
        warmup_bars=warmup_bars,
    )
    
    return processor.process_bar(bars, pair)


# Example usage functions
def example_full_pipeline():
    """Example: Run full pipeline from config"""
    # Create pipeline from config
    components = create_full_pipeline(
        config_path="config/pipeline.yaml",
        enable_lineage=True,
        enable_feature_store=True,
        enable_quality_gates=True,
    )
    
    # Run pipeline
    report = components.orchestrator.run()
    
    return report
