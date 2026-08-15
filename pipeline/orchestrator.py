"""
Pipeline Orchestrator
=====================
Orchestrates the data pipeline stages.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import polars as pl

from pipeline.config import PipelineConfig, PipelineStageName, load_pipeline_config
from contracts.validation.gates import PipelineStageValidator
from contracts.validation.drift import SchemaDriftDetector
from contracts.validation.reporter import ValidationReporter, ValidationReport
from pipeline.quality_gates import DataQualityGates, create_quality_gates
from feature_store.store import FeatureStore, ParquetFeatureStore
from feature_store.materializer import FeatureMaterializer, MaterializationJob
from lineage.tracker import LineageTracker, LineageEventType, LineageEvent
from lineage.store import FileLineageStore
from features.incremental import StreamingFeatureProcessor, IncrementalFeatureEngine
from data.data_ingestion import ForexDataPipeline, load_or_generate
from features.feature_engineering_pl import FeatureEngineer
from labeling.rl_reward_labeling import compute_rl_reward_labels_regime
from labeling.triple_barrier_labeling import compute_triple_barrier_labels
from training.dataset_builder import _build_chunk


class StageStatus(str, Enum):
    """Pipeline stage status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStage:
    """Pipeline stage definition"""
    name: PipelineStageName
    status: StageStatus = StageStatus.PENDING
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_seconds: float = 0.0
    input_data: Any = None
    output_data: Any = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)
    validation_report: Any = None
    quality_report: Any = None
    
    def to_dict(self) -> dict:
        return {
            "name": self.name.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "metadata": self.metadata,
        }


class PipelineOrchestrator:
    """
    Orchestrates the complete data pipeline.
    
    Runs stages in sequence with validation, quality gates, and lineage tracking.
    """
    
    def __init__(
        self,
        config: PipelineConfig | None = None,
        config_path: str | Path | None = None,
    ):
        self.config = config or load_pipeline_config(config_path)
        self.stages: dict[PipelineStageName, PipelineStage] = {}
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._init_components()
        
        # Initialize lineage
        if self.config.lineage.enabled:
            self.lineage_tracker = LineageTracker(
                store=FileLineageStore(Path(self.config.lineage.path) / "lineage.jsonl")
            )
        else:
            self.lineage_tracker = None
        
        # Initialize drift detector
        self.drift_detector = SchemaDriftDetector(
            reference_dir=self.output_dir / "drift_reference"
        )
        
        # Initialize validation reporter
        self.reporter = ValidationReporter(self.output_dir / "validation")
        
        # Stage validators
        self.validators = {
            PipelineStageName.INGESTION: PipelineStageValidator(
                stage=PipelineStageName.INGESTION,
                output_dir=self.output_dir / "validation",
            ),
            PipelineStageName.RESAMPLING: PipelineStageValidator(
                stage=PipelineStageName.RESAMPLING,
                output_dir=self.output_dir / "validation",
            ),
            PipelineStageName.FEATURE_ENGINEERING: PipelineStageValidator(
                stage=PipelineStageName.FEATURE_ENGINEERING,
                output_dir=self.output_dir / "validation",
            ),
            PipelineStageName.LABELING: PipelineStageValidator(
                stage=PipelineStageName.LABELING,
                output_dir=self.output_dir / "validation",
            ),
            PipelineStageName.DATASET_BUILD: PipelineStageValidator(
                stage=PipelineStageName.DATASET_BUILD,
                output_dir=self.output_dir / "validation",
            ),
        }
        
        # Quality gates
        if self.config.quality_gates.enabled:
            self.quality_gates = {
                "ingestion": create_quality_gates("ingestion", remediation_log_dir=self.config.quality_gates.log_dir),
                "resampling": create_quality_gates("resampling", remediation_log_dir=self.config.quality_gates.log_dir),
                "feature_engineering": create_quality_gates("feature_engineering", remediation_log_dir=self.config.quality_gates.log_dir),
            }
        else:
            self.quality_gates = {}
        
        # Incremental processor
        if self.config.incremental.enabled:
            self.incremental_processor = StreamingFeatureProcessor(
                state_dir=self.config.incremental.state_dir,
                warmup_bars=self.config.incremental.warmup_bars,
            )
        else:
            self.incremental_processor = None
        
        # Feature materializer
        if self.config.feature_store.enabled:
            self.materializer = FeatureMaterializer(
                feature_store_path=self.config.feature_store.path,
                quality_gates_dir=self.config.quality_gates.log_dir,
                lineage_dir=self.config.lineage.path,
            )
        else:
            self.materializer = None
    
    def _init_components(self):
        """Initialize pipeline components"""
        # Data pipeline
        self.data_pipeline = ForexDataPipeline(
            bar_freq=self.config.bars.freq,
            bar_type=self.config.bars.bar_type.value,
            info_bar_threshold=(
                self.config.bars.tick_threshold if self.config.bars.bar_type.value == "tick"
                else self.config.bars.volume_threshold if self.config.bars.bar_type.value == "volume"
                else self.config.bars.dollar_threshold
            ),
            session_filter=self.config.bars.session_filter,
            session_mode=self.config.bars.session_mode,
            session_start_utc=self.config.bars.session_start_utc,
            session_end_utc=self.config.bars.session_end_utc,
            apply_frac_diff=self.config.bars.apply_frac_diff,
            frac_diff_order=self.config.bars.frac_diff_order,
            gap_policy=self.config.bars.gap_policy,
            spread_cap_multiplier=self.config.bars.spread_cap_multiplier,
        )
        
        # Feature engineer
        self.feature_engineer = FeatureEngineer(
            enable_regime_gate=self.config.features.enable_regime_gate,
            enable_quality_gate=self.config.features.enable_quality_gate,
            enable_no_trade_zones=self.config.features.enable_no_trade_zones,
        )
    
    def run(self, pairs: list[str] | None = None) -> ValidationReport:
        """
        Run the complete pipeline.
        
        Args:
            pairs: Currency pairs to process (overrides config)
            
        Returns:
            ValidationReport with pipeline results
        """
        pairs = pairs or self.config.data_source.pairs
        run_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"[Pipeline] Starting run {run_id} for pairs: {pairs}")
        
        # Initialize stages
        self.stages = {
            PipelineStageName.INGESTION: PipelineStage(name=PipelineStageName.INGESTION),
            PipelineStageName.RESAMPLING: PipelineStage(name=PipelineStageName.RESAMPLING),
            PipelineStageName.FEATURE_ENGINEERING: PipelineStage(name=PipelineStageName.FEATURE_ENGINEERING),
            PipelineStageName.LABELING: PipelineStage(name=PipelineStageName.LABELING),
            PipelineStageName.DATASET_BUILD: PipelineStage(name=PipelineStageName.DATASET_BUILD),
        }
        
        if self.config.feature_store.enabled:
            self.stages[PipelineStageName.MATERIALIZATION] = PipelineStage(
                name=PipelineStageName.MATERIALIZATION
            )
        
        if self.config.validation.drift_detection_enabled:
            self.stages[PipelineStageName.VALIDATION] = PipelineStage(
                name=PipelineStageName.VALIDATION
            )
        
        stage_data = {}
        
        try:
            # Run each stage
            for pair in pairs:
                print(f"\n[Pipeline] Processing {pair}...")
                
                # Stage 1: Ingestion
                self._run_stage(PipelineStageName.INGESTION, pair, lambda: self._stage_ingestion(pair))
                
                # Stage 2: Resampling
                self._run_stage(PipelineStageName.RESAMPLING, pair, lambda: self._stage_resampling(pair))
                
                # Stage 3: Feature Engineering
                self._run_stage(PipelineStageName.FEATURE_ENGINEERING, pair, lambda: self._stage_features(pair))
                
                # Stage 4: Labeling
                self._run_stage(PipelineStageName.LABELING, pair, lambda: self._stage_labeling(pair))
                
                # Stage 5: Dataset Build
                self._run_stage(PipelineStageName.DATASET_BUILD, pair, lambda: self._stage_dataset(pair))
            
            # Stage 6: Materialization (all pairs)
            if self.config.feature_store.enabled:
                self._run_stage(PipelineStageName.MATERIALIZATION, "all", lambda: self._stage_materialization(pairs))
            
            # Stage 7: Validation (drift detection)
            if self.config.validation.drift_detection_enabled:
                self._run_stage(PipelineStageName.VALIDATION, "all", lambda: self._stage_validation())
            
            # Generate final report
            report = self.reporter.generate_report(
                stage_validators={k: v for k, v in self.validators.items() if k in self.stages},
                run_id=run_id,
            )
            
            # Save report
            self.reporter.save_report(report)
            self.reporter.save_html_report(report)
            
            print(f"\n[Pipeline] Run {run_id} completed: {report.overall_status}")
            
            return report
            
        except Exception as e:
            print(f"[Pipeline] Run failed: {e}")
            raise
    
    def _run_stage(self, stage_name: PipelineStageName, pair: str, stage_fn: Callable):
        """Run a pipeline stage with validation and error handling"""
        stage = self.stages[stage_name]
        stage.status = StageStatus.RUNNING
        stage.start_time = datetime.now()
        
        print(f"  [{stage_name.value}] Starting...")
        
        try:
            output = stage_fn()
            stage.output_data = output
            stage.status = StageStatus.COMPLETED
            stage.end_time = datetime.now()
            stage.duration_seconds = (stage.end_time - stage.start_time).total_seconds()
            
            # Run validation
            if stage_name in self.validators and output is not None:
                if isinstance(output, dict):
                    # For multi-pair stages, validate each
                    for p, df in output.items():
                        if isinstance(df, pl.DataFrame):
                            self.validators[stage_name].validate(df, pair=p)
                elif isinstance(output, pl.DataFrame):
                    self.validators[stage_name].validate(output, pair=pair)
            
            # Run quality gates
            if stage_name.value in self.quality_gates and output is not None:
                if isinstance(output, pl.DataFrame):
                    output, qc_report = self.quality_gates[stage_name.value].run(output, pair=pair)
                    stage.quality_report = qc_report
                    stage.output_data = output
            
            # Drift detection
            if self.config.validation.drift_detection_enabled and isinstance(output, pl.DataFrame):
                drift_report = self.drift_detector.detect_drift(
                    output, stage_name.value, pair, save_as_reference=True
                )
                stage.metadata["drift"] = drift_report.to_dict()
            
            # Lineage tracking
            if self.lineage_tracker and isinstance(output, pl.DataFrame):
                self.lineage_tracker.record_event(LineageEvent(
                    event_type=LineageEventType.TRANSFORM,
                    stage=stage_name.value,
                    pair=pair if pair != "all" else None,
                    output_data=output,
                    metadata={"duration_ms": stage.duration_seconds * 1000},
                ))
            
            print(f"  [{stage_name.value}] Completed in {stage.duration_seconds:.1f}s")
            
        except Exception as e:
            stage.status = StageStatus.FAILED
            stage.end_time = datetime.now()
            stage.duration_seconds = (stage.end_time - stage.start_time).total_seconds()
            stage.error = str(e)
            
            print(f"  [{stage_name.value}] Failed: {e}")
            raise
    
    # Stage implementations
    
    def _stage_ingestion(self, pair: str) -> pl.DataFrame:
        """Load raw tick data"""
        source_config = self.config.data_source
        
        if source_config.type.value == "synthetic":
            return load_or_generate(
                n_rows=source_config.synthetic_rows,
                pair=pair,
                base_price=source_config.base_price,
                spread_pips=source_config.spread_pips,
            )
        else:
            return load_or_generate(
                source=source_config.type.value,
                pair=pair,
                start=source_config.start_date,
                end=source_config.end_date,
            )
    
    def _stage_resampling(self, pair: str) -> pl.DataFrame:
        """Resample ticks to bars"""
        # Get ingestion output
        ingestion_stage = self.stages[PipelineStageName.INGESTION]
        ticks = ingestion_stage.output_data
        
        if ticks is None:
            raise ValueError("No tick data from ingestion stage")
        
        return self.data_pipeline.run(ticks, pair=pair)
    
    def _stage_features(self, pair: str) -> pl.DataFrame:
        """Compute features"""
        resampling_stage = self.stages[PipelineStageName.RESAMPLING]
        bars = resampling_stage.output_data
        
        if bars is None:
            raise ValueError("No bar data from resampling stage")
        
        # Check for incremental processing
        if self.incremental_processor:
            features = self.incremental_processor.process_bar(bars, pair)
            if features is not None:
                return features
        
        return self.feature_engineer.build(bars, pair=pair)
    
    def _stage_labeling(self, pair: str) -> pl.DataFrame:
        """Compute labels"""
        features_stage = self.stages[PipelineStageName.FEATURE_ENGINEERING]
        features = features_stage.output_data
        
        if features is None:
            raise ValueError("No feature data from feature engineering stage")
        
        bars_stage = self.stages[PipelineStageName.RESAMPLING]
        bars = bars_stage.output_data
        
        if bars is None:
            raise ValueError("No bar data for labeling")
        
        label_config = self.config.labeling
        
        if label_config.method.value in ["rl_reward", "both"]:
            labels = compute_rl_reward_labels_regime(
                features=features,
                bars=bars,
                lookahead_bars=label_config.lookahead_bars,
                profit_target_atr=label_config.profit_target_atr,
                stop_loss_atr=label_config.stop_loss_atr,
                transaction_cost_pips=label_config.transaction_cost_pips,
            )
        
        if label_config.method.value in ["triple_barrier", "both"]:
            tb_labels = compute_triple_barrier_labels(
                features=features,
                bars=bars,
                profit_target_atr=label_config.tb_profit_atr,
                stop_loss_atr=label_config.tb_stop_atr,
                horizon_bars=label_config.tb_horizon_bars,
            )
            
            if label_config.method.value == "both":
                # Combine labels
                labels = labels.join(tb_labels, on="timestamp_utc", how="outer", coalesce=True)
            else:
                labels = tb_labels
        
        return labels
    
    def _stage_dataset(self, pair: str) -> pl.DataFrame:
        """Build training dataset"""
        features_stage = self.stages[PipelineStageName.FEATURE_ENGINEERING]
        features = features_stage.output_data
        
        labeling_stage = self.stages[PipelineStageName.LABELING]
        labels = labeling_stage.output_data
        
        if features is None or labels is None:
            raise ValueError("Missing features or labels for dataset building")
        
        # This would integrate with the actual dataset_builder
        # For now, return aligned features + labels
        dataset = features.join(labels, on="timestamp_utc", how="inner")
        
        return dataset
    
    def _stage_materialization(self, pairs: list[str]) -> dict[str, pl.DataFrame]:
        """Materialize features to feature store"""
        if not self.materializer:
            raise ValueError("Feature materializer not initialized")
        
        job = MaterializationJob(
            version=f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            pairs=pairs,
            start_date=datetime.fromisoformat(self.config.data_source.start_date),
            end_date=datetime.fromisoformat(self.config.data_source.end_date),
            bar_freq=self.config.bars.freq,
            description=f"Pipeline materialization for {', '.join(pairs)}",
            tags={"pipeline": "true"},
        )
        
        result = self.materializer.materialize(job)
        
        if not result.success:
            raise RuntimeError(f"Materialization failed: {result.errors}")
        
        return {pair: None for pair in pairs}  # Data already in feature store
    
    def _stage_validation(self) -> dict:
        """Run validation checks"""
        results = {}
        
        # Check drift for all stages
        for stage_name, stage in self.stages.items():
            if stage.output_data is not None and isinstance(stage.output_data, pl.DataFrame):
                drift_report = self.drift_detector.detect_drift(
                    stage.output_data, stage_name.value, 
                    save_as_reference=True
                )
                results[stage_name.value] = drift_report.to_dict()
        
        return results
    
    def get_pipeline_status(self) -> dict:
        """Get current pipeline status"""
        return {
            "stages": {k.value: v.to_dict() for k, v in self.stages.items()},
            "overall": "completed" if all(s.status == StageStatus.COMPLETED for s in self.stages.values()) else
                      "failed" if any(s.status == StageStatus.FAILED for s in self.stages.values()) else
                      "running",
        }
    
    def save_status(self, filepath: str | Path):
        """Save pipeline status to file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        status = self.get_pipeline_status()
        with open(filepath, "w") as f:
            json.dump(status, f, indent=2, default=str)


# Convenience function
def create_orchestrator(
    config: PipelineConfig | None = None,
    config_path: str | Path | None = None,
) -> PipelineOrchestrator:
    """Create a pipeline orchestrator"""
    return PipelineOrchestrator(config=config, config_path=config_path)