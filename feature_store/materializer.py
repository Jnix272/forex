"""
Feature Materializer
====================
Materializes features from raw data and stores in feature store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from feature_store.store import FeatureStore, ParquetFeatureStore
from feature_store.registry import FeatureRegistry, get_registry
from features.feature_engineering_pl import FeatureEngineer
from contracts.validation.gates import PipelineStageValidator
from pipeline.quality_gates import DataQualityGates, create_quality_gates
from lineage.tracker import LineageTracker, LineageEventType, LineageEvent


@dataclass
class MaterializationJob:
    """Configuration for a materialization job"""
    version: str
    pairs: list[str]
    start_date: datetime
    end_date: datetime
    bar_freq: str = "1min"
    description: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    feature_engineer_config: dict | None = None
    data_source: str = "dukascopy"
    quality_gates: bool = True
    lineage_tracking: bool = True


@dataclass
class MaterializationResult:
    """Result of a materialization job"""
    job: MaterializationJob
    success: bool
    version: str
    pairs_processed: list[str]
    total_rows: int
    total_features: int
    duration_seconds: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    quality_reports: dict[str, Any] = field(default_factory=dict)
    lineage: dict | None = None


class FeatureMaterializer:
    """
    Materializes features from raw data and stores in feature store.
    
    Orchestrates the full pipeline: data loading -> feature engineering -> 
    quality gates -> feature store write.
    """
    
    def __init__(
        self,
        feature_store: FeatureStore | None = None,
        feature_store_path: str | Path = "./feature_store",
        feature_engineer: FeatureEngineer | None = None,
        data_loader: Any = None,
        quality_gates_dir: str | Path | None = None,
        lineage_dir: str | Path | None = None,
    ):
        self.feature_store = feature_store or ParquetFeatureStore(feature_store_path)
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.data_loader = data_loader
        self.quality_gates_dir = quality_gates_dir
        self.lineage_dir = lineage_dir
        
        # Initialize quality gates
        self.quality_gates = {
            "ingestion": create_quality_gates("ingestion", remediation_log_dir=quality_gates_dir),
            "resampling": create_quality_gates("resampling", remediation_log_dir=quality_gates_dir),
            "feature_engineering": create_quality_gates("feature_engineering", remediation_log_dir=quality_gates_dir),
        }
        
        # Initialize lineage tracker
        self.lineage_tracker = None
        if lineage_dir:
            from lineage.store import FileLineageStore
            self.lineage_tracker = LineageTracker(
                store=FileLineageStore(Path(lineage_dir) / "lineage.jsonl")
            )
    
    def materialize(self, job: MaterializationJob) -> MaterializationResult:
        """
        Run a materialization job.
        
        Args:
            job: Materialization job configuration
            
        Returns:
            MaterializationResult with job outcome
        """
        start_time = datetime.now()
        
        # Initialize lineage
        if self.lineage_tracker:
            self.lineage_tracker.record_event(LineageEvent(
                event_type=LineageEventType.DATASET_BUILD,
                stage="materialization_start",
                metadata={"job": job.__dict__},
            ))
        
        all_features = []
        pairs_processed = []
        total_rows = 0
        total_features = 0
        errors = []
        warnings = []
        quality_reports = {}
        
        for pair in job.pairs:
            try:
                print(f"[Materializer] Processing {pair}...")
                
                # Load raw data
                raw_data = self._load_raw_data(pair, job.start_date, job.end_date, job.data_source)
                if raw_data is None or len(raw_data) == 0:
                    warnings.append(f"No data for {pair}")
                    continue
                
                # Quality gate: ingestion
                if job.quality_gates:
                    raw_data, qc_report = self.quality_gates["ingestion"].run(raw_data, pair=pair)
                    quality_reports[f"{pair}_ingestion"] = qc_report.to_dict()
                
                # Resample to bars
                from data.data_ingestion import ForexDataPipeline
                pipeline = ForexDataPipeline(bar_freq=job.bar_freq)
                bars = pipeline.run(raw_data, pair=pair)
                
                # Quality gate: resampling
                if job.quality_gates:
                    bars, qc_report = self.quality_gates["resampling"].run(bars, pair=pair)
                    quality_reports[f"{pair}_resampling"] = qc_report.to_dict()
                
                # Feature engineering
                features = self.feature_engineer.build(bars, pair=pair)
                
                # Quality gate: feature engineering
                if job.quality_gates:
                    features, qc_report = self.quality_gates["feature_engineering"].run(features, pair=pair)
                    quality_reports[f"{pair}_feature_engineering"] = qc_report.to_dict()
                
                # Register features in registry
                self._register_features(features, pair)
                
                # Add pair identifier
                features = features.with_columns(pl.lit(pair).alias("pair"))
                
                all_features.append(features)
                pairs_processed.append(pair)
                total_rows += len(features)
                total_features = len(features.columns)
                
                # Lineage tracking
                if self.lineage_tracker:
                    self.lineage_tracker.record_event(LineageEvent(
                        event_type=LineageEventType.FEATURE_COMPUTE,
                        stage="feature_engineering",
                        pair=pair,
                        input_data=bars,
                        output_data=features,
                        metadata={"n_features": total_features},
                    ))
                
            except Exception as e:
                error_msg = f"Failed to process {pair}: {e}"
                print(f"[Materializer] {error_msg}")
                errors.append(error_msg)
        
        if not all_features:
            return MaterializationResult(
                job=job,
                success=False,
                version=job.version,
                pairs_processed=[],
                total_rows=0,
                total_features=0,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                errors=["No pairs successfully processed"],
            )
        
        # Combine all features
        combined = pl.concat(all_features, how="vertical_relaxed")
        combined = combined.sort(["pair", "timestamp_utc"])
        
        # Write to feature store
        try:
            lineage_data = {}
            if self.lineage_tracker:
                lineage_data = self.lineage_tracker.get_lineage_graph()
            
            version_meta = self.feature_store.write(
                combined,
                version=job.version,
                description=job.description,
                tags=job.tags,
                lineage=lineage_data,
            )
            
            # Final lineage event
            if self.lineage_tracker:
                self.lineage_tracker.record_event(LineageEvent(
                    event_type=LineageEventType.DATASET_BUILD,
                    stage="materialization_complete",
                    output_data=combined,
                    metadata={"version": job.version, "n_rows": len(combined)},
                ))
                
                # Export lineage
                lineage_path = Path(self.lineage_dir) / f"lineage_{job.version}.json"
                self.lineage_tracker.export_json(lineage_path)
            
        except Exception as e:
            return MaterializationResult(
                job=job,
                success=False,
                version=job.version,
                pairs_processed=pairs_processed,
                total_rows=total_rows,
                total_features=total_features,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                errors=[f"Feature store write failed: {e}"],
                quality_reports=quality_reports,
            )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return MaterializationResult(
            job=job,
            success=True,
            version=job.version,
            pairs_processed=pairs_processed,
            total_rows=len(combined),
            total_features=total_features,
            duration_seconds=duration,
            errors=errors,
            warnings=warnings,
            quality_reports=quality_reports,
            lineage=lineage_data if self.lineage_tracker else None,
        )
    
    def _load_raw_data(
        self, 
        pair: str, 
        start_date: datetime, 
        end_date: datetime,
        source: str,
    ) -> pl.DataFrame | None:
        """Load raw data for a pair"""
        if self.data_loader:
            return self.data_loader.load(pair, source, start_date, end_date)
        
        # Fallback to default loader
        try:
            from data.sources import ForexDataManager
            mgr = ForexDataManager(verbose=False)
            return mgr.load(pair, source=source, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        except Exception as e:
            print(f"[Materializer] Failed to load data for {pair}: {e}")
            return None
    
    def _register_features(self, features: pl.DataFrame, pair: str):
        """Register features in global registry"""
        registry = get_registry()
        
        for col in features.columns:
            if col in ["timestamp_utc", "pair"]:
                continue
            
            # Determine category from column name
            category = self._infer_category(col)
            
            registry.register(
                name=col,
                dtype=features.schema[col],
                category=category,
                source="computed",
                overwrite=True,
            )
    
    def _infer_category(self, col: str) -> str:
        """Infer feature category from column name"""
        col_lower = col.lower()
        
        if any(x in col_lower for x in ["ofi", "obi", "vpin", "spread", "liquidity", "kyle", "amihud"]):
            return "microstructure"
        elif any(x in col_lower for x in ["rsi", "macd", "stoch", "williams", "cci", "ret_", "momentum"]):
            return "momentum"
        elif any(x in col_lower for x in ["atr", "vol_", "bb_", "bollinger", "volatility"]):
            return "volatility"
        elif any(x in col_lower for x in ["regime", "hmm", "cpd", "hurst", "fractal"]):
            return "regime"
        elif any(x in col_lower for x in ["cross", "corr", "gold", "dxy", "yield", "carry"]):
            return "cross_asset"
        elif any(x in col_lower for x in ["sentiment", "fb_", "embed_", "news", "eco_", "buzz"]):
            return "sentiment_macro"
        elif any(x in col_lower for x in ["time_", "day_", "session", "london", "ny", "asia"]):
            return "temporal"
        elif any(x in col_lower for x in ["candle", "doji", "hammer", "engulf", "harami", "star", "soldier", "crow"]):
            return "candlestick"
        elif any(x in col_lower for x in ["vp_", "volume_profile", "poc", "vwap"]):
            return "volume_profile"
        elif any(x in col_lower for x in ["circuit", "drawdown", "var_", "position", "risk", "no_trade"]):
            return "risk_control"
        else:
            return "other"
    
    def materialize_incremental(
        self,
        version: str,
        pair: str,
        new_bars: pl.DataFrame,
        base_version: str | None = None,
    ) -> MaterializationResult:
        """
        Incrementally materialize new features for a pair.
        
        Args:
            version: New version to create
            pair: Currency pair
            new_bars: New bars to process
            base_version: Base version to extend (optional)
            
        Returns:
            MaterializationResult
        """
        # Load base features if provided
        base_features = None
        if base_version:
            try:
                base_features, _ = self.feature_store.read_latest(pair=pair)
            except Exception:
                pass
        
        # Process new bars
        features = self.feature_engineer.build(new_bars, pair=pair)
        features = features.with_columns(pl.lit(pair).alias("pair"))
        
        # Combine with base if available
        if base_features is not None and len(base_features) > 0:
            # Remove overlapping timestamps
            new_timestamps = set(features["timestamp_utc"].to_list())
            base_features = base_features.filter(
                ~pl.col("timestamp_utc").is_in(new_timestamps)
            )
            combined = pl.concat([base_features, features], how="vertical_relaxed")
        else:
            combined = features
        
        combined = combined.sort(["pair", "timestamp_utc"])
        
        # Write new version
        version_meta = self.feature_store.write(
            combined,
            version=version,
            description=f"Incremental update for {pair}",
            tags={"incremental": "true", "base_version": base_version or ""},
        )
        
        return MaterializationResult(
            job=MaterializationJob(
                version=version,
                pairs=[pair],
                start_date=new_bars["timestamp_utc"].min(),
                end_date=new_bars["timestamp_utc"].max(),
            ),
            success=True,
            version=version,
            pairs_processed=[pair],
            total_rows=len(combined),
            total_features=len(combined.columns),
            duration_seconds=0,
        )


# Convenience function
def create_materializer(
    feature_store_path: str | Path = "./feature_store",
    feature_engineer: FeatureEngineer | None = None,
    **kwargs
) -> FeatureMaterializer:
    """Create a feature materializer"""
    return FeatureMaterializer(
        feature_store_path=feature_store_path,
        feature_engineer=feature_engineer,
        **kwargs
    )