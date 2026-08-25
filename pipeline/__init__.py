"""
Configuration-Driven Pipeline Package
=====================================
Pipeline orchestration from YAML configuration.
"""

from pipeline.config import PipelineConfig, load_pipeline_config
from pipeline.orchestrator import PipelineOrchestrator, PipelineStage, StageStatus
from pipeline.quality_gates import DataQualityGates, create_quality_gates

__all__ = [
    "DataQualityGates",
    "PipelineConfig",
    "PipelineOrchestrator",
    "PipelineStage",
    "StageStatus",
    "create_quality_gates",
    "load_pipeline_config",
]

# Dependency inversion (R3): register the concrete quality-gate factory so
# feature_store.materializer can resolve gates without importing pipeline.
from feature_store.gates import set_quality_gate_factory  # noqa: E402

set_quality_gate_factory(create_quality_gates)
