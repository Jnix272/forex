"""
Quality Gate Factory Registry
=============================

Dependency-inversion seam between ``feature_store`` and ``pipeline``.

``feature_store.materializer`` needs quality-gate objects but must not import
from ``pipeline`` (pipeline already depends on feature_store). Instead, the
pipeline package registers its concrete factory here at import time, and the
materializer resolves gates through :func:`get_quality_gate_factory`.

If no implementation is registered (e.g. feature_store used standalone),
the materializer falls back to a no-op gate so behavior degrades gracefully
rather than raising an ImportError.
"""

from __future__ import annotations

from typing import Any, Protocol

_QUALITY_GATE_FACTORY: Any = None


class QualityGate(Protocol):
    """Minimal interface the materializer expects from a gate."""

    def run(self, data: Any, **kwargs: Any) -> tuple[Any, Any]: ...


def set_quality_gate_factory(factory: Any) -> None:
    """Register the concrete quality-gate factory (called by pipeline/__init__)."""
    global _QUALITY_GATE_FACTORY
    _QUALITY_GATE_FACTORY = factory


def get_quality_gate_factory() -> Any:
    """Return the registered factory, or None if none registered."""
    return _QUALITY_GATE_FACTORY


class NullQualityGate:
    """No-op gate used when no implementation has been registered.

    Mirrors the duck-type of pipeline.quality_gates.DataQualityGates.run,
    returning (data_unchanged, report) with a minimal to_dict-able report.
    """

    def __init__(self, name: str = "null", remediation_log_dir: Any = None):
        self.name = name
        self.remediation_log_dir = remediation_log_dir

    def run(self, data: Any, **kwargs: Any) -> tuple[Any, dict]:
        return data, {"gate": self.name, "passed": True, "checks": [], "remediations": []}


def create_quality_gates_default(stage: str, remediation_log_dir: Any = None) -> QualityGate:
    """Resolve a gate for `stage` via the registered factory or the null fallback."""
    factory = get_quality_gate_factory()
    if factory is not None:
        return factory(stage, remediation_log_dir=remediation_log_dir)
    return NullQualityGate(stage, remediation_log_dir)
