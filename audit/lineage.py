"""
audit/lineage.py - DEPRECATED shim.

The lineage-tracking implementation moved to the ``lineage`` package
(``lineage.provenance``). This module re-exports the merged API and emits a
DeprecationWarning once per attribute access so callers migrate to:

    from lineage import DataLineage, DecisionRecord, LineageStep, \\
        ModelRegistryRecord, decision_trail
"""

from __future__ import annotations

import warnings
from typing import Any

from lineage.provenance import (
    DataLineage,
    DecisionRecord,
    LineageStep,
    ModelRegistryRecord as _ModelRegistryRecord,
    _now_iso,
    decision_trail as _decision_trail,
)

__all__ = [
    "DataLineage",
    "DecisionRecord",
    "LineageStep",
    "ModelRegistryRecord",
    "decision_trail",
]


def _warn(name: str) -> None:
    warnings.warn(
        f"audit.lineage.{name} is deprecated; use lineage.{name} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def ModelRegistryRecord(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Deprecated wrapper around :func:`lineage.ModelRegistryRecord`."""
    _warn("ModelRegistryRecord")
    return _ModelRegistryRecord(*args, **kwargs)


def decision_trail(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Deprecated wrapper around :func:`lineage.decision_trail`."""
    _warn("decision_trail")
    return _decision_trail(*args, **kwargs)


# DataLineage / DecisionRecord / LineageStep are classes; wrap them lazily via
# module __getattr__ so the DeprecationWarning fires exactly once on first use.
_deprecated_classes = {
    "DataLineage": DataLineage,
    "DecisionRecord": DecisionRecord,
    "LineageStep": LineageStep,
}
_warned: set[str] = set()


def __getattr__(name: str) -> Any:
    if name in _deprecated_classes:
        if name not in _warned:
            _warned.add(name)
            warnings.warn(
                f"audit.lineage.{name} is deprecated; use lineage.{name} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return _deprecated_classes[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
