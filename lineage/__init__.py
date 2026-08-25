"""
Data Lineage Package
====================
Single source of truth for data provenance tracking:
raw ticks → features → trained models.

Public surface:
- Event-based tracking: LineageTracker / LineageEvent / LineageRecord,
  persisted via FileLineageStore / SQLiteLineageStore.
- Provenance/audit records (ported from the deprecated audit.lineage):
  DataLineage / LineageStep / DecisionRecord, ModelRegistryRecord,
  decision_trail.
"""

from lineage.provenance import (
    DataLineage,
    DecisionRecord,
    LineageStep,
    ModelRegistryRecord,
    decision_trail,
)
from lineage.store import FileLineageStore, LineageStore, SQLiteLineageStore
from lineage.tracker import LineageEvent, LineageRecord, LineageTracker

__all__ = [
    "DataLineage",
    "DecisionRecord",
    "FileLineageStore",
    "LineageEvent",
    "LineageRecord",
    "LineageStep",
    "LineageStore",
    "LineageTracker",
    "ModelRegistryRecord",
    "SQLiteLineageStore",
    "decision_trail",
]
