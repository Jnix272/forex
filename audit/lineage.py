"""
audit/lineage.py — data lineage, model registry records, decision trail.

Standard-library only (runs in CI / recovery shells without heavy deps).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LineageStep:
    """One processing step in the data → training pipeline."""
    step: str
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    data_hash: str = ""
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "name": self.name,
            "params": self.params,
            "data_hash": self.data_hash,
            "timestamp": self.timestamp,
        }


@dataclass
class DecisionRecord:
    """One audited decision (promotion / rollback / risk / drift)."""
    decision: str            # promote | rollback | risk_block | drift_alert ...
    model: str
    decision_made: bool
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "model": self.model,
            "decision_made": self.decision_made,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class DataLineage:
    """Record the provenance chain for a training run.

    Usage:
        lineage = DataLineage(dataset="EURUSD_1m", dataset_version="2026.07")
        lineage.add_step("preprocess", "clean_v2", params={"outliers": "mad"})
        lineage.add_step("feature_set", "features_v3", data_hash=features_hash)
        lineage.add_step("label", "label_v4", params={"horizon": 12})
        lineage.record_training_run(run_id="r123", params={"lr": 1e-4},
                                    seed=42, commit="abc123", env={"gpu": "A100"})
        lineage.save(path)
    """

    def __init__(self, dataset: str = "", dataset_version: str = "",
                 dataset_hash: str = ""):
        self.dataset = dataset
        self.dataset_version = dataset_version
        self.dataset_hash = dataset_hash
        self.steps: List[LineageStep] = []
        self.training_runs: List[Dict[str, Any]] = []
        self.created_at: str = _now_iso()

    def add_step(self, step: str, name: str, params: Optional[Dict[str, Any]] = None,
                 data_hash: str = "") -> LineageStep:
        s = LineageStep(step=step, name=name, params=params or {}, data_hash=data_hash)
        self.steps.append(s)
        return s

    def record_training_run(self, run_id: str, params: Optional[Dict[str, Any]] = None,
                            seed: Optional[int] = None, commit: Optional[str] = None,
                            env: Optional[Dict[str, Any]] = None,
                            model: str = "unknown") -> Dict[str, Any]:
        run = {
            "run_id": run_id,
            "model": model,
            "params": params or {},
            "seed": seed,
            "commit": commit,
            "env": env or {},
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "dataset_hash": self.dataset_hash,
            "steps": [s.to_dict() for s in self.steps],
            "timestamp": _now_iso(),
        }
        self.training_runs.append(run)
        return run

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "dataset_hash": self.dataset_hash,
            "created_at": self.created_at,
            "steps": [s.to_dict() for s in self.steps],
            "training_runs": self.training_runs,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "DataLineage":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        lineage = cls(data.get("dataset", ""), data.get("dataset_version", ""),
                      data.get("dataset_hash", ""))
        lineage.created_at = data.get("created_at", _now_iso())
        lineage.steps = [
            LineageStep(step=s.get("step", ""), name=s.get("name", ""),
                        params=s.get("params", {}), data_hash=s.get("data_hash", ""),
                        timestamp=s.get("timestamp", _now_iso()))
            for s in data.get("steps", [])
        ]
        lineage.training_runs = list(data.get("training_runs", []))
        return lineage


def ModelRegistryRecord(
    model: str,
    run_id: str,
    params: Optional[Dict[str, Any]] = None,
    data_hash: Optional[str] = None,
    code_commit: Optional[str] = None,
    seed: Optional[int] = None,
    env: Optional[Dict[str, Any]] = None,
    dataset_hash: Optional[str] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Model registry hook — a flat, queryable record of one model artifact."""
    return {
        "model": model,
        "run_id": run_id,
        "params": params or {},
        "data_hash": data_hash or "",
        "dataset_hash": dataset_hash or "",
        "code_commit": code_commit or "",
        "seed": seed,
        "env": env or {},
        "created_at": created_at or _now_iso(),
    }


def decision_trail(
    model: str,
    decision: str,
    decision_made: bool,
    details: Optional[Dict[str, Any]] = None,
    history: Optional[List[DecisionRecord]] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Append a decision to an audit trail (and optionally persist to JSON).

    ``history`` may be an existing list of DecisionRecord dicts; returns a dict
    with the full trail so callers can store it (e.g. alongside checkpoints).
    """
    record = DecisionRecord(decision=decision, model=model,
                            decision_made=decision_made, details=details or {})
    trail: List[Dict[str, Any]] = []
    if history:
        for h in history:
            if isinstance(h, DecisionRecord):
                trail.append(h.to_dict())
            else:
                trail.append(dict(h))
    trail.append(record.to_dict())

    result = {
        "model": model,
        "decision": decision,
        "decision_made": decision_made,
        "trail": trail,
        "timestamp": record.timestamp,
    }
    if path:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
    return result
