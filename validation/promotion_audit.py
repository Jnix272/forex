"""Priority 2 validation and promotion artifact auditor.

This module is intentionally standard-library only so it can run in CI, dry
promotion jobs, and recovery shells without importing torch/numpy/pandas.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_ARTIFACTS = (
    "train_summary.json",
    "fold_selection.json",
    "promotion_gate.json",
    "deployment.json",
    "manifest.json",
)


@dataclass(frozen=True)
class CalibrationGateConfig:
    max_ece: float = 0.08
    max_nll: float = 1.25
    max_confidence_accuracy_gap: float = 0.10


@dataclass(frozen=True)
class PromotionAuditConfig:
    allow_proxy_gate: bool = False
    require_deployment_success: bool = True
    require_model_diagnostics: bool = True
    require_leaderboard_rank: bool = False
    max_leaderboard_rank: int = 1
    calibration: CalibrationGateConfig = CalibrationGateConfig()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return data


def _nested_get(data: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


def _model_name(*docs: Mapping[str, Any]) -> str | None:
    for doc in docs:
        for key in ("model", "model_name", "name"):
            value = doc.get(key)
            if value:
                return str(value).lower()
        value = _nested_get(doc, "model", "name")
        if value:
            return str(value).lower()
    return None


def _calibration_from(train_summary: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    calibration = train_summary.get("calibration")
    if isinstance(calibration, Mapping) and calibration:
        return dict(calibration)
    calibration = diagnostics.get("calibration")
    if isinstance(calibration, Mapping):
        return dict(calibration)
    return {}


def _gate_input_type(promotion_gate: Mapping[str, Any]) -> str:
    for path in (
        ("gate_input_type",),
        ("input_type",),
        ("details", "gate_input_type"),
        ("details", "input_type"),
        ("metadata", "gate_input_type"),
    ):
        value = _nested_get(promotion_gate, *path)
        if value:
            return str(value).lower()
    return "unknown"


def _leaderboard_rank(diagnostics: Mapping[str, Any], model_name: str | None) -> int | None:
    rows = diagnostics.get("leaderboard")
    if not isinstance(rows, list) or not model_name:
        return None
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        row_model = str(row.get("model", row.get("model_name", ""))).lower()
        if row_model == model_name:
            try:
                return int(row.get("rank"))
            except Exception:
                return None
    return None


def validate_priority2_promotion(
    artifact_dir: str | Path,
    config: PromotionAuditConfig | None = None,
) -> dict[str, Any]:
    """Validate the Priority 2 promotion artifact chain.

    Returns a JSON-serializable report with `ready_for_production`, per-gate
    booleans, reasons, artifact paths, and calibration diagnostics.
    """
    cfg = config or PromotionAuditConfig()
    root = Path(artifact_dir)
    reasons = []
    gates: dict[str, bool] = {}
    docs: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, str] = {}

    for filename in REQUIRED_ARTIFACTS:
        path = root / filename
        artifacts[filename] = str(path)
        gates[f"artifact_{filename}"] = path.exists()
        if path.exists():
            try:
                docs[filename] = _read_json(path)
            except Exception as exc:
                docs[filename] = {}
                gates[f"artifact_{filename}"] = False
                reasons.append(f"{filename}: invalid JSON ({exc})")
        else:
            reasons.append(f"{filename}: missing")

    diagnostics_path = root / "model_diagnostics_report.json"
    artifacts["model_diagnostics_report.json"] = str(diagnostics_path)
    diagnostics = _read_json(diagnostics_path) if diagnostics_path.exists() else {}
    gates["model_diagnostics_present"] = diagnostics_path.exists() or not cfg.require_model_diagnostics
    if cfg.require_model_diagnostics and not diagnostics_path.exists():
        reasons.append("model_diagnostics_report.json: missing")

    train_summary = docs.get("train_summary.json", {})
    promotion_gate = docs.get("promotion_gate.json", {})
    deployment = docs.get("deployment.json", {})
    manifest = docs.get("manifest.json", {})

    promoted = bool(promotion_gate.get("promoted"))
    gates["promotion_gate_passed"] = promoted
    if not promoted:
        reasons.append("promotion_gate.json: promoted is not true")

    gate_input = _gate_input_type(promotion_gate)
    _ok_inputs = {"execution_backtest", "execution", "full_backtest"}
    gates["gate_input_not_proxy"] = cfg.allow_proxy_gate or gate_input in _ok_inputs
    if not gates["gate_input_not_proxy"]:
        reasons.append(
            f"promotion_gate.json: gate_input_type={gate_input!r} "
            f"(need execution_backtest; proxy/unknown cannot deploy without override)"
        )

    deploy_status = str(deployment.get("status", "")).lower()
    gates["deployment_success"] = (deploy_status == "success") or not cfg.require_deployment_success
    if cfg.require_deployment_success and deploy_status != "success":
        reasons.append(f"deployment.json: status is {deploy_status or 'missing'}")

    model = _model_name(train_summary, promotion_gate, manifest)
    manifest_model = _model_name(manifest)
    gates["manifest_model_consistent"] = not (model and manifest_model) or model == manifest_model
    if not gates["manifest_model_consistent"]:
        reasons.append(f"manifest.json: model mismatch ({manifest_model} != {model})")

    calibration = _calibration_from(train_summary, diagnostics)
    ece = float(calibration.get("ece", 999.0) or 999.0)
    nll = float(calibration.get("nll", 999.0) or 999.0)
    accuracy = calibration.get("accuracy")
    confidence = calibration.get("avg_confidence", calibration.get("confidence"))
    gap = 999.0
    if accuracy is not None and confidence is not None:
        gap = abs(float(confidence) - float(accuracy))

    gates["calibration_ece_ok"] = ece <= cfg.calibration.max_ece
    gates["calibration_nll_ok"] = nll <= cfg.calibration.max_nll
    gates["calibration_gap_ok"] = gap <= cfg.calibration.max_confidence_accuracy_gap
    if not gates["calibration_ece_ok"]:
        reasons.append(f"calibration: ece {ece:.6g} > {cfg.calibration.max_ece:.6g}")
    if not gates["calibration_nll_ok"]:
        reasons.append(f"calibration: nll {nll:.6g} > {cfg.calibration.max_nll:.6g}")
    if not gates["calibration_gap_ok"]:
        reasons.append(
            f"calibration: confidence/accuracy gap {gap:.6g} "
            f"> {cfg.calibration.max_confidence_accuracy_gap:.6g}"
        )

    rank = _leaderboard_rank(diagnostics, model)
    gates["leaderboard_rank_ok"] = (
        not cfg.require_leaderboard_rank
        or (rank is not None and rank <= cfg.max_leaderboard_rank)
    )
    if not gates["leaderboard_rank_ok"]:
        reasons.append(
            f"model_diagnostics_report.json: leaderboard rank {rank} "
            f"> {cfg.max_leaderboard_rank}"
        )

    ready = all(gates.values())
    return {
        "ready_for_production": ready,
        "model": model or "unknown",
        "artifact_dir": str(root),
        "gates": gates,
        "reasons": reasons,
        "gate_input_type": gate_input,
        "calibration": {
            "ece": ece,
            "nll": nll,
            "confidence_accuracy_gap": gap,
            "thresholds": {
                "max_ece": cfg.calibration.max_ece,
                "max_nll": cfg.calibration.max_nll,
                "max_confidence_accuracy_gap": cfg.calibration.max_confidence_accuracy_gap,
            },
        },
        "leaderboard_rank": rank,
        "artifacts": artifacts,
    }


def write_priority2_promotion_report(
    artifact_dir: str | Path,
    output_path: str | Path | None = None,
    config: PromotionAuditConfig | None = None,
) -> dict[str, Any]:
    """Validate artifacts and write `priority2_promotion_report.json`."""
    report = validate_priority2_promotion(artifact_dir, config=config)
    out = Path(output_path) if output_path else Path(artifact_dir) / "priority2_promotion_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
