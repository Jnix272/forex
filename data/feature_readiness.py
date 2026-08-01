"""Priority 4 data and feature readiness gate.

Combines dataset manifest, feature schema, data quality, and pair readiness
artifacts into one pre-training decision report.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


@dataclass(frozen=True)
class DataFeatureGateConfig:
    expected_seq_len: Optional[int] = None
    expected_schema_hash: Optional[str] = None
    expected_feature_count: Optional[int] = None
    max_feature_nan_rate: float = 0.05
    max_label_class_share: float = 0.80
    max_missing_bars_per_pair: int = 0
    allow_pair_warnings: bool = True


def feature_schema_hash(ordered_features: Iterable[str]) -> str:
    return hashlib.sha256(",".join(str(x) for x in ordered_features).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return data


def _max_numeric_leaf(data: Any) -> float:
    if isinstance(data, Mapping):
        values = [_max_numeric_leaf(v) for v in data.values()]
        return max(values) if values else 0.0
    if isinstance(data, list):
        values = [_max_numeric_leaf(v) for v in data]
        return max(values) if values else 0.0
    try:
        return float(data)
    except Exception:
        return 0.0


def _pair_rows(report: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    if "pairs" in report and isinstance(report["pairs"], Mapping):
        return {str(k): v for k, v in report["pairs"].items() if isinstance(v, Mapping)}
    return {str(k): v for k, v in report.items() if isinstance(v, Mapping)}


def validate_data_feature_readiness(
    artifact_dir: str | Path,
    config: DataFeatureGateConfig | None = None,
) -> Dict[str, Any]:
    """Validate Priority 4 data/features before training starts."""
    cfg = config or DataFeatureGateConfig()
    root = Path(artifact_dir)
    reasons = []
    gates: Dict[str, bool] = {}
    artifacts: Dict[str, str] = {}

    paths = {
        "dataset_manifest": root / "dataset_manifest.json",
        "feature_schema": root / "feature_schema.json",
        "data_quality_report": root / "data_quality_report.json",
        "pair_readiness_report": root / "pair_readiness_report.json",
    }
    docs: Dict[str, Dict[str, Any]] = {}
    for name, path in paths.items():
        artifacts[f"{name}.json"] = str(path)
        gates[f"{name}_present"] = path.exists()
        if not path.exists():
            reasons.append(f"{path.name}: missing")
            docs[name] = {}
            continue
        try:
            docs[name] = _read_json(path)
        except Exception as exc:
            gates[f"{name}_present"] = False
            docs[name] = {}
            reasons.append(f"{path.name}: invalid JSON ({exc})")

    manifest = docs["dataset_manifest"]
    schema = docs["feature_schema"]
    quality = docs["data_quality_report"]
    pair_report = docs["pair_readiness_report"]

    seq_len = manifest.get("sequence_length", manifest.get("seq_len"))
    gates["sequence_length_ok"] = cfg.expected_seq_len is None or seq_len == cfg.expected_seq_len
    if not gates["sequence_length_ok"]:
        reasons.append(f"dataset_manifest.json: sequence_length {seq_len} != {cfg.expected_seq_len}")

    ordered_features = schema.get("ordered_features") or schema.get("features") or []
    computed_hash = feature_schema_hash(ordered_features) if ordered_features else None
    manifest_hash = manifest.get("schema_hash")
    schema_hash = schema.get("schema_hash")
    expected_hash = cfg.expected_schema_hash or manifest_hash or schema_hash
    gates["schema_hash_ok"] = bool(expected_hash) and schema_hash == expected_hash and (
        computed_hash is None or computed_hash == expected_hash
    )
    if not gates["schema_hash_ok"]:
        reasons.append(
            "feature_schema.json: schema hash mismatch "
            f"(manifest={manifest_hash}, schema={schema_hash}, computed={computed_hash})"
        )

    feature_count = schema.get("feature_count", len(ordered_features) if ordered_features else None)
    manifest_feature_count = manifest.get("feature_count")
    gates["feature_count_ok"] = (
        (cfg.expected_feature_count is None or feature_count == cfg.expected_feature_count)
        and (manifest_feature_count is None or feature_count == manifest_feature_count)
    )
    if not gates["feature_count_ok"]:
        reasons.append(
            "feature_schema.json: feature count mismatch "
            f"(manifest={manifest_feature_count}, schema={feature_count}, expected={cfg.expected_feature_count})"
        )

    nan_rate = _max_numeric_leaf(quality.get("feature_nan_rates", {}))
    gates["feature_nan_rate_ok"] = nan_rate <= cfg.max_feature_nan_rate
    if not gates["feature_nan_rate_ok"]:
        reasons.append(f"data_quality_report.json: feature nan rate {nan_rate:.6g} > {cfg.max_feature_nan_rate:.6g}")

    label_share = _max_numeric_leaf(quality.get("label_class_balance", {}))
    gates["label_balance_ok"] = label_share <= cfg.max_label_class_share
    if not gates["label_balance_ok"]:
        reasons.append(f"data_quality_report.json: label class share {label_share:.6g} > {cfg.max_label_class_share:.6g}")

    missing_bars = _max_numeric_leaf(quality.get("missing_bars_by_pair", {}))
    gates["missing_bars_ok"] = missing_bars <= cfg.max_missing_bars_per_pair
    if not gates["missing_bars_ok"]:
        reasons.append(f"data_quality_report.json: missing bars {missing_bars:.6g} > {cfg.max_missing_bars_per_pair}")

    pairs = _pair_rows(pair_report)
    failing_pairs = []
    warning_pairs = []
    for pair, row in pairs.items():
        status = str(row.get("status", "")).lower()
        if status == "fail":
            failing_pairs.append(pair)
        elif status == "warn":
            warning_pairs.append(pair)
    gates["pair_readiness_ok"] = not failing_pairs and (cfg.allow_pair_warnings or not warning_pairs)
    if failing_pairs:
        reasons.append(f"pair_readiness_report.json: failing pairs {sorted(failing_pairs)}")
    if warning_pairs and not cfg.allow_pair_warnings:
        reasons.append(f"pair_readiness_report.json: warning pairs blocked {sorted(warning_pairs)}")

    ready = all(gates.values())
    return {
        "ready_for_training": ready,
        "artifact_dir": str(root),
        "gates": gates,
        "reasons": reasons,
        "schema": {
            "feature_count": feature_count,
            "schema_hash": schema_hash,
            "computed_hash": computed_hash,
        },
        "quality": {
            "max_feature_nan_rate": nan_rate,
            "max_label_class_share": label_share,
            "max_missing_bars": missing_bars,
        },
        "pairs": {
            "count": len(pairs),
            "failing": sorted(failing_pairs),
            "warnings": sorted(warning_pairs),
        },
        "artifacts": artifacts,
    }


def write_data_feature_readiness_report(
    artifact_dir: str | Path,
    output_path: str | Path | None = None,
    config: DataFeatureGateConfig | None = None,
) -> Dict[str, Any]:
    """Validate data/features and write `priority4_data_feature_report.json`."""
    report = validate_data_feature_readiness(artifact_dir, config=config)
    out = Path(output_path) if output_path else Path(artifact_dir) / "priority4_data_feature_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
