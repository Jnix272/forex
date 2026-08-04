"""Generic sidecar registry for cache and model artifacts.

Sidecars are small files that make a large artifact usable and safe: manifests,
feature schemas, readiness reports, calibration reports, metadata JSON, scaler
files, or auxiliary target arrays. This module writes and validates a single
`sidecar_index.json` so related files can be checked as one artifact package.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SidecarSpec:
    name: str
    path: str | Path
    kind: str = "metadata"
    required: bool = True
    max_age_seconds: float | None = None


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _as_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def build_sidecar_index(
    artifact_root: str | Path,
    sidecars: Iterable[SidecarSpec],
    package_name: str = "artifact_package",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable index for sidecars under `artifact_root`."""
    root = Path(artifact_root)
    entries = []
    for spec in sidecars:
        path = Path(spec.path)
        if not path.is_absolute():
            path = root / path
        exists = path.exists()
        entry = {
            "name": spec.name,
            "kind": spec.kind,
            "path": _as_relative(path, root),
            "required": bool(spec.required),
            "exists": bool(exists),
            "size_bytes": int(path.stat().st_size) if exists else 0,
            "sha256": file_sha256(path) if exists and path.is_file() else "",
            "max_age_seconds": spec.max_age_seconds,
            "indexed_at": _utc_now(),
        }
        entries.append(entry)

    return {
        "package_name": package_name,
        "artifact_root": str(root),
        "created_at": _utc_now(),
        "metadata": dict(metadata or {}),
        "sidecars": entries,
    }


def write_sidecar_index(
    artifact_root: str | Path,
    sidecars: Iterable[SidecarSpec],
    output_path: str | Path | None = None,
    package_name: str = "artifact_package",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write `sidecar_index.json` for the supplied sidecar specs."""
    root = Path(artifact_root)
    index = build_sidecar_index(root, sidecars, package_name=package_name, metadata=metadata)
    out = Path(output_path) if output_path else root / "sidecar_index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return index


def _entry_path(root: Path, entry: Mapping[str, Any]) -> Path:
    path = Path(str(entry.get("path", "")))
    return path if path.is_absolute() else root / path


def validate_sidecar_index(
    index_path: str | Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate an existing sidecar index and return a pass/fail report."""
    path = Path(index_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    root = Path(data.get("artifact_root") or path.parent)
    now_dt = now or datetime.now(UTC)

    gates: dict[str, bool] = {}
    reasons = []
    checked = []
    for entry in data.get("sidecars", []):
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name", "unknown"))
        sidecar_path = _entry_path(root, entry)
        required = bool(entry.get("required", True))
        exists = sidecar_path.exists()
        size = sidecar_path.stat().st_size if exists else 0

        present_ok = exists or not required
        gates[f"{name}_present"] = present_ok
        if not present_ok:
            reasons.append(f"{name}: missing required sidecar {sidecar_path}")

        nonempty_ok = (not exists) or size > 0 or not required
        gates[f"{name}_nonempty"] = nonempty_ok
        if not nonempty_ok:
            reasons.append(f"{name}: sidecar is empty")

        expected_hash = str(entry.get("sha256", ""))
        actual_hash = file_sha256(sidecar_path) if exists and sidecar_path.is_file() else ""
        hash_ok = (not exists) or not expected_hash or actual_hash == expected_hash
        gates[f"{name}_hash_ok"] = hash_ok
        if not hash_ok:
            reasons.append(f"{name}: sha256 mismatch")

        max_age = entry.get("max_age_seconds")
        fresh_ok = True
        if exists and max_age is not None:
            age = now_dt.timestamp() - sidecar_path.stat().st_mtime
            fresh_ok = age <= float(max_age)
            if not fresh_ok:
                reasons.append(f"{name}: stale sidecar age {age:.1f}s > {float(max_age):.1f}s")
        gates[f"{name}_fresh"] = fresh_ok

        checked.append(
            {
                "name": name,
                "path": str(sidecar_path),
                "required": required,
                "exists": exists,
                "size_bytes": int(size),
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
            }
        )

    valid = all(gates.values())
    return {
        "valid": valid,
        "index_path": str(path),
        "package_name": data.get("package_name", "artifact_package"),
        "artifact_root": str(root),
        "gates": gates,
        "reasons": reasons,
        "checked": checked,
    }
