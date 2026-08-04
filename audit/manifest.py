"""
audit/manifest.py — generate / verify / regenerate reproducibility manifests.

A manifest is a JSON snapshot of every artifact + input + environment for a
training run. It is stored as ``manifest.json`` alongside the run's checkpoints
and is verified by the promotion auditor (``validation/promotion_audit.py``).

Standard-library only.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "manifest.json"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def compute_file_hash(path: str, algo: str = "sha256", chunk: int = 1 << 20) -> str:
    """Hash a single file (chunked, memory-safe)."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def compute_dir_hash(path: str, algo: str = "sha256", relative: bool = True) -> str:
    """Hash a directory by hashing (relative path → file hash) pairs in order."""
    root = Path(path)
    if not root.is_dir():
        return ""
    h = hashlib.new(algo)
    entries = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(root)) if relative else str(p)
            entries.append((rel, compute_file_hash(str(p), algo)))
    for rel, fh in entries:
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(fh.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def hash_inputs(inputs: dict[str, Any], algo: str = "sha256") -> str:
    """Deterministic hash of an arbitrary JSON-able input dict."""
    return hashlib.new(algo, _canonical_json(inputs).encode("utf-8")).hexdigest()


def git_info(repo_root: str | None = None) -> dict[str, str]:
    """Best-effort git commit + branch capture."""
    info: dict[str, str] = {}
    cwd = repo_root or os.getcwd()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
        info["commit"] = commit
    except Exception:
        info["commit"] = ""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd,
            stderr=subprocess.DEVNULL
        ).decode().strip()
        info["branch"] = branch
    except Exception:
        info["branch"] = ""
    return info


def capture_env(include_gpu: bool = True) -> dict[str, Any]:
    """Capture python / platform / library versions for reproducibility."""
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import importlib.metadata as md
        libs: dict[str, str] = {}
        for name in ("numpy", "pandas", "polars", "scipy", "torch",
                     "sklearn", "shap", "river", "xgboost", "lightgbm", "catboost"):
            try:
                libs[name] = md.version(name)
            except Exception:
                pass
        env["libs"] = libs
    except Exception:
        env["libs"] = {}
    if include_gpu:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL
            ).decode().strip().splitlines()
            env["gpu"] = out[0] if out else ""
        except Exception:
            env["gpu"] = ""
    return env


def generate_manifest(
    run_dir: str,
    run_id: str,
    model: str,
    params: dict[str, Any] | None = None,
    dataset: dict[str, Any] | None = None,
    seed: int | None = None,
    artifacts: dict[str, str] | None = None,
    inputs: dict[str, Any] | None = None,
    commit: str | None = None,
    env: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a reproducibility manifest dict.

    Parameters
    ----------
    run_dir    : Checkpoint directory; artifact hashes are computed relative to it.
    run_id     : Unique training run id.
    model      : Model name (picked up by promotion_audit's manifest consistency gate).
    params     : Training hyperparameters.
    dataset    : e.g. {"name": "EURUSD_1m", "version": "2026.07", "hash": "..."}.
    artifacts  : Map of logical name → path (absolute or relative to run_dir).
    inputs     : Any other input key/values folded into the manifest.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_hashes: dict[str, dict[str, str]] = {}
    for name, path in (artifacts or {}).items():
        p = Path(path)
        if not p.is_absolute():
            p = run_dir / p
        if p.exists() and p.is_file():
            artifact_hashes[name] = {"path": str(p), "sha256": compute_file_hash(str(p))}
        else:
            artifact_hashes[name] = {"path": str(p), "sha256": ""}

    dataset = dataset or {}
    data_hash = dataset.get("hash", "")
    if not data_hash:
        data_hash = hash_inputs(dataset)

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "run_id": run_id,
        "model": model,
        "created_at": _now_iso(),
        "git": commit if isinstance(commit, dict) else git_info(),
        "seed": seed,
        "params": params or {},
        "dataset": dataset,
        "data_hash": data_hash,
        "inputs": inputs or {},
        "artifacts": artifact_hashes,
        "env": env if env is not None else capture_env(),
    }
    manifest["hash"] = hash_inputs(manifest)
    return manifest


def write_manifest(manifest: dict[str, Any], run_dir: str, filename: str = MANIFEST_FILENAME) -> str:
    """Write a manifest JSON next to the run's checkpoints."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / filename
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return str(out)


def verify_manifest(manifest: dict[str, Any], run_dir: str | None = None) -> dict[str, Any]:
    """Verify a manifest's self-hash and artifact hashes.

    Returns a report with gates: ``hash_ok``, ``artifacts_ok`` and a boolean
    ``reproducible`` (all gates pass).
    """
    manifest = dict(manifest)
    stored_hash = manifest.pop("hash", "")
    gates: dict[str, bool] = {}
    details: dict[str, Any] = {}

    recomputed = hash_inputs(manifest)
    gates["hash_ok"] = stored_hash == recomputed
    details["stored_hash"] = stored_hash
    details["recomputed_hash"] = recomputed

    artifact_gates: dict[str, bool] = {}
    if run_dir is not None:
        run_dir = Path(run_dir)
        for name, entry in manifest.get("artifacts", {}).items():
            path = Path(entry.get("path", ""))
            if not path.is_absolute():
                path = run_dir / path
            if not path.is_file():
                artifact_gates[name] = False
                continue
            current = compute_file_hash(str(path))
            artifact_gates[name] = current == entry.get("sha256", "")
        gates["artifacts_ok"] = all(artifact_gates.values()) if artifact_gates else True
        details["artifacts"] = artifact_gates
    else:
        gates["artifacts_ok"] = True

    gates["reproducible"] = all(gates.values())
    details["manifest_version"] = manifest.get("manifest_version")
    details["model"] = manifest.get("model")
    details["run_id"] = manifest.get("run_id")
    return {
        "reproducible": bool(gates.get("reproducible")),
        "gates": gates,
        "details": details,
        "summary": "REPRODUCIBLE ✅" if gates.get("reproducible") else "NOT REPRODUCIBLE ❌",
    }


def _default_artifacts(run_dir: Path) -> dict[str, str]:
    """Guess artifact paths inside a checkpoint dir for regeneration."""
    names = ["production_best.pt", "production_prev.pt", "best_model.pt",
             "model_best.pt", "final.pt", "config.yaml", "train_summary.json"]
    artifacts = {}
    for n in names:
        p = run_dir / n
        if p.is_file():
            artifacts[n] = str(p)
    return artifacts


def regenerate_manifest(run_dir: str, run_id: str = "", model: str = "unknown",
                        **kwargs: Any) -> dict[str, Any]:
    """CLI/helper: rebuild a run's manifest from an existing checkpoint dir.

    Reuses existing inputs where a manifest is already present (so regenerating
    preserves the original lineage) and re-hashes artifacts on disk.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / MANIFEST_FILENAME
    existing: dict[str, Any] = {}
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            existing = json.load(f)

    artifacts = _default_artifacts(run_dir)
    inputs = dict(existing.get("inputs", {}))
    inputs.update(kwargs.get("inputs", {}) or {})
    dataset = existing.get("dataset") or kwargs.get("dataset")

    manifest = generate_manifest(
        run_dir=str(run_dir),
        run_id=run_id or existing.get("run_id", ""),
        model=model if model and model != "unknown" else existing.get("model", model or "unknown"),
        params=existing.get("params") or kwargs.get("params"),
        dataset=dataset,
        seed=existing.get("seed") or kwargs.get("seed"),
        artifacts=artifacts,
        inputs=inputs,
        commit=existing.get("git") or kwargs.get("commit"),
        env=existing.get("env") or kwargs.get("env"),
    )
    return manifest


def verify_manifests_in_tree(root: str) -> list[dict[str, Any]]:
    """Verify every manifest.json found under ``root`` (recursively)."""
    reports: list[dict[str, Any]] = []
    root = Path(root)
    for m in sorted(root.rglob(MANIFEST_FILENAME)):
        try:
            with open(m, encoding="utf-8") as f:
                manifest = json.load(f)
            report = verify_manifest(manifest, run_dir=str(m.parent))
            report["manifest_path"] = str(m)
            reports.append(report)
        except Exception as e:
            reports.append({
                "manifest_path": str(m),
                "reproducible": False,
                "error": str(e),
            })
    return reports
