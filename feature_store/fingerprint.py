"""Source-code and side-input fingerprints for feature caches.

Every cache of engineered features (training Zarr datasets, the per-pair
parquet feature cache, FeatureStore materializations) must be invalidated when
the code that computes the features, or a raw side input it joins in (COT,
news, economic calendar), changes. Config digests alone cannot see either.

This module only depends on the standard library so any cache layer can import
it without pulling in the feature pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Modules whose output ends up in the training feature matrix / labels.
FEATURE_CODE_GLOBS: tuple[str, ...] = (
    "features/engineering/*.py",
    "features/feature_engineering_pl.py",
    "features/regime_detection.py",
    "features/cot_features.py",
    "features/macro_features.py",
    "features/sentiment_fusion.py",
    "features/cross_asset_factors.py",
    "features/no_trade_zones.py",
    "features/feature_quality_monitor.py",
    "features/quality.py",
    "features/finbert_sentiment.py",
    "labeling/*.py",
    "data/data_ingestion.py",
    "data/historical_news.py",
    "data/cross_asset.py",
)

# FeatureStore materializers additionally depend on the registry definitions.
FEATURE_STORE_CODE_GLOBS: tuple[str, ...] = (
    *FEATURE_CODE_GLOBS,
    "data/feature_materializers.py",
    "data/feature_definitions.py",
)

FEATURE_SCALE_KEYS: tuple[str, ...] = (
    "atr_windows",
    "vol_windows",
    "ofi_windows",
    "momentum_windows",
    "vwap_window",
    "chop_window",
    "corr_window",
    "regime_window",
    "volatility_window",
)

_lock = threading.Lock()
_code_cache: dict[tuple, str] = {}


def _resolve(path: str | os.PathLike, root: Path) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return root / p


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def list_code_files(globs: Iterable[str] = FEATURE_CODE_GLOBS, root: Path | None = None) -> list[Path]:
    root = Path(root) if root is not None else REPO_ROOT
    seen: dict[str, Path] = {}
    for pattern in globs:
        for p in root.glob(pattern):
            if p.is_file() and "__pycache__" not in p.parts:
                seen[_rel(p, root)] = p
    return [seen[k] for k in sorted(seen)]


def code_fingerprint(globs: Iterable[str] = FEATURE_CODE_GLOBS, root: Path | None = None) -> str:
    """SHA-256 over (relative path, newline-normalised content) of the given sources.

    Newlines are normalised so a CRLF/LF checkout difference does not bust caches.
    """
    root = Path(root) if root is not None else REPO_ROOT
    files = list_code_files(tuple(globs), root)
    stat_key = tuple((_rel(p, root), p.stat().st_size, p.stat().st_mtime_ns) for p in files)
    with _lock:
        cached = _code_cache.get(stat_key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    for p in files:
        h.update(_rel(p, root).encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
        h.update(b"\0")
    digest = h.hexdigest()
    with _lock:
        _code_cache[stat_key] = digest
    return digest


def feature_code_fingerprint() -> str:
    return code_fingerprint(FEATURE_CODE_GLOBS)


def feature_store_code_fingerprint() -> str:
    return code_fingerprint(FEATURE_STORE_CODE_GLOBS)


def file_signature(path: str | os.PathLike, root: Path | None = None) -> dict[str, Any]:
    """Cheap identity of a raw side-input file: size + mtime (no content read)."""
    root = Path(root) if root is not None else REPO_ROOT
    p = _resolve(path, root)
    sig: dict[str, Any] = {"path": _rel(p, root)}
    try:
        st = p.stat()
    except OSError:
        sig["missing"] = True
        return sig
    sig["size"] = int(st.st_size)
    sig["mtime_ns"] = int(st.st_mtime_ns)
    return sig


def inputs_fingerprint(paths: Iterable[str | os.PathLike], root: Path | None = None) -> tuple[str, list[dict]]:
    sigs = sorted((file_signature(p, root) for p in paths if p), key=lambda s: s["path"])
    digest = hashlib.sha256(json.dumps(sigs, sort_keys=True).encode("utf-8")).hexdigest()
    return digest, sigs


def _yaml_features_section() -> dict:
    cfg_path = os.environ.get("FOREX_CONFIG") or os.environ.get("FOREX_RUN_CONFIG")
    if not cfg_path:
        return {}
    try:
        import yaml

        with open(cfg_path, encoding="utf-8") as f:
            root = yaml.safe_load(f) or {}
    except Exception:
        return {}
    sec = root.get("features") or {}
    return sec if isinstance(sec, dict) else {}


def effective_feature_scales(feature_scales: dict | None = None) -> dict[str, Any]:
    """Indicator windows as ``FeatureEngineer.__init__`` resolves them.

    ``features.*`` from YAML (FOREX_CONFIG) override ``settings.FEATURE_SCALES``
    key by key, exactly like ``features/engineering/core.py``.
    """
    if feature_scales is None:
        try:
            from config.settings import FEATURE_SCALES as feature_scales
        except Exception:
            feature_scales = {}
    yaml_fs = _yaml_features_section()
    out: dict[str, Any] = {}
    for k in FEATURE_SCALE_KEYS:
        v = yaml_fs.get(k, (feature_scales or {}).get(k))
        if v is not None:
            out[k] = list(v) if isinstance(v, (list, tuple)) else v
    return out


def stable_digest(payload: Any, n: int | None = None) -> str:
    d = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return d[:n] if n else d
