"""
Feature Store Package
=====================
Feature store for materialized features with versioning.

Exports are resolved lazily so light submodules (``feature_store.fingerprint``)
can be imported by cache layers without loading the feature pipeline.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_EXPORTS = {
    "DeltaFeatureStore": "feature_store.store",
    "FeatureMaterializer": "feature_store.materializer",
    "FeatureMetadata": "feature_store.registry",
    "FeatureRegistry": "feature_store.registry",
    "FeatureStore": "feature_store.store",
    "ParquetFeatureStore": "feature_store.store",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'feature_store' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value
