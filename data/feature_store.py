"""Deprecation shim: data.feature_store has moved to feature_store.polars_store.

This module re-exports everything from the canonical location so existing
imports keep working, emitting a one-time DeprecationWarning.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "data.feature_store is deprecated; import from feature_store.polars_store instead.",
    DeprecationWarning,
    stacklevel=2,
)

from feature_store.polars_store import (  # noqa: E402,F401
    BUILTIN_FEATURES,
    FeatureStore,
    compute_data_hash,
    compute_feature_hash,
    get_feature_store,
    materialize_features,
)

__all__ = [
    "BUILTIN_FEATURES",
    "FeatureStore",
    "compute_data_hash",
    "compute_feature_hash",
    "get_feature_store",
    "materialize_features",
]
