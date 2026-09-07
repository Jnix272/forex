"""
features/feature_engineering_pl.py — Backward-compat shim.

Implementation split into features/engineering/ sub-package.
All symbols remain importable from this path for existing callers.
"""

from features.engineering import *  # noqa: F401, F403
from features.engineering import (
    FeatureEngineer,
    sanitize_frame,
    CrossAssetFeatures,
    RegimeGateClassifier,
    _SANITIZE_NO_CLIP,
    _DEFAULT_PIT_DELAY_MINUTES,
    _with_available_time,
    _join_asof_available,
)
