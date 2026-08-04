"""Compatibility shim for the canonical Polars feature engineering pipeline.

The implementation lives in :mod:`features.feature_engineering_pl`. Keep this
module importable for older scripts/tests without maintaining a second,
diverging feature schema.
"""

from features.feature_engineering_pl import *  # noqa: F403
