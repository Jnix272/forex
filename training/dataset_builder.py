"""Chunked / multi-pair dataset construction for GPU training.

Extracted from ``training.train_gpu`` (see ``docs/CONTINUE.md``)."""
from __future__ import annotations

import gc
import hashlib
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import StandardScaler

from config.settings import FEATURES, LABELING
from data.cross_asset import load_cross_asset_panel
from data.data_ingestion import ForexDataPipeline, generate_synthetic_tick_data
from data.dataset_manifest import DatasetManifest
from data.historical_news import collect_headlines_for_range, load_historical_news_bundle
from data.sources import ForexDataManager
from features.feature_engineering_pl import FeatureEngineer
from features.finbert_sentiment import SentimentPipeline
from infrastructure.logging_utils import log_data_load
from infrastructure.numerics import sanitize_array
from labeling.rl_reward_labeling import (
    align_labels_with_features,
    compute_rl_reward_labels_regime,
)
from labeling.triple_barrier_labeling import compute_triple_barrier_labels
from training.cache_integrity import (
    _cache_target_col,
    _clamp_n_samples_to_disk,
    _delete_cache_artifacts,
    _effective_window_days,
    _get_cache_path,
    _iter_date_windows,
    _market_bar_arrays_from_feats,
    _postprocess_cache_integrity_check,
    _real_data_window_days,
    _resolve_cross_asset_source,
    _validate_cache_integrity,
    _verify_dataset,
    _warn_multitask_cache_sidecars,
)
from training.config_validate import _effective_max_seq_len
from training.core import _FIRST_CHUNK_COLS, _TRAIN_LOGGER
from training.gpu_cache_io import (
    ZARR as _ZARR_DEFAULT,
)
from training.gpu_cache_io import (
    ZARR_FEATURE_DTYPE,
    ZARR_LABEL_DTYPE,
    _x_path,
    _y_path,
    _zarr_open_group,
    make_training_zarr_compressor,
)
from training.gpu_cache_io import (
    _atr_path as _atr_path_default,
)
from training.gpu_cache_io import (
    _base_path as _base_path_default,
)
from training.gpu_cache_io import (
    _close_path as _close_path_default,
)
from training.gpu_cache_io import (
    _diff_path as _diff_path_default,
)
from training.gpu_cache_io import (
    _pq_path as _pq_path_default,
)
from training.gpu_cache_io import (
    _scaler_npz_path as _scaler_npz_path_default,
)
from training.gpu_cache_io import (
    _spread_path as _spread_path_default,
)
from training.gpu_cache_io import (
    _x_path as _x_path_default,
)
from training.gpu_cache_io import (
    _y_cls_path as _y_cls_path_default,
)
from training.gpu_cache_io import (
    _y_path as _y_path_default,
)
from training.gpu_cache_io import (
    _zarr_create as _zarr_create_default,
)
from training.gpu_cache_io import (
    _zarr_open_group as _zarr_open_group_default,
)

# Local module state (was train_gpu globals)
_FIRST_CHUNK_COLS: list | None = None  # noqa: F811
# Defaults until bind_host overlays train_gpu symbols
ZARR = _ZARR_DEFAULT
_x_path = _x_path_default  # noqa: F811
_y_path = _y_path_default  # noqa: F811
_pq_path = _pq_path_default
_y_cls_path = _y_cls_path_default
_diff_path = _diff_path_default
_close_path = _close_path_default
_atr_path = _atr_path_default
_spread_path = _spread_path_default
_base_path = _base_path_default
_scaler_npz_path = _scaler_npz_path_default
_zarr_open_group = _zarr_open_group_default  # noqa: F811
_zarr_create = _zarr_create_default
_PAIR_READINESS_STATS: dict = {}
_PAIR_ALIGNMENT_STATS: dict = {}
_TRAIN_LOGGER = None  # mirrored from train_gpu via bind_host / logging shims  # noqa: F811


# DS-002: historical context loaded before each real-data window so EMA/MACD
# /rolling stats are not cold-started at the window boundary.
_FEATURE_WARMUP_DAYS = 14

TimeKey = tuple[str, int | str]


def _warmup_load_start(win_start: str, warmup_days: int = _FEATURE_WARMUP_DAYS) -> str:
    """Return YYYY-MM-DD load start = win_start minus warmup days (UTC)."""
    return (pd.to_datetime(win_start, utc=True) - pd.Timedelta(days=int(warmup_days))).strftime("%Y-%m-%d")


def _safe_save_json(data, path) -> None:
    """Atomic JSON writer - write to tempfile, then ``os.replace``.

    A local copy of ``training.post_train._safe_save_json`` so this
    module is statically self-contained (the IDE sees the symbol and
    the function does not depend on the post_train import chain).
    Behavior is identical to the canonical version in post_train.py.
    """
    import json
    import os
    import tempfile
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _identity_scaler(n_features: int) -> StandardScaler:
    s = StandardScaler()
    s.mean_ = np.zeros(n_features)
    s.scale_ = np.ones(n_features)
    s.var_ = np.ones(n_features)
    s.n_features_in_ = n_features
    return s


def _set_scaler_feature_names(scaler: StandardScaler, columns) -> None:
    """Attach ordered feature names when fitting scalers on numpy arrays."""

    try:
        if getattr(scaler, "feature_names_in_", None) is not None:
            return

        names = [str(c) for c in columns]

        if names:
            scaler.feature_names_in_ = np.asarray(names, dtype=str)

    except Exception:
        pass


def _scaler_feature_names(scaler: StandardScaler | None) -> list[str]:

    try:
        names = getattr(scaler, "feature_names_in_", None)

        if names is not None:
            return [str(c) for c in list(names)]

    except Exception:
        pass

    return []


def _write_feature_schema_json(cache_path: Path, feature_names: list[str], args=None) -> None:

    if not feature_names:
        return

    try:
        import json

        with open(str(cache_path) + "_feature_schema.json", "w", encoding="utf-8") as f:
            json.dump([str(c) for c in feature_names], f)

    except Exception:
        pass

    if args is not None:
        _enforce_dataset_feature_schema(args, feature_names, cache_path, phase="final")


def _enforce_dataset_feature_schema(
    args,
    feature_names: list[str],
    cache_path: Path | str | None = None,
    *,
    phase: str = "final",
) -> dict:
    """
    Multi-part gate for dataset builds:

      1. curriculum_coverage / market_columns / feature_mask  (built schema)
      2. settings_yaml   - shared-key drift by section
      3. args_yaml       - resolved args vs YAML (silent load failures)

    Raises RuntimeError when errors are present and integrity_gate is on
    (override with args.feature_schema_gate).
    """
    from config.config_mismatch_audit import (
        audit_args_vs_yaml_mismatches,
        audit_settings_yaml_section_mismatches,
        load_yaml_config,
    )
    from config.curriculum_audit import (
        audit_built_dataset_schema,
        format_audit_warnings,
    )
    from config.feature_mask import FEATURE_MASK
    from config.settings import CURRICULUM as SETTINGS_CURRICULUM

    cur = getattr(args, "curriculum", None)
    if not isinstance(cur, dict):
        cur = SETTINGS_CURRICULUM
    schema_report = audit_built_dataset_schema(
        feature_names=list(feature_names or []),
        feature_groups=(cur.get("feature_groups") if isinstance(cur, dict) else None),
        feature_mask=FEATURE_MASK,
    )

    # Only compare args↔YAML when this run actually loaded a config file.
    # getattr(..., "config/run.yaml") would falsely flag argparse/strategy defaults
    # against disk YAML when --config was never passed.
    yaml_path = getattr(args, "config", None)
    if yaml_path:
        yaml_path = str(yaml_path)
        yaml_cfg = load_yaml_config(yaml_path)
        settings_report = audit_settings_yaml_section_mismatches(yaml_cfg, yaml_path=yaml_path)
        args_report = audit_args_vs_yaml_mismatches(args, yaml_cfg, yaml_path=yaml_path)
    else:
        yaml_path = "(no --config)"
        yaml_cfg = {}
        settings_report = {
            "errors": [],
            "warnings": [],
            "mismatches": [],
            "parts": {},
        }
        args_report = {"errors": [], "warnings": [], "mismatches": []}

    parts = {
        "built_schema": {
            "errors": list(schema_report.get("errors") or []),
            "warnings": list(schema_report.get("warnings") or []),
            "missing_from_schema": schema_report.get("missing_from_schema") or {},
            "mask_enabled_missing": schema_report.get("mask_enabled_missing") or [],
        },
        "settings_yaml": {
            "errors": list(settings_report.get("errors") or []),
            "warnings": list(settings_report.get("warnings") or []),
            "mismatches": settings_report.get("mismatches") or [],
            "sections": {
                k: {
                    "n_mismatches": len((v or {}).get("mismatches") or []),
                    "n_only_settings": len((v or {}).get("only_settings") or []),
                    "n_only_yaml": len((v or {}).get("only_yaml") or []),
                }
                for k, v in (settings_report.get("parts") or {}).items()
            },
        },
        "args_yaml": {
            "errors": list(args_report.get("errors") or []),
            "warnings": list(args_report.get("warnings") or []),
            "mismatches": args_report.get("mismatches") or [],
        },
    }

    errors: list[str] = []
    warnings: list[str] = []
    for part_name, part in parts.items():
        for err in part.get("errors") or []:
            errors.append(f"[{part_name}] {err}")
        for warn in part.get("warnings") or []:
            warnings.append(f"[{part_name}] {warn}")

    report = {
        "phase": phase,
        "n_features": int(schema_report.get("n_features") or 0),
        "parts": parts,
        "errors": errors,
        "warnings": warnings,
        "missing_from_schema": schema_report.get("missing_from_schema") or {},
        "mask_enabled_missing": schema_report.get("mask_enabled_missing") or [],
        "extras_not_in_mask": schema_report.get("extras_not_in_mask") or [],
    }

    for line in format_audit_warnings({"warnings": warnings}, prefix="[FeatureSchema]"):
        try:
            print(line)
        except Exception:
            print(line)
    for err in errors:
        try:
            print(f"[FeatureSchema] ERROR: {err}")
        except Exception:
            print(f"[FeatureSchema] ERROR: {err}")

    if cache_path is not None:
        try:
            import json

            payload = {
                "phase": phase,
                "n_features": report["n_features"],
                "parts": {
                    k: {
                        "errors": v.get("errors") or [],
                        "warnings": (v.get("warnings") or [])[:30],
                        **({"sections": v["sections"]} if "sections" in v else {}),
                        **(
                            {
                                "missing_from_schema": v.get("missing_from_schema"),
                                "mask_enabled_missing": (v.get("mask_enabled_missing") or [])[:20],
                            }
                            if k == "built_schema"
                            else {}
                        ),
                        **(
                            {
                                "mismatches": (v.get("mismatches") or [])[:40],
                            }
                            if k in ("settings_yaml", "args_yaml")
                            else {}
                        ),
                    }
                    for k, v in parts.items()
                },
                "errors": errors,
                "warnings": warnings[:60],
            }
            with open(str(cache_path) + "_feature_schema_audit.json", "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
        except Exception:
            pass

    gate = bool(getattr(args, "integrity_gate", True))
    if getattr(args, "feature_schema_gate", None) is not None:
        gate = bool(args.feature_schema_gate)
    if errors and gate:
        hint = ""
        if cache_path is not None:
            hint = f" See {cache_path}_feature_schema_audit.json."
        raise RuntimeError(f"Dataset feature-schema gate failed ({phase}): " + "; ".join(errors[:5]) + hint)
    elif errors:
        print(
            f"[FeatureSchema] WARN: {len(errors)} error(s) ignored "
            f"(feature_schema_gate/integrity_gate off): {errors[0]}"
        )
    else:
        print(
            f"[FeatureSchema] OK ({phase}): {report['n_features']} columns; "
            f"{len(warnings)} warning(s) across "
            f"{', '.join(parts.keys())}"
        )
    return report


def _maybe_enforce_feature_schema_early(args, cache_path: Path | str | None = None) -> None:
    """Fail-fast on first locked chunk schema during a long dataset build."""
    global _FIRST_CHUNK_COLS
    if not _FIRST_CHUNK_COLS:
        return
    if getattr(args, "_feature_schema_checked", False):
        return
    args._feature_schema_checked = True
    _enforce_dataset_feature_schema(args, list(_FIRST_CHUNK_COLS), cache_path, phase="first_chunk")


def _maybe_run_lookahead_guard(
    args,
    X_seq: np.ndarray,
    close_seq: np.ndarray | None = None,
) -> None:
    """TPA-D04: structural lookahead check on built chunks.

    Modes (via ``args.lookahead_guard_mode``):
      off     -> skip entirely
      fast    -> one-shot, last-timestep features only, no rolling / permutation
      full    -> one-shot, last-timestep features only, but with rolling +
                 permutation checks enabled (default; stronger lookahead detection)
      multi   -> run on the first N chunks (default 4), accumulating evidence
    """
    if getattr(args, "_lookahead_guard_checked", False):
        return
    if getattr(args, "lookahead_guard", None) is False:
        return
    if not bool(getattr(args, "integrity_gate", True)):
        return
    if X_seq is None or getattr(X_seq, "size", 0) == 0:
        return
    _mode = str(getattr(args, "lookahead_guard_mode", "full") or "full").lower()
    if _mode == "off":
        return
    _use_rolling = _mode in ("full", "multi")
    _use_perm = _mode in ("full", "multi")
    _run_multi = _mode == "multi"
    if _run_multi:
        # Only do per-chunk accumulation in multi mode. In single-shot
        # modes we mark the check as done after the first chunk.
        _done_attr = "_lookahead_guard_checked_multi"
    else:
        args._lookahead_guard_checked = True
        _done_attr = "_lookahead_guard_checked"
    if getattr(args, _done_attr, False):
        return
    args._lookahead_guard_checked = True
    args._lookahead_guard_checked_multi = True
    _max_chunks = max(1, int(getattr(args, "lookahead_guard_chunks", 4) or 4))
    _n_chunks_done = int(getattr(args, "_lookahead_guard_n_chunks", 0) or 0)
    if _run_multi and _n_chunks_done >= _max_chunks:
        return
    if _run_multi:
        args._lookahead_guard_n_chunks = _n_chunks_done + 1
    try:
        from features.lookahead_guard import LookaheadViolation, assert_no_lookahead

        arr = np.asarray(X_seq)
        if arr.ndim != 3 or arr.shape[0] < 64:
            return
        # Sample up to 4096 rows from the chunk to keep the cost bounded
        # even on large seq_len / n_features settings.
        _n_rows = int(arr.shape[0])
        if _n_rows > 4096:
            _step = max(1, _n_rows // 4096)
            _sel = np.arange(0, _n_rows, _step)[:4096]
            feats = arr[_sel, -1, :].astype(np.float64, copy=False)
            _ts = _sel.astype(np.int64)
            if close_seq is not None and len(close_seq) == _n_rows:
                _close_sel = np.asarray(close_seq, dtype=np.float64).ravel()[_sel]
            else:
                _close_sel = None
        else:
            feats = arr[:, -1, :].astype(np.float64, copy=False)
            _ts = np.arange(_n_rows, dtype=np.int64)
            _close_sel = (
                np.asarray(close_seq, dtype=np.float64).ravel()
                if close_seq is not None and len(close_seq) == _n_rows
                else None
            )
        names = list(_FIRST_CHUNK_COLS or []) or [f"f{i}" for i in range(feats.shape[1])]
        names = names[: feats.shape[1]]
        fwd = None
        if _close_sel is not None:
            closes = _close_sel
            fwd = np.full(len(closes), np.nan, dtype=np.float64)
            denom = np.maximum(closes[:-1], 1e-12)
            fwd[:-1] = (closes[1:] - closes[:-1]) / denom
        report = assert_no_lookahead(
            timestamps=_ts,
            features=feats,
            feature_names=names,
            forward_returns=fwd,
            rolling_check=_use_rolling,
            permutation_check=_use_perm,
        )
        _chunk_label = f"chunk {_n_chunks_done + 1}/{_max_chunks}" if _run_multi else "first chunk"
        print(
            f"[LookaheadGuard] mode={_mode} ({_chunk_label}) "
            f"rows={feats.shape[0]} feats={feats.shape[1]} | {report.summary()}"
        )
    except LookaheadViolation as exc:
        raise RuntimeError(
            f"Lookahead guard failed on first chunk: {exc}. Pass --no-integrity-gate to skip (not recommended)."
        ) from exc
    except Exception as exc:
        print(f"[LookaheadGuard] WARN: skipped ({type(exc).__name__}: {exc})")


def _build_multipair_feature_schema(
    scalers: dict,
    pairs: list[str],
    n_features_total: int,
) -> list[str]:
    """Return the ordered full schema for pair-concatenated caches."""

    names: list[str] = []

    for pair in pairs:
        pair_names = _scaler_feature_names(scalers.get(pair))

        if not pair_names:
            return []

        names.extend([f"{pair}::{name}" for name in pair_names])

    if len(names) != int(n_features_total):
        return []

    return names


def _save_scaler_npz(cache_path: Path, scaler: StandardScaler) -> None:
    if not hasattr(scaler, "mean_") or scaler.mean_ is None:
        return
    p = _scaler_npz_path(cache_path)
    payload = {
        "mean": scaler.mean_,
        "scale": scaler.scale_,
        "var": scaler.var_,
        "n_features_in_": int(scaler.n_features_in_),
        "n_samples_seen_": int(getattr(scaler, "n_samples_seen_", 0) or 0),
    }
    if hasattr(scaler, "feature_names_in_") and scaler.feature_names_in_ is not None:
        payload["feature_names"] = np.asarray([str(c) for c in scaler.feature_names_in_], dtype=str)

    np.savez(p, **payload)


def _load_scaler_npz(cache_path: Path) -> StandardScaler | None:
    p = _scaler_npz_path(cache_path)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    s = StandardScaler()
    s.mean_ = np.asarray(z["mean"], dtype=np.float64)
    s.scale_ = np.asarray(z["scale"], dtype=np.float64)
    s.var_ = np.asarray(z["var"], dtype=np.float64)
    s.n_features_in_ = int(z["n_features_in_"])
    if "n_samples_seen_" in z.files:
        s.n_samples_seen_ = int(z["n_samples_seen_"])
    if "feature_names" in z.files:
        s.feature_names_in_ = np.asarray(z["feature_names"], dtype=object)
    return s


def _leak_check_features_sample(
    cache_path: Path | str,
    max_sample: int = 5000,
) -> tuple[np.ndarray, list[str]] | None:
    """Read a small random sample of the *last-timestep* feature values from the
    Zarr cache for future-leak correlation scanning.

    Returns ``(X_2d, feature_names)`` where ``X_2d`` has shape
    ``(sample_n, n_features)`` - i.e. the feature vector from the final bar of
    each sampled sequence window (the bar the model actually predicts from).
    Returns ``None`` when the cache is not a Zarr store or has no feature data.
    """
    _cp = Path(cache_path)
    # zarr v2 writes .zgroup; v3 writes .zmetadata / .zarr.json. Check either.
    is_zarr_store = (
        str(_cp).endswith(".zarr")
        and _cp.is_dir()
        and (
            (_cp / ".zgroup").exists()
            or (_cp / ".zmetadata").exists()
            or (_cp / ".zarr.json").exists()
            or any(_cp.glob("zarr.json"))
        )
    )
    if not is_zarr_store:
        return None
    try:
        import zarr as _zarr  # type: ignore

        z: Any = _zarr.open(str(_cp), mode="r")
        if "X" not in z:
            return None
        X = z["X"]
        n_total = int(X.shape[0])
        if n_total == 0:
            return None
        feat_names: list[str] = []
        attrs = getattr(X, "attrs", None)
        if attrs is not None:
            feat_names = [str(c) for c in (attrs.get("columns", []) or [])]
        if len(feat_names) != int(X.shape[-1]):
            feat_names = [f"f{i}" for i in range(int(X.shape[-1]))]
        # Sample up to max_sample rows; take the last timestep (column index -1)
        # so each row is the feature vector the model sees for that sample.
        sample_n = min(max_sample, n_total)
        if sample_n == n_total:
            X_sample = np.asarray(X[:, -1, :], dtype=np.float32)
        else:
            step = max(1, n_total // sample_n)
            sel = np.arange(0, n_total, step)[:sample_n]
            X_sample = np.asarray(X[sel, -1, :], dtype=np.float32)
        return X_sample, feat_names
    except Exception:
        return None


def _label_contamination_check(
    cache_path: Path | str,
    args,
    seq_len: int | None = None,
) -> dict:
    """Reconstruct per-sample feature/label timestamps from the Zarr cache
    row ordering and verify that feature timestamps strictly precede their
    corresponding label timestamps.

    The cache is chronologically ordered: row *i* is the *i*-th valid bar
    after warmup.  Each sequence window starts at bar *i* and ends at bar
    *i + seq_len - 1*; the label (forward return) is computed from prices at
    ``i + seq_len - 1 + execution_delay`` (the first bar *after* the window
    that the trader could act on).  We therefore check that
    ``feature_ts[i] < label_ts[i]`` where ``label_ts = feature_ts + seq_len - 1 + delay``.
    """
    _cp = Path(cache_path)
    is_zarr_store = (
        str(_cp).endswith(".zarr")
        and _cp.is_dir()
        and (
            (_cp / ".zgroup").exists()
            or (_cp / ".zmetadata").exists()
            or (_cp / ".zarr.json").exists()
            or any(_cp.glob("zarr.json"))
        )
    )
    if not is_zarr_store:
        return {"ok": True, "violations": 0, "total_checked": 0, "note": "non-zarr cache"}
    try:
        import zarr as _zarr  # type: ignore

        z: Any = _zarr.open(str(_cp), mode="r")
        n_samples = int(z["X"].shape[0])
        if n_samples == 0:
            return {"ok": True, "violations": 0, "total_checked": 0, "note": "empty cache"}
    except Exception as _e:
        return {"ok": True, "violations": 0, "total_checked": 0, "note": f"zarr read error: {_e}"}

    _sl = int(seq_len) if seq_len is not None else int(getattr(args, "seq_len", 80))
    _delay = int(getattr(args, "execution_delay_bars", getattr(args, "lookahead_bars", 1)) or 1)
    # Feature timestamp = window start index; label timestamp = window start + offset
    feat_ts = np.arange(n_samples, dtype=np.int64)
    label_ts = feat_ts + (_sl - 1) + _delay
    return DatasetManifest.check_label_contamination(
        feature_timestamps=feat_ts,
        label_timestamps=label_ts,
        max_tolerance_seconds=0.0,
    )


def _ticks_have_usable_datetime_index(ticks) -> bool:
    """True when tick frame carries a usable datetime column for OHLC resampling.

    Accepts either a pandas DataFrame (with a DatetimeIndex) or a Polars
    DataFrame (with a `timestamp_utc` column). The training pipeline now
    hands Polars frames in by default; we return True for that path so the
    readiness summary prints correctly.
    """
    # Polars FastPath - no index concept, but a timestamp_utc column suffices
    if hasattr(ticks, "columns") and "timestamp_utc" in ticks.columns:
        return True
    idx = getattr(ticks, "index", None)
    if idx is None or len(idx) == 0:
        return False
    if isinstance(idx, pd.DatetimeIndex):
        return True
    return bool(pd.api.types.is_datetime64_any_dtype(idx))


def _normalize_tick_index_utc(ticks: pd.DataFrame) -> pd.DataFrame:
    """Ensure a proper UTC DatetimeIndex (pandas resample requires this)."""
    out = ticks.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out.index.name = "timestamp"
    return out


def _multipair_zero_samples_help(
    pair_ticks: dict | None,
) -> str:
    lines = ["Per-pair tick load summary:"]
    if not pair_ticks:
        lines.append("  (no tick dict - loader failed.)")
        return "\n".join(lines)
    for p, df in pair_ticks.items():
        if df is None:
            lines.append(f"  {p}: None")
            continue
        n = len(df)
        # Support both pandas (DatetimeIndex) and Polars (timestamp_utc col)
        idx = getattr(df, "index", None)
        kind = type(idx).__name__ if idx is not None else "no-index"
        dt_ok = _ticks_have_usable_datetime_index(df)
        lines.append(f"  {p}: ticks={n:,} index={kind} datetime_ok={dt_ok}")
        if n > 0:
            try:
                if idx is not None:
                    lines.append(f"       range: {idx.min()} -> {idx.max()}")
                elif "timestamp_utc" in df.columns:
                    ts = df["timestamp_utc"]
                    # Polars: min/max return scalar Series; pick the item
                    t0 = ts.min()
                    t1 = ts.max()
                    if hasattr(t0, "item"):
                        t0, t1 = t0.item(), t1.item()
                    lines.append(f"       range: {t0} -> {t1}")
            except Exception:
                pass
    lines.append(
        "If every pair shows ticks=0: hour files may be empty (blocked download or bad cache). "
        "Delete data/raw/dukascopy/<PAIR>/ and rerun, shorten the date range, or set "
        "data.full_day_data: true. If ticks>0 but datetime_ok=False, tick index is malformed."
    )
    return "\n".join(lines)


def _json_scalar(value):
    if value is None:
        return None
    try:
        if isinstance(value, np.generic):
            value = value.item()
    except Exception:
        pass
    try:
        import pandas as pd

        if isinstance(value, pd.Timestamp):
            return value.isoformat()
    except Exception:
        pass
    if isinstance(value, np.datetime64):
        return str(value.astype("datetime64[ns]"))
    if isinstance(value, (float, int, str, bool)):
        return value
    return str(value)


def _pair_readiness_entry(pair: str) -> dict:
    global _PAIR_READINESS_STATS
    if "_PAIR_READINESS_STATS" not in globals():
        _PAIR_READINESS_STATS = {}
    if pair not in _PAIR_READINESS_STATS:
        _PAIR_READINESS_STATS[pair] = {
            "pair": pair,
            "raw_ticks": 0,
            "windows_seen": 0,
            "timestamp_start": None,
            "timestamp_end": None,
            "duplicate_timestamps": 0,
            "_observed_hours": [],
            "empty_windows": 0,
            "timestamp_utc_normalized": False,
            "missing_columns": [],
            "schema_errors": [],
            "bars_after_resampling": 0,
            "dropped_bars": 0,
            "dropped_bars_by_reason": {
                "weekend": 0,
                "holiday": 0,
                "dead_bar": 0,
                "spread": 0,
                "atr": 0,
                "news": 0,
                "label_filter": 0,
            },
            "dropped_sequences": 0,
            "dropped_sequences_by_reason": {
                "row_quality": 0,
                "weekend": 0,
                "holiday": 0,
                "dead_bar": 0,
                "spread": 0,
                "atr": 0,
                "news": 0,
                "label_filter": 0,
                "zero_feature_window": 0,
                "invalid_reward_label": 0,
                "invalid_direction_label": 0,
                "path_quality_filter": 0,
            },
            "label_filter_counts": {
                "invalid_label": 0,
                "invalid_reward": 0,
                "invalid_direction_label": 0,
                "invalid_path_quality": 0,
            },
            "nan_count": 0,
            "posinf_count": 0,
            "neginf_count": 0,
            "seq_count": 0,
            "label_counts": {},
            "difficulty_counts": {"0": 0, "1": 0, "2": 0},
            "spread": {},
            "atr": {},
            "valid": True,
            "reasons": [],
        }
    return _PAIR_READINESS_STATS[pair]


def _bump_reason_counts(entry: dict, field: str, counts: dict | None) -> None:

    if not counts:
        return

    target = entry.setdefault(field, {})

    for key, value in counts.items():
        try:
            inc = int(value)

        except Exception:
            continue

        if inc <= 0:
            continue

        target[str(key)] = int(target.get(str(key), 0) or 0) + inc


def _update_pair_readiness_raw(pair: str, ticks) -> None:
    entry = _pair_readiness_entry(pair)
    entry["windows_seen"] += 1
    if ticks is None:
        entry["schema_errors"].append("ticks_none")
        entry["valid"] = False
        return
    n_ticks = len(ticks)

    if n_ticks == 0:
        entry["empty_windows"] = int(entry.get("empty_windows", 0) or 0) + 1

        return

    entry["raw_ticks"] += n_ticks

    cols = set(getattr(ticks, "columns", []))
    if "timestamp_utc" in cols:
        ts_raw = ticks["timestamp_utc"]
    else:
        ts_raw = getattr(ticks, "index", None)
        if ts_raw is None:
            entry["missing_columns"].append("timestamp_utc")
            entry["schema_errors"].append("timestamp_missing")
            entry["timestamp_utc_normalized"] = False
            entry["valid"] = False
            return

    if not ({"bid", "ask"}.issubset(cols) or {"bid_close", "ask_close"}.issubset(cols)):
        for col in ("bid", "ask"):
            if col not in cols:
                entry["missing_columns"].append(col)

    try:
        import pandas as pd
        # ts_raw may be a Polars Series (Datetime ns UTC), numpy array, or pandas Series/Index.

        if "polars" in str(type(ts_raw)).lower():
            # Force conversion to standard numpy datetime64 to avoid pyarrow/polars incompatibilities with pd.to_datetime

            ts_raw_pd = ts_raw.to_pandas()

            if hasattr(ts_raw_pd, "dt") and ts_raw_pd.dt.tz is not None:
                ts_raw_pd = ts_raw_pd.dt.tz_convert(None)

            ts_raw_pd = ts_raw_pd.astype("datetime64[ns]")

        elif hasattr(ts_raw, "to_pandas"):
            ts_raw_pd = ts_raw.to_pandas()

        else:
            ts_raw_pd = ts_raw

        ts_utc = pd.to_datetime(ts_raw_pd, utc=True, errors="coerce")

        valid_ts = ts_utc[~pd.isna(ts_utc)]
        if len(valid_ts):
            entry["timestamp_utc_normalized"] = True

            start = valid_ts.min()
            end = valid_ts.max()
            if entry["timestamp_start"] is None or str(start) < str(entry["timestamp_start"]):
                entry["timestamp_start"] = _json_scalar(start)
            if entry["timestamp_end"] is None or str(end) > str(entry["timestamp_end"]):
                entry["timestamp_end"] = _json_scalar(end)
            entry["duplicate_timestamps"] += int(pd.Series(valid_ts).duplicated().sum())
            try:
                hours = pd.Series(valid_ts).dt.floor("h").dropna().astype(str).unique().tolist()

                observed = set(entry.get("_observed_hours", []) or [])

                observed.update(hours)

                entry["_observed_hours"] = sorted(observed)

            except Exception:
                pass

        else:
            entry["schema_errors"].append("timestamp_parse_failed")
            entry["valid"] = False
    except Exception as exc:
        entry["schema_errors"].append(f"timestamp_check_failed:{type(exc).__name__}")
        entry["valid"] = False

    try:
        if {"bid", "ask"}.issubset(cols):
            bad_bidask = (ticks["ask"].astype(float) <= ticks["bid"].astype(float)).sum()
            if int(bad_bidask) > 0:
                entry["schema_errors"].append(f"ask_lte_bid:{int(bad_bidask)}")
                entry["valid"] = False
    except Exception:
        pass


def _update_pair_readiness_processed(
    pair: str,
    *,
    bars_count: int = 0,
    feature_frame=None,
    labels=None,
    dropped_rows: int = 0,
    dropped_sequences: int = 0,
    diff_seq=None,
    spread_seq=None,
    atr_seq=None,
) -> None:
    entry = _pair_readiness_entry(pair)
    entry["bars_after_resampling"] += int(bars_count or 0)
    entry["dropped_bars"] += int(dropped_rows or 0)
    entry["dropped_sequences"] += int(dropped_sequences or 0)
    if feature_frame is not None:
        try:
            mod = type(feature_frame).__module__
            if mod.startswith("polars"):
                num_cols = [
                    c
                    for c in feature_frame.columns
                    if getattr(feature_frame[c].dtype, "is_numeric", lambda: False)()
                    or str(feature_frame[c].dtype)
                    in (
                        "Float32",
                        "Float64",
                        "Int8",
                        "Int16",
                        "Int32",
                        "Int64",
                        "UInt8",
                        "UInt16",
                        "UInt32",
                        "UInt64",
                    )
                ]
                vals = feature_frame.select(num_cols).to_numpy() if num_cols else np.empty((0, 0), dtype=np.float64)
            else:
                numeric = feature_frame.select_dtypes(include=[np.number])
                vals = numeric.to_numpy(dtype=np.float64, copy=False)
            entry["nan_count"] += int(np.isnan(vals).sum())
            entry["posinf_count"] += int(np.isposinf(vals).sum())
            entry["neginf_count"] += int(np.isneginf(vals).sum())
        except Exception as exc:
            print(f"[PairReadiness] WARN: feature nonfinite counts failed for {pair}: {exc}")
    if labels is not None:
        try:
            label_col = "label" if "label" in labels.columns else None
            if label_col:
                counts = labels[label_col].value_counts(dropna=False).to_dict()
                for k, v in counts.items():
                    key = (
                        str(int(k))
                        if isinstance(k, (int, float, np.integer, np.floating)) and np.isfinite(k)
                        else str(k)
                    )
                    entry["label_counts"][key] = int(entry["label_counts"].get(key, 0) + int(v))
        except Exception as exc:
            print(f"[PairReadiness] WARN: label_counts failed for {pair}: {exc}")
    if diff_seq is not None:
        try:
            vals, counts = np.unique(np.asarray(diff_seq, dtype=np.uint8), return_counts=True)
            for v, c in zip(vals, counts, strict=False):
                entry["difficulty_counts"][str(int(v))] = int(entry["difficulty_counts"].get(str(int(v)), 0) + int(c))
        except Exception as exc:
            print(f"[PairReadiness] WARN: difficulty_counts failed for {pair}: {exc}")
    for name, arr in (("spread", spread_seq), ("atr", atr_seq)):
        if arr is None:
            continue
        try:
            a = np.asarray(arr, dtype=np.float64)
            a = a[np.isfinite(a)]
            if a.size:
                existing = entry.get(name, {})
                samples = existing.get("_samples", [])
                samples.extend(a[:: max(1, len(a) // 500)].tolist())
                samples = samples[-2000:]
                entry[name] = {
                    "median": float(np.median(samples)),
                    "p95": float(np.percentile(samples, 95)),
                    "max": float(np.max(samples)),
                    "_samples": samples,
                }
        except Exception as exc:
            print(f"[PairReadiness] WARN: {name} stats failed for {pair}: {exc}")


def _finalize_pair_readiness_report(args, cache_path, pairs, *, alignment: dict | None = None) -> dict:
    stats = globals().get("_PAIR_READINESS_STATS", {})
    pair_reports = []
    failed = False
    warnings = []
    for pair in pairs:
        entry = dict(stats.get(pair, _pair_readiness_entry(pair)))
        seq = int(entry.get("seq_count", 0) or 0)
        n_feat = int(entry.get("n_features", 0) or 0)
        seq_len = int(getattr(args, "seq_len", 60) or 60)
        total_values = max(1, seq * max(1, seq_len) * max(1, n_feat))
        nonfinite = int(entry.get("nan_count", 0) + entry.get("posinf_count", 0) + entry.get("neginf_count", 0))
        nonfinite_pct = 100.0 * nonfinite / total_values
        schema_errors = list(dict.fromkeys(entry.get("schema_errors", [])))

        reasons = list(dict.fromkeys(entry.get("reasons", [])))

        raw_ticks = int(entry.get("raw_ticks", 0) or 0)

        for err in schema_errors:
            if err == "timestamp_parse_failed":
                if raw_ticks == 0 or seq == 0:
                    reasons.append(err)

            else:
                reasons.append(err)

        if entry.get("missing_columns"):
            reasons.append("missing_required_columns")
        if seq == 0:
            reasons.append("zero_valid_sequences")
        if raw_ticks > 0 and not entry.get("timestamp_utc_normalized", False):
            reasons.append("timestamps_not_confirmed_utc")
        if nonfinite_pct > 1.0:
            reasons.append("feature_nonfinite_pct_gt_1")
        label_counts = entry.get("label_counts", {})
        non_hold = sum(int(v) for k, v in label_counts.items() if str(k) not in {"0", "0.0"})
        if label_counts and non_hold == 0:
            reasons.append("label_classes_collapsed_to_hold")
        diff_counts = entry.get("difficulty_counts", {})
        if diff_counts and int(diff_counts.get("1", 0)) == 0 and int(diff_counts.get("2", 0)) == 0 and seq > 0:
            warnings.append(f"{pair}: difficulty_all_easy")
        observed_hours = sorted(set(entry.get("_observed_hours", []) or []))

        missing_hours = []

        coverage = {}

        try:
            import pandas as pd

            if entry.get("timestamp_start") and entry.get("timestamp_end") and observed_hours:
                start_ts = pd.Timestamp(entry["timestamp_start"]).floor("h")

                end_ts = pd.Timestamp(entry["timestamp_end"]).floor("h")

                expected_hours = pd.date_range(start=start_ts, end=end_ts, freq="h", tz="UTC")

                expected_set = {str(ts) for ts in expected_hours}

                observed_set = {
                    str(
                        pd.Timestamp(h).tz_convert("UTC")
                        if pd.Timestamp(h).tzinfo
                        else pd.Timestamp(h).tz_localize("UTC")
                    )
                    for h in observed_hours
                }

                missing_hours = sorted(expected_set - observed_set)

                coverage = {
                    "expected_hours": len(expected_set),
                    "observed_hours": len(observed_set),
                    "missing_hours": len(missing_hours),
                    "coverage_pct": round(100.0 * len(observed_set) / max(1, len(expected_set)), 6),
                    "missing_hours_sample": missing_hours[:50],
                }

                if missing_hours:
                    warnings.append(f"{pair}: missing_hours={len(missing_hours)}")

        except Exception as exc:
            coverage = {"error": f"missing_hour_check_failed:{type(exc).__name__}"}

        status = "fail" if reasons else ("warn" if nonfinite > 0 else "pass")
        if status == "fail":
            failed = True
        clean_entry = {k: v for k, v in entry.items() if k not in {"valid", "_observed_hours"}}

        for metric in ("spread", "atr"):
            if isinstance(clean_entry.get(metric), dict):
                clean_entry[metric] = {k: v for k, v in clean_entry[metric].items() if not k.startswith("_")}
        clean_entry.update(
            {
                "n_features": n_feat,
                "nonfinite_count": nonfinite,
                "nonfinite_pct": round(nonfinite_pct, 6),
                "hour_coverage": coverage,
                "status": status,
                "reasons": list(dict.fromkeys(reasons)),
            }
        )
        pair_reports.append(clean_entry)

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "cache_path": str(cache_path),
        "data_source": str(getattr(args, "data_source", "")),
        "start_date": str(getattr(args, "data_start", "")),
        "end_date": str(getattr(args, "data_end", "")),
        "bar_freq": str(getattr(args, "bar_freq", "5min")),
        "seq_len": int(getattr(args, "seq_len", 0) or 0),
        "pairs": pair_reports,
        "alignment": alignment or {},
        "warnings": warnings,
        "status": "fail" if failed else ("warn" if warnings else "pass"),
    }
    return report


def _write_pair_readiness_report(args, cache_path, pairs, *, alignment: dict | None = None) -> dict:
    report = _finalize_pair_readiness_report(args, cache_path, pairs, alignment=alignment)
    path = Path(str(cache_path) + "_pair_readiness_report.json")
    _safe_save_json(report, path)
    print(f"[Pair Readiness] report -> {path} ({report['status'].upper()})")
    return report


def _compute_difficulty_scores(feats: pd.DataFrame, seq_len: int) -> np.ndarray:
    """
    B: Per-sample difficulty score (uint8: 0=easy, 1=medium, 2=hard).

    Difficulty is assigned per bar using four independent signals.
    The bar's score is the MAX of all signals (worst-case wins):

    Session signal (UTC hour):
      0 = easy   -> London / London-NY overlap: 07-17 UTC (peak liquidity)
      0 = easy   -> NY session: 13-21 UTC (high liquidity, strong trends)
      1 = medium -> Asian session: 00-07 UTC (thinner, range-bound)
      1 = medium -> Pre-London / post-NY: 05-07, 18-21 UTC
      2 = hard   -> Rollover / off-hours: 21-24 + 00-01 UTC (widest spreads,
                   swap charges, thin books)

    Spread signal (liquidity):
      Uses `liquidity_vacuum` if available (spread / median_spread ratio),
      else falls back to `spread_avg` / rolling median.
      1 = medium if spread > 1.5├ù median
      2 = hard   if spread > 2.0├ù median

    News window signal:
      2 = hard when `news_ok == 0` (within ┬▒15 min of high-impact release)
      1 = medium when `eco_surprise` is nonzero (data release bar itself)

    Volatility spike signal:
      2 = hard when `vol_ok == 0` (ATR > 3├ù rolling mean ΓÇö spike / flash crash)

    The difficulty of window i is that of its *last* bar (index i + seq_len - 1),
    matching how labels are assigned.
    """
    import pandas as pd

    idx = feats.index
    n = len(feats)
    hour_arr = np.array(idx.hour if isinstance(idx, pd.DatetimeIndex) else np.zeros(n, dtype=np.int32), dtype=np.int32)

    # -- Signal 1: session ----------------------------------------------------
    # hard = rollover (21–01 UTC): swap charges, thin books, wide spreads
    # medium = pure Asia (01–07 UTC) or late-NY tail (18–21 UTC)
    # easy = London peak (07–17) + NY core (13–21 overlap included in easy)
    is_rollover = (hour_arr >= 21) | (hour_arr < 1)  # 21:00–01:00 UTC
    is_asia = (hour_arr >= 1) & (hour_arr < 7)  # 01:00–07:00 UTC
    is_late_ny = (hour_arr >= 18) & (hour_arr < 21)  # 18:00–21:00 UTC
    session_diff = np.where(is_rollover, 2, np.where(is_asia | is_late_ny, 1, 0)).astype(np.uint8)

    base_diff = session_diff.copy()

    # -- Signal 2: spread / liquidity ----------------------------------------
    if "liquidity_vacuum" in feats.columns:
        lv = feats["liquidity_vacuum"].ffill().fillna(1.0).to_numpy(dtype=float)
        spread_diff = np.where(lv > 2.0, 2, np.where(lv > 1.5, 1, 0)).astype(np.uint8)
        base_diff = np.maximum(base_diff, spread_diff)
    elif "spread_avg" in feats.columns:
        spr = feats["spread_avg"].ffill().fillna(0.0).to_numpy(dtype=float)
        med_spr = pd.Series(spr).rolling(120, min_periods=10).median().ffill().fillna(0.0).to_numpy(dtype=float)
        spr_ratio = np.where(med_spr > 0, spr / np.maximum(med_spr, 1e-10), 1.0)
        spread_diff = np.where(spr_ratio > 2.0, 2, np.where(spr_ratio > 1.5, 1, 0)).astype(np.uint8)
        base_diff = np.maximum(base_diff, spread_diff)

    # -- Signal 3: news windows -----------------------------------------------
    # news_ok=0 means within ±15 min of a high-impact economic release.
    # These bars have erratic price action and artificially wide spreads.
    if "news_ok" in feats.columns:
        news_ok = feats["news_ok"].fillna(1.0).to_numpy(dtype=float)
        news_diff = np.where(news_ok < 0.5, 2, 0).astype(np.uint8)
        base_diff = np.maximum(base_diff, news_diff)

    # eco_surprise != 0 -> the release bar itself (not the buffer) -> medium
    if "eco_surprise" in feats.columns:
        eco = feats["eco_surprise"].fillna(0.0).to_numpy(dtype=float)
        eco_diff = np.where(eco != 0.0, np.maximum(np.ones(n, dtype=np.uint8), base_diff), base_diff).astype(np.uint8)
        base_diff = eco_diff

    # -- Signal 4: volatility spike -------------------------------------------
    # vol_ok=0 means ATR > 3× its rolling mean (flash crash / spike).
    # Model should not learn patterns from these bars.
    if "vol_ok" in feats.columns:
        vol_ok = feats["vol_ok"].fillna(1.0).to_numpy(dtype=float)
        spike_diff = np.where(vol_ok < 0.5, 2, 0).astype(np.uint8)
        base_diff = np.maximum(base_diff, spike_diff)

    # -- Align to sequence windows --------------------------------------------
    # Window i uses bars [i, i+seq_len). Its difficulty = last bar's score.
    n_seq = len(base_diff) - seq_len + 1
    if n_seq <= 0:
        return np.array([], dtype=np.uint8)
    diff_seq = base_diff[seq_len - 1 : seq_len - 1 + n_seq]
    return diff_seq.astype(np.uint8)


def _robust_clip_frame(feats: pd.DataFrame, *, q_low: float = 0.001, q_high: float = 0.999) -> pd.DataFrame:
    numeric = feats.select_dtypes(include=[np.number])
    if numeric.empty:
        return feats
    lo = numeric.quantile(q_low)
    hi = numeric.quantile(q_high)
    clipped = numeric.clip(lower=lo, upper=hi, axis=1)
    out = feats.copy()
    out[clipped.columns] = clipped
    return out


def _compute_row_quality_mask(
    bars: pd.DataFrame,
    feats: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.Series, dict, dict, dict]:

    bars_aligned = bars.reindex(feats.index).ffill()
    mask = pd.Series(True, index=feats.index)
    reason_masks: dict[str, pd.Series] = {}

    label_filter_counts = {
        "invalid_label": 0,
        "invalid_reward": 0,
        "invalid_direction_label": 0,
        "invalid_path_quality": 0,
    }

    def _reason(name: str, bad) -> None:

        nonlocal mask

        bad_s = pd.Series(bad, index=feats.index).fillna(False).astype(bool)

        reason_masks[name] = reason_masks.get(name, pd.Series(False, index=feats.index)) | bad_s

        mask &= ~bad_s

    try:
        idx = pd.DatetimeIndex(feats.index)

        if len(idx):
            weekday = idx.weekday

            hour = idx.hour

            weekend_closed = (weekday == 5) | ((weekday == 6) & (hour < 21)) | ((weekday == 4) & (hour >= 22))

            _reason("weekend", weekend_closed)

    except Exception:
        pass

    for col in ("holiday", "is_holiday", "holiday_flag", "market_holiday"):
        if col not in feats.columns:
            continue

        try:
            vals = feats[col]

            if pd.api.types.is_numeric_dtype(vals):
                _reason("holiday", vals.astype(float) > 0.5)

            else:
                _reason("holiday", vals.astype(str).str.lower().isin({"1", "true", "yes", "holiday"}))

        except Exception:
            pass

    dead_bar = pd.Series(False, index=feats.index)

    for col in ("open", "high", "low", "close"):
        if col in bars_aligned.columns:
            vals = bars_aligned[col].astype(float)
            dead_bar |= ~(np.isfinite(vals) & (vals > 0))

    if {"high", "low", "close"}.issubset(bars_aligned.columns):
        high = bars_aligned["high"].astype(float)

        low = bars_aligned["low"].astype(float)

        close = bars_aligned["close"].astype(float)

        dead_bar |= ~(high >= low)

        dead_bar |= ~close.between(low, high)

        if "open" in bars_aligned.columns:
            open_ = bars_aligned["open"].astype(float)

            flat_ohlc = ((high - low).abs() <= 1e-12) & ((open_ - close).abs() <= 1e-12)

            if "volume" in bars_aligned.columns:
                vol = pd.to_numeric(bars_aligned["volume"], errors="coerce").fillna(0.0)

                dead_bar |= flat_ohlc & (vol <= 0.0)

            else:
                dead_bar |= flat_ohlc

    _reason("dead_bar", dead_bar)

    for col in ("atr_6", "atr_20", "spread_pips", "liquidity_vacuum", "vol_ok", "news_ok"):
        if col in feats.columns:
            vals = feats[col].astype(float)
            bad_finite = ~np.isfinite(vals)

            if col == "news_ok":
                # Exclude high impact windows from training entirely!
                _reason("news", bad_finite | (vals <= 0.5))

            elif col in {"spread_pips", "liquidity_vacuum"}:
                _reason("spread", bad_finite)

            elif col in {"atr_6", "atr_20", "vol_ok"}:
                _reason("atr", bad_finite)

    if "atr_6" in feats.columns:
        atr = feats["atr_6"].astype(float)
        med_atr = atr.rolling(500, min_periods=50).median().ffill().fillna(atr.median())
        _reason("atr", (atr <= 0) | (atr > (med_atr * 25.0).replace(0, np.inf)))

    if "vol_ok" in feats.columns:
        _reason("atr", feats["vol_ok"].astype(float) <= 0.5)

    if "spread_pips" in feats.columns:
        spr = feats["spread_pips"].astype(float)
        med_spr = spr.rolling(500, min_periods=50).median().ffill().fillna(spr.median())
        _reason("spread", (spr < 0) | (spr > (med_spr * 20.0 + 0.1)))

    if "liquidity_vacuum" in feats.columns:
        _reason("spread", feats["liquidity_vacuum"].astype(float) > 20.0)

    if "label" in labels.columns:
        y = labels["label"].reindex(feats.index)
        bad_label = ~y.isin([-1, 0, 1])

        label_filter_counts["invalid_label"] += int(bad_label.fillna(True).sum())

        _reason("label_filter", bad_label)

    if "reward" in labels.columns:
        r = labels["reward"].reindex(feats.index).astype(float)
        bad_reward = pd.Series(~np.isfinite(r.to_numpy()), index=feats.index)

        label_filter_counts["invalid_reward"] += int(bad_reward.fillna(True).sum())

        _reason("label_filter", bad_reward)

    reason_counts = {k: int(v.fillna(False).sum()) for k, v in reason_masks.items()}

    return mask.fillna(False), reason_counts, label_filter_counts, reason_masks


def _sequence_quality_mask(
    X_arr: np.ndarray,
    row_ok: np.ndarray,
    seq_len: int,
    *,
    max_bad_frac: float = 0.05,
    max_zero_frac: float = 0.80,
) -> np.ndarray:
    n_seq = len(X_arr) - seq_len + 1
    if n_seq <= 0:
        return np.zeros(0, dtype=bool)
    row_bad = (~row_ok.astype(bool)).astype(np.float32)
    bad_counts = np.convolve(row_bad, np.ones(seq_len, dtype=np.float32), mode="valid")
    window_ok = bad_counts <= max(0.0, max_bad_frac) * seq_len
    zero_rows = (np.abs(X_arr).sum(axis=1) <= 1e-8).astype(np.float32)
    zero_counts = np.convolve(zero_rows, np.ones(seq_len, dtype=np.float32), mode="valid")
    window_ok &= zero_counts <= max_zero_frac * seq_len
    return window_ok.astype(bool)


def _sequence_quality_reason_masks(
    X_arr: np.ndarray,
    row_ok: np.ndarray,
    row_reason_masks: dict | None,
    seq_len: int,
    *,
    max_bad_frac: float = 0.05,
    max_zero_frac: float = 0.80,
) -> dict[str, np.ndarray]:

    n_seq = len(X_arr) - seq_len + 1

    if n_seq <= 0:
        return {}

    out: dict[str, np.ndarray] = {}

    row_bad = (~row_ok.astype(bool)).astype(np.float32)

    bad_counts = np.convolve(row_bad, np.ones(seq_len, dtype=np.float32), mode="valid")

    out["row_quality"] = bad_counts > max(0.0, max_bad_frac) * seq_len

    zero_rows = (np.abs(X_arr).sum(axis=1) <= 1e-8).astype(np.float32)

    zero_counts = np.convolve(zero_rows, np.ones(seq_len, dtype=np.float32), mode="valid")

    out["zero_feature_window"] = zero_counts > max_zero_frac * seq_len

    for name, values in (row_reason_masks or {}).items():
        try:
            arr = np.asarray(values, dtype=bool)

            if len(arr) != len(X_arr):
                continue

            reason_counts = np.convolve(arr.astype(np.float32), np.ones(seq_len, dtype=np.float32), mode="valid")

            out[str(name)] = reason_counts > 0

        except Exception:
            continue

    return out


# ─── Unified COT loader (used by main, parallel-worker, and single-pair paths) ─
# Hard-coded path matches the dataset-builder convention; the audit dated
# 2026-08-07 flagged divergent (silent) loads across call sites.
COT_PARQUET_PATH = Path("data/raw/cot/cot_financials_cleaned.parquet")


def load_cot(path: Path | str | None = None) -> pl.DataFrame | None:
    """Load the COT financials parquet, logging row count + status on every call.

    Returns None when the file is missing or unreadable. Polars is imported
    lazily here so importing this module does not require polars (matches the
    existing module-level lazy-import pattern used in `_build_chunk`).
    """
    p = Path(path) if path else COT_PARQUET_PATH
    _t0 = time.perf_counter()
    if not p.exists():
        log_data_load("cot", str(p), n_rows=0, status="skip_missing")
        return None
    try:
        import polars as pl

        df = pl.read_parquet(p)
        log_data_load("cot", str(p), n_rows=len(df), status="ok", t0=_t0, note=f"size_mb={p.stat().st_size / 1e6:.1f}")
        return df
    except Exception as _e:
        log_data_load("cot", str(p), n_rows=0, status="error", t0=_t0, exc=_e)
        return None


@dataclass(frozen=True)
class ChunkResult:
    """Result of ``_build_chunk`` - prefer named fields over positional indices."""

    X_seq: np.ndarray
    y_seq: np.ndarray
    diff_seq: np.ndarray
    pq_seq: np.ndarray
    y_cls_seq: np.ndarray
    close_seq: np.ndarray
    atr_seq: np.ndarray
    spread_seq: np.ndarray
    n_features: int
    time_idx: np.ndarray

    def __iter__(self):
        yield self.X_seq
        yield self.y_seq
        yield self.diff_seq
        yield self.pq_seq
        yield self.y_cls_seq
        yield self.close_seq
        yield self.atr_seq
        yield self.spread_seq
        yield self.n_features
        yield self.time_idx

    def __len__(self) -> int:
        return 10

    def __getitem__(self, key: Any):
        """Slice / index compatibility for legacy ``chunk[:4]`` unpacking."""
        items = (
            self.X_seq,
            self.y_seq,
            self.diff_seq,
            self.pq_seq,
            self.y_cls_seq,
            self.close_seq,
            self.atr_seq,
            self.spread_seq,
            self.n_features,
            self.time_idx,
        )
        return items[key]


# Back-compat alias
_ChunkResult = ChunkResult


def _chunk_result(
    X_seq,
    y_seq,
    diff_seq,
    pq_seq,
    y_cls_seq,
    close_seq,
    atr_seq,
    spread_seq,
    n_features,
    time_idx,
) -> ChunkResult:
    return ChunkResult(
        X_seq=X_seq,
        y_seq=y_seq,
        diff_seq=diff_seq,
        pq_seq=pq_seq,
        y_cls_seq=y_cls_seq,
        close_seq=close_seq,
        atr_seq=atr_seq,
        spread_seq=spread_seq,
        n_features=int(n_features),
        time_idx=time_idx,
    )


def _build_chunk(
    ticks_chunk,  # pd.DataFrame or pl.DataFrame - handed to ForexDataPipeline.run, which auto-converts
    fe: FeatureEngineer,
    scaler: StandardScaler,
    seq_len: int,
    chunk_idx: int,
    win_start: str | None = None,
    label_method: str = "rl_reward",
    target_col: str = "label",
    execution_delay_bars: int = 1,
    bar_freq: str = "1min",
    lookahead_bars: int | None = None,
    profit_target_atr: float | None = None,
    stop_loss_atr: float | None = None,
    cross_asset: dict[str, pd.Series] | None = None,
    sentiment_pipe: SentimentPipeline | None = None,
    pair: str = "EURUSD",
    historical_news_mode: str = "calendar",
    historical_news_file: str | None = None,
    economic_calendar_file: str | None = None,
    cot_data: Any = None,
    max_bad_frac: float | None = None,
    max_zero_frac: float | None = None,
) -> ChunkResult:
    """
    Process one tick chunk -> (X, y, diff, pq, y_cls, close, atr, spread, n_features, time_idx).

    close/atr/spread are per-sequence values at the label bar (last timestep of each window):
      - close: absolute FX quote (mid_close or close from bars)
      - atr: ATR in price units (matches ForexTradingEnv SL/TP)
      - spread: bid-ask width in price units (spread_pips * pip_size)
    """
    import pandas as pd
    import polars as pl

    _empty = (
        np.array([]),
        np.array([]),
        np.array([], dtype=np.uint8),
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
    )
    _empty_time = np.array([], dtype="datetime64[ns]")

    # Graceful exit for empty/bad chunks (e.g., all vendor hour files missing/empty)
    if ticks_chunk is None or len(ticks_chunk) == 0:
        _update_pair_readiness_raw(pair, ticks_chunk)
        return _chunk_result(*_empty, 0, _empty_time)
    _update_pair_readiness_raw(pair, ticks_chunk)
    if "timestamp_utc" not in ticks_chunk.columns:
        return _chunk_result(*_empty, 0, _empty_time)

    pipeline = ForexDataPipeline(
        bar_freq=str(bar_freq or "1min"),
        session_filter=False,
        apply_frac_diff=False,
        session_mode="dst",
        add_session_label=True,
        spread_cap_multiplier=3.0,
    )
    bars = pipeline.run(ticks_chunk, pair=pair)  # Polars DataFrame
    if len(bars) < seq_len + 20:
        return _chunk_result(*_empty, 0, _empty_time)

    # Time bounds from Polars - avoid Polars→Pandas→Polars round-trip for FE.
    _ts = bars["timestamp_utc"]
    _t0 = _ts.min()
    _t1 = _ts.max()
    if hasattr(_t0, "to_pydatetime") and callable(getattr(_t0, "to_pydatetime", None)):
        _t0 = _t0.to_pydatetime()
    if hasattr(_t1, "to_pydatetime") and callable(getattr(_t1, "to_pydatetime", None)):
        _t1 = _t1.to_pydatetime()

    news_bundle = load_historical_news_bundle(
        _t0,
        _t1,
        pair,
        mode=historical_news_mode,
        news_file=historical_news_file,
        calendar_file=economic_calendar_file,
    )
    # Pre-computed sentiment_score from historical_news_combined.parquet already
    # populates news_bundle.sentiment via data/historical_news.py. The live override
    # below only kicks in when (a) a SentimentPipeline is wired, and (b) the bundle
    # has news rows WITHOUT a pre-computed score (URL-fallback rows are NULL there
    # by design and must not be re-scored). See docs/NEWS_DATA_GUIDE.md §0.
    sent_pl = news_bundle.sentiment if news_bundle.sentiment is not None else None
    if sentiment_pipe is not None and news_bundle.news_events_df is not None and news_bundle.news_events_df.height > 0:
        try:
            news_pdf = news_bundle.news_events_df
            cols = set(news_pdf.columns)
            text_col = "headline" if "headline" in cols else ("text" if "text" in cols else None)
            ts_col = "timestamp_utc" if "timestamp_utc" in cols else ("timestamp" if "timestamp" in cols else None)
            if text_col is None or ts_col is None:
                raise ValueError(f"news_pdf missing text/ts col; has={sorted(cols)}")
            head_series = news_pdf[text_col].cast(pl.Utf8).fill_null("")
            is_url = head_series.str.to_lowercase().str.starts_with("http")
            is_blank = head_series.str.strip_chars().str.len_chars() == 0
            mask = ~(is_url | is_blank)
            if "sentiment_score" in cols:
                pre = news_pdf["sentiment_score"].cast(pl.Float64)
                mask = mask & pre.is_null()
            to_score_idx = [i for i, m in enumerate(mask.to_list()) if m]
            if to_score_idx:
                headlines_all = head_series.to_list()
                headlines = [headlines_all[i] for i in to_score_idx]
                scores = sentiment_pipe.score_headlines_batch(headlines)
                ts_all = news_pdf[ts_col].to_list()
                sent_override = pd.DataFrame(
                    {
                        "timestamp_utc": [ts_all[i] for i in to_score_idx],
                        "sentiment": scores,
                    }
                )
                # augment the bundle sentiment (if any) with newly scored rows
                new_pl = pl.from_pandas(sent_override)
                if sent_pl is not None and sent_pl.height > 0:
                    sent_pl = pl.concat([sent_pl, new_pl], how="vertical_relaxed")
                else:
                    sent_pl = new_pl
        except Exception as _sent_exc:
            print(f"[Chunk] sentiment override skipped: {_sent_exc}")

    cross_pl = None
    if cross_asset:
        cross_pl = {}
        for k, v in cross_asset.items():
            df_v = v.to_frame(name="value").reset_index()
            if "index" in df_v.columns:
                df_v = df_v.rename(columns={"index": "timestamp_utc"})
            if "timestamp" in df_v.columns:
                df_v = df_v.rename(columns={"timestamp": "timestamp_utc"})
            cross_pl[k] = pl.from_pandas(df_v)

    eco_act_pl = news_bundle.eco_actual
    eco_fc_pl = news_bundle.eco_forecast
    eco_prior_pl = getattr(news_bundle, "eco_prior", None)
    news_events = news_bundle.news_events or None
    art_counts_pl = news_bundle.article_counts
    news_cats_pl = news_bundle.category_flags

    _fe_kwargs = {
        "cross_asset": cross_pl,
        "sentiment": sent_pl,
        "eco_act": eco_act_pl,
        "eco_fc": eco_fc_pl,
        "eco_prior": eco_prior_pl,
        "art_counts": art_counts_pl,
        "finbert_embs": news_bundle.finbert_embeddings,
        "news_events": news_events,
        "cot_data": cot_data,
        "pair": pair,
        "news_cats": news_cats_pl,
    }

    # DS-002: when win_start is set, bars include warmup prefix - build with
    # warmup context then keep only the target window (EMA/MACD cold-start safe).
    if win_start:
        import pandas as pd

        ws_dt = pd.to_datetime(win_start, utc=True)
        _ws_lit = pl.lit(ws_dt)
        warmup_bars = bars.filter(pl.col("timestamp_utc") < _ws_lit)
        target_bars = bars.filter(pl.col("timestamp_utc") >= _ws_lit)
        if len(warmup_bars) > 0 and len(target_bars) >= seq_len + 10:
            F = fe.build_with_warmup(target_bars, warmup_bars, **_fe_kwargs)
            bars = target_bars
        else:
            F = fe.build(bars, **_fe_kwargs)
            if len(target_bars) > 0:
                F = F.filter(pl.col("timestamp_utc") >= _ws_lit)
                bars = target_bars
    else:
        F = fe.build(bars, **_fe_kwargs)

    # Normalize join-key precision once (pandas bridges often emit μs).
    _ts_ns = pl.Datetime("ns", "UTC")
    F = F.with_columns(pl.col("timestamp_utc").cast(_ts_ns))
    bars = bars.with_columns(pl.col("timestamp_utc").cast(_ts_ns))

    # Stay in Polars through mask / news_ok; only bridge to pandas for labeling
    # APIs that still require DatetimeIndex frames (one-way, never back to Polars).
    from config.feature_mask import apply_feature_mask as _apply_fm

    F = _apply_fm(F)
    if "news_ok" in F.columns:
        news_nt = (1.0 - pl.col("news_ok").cast(pl.Float64)).clip(0.0, 1.0)
        if "no_trade_score" in F.columns:
            F = F.with_columns(
                pl.max_horizontal(pl.col("no_trade_score").cast(pl.Float64), news_nt).alias("no_trade_score")
            )
        else:
            F = F.with_columns(news_nt.alias("no_trade_score"))
    if len(F) < seq_len + 10:
        return _chunk_result(*_empty, 0, _empty_time)

    # Thin one-way pandas bridge for labeling + row-quality (DatetimeIndex APIs).
    bars_pd = bars.to_pandas()
    if "timestamp_utc" in bars_pd.columns:
        bars_pd = bars_pd.set_index("timestamp_utc")
    bars_pd.index.name = "timestamp"
    feats_pd = F.to_pandas().set_index("timestamp_utc")
    # Labeling-only aux from bars (Utf8 / overlap floats) - keep out of X.
    # session_label + asia_london are aux; london_ny may already be in F as the
    # DST-aware (or fixed-UTC fallback) curriculum session feature.
    _aux_join = [c for c in ("session_label", "asia_london") if c in bars_pd.columns and c not in feats_pd.columns]
    if _aux_join:
        feats_pd = feats_pd.join(bars_pd[_aux_join], how="left")
    # If bars have DST london_ny and F still has crude/missing, prefer bars for
    # labeling policy (cost/horizon); X already locked from F.
    if "london_ny" in bars_pd.columns and "london_ny" in feats_pd.columns:
        # Overwrite labeling view with DST-aware bars flag when available.
        feats_pd["london_ny"] = bars_pd["london_ny"].reindex(feats_pd.index).fillna(feats_pd["london_ny"])
    elif "london_ny" in bars_pd.columns and "london_ny" not in feats_pd.columns:
        feats_pd = feats_pd.join(bars_pd[["london_ny"]], how="left")

    if label_method == "triple_barrier":
        labels = compute_triple_barrier_labels(
            bars_pd,
            feats_pd,
            vertical_bars=int(lookahead_bars or LABELING["lookahead_bars"]),
            profit_atr_mult=float(profit_target_atr or LABELING["profit_target_atr"]),
            stop_atr_mult=float(stop_loss_atr or LABELING["stop_loss_atr"]),
            pip_size=LABELING["pip_size"],
            execution_delay_bars=int(execution_delay_bars),
        )
    else:
        # Use regime-conditional labeling: barrier widths, session costs, bad-win
        # penalties, no-trade zones, and multi-target outputs (path_quality,
        # confidence_target) are all derived from feature columns when present.
        labels = compute_rl_reward_labels_regime(
            bars_pd,
            feats_pd,
            lookahead_bars=int(lookahead_bars or LABELING["lookahead_bars"]),
            pip_size=LABELING["pip_size"],
            session_col="session_label" if "session_label" in feats_pd.columns else None,
            regime_col="regime_class" if "regime_class" in feats_pd.columns else None,
            no_trade_col="no_trade_score" if "no_trade_score" in feats_pd.columns else None,
            latency_col="expected_latency_ms" if "expected_latency_ms" in feats_pd.columns else None,
            execution_delay_bars=int(execution_delay_bars),
        )
    row_quality, row_drop_reasons, label_filter_counts, row_reason_masks = _compute_row_quality_mask(
        bars_pd, feats_pd, labels
    )

    # Align in Polars (labels converted once; feature matrix never round-trips).
    X, y, sidecar = align_labels_with_features(labels, F, target_col=target_col)

    # Track stats for Pair Readiness Gate
    stats = _pair_readiness_entry(pair)
    _n_feat_cols = (
        int(X.width)
        if hasattr(X, "width")
        else (int(X.shape[1]) if hasattr(X, "shape") and len(getattr(X, "shape", ())) > 1 else 0)
    )
    stats["n_features"] = _n_feat_cols or int(stats.get("n_features", 0) or 0)
    _update_pair_readiness_processed(
        pair,
        bars_count=len(bars_pd),
        feature_frame=X,
        labels=labels,
    )
    _bump_reason_counts(stats, "dropped_bars_by_reason", row_drop_reasons)

    _bump_reason_counts(stats, "label_filter_counts", label_filter_counts)

    if len(X) < seq_len:
        return _chunk_result(*_empty, 0, _empty_time)

    x_index = pd.DatetimeIndex(pd.to_datetime(sidecar["timestamp_utc"].to_numpy(), utc=True))

    row_reason_values = {
        name: mask.reindex(x_index).fillna(False).to_numpy(dtype=bool) for name, mask in (row_reason_masks or {}).items()
    }

    row_quality = row_quality.reindex(x_index).fillna(False).to_numpy(dtype=bool)
    bad_rows = int((~row_quality).sum())
    if bad_rows:
        stats["dropped_bars"] += bad_rows
        print(f"[DataQuality] Chunk {chunk_idx}: flagged {bad_rows:,}/{len(row_quality):,} low-quality row(s)")

    global _FIRST_CHUNK_COLS
    cols = list(X.columns)
    if _FIRST_CHUNK_COLS is None:
        _FIRST_CHUNK_COLS = cols
    elif cols != _FIRST_CHUNK_COLS:
        missing = sorted(set(_FIRST_CHUNK_COLS) - set(cols))
        extra = sorted(set(cols) - set(_FIRST_CHUNK_COLS))
        if missing or extra:
            raise ValueError(f"Feature schema/order changed between chunks. Missing={missing}, Extra={extra}")
        # Same features, different order - align to the first chunk.
        if hasattr(X, "select") and callable(getattr(X, "select", None)):
            X = X.select(_FIRST_CHUNK_COLS)
        else:
            X = X[_FIRST_CHUNK_COLS]
        cols = list(X.columns)

    X_arr = sanitize_array(
        X.to_numpy(),
        context="chunk features before scaling",
        clip_range=None,  # Don't hard-clip - NaN/Inf handled by col_medians below
    )

    # Sanitize: replace +-inf / NaN (can arise from log-return or cross-asset
    # derived features) with per-column medians so features where 0 is a
    # meaningful signal (MACD, ROC, ATR ratio, etc.) are not corrupted.
    n_feat = X_arr.shape[1]

    if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != n_feat:
        import logging

        logging.getLogger("train_gpu").debug(f"Mismatch! Scaler expects {scaler.n_features_in_}, but got {n_feat}.")
        try:
            if hasattr(scaler, "feature_names_in_"):
                logging.getLogger("train_gpu").debug(f"Scaler expected columns: {list(scaler.feature_names_in_)}")
                missing = set(scaler.feature_names_in_) - set(cols)
                extra = set(cols) - set(scaler.feature_names_in_)
                logging.getLogger("train_gpu").debug(f"Missing columns: {missing}")
                logging.getLogger("train_gpu").debug(f"Extra columns: {extra}")
        except Exception:
            pass
        raise ValueError(f"Feature count mismatch: {n_feat} vs {scaler.n_features_in_}")

    _existing_feature_names = getattr(scaler, "feature_names_in_", None)

    # Scaler fit removed here to prevent D3 leakage. Scaling should be fit per-fold.
    # BUT: the schema/column order is fixed right here (cols is the canonical
    # Polars feature order). Attach those names to the scaler so downstream
    # `_build_multipair_feature_schema` can deserialise feature provenance for
    # the multi-pair schema-JSON sidecar that the post-process integrity gate
    # requires. `_set_scaler_feature_names` no-ops if names already attached.
    # (FSBUG-2026-08-07: without this, _build_multipair_feature_schema returns
    # [] because scaler.feature_names_in_ is None, so the schema JSON file is
    # never written and the integrity gate fails with "Multi-pair feature
    # schema missing" - see cache_integrity.py:529.)
    if _existing_feature_names is None and cols:
        _set_scaler_feature_names(scaler, cols)

    # Compute per-column finite medians BEFORE the main sanitization pass.
    # Any column with no finite values at all falls back to fill_value=0.0.
    _finite_X = np.where(np.isfinite(X_arr), X_arr, np.nan)
    _col_medians = np.nanmedian(_finite_X, axis=0)  # shape (n_feat,)

    X_arr = sanitize_array(
        X_arr,
        col_medians=_col_medians,
        context="chunk features unscaled",
        clip_range=None,  # NaN/Inf handled by col_medians, values clipped later by scaler
    )
    X_arr = np.asarray(X_arr, dtype=np.float32)

    # Path-quality gating: bars where the winning trade had a noisy/meandering path
    # (path_quality < 0.2) are relabelled as hold (0) to suppress gradient noise.
    y_arr = np.asarray(y.to_numpy(), dtype=np.float32)
    y_arr = sanitize_array(y_arr, context="chunk labels", clip_range=(-50.0, 50.0))

    if "path_quality" in sidecar.columns:
        pq_arr = np.asarray(sidecar["path_quality"].to_numpy(), dtype=np.float32)
    else:
        pq_arr = np.ones(len(y_arr), dtype=np.float32)
    # Reuse the labeling pandas bridge (aligned rows only) for market/difficulty/
    # no_trade helpers - avoids a second Polars join on timestamp precision.
    feats_aligned = feats_pd.reindex(x_index)
    if "no_trade_score" in feats_aligned.columns:
        no_trade_arr = feats_aligned["no_trade_score"].fillna(0.0).to_numpy(dtype=np.float32)
        pq_arr = pq_arr * np.clip(1.0 - no_trade_arr, 0.0, 1.0)

    # Build sliding window sequences
    # Window i uses rows [i, i+seq_len), label is the bar at i+seq_len-1 (last bar).
    # sliding_window_view produces (N - seq_len + 1) windows;
    # labels start at index seq_len-1 so there are (N - seq_len + 1) of them too.
    n_seq = len(X_arr) - seq_len + 1
    if n_seq <= 0:
        return _chunk_result(*_empty, n_feat, _empty_time)

    seq_ok = _sequence_quality_mask(
        X_arr,
        row_quality,
        seq_len,
        max_bad_frac=float(max_bad_frac if max_bad_frac is not None else 0.05),
        max_zero_frac=float(max_zero_frac if max_zero_frac is not None else 0.80),
    )
    seq_reason_masks = _sequence_quality_reason_masks(
        X_arr,
        row_quality,
        row_reason_values,
        seq_len,
        max_bad_frac=float(max_bad_frac if max_bad_frac is not None else 0.05),
        max_zero_frac=float(max_zero_frac if max_zero_frac is not None else 0.80),
    )

    X_seq = np.lib.stride_tricks.sliding_window_view(X_arr, (seq_len, n_feat)).squeeze(1)  # (n_seq, seq_len, n_feat)
    X_seq = np.ascontiguousarray(X_seq, dtype=np.float32)
    y_seq = np.asarray(y_arr[seq_len - 1 :], dtype=np.float32)  # label = last bar of each window
    if "label" in sidecar.columns:
        lbl_arr = np.asarray(sidecar["label"].to_numpy(), dtype=np.float32)
        y_cls_seq = np.asarray(lbl_arr[seq_len - 1 :], dtype=np.float32)
    else:
        y_cls_seq = np.sign(y_seq).astype(np.float32)
    pq_seq = pq_arr[seq_len - 1 :]  # path quality aligned with y_seq
    pq_seq = np.asarray(pq_seq, dtype=np.float32)
    close_seq, atr_seq, spread_seq = _market_bar_arrays_from_feats(feats_aligned, x_index, fe, seq_len)
    time_idx = x_index[seq_len - 1 :]

    # B: per-sample difficulty scores aligned with X (inner-joined features)
    try:
        diff_seq = _compute_difficulty_scores(feats_aligned, seq_len)
    except Exception as e:
        print(f"[DiffCurriculum] Difficulty scoring failed ({e}); defaulting all samples to easy (stage 0).")
        diff_seq = np.zeros(len(y_seq), dtype=np.uint8)

    if len(diff_seq) != len(y_seq):
        print(
            f"[DiffCurriculum] Difficulty length mismatch ({len(diff_seq)} vs {len(y_seq)}); "
            "defaulting all samples to easy (stage 0)."
        )
        diff_seq = np.zeros(len(y_seq), dtype=np.uint8)

    label_ok = np.isfinite(y_seq)
    if target_col == "label":
        label_ok &= np.isin(np.round(y_seq), [-1.0, 0.0, 1.0])
    cls_ok = np.isfinite(y_cls_seq) & np.isin(np.round(y_cls_seq), [-1.0, 0.0, 1.0])
    pq_ok = np.isfinite(pq_seq) & (pq_seq >= 0.0) & (pq_seq <= 1.0)
    target_reason = "invalid_reward_label" if target_col != "label" else "label_filter"

    if target_reason in seq_reason_masks:
        seq_reason_masks[target_reason] = np.asarray(seq_reason_masks[target_reason], dtype=bool) | ~label_ok

    else:
        seq_reason_masks[target_reason] = ~label_ok

    seq_reason_masks["invalid_direction_label"] = ~cls_ok

    seq_reason_masks["path_quality_filter"] = ~pq_ok

    _bump_reason_counts(
        stats,
        "label_filter_counts",
        {
            "invalid_direction_label": int((~cls_ok).sum()),
            "invalid_path_quality": int((~pq_ok).sum()),
        },
    )

    keep = seq_ok & label_ok & cls_ok & pq_ok
    dropped = int((~keep).sum())
    if dropped:
        print(f"[DataQuality] Chunk {chunk_idx}: dropped {dropped:,}/{len(keep):,} low-quality sequence(s)")
        stats["dropped_sequences"] += dropped
        _bump_reason_counts(
            stats,
            "dropped_sequences_by_reason",
            {name: int((np.asarray(mask, dtype=bool) & ~keep).sum()) for name, mask in seq_reason_masks.items()},
        )

    if not keep.any():
        return _chunk_result(*_empty, n_feat, _empty_time)
    X_seq = X_seq[keep]
    y_seq = y_seq[keep]
    diff_seq = diff_seq[keep]
    pq_seq = pq_seq[keep]
    y_cls_seq = y_cls_seq[keep]
    close_seq = close_seq[keep]
    atr_seq = atr_seq[keep]
    spread_seq = spread_seq[keep]
    time_idx = time_idx[keep]
    stats["seq_count"] += len(y_seq)
    stats["difficulty_counts"] = stats.get("difficulty_counts", {"0": 0, "1": 0, "2": 0})
    _update_pair_readiness_processed(
        pair,
        dropped_sequences=0,
        diff_seq=diff_seq,
        spread_seq=spread_seq,
        atr_seq=atr_seq,
    )

    return _chunk_result(
        X_seq,
        y_seq,
        diff_seq,
        pq_seq,
        y_cls_seq,
        close_seq,
        atr_seq,
        spread_seq,
        n_feat,
        time_idx,
    )


def _scaler_npz_path_pair(cache_path: Path, pair: str) -> Path:
    """Per-pair scaler sidecar: dataset_EURUSD-GBPUSD_..._scaler_EURUSD.npz"""
    return Path(_base_path(str(cache_path)) + f"_scaler_{pair}")


def _build_multipair_chunk(
    pair_ticks: dict,
    fe: FeatureEngineer,
    scalers: dict,
    seq_len: int,
    chunk_idx: int,
    win_start: str | None = None,
    label_method: str = "rl_reward",
    target_col: str = "label",
    execution_delay_bars: int = 1,
    bar_freq: str = "1min",
    lookahead_bars: int | None = None,
    profit_target_atr: float | None = None,
    stop_loss_atr: float | None = None,
    cross_asset: dict[str, pd.Series] | None = None,
    sentiment_pipe: SentimentPipeline | None = None,
    historical_news_mode: str = "calendar",
    historical_news_file: str | None = None,
    economic_calendar_file: str | None = None,
    cot_data: Any = None,
    max_bad_frac: float | None = None,
    max_zero_frac: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Process raw ticks for P pairs into joint sequences.

    pair_ticks : {pair_name: pd.DataFrame}
    scalers    : {pair_name: StandardScaler}  ΓÇö one per pair, fitted in-place

    Returns
    -------
    X : (N, T, P * F_per_pair)   ΓÇö pairs concatenated on feature axis
    y : (N,)                      ΓÇö mean label across pairs
    y_cls : (N,)                  ΓÇö consensus direction {-1,0,+1} across pairs
    pq : (N,)                     ΓÇö mean path-quality across pairs
    diff : (N,) uint8             ΓÇö max difficulty across pairs (curriculum)
    close/atr/spread : (N,) from the first pair with valid sequences (RL market path)
    n_features_total : int
    """
    global _FIRST_CHUNK_COLS
    _FIRST_CHUNK_COLS = None  # fresh schema lock per multi-pair build

    pair_Xs: dict = {}
    pair_ys: dict = {}
    pair_ycls: dict = {}
    pair_pqs: dict = {}
    pair_diffs: dict = {}
    pair_times: dict = {}
    market_close = market_atr = market_spread = None

    for pair, ticks in pair_ticks.items():
        chunk_result = _build_chunk(
            ticks,
            fe,
            scalers[pair],
            seq_len,
            chunk_idx,
            win_start=win_start,
            label_method=label_method,
            target_col=target_col,
            execution_delay_bars=execution_delay_bars,
            bar_freq=bar_freq,
            lookahead_bars=lookahead_bars,
            profit_target_atr=profit_target_atr,
            stop_loss_atr=stop_loss_atr,
            cross_asset=cross_asset,
            sentiment_pipe=sentiment_pipe,
            pair=pair,
            historical_news_mode=historical_news_mode,
            historical_news_file=historical_news_file,
            economic_calendar_file=economic_calendar_file,
            cot_data=cot_data,
            max_bad_frac=max_bad_frac,
            max_zero_frac=max_zero_frac,
        )
        _cr_tuple = tuple(chunk_result)
        if len(_cr_tuple) == 10:
            (X_seq, y_seq, diff_seq, pq_seq, y_cls_seq, close_seq, atr_seq, spread_seq, _, time_idx) = _cr_tuple
        elif len(_cr_tuple) == 9:
            (X_seq, y_seq, diff_seq, pq_seq, y_cls_seq, close_seq, atr_seq, spread_seq, _) = _cr_tuple
            # Older tests/mocks predate timestamp-aware alignment. Preserve their
            # positional behavior while real _build_chunk paths return time_idx.
            time_idx = np.arange(len(y_seq))
        else:
            raise ValueError(f"_build_chunk returned {len(_cr_tuple)} values; expected 9 or 10")
        if X_seq.size == 0:
            continue
        pair_Xs[pair] = X_seq  # (N, T, F)
        pair_ys[pair] = y_seq  # (N,)
        pair_ycls[pair] = y_cls_seq
        pair_pqs[pair] = pq_seq
        pair_diffs[pair] = diff_seq
        pair_times[pair] = time_idx
        if market_close is None:
            market_close, market_atr, market_spread = close_seq, atr_seq, spread_seq

    _empty8 = (
        np.array([]),
        np.array([]),
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([], dtype=np.uint8),
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
    )
    if not pair_Xs:
        return *_empty8, 0

    missing = [p for p in pair_ticks if p not in pair_Xs]
    if missing:
        print(f"Warning: Required pair(s) produced no usable sequences: {missing}. Skipping chunk.")
        return *_empty8, 0

    # PAIR READINESS GATE
    print("\n[Pair Readiness]")
    gate_failed = False
    global _PAIR_READINESS_STATS
    if "_PAIR_READINESS_STATS" in globals():
        for p in pair_ticks:
            if p not in _PAIR_READINESS_STATS:
                continue
            stats = _PAIR_READINESS_STATS[p]
            n_features = pair_Xs[p].shape[2] if p in pair_Xs else 120
            seq_len_val = pair_Xs[p].shape[1] if p in pair_Xs else 60
            total_values = max(1, stats.get("seq_count", 1) * seq_len_val * n_features)
            nan_pct = (stats.get("nan_count", 0) / total_values) * 100
            if stats.get("seq_count", 0) == 0 or nan_pct > 1.0:
                status = "FAIL"
                gate_failed = True
                stats.setdefault("reasons", []).append("chunk_pair_readiness_failed")
            elif nan_pct > 0.0:
                status = "WARN"
            else:
                status = "PASS"

            print(
                f"  {p} {status}  seq={stats.get('seq_count', 0):,} dropped={stats.get('dropped_bars', 0):,} nan_pct={nan_pct:.2f}%"
            )

    if gate_failed:
        print("[Pair Readiness] WARN: chunk-level gate failure recorded; final JSON report will fail the build.")

    # Timestamp inner join. Build explicit timestamp -> row index maps instead
    # of boolean masks so duplicate or out-of-order timestamps cannot leave
    # pairs with different row counts or mismatched sample order.
    def _time_key(value: Any) -> TimeKey:
        if isinstance(value, np.datetime64):
            return ("dt", int(value.astype("datetime64[ns]").astype(np.int64)))
        if hasattr(value, "to_datetime64"):
            dt_value = value.to_datetime64()
            return ("dt", int(dt_value.astype("datetime64[ns]").astype(np.int64)))
        if hasattr(value, "value") and value.__class__.__name__ == "Timestamp":
            return ("dt", int(value.value))
        if isinstance(value, np.generic):
            # ``.item()`` returns a hashable Python scalar (int / float / str).
            value = value.item()
        # Final fallback: stringify non-hashable values.  The set-intersection
        # below requires hashable keys, so a raw ndarray would fail at runtime
        # - we coerce to its first element's repr to keep the key unique
        # while staying hashable.
        if not isinstance(value, (str, int, float, bytes, tuple, frozenset)):
            return ("raw", repr(value))
        return ("raw", value)

    def _time_key_json(key: TimeKey) -> str:
        try:
            kind, value = key
            if kind == "dt":
                import pandas as pd

                return pd.Timestamp(int(value), unit="ns", tz="UTC").isoformat()
        except Exception:
            pass
        return str(_json_scalar(key))

    time_maps: dict[str, dict[TimeKey, int]] = {}
    common_keys_set: set[TimeKey] | None = None
    for p, times in pair_times.items():
        idx_by_key: dict[TimeKey, int] = {}
        for i, t in enumerate(np.asarray(times)):
            idx_by_key.setdefault(_time_key(t), i)
        time_maps[p] = idx_by_key
        keys = set(idx_by_key)
        common_keys_set = keys if common_keys_set is None else common_keys_set.intersection(keys)

    common_keys: list[TimeKey] = sorted(common_keys_set or set())
    if len(common_keys) == 0:
        globals()["_PAIR_ALIGNMENT_STATS"] = {
            "status": "fail",
            "reason": "no_common_timestamps",
            "input_sequence_counts": {p: len(pair_times.get(p, [])) for p in pair_ticks},
        }
        return *_empty8, 0

    _sample = next(iter(pair_Xs.values()))
    n_feat_per_pair = _sample.shape[2]

    y_list: list = []
    ycls_list: list = []
    pq_list: list = []
    diff_list: list = []
    pair_indices: dict = {}
    for pair in pair_ticks:
        idx = np.asarray([time_maps[pair][k] for k in common_keys], dtype=np.int64)
        pair_indices[pair] = idx
        y_list.append(pair_ys[pair][idx])
        ycls_list.append(pair_ycls[pair][idx])
        pq_list.append(pair_pqs[pair][idx])
        diff_list.append(pair_diffs[pair][idx])

    first_pair = next(iter(pair_ticks.keys()))
    market_idx = np.asarray([time_maps[first_pair][k] for k in common_keys], dtype=np.int64)
    if market_close is None or market_atr is None or market_spread is None:
        return *_empty8, 0
    market_close = market_close[market_idx]
    market_atr = market_atr[market_idx]
    market_spread = market_spread[market_idx]

    expected_rows = len(common_keys)
    row_counts = {pair: len(pair_indices[pair]) for pair in pair_ticks}
    input_counts = {p: len(pair_times[p]) for p in pair_ticks}
    dropped_by_inner_join = {p: int(max(0, input_counts[p] - expected_rows)) for p in pair_ticks}
    diff_vals, diff_counts = np.unique(np.max(np.stack(diff_list, axis=1), axis=1).astype(np.uint8), return_counts=True)
    globals()["_PAIR_ALIGNMENT_STATS"] = {
        "status": "pass",
        "pair_align": "inner",
        "common_sequence_count": int(expected_rows),
        "timestamp_start": _time_key_json(common_keys[0]) if common_keys else None,
        "timestamp_end": _time_key_json(common_keys[-1]) if common_keys else None,
        "input_sequence_counts": input_counts,
        "dropped_by_inner_join": dropped_by_inner_join,
        "drop_pct_by_pair": {
            p: round(100.0 * dropped_by_inner_join[p] / max(1, input_counts[p]), 6) for p in pair_ticks
        },
        "difficulty_counts_joint": {str(int(v)): int(c) for v, c in zip(diff_vals, diff_counts, strict=False)},
    }
    if any(n != expected_rows for n in row_counts.values()):
        globals()["_PAIR_ALIGNMENT_STATS"]["status"] = "fail"
        globals()["_PAIR_ALIGNMENT_STATS"]["reason"] = "mismatched_rows_after_alignment"
        raise RuntimeError(
            f"Pair timestamp alignment produced mismatched rows: expected={expected_rows}, per_pair={row_counts}"
        )

    # Build the aligned tensor directly instead of materializing per-pair
    # fancy-indexed copies and then concatenating them.  On 3-pair/full-feature
    # Windows runs this avoids multiple extra ~1 GiB peak allocations.
    pair_order = list(pair_ticks.keys())
    n_total_features = n_feat_per_pair * len(pair_order)
    try:
        X_multi = np.empty(
            (expected_rows, _sample.shape[1], n_total_features),
            dtype=np.float32,
        )
    except MemoryError as exc:
        need_gib = (expected_rows * _sample.shape[1] * n_total_features * np.dtype(np.float32).itemsize) / (1024**3)
        raise MemoryError(
            f"Unable to allocate aligned multi-pair tensor "
            f"({expected_rows:,} x {_sample.shape[1]} x {n_total_features}; "
            f"~{need_gib:.2f} GiB). Reduce real_data_window_days, seq_len, "
            "number of pairs, or disable high-cardinality feature groups."
        ) from exc

    row_batch = max(1, min(expected_rows, 256))
    for pair_pos, pair in enumerate(pair_order):
        src = pair_Xs[pair]
        idx = pair_indices[pair]
        feat_start = pair_pos * n_feat_per_pair
        feat_end = feat_start + n_feat_per_pair
        for row_start in range(0, expected_rows, row_batch):
            row_end = min(expected_rows, row_start + row_batch)
            X_multi[row_start:row_end, :, feat_start:feat_end] = src[idx[row_start:row_end]]
        pair_Xs[pair] = None
        gc.collect()

    y_multi = np.mean(np.stack(y_list, axis=1), axis=1).astype(np.float32)  # (N,)
    cls_mean = np.mean(np.stack(ycls_list, axis=1), axis=1)
    y_cls_multi = np.where(
        np.abs(cls_mean) < 0.33,
        0.0,
        np.sign(cls_mean),
    ).astype(np.float32)
    pq_multi = np.mean(np.stack(pq_list, axis=1), axis=1).astype(np.float32)
    diff_multi = np.max(np.stack(diff_list, axis=1), axis=1).astype(np.uint8)
    n_min = X_multi.shape[0]
    return (
        X_multi,
        y_multi,
        y_cls_multi,
        pq_multi,
        diff_multi,
        market_close[:n_min],
        market_atr[:n_min],
        market_spread[:n_min],
        X_multi.shape[2],
    )


def _merge_scalers(scaler_list: list[StandardScaler]) -> StandardScaler:
    """Merge independently fitted StandardScalers using the parallel merge formula.

    Each scaler must have been fitted via partial_fit / fit so that
    ``n_samples_seen_``, ``mean_``, and ``var_`` are populated.
    Returns a new StandardScaler with combined statistics.
    """
    from sklearn.preprocessing import StandardScaler

    combined = StandardScaler()
    if not scaler_list:
        return combined
    valid = [
        s
        for s in scaler_list
        if hasattr(s, "n_samples_seen_")
        and s.n_samples_seen_ is not None
        and getattr(s, "mean_", None) is not None
        and getattr(s, "var_", None) is not None
    ]
    if not valid:
        return scaler_list[0] if scaler_list else combined
    if len(valid) == 1:
        return valid[0]

    first_mean = valid[0].mean_
    if first_mean is None:
        return valid[0]

    total_n = sum(int(np.atleast_1d(s.n_samples_seen_)[0]) for s in valid)
    if total_n == 0:
        return valid[0]

    n_features = len(first_mean)
    combined_mean = np.zeros(n_features, dtype=np.float64)
    for s in valid:
        if s.mean_ is None:
            continue
        n = int(np.atleast_1d(s.n_samples_seen_)[0])
        combined_mean += n * s.mean_
    combined_mean /= total_n

    combined_var = np.zeros(n_features, dtype=np.float64)
    for s in valid:
        if s.var_ is None or s.mean_ is None:
            continue
        n = int(np.atleast_1d(s.n_samples_seen_)[0])
        combined_var += n * (s.var_ + (s.mean_ - combined_mean) ** 2)
    combined_var /= total_n

    combined.mean_ = combined_mean
    combined.var_ = combined_var
    combined.scale_ = np.sqrt(combined_var)
    combined.scale_[combined.scale_ == 0] = 1.0
    combined.n_samples_seen_ = np.full(n_features, total_n, dtype=np.int64)
    combined.n_features_in_ = n_features
    if hasattr(valid[0], "feature_names_in_"):
        combined.feature_names_in_ = valid[0].feature_names_in_
    return combined


def _parallel_window_worker(worker_args: dict):
    """Top-level picklable function for ProcessPoolExecutor window processing.

    Each worker creates its own FeatureEngineer, ForexDataManager,
    SentimentPipeline, and StandardScalers so that no shared mutable state
    crosses the process boundary.

    Returns
    -------
    dict with keys: window_idx, X, y, y_cls, pq, diff, close, atr, spread,
                    n_feat, scalers (dict of fitted StandardScaler per pair),
                    or error string on failure.
    """
    try:
        win_start = worker_args["win_start"]
        win_end = worker_args["win_end"]
        window_idx = worker_args["window_idx"]
        pairs = worker_args["pairs"]
        data_source = worker_args["data_source"]
        full_day_data = worker_args["full_day_data"]
        seq_len = worker_args["seq_len"]
        label_method = worker_args["label_method"]
        target_col = worker_args["target_col"]
        execution_delay_bars = worker_args["execution_delay_bars"]
        bar_freq = worker_args["bar_freq"]
        lookahead_bars = worker_args["lookahead_bars"]
        profit_target_atr = worker_args["profit_target_atr"]
        stop_loss_atr = worker_args["stop_loss_atr"]
        cross_asset = worker_args.get("cross_asset")
        sentiment_mode = worker_args["sentiment_mode"]
        historical_news_mode = worker_args["historical_news_mode"]
        historical_news_file = worker_args.get("historical_news_file")
        economic_calendar_file = worker_args.get("economic_calendar_file")
        cot_data_path = worker_args.get("cot_data_path")

        from sklearn.preprocessing import StandardScaler

        from config.settings import FEATURES
        from data.sources import ForexDataManager
        from features.feature_engineering_pl import FeatureEngineer

        fe = FeatureEngineer(
            atr_window=FEATURES["atr_window"],
            ofi_window=FEATURES["ofi_window"],
            tar_window=FEATURES["trade_arrival_window"],
            rsi_period=FEATURES["rsi_period"],
            macd_fast=FEATURES["macd_fast"],
            macd_slow=FEATURES["macd_slow"],
            macd_signal=FEATURES["macd_signal"],
            bb_window=FEATURES["bollinger_window"],
            bb_std=FEATURES["bollinger_std"],
            lag_windows=FEATURES["lag_windows"],
            enable_no_trade_zones=True,
        )
        scalers = {p: StandardScaler() for p in pairs}
        mgr = ForexDataManager(verbose=False)

        # DS-002: load warmup overlap, then slice features/labels at win_start.
        load_start = _warmup_load_start(win_start)
        pair_ticks = {}
        for p in pairs:
            pair_ticks[p] = mgr.load(
                pair=p,
                source=data_source,
                start=load_start,
                end=win_end,
                session_only=not full_day_data,
            )

        sentiment_pipe = None
        if str(sentiment_mode).lower() != "off":
            try:
                from features.finbert_sentiment import SentimentPipeline

                pref = "finbert" if str(sentiment_mode).lower() == "finbert" else "vader"
                sentiment_pipe = SentimentPipeline(prefer_backend=pref, use_cache=True)
            except Exception:
                sentiment_pipe = None

        if sentiment_pipe is not None and str(historical_news_mode).lower() == "full":
            try:
                from data.historical_news import collect_headlines_for_range as _chr

                _headlines = _chr(
                    win_start,
                    win_end,
                    pairs,
                    news_file=historical_news_file,
                    calendar_file=economic_calendar_file,
                )
                if _headlines:
                    sentiment_pipe.prefetch_headlines(_headlines)
                del _headlines
            except Exception as _pf_err:
                print(f"[Sentiment] prefetch skipped ({_pf_err})")
        cot_data = None
        if cot_data_path:
            cot_data = load_cot(cot_data_path)

        result = _build_multipair_chunk(
            pair_ticks,
            fe,
            scalers,
            seq_len,
            window_idx,
            win_start=win_start,
            label_method=label_method,
            target_col=target_col,
            execution_delay_bars=execution_delay_bars,
            bar_freq=bar_freq,
            lookahead_bars=lookahead_bars,
            profit_target_atr=profit_target_atr,
            stop_loss_atr=stop_loss_atr,
            cross_asset=cross_asset,
            sentiment_pipe=sentiment_pipe,
            historical_news_mode=historical_news_mode,
            historical_news_file=historical_news_file,
            economic_calendar_file=economic_calendar_file,
            cot_data=cot_data,
            max_bad_frac=float(worker_args.get("max_bad_frac", 0.05)),
            max_zero_frac=float(worker_args.get("max_zero_frac", 0.80)),
        )
        X_seq, y_seq, y_cls_seq, pq_seq, diff_seq, close_seq, atr_seq, spread_seq, n_feat = result

        return {
            "window_idx": window_idx,
            "X": X_seq,
            "y": y_seq,
            "y_cls": y_cls_seq,
            "pq": pq_seq,
            "diff": diff_seq,
            "close": close_seq,
            "atr": atr_seq,
            "spread": spread_seq,
            "n_feat": n_feat,
            "scalers": scalers,
        }
    except Exception:
        return {"window_idx": worker_args.get("window_idx", -1), "error": traceback.format_exc()}


def _build_multipair_dataset(
    args,
    pairs: list[str],
    cache_path: Path,
    fe: FeatureEngineer,
) -> tuple[str, int, int, StandardScaler]:
    """
    Multi-pair variant of build_dataset_chunked.
    Loads ticks for all pairs in parallel (dukascopy) or sequentially (other sources),
    builds joint (N, T, P*F) sequences, and writes them to Zarr (primary) / NPY (fallback).
    Returns (cache_path, n_samples, n_features, first_pair_scaler).
    """
    print(f"\n[MultiPair] {len(pairs)} pairs: {', '.join(pairs)}")
    _start = getattr(args, "data_start", "N/A")
    _end = getattr(args, "data_end", "N/A")
    print(f"            Source: {args.data_source} | {_start} -> {_end}")

    # ── Data coverage check ───────────────────────────────────────────
    from training.data_coverage import validate_pair_coverage

    if args.data_source != "synthetic":
        _coverage_valid, _coverage_report = validate_pair_coverage(
            pairs=pairs,
            data_source=args.data_source,
            min_years=getattr(args, "min_pair_years", 2),
            expected_years=getattr(args, "expected_pair_years", 18),
        )
    else:
        _coverage_valid, _coverage_report = pairs, []
    _skipped = [p for p in pairs if p not in _coverage_valid]
    if _skipped:
        print(f"[Coverage] ⚠  {len(_skipped)}/{len(pairs)} pairs have insufficient data (<2 years)")
        print(f"[Coverage]    Skipped: {', '.join(_skipped)}")
        print(f"[Coverage]    Training on: {', '.join(_coverage_valid) if _coverage_valid else 'NONE'}")
        if len(_coverage_valid) < 2:
            raise RuntimeError(
                f"Only {len(_coverage_valid)} pair(s) available for multi-pair training. "
                f"Need at least 2 pairs with >= 2 years of data. "
                f"Run: python scripts/download_data.py --pairs {' '.join(p for p in pairs if p not in _coverage_valid)}"
            )
        pairs = [p for p in pairs if p in _coverage_valid]
    else:
        low = [r["pair"] for r in _coverage_report if r["status"] == "LOW"]
        if low:
            print(f"[Coverage] ℹ  {len(low)} pairs have shorter-than-expected history (low coverage): {', '.join(low)}")  # noqa: RUF001

    use_real_cross = args.cross_asset_mode == "real" or (
        args.cross_asset_mode == "auto" and args.data_source != "synthetic"
    )
    cross_asset_source = _resolve_cross_asset_source(args)
    # Treat empty env var the same as "not set" so the default data/processed/cross_asset
    # path is always used when CROSS_ASSET_CACHE_DIR is unset or blank.
    cross_asset_cache_dir = os.getenv("CROSS_ASSET_CACHE_DIR", "").strip() or str(Path(args.data_cache) / "cross_asset")
    cross_asset = None
    sentiment_pipe = None
    cot_path = COT_PARQUET_PATH
    cot_data = load_cot(cot_path)
    if str(getattr(args, "sentiment_mode", "finbert")).lower() != "off":
        try:
            pref = "finbert" if str(args.sentiment_mode).lower() == "finbert" else "vader"
            sentiment_pipe = SentimentPipeline(prefer_backend=pref, use_cache=True)
            print(f"[Sentiment] mode={args.sentiment_mode} enabled")
        except Exception as e:
            print(f"[Sentiment] WARN: init failed ({e}) ΓÇö disabling sentiment features")
            sentiment_pipe = None
    if sentiment_pipe is not None and str(getattr(args, "historical_news_mode", "calendar")).lower() == "full":
        try:
            _headlines = collect_headlines_for_range(
                args.data_start,
                args.data_end,
                pairs,
                news_file=getattr(args, "historical_news_file", None),
                calendar_file=getattr(args, "economic_calendar_file", None),
            )
            if _headlines:
                sentiment_pipe.prefetch_headlines(_headlines)
            del _headlines
        except Exception as _pf_err:
            print(f"[Sentiment] prefetch skipped ({_pf_err})")

    if use_real_cross:
        try:
            cross_asset = load_cross_asset_panel(
                start=args.data_start,
                end=args.data_end,
                cache_dir=cross_asset_cache_dir,
                source=cross_asset_source,
            )
            print(
                f"[CrossAsset] Loaded external assets: {len(cross_asset)} "
                f"(source={cross_asset_source}, cache={cross_asset_cache_dir})"
            )
        except Exception as e:
            print(f"[CrossAsset] WARN: external load failed ({e}) ΓÇö falling back to synthetic")
            cross_asset = None

    scalers = {p: StandardScaler() for p in pairs}
    n_features = 0
    total_samples = 0
    z_store = None  # zarr (primary)
    pair_ticks: dict | None = None
    globals()["_PAIR_READINESS_STATS"] = {}
    globals()["_PAIR_ALIGNMENT_STATS"] = {}

    use_zarr = bool(ZARR)
    if ZARR and not use_zarr:
        print("[MultiPair] Zarr writes disabled on Windows; using NPY memmap cache.")
    if use_zarr:
        _compressor = make_training_zarr_compressor(
            getattr(args, "zarr_cname", None),
            getattr(args, "zarr_clevel", None),
        )
        _cname = getattr(_compressor, "cname", "lz4")
        if isinstance(_cname, bytes):
            _cname = _cname.decode()
        print(f"[Data] Zarr compressor=Blosc(cname={_cname}, clevel={getattr(_compressor, 'clevel', 1)})")
    else:
        # Out-of-core binary file states to prevent RAM OOM
        bin_state = {
            "opened": False,
            "x_f": None,
            "y_f": None,
            "ycls_f": None,
            "pq_f": None,
            "diff_f": None,
            "total": 0,
            "x_shape": None,
            "y_shape": None,
        }
    store_rl_sidecars = args.label_method == "rl_reward"

    def _sidecar_or_default(pq_seq, diff_seq, n_rows: int):
        pq = pq_seq if pq_seq is not None else np.ones(n_rows, dtype=np.float32)
        diff = diff_seq if diff_seq is not None else np.zeros(n_rows, dtype=np.uint8)
        return pq, diff

    def _append_chunk(X_seq, y_seq, y_cls_seq, pq_seq, diff_seq, close_seq, atr_seq, spread_seq):
        nonlocal z_store, total_samples
        n_rows = len(X_seq)
        pq_arr, diff_arr = _sidecar_or_default(pq_seq, diff_seq, n_rows)
        if use_zarr:
            _zs: Any = z_store
            if _zs is None:
                if getattr(args, "_resume_zarr", False):
                    z_store = _zs = _zarr_open_group(str(cache_path), mode="a")
                    existing_samples = int(_zs["X"].shape[0]) if "X" in _zs else 0
                    total_samples += existing_samples
                    _zs["X"].append(np.asarray(X_seq, dtype=ZARR_FEATURE_DTYPE))
                    _zs["y"].append(y_seq)
                    _zs["y_cls"].append(y_cls_seq)
                    _zs["close"].append(close_seq)
                    _zs["atr"].append(atr_seq)
                    _zs["spread"].append(spread_seq)
                    if store_rl_sidecars:
                        if "pq" in _zs:
                            _zs["pq"].append(pq_arr)
                        if "diff" in _zs:
                            _zs["diff"].append(diff_arr)
                else:
                    if __import__("pathlib").Path(cache_path).exists():
                        for _retry in range(10):
                            __import__("shutil").rmtree(cache_path, ignore_errors=True)
                            if not __import__("pathlib").Path(cache_path).exists():
                                break
                            __import__("time").sleep(0.5)
                    z_store = _zs = _zarr_open_group(str(cache_path), mode="w")
                    # 2048-row chunks: ~30x fewer decompressions per epoch vs 64-row chunks.
                    # X stored as FP16 (mixed-precision cache); labels/market stay FP32.
                    # Use resizeable arrays (shape=(0,)+dims) for safe .append() across windows.
                    c0 = (min(2048, len(X_seq)), *X_seq.shape[1:])
                    _zarr_create(
                        _zs,
                        "X",
                        shape=(0, *X_seq.shape[1:]),
                        chunks=c0,
                        dtype=ZARR_FEATURE_DTYPE,
                        compressor=_compressor,
                    )
                    _zarr_create(
                        _zs, "y", shape=(0,), chunks=(c0[0],), dtype=ZARR_LABEL_DTYPE, compressor=_compressor
                    )
                    _zarr_create(
                        _zs, "y_cls", shape=(0,), chunks=(c0[0],), dtype=ZARR_LABEL_DTYPE, compressor=_compressor
                    )
                    _zarr_create(
                        _zs, "close", shape=(0,), chunks=(c0[0],), dtype=ZARR_LABEL_DTYPE, compressor=_compressor
                    )
                    _zarr_create(
                        _zs, "atr", shape=(0,), chunks=(c0[0],), dtype=ZARR_LABEL_DTYPE, compressor=_compressor
                    )
                    _zarr_create(
                        _zs, "spread", shape=(0,), chunks=(c0[0],), dtype=ZARR_LABEL_DTYPE, compressor=_compressor
                    )
                    if store_rl_sidecars:
                        _zarr_create(
                            _zs, "pq", shape=(0,), chunks=(c0[0],), dtype=ZARR_LABEL_DTYPE, compressor=_compressor
                        )
                        _zarr_create(
                            _zs, "diff", shape=(0,), chunks=(c0[0],), dtype="uint8", compressor=_compressor
                        )
                    _zs["X"].append(np.asarray(X_seq, dtype=ZARR_FEATURE_DTYPE))
                    _zs["y"].append(y_seq)
                    _zs["y_cls"].append(y_cls_seq)
                    _zs["close"].append(close_seq)
                    _zs["atr"].append(atr_seq)
                    _zs["spread"].append(spread_seq)
                    if store_rl_sidecars:
                        _zs["pq"].append(pq_arr)
                        _zs["diff"].append(diff_arr)
            else:
                _zs["X"].append(np.asarray(X_seq, dtype=ZARR_FEATURE_DTYPE))
                _zs["y"].append(y_seq)
                _zs["y_cls"].append(y_cls_seq)
                _zs["close"].append(close_seq)
                _zs["atr"].append(atr_seq)
                _zs["spread"].append(spread_seq)
                if store_rl_sidecars:
                    if "pq" in _zs:
                        _zs["pq"].append(pq_arr)
                    if "diff" in _zs:
                        _zs["diff"].append(diff_arr)
        else:
            if not bin_state["opened"]:
                bin_state["x_f"] = open(str(_x_path(cache_path)) + ".bin", "wb")  # noqa: SIM115
                bin_state["y_f"] = open(str(_y_path(cache_path)) + ".bin", "wb")  # noqa: SIM115
                bin_state["ycls_f"] = open(_y_cls_path(cache_path).replace(".npy", ".bin"), "wb")  # noqa: SIM115
                bin_state["close_f"] = open(str(cache_path) + "_close.bin", "wb")  # noqa: SIM115
                bin_state["atr_f"] = open(str(cache_path) + "_atr.bin", "wb")  # noqa: SIM115
                bin_state["spread_f"] = open(str(cache_path) + "_spread.bin", "wb")  # noqa: SIM115
                if store_rl_sidecars:
                    bin_state["pq_f"] = open(str(cache_path) + "_pq.bin", "wb")  # noqa: SIM115
                    bin_state["diff_f"] = open(str(cache_path) + "_diff.bin", "wb")  # noqa: SIM115
                bin_state["x_shape"] = list(X_seq.shape)
                bin_state["y_shape"] = list(y_seq.shape)
                bin_state["opened"] = True

            X_seq.tofile(bin_state["x_f"])
            y_seq.tofile(bin_state["y_f"])
            bin_state["ycls_f"].write(y_cls_seq.tobytes())
            bin_state["close_f"].write(close_seq.tobytes())
            bin_state["atr_f"].write(atr_seq.tobytes())
            bin_state["spread_f"].write(spread_seq.tobytes())
            if store_rl_sidecars:
                bin_state["pq_f"].write(pq_arr.tobytes())
                bin_state["diff_f"].write(diff_arr.tobytes())
            bin_state["total"] += len(X_seq)

    # -- Load ticks ----------------------------------------------------------
    real_windows_handled = False
    if args.data_source != "synthetic":
        real_windows_handled = True
        _base_days = _real_data_window_days(args)
        window_days = _effective_window_days(args)
        date_windows = _iter_date_windows(args.data_start, args.data_end, window_days)
        _batch_n = max(1, int(getattr(args, "window_batch_days", 1) or 1))
        if _batch_n > 1:
            print(
                f"[MultiPair] Real-data windows: {len(date_windows)} x {window_days} day(s) "
                f"(base {_base_days}d x {_batch_n} batch)"
            )
        else:
            print(f"[MultiPair] Real-data windows: {len(date_windows)} x {window_days} day(s)")

        mgr = ForexDataManager(verbose=True)
        _build_workers = max(1, int(getattr(args, "dataset_build_workers", 1) or 1))

        resume_idx = -1
        if getattr(args, "_resume_zarr", False):
            try:
                import json

                with open(str(cache_path) + "_resume.json") as f:
                    resume_idx = json.load(f).get("last_completed_window_idx", -1)
            except Exception:
                pass

        def _load_window_ticks(win_start, win_end):
            """Load ticks for all pairs with DS-002 warmup overlap before win_start."""
            load_start = _warmup_load_start(win_start)
            ticks = {}
            for p in pairs:
                ticks[p] = mgr.load(
                    pair=p,
                    source=args.data_source,
                    start=load_start,
                    end=win_end,
                    session_only=not getattr(args, "full_day_data", False),
                )
            return ticks

        if _build_workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            print(f"[MultiPair] Parallel tick loading: {_build_workers} threads")

        _pending = list(enumerate(date_windows))
        _pending = [(idx, ws, we) for idx, (ws, we) in _pending if idx > resume_idx]

        _pw_workers = max(1, int(getattr(args, "parallel_window_workers", 1) or 1))

        if _pw_workers > 1:
            # ── PARALLEL window processing (multi-process) ──────────────
            from concurrent.futures import ProcessPoolExecutor, as_completed

            print(f"[MultiPair] Parallel window processing: {_pw_workers} processes")

            _cot_path_str = str(COT_PARQUET_PATH)
            _cot_path_exists = COT_PARQUET_PATH.exists()

            _worker_args_all = []
            for idx, ws, we in _pending:
                _worker_args_all.append(
                    {
                        "win_start": ws,
                        "win_end": we,
                        "window_idx": idx,
                        "pairs": list(pairs),
                        "data_source": args.data_source,
                        "full_day_data": bool(getattr(args, "full_day_data", False)),
                        "seq_len": _effective_max_seq_len(args),
                        "label_method": args.label_method,
                        "target_col": _cache_target_col(args),
                        "execution_delay_bars": int(getattr(args, "execution_delay_bars", 1)),
                        "bar_freq": str(getattr(args, "bar_freq", "5min")),
                        "lookahead_bars": int(getattr(args, "lookahead_bars", LABELING["lookahead_bars"])),
                        "profit_target_atr": float(getattr(args, "profit_target_atr", LABELING["profit_target_atr"])),
                        "stop_loss_atr": float(getattr(args, "stop_loss_atr", LABELING["stop_loss_atr"])),
                        "cross_asset": cross_asset,
                        "sentiment_mode": str(getattr(args, "sentiment_mode", "finbert")),
                        "historical_news_mode": str(getattr(args, "historical_news_mode", "calendar")),
                        "historical_news_file": getattr(args, "historical_news_file", None),
                        "economic_calendar_file": getattr(args, "economic_calendar_file", None),
                        "cot_data_path": _cot_path_str if _cot_path_exists else None,
                        "max_bad_frac": float(getattr(args, "max_bad_frac", 0.05)),
                        "max_zero_frac": float(getattr(args, "max_zero_frac", 0.80)),
                    }
                )

            _all_worker_scalers: dict[str, list] = {p: [] for p in pairs}
            _batch_sz = max(_pw_workers, 2)
            _n_errors = 0

            with ProcessPoolExecutor(max_workers=_pw_workers) as _p_pool:
                for _b_start in range(0, len(_worker_args_all), _batch_sz):
                    _batch = _worker_args_all[_b_start : _b_start + _batch_sz]
                    _batch_results: dict[int, dict] = {}

                    _futures = {_p_pool.submit(_parallel_window_worker, wa): wa["window_idx"] for wa in _batch}
                    for _fut in as_completed(_futures):
                        _res = _fut.result()
                        _widx = _res["window_idx"]
                        if "error" in _res:
                            print(f"  [Window {_widx + 1}/{len(date_windows)}] FAILED:\n{_res['error']}")
                            _n_errors += 1
                            continue
                        _batch_results[_widx] = _res

                for _wa in _batch:
                    _widx = _wa["window_idx"]
                    if _widx not in _batch_results:
                        continue
                    _res = _batch_results[_widx]
                    _X = _res["X"]
                    if _X is None or _X.size == 0:
                        del _batch_results[_widx]
                        continue

                    n_features = _res["n_feat"]
                    total_samples += len(_X)
                    _append_chunk(
                        _res["X"],
                        _res["y"],
                        _res["y_cls"],
                        _res["pq"],
                        _res["diff"],
                        _res["close"],
                        _res["atr"],
                        _res["spread"],
                    )
                    print(
                        f"  [Window {_widx + 1}/{len(date_windows)}] "
                        f"{len(_X):,} joint sequences | {total_samples:,} total"
                    )

                    for _p in pairs:
                        if _p in _res.get("scalers", {}):
                            _all_worker_scalers[_p].append(_res["scalers"][_p])

                    if not getattr(args, "_feature_schema_checked", False):
                        _partial = {p: _all_worker_scalers[p][0] for p in pairs if _all_worker_scalers.get(p)}
                        if len(_partial) == len(pairs):
                            args._feature_schema_checked = True
                            _enforce_dataset_feature_schema(
                                args,
                                _build_multipair_feature_schema(_partial, pairs, n_features),
                                cache_path,
                                phase="first_chunk",
                            )

                    try:
                        import json as _json_mod

                        with open(str(cache_path) + "_resume.json", "w") as _rf:
                            _json_mod.dump({"last_completed_window_idx": _widx}, _rf)
                    except Exception:
                        pass

                    del _batch_results[_widx]

                gc.collect()
                if _TRAIN_LOGGER:
                    _TRAIN_LOGGER.heartbeat()

            for _p in pairs:
                if _all_worker_scalers[_p]:
                    scalers[_p] = _merge_scalers(_all_worker_scalers[_p])

            if _n_errors:
                print(f"[MultiPair] WARNING: {_n_errors} window(s) failed during parallel processing")

        else:
            # ── SEQUENTIAL window processing (original behavior) ────────
            _pool = None
            if _build_workers > 1:
                from concurrent.futures import ThreadPoolExecutor

                print(f"[MultiPair] Parallel tick loading: {_build_workers} threads")

            if _build_workers > 1:
                _pool = ThreadPoolExecutor(max_workers=_build_workers)
                _prefetch = {}
                _look_ahead = min(_build_workers, 3)
                for i in range(min(_look_ahead, len(_pending))):
                    idx, ws, we = _pending[i]
                    _prefetch[idx] = _pool.submit(_load_window_ticks, ws, we)

            try:
                for _q_pos, (window_idx, win_start, win_end) in enumerate(_pending):
                    print(f"  [Window {window_idx + 1}/{len(date_windows)}] {win_start} -> {win_end}")

                    try:
                        if _pool is not None:
                            pair_ticks = _prefetch.pop(window_idx).result()
                            _next_q = _q_pos + _look_ahead
                            if _next_q < len(_pending):
                                _ni, _ns, _ne = _pending[_next_q]
                                _prefetch[_ni] = _pool.submit(_load_window_ticks, _ns, _ne)
                        else:
                            pair_ticks = _load_window_ticks(win_start, win_end)

                        if pair_ticks is None:
                            continue

                        X_seq, y_seq, y_cls_seq, pq_seq, diff_seq, close_seq, atr_seq, spread_seq, n_feat = (
                            _build_multipair_chunk(
                                pair_ticks,
                                fe,
                                scalers,
                                args.seq_len,
                                window_idx,
                                win_start=win_start,
                                label_method=args.label_method,
                                target_col=_cache_target_col(args),
                                execution_delay_bars=int(getattr(args, "execution_delay_bars", 1)),
                                bar_freq=str(getattr(args, "bar_freq", "5min")),
                                lookahead_bars=int(getattr(args, "lookahead_bars", LABELING["lookahead_bars"])),
                                profit_target_atr=float(
                                    getattr(args, "profit_target_atr", LABELING["profit_target_atr"])
                                ),
                                stop_loss_atr=float(getattr(args, "stop_loss_atr", LABELING["stop_loss_atr"])),
                                cross_asset=cross_asset,
                                sentiment_pipe=sentiment_pipe,
                                historical_news_mode=str(getattr(args, "historical_news_mode", "calendar")),
                                historical_news_file=getattr(args, "historical_news_file", None),
                                economic_calendar_file=getattr(args, "economic_calendar_file", None),
                                cot_data=cot_data,
                                max_bad_frac=float(getattr(args, "max_bad_frac", 0.05)),
                                max_zero_frac=float(getattr(args, "max_zero_frac", 0.80)),
                            )
                        )
                    except Exception as e:
                        import traceback

                        print(f"\n[CRITICAL ERROR] Failed during Window {window_idx + 1} ({win_start} -> {win_end}):")
                        traceback.print_exc()
                        print(f"[MultiPair] Skipping window {window_idx + 1} due to error.")
                        if _pool is not None and getattr(e, "shutdown", False):
                            pass  # Pool is already shutting down
                        # We don't want to crash the whole pipeline just because one window threw an error
                        # unless it's a MemoryError.
                        if isinstance(e, MemoryError):
                            sys.exit(1)
                        continue

                    if X_seq is None or X_seq.size == 0:
                        del pair_ticks
                        gc.collect()
                        continue
                    n_features = n_feat
                    total_samples += len(X_seq)
                    _append_chunk(X_seq, y_seq, y_cls_seq, pq_seq, diff_seq, close_seq, atr_seq, spread_seq)
                    print(f"    {len(X_seq):,} joint sequences | {total_samples:,} total")
                    if not getattr(args, "_feature_schema_checked", False):
                        args._feature_schema_checked = True
                        _enforce_dataset_feature_schema(
                            args,
                            _build_multipair_feature_schema(scalers, pairs, n_features),
                            cache_path,
                            phase="first_chunk",
                        )
                    if _TRAIN_LOGGER:
                        _TRAIN_LOGGER.heartbeat()
                    del pair_ticks, X_seq, y_seq, y_cls_seq, pq_seq, diff_seq
                    del close_seq, atr_seq, spread_seq
                    try:
                        import json

                        with open(str(cache_path) + "_resume.json", "w") as f:
                            json.dump({"last_completed_window_idx": window_idx}, f)
                    except Exception:
                        pass
                    gc.collect()
            finally:
                if _pool is not None:
                    _pool.shutdown(wait=False)

    if args.data_source == "synthetic":
        n_remaining = args.n_ticks
        chunk_n = 0
        while n_remaining > 0:
            chunk_n_ticks = min(args.chunk_size, n_remaining)
            pair_ticks = {p: generate_synthetic_tick_data(n_rows=chunk_n_ticks) for p in pairs}
            X_seq, y_seq, y_cls_seq, pq_seq, diff_seq, close_seq, atr_seq, spread_seq, n_feat = _build_multipair_chunk(
                pair_ticks,
                fe,
                scalers,
                args.seq_len,
                chunk_n,
                args.label_method,
                target_col=_cache_target_col(args),
                execution_delay_bars=int(getattr(args, "execution_delay_bars", 1)),
                bar_freq=str(getattr(args, "bar_freq", "5min")),
                lookahead_bars=int(getattr(args, "lookahead_bars", LABELING["lookahead_bars"])),
                profit_target_atr=float(getattr(args, "profit_target_atr", LABELING["profit_target_atr"])),
                stop_loss_atr=float(getattr(args, "stop_loss_atr", LABELING["stop_loss_atr"])),
                cross_asset=cross_asset,
                sentiment_pipe=sentiment_pipe,
                historical_news_mode=str(getattr(args, "historical_news_mode", "calendar")),
                historical_news_file=getattr(args, "historical_news_file", None),
                economic_calendar_file=getattr(args, "economic_calendar_file", None),
                cot_data=cot_data,
                max_bad_frac=float(getattr(args, "max_bad_frac", 0.05)),
                max_zero_frac=float(getattr(args, "max_zero_frac", 0.80)),
            )
            if X_seq is not None and X_seq.size > 0:
                n_features = n_feat
                total_samples += len(X_seq)
                _append_chunk(X_seq, y_seq, y_cls_seq, pq_seq, diff_seq, close_seq, atr_seq, spread_seq)
                pct = min((args.n_ticks - n_remaining + chunk_n_ticks) / args.n_ticks * 100, 100)
                print(f"  Chunk {chunk_n + 1} | {len(X_seq):,} seqs | {pct:.0f}%")
                if not getattr(args, "_feature_schema_checked", False):
                    args._feature_schema_checked = True
                    _enforce_dataset_feature_schema(
                        args,
                        _build_multipair_feature_schema(scalers, pairs, n_features),
                        cache_path,
                        phase="first_chunk",
                    )
            n_remaining -= chunk_n_ticks
            chunk_n += 1

    elif not real_windows_handled:
        # Real data: load all pairs at once then process
        mgr = ForexDataManager(verbose=True)
        pair_ticks = {}
        for p in pairs:
            print(f"  Loading {p}...")
            pair_ticks[p] = mgr.load(
                pair=p,
                source=args.data_source,
                start=args.data_start,
                end=args.data_end,
                session_only=not getattr(args, "full_day_data", False),
            )

        X_seq, y_seq, y_cls_seq, pq_seq, diff_seq, close_seq, atr_seq, spread_seq, n_feat = _build_multipair_chunk(
            pair_ticks,
            fe,
            scalers,
            args.seq_len,
            0,
            args.label_method,
            target_col=_cache_target_col(args),
            execution_delay_bars=int(getattr(args, "execution_delay_bars", 1)),
            bar_freq=str(getattr(args, "bar_freq", "5min")),
            lookahead_bars=int(getattr(args, "lookahead_bars", LABELING["lookahead_bars"])),
            profit_target_atr=float(getattr(args, "profit_target_atr", LABELING["profit_target_atr"])),
            stop_loss_atr=float(getattr(args, "stop_loss_atr", LABELING["stop_loss_atr"])),
            cross_asset=cross_asset,
            sentiment_pipe=sentiment_pipe,
            historical_news_mode=str(getattr(args, "historical_news_mode", "calendar")),
            historical_news_file=getattr(args, "historical_news_file", None),
            economic_calendar_file=getattr(args, "economic_calendar_file", None),
            cot_data=cot_data,
            max_bad_frac=float(getattr(args, "max_bad_frac", 0.05)),
            max_zero_frac=float(getattr(args, "max_zero_frac", 0.80)),
        )
        if X_seq.size > 0:
            n_features = n_feat
            total_samples = len(X_seq)
            _append_chunk(X_seq, y_seq, y_cls_seq, pq_seq, diff_seq, close_seq, atr_seq, spread_seq)
            print(f"  {total_samples:,} joint sequences ├ù {n_features} features")
            if not getattr(args, "_feature_schema_checked", False):
                args._feature_schema_checked = True
                _enforce_dataset_feature_schema(
                    args,
                    _build_multipair_feature_schema(scalers, pairs, n_features),
                    cache_path,
                    phase="first_chunk",
                )

    if total_samples == 0:
        err = (
            "[MultiPair] No usable samples produced. Check date range and data source.\n"
            + _multipair_zero_samples_help(None)
        )
        print(err)
        raise RuntimeError(err)

    # -- Finalise cache -------------------------------------------------------
    if use_zarr and z_store is not None:
        z_store.attrs["total_samples"] = total_samples
        z_store.attrs["n_features"] = n_features
        z_store.attrs["seq_len"] = _effective_max_seq_len(args)
        z_store.attrs["n_pairs"] = len(pairs)
        z_store.attrs["pairs"] = ",".join(pairs)
        z_store.attrs["strategy_mode"] = str(getattr(args, "strategy_mode", "scalping"))
        z_store.attrs["bar_freq"] = str(getattr(args, "bar_freq", "5min"))
        z_store.attrs["lookahead_bars"] = int(getattr(args, "lookahead_bars", LABELING["lookahead_bars"]))
        import json

        meta = {
            "total_samples": int(total_samples),
            "n_features": int(n_features),
            "seq_len": int(_effective_max_seq_len(args)),
            "n_pairs": len(pairs),
            "pairs": list(pairs),
            "strategy_mode": str(getattr(args, "strategy_mode", "scalping")),
            "bar_freq": str(getattr(args, "bar_freq", "5min")),
            "lookahead_bars": int(getattr(args, "lookahead_bars", LABELING["lookahead_bars"])),
            "has_rl_market": True,
            "target_col": _cache_target_col(args),
            "y_cls_source": "labels.label",
        }
        with open(str(cache_path) + "_manifest.json", "w") as f:
            json.dump(meta, f)
        _write_feature_schema_json(
            cache_path,
            _build_multipair_feature_schema(scalers, pairs, n_features),
            args,
        )

        _save_scaler_npz(cache_path, scalers[pairs[0]])
    else:
        if bin_state["opened"]:
            bin_state["x_f"].close()
            bin_state["y_f"].close()
            if bin_state.get("ycls_f"):
                bin_state["ycls_f"].close()
            if bin_state.get("pq_f"):
                bin_state["pq_f"].close()
            if bin_state.get("diff_f"):
                bin_state["diff_f"].close()
            for _mk in ("close_f", "atr_f", "spread_f"):
                if bin_state.get(_mk):
                    bin_state[_mk].close()

            final_x_shape = (bin_state["total"], *bin_state["x_shape"][1:])
            final_y_shape = (bin_state["total"], *bin_state["y_shape"][1:])
            final_1d = (bin_state["total"],)

            x_mmap = np.memmap(str(_x_path(cache_path)) + ".bin", dtype="float32", mode="r", shape=final_x_shape)
            y_mmap = np.memmap(str(_y_path(cache_path)) + ".bin", dtype="float32", mode="r", shape=final_y_shape)
            np.save(_x_path(cache_path), x_mmap)
            np.save(_y_path(cache_path), y_mmap)

            ycls_bin = _y_cls_path(cache_path).replace(".npy", ".bin")
            if Path(ycls_bin).exists():
                ycls_mmap = np.memmap(ycls_bin, dtype="float32", mode="r", shape=final_1d)
                np.save(_y_cls_path(cache_path), ycls_mmap)
            if store_rl_sidecars:
                pq_bin = str(cache_path) + "_pq.bin"
                diff_bin = str(cache_path) + "_diff.bin"
                if Path(pq_bin).exists():
                    pq_mmap = np.memmap(pq_bin, dtype="float32", mode="r", shape=final_1d)
                    np.save(_pq_path(cache_path), pq_mmap)
                if Path(diff_bin).exists():
                    diff_mmap = np.memmap(diff_bin, dtype="uint8", mode="r", shape=final_1d)
                    np.save(_diff_path(cache_path), diff_mmap)
            for _bin_suffix, _save_fn in (
                ("_close.bin", _close_path),
                ("_atr.bin", _atr_path),
                ("_spread.bin", _spread_path),
            ):
                _bpath = str(cache_path) + _bin_suffix
                if Path(_bpath).exists():
                    _mmap = np.memmap(_bpath, dtype="float32", mode="r", shape=final_1d)
                    np.save(_save_fn(cache_path), _mmap)

            import json

            meta = {
                "total_samples": bin_state["total"],
                "n_features": final_x_shape[2],
                "seq_len": _effective_max_seq_len(args),
                "n_pairs": len(pairs),
                "pairs": list(pairs),
                "strategy_mode": str(getattr(args, "strategy_mode", "scalping")),
                "bar_freq": str(getattr(args, "bar_freq", "5min")),
                "lookahead_bars": int(getattr(args, "lookahead_bars", LABELING["lookahead_bars"])),
                "has_rl_market": True,
                "target_col": _cache_target_col(args),
                "y_cls_source": "labels.label",
            }
            with open(str(cache_path) + "_manifest.json", "w") as f:
                json.dump(meta, f)
            _write_feature_schema_json(
                cache_path,
                _build_multipair_feature_schema(scalers, pairs, final_x_shape[2]),
                args,
            )

            total_samples = bin_state["total"]
            n_features = final_x_shape[2]

        _save_scaler_npz(cache_path, scalers[pairs[0]])

    for p, sc in scalers.items():
        _save_scaler_npz(_scaler_npz_path_pair(cache_path, p), sc)

    readiness_report = _write_pair_readiness_report(
        args,
        cache_path,
        pairs,
        alignment=globals().get("_PAIR_ALIGNMENT_STATS", {}),
    )
    if readiness_report.get("status") == "fail":
        raise RuntimeError(f"Pair Readiness Gate Failed. See {cache_path!s}_pair_readiness_report.json")

    _postprocess_cache_integrity_check(str(cache_path), args, context="MultiPair")

    print(f"\n[MultiPair] Dataset built: {total_samples:,} samples ├ù {n_features} features")
    _verify_dataset(str(cache_path), args, total_samples, n_features, context="MultiPair")
    print(f"            Cached: {cache_path}")

    # ── Write enriched dataset manifest + build log ──────────
    try:
        from data.dataset_manifest import DatasetManifest

        _dm = DatasetManifest(str(Path(cache_path).parent))
        _dm.log_build_event("build_complete", n_rows=total_samples, n_features=n_features)
        _dm.write_manifest(
            source=str(getattr(args, "data_source", "dukascopy")),
            pairs=pairs,
            start=str(getattr(args, "data_start", "")),
            end=str(getattr(args, "data_end", "")),
            freq=str(getattr(args, "bar_freq", "5min")),
            news_mode=bool(str(getattr(args, "historical_news_mode", "calendar")).lower() not in ("none", "false", "0")),
            feature_count=n_features,
            label_method=str(getattr(args, "label_method", "rl_reward")),
            seq_len=int(args.seq_len),
            schema_hash="",
            feature_list=[],
            n_rows_total=total_samples,
            lookahead_bars=int(getattr(args, "lookahead_bars", LABELING.get("lookahead_bars", 30))),
            embargo_bars=int(getattr(args, "embargo_bars", LABELING.get("embargo_bars", 60))),
            purge_bars=int(getattr(args, "purge_bars", 120)),
        )
    except Exception as _m_err:
        print(f"[Manifest] write failed ({_m_err})")

    # ── Future-leak check ────────────────────────────────────
    try:
        # FIX: previously passed ``None`` as feature_df which always short-circuited
        # the correlation scan (see ``DatasetManifest.check_future_leak``). Now we
        # read a small sample of last-timestep features from the Zarr cache and
        # let the correlation check actually run.
        _feats = _leak_check_features_sample(cache_path, max_sample=5000)
        _fwd_ret = None
        if str(cache_path).endswith(".zarr") and Path(cache_path).is_dir():
            import zarr as _zarr

            _z: Any = _zarr.open(str(cache_path), mode="r")
            if "y" in _z:
                _fwd_ret = np.asarray(_z["y"][:], dtype=np.float32)
        if _feats is not None and _fwd_ret is not None and len(_fwd_ret) > 0:
            _X_2d, _feat_names = _feats
            import pandas as _pd

            _feat_df = _pd.DataFrame(_X_2d, columns=_feat_names)
            _leaks = DatasetManifest.check_future_leak(
                _feat_df,
                _fwd_ret.tolist(),
                max_abs_corr=0.30,
            )
            if _leaks:
                print(
                    f"[LeakCheck] {len(_leaks)} feature(s) correlated with "
                    f"forward returns (|r| > 0.30). Review for data leakage."
                )
            else:
                print("[LeakCheck] No features correlated with forward returns.")
        # Label contamination: feature window timestamps < label timestamps.
        _lc = _label_contamination_check(cache_path, args)
        if not _lc.get("ok", True):
            _n_v = int(_lc.get("violations", 0))
            _n_t = int(_lc.get("total_checked", 0))
            print(f"[LabelContamination] FAIL: {_n_v}/{_n_t} samples have feature_ts >= label_ts.")
        elif _lc.get("total_checked", 0) > 0:
            print(f"[LabelContamination] PASS: {_lc['total_checked']:,} samples verified.")
    except Exception as _le_err:
        print(f"[LeakCheck] skipped ({_le_err})")

    # ── Lockbox reservation ──────────────────────────────────
    try:
        _lookback_days = int(getattr(args, "real_data_window_days", 7) or 7)
        _lockbox_end = str(getattr(args, "data_end", datetime.now(UTC).strftime("%Y-%m-%d")))
        _lockbox_start = (datetime.fromisoformat(_lockbox_end) - timedelta(days=_lookback_days)).strftime("%Y-%m-%d")
        DatasetManifest.reserve_lockbox(str(Path(cache_path).parent), _lockbox_start, _lockbox_end)
    except Exception:
        pass

    return str(cache_path), total_samples, n_features, scalers[pairs[0]]


def build_dataset_chunked(args) -> tuple[str, int, int, StandardScaler]:
    """
    Ingest up to 20M ticks in chunks, write sequences to Zarr (primary) / NPY memmap (fallback).

    Returns: (cache_path, n_samples, n_features, scaler)
    """

    use_zarr = bool(ZARR)
    pairs = getattr(args, "pairs", []) or []
    if isinstance(pairs, str):
        pairs = [p.strip() for p in pairs.split(",")]
    if not pairs and getattr(args, "pair", None):
        pairs = [args.pair]
    is_multi = len(pairs) > 1
    if not is_multi:
        globals()["_PAIR_READINESS_STATS"] = {}
        globals()["_PAIR_ALIGNMENT_STATS"] = {}

    def _cache_present(cp: Path) -> bool:
        if str(cp).endswith(".zarr"):
            return cp.is_dir()
        return Path(_x_path(cp)).exists() and Path(_y_path(cp)).exists()

    # Multi-pair: delegate to dedicated function
    if is_multi:
        Path(args.data_cache).mkdir(parents=True, exist_ok=True)
        cache_path = _get_cache_path(args)
        if _cache_present(cache_path) and not args.force_rebuild:
            ok, reason = _validate_cache_integrity(str(cache_path), args)
            if not ok:
                if getattr(args, "auto_rebuild_on_mismatch", False):
                    print(f"[MultiPair] Cache mismatch ({reason}) - rebuilding.")
                    _delete_cache_artifacts(str(cache_path))
                elif getattr(args, "integrity_gate", True):
                    raise RuntimeError(f"Cache integrity failed: {reason}. Use --force-rebuild.")
        if _cache_present(cache_path) and not args.force_rebuild:
            cache_format = "zarr" if str(cache_path).endswith(".zarr") else "npy"
            if getattr(args, "_resume_zarr", False):
                pass  # Fall through to rebuild
            elif cache_format == "zarr":
                import zarr

                z_store: Any = zarr.open(cache_path, mode="r")
                if "total_samples" not in z_store.attrs:
                    args._resume_zarr = True
                else:
                    n_samples = int(z_store.attrs["total_samples"])
                    n_features = int(z_store.attrs["n_features"])
                    scaler = _load_scaler_npz(Path(cache_path)) or _identity_scaler(n_features)
                    n_samples = _clamp_n_samples_to_disk(str(cache_path), n_samples)
                    print(f"[MultiPair] {n_samples:,} samples x {n_features} features (cached)")
                    _warn_multitask_cache_sidecars(str(cache_path), args)
                    args._n_pairs = len(pairs)
                    args._f_per_pair = n_features // len(pairs)
                    return str(cache_path), n_samples, n_features, scaler
            else:
                import json

                meta_path = str(cache_path) + "_meta.json"
                if not os.path.exists(meta_path):
                    args._resume_zarr = True
                else:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    n_samples = int(meta["total_samples"])
                    n_features = int(meta["n_features"])
                    scaler = _load_scaler_npz(Path(cache_path)) or _identity_scaler(n_features)
                    n_samples = _clamp_n_samples_to_disk(str(cache_path), n_samples)
                    print(f"[MultiPair] {n_samples:,} samples x {n_features} features (cached)")
                    _warn_multitask_cache_sidecars(str(cache_path), args)
                    args._n_pairs = len(pairs)
                    args._f_per_pair = n_features // len(pairs)
                    return str(cache_path), n_samples, n_features, scaler
        fe = FeatureEngineer(
            atr_window=FEATURES["atr_window"],
            ofi_window=FEATURES["ofi_window"],
            tar_window=FEATURES["trade_arrival_window"],
            rsi_period=FEATURES["rsi_period"],
            macd_fast=FEATURES["macd_fast"],
            macd_slow=FEATURES["macd_slow"],
            macd_signal=FEATURES["macd_signal"],
            bb_window=FEATURES["bollinger_window"],
            bb_std=FEATURES["bollinger_std"],
            lag_windows=FEATURES["lag_windows"],
            enable_no_trade_zones=True,
        )
        cache_str, n_samples, n_features, scaler = _build_multipair_dataset(
            args,
            pairs,
            cache_path,
            fe,
        )
        args._n_pairs = len(pairs)
        args._f_per_pair = n_features // len(pairs)
        return cache_str, n_samples, n_features, scaler

    cache_path = _get_cache_path(args)
    Path(args.data_cache).mkdir(parents=True, exist_ok=True)
    _cache_engine = "zarr" if use_zarr and str(cache_path).endswith(".zarr") else "npy"
    _news_mode = str(getattr(args, "historical_news_mode", "calendar") or "calendar").lower()
    print(f"[Data] Cache target: {cache_path}")
    print(f"[Data] Cache engine: {_cache_engine.upper()} | historical_news_mode={_news_mode}")

    if _cache_present(cache_path) and not args.force_rebuild:
        ok, reason = _validate_cache_integrity(str(cache_path), args)
        if not ok:
            if getattr(args, "auto_rebuild_on_mismatch", False):
                print(f"[Data] WARN: cache integrity mismatch ({reason}) — auto rebuilding.")
                _delete_cache_artifacts(str(cache_path))
            elif getattr(args, "integrity_gate", True):
                raise RuntimeError(
                    f"Cache integrity check failed: {reason}. Run with --force-rebuild or --auto-rebuild-on-mismatch."
                )
        print(f"\n[Data] Found cached dataset: {cache_path}")
    if _cache_present(cache_path) and not args.force_rebuild:
        # -- Zarr cache ------------------------------------------------------
        if use_zarr and str(cache_path).endswith(".zarr") and cache_path.is_dir():
            z: Any = _zarr_open_group(str(cache_path), mode="r")
            _zx: Any = z["X"]
            _zy: Any = z["y"]
            n_samples = min(int(_zx.shape[0]), int(_zy.shape[0]))
            n_features = int(_zx.shape[2])
            scaler = _load_scaler_npz(Path(cache_path)) or _identity_scaler(n_features)
            n_samples = _clamp_n_samples_to_disk(str(cache_path), n_samples)
            print(f"[Data] {n_samples:,} samples × {n_features} features (zarr cache)")
            _warn_multitask_cache_sidecars(str(cache_path), args)
            return str(cache_path), n_samples, n_features, scaler
        pass
    print(f"\n[Data] Building 20M tick dataset ΓÇö chunk size: {args.chunk_size:,}")
    _pairs_display = ", ".join(pairs)
    print(f"       Source: {args.data_source} | Pairs: {_pairs_display}")
    print(f"       News mode: {_news_mode} | Cache engine: {_cache_engine.upper()}")
    use_real_cross = args.cross_asset_mode == "real" or (
        args.cross_asset_mode == "auto" and args.data_source != "synthetic"
    )
    cross_asset_source = _resolve_cross_asset_source(args)
    # Treat empty env var the same as "not set" so the default data/processed/cross_asset
    # path is always used when CROSS_ASSET_CACHE_DIR is unset or blank.
    cross_asset_cache_dir = os.getenv("CROSS_ASSET_CACHE_DIR", "").strip() or str(Path(args.data_cache) / "cross_asset")
    cross_asset = None
    if use_real_cross:
        try:
            cross_asset = load_cross_asset_panel(
                start=args.data_start,
                end=args.data_end,
                cache_dir=cross_asset_cache_dir,
                source=cross_asset_source,
            )
            print(
                f"[CrossAsset] Loaded external assets: {len(cross_asset)} "
                f"(source={cross_asset_source}, cache={cross_asset_cache_dir})"
            )
        except Exception as e:
            print(f"[CrossAsset] WARN: external load failed ({e}) ΓÇö falling back to synthetic")
            cross_asset = None

    fe = FeatureEngineer(
        atr_window=FEATURES["atr_window"],
        ofi_window=FEATURES["ofi_window"],
        tar_window=FEATURES["trade_arrival_window"],
        rsi_period=FEATURES["rsi_period"],
        macd_fast=FEATURES["macd_fast"],
        macd_slow=FEATURES["macd_slow"],
        macd_signal=FEATURES["macd_signal"],
        bb_window=FEATURES["bollinger_window"],
        bb_std=FEATURES["bollinger_std"],
        lag_windows=FEATURES["lag_windows"],
        enable_no_trade_zones=True,
    )
    scaler = StandardScaler()
    chunk_n = 0
    total_samples = 0
    n_features_total = 0
    z_store = None  # zarr group (primary)

    # Load COT data
    cot_path = COT_PARQUET_PATH
    cot_data = load_cot(cot_path)

    # Sentiment pipeline (single-pair path)
    sentiment_pipe = None
    if str(getattr(args, "sentiment_mode", "finbert")).lower() != "off":
        try:
            pref = "finbert" if str(args.sentiment_mode).lower() == "finbert" else "vader"
            sentiment_pipe = SentimentPipeline(prefer_backend=pref, use_cache=True)
            print(f"[Sentiment] mode={args.sentiment_mode} enabled")
        except Exception as e:
            print(f"[Sentiment] WARN: init failed ({e}) - disabling sentiment features")
            sentiment_pipe = None

    # Pre-score all headlines across windows so per-chunk calls are cache hits
    if sentiment_pipe is not None and str(getattr(args, "historical_news_mode", "calendar")).lower() == "full":
        try:
            _headlines = collect_headlines_for_range(
                args.data_start,
                args.data_end,
                [str(getattr(args, "pair", "EURUSD"))],
                news_file=getattr(args, "historical_news_file", None),
                calendar_file=getattr(args, "economic_calendar_file", None),
            )
            if _headlines:
                sentiment_pipe.prefetch_headlines(_headlines)
            del _headlines
        except Exception as _pf_err:
            print(f"[Sentiment] prefetch skipped ({_pf_err})")

    if ZARR and not use_zarr:
        print("[Data] Zarr writes disabled on Windows; using NPY memmap cache.")
    if use_zarr:
        # Blosc codec: Linux → lz4@1 (local FS), else zstd@3; overridable via args.
        _compressor = make_training_zarr_compressor(
            getattr(args, "zarr_cname", None),
            getattr(args, "zarr_clevel", None),
        )
        _cname = getattr(_compressor, "cname", "lz4")
        if isinstance(_cname, bytes):
            _cname = _cname.decode()
        print(f"[Data] Zarr compressor=Blosc(cname={_cname}, clevel={getattr(_compressor, 'clevel', 1)})")
    else:
        # Stream raw bytes to disk to avoid 200GB RAM usage and Windows 32-bit NPY save overflows
        _x_fp = open(_x_path(cache_path).replace(".npy", ".bin"), "wb")  # noqa: SIM115
        _y_fp = open(_y_path(cache_path).replace(".npy", ".bin"), "wb")  # noqa: SIM115
        _diff_fp = open(str(cache_path) + "_diff.bin", "wb") if args.label_method == "rl_reward" else None  # noqa: SIM115
        _pq_fp = open(str(cache_path) + "_pq.bin", "wb") if args.label_method == "rl_reward" else None  # noqa: SIM115
        _ycls_fp = open(_y_cls_path(cache_path).replace(".npy", ".bin"), "wb")  # noqa: SIM115
        _close_fp = open(str(cache_path) + "_close.bin", "wb")  # noqa: SIM115
        _atr_fp = open(str(cache_path) + "_atr.bin", "wb")  # noqa: SIM115
        _spread_fp = open(str(cache_path) + "_spread.bin", "wb")  # noqa: SIM115
        _n_samples_written = 0

    # -- Load all ticks or generate in chunks ---------------------------------
    if args.data_source == "synthetic":
        print(f"[Data] Generating {args.n_ticks:,} synthetic ticks in chunks...")
        chunk_specs = [
            (idx, None, None, min(args.chunk_size, max(args.n_ticks - idx * args.chunk_size, 0)))
            for idx in range((args.n_ticks + args.chunk_size - 1) // args.chunk_size)
        ]
    else:
        _base_days = _real_data_window_days(args)
        window_days = _effective_window_days(args)
        date_windows = _iter_date_windows(args.data_start, args.data_end, window_days)
        chunk_specs = [(idx, win_start, win_end, None) for idx, (win_start, win_end) in enumerate(date_windows)]
        _batch_n = max(1, int(getattr(args, "window_batch_days", 1) or 1))
        if _batch_n > 1:
            print(
                f"[Data] Real-data windows: {len(date_windows)} x {window_days} day(s) "
                f"(base {_base_days}d x {_batch_n} batch)"
            )
        else:
            print(f"[Data] Real-data windows: {len(date_windows)} x {window_days} day(s)")

    mgr = None if args.data_source == "synthetic" else ForexDataManager(verbose=True)

    global _FIRST_CHUNK_COLS
    _FIRST_CHUNK_COLS = None  # fresh schema lock for this cache build

    for chunk_n, win_start, win_end, chunk_ticks in chunk_specs:
        t0 = time.time()

        try:
            if args.data_source == "synthetic":
                ticks_chunk = generate_synthetic_tick_data(n_rows=int(chunk_ticks or 100000))
            else:
                load_start = _warmup_load_start(win_start) if win_start else win_start
                print(
                    f"[Data] Loading {args.data_source} for {args.pair} "
                    f"({win_start} -> {win_end}, load from {load_start} with warmup). "
                    "First run downloads many hourly files; this can take tens of minutes to hours."
                )
                if mgr is None:
                    mgr = ForexDataManager(verbose=True)
                ticks_chunk = mgr.load(
                    pair=args.pair,
                    source=args.data_source,
                    start=load_start,
                    end=win_end,
                    session_only=(not getattr(args, "full_day_data", False)),
                )

            X_seq, y_seq, diff_seq, pq_seq, y_cls_seq, close_seq, atr_seq, spread_seq, n_feat, _time_idx = _build_chunk(
                ticks_chunk,
                fe,
                scaler,
                seq_len=_effective_max_seq_len(args),
                chunk_idx=chunk_n,
                win_start=win_start,
                label_method=args.label_method,
                target_col=_cache_target_col(args),
                execution_delay_bars=int(getattr(args, "execution_delay_bars", 1)),
                bar_freq=str(getattr(args, "bar_freq", "5min")),
                lookahead_bars=int(getattr(args, "lookahead_bars", LABELING["lookahead_bars"])),
                profit_target_atr=float(getattr(args, "profit_target_atr", LABELING["profit_target_atr"])),
                stop_loss_atr=float(getattr(args, "stop_loss_atr", LABELING["stop_loss_atr"])),
                cross_asset=cross_asset,
                sentiment_pipe=sentiment_pipe,
                pair=str(getattr(args, "pair", "EURUSD")),
                historical_news_mode=str(getattr(args, "historical_news_mode", "calendar")),
                historical_news_file=getattr(args, "historical_news_file", None),
                economic_calendar_file=getattr(args, "economic_calendar_file", None),
                cot_data=cot_data,
                max_bad_frac=float(getattr(args, "max_bad_frac", 0.05)),
                max_zero_frac=float(getattr(args, "max_zero_frac", 0.80)),
            )
            del ticks_chunk
            gc.collect()

        except Exception as _exc:
            print(f"[Data] chunk {chunk_n} build failed", _exc)
            raise

        if X_seq.size == 0:
            continue

        _maybe_enforce_feature_schema_early(args, cache_path)
        _maybe_run_lookahead_guard(args, X_seq, close_seq)

        n_features_total = n_feat
        n_samples_chunk = len(X_seq)
        total_samples += n_samples_chunk

        if use_zarr:
            _zs: Any = z_store
            if _zs is None:
                if getattr(args, "_resume_zarr", False):
                    z_store = _zs = _zarr_open_group(str(cache_path), mode="a")
                    existing_samples = int(_zs["X"].shape[0]) if "X" in _zs else 0
                    total_samples += existing_samples
                else:
                    if __import__("pathlib").Path(cache_path).exists():
                        for _retry in range(10):
                            __import__("shutil").rmtree(cache_path, ignore_errors=True)
                            if not __import__("pathlib").Path(cache_path).exists():
                                break
                            __import__("time").sleep(0.5)
                    z_store = _zs = _zarr_open_group(str(cache_path), mode="w")
                    # FIX: was (min(64, n_samples_chunk),) - sub-optimal chunks
                    # make ZarrStreamDataset decompress many small blocks per
                    # epoch. Multi-pair path uses 2048 (line 2814); align the
                    # single-pair path to the same value so reader/builder
                    # contract symmetry is preserved. The integrity check at
                    # cache_integrity.py:911 also warns when chunks < 64.
                    c0 = (min(2048, n_samples_chunk), *X_seq.shape[1:])
                    _zarr_create(
                        _zs,
                        "X",
                        shape=(0, *X_seq.shape[1:]),
                        chunks=c0,
                        dtype=ZARR_FEATURE_DTYPE,
                        compressor=_compressor,
                    )
                    _zarr_create(
                        _zs,
                        "y",
                        shape=(0,),
                        chunks=(c0[0],),
                        dtype=ZARR_LABEL_DTYPE,
                        compressor=_compressor,
                    )
                    _zarr_create(
                        _zs,
                        "diff",
                        shape=(0,),
                        chunks=(c0[0],),
                        dtype="uint8",
                        compressor=_compressor,
                    )
                    _zarr_create(
                        _zs,
                        "pq",
                        shape=(0,),
                        chunks=(c0[0],),
                        dtype=ZARR_LABEL_DTYPE,
                        compressor=_compressor,
                    )
                    _zarr_create(
                        _zs,
                        "y_cls",
                        shape=(0,),
                        chunks=(c0[0],),
                        dtype=ZARR_LABEL_DTYPE,
                        compressor=_compressor,
                    )
                    _zarr_create(
                        _zs,
                        "close",
                        shape=(0,),
                        chunks=(c0[0],),
                        dtype=ZARR_LABEL_DTYPE,
                        compressor=_compressor,
                    )
                    _zarr_create(
                        _zs,
                        "atr",
                        shape=(0,),
                        chunks=(c0[0],),
                        dtype=ZARR_LABEL_DTYPE,
                        compressor=_compressor,
                    )
                    _zarr_create(
                        _zs,
                        "spread",
                        shape=(0,),
                        chunks=(c0[0],),
                        dtype=ZARR_LABEL_DTYPE,
                        compressor=_compressor,
                    )

            _zs["X"].append(np.asarray(X_seq, dtype=ZARR_FEATURE_DTYPE))
            _zs["y"].append(y_seq)
            _zs["y_cls"].append(y_cls_seq)
            _zs["close"].append(close_seq)
            if "atr" in _zs:
                _zs["atr"].append(atr_seq)
            if "spread" in _zs:
                _zs["spread"].append(spread_seq)
            # Keep sidecar lengths aligned with X even when a chunk omits them.
            if "diff" in _zs:
                _zs["diff"].append(diff_seq if diff_seq is not None else np.zeros(n_samples_chunk, dtype=np.uint8))
            if "pq" in _zs:
                _zs["pq"].append(pq_seq if pq_seq is not None else np.ones(n_samples_chunk, dtype=np.float32))
        else:
            _x_fp.write(X_seq.tobytes())
            _y_fp.write(y_seq.tobytes())
            _ycls_fp.write(y_cls_seq.tobytes())
            _close_fp.write(close_seq.tobytes())
            _atr_fp.write(atr_seq.tobytes())
            _spread_fp.write(spread_seq.tobytes())
            if args.label_method == "rl_reward":
                if _diff_fp is not None:
                    _diff_fp.write(diff_seq.tobytes())
                if _pq_fp is not None:
                    _pq_fp.write(pq_seq.tobytes())
            _n_samples_written += n_samples_chunk

        elapsed = time.time() - t0
        if args.data_source == "synthetic":
            done = min((chunk_n + 1) * args.chunk_size, args.n_ticks)
            pct = min(done / args.n_ticks * 100, 100)
            print(
                f"  Chunk {chunk_n + 1} | {n_samples_chunk:,} seqs | "
                f"{elapsed:.1f}s | {pct:.0f}% ({total_samples:,} total)"
            )
        else:
            print(
                f"  Window {chunk_n + 1}/{len(chunk_specs)} | {n_samples_chunk:,} seqs | "
                f"{elapsed:.1f}s | {total_samples:,} total"
            )

        # Structured build log
        try:
            from data.dataset_manifest import DatasetManifest

            _manifest_dir = (
                Path(cache_path).parent if str(cache_path).endswith((".zarr", "")) else Path(cache_path).parent  # noqa: RUF034
            )
            _dm = DatasetManifest(str(_manifest_dir))
            _dm.log_build_event(
                "chunk_built",
                pair=str(getattr(args, "pair", "EURUSD")),
                chunk_idx=chunk_n,
                n_rows=n_samples_chunk,
                n_features=n_features_total,
                duration_s=elapsed,
            )
        except Exception:
            pass

        if _TRAIN_LOGGER:
            _TRAIN_LOGGER.heartbeat()
        del X_seq, y_seq, diff_seq, pq_seq, y_cls_seq, close_seq, atr_seq, spread_seq
        gc.collect()

    if total_samples == 0:
        err = (
            "[Data] No usable samples were produced from the selected date range/source. "
            "Likely causes: vendor returned mostly empty hour files, wrong pair/date range, "
            "or blocked data endpoint. Try a shorter recent range first and verify raw cache."
        )
        print(err)
        raise RuntimeError(err)

    # -- Finalise storage ------------------------------------------------------
    if use_zarr and z_store is not None:
        z_store.attrs["total_samples"] = total_samples
        z_store.attrs["n_features"] = n_features_total
        z_store.attrs["seq_len"] = _effective_max_seq_len(args)
        z_store.attrs["strategy_mode"] = str(getattr(args, "strategy_mode", "scalping"))
        z_store.attrs["bar_freq"] = str(getattr(args, "bar_freq", "5min"))
        z_store.attrs["lookahead_bars"] = int(getattr(args, "lookahead_bars", LABELING["lookahead_bars"]))
        meta = {
            "total_samples": int(total_samples),
            "seq_len": int(_effective_max_seq_len(args)),
            "n_features": int(n_features_total),
            "label_method": args.label_method,
            "strategy_mode": str(getattr(args, "strategy_mode", "scalping")),
            "bar_freq": str(getattr(args, "bar_freq", "5min")),
            "lookahead_bars": int(getattr(args, "lookahead_bars", LABELING["lookahead_bars"])),
            "has_rl_market": True,
            "pairs": pairs,
            "target_col": _cache_target_col(args),
            "y_cls_source": "labels.label",
        }
        import json

        with open(str(cache_path) + "_manifest.json", "w") as f:
            json.dump(meta, f)
        _write_feature_schema_json(cache_path, _scaler_feature_names(scaler), args)

        _save_scaler_npz(cache_path, scaler)
    else:
        _x_fp.close()
        _y_fp.close()
        if _diff_fp:
            _diff_fp.close()
        if _pq_fp:
            _pq_fp.close()
        _ycls_fp.close()
        _close_fp.close()
        _atr_fp.close()
        _spread_fp.close()

        final_x_shape = (_n_samples_written, args.seq_len, n_features_total)
        final_y_shape = (_n_samples_written,)
        final_1d = (_n_samples_written,)
        if _n_samples_written > 0:
            x_mmap = np.memmap(
                _x_path(cache_path).replace(".npy", ".bin"),
                dtype="float32",
                mode="r",
                shape=final_x_shape,
            )
            y_mmap = np.memmap(
                _y_path(cache_path).replace(".npy", ".bin"),
                dtype="float32",
                mode="r",
                shape=final_y_shape,
            )
            np.save(_x_path(cache_path), x_mmap)
            np.save(_y_path(cache_path), y_mmap)

            ycls_bin = _y_cls_path(cache_path).replace(".npy", ".bin")
            if Path(ycls_bin).exists():
                ycls_mmap = np.memmap(ycls_bin, dtype="float32", mode="r", shape=final_1d)
                np.save(_y_cls_path(cache_path), ycls_mmap)
            if args.label_method == "rl_reward":
                pq_bin = str(cache_path) + "_pq.bin"
                diff_bin = str(cache_path) + "_diff.bin"
                if Path(pq_bin).exists():
                    pq_mmap = np.memmap(pq_bin, dtype="float32", mode="r", shape=final_1d)
                    np.save(_pq_path(cache_path), pq_mmap)
                if Path(diff_bin).exists():
                    diff_mmap = np.memmap(diff_bin, dtype="uint8", mode="r", shape=final_1d)
                    np.save(_diff_path(cache_path), diff_mmap)
        for _bin_suffix, _save_fn in (
            ("_close.bin", _close_path),
            ("_atr.bin", _atr_path),
            ("_spread.bin", _spread_path),
        ):
            _bpath = str(cache_path) + _bin_suffix
            if Path(_bpath).exists() and _n_samples_written > 0:
                _mmap = np.memmap(_bpath, dtype="float32", mode="r", shape=final_1d)
                np.save(_save_fn(cache_path), _mmap)

        # Save metadata for memmap loading
        meta = {
            "total_samples": _n_samples_written,
            "seq_len": _effective_max_seq_len(args),
            "n_features": n_features_total,
            "label_method": args.label_method,
            "strategy_mode": str(getattr(args, "strategy_mode", "scalping")),
            "bar_freq": str(getattr(args, "bar_freq", "5min")),
            "lookahead_bars": int(getattr(args, "lookahead_bars", LABELING["lookahead_bars"])),
            "has_rl_market": True,
            "pairs": pairs,
            "target_col": _cache_target_col(args),
            "y_cls_source": "labels.label",
        }
        import json

        with open(str(cache_path) + "_manifest.json", "w") as f:
            json.dump(meta, f)
        _write_feature_schema_json(cache_path, _scaler_feature_names(scaler), args)

        _save_scaler_npz(cache_path, scaler)

    readiness_report = _write_pair_readiness_report(
        args,
        cache_path,
        pairs,
        alignment=globals().get("_PAIR_ALIGNMENT_STATS", {}),
    )
    if readiness_report.get("status") == "fail":
        raise RuntimeError(f"Pair Readiness Gate Failed. See {cache_path!s}_pair_readiness_report.json")

    print(f"\n[Data] Dataset built: {total_samples:,} samples ├ù {n_features_total} features ├ù seq_len {args.seq_len}")
    _postprocess_cache_integrity_check(str(cache_path), args, context="Data")
    _verify_dataset(str(cache_path), args, total_samples, n_features_total, context="Data")
    print(f"       Cached at: {cache_path}")

    # ── Write enriched dataset manifest + build log ──────────────────
    try:
        from data.dataset_manifest import DatasetManifest

        _dm = DatasetManifest(str(Path(cache_path).parent))
        _dm.log_build_event("build_complete", n_rows=total_samples, n_features=n_features_total)
        from training.cache_integrity import _compute_content_hash

        _dm.write_manifest(
            source=str(getattr(args, "data_source", "dukascopy")),
            pairs=pairs,
            start=str(getattr(args, "data_start", "")),
            end=str(getattr(args, "data_end", "")),
            freq=str(getattr(args, "bar_freq", "5min")),
            news_mode=bool(str(getattr(args, "historical_news_mode", "calendar")).lower() not in ("none", "false", "0")),
            feature_count=n_features_total,
            label_method=str(getattr(args, "label_method", "rl_reward")),
            seq_len=int(args.seq_len),
            schema_hash=hashlib.sha256(",".join(_scaler_feature_names(scaler)).encode()).hexdigest()[:16] if scaler else "",
            feature_list=list(_scaler_feature_names(scaler)) if scaler else [],
            n_rows_total=total_samples,
            lookahead_bars=int(getattr(args, "lookahead_bars", LABELING.get("lookahead_bars", 30))),
            embargo_bars=int(getattr(args, "embargo_bars", LABELING.get("embargo_bars", 60))),
            purge_bars=int(getattr(args, "purge_bars", 120)),
            content_hash=_compute_content_hash(args),
        )
    except Exception as _m_err:
        print(f"[Manifest] write failed ({_m_err})")

    # ── Future-leak check ────────────────────────────────────────────
    try:
        # FIX: previously passed ``None`` as feature_df. Now we use the same
        # helper to load a sample of last-timestep features from the cache.
        _feats = _leak_check_features_sample(cache_path, max_sample=5000)
        _fwd_ret = None
        if str(cache_path).endswith(".zarr") and Path(cache_path).is_dir():
            import zarr as _zarr

            _z: Any = _zarr.open(str(cache_path), mode="r")
            if "y" in _z:
                _fwd_ret = np.asarray(_z["y"][:], dtype=np.float32)
        if _feats is not None and _fwd_ret is not None and len(_fwd_ret) > 0:
            _X_2d, _feat_names = _feats
            import pandas as _pd

            _feat_df = _pd.DataFrame(_X_2d, columns=_feat_names)
            _leaks = DatasetManifest.check_future_leak(
                _feat_df,
                _fwd_ret.tolist(),
                max_abs_corr=0.30,
            )
            if _leaks:
                print(
                    f"[LeakCheck] {len(_leaks)} feature(s) correlated with "
                    f"forward returns (|r| > 0.30). Review for data leakage."
                )
            else:
                print("[LeakCheck] No features correlated with forward returns.")
        # Label contamination: feature window timestamps < label timestamps.
        _lc = _label_contamination_check(cache_path, args)
        if not _lc.get("ok", True):
            _n_v = int(_lc.get("violations", 0))
            _n_t = int(_lc.get("total_checked", 0))
            print(f"[LabelContamination] FAIL: {_n_v}/{_n_t} samples have feature_ts >= label_ts.")
        elif _lc.get("total_checked", 0) > 0:
            print(f"[LabelContamination] PASS: {_lc['total_checked']:,} samples verified.")
    except Exception as _le_err:
        print(f"[LeakCheck] skipped ({_le_err})")

    # ── Lockbox reservation ──────────────────────────────────────────
    try:
        _lookback_days = int(getattr(args, "real_data_window_days", 7) or 7)
        _lockbox_end = str(getattr(args, "data_end", datetime.now(UTC).strftime("%Y-%m-%d")))
        _lockbox_start = (datetime.fromisoformat(_lockbox_end) - timedelta(days=_lookback_days)).strftime("%Y-%m-%d")
        DatasetManifest.reserve_lockbox(str(Path(cache_path).parent), _lockbox_start, _lockbox_end)
    except Exception:
        pass

    # ── Data Quality Report ──────────────────────────────────────────
    try:
        from zarr import open as zarr_open

        from data.data_quality_report import DataQualityReporter

        reporter = DataQualityReporter(str(Path(cache_path).parent / "quality"))
        z_store: Any = zarr_open(cache_path, mode="r")
        _n_samp = min(10000, total_samples)
        X_sample = np.array(z_store["X"][:_n_samp], dtype=np.float32)
        y_sample = np.array(z_store["y"][:_n_samp], dtype=np.float32)
        y_cls = (
            np.array(z_store["y_cls"][:_n_samp], dtype=np.float32) if "y_cls" in z_store else None
        )

        # Compute basic quality stats from cache
        feat_nan_rates = {str(i): float(np.isnan(X_sample).mean(axis=0)[i]) for i in range(X_sample.shape[2])}
        class_balance = {}
        if y_cls is not None:
            labels = y_cls[y_cls != 0]
            if len(labels) > 0:
                class_balance["long"] = float(np.mean(labels == 1))
                class_balance["short"] = float(np.mean(labels == -1))
                class_balance["hold"] = float(np.mean(y_cls == 0))

        reporter.generate_report(
            missing_bars={"total": 0},
            zero_volume_periods={"total": 0},
            spread_outliers={"total": 0},
            feature_nan_rates={str(k): v for k, v in enumerate(feat_nan_rates.values()) if v > 0},
            label_class_balance=class_balance,
            reward_dist={"mean": float(y_sample.mean()), "std": float(y_sample.std())},
            per_regime_counts={},
        )
    except Exception as _dq_e:
        print(f"[DataQuality] Report generation skipped: {_dq_e}")

    return str(cache_path), total_samples, n_features_total, scaler


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-TIMEFRAME TENSOR BUILDER
# Provides the missing data-pipeline link for models.ensemble.MultiTimeframeAttention.
# ─────────────────────────────────────────────────────────────────────────────


def build_multitf_tensors(
    X_seq: np.ndarray,
    base_tf_minutes: int = 1,
    target_tfs: list[int] | None = None,
) -> list[np.ndarray]:
    """Downsample a 1-min ``X_seq`` into coarser timeframe views.

    Produces a list of arrays that can be passed directly to
    ``MultiTimeframeAttention.forward(x_list)``.  The algorithm is
    **lookahead-free**: each coarser bar uses only the last 1-min bar within
    its completed window, so a 5-min bar at index *i* corresponds to 1-min
    bars ``[i-4, i]`` -- all of which have already closed by the time the
    sequence ends.

    Parameters
    ----------
    X_seq : np.ndarray
        Shape ``(N, T, F)`` -- the standard sliding-window feature tensor
        produced by ``dataset_builder``.  Each sample ``X_seq[n]`` is a
        sequence of ``T`` 1-min bars ending at time ``t_n``.
    base_tf_minutes : int
        The bar frequency of ``X_seq`` in minutes (default 1).
    target_tfs : list[int]
        Target timeframes in minutes.  Defaults to ``[1, 5, 15]``.
        Values must be multiples of ``base_tf_minutes``.

    Returns
    -------
    list[np.ndarray]
        One array per target timeframe, ordered as ``target_tfs``.
        - ``[0]`` shape ``(N, T,       F)``  -- unchanged 1-min view
        - ``[1]`` shape ``(N, T//5,    F)``  -- 5-min view (last bar of each 5)
        - ``[2]`` shape ``(N, T//15,   F)``  -- 15-min view
    """
    if target_tfs is None:
        target_tfs = [base_tf_minutes, 5, 15]

    _N, T, _F = X_seq.shape
    result = []

    for tf in target_tfs:
        stride = max(1, tf // base_tf_minutes)
        if stride == 1:
            result.append(X_seq.astype(np.float32, copy=False))
            continue

        # Take every stride-th bar starting from the last bar (index T-1)
        # working backwards, then reverse so time is ascending.
        # This picks the LAST bar of each completed coarser window -- no lookahead.
        coarse_indices = list(range(T - 1, -1, -stride))[::-1]
        if not coarse_indices:
            coarse_indices = [T - 1]

        coarse = X_seq[:, coarse_indices, :]  # (N, n_coarse, F)
        result.append(np.ascontiguousarray(coarse, dtype=np.float32))

    return result


def build_multitf_dataset(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    base_tf_minutes: int = 1,
    target_tfs: list[int] | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Convenience wrapper: returns ``(tf_views, labels)`` ready for training.

    The first element of ``tf_views`` is the full-resolution sequence.
    Subsequent elements are downsampled coarser-timeframe views.

    Usage with MultiTimeframeAttention::

        tf_views, y = build_multitf_dataset(X_seq, y_seq)
        import torch
        x_list = [torch.from_numpy(v) for v in tf_views]
        model = MultiTimeframeAttention(input_size=F)
        pred = model(x_list)
    """
    if target_tfs is None:
        target_tfs = [base_tf_minutes, 5, 15]
    views = build_multitf_tensors(X_seq, base_tf_minutes=base_tf_minutes, target_tfs=target_tfs)
    return views, y_seq
