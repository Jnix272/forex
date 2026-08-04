"""Cache path helpers, integrity checks, and dataset verification.\n\nSee docs/CONTINUE.md."""
from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from config.settings import FEATURES, LABELING
from training.gpu_cache_io import (
    ZARR,
    _atr_path,
    _close_path,
    _diff_path,
    _pq_path,
    _spread_path,
    _x_path,
    _y_cls_path,
    _y_path,
    _zarr_open_group,
)

_HOST = None
_BOUND = False
_HOST_DEPS = (
    '_log_error',
    '_log_warn',
    '_log_info',
    '_effective_max_seq_len',
    'LABELING',
    'FEATURES',
    'PATHS',
    'ZARR',
    'sanitize_array',
    'DatasetManifest',
)


def bind_host(host_mod) -> None:
    global _HOST, _BOUND
    _HOST = host_mod
    g = globals()
    for name in _HOST_DEPS:
        if hasattr(host_mod, name):
            g[name] = getattr(host_mod, name)
    _BOUND = True


def _ensure_bound() -> None:
    import training.train_gpu as tg
    bind_host(tg)

# -----------------------------------------------------------------------------
# PHASE 1 ΓÇö CHUNKED DATA PIPELINE
# -----------------------------------------------------------------------------

def _get_pairs(args) -> list[str]:
    """Return the list of pairs to train on, from --pairs or --pair."""
    _ensure_bound()
    raw = getattr(args, "pairs", None)
    if not raw:
        return [args.pair.upper()]
    if isinstance(raw, list):
        return [p.strip().upper() for p in raw if p and p.strip()]
    return [p.strip().upper() for p in str(raw).split(",") if p.strip()]


def _real_data_window_days(args) -> int:
    """
    Choose a conservative date-window size for real-data ingestion.

    We derive this from ``chunk_size`` so the real-data path respects the same
    RAM safety valve as the synthetic chunked builder.
    """
    explicit = int(getattr(args, "real_data_window_days", 0) or 0)
    if explicit > 0:
        return explicit

    session_hours = 24 if getattr(args, "full_day_data", False) else 11
    # Conservative FX tick density estimate to keep per-window RAM bounded.
    est_ticks_per_hour = 10_000
    est_ticks_per_day = max(session_hours * est_ticks_per_hour, 1)
    chunk_size = max(int(getattr(args, "chunk_size", 500_000) or 500_000), 1)
    days = max(1, chunk_size // est_ticks_per_day)
    return min(max(int(days), 1), 31)


def _effective_window_days(args) -> int:
    """Return per-window day count after applying the batch multiplier.

    effective = real_data_window_days * window_batch_days

    ``window_batch_days`` groups consecutive base windows into a single
    processing batch so feature engineering sees more context at window
    boundaries.  Default is 1 (no batching, backward compatible).
    """
    base = _real_data_window_days(args)
    batch = max(1, int(getattr(args, "window_batch_days", 1) or 1))
    return base * batch


def _iter_date_windows(start: str, end: str, window_days: int) -> list[tuple[str, str]]:
    """Split an inclusive YYYY-MM-DD range into inclusive date windows."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end, "%Y-%m-%d")
    if end_dt < start_dt:
        raise ValueError(f"data_end ({end}) is earlier than data_start ({start})")

    windows: list[tuple[str, str]] = []
    current = start_dt
    step = max(int(window_days), 1)
    while current <= end_dt:
        win_end = min(current + timedelta(days=step - 1), end_dt)
        windows.append((current.strftime("%Y-%m-%d"), win_end.strftime("%Y-%m-%d")))
        current = win_end + timedelta(days=1)
    return windows


def _resolve_cross_asset_source(args) -> str:
    """Resolve cross-asset provider with env override matching downloader behavior."""
    return str(
        os.getenv("CROSS_ASSET_SOURCE", "").strip()
        or getattr(args, "cross_asset_provider", "auto")
        or "auto"
    ).strip().lower()


def _cache_target_col(args) -> str:

    """Keep cache y as reward/PnL; direction labels live in y_cls sidecar.



    Classification models train from y_cls, but validation Sharpe and promotion

    gates need continuous reward/PnL in y. Do not switch cache y to the class

    label just because the supervised loss is cross_entropy.

    """

    return "reward"





def _get_cache_path(args) -> Path:
    _ensure_bound()
    pairs    = _get_pairs(args)
    pair_tag = "-".join(sorted(pairs))
    target_col = _cache_target_col(args)

    exec_delay = int(getattr(args, "execution_delay_bars", 1))
    strategy = str(getattr(args, "strategy_mode", "scalping") or "scalping").lower()
    bar_freq = str(getattr(args, "bar_freq", "1min") or "1min").lower()
    lookahead = int(getattr(args, "lookahead_bars", LABELING.get("lookahead_bars", 15)))
    tp_atr = float(getattr(args, "profit_target_atr", LABELING.get("profit_target_atr", 1.5)))
    sl_atr = float(getattr(args, "stop_loss_atr", LABELING.get("stop_loss_atr", 0.8)))
    news_mode = str(getattr(args, "historical_news_mode", "calendar") or "calendar").lower()
    news_tag = f"news-{news_mode}"
    ca_mode = str(getattr(args, "cross_asset_mode", "auto") or "auto").lower()
    ca_source = _resolve_cross_asset_source(args)
    ca_tag = f"ca-{ca_mode}-{ca_source}"
    # label_exit_mode / warmup_days bust caches after DS-001 / DS-002 fixes.
    exit_mode = str(getattr(args, "label_exit_mode", "bid_ask") or "bid_ask").lower()
    warmup_days = int(getattr(args, "feature_warmup_days", 14) or 14)
    tag      = (
        f"{strategy}_{bar_freq}_{pair_tag}_{args.n_ticks}_{args.data_source}_{args.seq_len}_"
        f"{args.label_method}_{target_col}_lh{lookahead}_tp{tp_atr:g}_sl{sl_atr:g}_"
        f"exec{exec_delay}_lexit-{exit_mode}_wu{warmup_days}_{news_tag}_{ca_tag}"
    )
    if getattr(args, "data_start", None) and getattr(args, "data_end", None):
        tag += f"_{args.data_start}_{args.data_end}"
    use_zarr_cache = bool(ZARR)
    ext = ".zarr" if use_zarr_cache else ""
    return Path(args.data_cache) / f"dataset_{tag}{ext}"




_RL_MARKET_ZARR_KEYS = ("close", "atr", "spread")


def _cache_has_rl_market_arrays(cache_path: str) -> bool:
    """True when close/atr/spread exist with the same row count as X."""
    p = Path(cache_path)
    n_x = _on_disk_sequence_count(cache_path)
    if n_x is None:
        return False
    if ZARR and p.is_dir() and (p / ".zgroup").exists():
        try:
            z = _zarr_open_group(cache_path, mode="r")
            if not all(k in z for k in _RL_MARKET_ZARR_KEYS):
                return False
            return all(int(z[k].shape[0]) == int(n_x) for k in _RL_MARKET_ZARR_KEYS)
        except Exception:
            return False
    for fn in (_close_path, _atr_path, _spread_path):
        fp = Path(fn(cache_path))
        if not fp.exists():
            return False
        try:
            if int(np.load(str(fp), mmap_mode="r").shape[0]) != int(n_x):
                return False
        except Exception:
            return False
    return True


def _require_rl_market_cache(cache_path: str) -> None:
    if _cache_has_rl_market_arrays(cache_path):
        return
    raise RuntimeError(
        "[RL] Cache missing real market arrays (close, atr, spread). "
        "Rebuild with: .\\.venv-gpu\\Scripts\\python.exe scripts\\train.py --rebuild-cache "
        "or training\\train_gpu.py --force-rebuild"
    )


def _load_rl_market_from_cache(cache_path: str, start: int, n_env: int
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load per-sequence close/ATR/spread aligned with X/y indices."""
    end = start + n_env
    if ZARR and str(cache_path).endswith(".zarr") and Path(cache_path).is_dir():
        z = _zarr_open_group(cache_path, mode="r")
        prices = np.asarray(z["close"][start:end], dtype=np.float32)
        atr = np.asarray(z["atr"][start:end], dtype=np.float32)
        spreads = np.asarray(z["spread"][start:end], dtype=np.float32)
    else:
        prices = np.asarray(np.load(_close_path(cache_path), mmap_mode="r")[start:end],
                            dtype=np.float32)
        atr = np.asarray(np.load(_atr_path(cache_path), mmap_mode="r")[start:end],
                         dtype=np.float32)
        spreads = np.asarray(np.load(_spread_path(cache_path), mmap_mode="r")[start:end],
                             dtype=np.float32)
    return prices, atr, spreads


def _market_bar_arrays_from_feats(
    feats,
    x_index,
    fe: FeatureEngineer,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bar-level close/ATR/spread aligned with feature rows (before sequence filter)."""
    close_col = "mid_close" if "mid_close" in feats.columns else "close"
    if close_col not in feats.columns:
        raise ValueError(
            f"[Data] RL market cache requires '{close_col}' or 'close' in features"
        )
    atr_w = int(getattr(fe, "atr_w", FEATURES.get("atr_window", 6)))
    atr_col = f"atr_{atr_w}"
    if atr_col not in feats.columns:
        atr_col = next(
            (c for c in (f"atr_{atr_w}", "atr_6", "atr_20", "atr") if c in feats.columns),
            None,
        )
    if atr_col is None:
        raise ValueError("[Data] RL market cache requires an ATR column (e.g. atr_6)")
    pip = float(LABELING.get("pip_size", 0.0001))
    close_bars = feats[close_col].reindex(x_index).astype(np.float64).values
    atr_bars = feats[atr_col].reindex(x_index).astype(np.float64).values
    if "spread_pips" in feats.columns:
        spread_bars = feats["spread_pips"].reindex(x_index).astype(np.float64).values * pip
    elif "spread_avg" in feats.columns:
        spread_bars = feats["spread_avg"].reindex(x_index).astype(np.float64).values
    else:
        print(
            "[Data] WARN: no spread_pips/spread_avg in feature frame — "
            "synthesizing default spread (0.5 pip) for RL market cache. "
            "Enable spread features in FEATURE_MASK / FeatureEngineer build."
        )
        spread_bars = np.full(len(x_index), 0.5 * pip, dtype=np.float64)
    close_seq = np.asarray(close_bars[seq_len - 1:], dtype=np.float32)
    atr_seq = np.asarray(np.maximum(atr_bars[seq_len - 1:], pip), dtype=np.float32)
    spread_seq = np.asarray(np.maximum(spread_bars[seq_len - 1:], pip * 0.1), dtype=np.float32)
    return close_seq, atr_seq, spread_seq


def _resolve_pair_feat_indices(feat_names: list | None, f_per_pair: int) -> tuple[int, int]:
    """Return (return_idx, atr_idx) within each pair's feature slice."""
    names = [str(c).split("::")[-1] for c in list(feat_names or [])[:f_per_pair]]

    if not names:
        return 0, min(1, f_per_pair - 1)
    ret_candidates = [f"ret_{w}" for w in (5, 20, 60, 30, 120)] + ["return", "ret"]

    atr_w = int(FEATURES.get("atr_window", 6))
    atr_candidates = [f"atr_{atr_w}", "atr_6", "atr_20", "atr"]
    ri = next((names.index(c) for c in ret_candidates if c in names), 0)
    ai = next((names.index(c) for c in atr_candidates if c in names), min(1, f_per_pair - 1))
    return ri, ai


def _promotion_holdout_n(n_samples: int, args) -> int:
    """Bars reserved for promotion gate — never used in walk-forward CV."""
    frac = min(max(float(getattr(args, "promote_forward_frac", 0.1)), 0.01), 0.5)
    # Tiny/quick/synthetic caches cannot spare a hard floor of 50 bars.
    floor = 50
    if bool(getattr(args, "quick_mode", False)) or int(n_samples) < 200:
        floor = max(1, min(50, max(1, int(n_samples) // 5)))
    return min(200_000, max(floor, int(n_samples * frac)))


def _trainable_max_index(n_total: int, args) -> int:
    """Last exclusive index usable for pretrain/RL (excludes holdout + embargo)."""
    n_total = max(0, int(n_total))
    return max(0, n_total - _promotion_holdout_n(n_total, args) - _embargo_bars(args))












def _on_disk_sequence_count(cache_path: str) -> int | None:
    """
    Rows actually readable by MemmapSequenceDataset.
    Priority: Zarr directory store > NPY memmap sidecars.
    """
    # 1. Zarr directory store
    if ZARR and str(cache_path).endswith(".zarr") and Path(cache_path).is_dir():
        try:
            z = _zarr_open_group(cache_path, mode="r")
            return int(min(z["X"].shape[0], z["y"].shape[0]))
        except Exception:
            return None
    # 2. NPY memory-map sidecars
    px, py = Path(_x_path(cache_path)), Path(_y_path(cache_path))
    if px.exists() and py.exists():
        X = np.load(str(px), mmap_mode="r")
        y = np.load(str(py), mmap_mode="r")
        return int(min(X.shape[0], y.shape[0]))
    return None


def _clamp_n_samples_to_disk(cache_path: str, n_samples: int) -> int:
    """Clamp n_samples to actual on-disk row count to prevent OOB in DataLoader workers."""
    n_disk = _on_disk_sequence_count(cache_path)
    if n_disk is None or n_disk >= n_samples:
        return n_samples
    print(f"[Data] WARN: on-disk arrays have {n_disk:,} rows but pipeline reported "
          f"{n_samples:,} ΓÇö clamping to {n_disk:,} (check X/Y export parity)")
    return n_disk


def _cache_length_snapshot(cache_path: str) -> dict:
    """
    Return cache lengths for integrity checks.
    Keys may include: zarr_X, zarr_y, npy_X, npy_y.
    """
    out: dict = {}
    p = Path(cache_path)
    # Zarr directory store
    if ZARR and p.is_dir() and (p / ".zgroup").exists():
        try:
            z = _zarr_open_group(cache_path, mode="r")
            if "X" in z: out["zarr_X"] = int(z["X"].shape[0])
            if "y" in z: out["zarr_y"] = int(z["y"].shape[0])
            if "y_cls" in z: out["zarr_y_cls"] = int(z["y_cls"].shape[0])
            if "pq" in z: out["zarr_pq"] = int(z["pq"].shape[0])
            if "diff" in z: out["zarr_diff"] = int(z["diff"].shape[0])
            for mk in _RL_MARKET_ZARR_KEYS:
                if mk in z:
                    out[f"zarr_{mk}"] = int(z[mk].shape[0])
        except Exception:
            out["zarr_unreadable"] = 1
    px, py = Path(_x_path(cache_path)), Path(_y_path(cache_path))
    try:
        import numpy.lib.format as np_fmt
        if px.exists():
            with open(px, "rb") as f:
                version = np_fmt.read_magic(f)
                shape, fortran, dtype = np_fmt._read_array_header(f, version)
            out["npy_X"] = int(shape[0])
        if py.exists():
            with open(py, "rb") as f:
                version = np_fmt.read_magic(f)
                shape, fortran, dtype = np_fmt._read_array_header(f, version)
            out["npy_y"] = int(shape[0])
        for key, path in (
            ("npy_y_cls", _y_cls_path(cache_path)),
            ("npy_pq", _pq_path(cache_path)),
            ("npy_diff", _diff_path(cache_path)),
            ("npy_close", _close_path(cache_path)),
            ("npy_atr", _atr_path(cache_path)),
            ("npy_spread", _spread_path(cache_path)),
        ):
            pp = Path(path)
            if pp.exists():
                with open(pp, "rb") as f:
                    version = np_fmt.read_magic(f)
                    shape, fortran, dtype = np_fmt._read_array_header(f, version)
                out[key] = int(shape[0])
    except Exception as e:
        print(f"[Cache] Direct NPY header read failed: {e}. Falling back to mmap.")
        if px.exists():
            out["npy_X"] = int(np.load(str(px), mmap_mode="r").shape[0])
        if py.exists():
            out["npy_y"] = int(np.load(str(py), mmap_mode="r").shape[0])
        for key, path in (
            ("npy_y_cls", _y_cls_path(cache_path)),
            ("npy_pq", _pq_path(cache_path)),
            ("npy_diff", _diff_path(cache_path)),
            ("npy_close", _close_path(cache_path)),
            ("npy_atr", _atr_path(cache_path)),
            ("npy_spread", _spread_path(cache_path)),
        ):
            pp = Path(path)
            if pp.exists():
                out[key] = int(np.load(str(pp), mmap_mode="r").shape[0])
    return out


def _validate_cache_integrity(cache_path: str, args=None) -> tuple[bool, str]:
    _ensure_bound()
    snap = _cache_length_snapshot(cache_path)
    problems = []

    # Manifest validation
    if args and not getattr(args, "ignore_manifest", False):
        manifest_path = Path(cache_path).with_name(Path(cache_path).name + "_manifest.json")
        manifest: dict = {}
        if not manifest_path.exists():
            # Support legacy _meta.json as a fallback
            legacy_meta = Path(cache_path).with_name(Path(cache_path).name + "_meta.json")
            if legacy_meta.exists():
                manifest_path = legacy_meta
            else:
                # Zarr stores may only have attrs (no sidecar) ΓÇö synthesize a manifest
                p_cache = Path(cache_path)
                if ZARR and p_cache.is_dir() and (p_cache / ".zgroup").exists():
                    try:
                        import zarr
                        z_store = zarr.open(str(p_cache), mode="r")
                        manifest = dict(getattr(z_store, "attrs", {}) or {})
                    except Exception:
                        manifest = {}
                if not manifest:
                    problems.append("dataset_manifest.json missing")

        if manifest_path.exists() and not manifest:
            import json
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
            except Exception as e:
                problems.append(f"Manifest read error: {e}")

        if manifest:
            try:
                expected_pairs = _get_pairs(args)
                manifest_pairs = manifest.get("pairs")
                if len(expected_pairs) > 1 and not manifest_pairs:

                    problems.append("Manifest missing pairs for multi-pair cache")

                if manifest_pairs:
                    if isinstance(manifest_pairs, str):
                        manifest_pairs = [p.strip().upper() for p in manifest_pairs.split(",") if p.strip()]
                    if manifest_pairs != expected_pairs:
                        problems.append(f"Manifest mismatch: pairs {manifest_pairs} != requested {expected_pairs}")
                    if len(expected_pairs) > 1:

                        schema_path = Path(str(cache_path) + "_feature_schema.json")

                        if not schema_path.exists():

                            problems.append("Multi-pair feature schema missing")

                        else:

                            try:

                                schema = json.loads(schema_path.read_text(encoding="utf-8"))

                                expected_n = int(manifest.get("n_features", 0) or 0)

                                if not isinstance(schema, list) or len(schema) != expected_n:

                                    problems.append(

                                        f"Multi-pair feature schema length {len(schema) if isinstance(schema, list) else 'invalid'} != n_features {expected_n}"

                                    )

                            except Exception as e:

                                problems.append(f"Multi-pair feature schema unreadable: {e}")

                if manifest.get("seq_len"):
                    requested_seq = _effective_max_seq_len(args)
                    if int(manifest.get("seq_len")) != int(requested_seq):
                        problems.append(
                            f"Manifest mismatch: seq_len {manifest.get('seq_len')} != "
                            f"required max {requested_seq} (training.seq_len / curriculum target)"
                        )
                if (

                    getattr(args, "label_method", "") == "rl_reward"

                    and manifest.get("y_cls_source") != "labels.label"

                ):

                    problems.append(

                        "Manifest y_cls_source is stale/missing; rebuild required so y_cls uses true direction labels"

                    )

                # We can also check bar_freq, strategy_mode, etc
            except Exception as e:
                problems.append(f"Manifest validation error: {e}")

    p = Path(cache_path)
    x_len = snap.get("zarr_X", snap.get("npy_X"))
    if args and getattr(args, "label_method", "") == "rl_reward" and x_len is not None:
        has_y_cls = "zarr_y_cls" in snap or "npy_y_cls" in snap
        if not has_y_cls:
            problems.append("RL reward cache missing y_cls direction sidecar")

    # Zarr checks
    if ZARR and p.is_dir() and (p / ".zgroup").exists():
        if snap.get("zarr_unreadable", 0) == 1:
            problems.append("Zarr store unreadable/corrupt")
        elif "zarr_X" not in snap or "zarr_y" not in snap:
            problems.append("Zarr store missing required arrays: X and/or y")
        if "zarr_y_cls" in snap and snap["zarr_y_cls"] != snap.get("zarr_X"):
            problems.append(
                f"Zarr y_cls={snap['zarr_y_cls']:,} != X={snap.get('zarr_X', 0):,}"
            )
        if "zarr_pq" in snap and snap["zarr_pq"] != snap.get("zarr_X"):
            problems.append(
                f"Zarr pq={snap['zarr_pq']:,} != X={snap.get('zarr_X', 0):,}"
            )
        if "zarr_diff" in snap and snap["zarr_diff"] != snap.get("zarr_X"):
            problems.append(
                f"Zarr diff={snap['zarr_diff']:,} != X={snap.get('zarr_X', 0):,}"
            )
        for mk in _RL_MARKET_ZARR_KEYS:
            zk = f"zarr_{mk}"
            if zk in snap and snap[zk] != snap.get("zarr_X"):
                problems.append(f"Zarr {mk}={snap[zk]:,} != X={snap.get('zarr_X', 0):,}")
    if "zarr_X" in snap and "zarr_y" in snap and snap["zarr_X"] != snap["zarr_y"]:
        problems.append(f"Zarr X={snap['zarr_X']:,} != y={snap['zarr_y']:,}")
    if "npy_X" in snap and "npy_y" in snap and snap["npy_X"] != snap["npy_y"]:
        problems.append(f"NPY X={snap['npy_X']:,} != y={snap['npy_y']:,}")
    for key in ("npy_y_cls", "npy_pq", "npy_diff", "npy_close", "npy_atr", "npy_spread"):
        if key in snap and "npy_X" in snap and snap[key] != snap["npy_X"]:
            problems.append(f"{key}={snap[key]:,} != NPY X={snap['npy_X']:,}")
    if not problems:
        return True, ""
    return False, " | ".join(problems)


def _verify_dataset(
    cache_path: str,
    args,
    n_samples: int,
    n_features: int,
    context: str = "Data",
) -> dict:
    """Comprehensive post-build verification of features, labels, and alignment.

    Returns a report dict with per-feature stats, label distribution,
    alignment checks, and anomaly flags.  All results are appended to
    build_log.jsonl and printed to stdout.
    """
    _ensure_bound()
    from data.dataset_manifest import DatasetManifest

    report = {
        "context": context,
        "n_samples": n_samples,
        "n_features": n_features,
        "features": {},
        "labels": {},
        "alignment": {},
        "anomalies": [],
        "timestamp": datetime.now(UTC).isoformat(),
    }

    try:
        cache_p = Path(cache_path)
        is_zarr = cache_p.is_dir() and (cache_p / ".zgroup").exists()

        # ── Load a sample for verification ──────────────────────
        sample_n = min(n_samples, 5000) if n_samples > 0 else 0
        if sample_n == 0:
            report["anomalies"].append("zero_samples")
            return report

        if is_zarr:
            import zarr as _zarr
            z = _zarr.open(str(cache_path), mode="r")
            X_sample = z["X"][:sample_n]
            y_sample = z["y"][:sample_n]
            y_cls_sample = z.get("y_cls", None)
            if y_cls_sample is not None:
                y_cls_sample = y_cls_sample[:sample_n]
            time_idx_sample = None
            try:
                time_idx_sample = z["X"].attrs.get("time_idx")
            except Exception:
                pass
        else:
            x_path = Path(_x_path(cache_path))
            y_path = Path(_y_path(cache_path))
            X_sample = np.load(x_path, mmap_mode="r")[:sample_n]
            y_sample = np.load(y_path, mmap_mode="r")[:sample_n]
            y_cls_sample = None
            yc_path = Path(_y_cls_path(cache_path))
            if yc_path.exists():
                try:
                    y_cls_sample = np.load(str(yc_path), mmap_mode="r")[:sample_n]
                except Exception:
                    pass
            time_idx_sample = None

        # ── Per-feature statistics ──────────────────────────────
        if X_sample.ndim == 3:
            for feat_idx in range(min(X_sample.shape[2], 20)):
                col_data = X_sample[:, :, feat_idx].flatten()
                finite = col_data[np.isfinite(col_data)]
                nan_rate = 1.0 - (len(finite) / max(len(col_data), 1))
                feat_name = f"feature_{feat_idx}"
                try:
                    if hasattr(args, "_feat_names") and feat_idx < len(args._feat_names):
                        feat_name = args._feat_names[feat_idx]
                except Exception:
                    pass
                feat_report = {
                    "min": float(np.min(finite)) if len(finite) > 0 else None,
                    "max": float(np.max(finite)) if len(finite) > 0 else None,
                    "mean": float(np.mean(finite)) if len(finite) > 0 else None,
                    "std": float(np.std(finite)) if len(finite) > 0 else None,
                    "nan_rate": round(nan_rate, 6),
                    "n_finite": len(finite),
                }
                report["features"][feat_name] = feat_report
                if nan_rate > 0.05:
                    report["anomalies"].append(f"high_nan_{feat_name}:{nan_rate:.4f}")
                if len(finite) > 0 and np.std(finite) < 1e-12:
                    report["anomalies"].append(f"zero_variance_{feat_name}")

        # ── Label statistics ────────────────────────────────────
        y_finite = y_sample[np.isfinite(y_sample)]
        if len(y_finite) > 0:
            report["labels"] = {
                "mean_reward": round(float(np.mean(y_finite)), 6),
                "std_reward": round(float(np.std(y_finite)), 6),
                "min_reward": round(float(np.min(y_finite)), 6),
                "max_reward": round(float(np.max(y_finite)), 6),
                "n_finite": len(y_finite),
                "nan_rate": round(1.0 - len(y_finite) / len(y_sample), 6),
            }
            # Direction distribution if y_cls available
            if y_cls_sample is not None:
                y_cls_finite = y_cls_sample[np.isfinite(y_cls_sample)]
                if len(y_cls_finite) > 0:
                    unique, counts = np.unique(y_cls_finite, return_counts=True)
                    dist = {int(v): int(c) for v, c in zip(unique, counts)}
                    report["labels"]["direction_dist"] = dist
                    total = sum(dist.values())
                    for side, cnt in dist.items():
                        pct = cnt / total * 100
                        if pct < 5 and side != 0:
                            report["anomalies"].append(
                                f"rare_direction_{side}: {pct:.1f}%"
                            )
        else:
            report["anomalies"].append("all_labels_nan")

        # ── Alignment check ─────────────────────────────────────
        report["alignment"] = {
            "n_samples": n_samples,
            "n_features": n_features,
            "X_shape": list(X_sample.shape),
            "y_shape": list(y_sample.shape),
            "y_cls_available": y_cls_sample is not None,
        }

        # ── Time index monotonicity (if available) ──────────────
        if time_idx_sample is not None:
            try:
                tidx = np.asarray(time_idx_sample[:sample_n], dtype=np.int64)
                if len(tidx) > 1:
                    monotonic = bool(np.all(np.diff(tidx) >= 0))
                    report["alignment"]["time_index_monotonic"] = monotonic
                    if not monotonic:
                        report["anomalies"].append("time_index_not_monotonic")
            except Exception:
                pass

    except Exception as e:
        report["anomalies"].append(f"verification_error: {e}")

    # ── Log results ──────────────────────────────────────────
    try:
        dm = DatasetManifest(str(Path(cache_path).parent))
        dm.log_build_event(
            "verification_complete",
            n_rows=n_samples,
            n_features=n_features,
            extra={
                "anomaly_count": len(report["anomalies"]),
                "anomalies": report["anomalies"],
                "label_mean": report["labels"].get("mean_reward"),
                "label_std": report["labels"].get("std_reward"),
                "direction_dist": report["labels"].get("direction_dist"),
            },
        )
    except Exception:
        pass

    # ── Print summary ────────────────────────────────────────
    anomaly_str = ""
    if report["anomalies"]:
        anomaly_str = f" ⚠ ANOMALIES: {', '.join(report['anomalies'][:10])}"
        if len(report["anomalies"]) > 10:
            anomaly_str += f" (+{len(report['anomalies']) - 10} more)"

    feat_nan = [f for f, s in report["features"].items() if s.get("nan_rate", 0) > 0.05]
    if feat_nan:
        anomaly_str += f" | high-NaN features: {len(feat_nan)}"

    print(
        f"[{context}] Verify: {n_samples:,} samples x {n_features} features"
        f" | reward μ={report['labels'].get('mean_reward', 'N/A')}"
        f" σ={report['labels'].get('std_reward', 'N/A')}"
        f"{anomaly_str}"
    )

    return report


def _postprocess_cache_integrity_check(cache_path: str, args, *, context: str = "Data") -> None:
    """Fail immediately if a freshly processed cache is incomplete or inconsistent."""
    _ensure_bound()
    ok, reason = _validate_cache_integrity(cache_path, args)
    if not ok:
        raise RuntimeError(
            f"[{context}] Post-processing cache integrity failed: {reason}. "
            "Delete/rebuild the processed cache before training."
        )
    snap = _cache_length_snapshot(cache_path)
    n_rows = snap.get("zarr_X", snap.get("npy_X", 0))
    print(f"[{context}] Post-processing cache integrity PASS ({int(n_rows):,} rows)")


def _cache_has_multitask_sidecars(cache_path: str) -> bool:
    """True when y_cls sidecar exists with the same row count as X."""
    p = Path(cache_path)
    n_x = _on_disk_sequence_count(cache_path)
    if n_x is None:
        return False
    if ZARR and p.is_dir() and (p / ".zgroup").exists():
        try:
            z = _zarr_open_group(cache_path, mode="r")
            if "y_cls" not in z:
                return False
            return int(z["y_cls"].shape[0]) == int(n_x)
        except Exception:
            return False
    yp = Path(_y_cls_path(cache_path))
    if not yp.exists():
        return False
    try:
        return int(np.load(str(yp), mmap_mode="r").shape[0]) == int(n_x)
    except Exception:
        return False


def _warn_multitask_cache_sidecars(cache_path: str, args) -> None:
    """Hint to rebuild when multitask training expects y_cls/pq sidecars."""
    if not getattr(args, "multitask", False):
        return
    if _cache_has_multitask_sidecars(cache_path):
        return
    msg = (
        "[Data] Multitask is enabled but cache has no y_cls sidecar "
        f"({cache_path}). Direction/confidence heads will threshold rewards "
        "instead of true class labels.\n"
        "  Rebuild cache: .\\.venv-gpu\\Scripts\\python.exe scripts\\train.py "
        "--rebuild-cache\n"
        "  Or: training\\train_gpu.py --force-rebuild"
    )
    print(msg, flush=True)
    _log_warn(msg)


def _delete_cache_artifacts(cache_path: str) -> None:
    _ensure_bound()
    import shutil as _shutil
    p = Path(cache_path)
    # Zarr is a directory ΓÇö use shutil.rmtree
    if p.is_dir() and str(cache_path).endswith(".zarr"):
        _shutil.rmtree(p)
        print(f"[Data] Removed corrupt zarr store: {p}")
    elif p.exists():
        p.unlink()
        print(f"[Data] Removed corrupt cache artifact: {p}")
    for fp in (
        Path(_x_path(cache_path)), Path(_y_path(cache_path)), _scaler_npz_path(p),
        Path(_diff_path(cache_path)), Path(_pq_path(cache_path)),
        Path(_y_cls_path(cache_path)),
        Path(_close_path(cache_path)), Path(_atr_path(cache_path)),
        Path(_spread_path(cache_path)),
        Path(str(cache_path) + "_manifest.json"),

        Path(str(cache_path) + "_meta.json"),

        Path(str(cache_path) + "_resume.json"),

        Path(str(cache_path) + "_feature_schema.json"),

        Path(str(cache_path) + "_pair_readiness_report.json"),
    ):
        if fp.exists():
            fp.unlink()
            print(f"[Data] Removed corrupt cache artifact: {fp}")
