"""
Per-pair feature cache for fast dataset building.

Caches the full FeatureEngineer output as parquet files partitioned by month,
avoiding redundant recomputation of 170-220 features across overlapping windows.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl

try:
    import filelock as _filelock_mod
    _HAS_FILELOCK = True
except ImportError:
    _HAS_FILELOCK = False

DEFAULT_CACHE_DIR = "data/features"
CACHE_VERSION = "v1"


def cache_version() -> str:
    """Monthly-cache directory tag: schema version + feature-code fingerprint.

    A fixed "v1" would keep serving features computed by older code.
    """
    from feature_store.fingerprint import feature_code_fingerprint

    return f"{CACHE_VERSION}-{feature_code_fingerprint()[:12]}"


def feat_cache_path(pair: str, cache_dir: str = DEFAULT_CACHE_DIR, version: str | None = None) -> Path:
    """Path to the per-pair feature cache directory."""
    return Path(cache_dir) / (version or cache_version()) / f"{pair.upper().replace('/', '')}"


def feat_cache_exists(pair: str, cache_dir: str = DEFAULT_CACHE_DIR, version: str | None = None) -> bool:
    """Check if a feature cache exists for this pair."""
    p = feat_cache_path(pair, cache_dir, version)
    return p.is_dir() and any(p.glob("*.parquet"))


def _month_key(dt) -> str:
    """Normalize to YYYY-MM cache key."""
    if hasattr(dt, "strftime"):
        return dt.strftime("%Y-%m")
    return str(dt)[:7]


def build_pair_feature_cache(
    pair: str,
    start: str,
    end: str,
    *,
    cache_dir: str = DEFAULT_CACHE_DIR,
    bar_freq: str = "5m",
    fe_kwargs: dict | None = None,
    news_mode: str = "calendar",
    news_file: str | None = None,
    calendar_file: str | None = None,
    cot_data: dict | None = None,
    cross_asset: dict | None = None,
    sentiment_pipe=None,
    data_mgr=None,
    overwrite: bool = False,
) -> str:
    """
    Build full-history feature cache for a single pair.

    Processes the full date range month-by-month, runs feature engineering
    once per month, and saves to partitioned parquet files.

    Args:
        pair: Currency pair (e.g., "EURUSD")
        start, end: Date range (ISO format)
        cache_dir: Root directory for feature caches
        bar_freq: Bar frequency
        fe_kwargs: Extra args for FeatureEngineer.build()
        news_mode, news_file, calendar_file: News bundle config
        cot_data: Pre-loaded COT dataframe
        cross_asset: Cross-asset panel dict
        sentiment_pipe: Sentiment pipeline instance
        data_mgr: ForexDataManager instance (if None, creates one)
        overwrite: Rebuild even if cache exists

    Returns:
        Path to the cache directory
    """
    from datetime import datetime, timedelta

    t0 = time.time()
    output = feat_cache_path(pair, cache_dir)
    output.mkdir(parents=True, exist_ok=True)

    if feat_cache_exists(pair, cache_dir) and not overwrite:
        print(f"[FeatCache] {pair} cache exists at {output}")
        return str(output)

    # Initialize data infrastructure
    from data.data_ingestion import ForexDataPipeline
    from data.sources import ForexDataManager
    from features.feature_engineering_pl import FeatureEngineer

    mgr = data_mgr or ForexDataManager()
    fe = None
    if fe is None:
        # Same construction as training/dataset_builder (canonical config).
        from config.settings import FEATURES
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

    # Process month by month
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    month = datetime(start_dt.year, start_dt.month, 1)

    total_bars = 0
    months_processed = 0

    while month < end_dt:
        next_month = (month + timedelta(days=32)).replace(day=1)
        month.strftime("%Y-%m-%d")
        m_end = min(next_month, end_dt).strftime("%Y-%m-%d")
        month_key = _month_key(month)

        output_file = output / f"{month_key}.parquet"
        if output_file.exists() and not overwrite:
            print(f"[FeatCache] {pair} {month_key} already cached, skipping")
            month = next_month
            continue

        # Load ticks for this month (with 2-week warmup)
        warmup_start = (month - timedelta(days=14)).strftime("%Y-%m-%d")
        try:
            ticks = mgr.load(pair=pair, source="dukascopy", start=warmup_start, end=m_end)
        except Exception as e:
            print(f"[FeatCache] {pair} {month_key}: tick load failed ({e}), skipping")
            month = next_month
            continue

        if ticks is None or len(ticks) == 0:
            print(f"[FeatCache] {pair} {month_key}: no ticks, skipping")
            month = next_month
            continue

        # Resample bars (canonical impl lives in data/data_ingestion;
        # training/dataset_builder re-exports it)
        pipeline = ForexDataPipeline(
            bar_freq=bar_freq or "5m",
            session_filter=False,
            apply_frac_diff=False,
            session_mode="dst",
            add_session_label=True,
            spread_cap_multiplier=3.0,
        )
        bars = pipeline.run(ticks, pair=pair)

        # Filter to target month
        m_start_ts = pl.lit(month.strftime("%Y-%m-%d")).str.to_datetime()
        m_end_ts = pl.lit(m_end).str.to_datetime()
        bars = bars.filter((pl.col("timestamp_utc") >= m_start_ts) & (pl.col("timestamp_utc") < m_end_ts))

        if len(bars) < 50:
            print(f"[FeatCache] {pair} {month_key}: only {len(bars)} bars, skipping")
            month = next_month
            continue

        # Feature engineering
        try:
            F = fe.build(
                bars,
                cross_asset=cross_asset or {},
                sentiment_pipe=sentiment_pipe,
                news_mode=news_mode or "calendar",
                news_file=news_file,
                calendar_file=calendar_file,
                cot_data=cot_data,
                pair=pair,
                **(fe_kwargs or {}),
            )
        except Exception as e:
            print(f"[FeatCache] {pair} {month_key}: FE failed ({e}), skipping")
            month = next_month
            continue

        if F is None or len(F) == 0 or len(F.columns) < 5:
            print(f"[FeatCache] {pair} {month_key}: no features produced, skipping")
            month = next_month
            continue

        # Save (file-locked to prevent concurrent build corruption)
        if _HAS_FILELOCK:
            with _filelock_mod.FileLock(str(output_file) + ".lock", timeout=60):
                F.write_parquet(output_file)
        else:
            F.write_parquet(output_file)
        n_cols = len(F.columns)
        n_rows = len(F)
        total_bars += n_rows
        months_processed += 1
        print(f"[FeatCache] {pair} {month_key}: {n_rows:,} bars x {n_cols} features → {output_file.name}")

        month = next_month

    elapsed = time.time() - t0
    # Write manifest
    manifest_path = output / "_manifest.json"
    import json

    manifest = {
        "pair": pair,
        "start": start,
        "end": end,
        "version": output.parent.name,
        "total_bars": total_bars,
        "months": months_processed,
        "cache_dir": str(output),
        "columns": len(F.columns) if months_processed > 0 else 0,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[FeatCache] {pair}: {total_bars:,} bars x {months_processed} months in {elapsed:.0f}s")
    print(f"[FeatCache] Path: {output}")

    return str(output)


def load_cached_features(
    pair: str,
    start: str,
    end: str,
    *,
    cache_dir: str = DEFAULT_CACHE_DIR,
    version: str | None = None,
    columns: list[str] | None = None,
) -> pl.DataFrame | None:
    """
    Load cached features for a date range.

    Returns None if the cache doesn't exist or the date range has no data.
    """
    from datetime import datetime, timedelta

    cache = feat_cache_path(pair, cache_dir, version)
    if not cache.is_dir():
        return None

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    month = datetime(start_dt.year, start_dt.month, 1)

    frames = []
    while month < end_dt:
        next_month = (month + timedelta(days=32)).replace(day=1)
        f = cache / f"{_month_key(month)}.parquet"
        if f.exists():
            df = pl.read_parquet(f)
            # Filter to requested date range
            m_end = next_month.strftime("%Y-%m-%d")
            df = df.filter(
                (pl.col("timestamp_utc") >= pl.lit(start).str.to_datetime())
                & (pl.col("timestamp_utc") < pl.lit(m_end).str.to_datetime())
            )
            if len(df) > 0:
                if columns:
                    df = df.select([c for c in columns if c in df.columns])
                frames.append(df)
        month = next_month

    if not frames:
        return None

    result = pl.concat(frames)
    # Re-filter to exact range
    result = result.filter(
        (pl.col("timestamp_utc") >= pl.lit(start).str.to_datetime())
        & (pl.col("timestamp_utc") < pl.lit(end).str.to_datetime())
    )
    return result if len(result) > 0 else None


def build_single_pass_dataset(
    pairs: list[str],
    start: str,
    end: str,
    cache_path: str,
    *,
    feature_cache_dir: str = DEFAULT_CACHE_DIR,
    seq_len: int = 80,
    fe=None,
    scalers: dict | None = None,
    label_method: str = "rl_reward",
    lookahead_bars: int = 30,
    profit_atr_mult: float = 1.2,
    stop_atr_mult: float = 0.8,
    data_mgr=None,
) -> tuple[str, int, int]:
    """
    Single-pass dataset builder - processes full history once instead of window-by-window.

    Loads pre-built feature caches for all pairs, aligns timestamps, computes labels
    in one pass, builds sliding windows, and writes to Zarr.

    Returns (cache_path, n_samples, n_features_total).
    """
    import time
    from pathlib import Path

    t0 = time.time()
    print(f"\n[SinglePass] {len(pairs)} pairs: {', '.join(pairs)}")
    print(f"[SinglePass] {start} → {end} | seq_len={seq_len} | lookahead={lookahead_bars}")

    # Check feature caches exist
    missing = [p for p in pairs if not feat_cache_exists(p, feature_cache_dir)]
    if missing:
        raise RuntimeError(
            f"Feature cache not found for: {', '.join(missing)}. "
            f"Run: python data/feature_cache.py --pairs {' '.join(missing)}"
        )

    # Load features for all pairs
    pair_features = {}
    pair_times = {}
    for pair in pairs:
        f_df = load_cached_features(pair, start, end, cache_dir=feature_cache_dir)
        if f_df is None or len(f_df) == 0:
            raise RuntimeError(f"No cached features for {pair} in {start}→{end}")
        pair_features[pair] = f_df
        pair_times[pair] = f_df["timestamp_utc"].to_numpy()
        print(f"[SinglePass] {pair}: {len(f_df):,} bars loaded from cache")

    # Align pairs on timestamp (inner join)
    common_ts = pair_times[pairs[0]]
    for p in pairs[1:]:
        common_set = set(common_ts)
        other_set = set(pair_times[p])
        common_ts = np.array(sorted(common_set & other_set))

    if len(common_ts) == 0:
        raise RuntimeError("No common timestamps across pairs")

    print(f"[SinglePass] Aligned: {len(common_ts):,} common timestamps")

    # Build aligned feature matrix per pair
    pair_Xs = {}
    pair_markets = {}
    for p in pairs:
        df = pair_features[p]
        # Create timestamp index for fast lookup
        ts_map = {t: i for i, t in enumerate(df["timestamp_utc"].to_numpy())}
        idx = np.array([ts_map.get(t, -1) for t in common_ts], dtype=np.int64)
        valid = idx >= 0
        common_ts_filtered = common_ts[valid]
        idx = idx[valid]

        # Extract feature columns (exclude timestamp, close, atr, spread, etc.)
        exclude_cols = {
            "timestamp_utc",
            "close",
            "mid_close",
            "bid_close",
            "ask_close",
            "atr_6",
            "atr_20",
            "spread_pips",
            "pair",
            "source",
        }
        feat_cols = [c for c in df.columns if c not in exclude_cols]
        X = np.asarray(df.select(feat_cols).to_numpy(), dtype=np.float32)
        pair_Xs[p] = X[idx]

        # Extract market columns for labeling
        mkt = {}
        for col in ["close", "mid_close", "atr_6", "atr_20", "spread_pips"]:
            if col in df.columns:
                mkt[col] = np.asarray(df[col].to_numpy(), dtype=np.float32)[idx]
        pair_markets[p] = mkt

    common_ts = common_ts_filtered
    n_total = len(common_ts)

    if n_total < seq_len + lookahead_bars:
        raise RuntimeError(f"Only {n_total} aligned bars - need {seq_len + lookahead_bars}")

    # Build sliding windows from aligned data
    n_feat_per_pair = pair_Xs[pairs[0]].shape[1]
    n_total_features = n_feat_per_pair * len(pairs)

    # Stack all pairs into one feature matrix
    X_aligned = np.concatenate([pair_Xs[p] for p in pairs], axis=1)  # (n, F_total)

    # Build sliding windows
    from numpy.lib.stride_tricks import sliding_window_view

    X_seq = sliding_window_view(X_aligned, (seq_len, n_total_features))  # (n-seq+1, seq_len, F)
    X_seq = np.ascontiguousarray(X_seq)

    # Compute labels on the first pair's market data
    close = pair_markets[pairs[0]].get("close", pair_markets[pairs[0]].get("mid_close"))
    atr_col = pair_markets[pairs[0]].get("atr_6")
    spread_col = pair_markets[pairs[0]].get("spread_pips")

    if close is None or atr_col is None:
        raise RuntimeError("Missing close or atr columns in feature cache")

    # Simple labeling: use Numba-accelerated scan
    from labeling.rl_reward_numba import _numba_available, _scan_barriers_simple

    delay = 1
    entry_long = close + 0.00001
    entry_short = close - 0.00001
    exit_long = entry_short.copy()
    exit_short = entry_long.copy()
    valid_market = np.ones(n_total, dtype=bool)

    if _numba_available():
        reward_long, reward_short = _scan_barriers_simple(
            close.astype(np.float64),
            entry_long.astype(np.float64),
            entry_short.astype(np.float64),
            exit_long.astype(np.float64),
            exit_short.astype(np.float64),
            atr_col.astype(np.float64),
            valid_market,
            profit_atr_mult,
            stop_atr_mult,
            1.5,
            0.0001,
            lookahead_bars,
            delay,
        )
    else:
        reward_long = np.zeros(n_total, dtype=np.float32)
        reward_short = np.zeros(n_total, dtype=np.float32)

    np.maximum(reward_long, reward_short)
    label = np.select(
        [(reward_long > 1.5) & (reward_long >= reward_short), (reward_short > 1.5) & (reward_short > reward_long)],
        [1, -1],
        default=0,
    )

    # Align labels with windows
    n_windows = X_seq.shape[0]
    y_seq = label[seq_len - 1 : seq_len - 1 + n_windows].astype(np.float32)
    y_cls_seq = y_seq.copy()

    # Quality filter
    keep = np.ones(n_windows, dtype=bool)
    keep[: n_windows - lookahead_bars - delay] = True
    keep[-(lookahead_bars + delay) :] = False

    X_seq = X_seq[keep]
    y_seq = y_seq[keep]
    y_cls_seq = y_cls_seq[keep]

    # Extract close/atr/spread at label bar
    label_idx = np.arange(seq_len - 1, seq_len - 1 + n_windows)[keep]
    close_seq = close[label_idx].astype(np.float32)
    atr_seq = atr_col[label_idx].astype(np.float32)
    spread_seq = (
        spread_col[label_idx].astype(np.float32)
        if spread_col is not None
        else np.zeros(len(label_idx), dtype=np.float32)
    )

    # Write Zarr

    from common.cache_io import (
        ZARR_FEATURE_DTYPE,
        ZARR_LABEL_DTYPE,
        _zarr_create,
        _zarr_open_group,
        make_training_zarr_compressor,
    )

    cp = Path(cache_path)
    cp.parent.mkdir(parents=True, exist_ok=True)

    if cp.exists():
        import shutil

        shutil.rmtree(cp, ignore_errors=True)

    _compressor = make_training_zarr_compressor()
    _chunk_rows = min(4096, len(X_seq))
    c0 = (_chunk_rows, *X_seq.shape[1:])

    z_store = _zarr_open_group(str(cp), mode="w")
    _zarr_create(z_store, "X", shape=X_seq.shape, chunks=c0, dtype=ZARR_FEATURE_DTYPE, compressor=_compressor)
    _zarr_create(z_store, "y", shape=y_seq.shape, chunks=(c0[0],), dtype=ZARR_LABEL_DTYPE, compressor=_compressor)
    _zarr_create(
        z_store, "y_cls", shape=y_cls_seq.shape, chunks=(c0[0],), dtype=ZARR_LABEL_DTYPE, compressor=_compressor
    )
    _zarr_create(
        z_store, "close", shape=close_seq.shape, chunks=(c0[0],), dtype=ZARR_LABEL_DTYPE, compressor=_compressor
    )
    _zarr_create(z_store, "atr", shape=atr_seq.shape, chunks=(c0[0],), dtype=ZARR_LABEL_DTYPE, compressor=_compressor)
    _zarr_create(
        z_store, "spread", shape=spread_seq.shape, chunks=(c0[0],), dtype=ZARR_LABEL_DTYPE, compressor=_compressor
    )

    z_store["X"][:] = np.asarray(X_seq, dtype=ZARR_FEATURE_DTYPE)
    z_store["y"][:] = y_seq
    z_store["y_cls"][:] = y_cls_seq
    z_store["close"][:] = close_seq
    z_store["atr"][:] = atr_seq
    z_store["spread"][:] = spread_seq

    z_store.attrs["total_samples"] = len(X_seq)
    z_store.attrs["n_features"] = n_total_features
    z_store.attrs["seq_len"] = seq_len
    z_store.attrs["single_pass"] = True

    elapsed = time.time() - t0
    print(f"[SinglePass] {len(X_seq):,} windows, {n_total_features} features in {elapsed:.0f}s")
    print(f"[SinglePass] Zarr saved → {cp}")

    return str(cp), len(X_seq), n_total_features


# ════════════════════════════════════════════════════════════════════════════
# Per-window feature cache (training/dataset_builder._build_chunk)
# ════════════════════════════════════════════════════════════════════════════
#
# Build-time only: the live engine never reads it, so live features are
# unaffected. Each entry is the exact FeatureEngineer output for one
# (pair, window) *before* the feature mask, keyed by everything that can change
# it: the bars themselves (ticks, window range incl. warmup, bar construction),
# every side input passed to FeatureEngineer.build (content digests), the COT
# parquet file signature, feature-code fingerprint, dataset build version,
# FeatureEngineer settings and features.* windows. Any failure falls back to a
# normal feature build with a warning.

WINDOW_CACHE_SCHEMA = 1
_COT_PARQUET = Path("data/raw/cot/cot_financials_cleaned.parquet")

# FeatureEngineer.build kwargs each known slow column depends on (besides bars).
# Unknown slow columns are keyed on all inputs (conservative).
SLOW_COL_INPUTS: dict[str, tuple[str, ...]] = {
    "sentiment_decayed": ("sentiment", "news_events"),
    "sentiment_raw": ("sentiment", "news_events"),
    "eco_surprise": ("eco_act", "eco_fc", "eco_prior"),
    "eco_revision": ("eco_act", "eco_fc", "eco_prior"),
    "hurst_exponent": (),
    "cot_net_hf": ("cot_data",),
    "cot_net_comm": ("cot_data",),
    "cot_hf_mom_4w": ("cot_data",),
    "cot_extreme": ("cot_data",),
}

_WARNED: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(f"[FeatCache] WARN: {msg}", flush=True)


def _yaml_root(config_path: str | None = None) -> dict:
    import os

    cfg_path = os.environ.get("FOREX_CONFIG") or os.environ.get("FOREX_RUN_CONFIG") or config_path
    if not cfg_path or not os.path.exists(str(cfg_path)):
        return {}
    try:
        import yaml

        with open(cfg_path, encoding="utf-8-sig") as f:
            root = yaml.safe_load(f) or {}
        return root if isinstance(root, dict) else {}
    except Exception:
        return {}


def normalize_slow_cols(cols) -> list[str]:
    return ["hurst_exponent" if c == "hurst" else str(c) for c in (cols or [])]


def resolve_feature_cache_config(args=None, yaml_root: dict | None = None) -> dict:
    """Effective feature-cache settings.

    Precedence (low -> high): settings.FEATURE_CACHE, FOREX_CONFIG YAML
    (feature_cache.* merged key by key, data.use_feature_cache,
    data.feature_cache_dir), then parsed CLI args (args.feature_cache dict,
    args.use_feature_cache, args.feature_cache_dir).

    The cache is active only when ``feature_cache.enabled`` AND
    ``data.use_feature_cache`` are both true.
    """
    try:
        from config.settings import FEATURE_CACHE
    except Exception:
        FEATURE_CACHE = {}
    fc: dict = dict(FEATURE_CACHE or {})
    root = _yaml_root(getattr(args, "config", None)) if yaml_root is None else yaml_root
    fc.update(root.get("feature_cache") or {})
    data_cfg = root.get("data") or {}
    use_cache = bool(data_cfg.get("use_feature_cache", False))
    cache_dir = data_cfg.get("feature_cache_dir") or DEFAULT_CACHE_DIR
    if args is not None:
        a_fc = getattr(args, "feature_cache", None)
        if isinstance(a_fc, dict):
            fc.update(a_fc)
        if getattr(args, "use_feature_cache", None) is not None:
            use_cache = bool(args.use_feature_cache)
        if getattr(args, "feature_cache_dir", None):
            cache_dir = str(args.feature_cache_dir)
    section_enabled = bool(fc.get("enabled", False))
    return {
        "enabled": section_enabled and use_cache,
        "section_enabled": section_enabled,
        "use_feature_cache": use_cache,
        "cache_dir": str(cache_dir),
        "slow_cols": normalize_slow_cols(fc.get("slow_cols")),
        "ofi_z_threshold": fc.get("ofi_z_threshold"),
        "regime_window": fc.get("regime_window"),
    }


def warn_unimplemented_feature_cache_keys(cfg: dict) -> None:
    """ofi_z_threshold / feature_cache.regime_window have no consumer that would not
    change the trained feature layout (and break live parity); say so once."""
    if cfg.get("ofi_z_threshold") is not None:
        _warn_once(
            "ofi_z_threshold",
            "feature_cache.ofi_z_threshold is accepted but not used: no feature consumes an OFI "
            "z-score threshold, and adding one would change the feature layout vs live/checkpoints.",
        )
    if cfg.get("regime_window") is not None:
        _warn_once(
            "fc_regime_window",
            "feature_cache.regime_window is accepted but not used: the regime window is "
            "features.regime_window (FeatureEngineer.vol_regime_w / cross-asset regime).",
        )
    if cfg.get("section_enabled") and not cfg.get("use_feature_cache"):
        _warn_once(
            "fc_gate",
            "feature_cache.enabled is true but data.use_feature_cache is false - window feature cache is off.",
        )


# ── content digests ─────────────────────────────────────────────────────────

_DIGEST_MEMO: dict[int, tuple[object, str]] = {}


def frame_digest(obj) -> str:
    """Content digest of a feature-builder input. Raises on unsupported types so an
    un-hashable input disables caching instead of risking a key collision."""
    import hashlib
    import json

    if obj is None:
        return "none"
    memo = _DIGEST_MEMO.get(id(obj))
    if memo is not None and memo[0] is obj:
        return memo[1]
    h = hashlib.sha256()
    if isinstance(obj, pl.DataFrame):
        h.update(b"pl")
        h.update(json.dumps([(c, str(t)) for c, t in obj.schema.items()]).encode())
        h.update(str(obj.height).encode())
        if obj.height:
            h.update(obj.hash_rows(seed=0, seed_1=1, seed_2=2, seed_3=3).to_numpy().tobytes())
    elif isinstance(obj, pl.Series):
        h.update(b"pls" + str(obj.dtype).encode() + str(len(obj)).encode())
        if len(obj):
            h.update(obj.hash(seed=0, seed_1=1, seed_2=2, seed_3=3).to_numpy().tobytes())
    elif isinstance(obj, np.ndarray):
        h.update(b"np" + str(obj.dtype).encode() + str(obj.shape).encode())
        h.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, dict):
        h.update(b"dict")
        for k in sorted(obj, key=str):
            h.update(str(k).encode() + b"=" + frame_digest(obj[k]).encode() + b";")
    elif isinstance(obj, (list, tuple)):
        h.update(b"seq")
        for v in obj:
            h.update(frame_digest(v).encode() + b";")
    elif isinstance(obj, (str, int, float, bool)):
        h.update(b"s" + repr(obj).encode())
    else:
        import pandas as pd

        if isinstance(obj, (pd.DataFrame, pd.Series)):
            h.update(b"pd" + str(getattr(obj, "shape", "")).encode())
            if isinstance(obj, pd.DataFrame):
                h.update(json.dumps([(str(c), str(t)) for c, t in obj.dtypes.items()]).encode())
            else:
                h.update(str(obj.dtype).encode() + str(obj.name).encode())
            h.update(pd.util.hash_pandas_object(obj, index=True).to_numpy().tobytes())
        elif hasattr(obj, "isoformat"):
            h.update(b"dt" + obj.isoformat().encode())
        else:
            raise TypeError(f"cannot digest feature input of type {type(obj).__name__}")
    digest = h.hexdigest()
    if isinstance(obj, (pl.DataFrame, np.ndarray)) or type(obj).__name__ in ("DataFrame", "Series"):
        if len(_DIGEST_MEMO) > 64:
            _DIGEST_MEMO.clear()
        _DIGEST_MEMO[id(obj)] = (obj, digest)
    return digest


def _simple_state(obj, depth: int = 1) -> dict:
    out: dict = {}
    for k, v in sorted(vars(obj).items()):
        if k.startswith("_"):
            continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float, str, bool)) for x in v):
            out[k] = list(v)
        elif depth > 0 and hasattr(v, "__dict__") and not callable(v):
            out[k] = _simple_state(v, depth - 1)
    return out


def fe_settings_digest(fe) -> str:
    from feature_store.fingerprint import effective_feature_scales, stable_digest

    return stable_digest({"fe": _simple_state(fe), "feature_scales": effective_feature_scales()}, 32)


class WindowFeatureCache:
    """Atomic parquet cache of FeatureEngineer output per (pair, window)."""

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR, slow_cols=None, build_version: str = ""):
        self.root = Path(cache_dir)
        self.slow_cols = normalize_slow_cols(slow_cols)
        self.build_version = str(build_version)
        self.stats = {"hits": 0, "misses": 0, "slow_hits": 0, "slow_writes": 0, "errors": 0}

    @classmethod
    def from_config(cls, cfg: dict, build_version: str = "") -> WindowFeatureCache | None:
        if not cfg.get("enabled"):
            return None
        return cls(cfg.get("cache_dir") or DEFAULT_CACHE_DIR, cfg.get("slow_cols"), build_version)

    # ── keys ────────────────────────────────────────────────────────────
    def key_parts(self, *, pair, bars, fe, fe_kwargs, win_start, bar_freq, seq_len) -> dict:
        from feature_store.fingerprint import feature_code_fingerprint, file_signature

        return {
            "schema": WINDOW_CACHE_SCHEMA,
            "build_version": self.build_version,
            "code": feature_code_fingerprint(),
            "polars": pl.__version__,
            "pair": str(pair).upper(),
            "win_start": str(win_start) if win_start else None,
            "bar_freq": str(bar_freq),
            "seq_len": int(seq_len),
            "bars": frame_digest(bars),
            "fe_settings": fe_settings_digest(fe),
            "cot_file": file_signature(_COT_PARQUET),
            "inputs": {str(k): frame_digest(v) for k, v in sorted(fe_kwargs.items())},
        }

    def _slow_parts(self, parts: dict) -> dict:
        needed: set[str] = set()
        for col in self.slow_cols:
            deps = SLOW_COL_INPUTS.get(col)
            needed.update(parts["inputs"].keys() if deps is None else deps)
        needed.add("pair")
        slow = {k: v for k, v in parts.items() if k != "inputs"}
        slow["inputs"] = {k: v for k, v in parts["inputs"].items() if k in needed}
        slow["slow_cols"] = sorted(self.slow_cols)
        return slow

    @staticmethod
    def _hash(parts: dict) -> str:
        from feature_store.fingerprint import stable_digest

        return stable_digest(parts)

    def window_path(self, parts: dict) -> Path:
        h = self._hash(parts)
        return self.root / "windows" / parts["pair"] / h[:2] / f"{h}.parquet"

    def slow_path(self, parts: dict) -> Path:
        h = self._hash(self._slow_parts(parts))
        return self.root / "slow" / parts["pair"] / h[:2] / f"{h}.parquet"

    # ── io ──────────────────────────────────────────────────────────────
    @staticmethod
    def _atomic_write(df: pl.DataFrame, path: Path) -> None:
        import os
        import uuid

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            df.write_parquet(tmp, compression="zstd")
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()

    @staticmethod
    def _read(path: Path) -> pl.DataFrame | None:
        if not path.exists():
            return None
        try:
            return pl.read_parquet(path)
        except Exception as e:
            _warn_once(f"corrupt:{path}", f"unreadable cache entry {path.name} ({e}); recomputing")
            try:
                path.unlink()
            except OSError:
                pass
            return None

    # ── main entry ──────────────────────────────────────────────────────
    def get_or_build(self, build_fn, *, pair, bars, fe, fe_kwargs, win_start, bar_freq, seq_len) -> pl.DataFrame:
        try:
            parts = self.key_parts(
                pair=pair, bars=bars, fe=fe, fe_kwargs=fe_kwargs, win_start=win_start, bar_freq=bar_freq, seq_len=seq_len
            )
            path = self.window_path(parts)
            cached = self._read(path)
        except Exception as e:
            self.stats["errors"] += 1
            _warn_once(f"key:{type(e).__name__}", f"cache key failed ({e}); computing features without cache")
            return build_fn()
        if cached is not None:
            self.stats["hits"] += 1
            return cached

        self.stats["misses"] += 1
        F = build_fn()
        try:
            F = self._apply_slow_cols(F, parts)
            self._atomic_write(F, path)
        except Exception as e:
            self.stats["errors"] += 1
            _warn_once(f"write:{type(e).__name__}", f"cache write failed ({e}); features used uncached")
        return F

    def _apply_slow_cols(self, F: pl.DataFrame, parts: dict) -> pl.DataFrame:
        """Slow columns are stored separately, keyed only by the inputs they depend on.

        When a cached slow entry matches (its inputs are unchanged) its values replace
        the freshly computed ones; with deterministic feature code they are identical,
        so this pins slow columns across changes to unrelated inputs. FeatureEngineer
        has no per-column entry point, so the full build still runs on a window miss.
        """
        cols = [c for c in self.slow_cols if c in F.columns]
        if not cols or "timestamp_utc" not in F.columns:
            return F
        spath = self.slow_path(parts)
        slow = self._read(spath)
        if slow is not None and slow.height == F.height and all(c in slow.columns for c in cols):
            if slow["timestamp_utc"].cast(F.schema["timestamp_utc"]).equals(F["timestamp_utc"]):
                self.stats["slow_hits"] += 1
                return F.with_columns([slow[c].cast(F.schema[c]).alias(c) for c in cols])
        self._atomic_write(F.select(["timestamp_utc", *cols]), spath)
        self.stats["slow_writes"] += 1
        return F

    def summary(self) -> str:
        s = self.stats
        return (
            f"[FeatCache] windows hit={s['hits']} miss={s['misses']} slow_hit={s['slow_hits']} "
            f"errors={s['errors']} dir={self.root}"
        )
