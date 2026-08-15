"""
Per-pair feature cache for fast dataset building.

Caches the full FeatureEngineer output as parquet files partitioned by month,
avoiding redundant recomputation of 170-220 features across overlapping windows.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

DEFAULT_CACHE_DIR = "data/features"
CACHE_VERSION = "v1"


def feat_cache_path(pair: str, cache_dir: str = DEFAULT_CACHE_DIR, version: str = CACHE_VERSION) -> Path:
    """Path to the per-pair feature cache directory."""
    return Path(cache_dir) / version / f"{pair.upper().replace('/', '')}"


def feat_cache_exists(pair: str, cache_dir: str = DEFAULT_CACHE_DIR, version: str = CACHE_VERSION) -> bool:
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
    fe_kwargs: dict = None,
    news_mode: str = "calendar",
    news_file: str = None,
    calendar_file: str = None,
    cot_data: dict = None,
    cross_asset: dict = None,
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
    from data.sources import ForexDataManager
    from training.dataset_builder import _ensure_bound
    import training.dataset_builder as dsb

    _ensure_bound()
    mgr = data_mgr or ForexDataManager()
    fe = dsb._make_feature_engineer()

    # Process month by month
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    month = datetime(start_dt.year, start_dt.month, 1)

    total_bars = 0
    months_processed = 0

    while month < end_dt:
        next_month = (month + timedelta(days=32)).replace(day=1)
        m_start = month.strftime("%Y-%m-%d")
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

        # Resample bars
        from training.dataset_builder import ForexDataPipeline
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
        bars = bars.filter(
            (pl.col("timestamp_utc") >= m_start_ts) &
            (pl.col("timestamp_utc") < m_end_ts)
        )

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
                **(fe_kwargs or {})
            )
        except Exception as e:
            print(f"[FeatCache] {pair} {month_key}: FE failed ({e}), skipping")
            month = next_month
            continue

        if F is None or len(F) == 0 or len(F.columns) < 5:
            print(f"[FeatCache] {pair} {month_key}: no features produced, skipping")
            month = next_month
            continue

        # Save
        F.write_parquet(output_file)
        n_cols = len(F.columns)
        n_rows = len(F)
        total_bars += n_rows
        months_processed += 1
        print(f"[FeatCache] {pair} {month_key}: {n_rows:,} bars × {n_cols} features → {output_file.name}")

        month = next_month

    elapsed = time.time() - t0
    # Write manifest
    manifest_path = output / "_manifest.json"
    import json
    manifest = {
        "pair": pair,
        "start": start,
        "end": end,
        "version": CACHE_VERSION,
        "total_bars": total_bars,
        "months": months_processed,
        "cache_dir": str(output),
        "columns": len(F.columns) if months_processed > 0 else 0,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[FeatCache] {pair}: {total_bars:,} bars × {months_processed} months in {elapsed:.0f}s")
    print(f"[FeatCache] Path: {output}")

    return str(output)


def load_cached_features(
    pair: str,
    start: str,
    end: str,
    *,
    cache_dir: str = DEFAULT_CACHE_DIR,
    version: str = CACHE_VERSION,
    columns: list[str] = None,
) -> Optional[pl.DataFrame]:
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
                (pl.col("timestamp_utc") >= pl.lit(start).str.to_datetime()) &
                (pl.col("timestamp_utc") < pl.lit(m_end).str.to_datetime())
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
        (pl.col("timestamp_utc") >= pl.lit(start).str.to_datetime()) &
        (pl.col("timestamp_utc") < pl.lit(end).str.to_datetime())
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
    scalers: dict = None,
    label_method: str = "rl_reward",
    lookahead_bars: int = 30,
    profit_atr_mult: float = 1.2,
    stop_atr_mult: float = 0.8,
    data_mgr=None,
) -> tuple[str, int, int]:
    """
    Single-pass dataset builder — processes full history once instead of window-by-window.

    Loads pre-built feature caches for all pairs, aligns timestamps, computes labels
    in one pass, builds sliding windows, and writes to Zarr.

    Returns (cache_path, n_samples, n_features_total).
    """
    import hashlib, json, time
    import numpy as np, pandas as pd
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
        exclude_cols = {"timestamp_utc", "close", "mid_close", "bid_close", "ask_close",
                        "atr_6", "atr_20", "spread_pips", "pair", "source"}
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
        raise RuntimeError(f"Only {n_total} aligned bars — need {seq_len + lookahead_bars}")

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
    from labeling.rl_reward_numba import _scan_barriers_simple, _numba_available
    delay = 1
    entry_long = close + 0.00001
    entry_short = close - 0.00001
    exit_long = entry_short.copy()
    exit_short = entry_long.copy()
    valid_market = np.ones(n_total, dtype=bool)

    if _numba_available():
        reward_long, reward_short = _scan_barriers_simple(
            close.astype(np.float64), entry_long.astype(np.float64),
            entry_short.astype(np.float64), exit_long.astype(np.float64),
            exit_short.astype(np.float64), atr_col.astype(np.float64),
            valid_market, profit_atr_mult, stop_atr_mult, 1.5, 0.0001,
            lookahead_bars, delay,
        )
    else:
        reward_long = np.zeros(n_total, dtype=np.float32)
        reward_short = np.zeros(n_total, dtype=np.float32)

    reward = np.maximum(reward_long, reward_short)
    label = np.select(
        [(reward_long > 1.5) & (reward_long >= reward_short),
         (reward_short > 1.5) & (reward_short > reward_long)],
        [1, -1], default=0
    )

    # Align labels with windows
    n_windows = X_seq.shape[0]
    y_seq = label[seq_len-1:seq_len-1+n_windows].astype(np.float32)
    y_cls_seq = y_seq.copy()

    # Quality filter
    keep = np.ones(n_windows, dtype=bool)
    keep[:n_windows - lookahead_bars - delay] = True
    keep[-(lookahead_bars + delay):] = False

    X_seq = X_seq[keep]
    y_seq = y_seq[keep]
    y_cls_seq = y_cls_seq[keep]

    # Extract close/atr/spread at label bar
    label_idx = np.arange(seq_len - 1, seq_len - 1 + n_windows)[keep]
    close_seq = close[label_idx].astype(np.float32)
    atr_seq = atr_col[label_idx].astype(np.float32)
    spread_seq = (spread_col[label_idx].astype(np.float32) if spread_col is not None
                  else np.zeros(len(label_idx), dtype=np.float32))

    # Write Zarr
    from training.gpu_cache_io import (_zarr_open_group, _zarr_create,
                                        ZARR_FEATURE_DTYPE, ZARR_LABEL_DTYPE,
                                        make_training_zarr_compressor)
    from pathlib import Path
    import os

    cp = Path(cache_path)
    cp.parent.mkdir(parents=True, exist_ok=True)

    if cp.exists():
        import shutil
        shutil.rmtree(cp, ignore_errors=True)

    _compressor = make_training_zarr_compressor()
    _chunk_rows = min(4096, len(X_seq))
    c0 = (_chunk_rows,) + X_seq.shape[1:]

    z_store = _zarr_open_group(str(cp), mode="w")
    _zarr_create(z_store, "X", shape=X_seq.shape, chunks=c0, dtype=ZARR_FEATURE_DTYPE, compressor=_compressor)
    _zarr_create(z_store, "y", shape=y_seq.shape, chunks=(c0[0],), dtype=ZARR_LABEL_DTYPE, compressor=_compressor)
    _zarr_create(z_store, "y_cls", shape=y_cls_seq.shape, chunks=(c0[0],), dtype=ZARR_LABEL_DTYPE, compressor=_compressor)
    _zarr_create(z_store, "close", shape=close_seq.shape, chunks=(c0[0],), dtype=ZARR_LABEL_DTYPE, compressor=_compressor)
    _zarr_create(z_store, "atr", shape=atr_seq.shape, chunks=(c0[0],), dtype=ZARR_LABEL_DTYPE, compressor=_compressor)
    _zarr_create(z_store, "spread", shape=spread_seq.shape, chunks=(c0[0],), dtype=ZARR_LABEL_DTYPE, compressor=_compressor)

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
