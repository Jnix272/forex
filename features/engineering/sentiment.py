"""
features/engineering/sentiment.py
Sentiment features: decay, buzz, FinBERT projection, sentiment tiers.
"""

import numpy as np
import polars as pl


def sentiment_decay(s_df: pl.DataFrame, lam: float = 0.1) -> np.ndarray:
    if s_df is None or len(s_df) == 0:
        return np.zeros(0, dtype=float)
    import pandas as pd

    ts_ns = pd.to_datetime(s_df["timestamp_utc"].to_numpy(), utc=True).view("int64").astype(np.int64)
    # Fill null sentiment with 0 (neutral) before processing
    vals = np.nan_to_num(s_df["sentiment_raw"].to_numpy().astype(float), nan=0.0)
    event_mask = vals != 0.0
    last_idx = np.maximum.accumulate(np.where(event_mask, np.arange(len(vals)), -1))
    dec = np.zeros(len(vals), dtype=float)
    valid = last_idx >= 0
    if valid.any():
        elapsed = (ts_ns[valid] - ts_ns[last_idx[valid]]) / 1e9
        dec[valid] = vals[last_idx[valid]] * np.exp(-float(lam) * elapsed)
    return dec


def buzz_score(window: int = 5) -> pl.Expr:
    return pl.col("article_counts").rolling_sum(window).fill_null(0.0).alias("buzz")


def proj_finbert(emb, dim=8):
    rng = np.random.default_rng(0)
    P = rng.standard_normal((768, dim)).astype(np.float32)
    P /= np.linalg.norm(P, axis=0, keepdims=True) + 1e-9
    e = emb.reshape(1, -1) if emb.ndim == 1 else emb
    return (e @ P).squeeze()


def sentiment_tiers(df: pl.DataFrame, decay_lam: float = 0.1, fb_dim: int = 8) -> pl.DataFrame:
    """Add three sentiment columns:
    1. raw sentiment (if present)
    2. decayed sentiment (exponential decay)
    3. FinBERT projection to low dim space.

    Null sentiment values are treated as 0/neutral.
    """
    # raw sentiment assumed in column "sentiment"
    if "sentiment" in df.columns:
        # Fill null sentiment with 0 (neutral) before processing
        df = df.with_columns([pl.col("sentiment").fill_null(0.0).alias("sentiment_raw")])
        df = df.with_columns(pl.Series("sentiment_decayed", sentiment_decay(df, decay_lam)))
    else:
        df = df.with_columns([pl.lit(0.0).alias("sentiment_raw"), pl.lit(0.0).alias("sentiment_decayed")])
    # FinBERT projection - placeholder zeros if embeddings not provided
    for i in range(fb_dim):
        df = df.with_columns([pl.lit(0.0).alias(f"fb_{i}")])
    return df


def news_filter_expr(news_events: list, buf_min: int = 15) -> list[pl.Expr]:
    # Polars implementation using list matching or just looping if list is small.
    # This is intentionally left as a no-op placeholder for callers that
    # integrate an event dataset upstream; it does not create additional columns.
    del news_events, buf_min
    return []
