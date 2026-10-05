"""
features/engineering/sentiment.py
Sentiment features: decay, buzz, FinBERT projection, sentiment tiers.
"""

import numpy as np
import polars as pl


EVENT_TIME_COL = "_sentiment_event_time"


def _to_ns(series: pl.Series) -> np.ndarray:
    return series.cast(pl.Datetime("ns", "UTC")).cast(pl.Int64).to_numpy()


def sentiment_decay(s_df: pl.DataFrame, lam: float = 0.1) -> np.ndarray:
    """Exponentially decay ``sentiment_raw`` by hours since its source event.

    ``lam`` is per hour (0.1 -> ~6.9h half-life). The event time comes from
    ``EVENT_TIME_COL`` (the asof-joined event's availability time). Without it,
    a new event is inferred wherever the forward-filled value changes: every
    bar of an asof-joined series is non-zero, so "last non-zero bar" would
    reset the clock each bar and the decay would never apply.
    """
    if s_df is None or len(s_df) == 0:
        return np.zeros(0, dtype=float)

    vals = np.nan_to_num(s_df["sentiment_raw"].to_numpy().astype(float), nan=0.0)
    bar_ns = _to_ns(s_df["timestamp_utc"])
    dec = np.zeros(len(vals), dtype=float)

    if EVENT_TIME_COL in s_df.columns:
        ev = s_df[EVENT_TIME_COL]
        valid = ev.is_not_null().to_numpy()
        ev_ns = _to_ns(ev.fill_null(s_df["timestamp_utc"]))
    else:
        idx = np.arange(len(vals))
        changed = np.r_[True, vals[1:] != vals[:-1]] & (vals != 0.0)
        last_idx = np.maximum.accumulate(np.where(changed, idx, -1))
        valid = (last_idx >= 0) & (vals != 0.0)
        ev_ns = bar_ns[np.maximum(last_idx, 0)]

    if valid.any():
        elapsed_h = np.maximum(bar_ns[valid] - ev_ns[valid], 0) / 3.6e12
        dec[valid] = vals[valid] * np.exp(-float(lam) * elapsed_h)
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
