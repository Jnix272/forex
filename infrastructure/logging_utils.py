"""
Structured logging utilities for data pipeline.

Replaces bare print() with grep-able, structured log lines.
Usage:
    from infrastructure.logging_utils import log_data_load

    log_data_load("dukascopy", "data/raw/dukascopy/EURUSD/2024/", n_rows=1500000, status="success", t0=start_time)
    log_data_load("news_parquet", "data/raw/news/historical_news_combined.parquet", n_rows=8400000, status="success", t0=start_time)
    log_data_load("fred_yield", "FRED:US10Y", n_rows=0, status="fallback_synthetic", t0=start_time, exc=e)
"""

import logging
import time
from contextlib import contextmanager
from typing import Any

# Configure module logger
logger = logging.getLogger("forex.data")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def log_data_load(
    source: str,
    path: str,
    n_rows: int,
    status: str,
    t0: float | None = None,
    exc: Exception | None = None,
    note: str = "",
    **extra: Any,
) -> None:
    """
    Log a data load event with consistent structure.

    Args:
        source: Source identifier (dukascopy, news_parquet, fred, cross_asset, cot, etc.)
        path: File/URL/path being loaded
        n_rows: Number of rows loaded (0 if failed/empty)
        status: "success" | "empty" | "fallback_synthetic" | "skip_empty" | "error" | "cached" | "partial"
        t0: Start time from time.perf_counter() for duration calculation
        exc: Exception if status is "error"
        note: Additional context (e.g., "pair=EURUSD", "asset=US10Y")
        **extra: Additional key=value pairs for structured logging
    """
    duration_ms = int((time.perf_counter() - t0) * 1000) if t0 is not None else -1

    parts = [
        f"source={source}",
        f"path={path}",
        f"rows={n_rows:,}",
        f"status={status}",
        f"duration_ms={duration_ms}",
    ]
    if note:
        parts.append(f"note={note}")
    for k, v in extra.items():
        parts.append(f"{k}={v}")
    if exc:
        parts.append(f"error={type(exc).__name__}: {exc}")

    msg = " | ".join(parts)

    if status == "error":
        logger.error(msg)
    elif status in ("fallback_synthetic", "skip_empty", "empty", "partial"):
        logger.warning(msg)
    else:
        logger.info(msg)


@contextmanager
def timed_load(source: str, path: str, note: str = "", **extra: Any):
    """
    Context manager for timing and logging a data load.

    Usage:
        with timed_load("dukascopy", "data/raw/dukascopy/EURUSD/", pair="EURUSD") as ctx:
            df = load_data()
            ctx["n_rows"] = len(df)
            ctx["status"] = "success"
    """
    t0 = time.perf_counter()
    ctx = {"n_rows": 0, "status": "success", "extra": {}}
    try:
        yield ctx
    except Exception as e:
        ctx["status"] = "error"
        ctx["exc"] = e
        raise
    finally:
        log_data_load(
            source=source,
            path=path,
            n_rows=ctx.get("n_rows", 0),
            status=ctx.get("status", "success"),
            t0=t0,
            exc=ctx.get("exc"),
            note=note,
            **ctx.get("extra", {}),
            **extra,
        )


def log_feature_build(
    feature_group: str,
    n_rows: int,
    n_cols: int,
    status: str,
    t0: float | None = None,
    exc: Exception | None = None,
    note: str = "",
) -> None:
    """Log feature engineering step."""
    duration_ms = int((time.perf_counter() - t0) * 1000) if t0 is not None else -1
    parts = [
        f"feature_group={feature_group}",
        f"rows={n_rows:,}",
        f"cols={n_cols}",
        f"status={status}",
        f"duration_ms={duration_ms}",
    ]
    if note:
        parts.append(f"note={note}")
    if exc:
        parts.append(f"error={type(exc).__name__}: {exc}")

    msg = " | ".join(parts)
    if status == "error":
        logger.error(msg)
    elif status in ("failed", "fallback_zero"):
        logger.warning(msg)
    else:
        logger.info(msg)


def log_training_step(
    step: str,
    epoch: int | None = None,
    batch: int | None = None,
    metrics: dict | None = None,
    status: str = "progress",
    t0: float | None = None,
) -> None:
    """Log training step with optional metrics."""
    duration_ms = int((time.perf_counter() - t0) * 1000) if t0 is not None else -1
    parts = [f"step={step}", f"status={status}", f"duration_ms={duration_ms}"]
    if epoch is not None:
        parts.append(f"epoch={epoch}")
    if batch is not None:
        parts.append(f"batch={batch}")
    if metrics:
        for k, v in metrics.items():
            parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
    msg = " | ".join(parts)
    if status == "error":
        logger.error(msg)
    else:
        logger.info(msg)
