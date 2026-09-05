"""Shared time-formatting helpers used across monitoring, risk, and data."""

from __future__ import annotations

from datetime import datetime, timezone

UTC = timezone.utc


def now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def utcnow_str() -> str:
    """Return current UTC time as a human-readable string (YYYY-MM-DD HH:MM:SS)."""
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
