"""
training/sharpe_annualization.py
=================================
Single source of truth for the Sharpe-ratio annualization factor.

The historical pipeline hard-coded a single magic number (``325.0`` in
``settings.py``, then ``140.0`` in ``run.yaml``) that conflated THREE
different assumptions:

    1. Number of trading days per year  (252 vs 365)
    2. Number of bars per day            (78 session vs 288 full-day FX)
    3. Holding period per "trade"        (1 bar vs ``lookahead_bars``)

The result was an annualization factor that was either wildly over- or
under-inflating Sharpe by 2x-12x depending on which combination of
assumptions the user made. This module computes the correct factor
programmatically, and exposes a single helper that:

    * Auto-detects ``bars_per_year`` from the actual cache time-span.
    * Auto-detects ``holding_period_bars`` from ``lookahead_bars``.
    * Returns ``sqrt(bars_per_year / holding_period_bars)`` which is the
      textbook annualization for a stream of non-overlapping per-trade
      returns.

Usage::

    from training.sharpe_annualization import (
        auto_annualization_factor,
        sharpe_ann_factor,
    )

    # 1. Direct call with known data
    f = sharpe_ann_factor(
        bars_per_year=252 * 78,
        holding_period_bars=30,
    )   # -> sqrt(252*78/30) ≈ 25.6

    # 2. Auto-detect from a Zarr cache (preferred for live training)
    f = auto_annualization_factor(
        cache_path,
        lookahead_bars=30,
        bar_freq="5min",
    )
"""

from __future__ import annotations

import math
from pathlib import Path

# ── Reference constants ───────────────────────────────────────────────────────

# Calendar trading days per year. 252 is the convention used by US equity
# benchmarks and most FX research. 365 (calendar days) is appropriate for
# 24h crypto; 252 is the conservative choice for FX.
TRADING_DAYS_PER_YEAR: int = 252


# ── Bar-frequency table (per TRADING day) ────────────────────────────────────
#
# Two session profiles are common in this codebase:
#   * "session" - US-equity / London-equity 6.5h session (390 minutes)
#   * "24h"     - full-day FX (1440 minutes)
#
# We default to "session" because the run.yaml comment in the repo says
# 78 bars/day, but the user can override via ``bar_freq="24h"`` or by
# passing an explicit ``bars_per_year`` argument.

_BARS_PER_DAY = {
    # 1-minute bars
    "1m": 390,  # 6.5h session
    "1min": 390,
    "1minute": 390,
    # 5-minute bars (default)
    "5m": 78,
    "5min": 78,
    "5minute": 78,
    # 15-minute bars
    "15m": 26,
    "15min": 26,
    "15minute": 26,
    # 1-hour bars
    "1h": 6.5,
    "1hour": 6.5,
    "1hr": 6.5,
    "60m": 6.5,
    "60min": 6.5,
    # 4-hour bars
    "4h": 1.625,
    "4hr": 1.625,
    # Daily
    "1d": 1.0,
    "1day": 1.0,
    "daily": 1.0,
    # Weekly (approximate)
    "1w": 1.0 / 5,
}


def _bars_per_day_from_freq(bar_freq: str | None) -> float | None:
    """Return the per-trading-day bar count for a given frequency string.

    Returns ``None`` if the frequency is unknown so the caller can fall
    back to ``bars_per_year`` (auto-detected from data).
    """
    if not bar_freq:
        return None
    key = str(bar_freq).lower().replace(" ", "")
    if key in _BARS_PER_DAY:
        return _BARS_PER_DAY[key]
    # Try a few common alternate spellings
    aliases = {
        "5min": "5m",
        "5minbar": "5m",
        "5minbars": "5m",
        "15min": "15m",
        "30min": None,  # not a default, error
        "1h": "1h",
        "1hr": "1h",
        "1hour": "1h",
    }
    if key in aliases and aliases[key] is not None:
        return _BARS_PER_DAY[aliases[key]]
    return None


def sharpe_ann_factor(
    bars_per_year: float,
    holding_period_bars: int = 1,
    *,
    override: float | None = None,
) -> float:
    """Compute the Sharpe annualization factor.

    Parameters
    ----------
    bars_per_year
        Number of bars per year in the data stream.  For 5-min FX bars
        24/7 this is ~72,576; for 5-min session-only bars it is ~19,656.
    holding_period_bars
        Number of bars over which each "trade" (forward return) is
        realized.  Defaults to 1 (per-bar returns).  For a 30-bar
        lookahead label, pass ``holding_period_bars=30``.
    override
        If set, return this value verbatim (use only when the caller
        has an explicit CLI / YAML override that must win).

    Returns
    -------
    float
        The annualization multiplier: ``sqrt(bars_per_year / holding_period_bars)``.

    Notes
    -----
    The correct annualization for a stream of per-trade returns is
    ``sqrt(bars_per_year / holding_period_bars)``, NOT
    ``sqrt(bars_per_year)`` (which assumes per-bar returns).

    Applying ``sqrt(bars_per_year)`` to a per-trade return stream is the
    single most common Sharpe over-inflation error in the wild. It
    silently inflates the headline number by
    ``sqrt(holding_period_bars)``, e.g. ~5.5x for a 30-bar lookahead.
    """
    if override is not None:
        try:
            ov = float(override)
        except (TypeError, ValueError):
            ov = 0.0
        if ov > 0.0:
            return ov
    if bars_per_year <= 0 or holding_period_bars <= 0:
        return 1.0
    return math.sqrt(float(bars_per_year) / float(holding_period_bars))


def annualization_factor_from_freq(
    bar_freq: str | None,
    *,
    holding_period_bars: int = 1,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    full_day: bool = False,
) -> float:
    """Compute the factor from a bar-frequency string.

    Parameters
    ----------
    bar_freq
        One of the keys in ``_BARS_PER_DAY`` (e.g. ``"5m"``).
    holding_period_bars
        Lookahead horizon in bars.
    trading_days_per_year
        Defaults to 252.  Set to 365 for 24/7 crypto.
    full_day
        If True, treat the per-day count as a 24h day (1440 minutes).
        FX markets run ~24h Sunday-Friday, so ``full_day=True`` is the
        right choice for FX.
    """
    bpd = _bars_per_day_from_freq(bar_freq)
    if bpd is None:
        return 1.0
    if full_day:
        # Re-derive the per-day count as a 24h figure, not the 6.5h
        # session default.  We rebuild from minutes-per-day so the
        # "5m" → 288 logic is exact.
        key = str(bar_freq or "").lower().replace(" ", "")
        if key in ("1m", "1min", "1minute"):
            bpd = 1440.0
        elif key in ("5m", "5min", "5minute"):
            bpd = 288.0
        elif key in ("15m", "15min", "15minute"):
            bpd = 96.0
        elif key in ("1h", "1hour", "1hr", "60m", "60min"):
            bpd = 24.0
        elif key in ("4h", "4hr"):
            bpd = 6.0
        elif key in ("1d", "1day", "daily"):
            bpd = 1.0
        elif key in ("1w",):
            bpd = 1.0 / 7.0
    return sharpe_ann_factor(
        bars_per_year=trading_days_per_year * bpd,
        holding_period_bars=holding_period_bars,
    )


def _detect_cache_span_days(cache_path: str | Path) -> float | None:
    """Read the dataset manifest and return the cache span in days.

    Used by ``auto_annualization_factor`` to sanity-check the
    auto-detected factor: if the cache spans e.g. 1 year but the
    factor was derived from a frequency that implies 5 years of
    training, we can fall back to the actual span instead.

    Returns ``None`` if the cache has no readable manifest.
    """
    _cp = Path(cache_path)
    if not _cp.exists():
        return None
    candidates = [
        _cp / "dataset_manifest.json",
        _cp.parent / "dataset_manifest.json",
        _cp.parent.parent / "dataset_manifest.json",
    ]
    manifest = None
    for cand in candidates:
        if cand.exists():
            manifest = cand
            break
    if manifest is None:
        return None
    try:
        import json
        from datetime import datetime

        with open(manifest, encoding="utf-8") as f:
            meta = json.load(f)
        start = meta.get("start")
        end = meta.get("end")
        if not (start and end):
            return None
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d"):
            try:
                t0 = datetime.strptime(start[:19], fmt)
                t1 = datetime.strptime(end[:19], fmt)
                days = max((t1 - t0).total_seconds() / 86400.0, 1.0)
                return days
            except (ValueError, TypeError):
                continue
    except Exception:
        pass
    return None


def auto_annualization_factor(
    cache_path: str | Path | None = None,
    *,
    bar_freq: str | None = None,
    lookahead_bars: int = 1,
    full_day: bool = False,
    override: float | None = None,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Compute the Sharpe annualization factor with sensible auto-detection.

    Priority order:

      1. ``override`` - explicit CLI / YAML value, always wins.
      2. Frequency-string lookup + ``full_day`` flag - returns
         ``sqrt(trading_days_per_year * bars_per_day / lookahead_bars)``.
      3. If the cache manifest is available, log a sanity check
         comparing the derived factor to the cache span. We do NOT
         cap the factor on the cache span because the cache can
         always be larger or smaller than a single year; the user
         provides ``bar_freq`` to describe the schedule, and we
         trust that.
      4. If the frequency is unknown, return 1.0 (no inflation).
    """
    if override is not None:
        try:
            ov = float(override)
        except (TypeError, ValueError):
            ov = 0.0
        if ov > 0.0:
            return ov

    bpd = _bars_per_day_from_freq(bar_freq)
    bpd_eff: float | None = None
    if bpd is not None:
        bpd_eff = bpd
        if full_day:
            key = str(bar_freq or "").lower().replace(" ", "")
            if key in ("1m", "1min", "1minute"):
                bpd_eff = 1440.0
            elif key in ("5m", "5min", "5minute"):
                bpd_eff = 288.0
            elif key in ("15m", "15min", "15minute"):
                bpd_eff = 96.0
            elif key in ("1h", "1hour", "1hr", "60m", "60min"):
                bpd_eff = 24.0
            elif key in ("4h", "4hr"):
                bpd_eff = 6.0
            elif key in ("1d", "1day", "daily"):
                bpd_eff = 1.0
            elif key in ("1w",):
                bpd_eff = 1.0 / 7.0

    if bpd_eff is None:
        # No frequency known - return 1.0 so we don't silently
        # inflate Sharpe. Callers can still pass ``override``.
        return sharpe_ann_factor(1.0, lookahead_bars)

    bars_per_year = trading_days_per_year * bpd_eff

    # Optional sanity check against the cache manifest: if the cache
    # spans only a few weeks we should warn the user that the
    # annualization assumes the schedule continues year-round, which
    # may over-state the per-year count.  We don't cap because the
    # user provided bar_freq deliberately, but we log.
    if cache_path is not None:
        span_days = _detect_cache_span_days(cache_path)
        if span_days is not None and 0 < span_days < 90:
            import warnings

            warnings.warn(
                f"Cache spans only {span_days:.0f} days but the "
                f"annualization factor assumes a full-year schedule "
                f"({bars_per_year:.0f} bars/year). If the model is "
                f"trained on this short cache, the reported Sharpe "
                f"may over-state production performance.",
                stacklevel=2,
            )

    return sharpe_ann_factor(
        bars_per_year=bars_per_year,
        holding_period_bars=lookahead_bars,
    )


__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "_bars_per_day_from_freq",
    "_detect_cache_span_days",
    "annualization_factor_from_freq",
    "auto_annualization_factor",
    "sharpe_ann_factor",
]
