"""
trading/session_utils.py
=========================
Single source of truth for FX session classification (DST-aware).

Used by risk limits, live guards, ingestion overlap flags, and fill-cost
multipliers so paper / backtest / live share the same vocabulary:

  Primary (exclusive):  asia | london | ny | off
  Overlaps (flags):     asia_london | london_ny
  Policy key:           overlaps preferred over primary (see policy_key)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore


# Local session open/close times (stable across DST; UTC walls shift).
_SESSION_LOCAL_HOURS = {
    "asia": (time(9, 0), time(18, 0), "Asia/Tokyo"),
    "london": (time(8, 0), time(16, 30), "Europe/London"),
    "ny": (time(9, 30), time(16, 0), "America/New_York"),
}

# Production policy / limit / cost keys (no private "overlap" alias).
PRODUCTION_SESSION_KEYS = (
    "asia",
    "london",
    "ny",
    "asia_london",
    "london_ny",
    "off",
)

# Legacy → production name map (slippage calibrator, old risk "overlap", etc.).
_SESSION_ALIASES = {
    "overlap": "london_ny",
    "london/ny": "london_ny",
    "london-ny": "london_ny",
    "asia/london": "asia_london",
    "asia-london": "asia_london",
    "tokyo": "asia",
    "tokyo_asia": "asia",
    "sydney": "asia",  # APAC thin → asia bucket
    "overnight": "off",
    "off_hours": "off",
    "after_hours": "off",
}

# Fallback spread/cost scales when LABEL_REGIME is unavailable (mirrors settings).
_DEFAULT_SESSION_SPREAD_MULT = {
    "asia": 1.2,
    "london": 0.9,
    "ny": 0.9,
    "asia_london": 1.0,
    "london_ny": 0.85,
    "off": 1.5,
}


def normalize_session_name(session: str | None) -> str:
    """Map legacy session strings onto production keys."""
    if session is None:
        return "off"
    s = str(session).strip().lower().replace(" ", "_")
    if not s:
        return "off"
    return _SESSION_ALIASES.get(s, s)


@dataclass(frozen=True)
class SessionInfo:
    """DST-aware session snapshot at a single UTC instant."""

    primary: str
    asia_london: bool
    london_ny: bool
    active: frozenset[str]

    @property
    def policy_key(self) -> str:
        """Prefer overlap flags over exclusive primary (labeling / risk / fills)."""
        if self.asia_london:
            return "asia_london"
        if self.london_ny:
            return "london_ny"
        return self.primary or "off"


def get_session_hours_utc(session: str, date: datetime | None = None) -> tuple[int, int]:
    """Get current UTC hour boundaries for a trading session, accounting for DST.

    Args:
        session: one of 'asia', 'london', 'ny', 'off'
        date: reference date (defaults to now)

    Returns:
        (start_hour_utc, end_hour_utc) tuple
    """
    if session == "off":
        return (21, 24)

    if session not in _SESSION_LOCAL_HOURS:
        return (0, 24)

    local_open, local_close, tz_name = _SESSION_LOCAL_HOURS[session]
    tz = ZoneInfo(tz_name)

    if date is None:
        date = datetime.now(UTC)

    ref_date = date.date() if hasattr(date, "date") else date

    from datetime import datetime as dt

    open_local = dt.combine(ref_date, local_open, tzinfo=tz)
    close_local = dt.combine(ref_date, local_close, tzinfo=tz)

    open_utc = open_local.astimezone(UTC).hour
    close_utc = close_local.astimezone(UTC).hour

    if close_utc <= open_utc:
        close_utc += 24

    return (open_utc, close_utc)


def _in_local_window(dt_utc: datetime, session: str) -> bool:
    """True when ``dt_utc`` falls inside the session's local open/close window."""
    if session not in _SESSION_LOCAL_HOURS:
        return False
    local_open, local_close, tz_name = _SESSION_LOCAL_HOURS[session]
    local = dt_utc.astimezone(ZoneInfo(tz_name))
    t = local.time()
    return local_open <= t <= local_close


def classify_session(dt_utc: datetime | None = None) -> SessionInfo:
    """DST-aware session classification - shared SoT for risk / live / fills."""
    if dt_utc is None:
        dt_utc = datetime.now(UTC)
    elif dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=UTC)
    else:
        dt_utc = dt_utc.astimezone(UTC)

    active = {name for name in ("asia", "london", "ny") if _in_local_window(dt_utc, name)}
    asia_london = "asia" in active and "london" in active
    london_ny = "london" in active and "ny" in active

    # Exclusive primary: last writer wins in ingestion order asia→london→ny;
    # prefer liquid western sessions when multiple are active.
    if "ny" in active and not london_ny:
        primary = "ny"
    elif "london" in active:
        primary = "london"
    elif "ny" in active:
        primary = "ny"
    elif "asia" in active:
        primary = "asia"
    else:
        primary = "off"

    # During london_ny overlap, exclusive label follows ingestion (ny overwrites).
    if london_ny:
        primary = "ny"
    elif asia_london:
        primary = "london"

    return SessionInfo(
        primary=primary,
        asia_london=asia_london,
        london_ny=london_ny,
        active=frozenset(active),
    )


def get_current_session(dt_utc: datetime | None = None) -> str:
    """Exclusive primary session label (asia/london/ny/off)."""
    return classify_session(dt_utc).primary


def resolve_session_policy_key(
    session: str | None = None,
    *,
    asia_london: bool | float = False,
    london_ny: bool | float = False,
    now: datetime | None = None,
) -> str:
    """Policy lookup key: overlaps → primary; optional ``now`` fills gaps."""
    if asia_london:
        return "asia_london"
    if london_ny:
        return "london_ny"
    if session is not None:
        return normalize_session_name(session)
    return classify_session(now).policy_key


def is_session_active(session: str, dt_utc: datetime | None = None) -> bool:
    """Check if a specific session window is currently active (DST-aware)."""
    info = classify_session(dt_utc)
    key = normalize_session_name(session)
    if key in ("asia", "london", "ny"):
        return key in info.active
    if key == "london_ny":
        return info.london_ny
    if key == "asia_london":
        return info.asia_london
    if key == "off":
        return info.primary == "off"
    return info.primary == key


def session_spread_mult(
    session: str | None = None,
    *,
    asia_london: bool | float = False,
    london_ny: bool | float = False,
    now: datetime | None = None,
    scales: dict | None = None,
) -> float:
    """Shared session → spread/cost multiplier for fills and slippage.

    Reads ``LABEL_REGIME.session_cost_scale`` when available so RL costs,
    backtest fills, and slippage calibrator share one table.
    """
    key = resolve_session_policy_key(
        session,
        asia_london=asia_london,
        london_ny=london_ny,
        now=now,
    )
    table = scales
    if table is None:
        try:
            from config.settings import LABEL_REGIME as _LR

            table = _LR.get("session_cost_scale") or _DEFAULT_SESSION_SPREAD_MULT
        except Exception:
            table = _DEFAULT_SESSION_SPREAD_MULT
    return float(table.get(key, 1.0))


def default_session_slip_factors(scales: dict | None = None) -> dict[str, float]:
    """Slippage session factors relative to ``london_ny`` (tightest = 1.0)."""
    mults = dict(_DEFAULT_SESSION_SPREAD_MULT)
    if scales:
        mults.update({normalize_session_name(k): float(v) for k, v in scales.items()})
    else:
        try:
            from config.settings import LABEL_REGIME as _LR

            raw = _LR.get("session_cost_scale") or {}
            mults.update({normalize_session_name(k): float(v) for k, v in raw.items()})
        except Exception:
            pass
    ref = float(mults.get("london_ny", 0.85)) or 0.85
    return {k: float(v) / ref for k, v in mults.items() if k in PRODUCTION_SESSION_KEYS}
