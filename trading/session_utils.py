"""
trading/session_utils.py
=========================
PIPE-005: DST-aware session window computation.

Replaces hardcoded UTC hour ranges with timezone-aware boundaries that
correctly shift during Daylight Saving Time transitions.
"""

from datetime import UTC, datetime, time

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo


# Local session open/close times (these DON'T change with DST)
_SESSION_LOCAL_HOURS = {
    "asia":   (time(9, 0), time(18, 0), "Asia/Tokyo"),
    "london": (time(8, 0), time(16, 30), "Europe/London"),
    "ny":     (time(9, 30), time(16, 0), "America/New_York"),
}


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

    # Create timezone-aware datetimes at the local session times
    from datetime import datetime as dt
    open_local = dt.combine(ref_date, local_open, tzinfo=tz)
    close_local = dt.combine(ref_date, local_close, tzinfo=tz)

    # Convert to UTC hours
    open_utc = open_local.astimezone(UTC).hour
    close_utc = close_local.astimezone(UTC).hour

    # Handle day wrap
    if close_utc <= open_utc:
        close_utc += 24

    return (open_utc, close_utc)


def get_current_session(dt_utc: datetime | None = None) -> str:
    """Determine which trading session is currently active."""
    if dt_utc is None:
        dt_utc = datetime.now(UTC)

    hour = dt_utc.hour

    for session in ("london", "ny", "asia"):
        start, end = get_session_hours_utc(session, dt_utc)
        if start <= hour < (end % 24 if end > 24 else end):
            return session
        if end > 24 and (hour >= start or hour < end - 24):
            return session

    return "off"


def is_session_active(session: str, dt_utc: datetime | None = None) -> bool:
    """Check if a specific session is currently active."""
    return get_current_session(dt_utc) == session
