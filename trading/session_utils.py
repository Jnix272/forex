"""
trading/session_utils.py
========================
Compatibility shim: the implementation moved to ``contracts/session_utils.py``
to break the risk -> trading coupling. Re-exports everything so existing
imports keep working.
"""

from contracts.session_utils import (  # noqa: F401
    PRODUCTION_SESSION_KEYS,
    SessionInfo,
    classify_session,
    default_session_slip_factors,
    get_current_session,
    get_session_hours_utc,
    is_session_active,
    normalize_session_name,
    resolve_session_policy_key,
    session_spread_mult,
)

__all__ = [
    "PRODUCTION_SESSION_KEYS",
    "SessionInfo",
    "classify_session",
    "default_session_slip_factors",
    "get_current_session",
    "get_session_hours_utc",
    "is_session_active",
    "normalize_session_name",
    "resolve_session_policy_key",
    "session_spread_mult",
]
