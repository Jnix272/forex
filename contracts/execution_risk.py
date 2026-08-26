"""
contracts/execution_risk.py
============================
Shared execution-risk value types used by both ``risk/`` and ``trading/``.

This module exists to break the risk <-> trading mutual coupling: it holds the
pure dataclasses / constants that both sides need, with no imports from either
package. Behavioral modules stay where they are; only shared vocabulary lives
here.
"""

from __future__ import annotations

from typing import Any

# Default "do nothing" action code shared by guards and the live engine.
HOLD = 1


from dataclasses import dataclass


@dataclass
class GuardResult:
    """Outcome of a pre-trade guard check (blocked / reason / sizing hints)."""

    blocked: bool
    reason: str = ""
    details: dict[str, Any] | None = None
    action: int = HOLD
    size_multiplier: float = 1.0
    confidence_threshold: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocked": bool(self.blocked),
            "reason": self.reason,
            "details": self.details or {},
            "action": int(self.action),
            "size_multiplier": float(self.size_multiplier),
            "confidence_threshold": float(self.confidence_threshold),
        }
