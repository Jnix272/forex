"""
Live trading guard rails.

These guards run after features are built and before broker order submission.
They are deliberately deterministic and auditable: every block returns a reason
that can be written to the live journal.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
try:
    import polars as pl
except ImportError:  # pragma: no cover - optional for legacy callers
    pl = None

from data.historical_news import _filter_relevant, _load_events


HOLD = 1
_SPECIAL_EVENTS = ("nfp", "nonfarm", "non-farm", "cpi", "fomc", "ecb", "boe", "boj", "rate")


def _is_polars_frame(frame: Any) -> bool:
    return pl is not None and isinstance(frame, pl.DataFrame)


def _last_numeric(features: Any, col: str, default: float = 0.0) -> float:
    if col not in features.columns or len(features) == 0:
        return float(default)
    try:
        if _is_polars_frame(features):
            value = features.select(pl.col(col).tail(1)).item()
        else:
            value = pd.to_numeric(features[col], errors="coerce").iloc[-1]
        return float(value)
    except Exception:
        return float(default)


def _tail_median(features: Any, col: str, lookback: int, default: float = 0.0) -> float:
    if col not in features.columns or len(features) == 0:
        return float(default)
    try:
        if _is_polars_frame(features):
            value = features.select(pl.col(col).tail(int(lookback)).median()).item()
        else:
            value = features[col].tail(int(lookback)).median()
        return float(value)
    except Exception:
        return float(default)


@dataclass
class GuardResult:
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


class EconomicCalendarGuard:
    def __init__(
        self,
        pair: str,
        *,
        calendar_file: Optional[str] = None,
        block_before_min: int = 0,
        block_after_min: int = 0,
        special_before_min: int = 2,
        special_after_min: int = 2,
        flatten_before_event: bool = False,
    ):
        self.pair = pair
        self.calendar_file = calendar_file or os.getenv("LIVE_ECONOMIC_CALENDAR_FILE") or os.getenv("ECONOMIC_CALENDAR_FILE")
        self.block_before_min = int(block_before_min)
        self.block_after_min = int(block_after_min)
        self.special_before_min = int(special_before_min)
        self.special_after_min = int(special_after_min)
        self.flatten_before_event = bool(flatten_before_event)
        self._events = pd.DataFrame()
        self._loaded_at = pd.Timestamp(0, tz="UTC")

    def _refresh(self, now: pd.Timestamp) -> None:
        if (now - self._loaded_at).total_seconds() < 60:
            return
        raw = _load_events(None, self.calendar_file)
        self._events = _filter_relevant(raw, now - pd.Timedelta(days=1), now + pd.Timedelta(days=7), self.pair)
        self._loaded_at = now

    def check(self, now=None) -> GuardResult:
        now_ts = pd.Timestamp(now or pd.Timestamp.utcnow())
        if now_ts.tzinfo is None:
            now_ts = now_ts.tz_localize("UTC")
        else:
            now_ts = now_ts.tz_convert("UTC")
        self._refresh(now_ts)
        if self._events.empty:
            return GuardResult(False, details={"events_loaded": 0})

        for _, row in self._events.iterrows():
            event_time = pd.Timestamp(row["timestamp_utc"]).tz_convert("UTC")
            name = str(row.get("headline", row.get("event", "")))
            low_name = name.lower()
            special = any(token in low_name for token in _SPECIAL_EVENTS)
            before = self.special_before_min if special else self.block_before_min
            after = self.special_after_min if special else self.block_after_min
            start = event_time - pd.Timedelta(minutes=before)
            end = event_time + pd.Timedelta(minutes=after)
            if start <= now_ts <= end:
                return GuardResult(
                    True,
                    reason="economic_calendar_block",
                    details={
                        "event": name,
                        "currency": str(row.get("currency", "")),
                        "impact": str(row.get("impact", "")),
                        "event_time": event_time.isoformat(),
                        "minutes_to_event": (event_time - now_ts).total_seconds() / 60.0,
                        "flatten_before_event": self.flatten_before_event,
                    },
                )
        return GuardResult(False, details={"events_loaded": int(len(self._events))})


class SpreadVolatilityGuard:
    def __init__(
        self,
        *,
        max_spread_pips: float = 2.5,
        spread_median_mult: float = 2.5,
        atr_median_mult: float = 3.0,
        vol_median_mult: float = 3.0,
        lookback: int = 60,
    ):
        self.max_spread_pips = float(max_spread_pips)
        self.spread_median_mult = float(spread_median_mult)
        self.atr_median_mult = float(atr_median_mult)
        self.vol_median_mult = float(vol_median_mult)
        self.lookback = int(lookback)

    @staticmethod
    def _last(features, col: str, default: float = 0.0) -> float:
        return _last_numeric(features, col, default)

    def check(self, features, *, bid: Optional[float] = None, ask: Optional[float] = None) -> GuardResult:
        spread = self._last(features, "spread_pips", 0.0)
        if bid and ask and bid > 0 and ask > bid:
            spread = max(spread, (ask - bid) * 10_000)
        if spread > self.max_spread_pips:
            return GuardResult(True, "spread_too_wide", {"spread_pips": spread, "max_spread_pips": self.max_spread_pips})

        if "spread_pips" in features.columns and len(features) >= self.lookback:
            med = _tail_median(features, "spread_pips", self.lookback)
            if med > 0 and spread > self.spread_median_mult * med:
                return GuardResult(True, "spread_spike", {"spread_pips": spread, "median": med, "mult": spread / med})

        atr_cols = [c for c in features.columns if str(c).startswith("atr_")]
        if atr_cols and len(features) >= self.lookback:
            atr_col = atr_cols[0]
            atr = self._last(features, atr_col)
            med = _tail_median(features, atr_col, self.lookback)
            if med > 0 and atr > self.atr_median_mult * med:
                return GuardResult(True, "atr_spike", {"atr": atr, "median": med, "mult": atr / med})

        if "vol_20" in features.columns and len(features) >= self.lookback:
            vol = self._last(features, "vol_20")
            med = _tail_median(features, "vol_20", self.lookback)
            if med > 0 and vol > self.vol_median_mult * med:
                return GuardResult(True, "volatility_unstable", {"vol_20": vol, "median": med, "mult": vol / med})

        return GuardResult(False, details={"spread_pips": spread})


class RegimeRouter:
    def __init__(self, *, rollover_start_utc: int = 21, rollover_end_utc: int = 1):
        self.rollover_start_utc = int(rollover_start_utc)
        self.rollover_end_utc = int(rollover_end_utc)

    def route(self, features, *, now=None, calendar_blocked: bool = False) -> GuardResult:
        now_ts = pd.Timestamp(now or pd.Timestamp.utcnow())
        hour = int(now_ts.hour)
        if calendar_blocked:
            return GuardResult(True, "news_block", {"regime": "news_block"}, size_multiplier=0.0, confidence_threshold=1.0)
        in_rollover = hour >= self.rollover_start_utc or hour <= self.rollover_end_utc
        if in_rollover:
            return GuardResult(True, "rollover", {"regime": "rollover", "hour_utc": hour}, size_multiplier=0.0, confidence_threshold=1.0)
        if "regime_break_prob" in features.columns:
            rbp = _last_numeric(features, "regime_break_prob")
            if rbp >= 0.75:
                return GuardResult(False, "high_vol", {"regime": "high_vol", "regime_break_prob": rbp}, size_multiplier=0.5, confidence_threshold=0.65)
        return GuardResult(False, "normal", {"regime": "normal"}, size_multiplier=1.0, confidence_threshold=0.45)


class DisagreementGate:
    def __init__(self, *, min_confidence: float = 0.45, enabled: bool = True):
        self.min_confidence = float(min_confidence)
        self.enabled = bool(enabled)

    @staticmethod
    def _safe_action(model, obs) -> Optional[int]:
        try:
            return int(model.select_action(obs))
        except Exception:
            return None

    def check(self, action: int, obs, *, fast_model=None, slow_model=None, confidence: Optional[float] = None) -> GuardResult:
        if confidence is not None and float(confidence) < self.min_confidence:
            return GuardResult(True, "low_confidence", {"confidence": float(confidence), "min_confidence": self.min_confidence})
        if not self.enabled or fast_model is None or slow_model is None:
            return GuardResult(False, details={"confidence": confidence})
        fast_action = self._safe_action(fast_model, obs)
        slow_action = self._safe_action(slow_model, obs)
        votes = [a for a in (int(action), fast_action, slow_action) if a is not None]
        disagreement = len(set(votes)) > 1
        if disagreement and int(action) != HOLD:
            return GuardResult(True, "model_disagreement", {"action": int(action), "fast": fast_action, "slow": slow_action})
        return GuardResult(False, details={"action": int(action), "fast": fast_action, "slow": slow_action, "confidence": confidence})


class NoTradeZoneGate:
    """
    Learned/heuristic no-trade gate (Improvement #7).

    Blocks orders when the feature frame carries a ``no_trade_score`` at or
    above ``threshold`` (0-1, 1 = strong no-trade signal). Falls back to the
    offline heuristic scorer when the column is absent, so the gate works on
    raw feature frames too. Disabled by default to preserve existing behavior.
    """

    def __init__(self, *, threshold: float = 0.70, enabled: bool = False):
        self.threshold = float(threshold)
        self.enabled = bool(enabled)

    def check(self, features: Any) -> GuardResult:
        if not self.enabled:
            return GuardResult(False, "no_trade_disabled", {"enabled": False})
        score = None
        if "no_trade_score" in features.columns:
            try:
                if _is_polars_frame(features):
                    score = float(features.select(pl.col("no_trade_score").tail(1)).item())
                else:
                    score = float(pd.to_numeric(features["no_trade_score"], errors="coerce").iloc[-1])
            except Exception:
                score = None
        if score is None:
            score = self._heuristic(features)
        if score is None:
            return GuardResult(False, "no_trade_unavailable", {"score": None})
        if score >= self.threshold:
            return GuardResult(True, "no_trade_zone", {"no_trade_score": float(score), "threshold": self.threshold}, size_multiplier=0.0, confidence_threshold=1.0)
        return GuardResult(False, "no_trade_ok", {"no_trade_score": float(score)})

    @staticmethod
    def _heuristic(features: Any) -> Optional[float]:
        try:
            from features.no_trade_zones import compute_heuristic_no_trade_score
            cols = ["atr_6", "spread_pips", "adx_14", "rsi_14"]
            if _is_polars_frame(features):
                present = [c for c in cols if c in features.columns]
                if not present:
                    return None
                pdf = features.select(present).to_pandas()
            else:
                present = [c for c in cols if c in features.columns]
                if not present:
                    return None
                pdf = features[present]
            vals = compute_heuristic_no_trade_score(pdf)
            return float(vals[-1]) if len(vals) else None
        except Exception:
            return None


class TradeJournal:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, payload: dict[str, Any]) -> None:
        safe = json.loads(json.dumps(payload, default=str))
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(safe, separators=(",", ":")) + "\n")
