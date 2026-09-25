"""
Live trading guard rails.

These guards run after features are built and before broker order submission.
They are deliberately deterministic and auditable: every block returns a reason
that can be written to the live journal.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional for legacy callers
    pl = None

from config.settings import price_to_pips
from data.historical_news import _filter_relevant, _load_events

# Shared guard value types now live in contracts (risk <-> trading decoupling).
from contracts.execution_risk import GuardResult, HOLD  # noqa: F401  # re-export

_SPECIAL_EVENT_REGEX = re.compile(
    r"\b(nfp|nonfarm|non-farm|cpi|fomc|ecb|boe|boj|interest rate|fed|central bank|gdp|retail sales|retail|pmi|ism)\b",
    re.IGNORECASE,
)


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


class EconomicCalendarGuard:
    # Tiered windows: critical (NFP) 60/30 flatten, high (CPI/GDP/Retail/FOMC) 30/15, medium 15/10
    TIER_WINDOWS = {
        "critical": (60, 30, True),
        "high": (30, 15, False),
        "medium": (15, 10, False),
        "low": (0, 0, False),
    }

    def __init__(
        self,
        pair: str,
        *,
        calendar_file: str | None = None,
        block_before_min: int = 15,
        block_after_min: int = 10,
        special_before_min: int = 30,
        special_after_min: int = 15,
        flatten_before_event: bool = False,
        tier_windows: dict | None = None,
    ):
        self.pair = pair
        self.calendar_file = (
            calendar_file or os.getenv("LIVE_ECONOMIC_CALENDAR_FILE") or os.getenv("ECONOMIC_CALENDAR_FILE")
        )
        self.block_before_min = int(block_before_min)
        self.block_after_min = int(block_after_min)
        self.special_before_min = int(special_before_min)
        self.special_after_min = int(special_after_min)
        self.flatten_before_event = bool(flatten_before_event)
        # Allow run.yaml news.tiers override: {"critical":[60,30],"high":[30,15]}
        if tier_windows:
            try:
                for k, v in tier_windows.items():
                    self.TIER_WINDOWS[str(k).lower()] = tuple(v)  # type: ignore
            except Exception:
                pass
        self._events = pd.DataFrame()
        self._loaded_at = pd.Timestamp(0, tz="UTC")

    def _refresh(self, now: pd.Timestamp) -> None:
        if (now - self._loaded_at).total_seconds() < 60:
            return
        start_ts = now - pd.Timedelta(days=1)
        end_ts = now + pd.Timedelta(days=7)
        raw = _load_events("", self.calendar_file, start_ts=start_ts, end_ts=end_ts)
        self._events = _filter_relevant(raw, start_ts, end_ts, self.pair)
        self._loaded_at = now

    def check(self, now=None) -> GuardResult:
        now_ts = pd.Timestamp(now or pd.Timestamp.utcnow())
        if now_ts.tzinfo is None:
            now_ts = now_ts.tz_localize("UTC")
        else:
            now_ts = now_ts.tz_convert("UTC")
        self._refresh(now_ts)
        if self._events is None or len(self._events) == 0:
            return GuardResult(False, details={"events_loaded": 0})

        events_df = self._events.to_pandas() if hasattr(self._events, "to_pandas") else self._events
        if getattr(events_df, "empty", True):
            return GuardResult(False, details={"events_loaded": 0})

        # Build sorted, merged windows to avoid double-counting overlapping 08:30 CPI + 08:50 FOMC
        candidates: list[tuple[pd.Timestamp, pd.Timestamp, dict]] = []
        for _, row in events_df.iterrows():
            ts = pd.Timestamp(row["timestamp_utc"])
            event_time = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            name = str(row.get("headline", row.get("event", "")))
            low_name = name.lower()
            impact_raw = str(row.get("impact", "")).strip().lower()
            # Tiered impact
            if impact_raw in ("critical", "3", "red") or "crit" in impact_raw:
                tier = "critical"
            elif impact_raw in ("high", "high impact") or "high" in impact_raw:
                tier = "high"
            elif impact_raw in ("medium", "moderate"):
                tier = "medium"
            else:
                tier = "low"
            if tier == "low":
                continue
            # Legacy special regex now only upgrades high→critical for NFP-style
            special = bool(_SPECIAL_EVENT_REGEX.search(low_name))
            if tier == "high" and special and "nfp" in low_name:
                tier = "critical"
            before, after, tier_flatten = self.TIER_WINDOWS.get(tier, (15, 10, False))
            # Env/config can still force flatten for critical only and only if position is in event currency
            flatten = bool(self.flatten_before_event or tier_flatten)
            # Currency-specific flatten: only flatten if engine pair holds the event currency
            # (ECB EUR → EURUSD/GBPUSD not USDJPY), otherwise just HOLD
            if flatten:
                ccy = str(row.get("currency", "")).upper().strip()
                if ccy and ccy not in self.pair.upper():
                    flatten = False
            if before == 0 and after == 0:
                continue
            start = event_time - pd.Timedelta(minutes=before)
            end = event_time + pd.Timedelta(minutes=after)
            candidates.append((start, end, {
                "event": name, "currency": str(row.get("currency","")), "impact": str(row.get("impact","")),
                "event_time": event_time.isoformat(), "tier": tier, "special": special,
                "flatten_before_event": flatten, "before": before, "after": after,
                "minutes_to_event": (event_time - now_ts).total_seconds()/60.0,
            }))
        if not candidates:
            return GuardResult(False, details={"events_loaded": len(self._events)})
        # Sort and merge overlapping windows for graduated re-entry (0.5× size next 15m)
        candidates.sort(key=lambda x: x[0])
        merged: list[tuple[pd.Timestamp, pd.Timestamp, dict]] = []
        for s, e, d in candidates:
            if not merged or s > merged[-1][1]:
                merged.append((s, e, d))
            else:
                # Overlap: extend end, keep earliest event details but mark merged
                prev_s, prev_e, prev_d = merged[-1]
                new_e = max(prev_e, e)
                prev_d = {**prev_d, "merged": True, "merged_count": int(prev_d.get("merged_count",1))+1}
                merged[-1] = (prev_s, new_e, prev_d)
        for s, e, d in merged:
            if s <= now_ts <= e:
                # Graduated re-entry: after event, size 0.5× for 15m already encoded via RegimeRouter; economics just blocks
                return GuardResult(True, reason="economic_calendar_block", details=d)
            # 15m graduated tail: 0.5× not blocked but logged for live_engine to size 0.5
            tail_end = e + pd.Timedelta(minutes=15)
            if e < now_ts <= tail_end:
                return GuardResult(False, details={**d, "graduated_tail": True, "size_multiplier": 0.5})
        return GuardResult(False, details={"events_loaded": len(self._events)})


class SpreadVolatilityGuard:
    def __init__(
        self,
        *,
        max_spread_pips: float | None = None,
        spread_median_mult: float | None = None,
        atr_median_mult: float = 3.0,
        vol_median_mult: float = 3.0,
        lookback: int = 60,
        pair: str = "EURUSD",
    ):
        pair_u = str(pair).upper()
        # Pair-specific defaults: CAD/JPY naturally wider (2.1pips normal for USDCAD vs 1.3 EURUSD)
        if "JPY" in pair_u:
            def_max, def_mult = 3.0, 3.0
        elif "CAD" in pair_u:
            def_max, def_mult = 3.0, 3.5
        else:
            def_max, def_mult = 2.5, 2.5
        self.max_spread_pips = float(max_spread_pips if max_spread_pips is not None else def_max)
        self.spread_median_mult = float(spread_median_mult if spread_median_mult is not None else def_mult)
        self.atr_median_mult = float(atr_median_mult)
        self.vol_median_mult = float(vol_median_mult)
        self.lookback = int(lookback)
        self.pair = str(pair).upper()

    @staticmethod
    def _last(features, col: str, default: float = 0.0) -> float:
        return _last_numeric(features, col, default)

    def check(self, features, *, bid: float | None = None, ask: float | None = None) -> GuardResult:
        spread = self._last(features, "spread_pips", 0.0)
        if bid and ask and bid > 0 and ask > bid:
            spread = max(spread, price_to_pips(float(ask) - float(bid), self.pair))
        if spread > self.max_spread_pips:
            return GuardResult(
                True, "spread_too_wide", {"spread_pips": spread, "max_spread_pips": self.max_spread_pips}
            )

        if "spread_pips" in features.columns and len(features) >= self.lookback:
            med = _tail_median(features, "spread_pips", self.lookback)
            if med > 0 and spread > self.spread_median_mult * med:
                return GuardResult(True, "spread_spike", {"spread_pips": spread, "median": med, "mult": spread / med})

        atr_cand = [c for c in ("atr_6", "atr_14", "atr_20") if c in features.columns]
        atr_col = atr_cand[0] if atr_cand else None
        if not atr_col:
            atr_cols = [c for c in features.columns if str(c).startswith("atr_") and not str(c).startswith("atr_ratio")]
            atr_col = atr_cols[0] if atr_cols else None
        if atr_col and len(features) >= self.lookback:
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
    def __init__(self, *, rollover_start_utc: int = 21, rollover_end_utc: int = 22):
        self.rollover_start_utc = int(rollover_start_utc)
        self.rollover_end_utc = int(rollover_end_utc)

    def route(self, features, *, now=None, calendar_blocked: bool = False) -> GuardResult:
        now_ts = pd.Timestamp(now or pd.Timestamp.utcnow())
        hour = int(now_ts.hour)
        if calendar_blocked:
            return GuardResult(
                True, "news_block", {"regime": "news_block"}, size_multiplier=0.0, confidence_threshold=1.0
            )
        if self.rollover_start_utc <= self.rollover_end_utc:
            in_rollover = self.rollover_start_utc <= hour < self.rollover_end_utc
        else:
            in_rollover = hour >= self.rollover_start_utc or hour < self.rollover_end_utc
        if in_rollover:
            return GuardResult(
                True,
                "rollover",
                {"regime": "rollover", "hour_utc": hour},
                size_multiplier=0.0,
                confidence_threshold=1.0,
            )
        if "regime_break_prob" in features.columns:
            rbp = _last_numeric(features, "regime_break_prob")
            if rbp >= 0.75:
                return GuardResult(
                    False,
                    "high_vol",
                    {"regime": "high_vol", "regime_break_prob": rbp},
                    size_multiplier=0.5,
                    confidence_threshold=0.65,
                )
        return GuardResult(False, "normal", {"regime": "normal"}, size_multiplier=1.0, confidence_threshold=0.45)


class DisagreementGate:
    def __init__(self, *, min_confidence: float = 0.45, enabled: bool = True):
        self.min_confidence = float(min_confidence)
        self.enabled = bool(enabled)

    @staticmethod
    def _safe_action(model, obs) -> int | None:
        try:
            if hasattr(model, "peek_raw"):
                return int(model.peek_raw(obs))
            return int(model.select_action(obs))
        except Exception:
            return None

    def check(
        self,
        action: int,
        obs,
        *,
        fast_model=None,
        slow_model=None,
        confidence: float | None = None,
        bypass_disagreement: bool = False,
        fast_action: int | None = None,
    ) -> GuardResult:
        if confidence is not None and float(confidence) < self.min_confidence:
            return GuardResult(
                True, "low_confidence", {"confidence": float(confidence), "min_confidence": self.min_confidence}
            )
        if not self.enabled or bypass_disagreement:
            return GuardResult(False, details={"confidence": confidence})
        if fast_action is not None:
            resolved_fast = int(fast_action)
        elif action is not None:
            resolved_fast = int(action)
        elif fast_model is not None:
            resolved_fast = self._safe_action(fast_model, obs)
        else:
            resolved_fast = None

        if slow_model is None:
            return GuardResult(False, details={"confidence": confidence})
        slow_action = self._safe_action(slow_model, obs)
        votes = [a for a in (int(action), resolved_fast, slow_action) if a is not None]
        disagreement = len(set(votes)) > 1
        if disagreement and int(action) != HOLD:
            return GuardResult(
                True, "model_disagreement", {"action": int(action), "fast": resolved_fast, "slow": slow_action}
            )
        return GuardResult(
            False, details={"action": int(action), "fast": resolved_fast, "slow": slow_action, "confidence": confidence}
        )


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
            return GuardResult(
                True,
                "no_trade_zone",
                {"no_trade_score": float(score), "threshold": self.threshold},
                size_multiplier=0.0,
                confidence_threshold=1.0,
            )
        return GuardResult(False, "no_trade_ok", {"no_trade_score": float(score)})

    @staticmethod
    def _heuristic(features: Any) -> float | None:
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
    def __init__(self, path: str | Path, db_sink: Any = None, default_pair: str = ""):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db_sink = db_sink
        self.default_pair = str(default_pair)

    def record(self, payload: dict[str, Any]) -> None:
        safe = json.loads(json.dumps(payload, default=str))
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(safe, separators=(",", ":")) + "\n")
        if self.db_sink is not None:
            try:
                self.db_sink.record_trade(
                    pair=str(safe.get("pair") or self.default_pair or ""),
                    event=str(safe.get("event") or ""),
                    action=str(safe.get("action") or safe.get("side") or ""),
                    lots=float(safe.get("lots") or safe.get("qty") or 0.0),
                    price=float(safe.get("price") or safe.get("mid") or safe.get("entry") or 0.0),
                    order_id=str(safe.get("order_id") or safe.get("id") or ""),
                    reason=str(safe.get("reason") or ""),
                    pnl=float(safe.get("pnl") or 0.0),
                    details=safe,
                )
            except Exception:
                pass
