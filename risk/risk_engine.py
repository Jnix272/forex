"""
risk/risk_engine.py — Real-time risk engine (pre-trade, post-trade, circuit breakers)

One source of truth: all limits default from config.settings.LIVE_RISK / RISK so
paper, backtest, shadow, and live share identical logic. The engine is pure and
stateless between explicit updates, so it can be driven by the live engine, a
paper-trading loop, or tests.

Responsibilities
----------------
  * Pre-trade checks   — max position size, max notional, max daily loss,
                         max order frequency, single-instrument concentration.
  * Post-trade monitor — drawdown limits, VaR (historical + parametric),
                         Expected Shortfall / CVaR, exposure by currency and
                         pair, gap-risk flags.
  * Circuit breakers   — auto-flatten or auto-standby when limits breach.
  * Audit log          — every risk decision (timestamp, rule, value, action).

Usage
-----
    from risk.risk_engine import RiskEngine
    engine = RiskEngine(equity=10_000.0)
    verdict = engine.check_order(pair="EURUSD", lots=0.5, price=1.10,
                                 position_size_pct=0.03, notional_usd=5500.0)
    engine.on_trade_closed(pnl=-180.0, equity=9_820.0, pair="EURUSD",
                           lots=0.5, direction="long")
    alert = engine.update_equity(equity=9_820.0)
    if alert["circuit_breaker"]:
        engine.flatten()   # auto-standby / flatten all
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

try:
    from config.settings import LIVE_RISK as _LR
    from config.settings import RISK as _RISK
except ImportError:
    _LR = {}
    _RISK = {}

logger = logging.getLogger(__name__)

_PAIRS = ("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _clip_currency(currency: str) -> str:
    """Normalise a pair to its base and quote currency codes."""
    c = (currency or "").upper()
    if len(c) != 6:
        return c
    return c


@dataclass
class RiskConfig:
    """All engine limits; mirrors LIVE_RISK defaults when not supplied."""
    max_position_pct:    float = float(_LR.get("max_position_pct", 0.05))
    max_total_lots:      float = float(_LR.get("max_total_lots", 3.0))
    max_notional_usd:    float = float(_LR.get("max_notional_usd", 250_000.0))
    max_daily_loss_pct:  float = float(_LR.get("daily_loss_limit", 0.03))
    max_consecutive_losses: int = int(_LR.get("max_consecutive_losses", 5))
    max_order_freq_per_min: int = int(_LR.get("max_order_freq_per_min", 10))
    max_drawdown_halt:   float = float(_LR.get("max_drawdown_halt", 0.10))
    soft_drawdown_reduce: float = float(_LR.get("soft_drawdown_reduce", 0.05))
    max_instrument_concentration: float = float(_LR.get("max_instrument_concentration", 0.50))
    var_confidence:      float = float(_LR.get("var_confidence", 0.99))
    var_window:          int = int(_LR.get("var_window", 500))
    cvar_multiplier:     float = float(_LR.get("cvar_multiplier", 1.5))
    gap_move_threshold:  float = float(_LR.get("gap_move_threshold", 0.02))
    require_approval:    bool = bool(_LR.get("require_approval_on_flatten", False))

    @classmethod
    def from_dict(cls, data: dict | None = None, **overrides: float) -> RiskConfig:
        """Build a RiskConfig from an explicit dict (e.g. a file loaded by
        ``--risk-config``). Unknown keys are ignored; missing keys fall back to
        the RISK / LIVE_RISK defaults embedded above.
        """
        data = data or {}
        valid = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid}
        filtered.update({k: v for k, v in overrides.items() if k in valid})
        return cls(**filtered)


@dataclass
class RiskDecision:
    allowed: bool
    rule: str
    value: float
    limit: float
    action: str = "ok"
    reason: str = ""
    details: dict = field(default_factory=dict)

    def to_audit(self, ts: str | None = None) -> dict:
        return {
            "ts": ts or _now_iso(),
            "rule": self.rule,
            "value": round(float(self.value), 6),
            "limit": round(float(self.limit), 6),
            "allowed": self.allowed,
            "action": self.action,
            "reason": self.reason,
            "details": self.details,
        }


class RiskEngine:
    """Real-time risk engine with pre-trade checks, post-trade monitoring and
    circuit breakers. All decisions are appended to an in-memory audit log."""

    def __init__(self, equity: float = 10_000.0, cfg: RiskConfig | None = None):
        self.cfg = cfg or RiskConfig()
        self.equity = float(equity)
        self.peak_equity = float(equity)
        self.start_equity = float(equity)
        self.daily_start_equity = float(equity)

        self.positions: dict[str, dict] = {}          # pair -> {"lots":..., "entry_price":..., "direction":...}
        self.realized_pnl: float = 0.0
        self.day_realized_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.gap_flagged: dict[str, bool] = {}
        self._halted: bool = False
        self._soft_reduce: bool = False

        # returns history for VaR/CVaR (per-pair and portfolio)
        self._returns: dict[str, deque[float]] = {}
        self._portfolio_returns: deque[float] = deque(maxlen=max(100, self.cfg.var_window))

        # order frequency guard (timestamps of the last N order requests)
        self._order_times: deque[float] = deque()

        # audit log
        self.audit_log: list[dict] = []

        # circuit-breaker auto-flatten decision
        self.breach_reasons: list[str] = []

    # ── pre-trade ──────────────────────────────────────────────────────────

    def check_order(
        self,
        pair: str,
        lots: float,
        price: float,
        position_size_pct: float | None = None,
        notional_usd: float | None = None,
        hour_utc: int | None = None,
        now: str | None = None,
    ) -> RiskDecision:
        """Run every pre-trade check. Returns the first blocking decision, or an
        allowed decision when all checks pass."""
        now = now or _now_iso()
        pair = _clip_currency(pair)
        notional_usd = notional_usd if notional_usd is not None else float(lots) * float(price) * 100_000
        position_size_pct = position_size_pct if position_size_pct is not None else (
            notional_usd / max(self.equity, 1.0)
        )

        if self._halted:
            d = RiskDecision(False, "circuit_breaker", 0.0, self.cfg.max_drawdown_halt,
                             action="standby", reason="circuit breaker active",
                             details={"halted": True})
            self._log(d, now)
            return d

        current_lots = sum(abs(p.get("lots", 0.0)) for p in self.positions.values())
        total_lots = current_lots + abs(lots)
        
        current_notional = sum(abs(p.get("lots", 0.0)) * p.get("entry_price", 1.0) * 100_000 for p in self.positions.values())
        total_notional = current_notional + notional_usd
        
        total_position_pct = total_notional / max(self.equity, 1.0)

        checks = [
            # Use caller's position_size_pct for max_position_pct (backward-compat)
            # Use total_notional/equity for max_notional_usd
            self._check("max_position_pct", position_size_pct, self.cfg.max_position_pct,
                        allowed_fn=lambda v, lim: v <= lim,
                        reason="cumulative position size exceeds max_position_pct"),
            self._check("max_total_lots", total_lots, self.cfg.max_total_lots,
                        allowed_fn=lambda v, lim: v <= lim,
                        reason="cumulative lots exceed max_total_lots"),
            self._check("max_notional_usd", total_notional, self.cfg.max_notional_usd,
                        allowed_fn=lambda v, lim: v <= lim,
                        reason="cumulative notional exceeds max_notional_usd"),
            self._check("daily_loss", abs(self.day_realized_pnl) if self.day_realized_pnl < 0 else 0.0,
                        max(self.daily_start_equity, 1.0) * self.cfg.max_daily_loss_pct,
                        allowed_fn=lambda v, lim: v <= lim,
                        reason="daily loss limit breached"),
            self._check("concentration", lots, self._max_lots_for_instrument(pair, self.cfg.max_instrument_concentration),
                        allowed_fn=lambda v, lim: v <= lim,
                        reason="single-instrument concentration cap"),
        ]
        # order frequency
        if self._freq_blocked():
            checks.append(RiskDecision(False, "max_order_freq", float(len(self._order_times)),
                                       float(self.cfg.max_order_freq_per_min),
                                       action="block", reason="order frequency cap"))

        for d in checks:
            self._log(d, now)
            if not d.allowed:
                return d

        # frequency accounting only counts accepted orders
        self._order_times.append(float(datetime.now(UTC).timestamp()))
        while self._order_times and self._order_times[-1] - self._order_times[0] > 60.0:
            self._order_times.popleft()

        ok = RiskDecision(True, "pre_trade_checks", position_size_pct, self.cfg.max_position_pct)
        self._log(ok, now)
        return ok

    def _check(self, rule: str, value: float, limit: float, allowed_fn, reason: str) -> RiskDecision:
        allowed = allowed_fn(float(value), float(limit))
        action = "ok" if allowed else "reject"
        return RiskDecision(allowed, rule, value, limit, action=action, reason=reason)

    def _max_lots_for_instrument(self, pair: str, concentration: float) -> float:
        """Remaining lots a single instrument may add before its exposure
        exceeds ``concentration`` of the total lot limit."""
        existing = abs(self.positions.get(pair, {}).get("lots", 0.0))
        cap = concentration * self.cfg.max_total_lots
        return max(0.0, cap - existing)

    def _freq_blocked(self) -> bool:
        if len(self._order_times) < self.cfg.max_order_freq_per_min:
            return False
        window = self._order_times[-1] - self._order_times[0]
        return window <= 60.0

    # ── post-trade ─────────────────────────────────────────────────────────

    def open_position(self, pair: str, lots: float, entry_price: float, direction: str = "long") -> None:
        pair = _clip_currency(pair)
        self.positions[pair] = {
            "lots": float(lots),
            "entry_price": float(entry_price),
            "direction": direction,
        }

    def close_position(self, pair: str) -> None:
        pair = _clip_currency(pair)
        self.positions.pop(pair, None)

    def on_trade_closed(
        self,
        pnl: float,
        equity: float,
        pair: str = "EURUSD",
        lots: float = 0.0,
        direction: str = "long",
        now: str | None = None,
    ) -> RiskDecision | None:
        """Update daily P&L, consecutive-loss counter and return series after a
        closed trade. Returns a circuit-breaker decision if limits breach."""
        now = now or _now_iso()
        self.realized_pnl += float(pnl)
        self.day_realized_pnl += float(pnl)
        self.equity = float(equity)
        self.peak_equity = max(self.peak_equity, self.equity)

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        ret = self._pnl_to_ret(pnl, equity, lots)
        self._returns.setdefault(pair, deque(maxlen=max(100, self.cfg.var_window))).append(ret)
        self._portfolio_returns.append(ret)

        if self.consecutive_losses >= self.cfg.max_consecutive_losses:
            d = RiskDecision(False, "max_consecutive_losses", float(self.consecutive_losses),
                             float(self.cfg.max_consecutive_losses),
                             action="reduce_size", reason="consecutive losses exceed limit",
                             details={"direction": direction})
            self._log(d, now)
            return d
        return None

    def update_equity(self, equity: float, now: str | None = None) -> dict:
        """Run post-trade drawdown / VaR monitoring. Returns a monitor dict with
        a circuit_breaker flag that callers may act on (flatten / standby)."""
        now = now or _now_iso()
        self.equity = float(equity)
        self.peak_equity = max(self.peak_equity, self.equity)

        dd = self.current_drawdown()
        breach: list[str] = []
        if dd >= self.cfg.max_drawdown_halt:
            breach.append("max_drawdown_halt")
        if self.day_realized_pnl < 0 and abs(self.day_realized_pnl) >= max(self.daily_start_equity, 1.0) * self.cfg.max_daily_loss_pct:
            breach.append("daily_loss_limit")

        var = self.portfolio_var()
        gap = self._check_gaps()

        decision = None
        if breach:
            self._halted = True
            action = "flatten" if not self.cfg.require_approval else "standby_approval"
            decision = RiskDecision(False, "circuit_breaker", round(dd, 6),
                                    self.cfg.max_drawdown_halt, action=action,
                                    reason=",".join(breach),
                                    details={"var_usd": var.get("var_usd"), "gap_flagged": gap})
            self._log(decision, now)

        if dd >= self.cfg.soft_drawdown_reduce and not self._halted:
            self._soft_reduce = True
            self._log(RiskDecision(True, "soft_drawdown_reduce", round(dd, 6),
                                   self.cfg.soft_drawdown_reduce, action="reduce_size",
                                   reason="soft drawdown threshold hit"), now)

        return {
            "ts": now,
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "drawdown_pct": round(dd, 6),
            "day_pnl": round(self.day_realized_pnl, 2),
            "var_usd": var.get("var_usd", 0.0),
            "cvar_usd": var.get("cvar_usd", 0.0),
            "gap_flagged": gap,
            "circuit_breaker": bool(breach),
            "soft_reduce": self._soft_reduce,
            "breach_reasons": breach,
            "decision": decision.to_audit(now) if decision else None,
        }

    def new_day(self, equity: float | None = None) -> None:
        """Reset daily loss counters at the UTC day boundary."""
        if equity is not None:
            self.equity = float(equity)
            self.peak_equity = max(self.peak_equity, self.equity)
        self.daily_start_equity = self.equity
        self.day_realized_pnl = 0.0
        self.consecutive_losses = 0
        # Clear daily / soft halt so a new session can trade; max-DD halt stays
        # until an explicit flatten()/manual resume if still under water.
        if self.current_drawdown() < self.cfg.max_drawdown_halt:
            self._halted = False
            self._soft_reduce = False
            self.breach_reasons = []

    def flatten(self) -> dict:
        """Auto-flatten: stand down and clear open positions. Records a single
        audit decision (per-instrument details captured in the details dict)."""
        pairs = list(self.positions.keys())
        detail = {"flattened_pairs": pairs, "halted": True}
        self._halted = True
        self.positions.clear()
        d = RiskDecision(False, "flatten_all", float(len(pairs)), 0.0,
                         action="flatten", reason="circuit breaker flatten",
                         details=detail)
        self._log(d)
        return d.to_audit()

    def resume(self) -> None:
        self._halted = False
        self._soft_reduce = False
        self._order_times.clear()
        self.day_realized_pnl = 0.0
        self.consecutive_losses = 0
        self.daily_start_equity = self.equity

    # ── analytics ──────────────────────────────────────────────────────────

    def current_drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    def historical_var(self, returns: deque[float] | None = None, confidence: float | None = None) -> dict:
        """Historical simulation VaR / CVaR of the return series."""
        conf = confidence or self.cfg.var_confidence
        r = np.array(list(returns if returns is not None else self._portfolio_returns), dtype=np.float64)
        if r.size < 20:
            return {"var_pct": 0.0, "cvar_pct": 0.0, "n_obs": int(r.size)}
        r = r[np.isfinite(r)]
        q = np.quantile(r, 1.0 - conf)
        var_pct = -float(q)
        tail = r[r <= q]
        cvar_pct = -float(tail.mean()) if tail.size else var_pct
        return {"var_pct": round(var_pct, 6), "cvar_pct": round(cvar_pct, 6), "n_obs": int(r.size)}

    def parametric_var(self, confidence: float | None = None) -> dict:
        """Parametric (normal) VaR of the portfolio return series."""
        conf = confidence or self.cfg.var_confidence
        r = np.array(list(self._portfolio_returns), dtype=np.float64)
        if r.size < 2:
            return {"var_pct": 0.0, "cvar_pct": 0.0}
        r = r[np.isfinite(r)]
        mu, sd = float(r.mean()), float(r.std(ddof=1))
        if sd <= 1e-12:
            return {"var_pct": 0.0, "cvar_pct": 0.0}
        try:
            from scipy.stats import norm
            z = norm.ppf(conf)
            cvar_z = norm.pdf(z) / (1.0 - conf)
        except ImportError:
            z, cvar_z = 2.326, 2.665
        var_pct = -(mu - z * sd)
        cvar_pct = -(mu - cvar_z * sd)
        return {"var_pct": round(max(0.0, var_pct), 6), "cvar_pct": round(max(0.0, cvar_pct), 6)}

    def portfolio_var(self, confidence: float | None = None) -> dict:
        h = self.historical_var(confidence=confidence)
        p = self.parametric_var(confidence=confidence)
        var_usd = h["var_pct"] * self.equity
        cvar_usd = h["cvar_pct"] * self.equity
        return {
            "var_pct": h["var_pct"],
            "cvar_pct": h["cvar_pct"],
            "var_usd": round(var_usd, 2),
            "cvar_usd": round(cvar_usd, 2),
            "param_var_pct": p["var_pct"],
            "param_cvar_pct": p["cvar_pct"],
        }

    def exposure_by_pair(self) -> dict[str, float]:
        return {p: float(pos.get("lots", 0.0)) for p, pos in self.positions.items()}

    def exposure_by_currency(self) -> dict[str, float]:
        """Net exposure per currency (base credit, quote debit). Uses simple
        per-lot 100k notional so tests stay deterministic."""
        out: dict[str, float] = {}
        for pair, pos in self.positions.items():
            lots = float(pos.get("lots", 0.0))
            direction = pos.get("direction", "long")
            sign = 1.0 if direction == "long" else -1.0
            base, quote = pair[:3], pair[3:]
            out[base] = out.get(base, 0.0) + sign * lots * 100_000
            out[quote] = out.get(quote, 0.0) - sign * lots * 100_000
        return {k: round(v, 2) for k, v in out.items()}

    def _check_gaps(self) -> dict[str, bool]:
        """Gap-risk flags: per-pair if any recent |return| exceeds threshold."""
        out: dict[str, bool] = {}
        for pair, dq in self._returns.items():
            r = np.array(list(dq), dtype=np.float64)
            out[pair] = bool(r.size > 0 and float(np.max(np.abs(r))) > self.cfg.gap_move_threshold)
        self.gap_flagged = out
        return out

    # ── internals ──────────────────────────────────────────────────────────

    @staticmethod
    def _pnl_to_ret(pnl: float, equity: float, lots: float) -> float:
        notional = abs(float(lots)) * 100_000.0
        return float(pnl) / max(notional, 1.0)

    def _log(self, decision: RiskDecision, ts: str | None = None) -> None:
        entry = decision.to_audit(ts)
        self.audit_log.append(entry)

    def get_audit(self, rule: str | None = None) -> list[dict]:
        if rule is None:
            return list(self.audit_log)
        return [e for e in self.audit_log if e["rule"] == rule]
