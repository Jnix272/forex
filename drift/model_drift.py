"""
drift/model_drift.py — Model drift & deployment safety (Improvement #5)

Champion-challenger model management for live inference:

  * ChampionChallengerHarness  — shadow-mode evaluation: challenger predicts in
    parallel with the champion and we compare their performance on the same live
    trade/error stream (equity, Sharpe, drawdown, PSR).
  * CanaryRollout              — gradually route `x%` of live signals to a
    challenger, escalating the fraction on success and de-escalating on failure.
  * AutomatedRollbackMonitor   — watch live challenger/champion performance
    against a deployment-time baseline; breach thresholds (drawdown, PSR drop,
    error spike) → rollback alert (optional callback for checkpoint swap).

Integration: challenger vs champion comparisons reuse the repo's promotion gate
(``validation/promotion_gate.PromotionGate.evaluate_from_history``) and PSR from
``evaluation.metrics``, so promotion/readiness decisions stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from validation.promotion_gate import PromotionGate

try:
    from evaluation.metrics import probabilistic_sharpe_ratio as _psr_fn
except Exception:  # pragma: no cover - evaluation.metrics ships with repo
    def _psr_fn(sr_hat, sr_benchmark, n_obs, skewness=0.0, kurtosis=3.0):  # type: ignore
        return 0.5


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_sharpe(pnls: np.ndarray, annual_factor: float = 252.0) -> float:
    pnls = np.asarray(pnls, dtype=np.float64)
    if pnls.size < 5:
        return 0.0
    std = float(pnls.std(ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(pnls.mean() / std * np.sqrt(annual_factor))


def _safe_win_rate(pnls: np.ndarray) -> float:
    pnls = np.asarray(pnls, dtype=np.float64)
    if pnls.size == 0:
        return 0.5
    return float((pnls > 0).mean())


def _max_drawdown_pct(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=np.float64)
    if eq.size < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.maximum(peak, 1e-9)
    return float(abs(dd.min()))


# ═════════════════════════════════════════════════════════════════════════════
# Rolling model stats
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelStats:
    """Rolling window of trade results + prediction errors for one model."""
    model_id: str
    maxlen: int = 500
    pnls: List[float] = field(default_factory=list)
    equity: List[float] = field(default_factory=list)
    errors: List[bool] = field(default_factory=list)
    n_trades: int = 0
    n_errors: int = 0

    def record_trade(self, pnl: float, equity: Optional[float] = None) -> None:
        self.pnls.append(float(pnl))
        if equity is not None:
            self.equity.append(float(equity))
        if len(self.pnls) > self.maxlen:
            self.pnls.pop(0)
        self.n_trades += 1

    def record_error(self, is_error: bool) -> None:
        self.errors.append(bool(is_error))
        if len(self.errors) > self.maxlen:
            self.errors.pop(0)
        self.n_errors += int(bool(is_error))

    @property
    def pnl_array(self) -> np.ndarray:
        return np.asarray(self.pnls, dtype=np.float64)

    @property
    def equity_array(self) -> np.ndarray:
        return np.asarray(self.equity, dtype=np.float64)

    @property
    def error_rate(self) -> float:
        if not self.errors:
            return 0.0
        return float(np.mean(self.errors))

    def summary(self, annual_factor: float = 252.0) -> Dict[str, float]:
        return {
            "n_trades": float(len(self.pnls)),
            "sharpe": round(_safe_sharpe(self.pnl_array, annual_factor), 4),
            "win_rate": round(_safe_win_rate(self.pnl_array), 4),
            "max_drawdown": round(_max_drawdown_pct(self.equity_array), 4),
            "error_rate": round(self.error_rate, 4),
        }

    def psr(self, annual_factor: float = 252.0) -> float:
        pnls = self.pnl_array
        if pnls.size < 5:
            return 0.5
        sr = _safe_sharpe(pnls, annual_factor)
        sk = float(((pnls - pnls.mean()) / max(pnls.std(ddof=1), 1e-12) ** 3).mean()) if pnls.std(ddof=1) > 1e-12 else 0.0
        ku = float((((pnls - pnls.mean()) / max(pnls.std(ddof=1), 1e-12)) ** 4).mean()) if pnls.std(ddof=1) > 1e-12 else 3.0
        return _psr_fn(sr, 0.0, pnls.size, sk, ku)


# ═════════════════════════════════════════════════════════════════════════════
# Champion-challenger harness
# ═════════════════════════════════════════════════════════════════════════════

class ChampionChallengerHarness:
    """Shadow-mode comparison of challenger vs champion on a live stream.

    Both models receive the same trades/errors; comparisons are made on rolling
    windows so a challenger can be judged against the champion it must beat.
    """

    def __init__(
        self,
        champion: str,
        challengers: Sequence[str] = (),
        window: int = 300,
        sharpe_margin: float = 0.2,
        promotion_gate: Optional[PromotionGate] = None,
        annual_factor: float = 252.0,
    ):
        self.champion = champion
        self.challengers = list(challengers)
        self.window = window
        self.sharpe_margin = sharpe_margin
        self.annual_factor = annual_factor
        self._gate = promotion_gate or PromotionGate()
        self._stats: Dict[str, ModelStats] = {champion: ModelStats(champion, window)}
        for c in self.challengers:
            self._stats.setdefault(c, ModelStats(c, window))

    def add_challenger(self, model_id: str) -> None:
        if model_id not in self._stats:
            self._stats[model_id] = ModelStats(model_id, self.window)
            self.challengers.append(model_id)

    def record_trade(self, model_id: str, pnl: float, equity: Optional[float] = None) -> None:
        self._stats.setdefault(model_id, ModelStats(model_id, self.window)).record_trade(pnl, equity)

    def record_error(self, model_id: str, is_error: bool) -> None:
        self._stats.setdefault(model_id, ModelStats(model_id, self.window)).record_error(is_error)

    def stats(self, model_id: Optional[str] = None) -> Dict[str, Dict]:
        if model_id is not None:
            return {model_id: self._stats[model_id].summary(self.annual_factor)}
        return {k: s.summary(self.annual_factor) for k, s in self._stats.items()}

    def compare(self, challenger: str, min_trades: int = 30) -> Dict:
        """Evaluate challenger vs champion; returns gate result + relative metrics."""
        ch = self._stats[self.champion]
        ca = self._stats[challenger]
        if len(ca.pnls) < min_trades or len(ch.pnls) < min_trades:
            return {
                "challenger": challenger,
                "champion": self.champion,
                "ready": False,
                "n_challenger_trades": len(ca.pnls),
                "n_champion_trades": len(ch.pnls),
                "beats_champion": False,
                "gate": None,
            }

        challenger_gate = self._gate.evaluate_from_history(
            trade_pnls=ca.pnls, equity_curve=ca.equity_array.tolist() if ca.equity else [],
            n_bars=max(len(ca.pnls), 1),
        )
        champion_gate = self._gate.evaluate_from_history(
            trade_pnls=ch.pnls, equity_curve=ch.equity_array.tolist() if ch.equity else [],
            n_bars=max(len(ch.pnls), 1),
        )

        ch_sr = _safe_sharpe(ch.pnl_array, self.annual_factor)
        ca_sr = _safe_sharpe(ca.pnl_array, self.annual_factor)
        beats = ca_sr > ch_sr + self.sharpe_margin

        return {
            "challenger": challenger,
            "champion": self.champion,
            "ready": True,
            "n_challenger_trades": len(ca.pnls),
            "n_champion_trades": len(ch.pnls),
            "challenger_sharpe": round(ca_sr, 4),
            "champion_sharpe": round(ch_sr, 4),
            "challenger_psr": round(ca.psr(self.annual_factor), 4),
            "champion_psr": round(ch.psr(self.annual_factor), 4),
            "beats_champion": beats,
            "gate": {
                "challenger_promoted": bool(challenger_gate.get("promoted", False)),
                "champion_promoted": bool(champion_gate.get("promoted", False)),
                "challenger_reasons": challenger_gate.get("reasons", []),
            },
        }


# ═════════════════════════════════════════════════════════════════════════════
# Canary rollout
# ═════════════════════════════════════════════════════════════════════════════

class CanaryRollout:
    """Route a growing fraction of live signals to a challenger.

    ``escalate()`` increases the canary fraction after a success signal;
    ``deescalate()`` cuts it back after a failure. Routing is deterministic per
    signal index so the same fraction always selects the same signals.
    """

    def __init__(
        self,
        champion: str,
        challenger: str,
        min_fraction: float = 0.05,
        max_fraction: float = 1.0,
        step: float = 0.1,
        seed: int = 0,
    ):
        self.champion = champion
        self.challenger = challenger
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.step = step
        self.fraction = min_fraction
        self._rng = np.random.default_rng(seed)
        self._n_signals = 0
        self._n_to_challenger = 0
        self._escalated = 0
        self._deescalated = 0

    def route(self, signal: Optional[object] = None) -> str:
        """Pick which model handles this signal."""
        self._n_signals += 1
        if self.fraction >= 1.0:
            self._n_to_challenger += 1
            return self.challenger
        if self._rng.random() < self.fraction:
            self._n_to_challenger += 1
            return self.challenger
        return self.champion

    def escalate(self) -> float:
        """Increase canary fraction after a success signal."""
        self._escalated += 1
        self.fraction = min(self.max_fraction, self.fraction + self.step)
        return self.fraction

    def deescalate(self) -> float:
        """Cut back canary fraction after a failure signal."""
        self._deescalated += 1
        self.fraction = max(self.min_fraction, self.fraction - self.step)
        return self.fraction

    def status(self) -> Dict:
        return {
            "champion": self.champion,
            "challenger": self.challenger,
            "fraction": round(self.fraction, 4),
            "n_signals": self._n_signals,
            "n_to_challenger": self._n_to_challenger,
            "escalations": self._escalated,
            "deescalations": self._deescalated,
        }


# ═════════════════════════════════════════════════════════════════════════════
# Automated rollback
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RollbackConfig:
    """Thresholds that fire an automated rollback."""
    max_drawdown_pct: float = 0.10        # equity drawdown breach
    min_psr: float = 0.90                 # PSR floor vs benchmark 0
    min_sharpe: float = 0.30              # rolling sharpe floor
    error_spike_ratio: float = 2.0        # live error rate > baseline × ratio
    min_baseline_errors: int = 20         # baseline needs this many errors
    min_trades: int = 30                  # rolling window needs this many trades
    annual_factor: float = 252.0


class AutomatedRollbackMonitor:
    """Monitor live performance vs a deployment-time baseline and fire rollback.

    On a breach, returns an alert dict and calls ``rollback_callback(alert)`` if
    provided (e.g. swap checkpoints via ``monitoring.demotion_monitor.atomic_swap``).
    """

    def __init__(
        self,
        model_id: str,
        config: Optional[RollbackConfig] = None,
        rollback_callback: Optional[Callable[[Dict], None]] = None,
        window: int = 300,
        verbose: bool = True,
    ):
        self.model_id = model_id
        self.cfg = config or RollbackConfig()
        self.rollback_callback = rollback_callback
        self.window = window
        self.verbose = verbose
        self._live = ModelStats(model_id, window)
        self._baseline_error_rate: Optional[float] = None
        self._baseline_psr: Optional[float] = None
        self._baseline_equity: List[float] = []
        self._rolled_back = False
        self._triggers: List[str] = []

    def set_baseline(
        self,
        trade_pnls: Optional[Sequence[float]] = None,
        equity_curve: Optional[Sequence[float]] = None,
        error_rate: Optional[float] = None,
        psr: Optional[float] = None,
    ) -> None:
        """Record the deployment-time baseline performance."""
        if trade_pnls is not None:
            arr = np.asarray(trade_pnls, dtype=np.float64)
            self._baseline_psr = _psr_fn(
                _safe_sharpe(arr, self.cfg.annual_factor), 0.0, arr.size,
                float(((arr - arr.mean()) / max(arr.std(ddof=1), 1e-12) ** 3).mean()) if arr.std(ddof=1) > 1e-12 else 0.0,
                float((((arr - arr.mean()) / max(arr.std(ddof=1), 1e-12)) ** 4).mean()) if arr.std(ddof=1) > 1e-12 else 3.0,
            )
        if equity_curve is not None:
            self._baseline_equity = list(equity_curve)
        if error_rate is not None:
            self._baseline_error_rate = float(error_rate)

    def on_trade_closed(self, pnl: float, equity: Optional[float] = None) -> Optional[Dict]:
        """Record a live trade; returns a rollback alert if thresholds breach."""
        self._live.record_trade(pnl, equity)
        if self._rolled_back:
            return None
        if len(self._live.pnls) < self.cfg.min_trades:
            return None
        alert = self._check()
        if alert is not None:
            self._rolled_back = True
            if self.rollback_callback is not None:
                try:
                    self.rollback_callback(alert)
                except Exception as e:  # pragma: no cover
                    alert["callback_error"] = str(e)
            if self.verbose:
                print(f"\n[AutomatedRollbackMonitor] ⬇️ ROLLBACK for {self.model_id}")
                for t in alert["triggers"]:
                    print(f"   • {t}")
        return alert

    def on_prediction_error(self, is_error: bool) -> None:
        """Record a prediction error on the live stream."""
        self._live.record_error(is_error)

    def _check(self) -> Optional[Dict]:
        triggers: List[str] = []
        eq = self._live.equity_array
        pnls = self._live.pnl_array

        dd = _max_drawdown_pct(eq) if eq.size >= 2 else 0.0
        if dd > self.cfg.max_drawdown_pct:
            triggers.append(f"MaxDD {dd:.2%} > limit {self.cfg.max_drawdown_pct:.2%}")

        psr = self._live.psr(self.cfg.annual_factor)
        if psr < self.cfg.min_psr:
            triggers.append(f"PSR {psr:.3f} < floor {self.cfg.min_psr:.2f}")

        sr = _safe_sharpe(pnls, self.cfg.annual_factor)
        if sr < self.cfg.min_sharpe:
            triggers.append(f"Sharpe {sr:.3f} < floor {self.cfg.min_sharpe:.2f}")

        if self._baseline_error_rate is not None and len(self._live.errors) >= self.cfg.min_baseline_errors:
            live_err = self._live.error_rate
            if live_err > self._baseline_error_rate * self.cfg.error_spike_ratio:
                triggers.append(
                    f"Error spike {live_err:.3f} > baseline {self._baseline_error_rate:.3f} × {self.cfg.error_spike_ratio:.1f}"
                )

        if not triggers:
            return None
        return {
            "model_id": self.model_id,
            "rolled_back": True,
            "timestamp": _now_iso(),
            "triggers": triggers,
            "live": self._live.summary(self.cfg.annual_factor),
            "baseline_error_rate": self._baseline_error_rate,
            "baseline_psr": self._baseline_psr,
        }

    def reset(self) -> None:
        self._live = ModelStats(self.model_id, self.window)
        self._rolled_back = False
        self._triggers = []


# ═════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═════════════════════════════════════════════════════════════════════════════

def run_model_drift_check(
    champion_id: str,
    challenger_ids: Sequence[str],
    champion_pnls: Sequence[float],
    challenger_pnls: Dict[str, Sequence[float]],
    champion_equity: Optional[Sequence[float]] = None,
    window: int = 300,
    annual_factor: float = 252.0,
) -> Dict:
    """One-call model drift audit: populate the harness from existing trade lists
    and emit a comparison report with structured events for alerting."""
    harness = ChampionChallengerHarness(
        champion=champion_id, challengers=challenger_ids,
        window=window, annual_factor=annual_factor,
    )
    for pnl in champion_pnls:
        harness.record_trade(champion_id, pnl)
    comparisons = []
    alerts = []
    for cid in challenger_ids:
        for pnl in challenger_pnls.get(cid, []):
            harness.record_trade(cid, pnl)
        cmp = harness.compare(cid)
        comparisons.append(cmp)
        if cmp.get("ready") and not cmp["beats_champion"]:
            alerts.append({
                "type": "model_drift",
                "event": "challenger_losing",
                "challenger": cid,
                "champion": champion_id,
                "challenger_sharpe": cmp["challenger_sharpe"],
                "champion_sharpe": cmp["champion_sharpe"],
            })
    return {
        "ts": _now_iso(),
        "champion": champion_id,
        "comparisons": comparisons,
        "stats": harness.stats(),
        "alert": any(a["type"] == "model_drift" for a in alerts),
        "events": alerts,
    }
