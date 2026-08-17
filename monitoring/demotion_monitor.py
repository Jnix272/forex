"""
monitoring/demotion_monitor.py
================================
Rolling live performance monitor with automatic model demotion and rollback.

Monitors a rolling window of live trades and fires a rollback when:
  • Live Sharpe (rolling) < sharpe_floor  (default 0.5)
  • Win rate (rolling)    < winrate_floor (default 45%)
  • Max drawdown breaches policy limit     (default 10%)
  • Page-Hinkley change detector fires on equity curve

On demotion trigger:
  1. Swaps active model back to checkpoints/production_prev.pt
  2. Sets .needs_retrain flag (checked by train_gpu.py)
  3. Returns a DemotionAlert dict (caught by DiscordAlerter in live engine)
  4. Logs the rollback event to MLflow (or filesystem fallback)

Usage:
    from monitoring.demotion_monitor import DemotionMonitor
    monitor = DemotionMonitor()

    # Call on each closed trade:
    monitor.on_trade_closed(pnl=150.0, equity=10_150.0,
                            regime="trending", direction="long")

    # Call on each bar (equity update):
    alert = monitor.on_bar(equity=10_150.0)
    if alert and alert["demoted"]:
        discord.send(alert)
"""

import math
import os
import shutil
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from config.settings import active_checkpoint_dir


def _get_checkpoint_paths():
    run_dir = active_checkpoint_dir()
    return run_dir / "production_best.pt", run_dir / "production_prev.pt", run_dir / "needs_retrain.flag"


def atomic_swap(src, dst) -> None:
    """Atomically replace `dst` with the contents of `src`.

    Copies `src` to a temp file in the SAME directory as `dst` (so os.replace
    stays on one filesystem) then os.replace()s it into place. os.replace is
    atomic on POSIX and Windows, so a concurrent reader of `dst` (e.g. the live
    inference loop) always sees either the old or the new file - never a
    half-written one. (B-M2 / B-C3)
    """
    import tempfile

    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=str(dst.parent))
    os.close(fd)
    try:
        shutil.copy2(src, tmp_name)
        os.replace(tmp_name, dst)  # atomic
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass


# ── Page-Hinkley change detector ─────────────────────────────────────────────


class PageHinkleyDetector:
    """
    Page-Hinkley test for detecting a persistent change in the mean of a series.

    A signal fires when the cumulative deviation from the running mean exceeds
    the threshold lambda. Resets automatically after firing.

    Parameters
    ----------
    delta  : Magnitude tolerance - small allowed mean shift.
    lambda_: Detection threshold - higher = fewer false positives.
    min_obs: Min observations before detector can fire.
    """

    def __init__(self, delta: float = 0.005, lambda_: float = 50.0, min_obs: int = 30):
        self.delta = delta
        self.lambda_ = lambda_
        self.min_obs = min_obs
        self._sum = 0.0
        self._n = 0
        self._min_sum = 0.0

    def reset(self):
        self._sum = 0.0
        self._n = 0
        self._min_sum = 0.0

    def add(self, value: float) -> bool:
        """
        Add one observation. Returns True if a change is detected.
        """
        self._n += 1
        self._sum += value - self.delta
        self._min_sum = min(self._min_sum, self._sum)
        if self._n < self.min_obs:
            return False
        ph = self._sum - self._min_sum
        if ph > self.lambda_:
            self.reset()
            return True
        return False


# ── rolling stats ─────────────────────────────────────────────────────────────


class _RollingWindow:
    """Efficient rolling window for Sharpe and win rate computation."""

    def __init__(self, maxlen: int = 300, trades_per_year: float = 252 * 50):
        self._pnls: deque[float] = deque(maxlen=maxlen)
        self._wins: deque[bool] = deque(maxlen=maxlen)
        self._equitys: deque[float] = deque(maxlen=maxlen)
        self._win_count = 0
        self._ann_factor = math.sqrt(trades_per_year)

    def add_trade(self, pnl: float, equity: float | None = None):
        if self._wins.maxlen and len(self._wins) == self._wins.maxlen:
            self._win_count -= int(self._wins[0])
        self._pnls.append(pnl)
        won = pnl > 0
        self._wins.append(won)
        self._win_count += int(won)
        if equity is not None:
            self._equitys.append(equity)

    def add_equity(self, equity: float):
        self._equitys.append(equity)

    @property
    def n_trades(self) -> int:
        return len(self._pnls)

    @property
    def sharpe(self) -> float:
        if len(self._pnls) < 10:
            return 0.0
        if len(self._equitys) >= len(self._pnls):
            eq = np.array(self._equitys)[-len(self._pnls) :]
            prev_eq = np.maximum(eq - np.array(self._pnls), 1e-9)
            arr = np.array(self._pnls) / prev_eq
        else:
            arr = np.array(self._pnls)
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        return float(arr.mean() / std * self._ann_factor) if std > 1e-9 else 0.0

    @property
    def win_rate(self) -> float:
        if len(self._wins) == 0:
            return 0.5
        return float(self._win_count) / len(self._wins)

    @property
    def max_drawdown(self) -> float:
        if len(self._equitys) < 2:
            return 0.0
        eq = np.array(self._equitys)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.maximum(peak, 1e-9)
        return float(abs(dd.min()))


# ── main monitor ─────────────────────────────────────────────────────────────


class DemotionMonitor:
    """
    Rolling live performance monitor with Page-Hinkley change detection.

    Parameters
    ----------
    sharpe_floor     : Demote if rolling Sharpe < this value.
    winrate_floor    : Demote if rolling win rate < this value.
    max_drawdown_pct : Demote if rolling max drawdown > this value.
    window_trades    : Number of recent trades used for rolling stats.
    ph_delta         : Page-Hinkley: magnitude tolerance.
    ph_lambda        : Page-Hinkley: detection threshold.
    auto_rollback    : If True, automatically swap checkpoints on demotion.
    """

    def __init__(
        self,
        sharpe_floor: float = 0.5,
        winrate_floor: float = 0.45,
        max_drawdown_pct: float = 0.10,
        window_trades: int = 300,
        ph_delta: float = 0.005,
        ph_lambda: float = 50.0,
        auto_rollback: bool = True,
        verbose: bool = True,
        trades_per_year: float = 252 * 50,
    ):
        self.sharpe_floor = sharpe_floor
        self.winrate_floor = winrate_floor
        self.max_dd_pct = max_drawdown_pct
        self.auto_rollback = auto_rollback
        self.verbose = verbose

        self.window_trades = window_trades
        self._window = _RollingWindow(maxlen=window_trades, trades_per_year=trades_per_year)
        self._ph = PageHinkleyDetector(delta=ph_delta, lambda_=ph_lambda)
        self._ph_fired = False  # B-M3: latched until consumed by _check_triggers
        self._demoted = False
        self._n_bars = 0

    # ── public API ──────────────────────────────────────────────────────────

    def on_trade_closed(
        self,
        pnl: float,
        equity: float,
        regime: str = "unknown",
        direction: str = "long",
    ):
        """Call every time a trade is closed."""
        self._window.add_trade(pnl, equity)
        # Page-Hinkley on equity changes (detects regime shift in performance).
        # B-M3: capture the detector's return value - previously it was discarded
        # so the change-detector trigger could never actually fire a demotion.
        if self._ph.add(-pnl):  # negative PnL = increasing loss trend
            self._ph_fired = True

    def on_bar(self, equity: float) -> dict | None:
        """
        Call every bar. Returns None normally, or a DemotionAlert dict
        if a de-promotion should be triggered.
        """
        self._n_bars += 1
        self._window.add_equity(equity)

        if self._demoted:
            return None  # already demoted, don't fire again

        if self._window.n_trades < min(30, self.window_trades):
            return None  # not enough data yet

        triggers = self._check_triggers()
        if triggers:
            return self._fire_demotion(triggers, equity)
        return None

    def status(self) -> dict:
        """Current rolling performance status."""
        return {
            "n_trades": self._window.n_trades,
            "sharpe": round(self._window.sharpe, 4),
            "win_rate": round(self._window.win_rate, 4),
            "max_drawdown": round(self._window.max_drawdown, 4),
            "demoted": self._demoted,
        }

    def reset(self):
        """Call after a new model is deployed to start fresh monitoring."""
        self._window = _RollingWindow(maxlen=self._window._pnls.maxlen)
        self._ph = PageHinkleyDetector()
        self._ph_fired = False
        self._demoted = False
        self._n_bars = 0
        _prod, _prev, needs_retrain = _get_checkpoint_paths()
        if needs_retrain.exists():
            needs_retrain.unlink()

    # ── internal ────────────────────────────────────────────────────────────

    def _check_triggers(self) -> list[str]:
        """Returns list of trigger reasons, empty if no demotion needed."""
        triggers = []
        sr = self._window.sharpe
        wr = self._window.win_rate
        dd = self._window.max_drawdown

        if sr < self.sharpe_floor:
            triggers.append(f"Sharpe {sr:.3f} < floor {self.sharpe_floor}")
        if wr < self.winrate_floor:
            triggers.append(f"WinRate {wr:.1%} < floor {self.winrate_floor:.0%}")
        if dd > self.max_dd_pct:
            triggers.append(f"MaxDD {dd:.1%} > limit {self.max_dd_pct:.0%}")
        if self._ph_fired:
            triggers.append("Page-Hinkley change detector fired on equity curve")
            self._ph_fired = False
        return triggers

    def _fire_demotion(self, triggers: list[str], equity: float) -> dict:
        """Execute demotion and return alert dict."""
        self._demoted = True
        timestamp = datetime.now(UTC).isoformat()

        # Signal retrain
        run_dir = active_checkpoint_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        prod, prev, needs_retrain = _get_checkpoint_paths()
        needs_retrain.write_text(f"Demotion at {timestamp}\n" + "\n".join(triggers))

        # Rollback checkpoint
        rollback_ok = False
        if self.auto_rollback and prev.exists():
            try:
                import logging

                logging.warning("Auto-rollback: restoring previous checkpoint.")
                atomic_swap(prev, prod)
                rollback_ok = True
            except Exception as e:
                print(f"[DemotionMonitor] Rollback failed: {e}")

        alert = {
            "demoted": True,
            "timestamp": timestamp,
            "triggers": triggers,
            "status": self.status(),
            "rollback_ok": rollback_ok,
            "equity": equity,
            "needs_retrain": True,
        }

        if self.verbose:
            print(f"\n[DemotionMonitor] ⬇️  DEMOTION TRIGGERED @ {timestamp}")
            for t in triggers:
                print(f"   • {t}")
            print(f"   Rollback: {'✓' if rollback_ok else '✗ (no prev checkpoint)'}")
            prod, prev, needs_retrain = _get_checkpoint_paths()
            print(f"   Retrain flag: {needs_retrain}")

        # Log to MLflow / filesystem
        self._log_demotion(alert)
        return alert

    def _log_demotion(self, alert: dict):
        try:
            from validation.mlflow_logger import MLflowModelLogger

            logger = MLflowModelLogger(verbose=False)
            logger.log_promotion(
                model_name="rollback_event",
                gate_result={"promoted": False, "details": self.status(), "reasons": alert["triggers"]},
                extra_tags={"event_type": "demotion"},
            )
        except Exception:
            import json as _json

            log_dir = active_checkpoint_dir().parent / "logs" / "demotions"
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            with open(log_dir / f"demotion_{ts}.json", "w") as f:
                _json.dump(alert, f, indent=2, default=str)


# ── smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(42)

    print("=== DemotionMonitor smoke test ===\n")
    mon = DemotionMonitor(
        sharpe_floor=0.5,
        winrate_floor=0.45,
        max_drawdown_pct=0.10,
        window_trades=50,
        auto_rollback=False,
        verbose=True,
    )

    # Simulate 80 good trades
    eq = 10_000.0
    for i in range(80):
        pnl = float(rng.normal(50, 150))
        eq += pnl
        mon.on_trade_closed(pnl=pnl, equity=eq)
        alert = mon.on_bar(eq)
        if alert and alert["demoted"]:
            print(f"\n  Demotion at trade {i}: {alert['triggers']}")
            break

    print(f"\n  Status after 80 good trades: {mon.status()}")

    # Simulate 50 bad trades (consecutive losses)
    print("\n  Simulating 50 losing trades...")
    for i in range(50):
        pnl = float(rng.normal(-100, 80))
        eq = max(eq + pnl, 1.0)
        mon.on_trade_closed(pnl=pnl, equity=eq)
        alert = mon.on_bar(eq)
        if alert and alert["demoted"]:
            print(f"\n  Demotion triggered at bad-trade {i}:")
            for t in alert["triggers"]:
                print(f"    • {t}")
            break

    print(f"\n  Final status: {mon.status()}")
    print("OK ✓")
