"""Shadow-mode deployer: runs candidate model in parallel with live model."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from config.settings import PATHS

__all__ = ["ShadowModeDeployer"]


class ShadowModeDeployer:
    """
    Runs the candidate (new) model in parallel with the live model.
    Both receive the same market data, but only the live model's
    signals are executed. Candidate signals are logged for comparison.

    After shadow_bars bars, automatic comparison determines if the
    candidate should replace the live model:
      - Sharpe improvement > min_improvement
      - Max drawdown not worse than tolerance
      - Signal agreement with live model > agreement_floor (sanity check)
    """

    def __init__(
        self,
        shadow_bars: int = 2000,
        min_sharpe_imp: float = 0.1,
        max_dd_worse: float = 0.02,
        agreement_floor: float = 0.6,
        log_dir: str | None = None,
    ):
        if log_dir is None:
            log_dir = PATHS["logs_shadow"]
        self.shadow_bars = shadow_bars
        self.min_sharpe = min_sharpe_imp
        self.max_dd_worse = max_dd_worse
        self.agree_floor = agreement_floor
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._live_signals: list = []
        self._candidate_signals: list = []
        self._live_returns: list = []
        self._cand_returns: list = []
        self._bar_count: int = 0

    def step(self, live_signal: int, candidate_signal: int, actual_return: float):
        """Record one bar of shadow comparison."""
        self._live_signals.append(live_signal)
        self._candidate_signals.append(candidate_signal)
        live_ret = actual_return if live_signal != 1 else 0.0
        cand_ret = actual_return if candidate_signal != 1 else 0.0
        self._live_returns.append(live_ret * (1 if live_signal == 0 else -1 if live_signal == 2 else 0))
        self._cand_returns.append(cand_ret * (1 if candidate_signal == 0 else -1 if candidate_signal == 2 else 0))
        self._bar_count += 1

    def should_promote(self) -> tuple[bool, dict]:
        """Determine if the candidate should replace the live model."""
        if self._bar_count < self.shadow_bars:
            return False, {"status": "insufficient_data", "bars": self._bar_count, "needed": self.shadow_bars}

        live_r = np.array(self._live_returns)
        cand_r = np.array(self._cand_returns)

        def sharpe(r):
            if r.std() < 1e-10:
                return 0.0
            return r.mean() / r.std() * np.sqrt(252 * 1440)

        def max_dd(r):
            cumr = np.cumprod(1 + r)
            peak = np.maximum.accumulate(cumr)
            return float((cumr - peak).min() / (peak + 1e-9))

        live_sharpe = sharpe(live_r)
        cand_sharpe = sharpe(cand_r)
        sharpe_imp = cand_sharpe - live_sharpe
        live_mdd = max_dd(live_r)
        cand_mdd = max_dd(cand_r)
        dd_delta = cand_mdd - live_mdd
        ls = np.array(self._live_signals)
        cs = np.array(self._candidate_signals)
        agreement = float((ls == cs).mean())

        promote = sharpe_imp >= self.min_sharpe and dd_delta >= -self.max_dd_worse and agreement >= self.agree_floor

        diag = {
            "status": "promote" if promote else "keep_live",
            "bars_evaluated": self._bar_count,
            "live_sharpe": round(live_sharpe, 4),
            "cand_sharpe": round(cand_sharpe, 4),
            "sharpe_improvement": round(sharpe_imp, 4),
            "live_max_dd": round(live_mdd, 4),
            "cand_max_dd": round(cand_mdd, 4),
            "dd_delta": round(dd_delta, 4),
            "signal_agreement": round(agreement, 4),
            "promotion_criteria": {
                "sharpe_ok": sharpe_imp >= self.min_sharpe,
                "dd_ok": dd_delta >= -self.max_dd_worse,
                "agree_ok": agreement >= self.agree_floor,
            },
        }

        def _to_native(obj):
            if isinstance(obj, dict):
                return {k: _to_native(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_native(v) for v in obj]
            if hasattr(obj, "item"):
                return obj.item()
            return obj

        log_path = self.log_dir / f"shadow_{datetime.now():%Y%m%d_%H%M}.json"
        with open(log_path, "w") as f:
            json.dump(_to_native(diag), f, indent=2)
        print(
            f"[Shadow] {'PROMOTE ✓' if promote else 'Keep live ✗'} | "
            f"ΔSharpe={sharpe_imp:+.4f} | ΔDD={dd_delta:+.4f} | Agreement={agreement:.2%}"
        )
        return promote, diag

    def reset(self):
        self._live_signals = []
        self._candidate_signals = []
        self._live_returns = []
        self._cand_returns = []
        self._bar_count = 0
