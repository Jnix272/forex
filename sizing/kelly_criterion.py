"""
sizing/kelly_criterion.py
==========================
Fractional Kelly position sizing with:
  - Quarter-Kelly default (conservative)
  - Volatility targeting (scales down in high-vol regimes)
  - Square Root market impact model
  - Both scaling strategies: pyramid winners + martingale losers
"""
import numpy as np

from risk.execution import RegimePositionSizer


def kelly_binary(win_prob: float, win_loss_ratio: float) -> float:
    """Kelly fraction = p - q/b where b=win/loss ratio."""
    if win_loss_ratio <= 0:
        return 0.0
    q = 1 - win_prob
    return win_prob - q / win_loss_ratio


def fractional_kelly(full_kelly: float, fraction: float = 0.25) -> float:
    return np.clip(full_kelly * fraction, 0, 1)


def vol_target_scalar(
    returns: np.ndarray,
    target_vol: float = 0.10,
    lookback: int = 20,
) -> float:
    if len(returns) < 2: return 1.0
    recent = returns[-lookback:] if len(returns) >= lookback else returns
    realized_vol = float(np.std(recent) * np.sqrt(252))
    return np.clip(target_vol / (realized_vol + 1e-9), 0.1, 3.0)


def square_root_impact(
    lots: float,
    adv_lots: float = 1000.0,
    perm_impact_coef: float = 0.1,
    pip_size: float = 0.0001,
) -> float:
    """Market impact (pips) = coef × sqrt(size/ADV)."""
    pct = lots / (adv_lots + 1e-9)
    impact_pips = perm_impact_coef * np.sqrt(pct)
    return impact_pips


class PositionSizer:
    """Legacy adapter around the canonical regime-aware risk sizer."""

    def __init__(self, equity=10_000, kelly_fraction=0.25,
                 max_position_pct=0.05, target_vol=0.10, pip_risk=20.0):
        self.equity    = equity
        self.frac      = kelly_fraction
        self.max_pct   = max_position_pct
        self.tvol      = target_vol
        self.pip_risk  = pip_risk
        self._sizer = RegimePositionSizer(
            base_kelly=kelly_fraction,
            max_pos_pct=max_position_pct,
            vol_target=target_vol,
            min_stop_pips=pip_risk,
        )

    def size_position(self, win_prob, win_loss_ratio, returns,
                      price, current_atr, lot_size=10_000):
        full_k  = kelly_binary(win_prob, win_loss_ratio)
        frac_k  = fractional_kelly(full_k, self.frac)
        self._sizer.lot_size = lot_size
        sized = self._sizer.size(
            equity=self.equity,
            win_prob=win_prob,
            win_loss_r=win_loss_ratio,
            returns=np.array(returns),
            atr=current_atr,
        )
        pip_val = lot_size * self._sizer.pip_size
        impact = square_root_impact(sized["lots"], pip_size=self._sizer.pip_size) * pip_val * sized["lots"]
        return {
            "lots": sized["lots"], "full_kelly": full_k, "frac_kelly": frac_k,
            "vol_scalar": sized["vol_scalar"], "risk_usd": sized["risk_usd"], "impact_usd": impact,
        }
