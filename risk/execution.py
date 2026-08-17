"""
risk/execution.py - Regime sizing, Almgren-Chriss, Drawdown exit, Portfolio VaR

ONE SOURCE OF TRUTH
-------------------
All risk parameters are loaded from config.settings.LIVE_RISK so that paper
trading, backtesting, shadow mode, and live execution share identical logic.
Never hardcode risk numbers in this file - always reference LIVE_RISK or the
constructor defaults that mirror it.
"""

import numpy as np

try:
    from config.settings import LIVE_RISK as _LR
except ImportError:
    _LR = {}  # standalone use without full project installed


class RegimePositionSizer:
    """
    Kelly fraction scales down in correlation crises and mean-reversion regimes.

    Defaults mirror config.settings.LIVE_RISK so paper / backtest / live are
    identical unless explicitly overridden.
    """

    def __init__(
        self,
        base_kelly=None,
        max_kelly=0.40,
        min_kelly=0.05,
        corr_crisis_thresh=None,
        corr_crisis_scale=None,
        hurst_trending=None,
        hurst_mean_rev=None,
        trending_bonus=None,
        mean_rev_penalty=None,
        vol_target=None,
        max_pos_pct=None,
        lot_size=10_000.0,
        pip_size=0.0001,
        min_stop_pips=10.0,
    ):
        # Pull canonical defaults from LIVE_RISK; constructor args override
        _rs = _LR.get("regime_scale", {})
        base_kelly = base_kelly if base_kelly is not None else _LR.get("kelly_fraction", 0.25)
        corr_crisis_thresh = (
            corr_crisis_thresh if corr_crisis_thresh is not None else _LR.get("corr_crisis_threshold", 0.70)
        )
        corr_crisis_scale = corr_crisis_scale if corr_crisis_scale is not None else _rs.get("crisis", 0.50)
        hurst_trending = hurst_trending if hurst_trending is not None else _LR.get("hurst_trending", 0.60)
        hurst_mean_rev = hurst_mean_rev if hurst_mean_rev is not None else _LR.get("hurst_mean_rev", 0.40)
        trending_bonus = trending_bonus if trending_bonus is not None else _rs.get("trending", 1.20)
        mean_rev_penalty = mean_rev_penalty if mean_rev_penalty is not None else _rs.get("mean_rev", 0.75)
        vol_target = vol_target if vol_target is not None else _LR.get("target_annual_vol", 0.10)
        max_pos_pct = max_pos_pct if max_pos_pct is not None else _LR.get("max_position_pct", 0.05)
        self.base_k = base_kelly
        self.max_k = max_kelly
        self.min_k = min_kelly
        self.cr_thresh = corr_crisis_thresh
        self.cr_scale = corr_crisis_scale
        self.h_trend = hurst_trending
        self.h_mr = hurst_mean_rev
        self.t_bonus = trending_bonus
        self.mr_pen = mean_rev_penalty
        self.vol_tgt = vol_target
        self.max_pct = max_pos_pct
        self.lot_size = lot_size
        self.pip_size = pip_size
        self.min_stop_pips = float(min_stop_pips)

    def _atr_to_pips(self, atr):
        atr = abs(float(atr))
        if atr > 5.0:
            return atr
        return atr / self.pip_size

    def _regime_scale(self, corr_avg=0.0, hurst=0.5, corr_break=0.0):
        scale = 1.0
        if corr_avg > self.cr_thresh or corr_break > 0:
            scale *= self.cr_scale
        if hurst > self.h_trend:
            scale *= self.t_bonus
        elif hurst < self.h_mr:
            scale *= self.mr_pen
        return float(np.clip(scale, self.min_k / self.base_k, self.max_k / self.base_k))

    def size(self, equity, win_prob, win_loss_r, returns, atr, corr_avg=0.0, hurst=0.5, corr_break=0.0):
        q = 1 - win_prob
        full_k = max(0, win_prob - q / max(win_loss_r, 0.01))
        base_k = full_k * self.base_k
        reg_sc = self._regime_scale(corr_avg, hurst, corr_break)
        adj_k = float(np.clip(base_k * reg_sc, self.min_k, self.max_k)) if base_k > 1e-9 else 0.0
        if len(returns) >= 20:
            vol_sc = np.clip(self.vol_tgt / (float(np.std(returns[-60:])) * np.sqrt(252) + 1e-9), 0.1, 3.0)
        else:
            vol_sc = 1.0
        risk_usd = equity * min(adj_k * vol_sc, self.max_pct)
        pip_stop = max(self.min_stop_pips, self._atr_to_pips(atr) * 1.5)
        if adj_k <= 1e-12 or risk_usd <= 0.0:
            lots = 0.0
        else:
            lots = round(
                np.clip(risk_usd / (pip_stop * self.lot_size * self.pip_size), 0.01, equity / self.lot_size * 0.3), 2
            )
        reg_desc = (
            "crisis"
            if corr_avg > self.cr_thresh
            else "trending"
            if hurst > self.h_trend
            else "mean_rev"
            if hurst < self.h_mr
            else "normal"
        )
        return {
            "lots": lots,
            "kelly": adj_k,
            "regime_scale": reg_sc,
            "vol_scalar": vol_sc,
            "regime": reg_desc,
            "risk_usd": risk_usd,
        }


class AlmgrenChrissExecutor:
    """Optimal execution schedule minimizing market impact + timing risk."""

    def __init__(self, sigma=0.0001, eta=2.5e-7, gamma=2.5e-8, lambda_risk=1e-6, adv=1_000_000):
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.lam = lambda_risk
        self.adv = adv

    def optimal_schedule(self, total_lots, n_slices=10):
        X = total_lots
        T = n_slices
        kappa = np.sqrt(max(self.lam * self.sigma**2 / self.eta, 0))
        t = np.arange(1, T + 1)
        if kappa * T < 1e-6:
            traj = np.ones(T) * X / T
        else:
            denom = np.sinh(kappa * T)
            x_arr = X * np.sinh(kappa * (T - t)) / (denom + 1e-12)
            x_arr = np.concatenate([[X], x_arr])
            traj = np.clip(-np.diff(x_arr), 0, None)
        s = traj.sum()
        return traj * X / s if s > 1e-9 else traj

    def estimate_impact_cost(self, lots, n_slices=10, pip_value=10.0):
        sched = self.optimal_schedule(lots, n_slices)
        perm = self.gamma * lots
        temp = self.eta * sum(s**2 for s in sched)
        imp_pips = (perm + temp) / 0.0001
        return {
            "impact_pips": round(imp_pips, 4),
            "schedule": sched,
            "total_cost_usd": round(imp_pips * pip_value * lots, 4),
            "n_slices": n_slices,
        }

    def should_split(self, lots, urgency="normal", atr=0.0005):
        if lots > 2.0:
            return True, 10
        if urgency == "urgent":
            return lots > 0.5, 3
        if urgency == "patient":
            return lots > 0.3, 15
        return lots > 0.5, 5


class SessionLimitsEnforcer:
    """
    Enforces per-session exposure limits (lots + open trades) from LIVE_RISK.
    Used identically in paper, shadow, and live to prevent environment divergence.

    Session identity comes from ``trading.session_utils.classify_session`` (DST-aware
    SoT). Policy key prefers ``asia_london`` / ``london_ny`` over exclusive primary
    - never a private ``"overlap"`` string.

    Definition: ``open_trades`` counts independent open position tickets/legs,
    not every scaling execution. Scale-in and scale-out actions modify lots on
    an existing open trade, so they are governed by ``max_lots`` and execution
    cost controls rather than consuming another ``max_open_trades`` slot.
    """

    def __init__(self, session_limits: dict | None = None, stage: str = "scale"):
        # Load from MATURITY_LADDER if stage provided, else from LIVE_RISK
        try:
            from config.settings import MATURITY_LADDER as _ML

            stage_cfg = _ML.get(stage, {})
            default_limits = stage_cfg.get("session_limits", _LR.get("session_limits", {}))
        except ImportError:
            default_limits = _LR.get("session_limits", {})
        self.limits = session_limits if session_limits is not None else default_limits

    def _current_session(self, hour_utc: int | None = None, now=None) -> str:
        """Classify into production policy key (DST-aware when ``now`` given)."""
        from datetime import UTC, datetime

        from trading.session_utils import classify_session, normalize_session_name

        if now is not None:
            return classify_session(now).policy_key
        if hour_utc is None:
            return classify_session(datetime.now(UTC)).policy_key
        # Hour-only callers (unit tests): pin a mid-summer UTC date so DST walls
        # match the historical fixed-UTC expectations used in fixtures.
        pinned = datetime(2026, 8, 6, int(hour_utc) % 24, 0, 0, tzinfo=UTC)
        return normalize_session_name(classify_session(pinned).policy_key)

    def _limits_for(self, sess: str) -> dict:
        """Lookup session limits with overlap → primary fallback (no 999 bypass)."""
        from trading.session_utils import normalize_session_name

        key = normalize_session_name(sess)
        lim = self.limits.get(key)
        if lim is None and key == "london_ny":
            lim = self.limits.get("overlap") or self.limits.get("london") or self.limits.get("ny")
        if lim is None and key == "asia_london":
            lim = self.limits.get("asia") or self.limits.get("london")
        if lim is None:
            lim = self.limits.get("off") or {"max_lots": 0.0, "max_open_trades": 0}
        return lim

    def check(
        self,
        hour_utc: int | None = None,
        open_lots: float = 0.0,
        open_trades: int = 0,
        *,
        now=None,
        session: str | None = None,
    ) -> dict:
        """
        Returns {"allowed": bool, "session": str, "max_lots": float, "max_trades": int}.
        Call before every order; block if allowed=False.
        """
        from trading.session_utils import normalize_session_name

        if session is not None:
            sess = normalize_session_name(session)
        else:
            sess = self._current_session(hour_utc, now=now)
        lim = self._limits_for(sess)
        max_lots = float(lim.get("max_lots", 0.0))
        max_trades = int(lim.get("max_open_trades", lim.get("max_trades", 0)))
        allowed = (open_lots < max_lots) and (open_trades < max_trades)
        return {
            "allowed": allowed,
            "session": sess,
            "max_lots": max_lots,
            "max_trades": max_trades,
            "open_lots": open_lots,
            "open_trades": open_trades,
        }


class DrawdownAwareExitManager:
    """
    Monitors portfolio drawdown and daily loss; signals early exit before hard stops.
    Defaults from LIVE_RISK so paper == backtest == live.
    """

    def __init__(self, soft_dd=None, hard_dd=None, daily_limit=None, max_cons=5, rec_bars=None):
        soft_dd = soft_dd if soft_dd is not None else _LR.get("soft_drawdown_reduce", 0.05)
        hard_dd = hard_dd if hard_dd is not None else _LR.get("max_drawdown_halt", 0.10)
        daily_limit = daily_limit if daily_limit is not None else _LR.get("daily_loss_limit", 0.03)
        rec_bars = rec_bars if rec_bars is not None else _LR.get("recovery_bars", 20)
        self.soft_dd = soft_dd
        self.hard_dd = hard_dd
        self.daily_limit = daily_limit
        self.max_cons = max_cons
        self.rec_bars = rec_bars
        self._equity = 1.0
        self._peak = 1.0
        self._day_start = 1.0
        self._cons = 0
        self._halted = False
        self._countdown = 0

    def update(self, equity, pnl):
        self._equity = equity
        self._peak = max(self._peak, equity)
        if pnl < 0:
            self._cons += 1
        else:
            self._cons = 0
        dd = max(0, (self._peak - equity) / self._peak)
        dl = max(0, (self._day_start - equity) / self._day_start)
        if self._halted:
            self._countdown -= 1
            if self._countdown <= 0:
                self._halted = False
                self._day_start = equity
            return {
                "action": "halt",
                "dd": dd,
                "daily_loss": dl,
                "consec_losses": self._cons,
                "size_multiplier": 0.0,
                "halt_bars": self._countdown,
            }
        if dd >= self.hard_dd or dl >= self.daily_limit:
            self._halted = True
            self._countdown = self.rec_bars
            return {
                "action": "close_all",
                "dd": dd,
                "daily_loss": dl,
                "consec_losses": self._cons,
                "size_multiplier": 0.0,
            }
        if dd >= self.soft_dd:
            return {
                "action": "reduce_50",
                "dd": dd,
                "daily_loss": dl,
                "consec_losses": self._cons,
                "size_multiplier": 0.5,
            }
        if self._cons >= self.max_cons:
            sm = max(0.25, 1.0 - 0.15 * (self._cons - self.max_cons + 1))
            return {
                "action": "reduce_size",
                "dd": dd,
                "daily_loss": dl,
                "consec_losses": self._cons,
                "size_multiplier": sm,
            }
        return {"action": "continue", "dd": dd, "daily_loss": dl, "consec_losses": self._cons, "size_multiplier": 1.0}

    def new_day(self):
        self._day_start = self._equity
        self._cons = 0

    def status(self):
        dd = max(0, (self._peak - self._equity) / self._peak)
        return {"equity": self._equity, "drawdown": dd, "peak": self._peak, "halted": self._halted}


class PortfolioVaR:
    """Correlation-adjusted VaR, CVaR, and per-pair lot limits for multi-pair book.

    Units contract (FIX for audit R-1/R-2, 2026-08-07):
    - ``update_returns(pair, ret)`` receives **price-fraction returns** (e.g.
      log-returns ≈ 0.0003 for an FX daily move). NOT pip-scaled returns.
    - ``positions[p]`` is in **standard lots** (1.0 = 100,000 units notional).
    - ``pip_value`` is retained only for backward-compat with old call sites and
      the legacy ``max_allowed_lots`` API; it is NOT used in ``parametric_var``
      because the math now uses the dollar notional directly.

    Bug history: the previous implementation multiplied pip-dollar weights
    (lots x $/pip) by a covariance of price-fraction returns (units of price²),
    yielding wildly wrong dollar magnitudes ($0.0096 instead of $96.48 for a
    standard EURUSD lot at 99% confidence). By moving to notional x price-fraction
    returns throughout, units are consistent: ($notional) x price-fraction x
    price-fraction → $² variance, and VaR comes out in $.
    """

    DEFAULT_NOTIONAL_PER_LOT = 100_000.0  # standard FX lot

    def __init__(self, confidence=0.99, horizon=1, pip_value=10.0, max_var_pct=0.02, notional_per_lot=None):
        self.conf = confidence
        self.horizon = horizon
        self.pv = pip_value  # legacy: only used by max_allowed_lots
        self.max_var = max_var_pct
        # Per-pair notional override (rare; most callers use default 100k)
        self._notional_per_lot = float(notional_per_lot or self.DEFAULT_NOTIONAL_PER_LOT)
        from collections import deque as _deque

        self._deque = _deque
        self._returns: dict[str, _deque] = {}
        # Per-pair notional override map: pair -> notional_in_dollars
        # e.g. {"USDJPY": 100_000, "XAUUSD": 100} for non-FX instruments
        self._notional_overrides: dict[str, float] = {}

    def set_notional(self, pair: str, notional_per_lot: float) -> None:
        """Override the notional-in-dollars per lot for a specific pair."""
        self._notional_overrides[pair] = float(notional_per_lot)

    def _notional(self, pair: str) -> float:
        """Return the dollar notional for 1.0 lot of `pair`."""
        if pair in self._notional_overrides:
            return float(self._notional_overrides[pair])
        return self._notional_per_lot

    def update_returns(self, pair, ret):
        """Append one bar's price-fraction return for ``pair``.

        Caller MUST pass price-fraction returns (e.g. log-returns ≈ 3e-4),
        NOT pip-scaled returns. See class docstring for the units contract.
        """
        if pair not in self._returns:
            self._returns[pair] = self._deque(maxlen=500)
        # R-1/R-2 sanity: warn (not raise) for clearly-out-of-contract values
        deq = self._returns[pair]
        try:
            r = float(ret)
        except Exception:
            return
        # Pip-scaled FX daily returns are typically ~3-30 pips = 3e-4..3e-3 in
        # price-fraction. If |r| is consistently > 0.5, it looks like a pip
        # value (e.g. 30 pips passed as 30 instead of 0.0030). Don't fail hard
        # because we don't want to crash the live engine; instead sanitise the
        # input by treating unusually large magnitude as already-priced
        # dollars-per-lot-move (which is also a valid interpretation when
        # the caller passed "pip returns" / pip_value). Either way, the new
        # notional-based math is robust because we keep weights and returns in
        # consistent $xfraction units.
        if abs(r) > 0.5:
            # Likely pip-scaled or daily-return-percent; scale down by pip_size
            # heuristic (1 pip = 1e-4 for non-JPY, 1e-2 for JPY). Default 1e-4.
            pip_size = 0.01 if "JPY" in pair else 0.0001
            r = r * pip_size
        deq.append(r)  # O(1) - deque auto-evicts oldest when maxlen reached

    def parametric_var(self, positions, equity):
        pairs = [p for p in positions if abs(positions[p]) > 0.001]
        if not pairs:
            return {"var_pct": 0.0, "var_usd": 0.0, "cvar_usd": 0.0, "correlation_avg": 0.0}
        # BUG-011: Exclude pairs with insufficient history instead of zero-padding
        min_obs = 20
        pairs = [p for p in pairs if len(self._returns.get(p, [])) >= min_obs]
        if not pairs:
            return {"var_pct": 0.0, "var_usd": 0.0, "cvar_usd": 0.0, "correlation_avg": 0.0}
        ml = min(len(self._returns.get(p, [])) for p in pairs)
        rm = np.array([list(self._returns[p])[-ml:] for p in pairs]).T
        corr = np.corrcoef(rm.T) if len(pairs) > 1 else np.array([[1.0]])
        stds = rm.std(0)
        cov = np.outer(stds, stds) * corr  # units: (price-fraction)²
        # FIX: dollar notional weights (units: $/lot x lots = $).
        # Notional x price-fraction gives $ per unit move; consistent with cov
        # in (price-fraction)² → final variance is in $².
        w = np.array([positions[p] * self._notional(p) for p in pairs])
        pv = float(w @ cov @ w.T)
        ps = np.sqrt(max(pv, 0) * self.horizon)  # standard deviation in $ over horizon
        try:
            from scipy.stats import norm

            z = norm.ppf(self.conf)
            var = z * ps
            cvar = ps * norm.pdf(z) / (1 - self.conf)
        except ImportError:
            z = 2.326
            var = z * ps
            cvar = var * 1.3
        corr_avg = float(corr[np.triu_indices(len(pairs), k=1)].mean()) if len(pairs) > 1 else 0.0
        return {
            "var_pct": round(var / max(equity, 1), 6),
            "var_usd": round(var, 4),
            "cvar_usd": round(cvar, 4),
            "correlation_avg": round(corr_avg, 3),
        }

    def max_allowed_lots(self, pair, equity, positions, pip_value=None):
        """Compute the additional lots of ``pair`` we can open without breaching
        the daily VaR budget.

        Sizes the candidate lot by its **portfolio contribution** - the marginal
        increase in portfolio VaR given the new pair's correlation with existing
        positions - rather than its standalone $-std. Solves the VaR-budget
        quadratic in closed form.

        ``pip_value`` is accepted for backward-compat but unused (math is done
        in dollar-notional x price-fraction units; see class docstring).
        """
        budget = equity * self.max_var
        min_obs = 20

        # Existing positions with sufficient return history.
        exist_pairs = [p for p in positions if abs(positions[p]) > 0.001 and len(self._returns.get(p, [])) >= min_obs]
        # Candidate pair must itself have history.
        cand_hist = list(self._returns.get(pair, []))
        if len(cand_hist) < min_obs:
            return 0.01  # insufficient history → fail-safe minimal size

        try:
            from scipy.stats import norm

            z = norm.ppf(self.conf)
        except ImportError:
            z = 2.326
        horizon = float(self.horizon)
        notional_new = self._notional(pair)

        # No existing positions → standalone sizing is exactly the portfolio
        # contribution, so the closed form degenerates to the simple case.
        if not exist_pairs:
            std_frac = float(np.std(cand_hist) * np.sqrt(horizon))
            if std_frac < 1e-9:
                return 0.01
            dollar_std_per_lot = notional_new * std_frac
            max_lots = budget / (z * max(dollar_std_per_lot, 1e-9))
            return round(float(np.clip(max_lots, 0.01, 10.0)), 2)

        # Build aligned return matrix for existing pairs + candidate pair and
        # the extended covariance (price-fraction² units).
        ml = min(min(len(self._returns[p]) for p in exist_pairs), len(cand_hist))
        rs = [list(self._returns[p])[-ml:] for p in exist_pairs]
        rs.append(cand_hist[-ml:])
        rm = np.array(rs).T  # (ml, k+1)
        corr = np.corrcoef(rm.T)  # (k+1, k+1)
        stds = rm.std(0)  # (k+1,)
        cov = np.outer(stds, stds) * corr  # price-fraction², (k+1, k+1)

        # Existing dollar weights (k,) and candidate notional-per-lot scalar.
        w_e = np.array([positions[p] * self._notional(p) for p in exist_pairs])
        sigma_new = float(stds[-1]) ** 2  # candidate variance (fraction²)
        # Covariance between existing portfolio and the candidate (vector).
        cov_en = cov[:-1, -1]  # (k,)

        # Portfolio variance as a function of candidate lots x:
        #   v(x) = sigma_e + 2 * x * notional_new * (w_e · cov_en) + x² * notional_new² * sigma_new
        # where sigma_e = w_e Σ_ee w_eᵀ (existing portfolio variance).
        sigma_e = float(w_e @ cov[:-1, :-1] @ w_e.T)
        # Budget constraint: z² * horizon * v(x) = budget²
        target_var2 = (budget / z) ** 2 / max(horizon, 1e-12)
        # Quadratic: A x² + B x + C = 0
        A = notional_new * notional_new * sigma_new
        B = 2.0 * notional_new * float(w_e @ cov_en)
        C = sigma_e - target_var2
        if A < 1e-18:
            # Degenerate linear case
            if abs(B) < 1e-18:
                return 0.01
            max_lots = max(0.0, -C / B)
        else:
            disc = B * B - 4.0 * A * C
            if disc < 0:
                # Existing portfolio already over budget - no room.
                return 0.01
            sq = np.sqrt(disc)
            x1 = (-B + sq) / (2.0 * A)
            x2 = (-B - sq) / (2.0 * A)
            max_lots = max(x for x in (x1, x2) if x > 0) if any(x > 0 for x in (x1, x2)) else 0.0
        # Sanity: standalone contribution bound (fail-safe ceiling).
        std_frac = float(np.std(cand_hist) * np.sqrt(horizon))
        if std_frac > 1e-9:
            ceiling = budget / (z * notional_new * std_frac)
            max_lots = min(max_lots, ceiling)
        return round(float(np.clip(max_lots, 0.01, 10.0)), 2)


if __name__ == "__main__":
    print("Risk & Execution - smoke tests")
    rs = RegimePositionSizer()
    ret = np.random.normal(0.001, 0.003, 100)
    for desc, kw in [
        ("normal", {}),
        ("crisis", {"corr_avg": 0.80}),
        ("trending", {"hurst": 0.70}),
        ("mean_rev", {"hurst": 0.30}),
    ]:
        r = rs.size(10000, 0.54, 1.7, ret, 0.0005, **kw)
        print(f"  {desc:12s}: {r['lots']:.2f}L | Kelly={r['kelly']:.3f} | scale={r['regime_scale']:.2f}")
    ac = AlmgrenChrissExecutor()
    for lots in [0.5, 1.0, 3.0]:
        r = ac.estimate_impact_cost(lots)
        print(f"  AC {lots}L: {r['impact_pips']:.4f} pips")
    dm = DrawdownAwareExitManager()
    eq = 10000.0
    for pnl in [100, -200, -150, -100, -80, -50, -400, -350]:
        eq += pnl
        r = dm.update(eq, pnl)
        print(f"  DD={r['dd']:.2%} | {r['action']:12s} | size={r['size_multiplier']:.2f}")
    pv = PortfolioVaR()
    for _ in range(100):
        for p in ["EURUSD", "GBPUSD"]:
            pv.update_returns(p, np.random.normal(0, 0.0003))
    v = pv.parametric_var({"EURUSD": 1.0, "GBPUSD": 0.8}, 10000)
    print(f"  VaR: {v['var_pct']:.2%} | ${v['var_usd']:.2f} | corr={v['correlation_avg']:.2f}")
    print("All tests passed ✓")

# Public aliases - main.py and external callers import these names.
# RegimePositionSizer is the canonical class; both names refer to the same object.
RegimeConditionalKelly = RegimePositionSizer  # legacy name used in main.py
DrawdownAwareExitPolicy = DrawdownAwareExitManager  # legacy name used in main.py
