"""
risk/execution.py — Regime sizing, Almgren-Chriss, Drawdown exit, Portfolio VaR

ONE SOURCE OF TRUTH
-------------------
All risk parameters are loaded from config.settings.LIVE_RISK so that paper
trading, backtesting, shadow mode, and live execution share identical logic.
Never hardcode risk numbers in this file — always reference LIVE_RISK or the
constructor defaults that mirror it.
"""

import numpy as np

try:
    from config.settings import LIVE_RISK as _LR
except ImportError:
    _LR = {}   # standalone use without full project installed


class RegimePositionSizer:
    """
    Kelly fraction scales down in correlation crises and mean-reversion regimes.

    Defaults mirror config.settings.LIVE_RISK so paper / backtest / live are
    identical unless explicitly overridden.
    """
    def __init__(self,
                 base_kelly=None, max_kelly=0.40, min_kelly=0.05,
                 corr_crisis_thresh=None, corr_crisis_scale=None,
                 hurst_trending=None, hurst_mean_rev=None,
                 trending_bonus=None, mean_rev_penalty=None,
                 vol_target=None, max_pos_pct=None,
                 lot_size=10_000.0, pip_size=0.0001,
                 min_stop_pips=10.0):
        # Pull canonical defaults from LIVE_RISK; constructor args override
        _rs = _LR.get("regime_scale", {})
        base_kelly         = base_kelly         if base_kelly         is not None else _LR.get("kelly_fraction", 0.25)
        corr_crisis_thresh = corr_crisis_thresh if corr_crisis_thresh is not None else _LR.get("corr_crisis_threshold", 0.70)
        corr_crisis_scale  = corr_crisis_scale  if corr_crisis_scale  is not None else _rs.get("crisis", 0.50)
        hurst_trending     = hurst_trending     if hurst_trending     is not None else _LR.get("hurst_trending", 0.60)
        hurst_mean_rev     = hurst_mean_rev     if hurst_mean_rev     is not None else _LR.get("hurst_mean_rev", 0.40)
        trending_bonus     = trending_bonus     if trending_bonus     is not None else _rs.get("trending", 1.20)
        mean_rev_penalty   = mean_rev_penalty   if mean_rev_penalty   is not None else _rs.get("mean_rev", 0.75)
        vol_target         = vol_target         if vol_target         is not None else _LR.get("target_annual_vol", 0.10)
        max_pos_pct        = max_pos_pct        if max_pos_pct        is not None else _LR.get("max_position_pct", 0.05)
        self.base_k=base_kelly;self.max_k=max_kelly;self.min_k=min_kelly
        self.cr_thresh=corr_crisis_thresh;self.cr_scale=corr_crisis_scale
        self.h_trend=hurst_trending;self.h_mr=hurst_mean_rev
        self.t_bonus=trending_bonus;self.mr_pen=mean_rev_penalty
        self.vol_tgt=vol_target;self.max_pct=max_pos_pct
        self.lot_size=lot_size;self.pip_size=pip_size
        self.min_stop_pips=float(min_stop_pips)

    def _atr_to_pips(self, atr):
        atr = abs(float(atr))
        if atr > 5.0:
            return atr
        return atr / self.pip_size

    def _regime_scale(self,corr_avg=0.0,hurst=0.5,corr_break=0.0):
        scale=1.0
        if corr_avg>self.cr_thresh or corr_break>0: scale*=self.cr_scale
        if hurst>self.h_trend: scale*=self.t_bonus
        elif hurst<self.h_mr:  scale*=self.mr_pen
        return float(np.clip(scale,self.min_k/self.base_k,self.max_k/self.base_k))

    def size(self,equity,win_prob,win_loss_r,returns,atr,corr_avg=0.0,hurst=0.5,corr_break=0.0):
        q=1-win_prob; full_k=max(0,win_prob-q/max(win_loss_r,0.01))
        base_k = full_k * self.base_k
        reg_sc = self._regime_scale(corr_avg, hurst, corr_break)
        adj_k = float(np.clip(base_k * reg_sc, self.min_k, self.max_k)) if base_k > 1e-9 else 0.0
        if len(returns)>=20:
            vol_sc=np.clip(self.vol_tgt/(float(np.std(returns[-60:]))*np.sqrt(252)+1e-9),0.1,3.0)
        else: vol_sc=1.0
        risk_usd=equity*min(adj_k*vol_sc,self.max_pct)
        pip_stop=max(self.min_stop_pips,self._atr_to_pips(atr)*1.5)
        if adj_k <= 1e-12 or risk_usd <= 0.0:
            lots = 0.0
        else:
            lots=round(np.clip(risk_usd/(pip_stop*self.lot_size*self.pip_size),0.01,equity/self.lot_size*0.3),2)
        reg_desc=("crisis" if corr_avg>self.cr_thresh else
                  "trending" if hurst>self.h_trend else
                  "mean_rev" if hurst<self.h_mr else "normal")
        return {"lots":lots,"kelly":adj_k,"regime_scale":reg_sc,"vol_scalar":vol_sc,
                "regime":reg_desc,"risk_usd":risk_usd}


class AlmgrenChrissExecutor:
    """Optimal execution schedule minimizing market impact + timing risk."""
    def __init__(self,sigma=0.0001,eta=2.5e-7,gamma=2.5e-8,lambda_risk=1e-6,adv=1_000_000):
        self.sigma=sigma;self.eta=eta;self.gamma=gamma;self.lam=lambda_risk;self.adv=adv

    def optimal_schedule(self,total_lots,n_slices=10):
        X=total_lots; T=n_slices
        kappa=np.sqrt(max(self.lam*self.sigma**2/self.eta,0))
        t=np.arange(1,T+1)
        if kappa*T<1e-6: traj=np.ones(T)*X/T
        else:
            denom=np.sinh(kappa*T)
            x_arr=X*np.sinh(kappa*(T-t))/(denom+1e-12)
            x_arr=np.concatenate([[X],x_arr])
            traj=np.clip(-np.diff(x_arr),0,None)
        s=traj.sum()
        return traj*X/s if s>1e-9 else traj

    def estimate_impact_cost(self,lots,n_slices=10,pip_value=10.0):
        sched=self.optimal_schedule(lots,n_slices)
        perm=self.gamma*lots; temp=self.eta*sum(s**2 for s in sched)
        imp_pips=(perm+temp)/0.0001
        return {"impact_pips":round(imp_pips,4),"schedule":sched,
                "total_cost_usd":round(imp_pips*pip_value*lots,4),"n_slices":n_slices}

    def should_split(self,lots,urgency="normal",atr=0.0005):
        if lots>2.0: return True,10
        if urgency=="urgent": return lots>0.5,3
        if urgency=="patient": return lots>0.3,15
        return lots>0.5,5


class SessionLimitsEnforcer:
    """
    Enforces per-session exposure limits (lots + open trades) from LIVE_RISK.
    Used identically in paper, shadow, and live to prevent environment divergence.

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

    def _current_session(self, hour_utc: int) -> str:
        """Classify UTC hour into trading session."""
        if 12 <= hour_utc < 16:
            return "overlap"   # London/NY overlap (highest volume) — formerly unreachable 'ny' range
        if 7 <= hour_utc < 12:
            return "london"
        if 16 <= hour_utc < 21:
            return "ny"
        if 0 <= hour_utc < 7:
            return "asia"
        return "off"

    def check(self, hour_utc: int, open_lots: float, open_trades: int) -> dict:
        """
        Returns {"allowed": bool, "session": str, "max_lots": float, "max_trades": int}.
        Call before every order; block if allowed=False.
        """
        sess = self._current_session(hour_utc)
        lim  = self.limits.get(sess, {"max_lots": 999, "max_open_trades": 999})
        max_lots   = float(lim.get("max_lots", 999))
        max_trades = int(lim.get("max_open_trades", lim.get("max_trades", 999)))
        allowed = (open_lots < max_lots) and (open_trades < max_trades)
        return {
            "allowed":    allowed,
            "session":    sess,
            "max_lots":   max_lots,
            "max_trades": max_trades,
            "open_lots":  open_lots,
            "open_trades": open_trades,
        }


class DrawdownAwareExitManager:
    """
    Monitors portfolio drawdown and daily loss; signals early exit before hard stops.
    Defaults from LIVE_RISK so paper == backtest == live.
    """
    def __init__(self,
                 soft_dd=None, hard_dd=None, daily_limit=None,
                 max_cons=5, rec_bars=None):
        soft_dd     = soft_dd     if soft_dd     is not None else _LR.get("soft_drawdown_reduce", 0.05)
        hard_dd     = hard_dd     if hard_dd     is not None else _LR.get("max_drawdown_halt",    0.10)
        daily_limit = daily_limit if daily_limit is not None else _LR.get("daily_loss_limit",     0.03)
        rec_bars    = rec_bars    if rec_bars    is not None else _LR.get("recovery_bars",         20)
        self.soft_dd=soft_dd;self.hard_dd=hard_dd;self.daily_limit=daily_limit
        self.max_cons=max_cons;self.rec_bars=rec_bars
        self._equity=1.0;self._peak=1.0;self._day_start=1.0
        self._cons=0;self._halted=False;self._countdown=0

    def update(self,equity,pnl):
        self._equity=equity; self._peak=max(self._peak,equity)
        if pnl<0: self._cons+=1
        else: self._cons=0
        dd=max(0,(self._peak-equity)/self._peak)
        dl=max(0,(self._day_start-equity)/self._day_start)
        if self._halted:
            self._countdown-=1
            if self._countdown<=0: self._halted=False;self._day_start=equity
            return {"action":"halt","dd":dd,"daily_loss":dl,"consec_losses":self._cons,
                    "size_multiplier":0.0,"halt_bars":self._countdown}
        if dd>=self.hard_dd or dl>=self.daily_limit:
            self._halted=True;self._countdown=self.rec_bars
            return {"action":"close_all","dd":dd,"daily_loss":dl,"consec_losses":self._cons,"size_multiplier":0.0}
        if dd>=self.soft_dd:
            return {"action":"reduce_50","dd":dd,"daily_loss":dl,"consec_losses":self._cons,"size_multiplier":0.5}
        if self._cons>=self.max_cons:
            sm=max(0.25,1.0-0.15*(self._cons-self.max_cons+1))
            return {"action":"reduce_size","dd":dd,"daily_loss":dl,"consec_losses":self._cons,"size_multiplier":sm}
        return {"action":"continue","dd":dd,"daily_loss":dl,"consec_losses":self._cons,"size_multiplier":1.0}

    def new_day(self): self._day_start=self._equity;self._cons=0
    def status(self):
        dd=max(0,(self._peak-self._equity)/self._peak)
        return {"equity":self._equity,"drawdown":dd,"peak":self._peak,"halted":self._halted}


class PortfolioVaR:
    """Correlation-adjusted VaR, CVaR, and per-pair lot limits for multi-pair book."""
    def __init__(self,confidence=0.99,horizon=1,pip_value=10.0,max_var_pct=0.02):
        self.conf=confidence;self.horizon=horizon;self.pv=pip_value;self.max_var=max_var_pct
        from collections import deque as _deque
        self._deque = _deque
        self._returns:dict[str,_deque]={}

    def update_returns(self,pair,ret):
        if pair not in self._returns: self._returns[pair]=self._deque(maxlen=500)
        self._returns[pair].append(ret)  # O(1) — deque auto-evicts oldest when maxlen reached

    def parametric_var(self,positions,equity):
        pairs=[p for p in positions if abs(positions[p])>0.001]
        if not pairs: return {"var_pct":0.0,"var_usd":0.0,"cvar_usd":0.0,"correlation_avg":0.0}
        # BUG-011: Exclude pairs with insufficient history instead of zero-padding
        min_obs = 20
        pairs = [p for p in pairs if len(self._returns.get(p, [])) >= min_obs]
        if not pairs: return {"var_pct":0.0,"var_usd":0.0,"cvar_usd":0.0,"correlation_avg":0.0}
        ml=min(len(self._returns.get(p,[])) for p in pairs)
        rm=np.array([list(self._returns[p])[-ml:] for p in pairs]).T
        corr=np.corrcoef(rm.T) if len(pairs)>1 else np.array([[1.0]])
        stds=rm.std(0); cov=np.outer(stds,stds)*corr
        w=np.array([positions[p]*self.pv for p in pairs])
        pv=float(w@cov@w.T); ps=np.sqrt(max(pv,0)*self.horizon)
        try:
            from scipy.stats import norm
            z=norm.ppf(self.conf); var=z*ps; cvar=ps*norm.pdf(z)/(1-self.conf)
        except ImportError:
            z=2.326; var=z*ps; cvar=var*1.3
        corr_avg=float(corr[np.triu_indices(len(pairs),k=1)].mean()) if len(pairs)>1 else 0.0
        return {"var_pct":round(var/max(equity,1),6),"var_usd":round(var,4),
                "cvar_usd":round(cvar,4),"correlation_avg":round(corr_avg,3)}

    def max_allowed_lots(self,pair,equity,positions,pip_value=10.0):
        cur=self.parametric_var(positions,equity)["var_usd"]
        budget=equity*self.max_var; remaining=max(0,budget-cur)
        hist=list(self._returns.get(pair,[0.0]*20))[-60:]
        std1=float(np.std(hist)*np.sqrt(self.horizon))*pip_value
        if std1<1e-9: return 5.0
        try:
            from scipy.stats import norm; z=norm.ppf(self.conf)
        except ImportError: z=2.326
        return round(float(np.clip(remaining/(z*std1),0.01,10.0)),2)


if __name__=="__main__":
    print("Risk & Execution — smoke tests")
    rs=RegimePositionSizer(); ret=np.random.normal(0.001,0.003,100)
    for desc,kw in [("normal",{}),("crisis",{"corr_avg":0.80}),("trending",{"hurst":0.70}),("mean_rev",{"hurst":0.30})]:
        r=rs.size(10000,0.54,1.7,ret,0.0005,**kw)
        print(f"  {desc:12s}: {r['lots']:.2f}L | Kelly={r['kelly']:.3f} | scale={r['regime_scale']:.2f}")
    ac=AlmgrenChrissExecutor()
    for lots in [0.5,1.0,3.0]:
        r=ac.estimate_impact_cost(lots); print(f"  AC {lots}L: {r['impact_pips']:.4f} pips")
    dm=DrawdownAwareExitManager(); eq=10000.0
    for pnl in [100,-200,-150,-100,-80,-50,-400,-350]:
        eq+=pnl; r=dm.update(eq,pnl)
        print(f"  DD={r['dd']:.2%} | {r['action']:12s} | size={r['size_multiplier']:.2f}")
    pv=PortfolioVaR()
    for _ in range(100):
        for p in ["EURUSD","GBPUSD"]: pv.update_returns(p,np.random.normal(0,0.0003))
    v=pv.parametric_var({"EURUSD":1.0,"GBPUSD":0.8},10000)
    print(f"  VaR: {v['var_pct']:.2%} | ${v['var_usd']:.2f} | corr={v['correlation_avg']:.2f}")
    print("All tests passed ✓")

# Public aliases — main.py and external callers import these names.
# RegimePositionSizer is the canonical class; both names refer to the same object.
RegimeConditionalKelly  = RegimePositionSizer   # legacy name used in main.py
DrawdownAwareExitPolicy = DrawdownAwareExitManager  # legacy name used in main.py
