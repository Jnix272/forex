"""Offline adapter around the actual zoo inference implementation.

Never imports a broker or starts LiveTradingEngine. Inference parity and
simulated execution parity are reported separately.
"""
from __future__ import annotations

import numpy as np

from trading.live_actions import LiveAction
from trading.zoo_policy import ReturnFit, Scale, hysteresis, portfolio_weights, specialist_edge


class ResearchZooAdapter:
    """Record real model outputs; optionally apply an explicit research policy.

    The uncalibrated zoo calls its actual select_action verbatim. A temporary
    instance-level score converter permits experimental calibration without
    editing the production inference class or any production checkpoint.
    """
    def __init__(self, engine, policy=None, component='zoo'):
        self.engine, self.policy, self.component = engine, policy, component
        self.raw_edges, self.scores, self.errors = {}, {}, []
        self.counter, self.last_target = 0, 0
        self.base_weights=tuple(engine.weights)
        self.original_converter=engine._proba_to_z
        for name in ('haelt','mamba'):
            member=getattr(engine,f'{name}_engine')
            if member is not None:
                setattr(engine,f'{name}_engine',self._recording_member(member,name))
        engine._proba_to_z=self._convert
        if component != 'zoo':
            names=('stationary','haelt','mamba')
            if component not in names:
                raise ValueError('Unknown component')
            if component != 'stationary' and getattr(engine,f'{component}_engine') is None:
                raise ValueError(f'Component unavailable: {component}')
            engine.weights=tuple(float(n == component) for n in names)
        elif policy is not None and policy.get('weights') is not None:
            w=np.asarray(policy['weights'],dtype=float)
            if w.shape != (3,) or np.any(w < 0) or not np.isfinite(w).all() or not np.isclose(w.sum(),1):
                raise ValueError('Invalid research blend weights')
            for i,name in enumerate(('haelt','mamba'),start=1):
                if w[i] and getattr(engine,f'{name}_engine') is None:
                    raise ValueError('Research blend requires a missing model')
            engine.weights=tuple(w)

    def _recording_member(self, member, name):
        owner=self
        class Recorder:
            def __getattr__(self,key):
                return getattr(member,key)
            def predict_proba(self,window):
                owner.active=name
                try:
                    p=member.predict_proba(window)
                    owner.raw_edges[name]=specialist_edge(p)
                    return p
                except Exception as exc:
                    owner.errors.append((name,str(exc)))
                    raise
        return Recorder()

    def _convert(self, probabilities):
        name=self.active
        if self.policy is None:
            z=self.original_converter(probabilities)
        else:
            z=Scale(**self.policy['scales'][name]).apply(specialist_edge(probabilities))
        self.scores[name]=float(z)
        return z

    def step(self, obs, *, t_ns, position, cost_bps, entry_allowed=True):
        self.raw_edges, self.scores, self.errors = {}, {}, []
        self.engine.set_agent_state(position_lots=position)
        actual=int(self.engine.select_action(obs))
        if self.errors:
            raise RuntimeError(f'Model errors cannot silently become zero signals in replay: {self.errors}')
        self.raw_edges['stationary']=float(self.engine.cs_engine.last_z_score)
        self.scores['stationary']=self.raw_edges['stationary']
        if not np.isfinite(list(self.scores.values())).all():
            raise ValueError('Nonfinite replay signal')
        self.counter+=1
        if self.policy is None:
            if not entry_allowed:
                target={int(LiveAction.BUY):1,int(LiveAction.SELL):-1}.get(actual,0)
                if target and target != position:
                    actual=int(LiveAction.CLOSE if position else LiveAction.HOLD)
            return actual
        if int(t_ns) < self.policy['validate_start_ns']:
            raise ValueError('Policy used before the end of calibration')
        interval=int(self.policy['decision_interval'])
        if interval < 1:
            raise ValueError('Decision interval must be positive')
        if (self.counter-1) % interval:
            # Preserve the filled position between decisions, not a stale target
            # whose execution may have been blocked by a guard.
            self.engine._pos_dir=position
            return int(LiveAction.BUY if position > 0 else LiveAction.SELL if position < 0 else LiveAction.HOLD)
        z=float(self.engine.last_z_score)
        desired=1 if z >= 0 else -1
        fit=ReturnFit(**self.policy['return_fit'])
        if not np.isfinite(cost_bps) or cost_bps < 0:
            raise ValueError('Known execution and financing costs are required')
        allowed=entry_allowed and fit.expected(z,desired) > cost_bps + fit.uncertainty_bps + self.policy['min_edge_bps']
        target,action=hysteresis(z,position,self.policy['entry'],self.policy['exit'],allowed)
        self.engine._pos_dir=target
        return action


def usd_pnl(pair, units, entry, price):
    pnl=units*(price-entry)
    if pair.endswith('USD'):
        return pnl
    if pair.startswith('USD'):
        return pnl/price
    raise ValueError('Explicit USD conversion is required for cross pairs')


def validate_market(t_ns, close, spread, funding_bps, pairs):
    t=np.asarray(t_ns,dtype=np.int64)
    c,s,f=np.asarray(close),np.asarray(spread),np.asarray(funding_bps)
    if len(t)<3 or np.any(np.diff(t)<=0):
        raise ValueError('Market timestamps must increase strictly')
    if c.shape != (len(t),len(pairs)) or s.shape != c.shape or f.shape != (*c.shape,2):
        raise ValueError('Prices, spreads, and signed financing must share timestamps and pair order')
    if not np.isfinite(c).all() or np.any(c<=0) or not np.isfinite(s).all() or np.any(s<0) or not np.isfinite(f).all():
        raise ValueError('Missing/invalid market data cannot become free execution')


def summarize_equity(t_ns, equity, turnover, trades):
    from training.honest_eval import bootstrap_sharpe_ci
    t=np.asarray(t_ns,dtype='datetime64[ns]')
    days=t.astype('datetime64[D]')
    # Sunday opening PnL belongs to Monday, never disappears from the sample.
    buckets=np.busday_offset(days,0,roll='forward')
    calendar=np.arange(buckets[0],buckets[-1]+np.timedelta64(1,'D'))
    calendar=calendar[np.is_busday(calendar)]
    daily=np.full(len(calendar),np.nan)
    for day,value in zip(buckets,equity):
        daily[np.searchsorted(calendar,day)]=value
    previous=float(equity[0])
    for i in range(len(daily)):
        if np.isnan(daily[i]):
            daily[i]=previous
        previous=daily[i]
    ret=np.diff(np.r_[equity[0],daily])/np.r_[equity[0],daily[:-1]]
    std=ret.std(ddof=1) if len(ret)>1 else 0.
    sr=float(ret.mean()/std*np.sqrt(252)) if std>0 else None
    centered=(ret-ret.mean())/(ret.std()+1e-15)
    ci=[None,None]
    if len(ret)>=30 and std>0:
        lo,hi,_=bootstrap_sharpe_ci(ret,np.sqrt(252),block=10)
        ci=[lo,hi]
    peak=np.maximum.accumulate(equity)
    return {'sharpe_daily':sr,'ci95_block10d':ci,'n_business_days':len(ret),
            'skewness':float(np.mean(centered**3)), 'kurtosis':float(np.mean(centered**4)),
            'net_return':float(equity[-1]/equity[0]-1),
            'max_drawdown':float(np.max((peak-equity)/peak)),
            'turnover_usd':float(turnover),'entries':int(trades),
            'positive_edge_evidence':bool(len(ret)>=60 and trades>=30 and ci[0] is not None and ci[0]>0)}


def replay(adapters, observations, t_ns, close, spread, funding_bps, pairs, *,
           initial_equity=100_000., slippage_pips=.7, commission_bps=0.,
           trade_allowed=None, mode='equal', verified_edge=False, cost_mult=1.,
           expected_financing_bps=None):
    """One-bar delayed mid fills, half spread/slippage each leg, exact USD MTM.

    Funding is a required signed LONG/SHORT cost in bps of notional for each
    observation interval (negative values represent credits). Trade guards are
    an explicit point-in-time boolean array. TIP, FIFO, stops and broker fill
    mechanics are outside this policy replay's execution model.
    """
    validate_market(t_ns,close,spread,funding_bps,pairs)
    t,c,s,f=np.asarray(t_ns),np.asarray(close),np.asarray(spread),np.asarray(funding_bps)
    if len(adapters)!=len(pairs) or len(observations)!=len(t):
        raise ValueError('Adapters and observations must align with the market')
    allowed=np.ones(c.shape,dtype=bool) if trade_allowed is None else np.asarray(trade_allowed)
    if allowed.dtype.kind!='b':
        raise ValueError('Entry guards must be explicit booleans, not missing numeric values')
    if allowed.shape != c.shape:
        raise ValueError('Entry guards must align with quotes')
    if expected_financing_bps is None:
        if any(getattr(a,'policy',None) is not None for a in adapters):
            raise ValueError('Cost-aware policies require a point-in-time financing forecast')
        expected_funding=np.zeros_like(c)
    else:
        expected_funding=np.asarray(expected_financing_bps,dtype=float)
        if expected_funding.shape!=c.shape or not np.isfinite(expected_funding).all():
            raise ValueError('Expected financing must align with quotes')
    if initial_equity<=0 or min(slippage_pips,commission_bps)<0 or cost_mult<=0:
        raise ValueError('Invalid equity or execution-cost settings')
    n,k=c.shape
    cash=initial_equity
    units=np.zeros(k)
    entry=c[0].copy()
    pending=np.zeros(k)
    equity=np.full(n,initial_equity)
    turnover=0.
    trades=0
    edge_trace={p:[] for p in pairs}
    scores={p:[] for p in pairs}
    day_returns=[]
    prior_day_close=c[0].copy()
    entry_events=np.zeros(n,dtype=int)
    turnover_events=np.zeros(n)
    current_day=np.datetime64(int(t[0]),'ns').astype('datetime64[D]')
    for i in range(n):
        prices=c[i]
        unrealized=np.array([usd_pnl(p,units[j],entry[j],prices[j]) for j,p in enumerate(pairs)])
        pre_equity=cash+unrealized.sum()
        if pre_equity<=0:
            raise ValueError('Simulated account equity exhausted')
        for j,pair in enumerate(pairs):
            direction=int(np.sign(units[j]))
            if direction:
                notional=abs(units[j])*(prices[j] if pair.endswith('USD') else 1.)
                cash-=notional*f[i,j,0 if direction>0 else 1]/1e4
            target=pending[j]
            target_units=target*pre_equity/(prices[j] if pair.endswith('USD') else 1.)
            # Rebalance only when direction changes. Holding a signal must not
            # create a fresh order every bar merely because equity fluctuates.
            if np.sign(target_units)==np.sign(units[j]):
                continue
            if target and not allowed[i,j]:
                target_units=0.
            delta=target_units-units[j]
            pip=.01 if 'JPY' in pair else .0001
            conversion=1. if pair.endswith('USD') else prices[j]
            cost=cost_mult*(abs(delta)*(.5*s[i,j]+.5*slippage_pips*pip)/conversion
                           +abs(delta)*(prices[j] if pair.endswith('USD') else 1.)*.5*commission_bps/1e4)
            cash+=unrealized[j]-cost
            turnover+=abs(delta)*(prices[j] if pair.endswith('USD') else 1.)
            turnover_events[i]+=abs(delta)*(prices[j] if pair.endswith('USD') else 1.)
            if target_units:
                trades+=1
                entry_events[i]+=1
            units[j]=target_units
            entry[j]=prices[j]
        equity[i]=cash+sum(usd_pnl(p,units[j],entry[j],prices[j]) for j,p in enumerate(pairs))
        day=np.datetime64(int(t[i]),'ns').astype('datetime64[D]')
        if day!=current_day:
            # Only completed prior-day prices may inform today's weights.
            if i>0:
                day_returns.append(c[i-1]/prior_day_close-1.)
                prior_day_close=c[i-1].copy()
            current_day=day
        directions=[]
        for j,(pair,adapter) in enumerate(zip(pairs,adapters)):
            pip=.01 if 'JPY' in pair else .0001
            round_trip=cost_mult*((s[i,j]+slippage_pips*pip)/prices[j]*1e4+commission_bps)
            # The caller supplies a conservative forecast of upcoming funding
            # in policy cost context; realized financing is charged separately.
            round_trip+=max(0.,float(expected_funding[i,j]))
            action=adapter.step(observations[i],t_ns=int(t[i]),position=int(np.sign(units[j])),
                                cost_bps=round_trip,entry_allowed=bool(allowed[i,j]))
            directions.append(1 if action==int(LiveAction.BUY) else -1 if action==int(LiveAction.SELL) else
                              0 if action==int(LiveAction.CLOSE) else int(np.sign(units[j])))
            edge_trace[pair].append(dict(adapter.raw_edges))
            scores[pair].append(dict(adapter.scores))
        pending=portfolio_weights(directions,day_returns,pairs,mode=mode,verified_edge=verified_edge)
    # Liquidate at the final observed quote, charging the exit leg.
    for j,pair in enumerate(pairs):
        pip=.01 if 'JPY' in pair else .0001
        conv=1. if pair.endswith('USD') else c[-1,j]
        equity[-1]-=cost_mult*abs(units[j])*(.5*s[-1,j]+.5*slippage_pips*pip)/conv
        equity[-1]-=cost_mult*abs(units[j])*(c[-1,j] if pair.endswith('USD') else 1.)*.5*commission_bps/1e4
        closing_notional=abs(units[j])*(c[-1,j] if pair.endswith('USD') else 1.)
        turnover+=closing_notional
        turnover_events[-1]+=closing_notional
    metrics=summarize_equity(t,equity,turnover,trades)
    return {'metrics':metrics,'equity':equity,'raw_edges':edge_trace,'scores':scores,
            'entry_events':entry_events,'turnover_events':turnover_events,
            'scope':'Actual zoo inference with simulated next-bar execution; excludes TIP, broker/FIFO and stop mechanics.'}
