"""Shared zoo policy primitives for inference and chronological research.

Experimental artifacts are explicit opt-ins; no production artifact is written
or discovered automatically by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from trading.live_actions import LiveAction


def specialist_edge(proba):
    p = np.asarray(proba, dtype=float).reshape(-1)
    if p.size != 3 or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1, atol=1e-4):
        raise ValueError('Expected finite SELL/HOLD/BUY probabilities summing to one')
    return float((p[2] - p[0]) * max(p[0] + p[2], 1e-6))


def hysteresis(z, position, entry=1.5, exit=-.3, entry_allowed=True):
    """One action contract for the stationary engine, zoo, and replay."""
    if not np.isfinite(z) or position not in (-1, 0, 1):
        raise ValueError('Invalid signal or position')
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(exit) or exit >= entry:
        raise ValueError('Invalid hysteresis thresholds')
    target = position
    if position == 0:
        target = 1 if z >= entry else (-1 if z <= -entry else 0)
    elif position == 1:
        target = -1 if z <= -entry else (0 if z < exit else 1)
    else:
        target = 1 if z >= entry else (0 if -z < exit else -1)
    if target and target != position and not entry_allowed:
        target = 0  # a rejected reversal must still permit closing the old side
    if target:
        action = LiveAction.BUY if target > 0 else LiveAction.SELL
    else:
        action = LiveAction.CLOSE if position else LiveAction.HOLD
    return target, int(action)


@dataclass(frozen=True)
class Scale:
    center: float
    scale: float

    def __post_init__(self):
        if not np.isfinite([self.center,self.scale]).all() or self.scale<=0:
            raise ValueError('Invalid calibration scale')

    def apply(self, edge):
        if not np.isfinite([self.center, self.scale, edge]).all() or self.scale <= 0:
            raise ValueError('Invalid calibration scale')
        return float(np.clip((edge - self.center) / self.scale, -3.5, 3.5))


@dataclass(frozen=True)
class ReturnFit:
    intercept_bps: float
    slope_bps: float
    uncertainty_bps: float

    def __post_init__(self):
        if not np.isfinite([self.intercept_bps,self.slope_bps,self.uncertainty_bps]).all() or self.uncertainty_bps<0:
            raise ValueError('Invalid expected-return uncertainty')

    def expected(self, z, direction):
        return float(direction * (self.intercept_bps + self.slope_bps * z))


def fit_scale(edges):
    x = np.asarray(edges, dtype=float)
    if len(x) < 30 or not np.isfinite(x).all() or x.std(ddof=1) < 1e-6:
        raise ValueError('Insufficient varying calibration predictions')
    return Scale(float(x.mean()), float(x.std(ddof=1)))


def fit_return(z, future_gross_bps):
    x, y = np.asarray(z, dtype=float), np.asarray(future_gross_bps, dtype=float)
    if x.shape != y.shape or len(x) < 30 or not np.isfinite([x, y]).all():
        raise ValueError('Invalid train-only return calibration samples')
    design = np.column_stack([np.ones(len(x)), x])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coef
    # Samples supplied by the caller must be nonoverlapping. This is estimation
    # uncertainty, not a probability that an individual trade will profit.
    uncertainty = 1.96 * residual.std(ddof=1) / np.sqrt(len(x))
    return ReturnFit(float(coef[0]), float(coef[1]), float(uncertainty))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_policy(path, *, pair, scales, return_fit, fit_start_ns, fit_end_ns,
                validate_start_ns, model_trained_through_ns, artifact_hashes,
                entry=1.5, exit=-.3, decision_interval=1, weights=None, min_edge_bps=0.):
    if not (model_trained_through_ns < fit_start_ns <= fit_end_ns < validate_start_ns):
        raise ValueError('Model training, calibration, and validation must be disjoint and chronological')
    if not artifact_hashes or any(sha256(p) != h for p, h in artifact_hashes.items()):
        raise ValueError('Missing or mismatched bound model artifacts')
    if not np.isfinite([decision_interval,min_edge_bps]).all() or decision_interval < 1 or int(decision_interval)!=decision_interval or min_edge_bps < 0:
        raise ValueError('Invalid decision interval or cost buffer')
    payload = {'version':1, 'pair':pair, 'scales':{k:asdict(v) for k,v in scales.items()},
               'return_fit':asdict(return_fit), 'fit_start_ns':int(fit_start_ns),
               'fit_end_ns':int(fit_end_ns), 'validate_start_ns':int(validate_start_ns),
               'model_trained_through_ns':int(model_trained_through_ns),
               'artifact_hashes':artifact_hashes, 'entry':entry, 'exit':exit,
               'decision_interval':int(decision_interval),'weights':weights,
               'min_edge_bps':min_edge_bps, 'experimental':True}
    target=Path(path)
    if target.exists():
        raise FileExistsError('Refusing to overwrite a policy artifact')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(payload,indent=2,allow_nan=False))


def load_policy(path, pair, required_artifacts):
    p=json.loads(Path(path).read_text())
    if p.get('version') != 1 or p.get('pair') != pair:
        raise ValueError('Policy version or pair mismatch')
    bound=p.get('artifact_hashes',{})
    for artifact in required_artifacts:
        key=str(Path(artifact).resolve())
        if key not in bound or sha256(key) != bound[key]:
            raise ValueError(f'Policy does not bind the loaded model artifact: {key}')
    if any(sha256(k) != v for k,v in bound.items()):
        raise ValueError('Policy artifact hashes changed')
    if not (p['model_trained_through_ns'] < p['fit_start_ns'] <= p['fit_end_ns'] < p['validate_start_ns']):
        raise ValueError('Policy calibration overlaps evaluation or model fitting')
    hysteresis(0.,0,p['entry'],p['exit'])
    if int(p['decision_interval']) < 1 or p['min_edge_bps'] < 0:
        raise ValueError('Invalid policy decision settings')
    for value in p['scales'].values():
        Scale(**value).apply(0.)
    ReturnFit(**p['return_fit'])
    return p


def portfolio_weights(directions, trailing_daily_returns, pairs, *, mode='equal',
                      verified_edge=False, gross_cap=1., pair_cap=.4, usd_cap=.5):
    d=np.asarray(directions,dtype=float)
    if not np.isin(d,[-1,0,1]).all() or len(d) != len(pairs):
        raise ValueError('Invalid portfolio directions')
    if mode not in ('equal','inverse_vol'):
        raise ValueError('Unknown sizing mode')
    if mode == 'inverse_vol' and not verified_edge:
        raise ValueError('Volatility sizing requires positive validation evidence')
    if not np.isfinite([gross_cap,pair_cap,usd_cap]).all() or min(gross_cap,pair_cap,usd_cap) <= 0:
        raise ValueError('Exposure caps must be positive')
    w=np.ones(len(d))
    history=np.asarray(trailing_daily_returns,dtype=float)
    if mode == 'inverse_vol' and len(history) >= 40:
        if history.ndim != 2 or history.shape[1] != len(d) or not np.isfinite(history).all():
            raise ValueError('Invalid trailing volatility history')
        w=1/np.maximum(history[-40:].std(axis=0,ddof=1),1e-4)
    w=w/w.sum()*gross_cap
    w=np.minimum(w,pair_cap)*d
    if any(not (p.endswith('USD') or p.startswith('USD')) for p in pairs):
        raise ValueError('Explicit currency exposure conversion is required for cross pairs')
    usd=np.array([-1 if p.endswith('USD') else 1 for p in pairs])
    net=abs(float(w @ usd))
    if net > usd_cap:
        w*=usd_cap/net
    return w
