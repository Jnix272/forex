"""Extract actual zoo signals, then tune only with chronological lineage.

No live-engine edits, broker calls, publication, or automatic deployment.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
for stream in (sys.stdout,sys.stderr):
    if hasattr(stream,'reconfigure'):
        stream.reconfigure(encoding='utf-8',errors='replace')

import numpy as np

from backtesting.zoo_replay import ResearchZooAdapter, replay, summarize_equity
from trading.live_actions import LiveAction
from trading.zoo_policy import Scale, ReturnFit, fit_scale, fit_return, hysteresis, save_policy, sha256

NAMES=('stationary','haelt','mamba')


def ns(date):
    return int(np.datetime64(date,'ns').astype(np.int64))


def assemble(caches,out):
    """Inner-join corrected single-pair bar features without zero-padding peers."""
    import zarr
    from feature_store.fingerprint import feature_code_fingerprint
    from scripts.research_sharpe_baseline import read_last_features
    pairs=['EURUSD','USDJPY','USDCAD']
    groups=[]; times=[]; schemas=[]; sources={}
    for pair,path in zip(pairs,caches):
        manifest_path=Path(str(path)+'_manifest.json')
        manifest=json.loads(manifest_path.read_text())
        if manifest.get('code_fingerprint')!=feature_code_fingerprint() or manifest.get('pairs')!=[pair]:
            raise ValueError(f'Corrected, explicitly identified cache required for {pair}')
        schema=json.loads(Path(str(path)+'_feature_schema.json').read_text())
        schema=[name.split('::')[-1] for name in schema]
        g=zarr.open(str(path),mode='r')
        if 't_ns' not in g or g['X'].shape[-1]!=146:
            raise ValueError('Single-pair cache lacks the canonical timestamp/feature contract')
        t=np.asarray(g['t_ns'][:],dtype=np.int64)
        if np.any(np.diff(t)<=0):
            raise ValueError('Source timestamps must increase strictly')
        if schemas and schema!=schemas[0]:
            raise ValueError('Pair feature order differs; refusing implicit reordering')
        groups.append(g); times.append(t); schemas.append(schema)
        sources[pair]={'path':str(path.resolve()),'manifest_sha256':sha256(manifest_path),
                       'rows':len(t),'content_hash':manifest.get('content_hash')}
    common=times[0]
    for t in times[1:]:
        common=np.intersect1d(common,t,assume_unique=True)
    if len(common)<1000:
        raise ValueError('Insufficient common corrected timestamps')
    if out.exists():
        raise FileExistsError('Refusing to replace an existing dataset')
    result=zarr.open_group(str(out),mode='w',zarr_format=2)
    x=result.create_array('X',shape=(len(common),1,438),chunks=(512,1,438),dtype='float32')
    close=np.empty((len(common),3),dtype='float64')
    spread=np.empty_like(close)
    for j,(g,t) in enumerate(zip(groups,times)):
        rows=np.searchsorted(t,common)
        values=read_last_features(g,rows)
        x[:,0,j*146:(j+1)*146]=values
        close[:,j]=np.asarray(g['close'][:])[rows]
        spread[:,j]=np.asarray(g['spread'][:])[rows]
        print(f'Joined {pairs[j]}: {len(common)} synchronized bar-feature rows',flush=True)
    for name,values in [('t_ns',common),('close_pairs',close),('spread_pairs',spread)]:
        result.create_array(name,data=values)
    meta={'pairs':pairs,'total_samples':len(common),'n_features':438,'seq_len':1,
          'source_seq_len':120,'code_fingerprint':feature_code_fingerprint(),'sources':sources,
          'purpose':'Bar-feature replay; actual zoo reconstructs its rolling history.',
          'alignment':'Strict timestamp inner join; no synthetic peer rows.'}
    Path(str(out)+'_manifest.json').write_text(json.dumps(meta,indent=2))


def artifact_paths(engine):
    paths=[engine.cs_engine.model_dir/'metadata.json',engine.cs_engine.model_dir/'cs_policy_weights.npz',
           engine.cs_engine.model_dir/f'{engine.pair.lower()}_4h_lgb.txt']
    for name in ('haelt','mamba'):
        member=getattr(engine,f'{name}_engine')
        if member is not None:
            cp=Path(member.checkpoint_path)
            paths.extend([cp,cp.with_name(cp.stem+'_scaler.npz'),cp.with_name(cp.stem+'_features.json'),
                          cp.with_suffix(cp.suffix+'.metadata.json')])
            paths.extend(cp.parent.glob('*_config.json'))
    return [p.resolve() for p in paths if p.is_file()]


def extract(cache,out,pairs,max_bars,plumbing_only,lineage=None):
    import torch
    import zarr
    from feature_store.fingerprint import feature_code_fingerprint
    from trading.inference_engines import MultiArchitectureSpecialistEnsembleInferenceEngine
    manifest=json.loads(Path(str(cache)+'_manifest.json').read_text())
    if not plumbing_only and manifest.get('code_fingerprint')!=feature_code_fingerprint():
        raise ValueError('Full research requires a corrected cache; stale caches are plumbing-only')
    if not plumbing_only and max_bars:
        raise ValueError('Capped extracts are plumbing-only, not performance evidence')
    source_pairs=manifest.get('pairs')
    if not source_pairs or any(p not in source_pairs for p in pairs):
        raise ValueError('Source manifest must explicitly identify requested pair slots')
    root=zarr.open(str(cache),mode='r')
    if 't_ns' not in root or 'close_pairs' not in root or 'spread_pairs' not in root:
        raise ValueError('Timestamp-aligned multi-pair market sidecars are required')
    n=min(root['X'].shape[0],max_bars) if max_bars else root['X'].shape[0]
    times=np.asarray(root['t_ns'][:n],dtype=np.int64)
    if np.any(np.diff(times)<=0):
        raise ValueError('Duplicate/out-of-order timestamps cannot be replayed')
    out.mkdir(parents=True,exist_ok=False)
    raw=np.full((n,len(pairs),3),np.nan)
    actions=np.ones((n,len(pairs)),dtype='int8')
    legacy_scores=np.zeros((n,len(pairs)))
    hysteresis_params={}
    adapters=[]; bindings={}; weights=[]
    for pair in pairs:
        engine=MultiArchitectureSpecialistEnsembleInferenceEngine(pair=pair,device=torch.device('cpu'))
        paths=artifact_paths(engine)
        bindings[pair]={str(p):sha256(p) for p in paths}
        weights.append(engine.weights)
        hysteresis_params[pair]={'entry':engine.z_entry,'exit':engine.z_exit}
        adapters.append(ResearchZooAdapter(engine))
    positions=np.zeros(len(pairs),dtype=int)
    for start in range(0,n,128):
        block=np.asarray(root['X'][start:min(n,start+128)])
        for local,window in enumerate(block):
            i=start+local
            obs=window[-1]
            if obs.size==438 and source_pairs!=['EURUSD','USDJPY','USDCAD']:
                raise ValueError('438-feature source must have the exact live 3-pair slot order')
            if obs.size not in (438,584):
                raise ValueError('Replay refuses unknown feature layouts')
            if obs.size==584 and source_pairs!=['EURUSD','USDJPY','GBPUSD','USDCAD']:
                raise ValueError('584-feature source must have canonical four-pair order')
            # Explicitly reuse the live engine's existing 3-to-4 expansion.
            expanded=adapters[0].engine.cs_engine._expand_to_584(obs)
            for j,adapter in enumerate(adapters):
                act=adapter.step(expanded,t_ns=int(times[i]),position=int(positions[j]),cost_bps=0.)
                actions[i,j]=act
                legacy_scores[i,j]=float(adapter.engine.last_z_score)
                positions[j]=1 if act==int(LiveAction.BUY) else -1 if act==int(LiveAction.SELL) else 0 if act==int(LiveAction.CLOSE) else positions[j]
                for k,name in enumerate(NAMES):
                    if name in adapter.raw_edges:
                        raw[i,j,k]=adapter.raw_edges[name]
        print(f'Extracted {min(n,start+128)}/{n} actual zoo observations',flush=True)
    for hashes in bindings.values():
        if any(sha256(p)!=h for p,h in hashes.items()):
            raise ValueError('Model artifacts changed during extraction')
    slots=[source_pairs.index(p) for p in pairs]
    np.savez_compressed(out/'signals.npz',raw=raw,actions=actions,t_ns=times,
                        close=np.asarray(root['close_pairs'][:n])[:,slots],
                        spread=np.asarray(root['spread_pairs'][:n])[:,slots],weights=np.asarray(weights),
                        legacy_scores=legacy_scores)
    provenance={'pairs':pairs,'source_cache':str(cache.resolve()),'source_manifest_sha256':sha256(Path(str(cache)+'_manifest.json')),
                'source_code_fingerprint':manifest.get('code_fingerprint'),'artifact_hashes':bindings,
                'plumbing_only':plumbing_only,'rows':n,'warmup_rows':max(720,120),
                'inference_source_sha256':sha256(ROOT/'trading/inference_engines.py'),
                'hysteresis_params':hysteresis_params,
                'feature_expansion':'Actual live expansion; absent GBPUSD is proxied from EURUSD for 438-feature inputs.',
                'limitation':'Recorded actions are plumbing diagnostics. Account performance requires delayed-fill replay with explicit funding and entry guards.'}
    if lineage is not None:
        provenance['lineage']=json.loads(lineage.read_text())
        provenance['lineage_file_sha256']=sha256(lineage)
    (out/'signals.json').write_text(json.dumps(provenance,indent=2))


def verify_lineage(meta,fit_start):
    """Require hash-bound training evidence, not a manually supplied date alone."""
    if meta.get('plumbing_only'):
        raise ValueError('Plumbing-only signals cannot be used for calibration or policy selection')
    lineage=meta.get('lineage',{})
    latest=0
    for pair,bound in meta['artifact_hashes'].items():
        item=lineage.get(pair)
        if not item:
            raise ValueError(f'No verified model training-window evidence for {pair}')
        evidence_path=Path(item['evidence_file'])
        if sha256(evidence_path)!=item['evidence_sha256']:
            raise ValueError('Training evidence changed')
        evidence=json.loads(evidence_path.read_text())
        if evidence.get('artifact_hashes')!=bound:
            raise ValueError('Training evidence does not bind the extracted model/scaler artifacts')
        # Evidence producer must record the actual latest training sample plus
        # its label horizon, not just a nominal dataset start/end date.
        cutoff=int(evidence['latest_training_label_t_ns'])
        if cutoff>=fit_start:
            raise ValueError('Models trained into calibration data')
        latest=max(latest,cutoff)
        if any(sha256(p)!=h for p,h in bound.items()):
            raise ValueError('Bound inference artifacts changed')
    return latest


class ScoreTape:
    """Frozen real-model predictions with the tested shared policy semantics."""
    def __init__(self,scores,policy):
        self.values=scores
        self.policy=policy
        self.i=0
        self.raw_edges={}; self.scores={}
    def step(self,obs,*,t_ns,position,cost_bps,entry_allowed=True):
        z=float(self.values[self.i]); self.i+=1
        self.scores={'selected':z}
        if (self.i-1)%self.policy['decision_interval']:
            return int(LiveAction.BUY if position>0 else LiveAction.SELL if position<0 else LiveAction.HOLD)
        fit=ReturnFit(**self.policy['return_fit'])
        desired=1 if z>=0 else -1
        allowed=entry_allowed and fit.expected(z,desired)>cost_bps+fit.uncertainty_bps+self.policy['min_edge_bps']
        return hysteresis(z,position,self.policy['entry'],self.policy['exit'],allowed)[1]


class LegacyScoreTape:
    """Actual cached zoo score with the exact extracted live thresholds."""
    policy=None
    raw_edges={}; scores={}
    def __init__(self,values,entry,exit):
        self.values=values; self.entry=entry; self.exit=exit; self.i=0
    def step(self,obs,*,position,entry_allowed=True,**kwargs):
        z=float(self.values[self.i]); self.i+=1
        return hysteresis(z,position,self.entry,self.exit,entry_allowed)[1]


def calibrated_scores(raw,scales,weights):
    z=np.zeros(raw.shape[:2])
    for j in range(raw.shape[1]):
        for k,name in enumerate(NAMES):
            if weights[j,k]==0:
                continue
            x=raw[:,j,k]
            if not np.isfinite(x).all():
                raise ValueError('Required model has missing predictions after warmup')
            values=x if k==0 else np.array([scales[j][name].apply(v) for v in x])
            z[:,j]+=weights[j,k]*values
    return z


def tune(signals,market,out,fit_start,val_start,test_start,test_end):
    from dataclasses import asdict
    from feature_store.fingerprint import feature_code_fingerprint
    meta=json.loads(signals.with_suffix('.json').read_text())
    if meta.get('source_code_fingerprint')!=feature_code_fingerprint():
        raise ValueError('Signal cache was not built with the current feature/label source')
    if meta.get('inference_source_sha256')!=sha256(ROOT/'trading/inference_engines.py'):
        raise ValueError('Inference implementation changed after signal extraction')
    latest=verify_lineage(meta,ns(fit_start))
    if not ns(fit_start)<ns(val_start)<ns(test_start)<ns(test_end):
        raise ValueError('Calibration, validation and test periods must be ordered')
    data=np.load(signals); costs=np.load(market)
    t=data['t_ns']; pairs=meta['pairs']; raw=data['raw']; base=data['weights']
    if not np.array_equal(costs['t_ns'],t) or list(costs['pairs'])!=pairs:
        raise ValueError('Financing and entry guards must match every signal timestamp and pair slot')
    funding=costs['funding_bps']; allowed=costs['trade_allowed']
    expected_funding=costs['expected_financing_bps']
    commission=float(costs['commission_bps'])
    slippage=float(costs['slippage_pips'])
    if funding.shape!=(*data['close'].shape,2) or allowed.shape!=data['close'].shape:
        raise ValueError('Explicit funding and entry guards are required')
    # Complete, nonoverlapping forward calibration outcomes, strictly before validation.
    rows=np.flatnonzero((t>=ns(fit_start))&(t<ns(val_start)))
    rows=rows[rows>=meta['warmup_rows']][::30]
    rows=rows[(rows+30<len(t))&(t[np.minimum(rows+30,len(t)-1)]<ns(val_start))]
    rows=rows[(t[rows+30]-t[rows])==30*300_000_000_000]
    rows=rows[rows+151<int(np.searchsorted(t,ns(val_start)))]
    if len(rows)<100:
        raise ValueError('Insufficient nonoverlapping, gap-free calibration outcomes')
    scales=[]
    for j in range(len(pairs)):
        scales.append({name:fit_scale(raw[rows,j,k]) for k,name in enumerate(NAMES) if k and base[j,k]>0})
    valid=np.flatnonzero((t>=ns(val_start))&(t<ns(test_start)))
    test=np.flatnonzero((t>=ns(test_start))&(t<ns(test_end)))
    if not len(valid) or not len(test) or valid[0]<meta['warmup_rows']:
        raise ValueError('Missing warmed-up validation or test observations')
    out.mkdir(parents=True,exist_ok=False)
    report={'status':'research_only','pairs':pairs,'trials':[],
            'scope':'Actual zoo prediction cache; simulated one-bar-delayed fills, not full live TIP/broker replay.',
            'test_limitation':'Previously inspected periods remain diagnostic; no automatic promotion.',
            'periods':{'fit_start':fit_start,'validation_start':val_start,'test_start':test_start,'test_end':test_end}}
    report['reference']={}
    for period,ids in [('validation',valid),('diagnostic_test',test)]:
        adapters=[LegacyScoreTape(data['legacy_scores'][ids,j],**meta['hysteresis_params'][pair])
                  for j,pair in enumerate(pairs)]
        ref=replay(adapters,np.zeros(len(ids)),t[ids],data['close'][ids],data['spread'][ids],funding[ids],
                   pairs,trade_allowed=allowed[ids],commission_bps=commission,slippage_pips=slippage)
        report['reference'][period]=ref['metrics']
    report['component_coverage']={pair:[NAMES[k] for k in range(3) if base[j,k]>0]
                                  for j,pair in enumerate(pairs)}
    variants={'zoo':base}
    for k,name in enumerate(NAMES):
        if np.any(base[:,k]>0):
            w=np.zeros_like(base); w[:,k]=(base[:,k]>0).astype(float); variants[name]=w
        w=base.copy(); w[:,k]=0
        if np.all(w.sum(axis=1)>0):
            variants[f'without_{name}']=w/w.sum(axis=1,keepdims=True)
    planned_trials=len(variants)*18
    from validation.promotion_gate import deflated_sharpe_ratio
    best=None
    for label,weights in variants.items():
        # Missing model columns are unused, not silently zero-filled.
        z=calibrated_scores(raw[meta['warmup_rows']:],scales,weights)
        offset=meta['warmup_rows']
        fits=[]
        for j,pair in enumerate(pairs):
            p=data['close']
            denominator=p[rows,j] if pair.endswith('USD') else p[rows+30,j]
            target=(p[rows+30,j]-p[rows,j])/denominator*1e4
            fits.append(fit_return(z[rows-offset,j],target))
        for entry,exit,interval in product((1.25,1.5,1.75),(-.3,0.),(1,6,12)):
            policies=[{'entry':entry,'exit':exit,'decision_interval':interval,'min_edge_bps':1.,
                       'return_fit':asdict(f)} for f in fits]
            adapters=[ScoreTape(z[valid-offset,j],policies[j]) for j in range(len(pairs))]
            result=replay(adapters,np.zeros(len(valid)),t[valid],data['close'][valid],data['spread'][valid],
                          funding[valid],pairs,trade_allowed=allowed[valid],
                          expected_financing_bps=expected_funding[valid],commission_bps=commission,slippage_pips=slippage)
            # Require improvement to survive both halves of chronological validation.
            halves=[]
            for ids in np.array_split(np.arange(len(valid)),2):
                halves.append(summarize_equity(t[valid][ids],result['equity'][ids],
                    result['turnover_events'][ids].sum(),result['entry_events'][ids].sum()))
            m=result['metrics']
            dsr=deflated_sharpe_ratio(m['sharpe_daily'],planned_trials,m['n_business_days'],
                                     m['skewness'],m['kurtosis'],periods_per_year=252) if m['sharpe_daily'] is not None else 0.
            row={'variant':label,'entry':entry,'exit':exit,'interval':interval,'metrics':m,'halves':halves,
                 'deflated_sharpe_probability':dsr}
            report['trials'].append(row)
            eligible=m['positive_edge_evidence'] and dsr>=.95 and m['max_drawdown']<=.12 and all(h['net_return']>0 for h in halves)
            reference_sr=report['reference']['validation']['sharpe_daily']
            eligible=eligible and (reference_sr is None or m['sharpe_daily']>reference_sr)
            score=m['ci95_block10d'][0] if eligible else -np.inf
            if best is None or score>best[0]:
                best=(score,label,weights,z,policies,fits)
            (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    if best is None or not np.isfinite(best[0]):
        report['selection']='No policy passes validation edge, stability, and drawdown checks. Sizing comparison skipped.'
    else:
        _,label,weights,z,policies,fits=best
        report['selection']=label
        report['test_comparisons']={}
        for mode,mult in [('equal',1.),('equal',1.5),('inverse_vol',1.)]:
            result=replay([ScoreTape(z[test-meta['warmup_rows'],j],policies[j]) for j in range(len(pairs))],
                          np.zeros(len(test)),t[test],data['close'][test],data['spread'][test],funding[test],pairs,
                          trade_allowed=allowed[test],mode=mode,verified_edge=True,cost_mult=mult,
                          expected_financing_bps=expected_funding[test],commission_bps=commission,slippage_pips=slippage)
            report['test_comparisons'][f'{mode}_cost_{mult}']=result['metrics']
        for j,pair in enumerate(pairs):
            if weights[j].sum()==0:
                report.setdefault('inactive_pairs',[]).append(pair)
                continue
            save_policy(out/f'{pair.lower()}_candidate_policy.json',pair=pair,scales=scales[j],return_fit=fits[j],
                        fit_start_ns=int(t[rows[0]]),fit_end_ns=int(t[rows[-1]+30]),validate_start_ns=ns(test_start),
                        model_trained_through_ns=latest,artifact_hashes=meta['artifact_hashes'][pair],
                        entry=policies[j]['entry'],exit=policies[j]['exit'],decision_interval=policies[j]['decision_interval'],
                        weights=weights[j].tolist(),min_edge_bps=1.)
    report['trial_count']=len(report['trials'])
    report['component_rule']='Component inclusion is selected on validation only; test results do not change weights.'
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='mode',required=True)
    e=sub.add_parser('extract'); e.add_argument('--cache',type=Path,required=True)
    e.add_argument('--out',type=Path,required=True); e.add_argument('--pairs',nargs='+',choices=['EURUSD','USDJPY','GBPUSD','USDCAD'],default=['EURUSD','USDJPY','USDCAD'])
    e.add_argument('--max-bars',type=int,default=0); e.add_argument('--plumbing-only',action='store_true')
    e.add_argument('--lineage',type=Path)
    v=sub.add_parser('tune'); v.add_argument('--signals',type=Path,required=True)
    v.add_argument('--market-costs',type=Path,required=True); v.add_argument('--out',type=Path,required=True)
    v.add_argument('--fit-start',required=True); v.add_argument('--validation-start',required=True)
    v.add_argument('--test-start',required=True); v.add_argument('--test-end',required=True)
    asemble=sub.add_parser('assemble')
    asemble.add_argument('--eurusd',type=Path,required=True)
    asemble.add_argument('--usdjpy',type=Path,required=True)
    asemble.add_argument('--usdcad',type=Path,required=True)
    asemble.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='extract':
        extract(a.cache,a.out,a.pairs,a.max_bars,a.plumbing_only,a.lineage)
    elif a.mode=='tune':
        tune(a.signals,a.market_costs,a.out,a.fit_start,a.validation_start,a.test_start,a.test_end)
    else:
        assemble([a.eurusd,a.usdjpy,a.usdcad],a.out)
