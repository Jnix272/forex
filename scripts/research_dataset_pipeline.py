"""Sequential research rebuild, simple baselines, and isolated HAELT retraining.

All outputs stay under an explicit research directory. No broker or deployment
entry points are invoked. State is written after every completed stage.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import hashlib
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def build_worker(config, result):
    from training.cli import parse_args
    from training.dataset_builder import build_dataset_chunked
    sys.argv = [sys.argv[0], '--config', str(config), '--build-only', '--sentiment-mode', 'off']
    args = parse_args()
    cache, rows, features, _ = build_dataset_chunked(args)
    import zarr
    import numpy as np
    z = zarr.open(cache, mode='r')
    if 't_ns' not in z:
        raise RuntimeError('Rebuilt cache has no timestamps')
    times = np.asarray(z['t_ns'][:], dtype=np.int64)
    if len(times) != rows or np.any(np.diff(times) <= 0):
        raise RuntimeError('Rebuilt timestamps are not strictly increasing and aligned')
    for key in ('X','y','y_cls','close','spread','atr'):
        if z[key].shape[0] != rows:
            raise RuntimeError(f'Misaligned cache array: {key}')
    write_json(result, {'cache':str(Path(cache).resolve()),'rows':rows,'features':features,
                        'start_t_ns':int(times[0]),'end_t_ns':int(times[-1]),
                        'sentiment_mode':'off', 'purpose':'isolated corrected-data experiment'})


def single_pair_training_config(cfg, resume=False):
    """A single-pair cache has scalar targets, not multi-pair sidecars."""
    if len(cfg['data']['pairs']) != 1:
        raise ValueError('This experiment requires exactly one pair per dataset')
    cfg['training']['per_pair_heads'] = False
    cfg['training']['resume'] = bool(resume)
    return cfg


def load_resume_state(out, fp, pairs):
    state = json.loads((out/'status.json').read_text())
    if state['status'] == 'running':
        raise RuntimeError('Experiment is marked running; inspect its process before resuming')
    if state['source_fingerprint'] != fp:
        raise RuntimeError('Cannot resume after feature/label/builder source changes')
    for pair in pairs:
        for name, expected in state['reference_artifacts'][pair].items():
            path = out/'reference'/pair/name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Reference snapshot changed: {path}')
        if f'build_{pair.lower()}' in state['completed']:
            result = json.loads((out/f'{pair.lower()}_dataset.json').read_text())
            if not Path(result['cache']).is_dir():
                raise RuntimeError(f'Completed cache missing for {pair}')
        if f'baseline_{pair.lower()}' in state['completed']:
            json.loads((out/'baseline'/pair/'results.json').read_text())
    shutil.copy2(out/'status.json', out/f"status_before_resume_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.json")
    state.update(status='running', pid=os.getpid(), resumed_utc=datetime.now(timezone.utc).isoformat())
    state.pop('error', None)
    return state


def run_job(out, pairs, deep, resume=False):
    import yaml
    from feature_store.fingerprint import feature_code_fingerprint
    fp = feature_code_fingerprint()
    state = {'status':'running', 'pid':os.getpid(), 'source_fingerprint':fp,
             'started_utc':datetime.now(timezone.utc).isoformat(), 'completed':[],
             'protocol':{'validation_years':[2023,2024], 'diagnostic_year':2025,
                         'history_years':['expanding',2], 'no_deployment':True,
                         'sentiment_mode':'off', 'reason':'calendar-only source cache; explicit reproducible setting'}}
    out.mkdir(parents=True,exist_ok=True)
    if resume:
        state = load_resume_state(out, fp, pairs)
    elif (out/'status.json').exists():
        raise RuntimeError('Experiment already exists; use --resume or a new directory')
    else:
        state['reference_artifacts']={}
    for pair in ([] if resume else pairs):
        src=ROOT/f'checkpoints/haelt_{pair.lower()}_v1/haelt'
        dst=out/'reference'/pair
        dst.mkdir(parents=True,exist_ok=True)
        files=[src/'haelt_best.pt',src/'haelt_best_scaler.npz',src/'haelt_best_features.json']
        files.extend(src.glob('*.json'))
        hashes={}
        for path in dict.fromkeys(files):
            if path.exists():
                target=dst/path.name
                if target.exists():
                    raise RuntimeError(f'Reference snapshot already exists; choose a new experiment directory: {target}')
                shutil.copy2(path,target)
                hashes[path.name]=hashlib.sha256(target.read_bytes()).hexdigest()
        if 'haelt_best.pt' not in hashes:
            raise RuntimeError(f'No reference checkpoint for {pair}')
        state['reference_artifacts'][pair]=hashes
    def stage(tag, command):
        if feature_code_fingerprint() != fp:
            raise RuntimeError('Feature/label/builder sources changed during the experiment')
        if tag in state['completed']:
            return
        state['stage']=tag
        state['command']=command
        write_json(out/'status.json',state)
        log_path = out/f'{tag}.log'
        if log_path.exists():
            log_path.rename(out/f"{tag}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.log")
        with log_path.open('w',encoding='utf-8') as log:
            env=os.environ.copy()
            env.update({'PYTHONPATH':str(ROOT),'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'2',
                        'PYTHONIOENCODING':'utf-8',
                        'MKL_NUM_THREADS':'2','FOREX_PURE_BCE':'1','FOREX_TRADE_FRACTION':'0.05'})
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        state['completed'].append(tag)
        write_json(out/'status.json',state)
    try:
        for pair in pairs:
            config=out/f'{pair.lower()}.yaml'
            result=out/f'{pair.lower()}_dataset.json'
            stage(f'build_{pair.lower()}',[sys.executable,'-u',str(Path(__file__).resolve()),
                  '--build-worker',str(config),'--result',str(result)])
            cache=Path(json.loads(result.read_text())['cache'])
            cmd=[sys.executable,'-u',str(ROOT/'scripts/research_sharpe_baseline.py'),
                 '--cache',str(cache),'--pair',pair,'--out',str(out/'baseline'/pair),
                 '--legacy-checkpoint',str(out/'reference'/pair/'haelt_best.pt')]
            stage(f'baseline_{pair.lower()}',cmd)
            if deep:
                import zarr
                import numpy as np
                times=np.asarray(zarr.open(str(cache),mode='r')['t_ns'][:],dtype=np.int64)
                max_samples=int(np.searchsorted(times,np.datetime64('2025-01-01','ns').astype(np.int64))) - 151
                cfg=single_pair_training_config(yaml.safe_load(config.read_text()), resume=resume)
                tr=cfg['training']
                tr.update({'enable_ewc':False,'enable_si':False,'swa_enabled':False,
                           'training_memory':False,'use_mixup':False,'sharpe_weight':0.,
                           'online_hard_mining':False,
                           'patience':3,'epochs':12,'lr':1.5e-5})
                cfg['curriculum']['manager']={'enabled':False}
                for key in ('self_paced','miner_feedback','loss_weighting'):
                    cfg['curriculum'][key]={'enabled':False}
                cfg.setdefault('online_miner',{})['enabled']=False
                cfg.setdefault('pretrain',{})['enabled']=False
                cfg['walk_forward']['enabled']=True
                cfg['walk_forward']['folds']=3
                cfg['validation']['n_splits']=3
                cfg['tracking']['run_name']=f'corrected_simple_haelt_{pair.lower()}'
                train_config=out/f'{pair.lower()}_simple_train.yaml'
                train_config.write_text(yaml.safe_dump(cfg,sort_keys=False))
                stage(f'train_{pair.lower()}',[sys.executable,'-u',str(ROOT/'training/train_gpu.py'),
                      '--config',str(train_config),'--model','haelt','--no-model-profile',
                      '--no-promotion-gate','--no-online-hard-mining','--no-pretrain',
                      '--no-training-memory','--no-per-pair-heads',
                      '--resume' if resume else '--no-resume',
                      '--max-samples',str(max_samples),'--sentiment-mode','off'])
                checkpoint=out/'checkpoints'/pair/'haelt'/'haelt_best.pt'
                if not checkpoint.exists():
                    raise RuntimeError(f'Fresh checkpoint missing: {checkpoint}')
                stage(f'compare_{pair.lower()}',cmd+['--candidate-checkpoint',str(checkpoint)])
        state['status']='complete'
        state['finished_utc']=datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        state['status']='failed'
        state['error']=str(exc)
        write_json(out/'status.json',state)
        raise
    write_json(out/'status.json',state)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'research/sharpe_20261019')
    p.add_argument('--pairs',nargs='+',choices=['EURUSD','USDCAD','USDJPY'],default=['EURUSD','USDCAD','USDJPY'])
    p.add_argument('--deep',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--build-worker',type=Path)
    p.add_argument('--result',type=Path)
    a=p.parse_args()
    if a.build_worker:
        if not a.result:
            p.error('--result is required for a build worker')
        build_worker(a.build_worker,a.result)
    else:
        run_job(a.out.resolve(),a.pairs,a.deep,a.resume)
