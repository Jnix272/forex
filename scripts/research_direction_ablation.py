"""Isolated direction/confidence ablation and ungated checkpoint diagnostics.

No broker entry points or production checkpoint directories are used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def direction_metrics(logits, confidence, labels, threshold=.52):
    """Raw direction quality excludes HOLD labels; policy accuracy includes them."""
    import numpy as np
    logits, confidence, labels = [np.asarray(x).reshape(-1) for x in (logits, confidence, labels)]
    if not len(labels) or not (len(logits) == len(confidence) == len(labels)):
        raise ValueError('Aligned nonempty predictions and signed labels required')
    if not np.isfinite(logits).all() or not np.isfinite(confidence).all():
        raise ValueError('Nonfinite checkpoint outputs')
    if not set(np.unique(labels)).issubset({-1, 0, 1}):
        raise ValueError('Expected signed SELL/HOLD/BUY labels')
    p = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    q = 1 / (1 + np.exp(-np.clip(confidence, -80, 80)))
    raw = np.where(p >= .5, 1, -1)
    tradable = labels != 0
    recalls = [float(np.mean(raw[labels == k] == k)) if np.any(labels == k) else None for k in (-1, 1)]
    composite = np.stack([(1-p)*q, 1-q, p*q], axis=-1)
    action = composite.argmax(axis=-1)-1
    action[composite.max(axis=-1) < threshold] = 0
    confusion = [[int(np.sum((labels == t) & (action == a))) for a in (-1, 0, 1)] for t in (-1, 0, 1)]
    probability = np.clip(p[tradable], 1e-7, 1-1e-7)
    target = (labels[tradable] == 1).astype(float)
    return {'n_rows': len(labels), 'n_direction_rows': int(tradable.sum()),
            'raw_direction_accuracy': float(np.mean(raw[tradable] == labels[tradable])) if tradable.any() else None,
            'raw_direction_balanced_accuracy': float(np.mean(recalls)) if None not in recalls else None,
            'sell_recall': recalls[0], 'buy_recall': recalls[1],
            'direction_log_loss': float(-np.mean(target*np.log(probability)+(1-target)*np.log(1-probability))) if len(target) else None,
            'direction_brier': float(np.mean((probability-target)**2)) if len(target) else None,
            'p_buy_quantiles': np.quantile(p, [0, .05, .5, .95, 1]).tolist(),
            'p_trade_quantiles': np.quantile(q, [0, .05, .5, .95, 1]).tolist(),
            'policy_class_order': ['SELL', 'HOLD', 'BUY'],
            'policy_confusion_true_rows_pred_columns': confusion,
            'policy_pred_counts': [int(np.sum(action == k)) for k in (-1, 0, 1)],
            'policy_accuracy': float(np.mean(action == labels)),
            'policy_nonhold_fraction': float(np.mean(action != 0)), 'policy_threshold': threshold}


def predict(cache, checkpoint, rows, batch_size=32):
    import numpy as np
    import torch
    import zarr
    from inference.pytorch_inference import PyTorchInferenceEngine
    torch.set_num_threads(2)
    engine = PyTorchInferenceEngine(checkpoint_path=str(checkpoint), model_name='haelt', seq_len=120, device='cpu')
    core = getattr(engine.model, 'inner', engine.model)
    root = zarr.open(str(cache), mode='r')
    direction, confidence = [], []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            ids = rows[start:start+batch_size]
            windows = np.asarray(root['X'].oindex[ids, :, :])
            x = torch.as_tensor(np.stack([engine._transform_window(w) for w in windows]), dtype=torch.float32)
            output = core(x)
            if not isinstance(output, (tuple, list)) or len(output) < 3:
                raise ValueError('Expected the same single-pair multitask architecture in both arms')
            direction.extend(output[0].float().reshape(-1).tolist())
            confidence.extend(output[2].float().reshape(-1).tolist())
    return np.asarray(direction), np.asarray(confidence)


def diagnose(cache, checkpoint, out, start, end, max_rows=0):
    import numpy as np
    import zarr
    root = zarr.open(str(cache), mode='r')
    t = np.asarray(root['t_ns'][:], dtype=np.int64)
    rows = np.flatnonzero((t >= np.datetime64(start, 'ns').astype(np.int64)) &
                          (t < np.datetime64(end, 'ns').astype(np.int64)))[::30]
    if max_rows:
        rows = rows[:max_rows]
    if not len(rows):
        raise ValueError('No rows in diagnostic interval')
    before = digest(checkpoint)
    logits, confidence = predict(cache, checkpoint, rows)
    if before != digest(checkpoint):
        raise ValueError('Checkpoint changed during diagnostics; use an immutable snapshot')
    report = direction_metrics(logits, confidence, np.asarray(root['y_cls'][:])[rows])
    report.update(checkpoint=str(checkpoint), checkpoint_sha256=before,
                  cache=str(cache), first_t_ns=int(t[rows[0]]), last_t_ns=int(t[rows[-1]]),
                  stride=30, capped=bool(max_rows), scope='Checkpoint diagnostic, not an out-of-sample performance claim')
    save(out, report)
    return report


def paired_configs(base, out):
    """Only confidence-loss weight differs between paired training arms."""
    configs = {}
    for arm, weight in [('direction_confidence', .3), ('direction_only', 0.)]:
        cfg = copy.deepcopy(base)
        cfg['paths']['checkpoint_dir'] = str(out/'checkpoints'/arm)
        cfg['tracking']['run_name'] = f'paired_{arm}_eurusd'
        cfg['training'].update(per_pair_heads=False, resume=False, early_stop_metric='val_loss',
                               early_stop_min_delta=0., sharpe_weight=0.)
        cfg.setdefault('direction_training', {})['warmup_epochs'] = 0
        cfg.setdefault('multitask', {}).update(enabled=True, w_ret=0., w_quantile=0., w_conf=weight,
                                              focal_gamma=0., entropy_weight=0., class_balance_weight=0.)
        configs[arm] = cfg
    return configs


def compare(cache, checkpoints, max_samples, out):
    """Both arms use the same direction-only policy and train-fitted hurdle.

    The untrained confidence head is never used to score the direction-only
    arm's economic performance. Composite-head diagnostics are separate.
    """
    import numpy as np
    import zarr
    from scripts.research_sharpe_baseline import score_policy, time_split
    root = zarr.open(str(cache), mode='r')
    t = np.asarray(root['t_ns'][:], dtype=np.int64)
    labels = np.asarray(root['y_cls'][:])
    close, spread = [np.asarray(root[k][:]) for k in ('close', 'spread')]
    train, test = time_split(t, '2025-01-01', '2025-12-30')
    train = train[train < max_samples][::30]
    # Fixed, chronological, evenly spaced training subset; no test-fitted hurdle.
    if len(train) > 2048:
        train = train[np.linspace(0, len(train)-1, 2048).astype(int)]
    test = test[::30]
    report = {'protocol': 'Paired loss ablation; 2025 was previously inspected and remains diagnostic only.',
              'policy': 'Identical direction-only probabilities, threshold=max(0.52, training 95th percentile).',
              'limitation': 'Fixed-horizon session-filtered proxy; no financing or separate commission modeled; not live-zoo replay.',
              'arms': {}}
    for arm, checkpoint in checkpoints.items():
        binding = digest(checkpoint)
        tr_logits, _ = predict(cache, checkpoint, train)
        te_logits, te_conf = predict(cache, checkpoint, test)
        p_tr = 1/(1+np.exp(-np.clip(tr_logits, -80, 80)))
        p = 1/(1+np.exp(-np.clip(te_logits, -80, 80)))
        hurdle = max(.52, float(np.quantile(np.maximum(p_tr, 1-p_tr), .95)))
        actions = np.where(p >= .5, 1., -1.)
        actions[np.maximum(p, 1-p) < hurdle] = 0
        if binding != digest(checkpoint):
            raise ValueError('Checkpoint changed during comparison')
        report['arms'][arm] = {'checkpoint_sha256': binding, 'threshold': hurdle,
            'confidence_head_trained': arm == 'direction_confidence',
            'train_calibration_rows': len(train), 'diagnostic_rows': len(test),
            'prediction_metrics': direction_metrics(te_logits, te_conf, labels[test]),
            'net_metrics': score_policy(actions, test, close, spread, t, .0001),
            'cost_stress_1_5x': score_policy(actions, test, close, spread, t, .0001, cost_mult=1.5)}
        save(out, report)
    return report


def run(source, out, wait_pid):
    import yaml
    from feature_store.fingerprint import feature_code_fingerprint
    out.mkdir(parents=True, exist_ok=True)
    status_path = out/'status.json'
    if status_path.exists():
        raise RuntimeError('Choose a new output directory; existing experiments are immutable')
    fp = feature_code_fingerprint()
    research_sources = [Path(__file__), ROOT/'scripts/research_sharpe_baseline.py']
    research_sources += list((ROOT/'training').glob('*.py')) + list((ROOT/'models').glob('*.py'))
    training_hashes = {str(path.resolve()): digest(path) for path in research_sources}
    base_path = source/'eurusd_simple_train.yaml'
    base = yaml.safe_load(base_path.read_text())
    configs = paired_configs(base, out)
    config_hashes = {}
    for arm, cfg in configs.items():
        path = out/f'{arm}.yaml'
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        config_hashes[arm] = digest(path)
    state = {'status': 'waiting', 'pid': os.getpid(), 'wait_pid': wait_pid, 'source_fingerprint': fp,
             'completed': [], 'no_deployment': True, 'config_hashes': config_hashes,
             'base_config_sha256': digest(base_path), 'training_source_hashes': training_hashes}
    save(status_path, state)
    try:
        if wait_pid:
            # Windows process handle remains bound to the original process even
            # if its PID is reused. Wait without busy polling or a training race.
            import ctypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.OpenProcess.restype = ctypes.c_void_p
            kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            handle = kernel.OpenProcess(0x00100000, False, wait_pid)
            if handle:
                try:
                    while kernel.WaitForSingleObject(handle, 30000) == 258:
                        pass
                finally:
                    kernel.CloseHandle(handle)
        if feature_code_fingerprint() != fp:
            raise RuntimeError('Feature/label code changed while experiment was queued')
        if any(digest(path) != expected for path, expected in training_hashes.items()):
            raise RuntimeError('Training or evaluation code changed while experiment was queued')
        original = json.loads((source/'status.json').read_text())
        if original['status'] == 'running':
            raise RuntimeError('Source experiment is still marked running; refusing concurrent GPU training')
        data = json.loads((source/'eurusd_dataset.json').read_text())
        cache = Path(data['cache'])
        manifest = json.loads(Path(str(cache)+'_manifest.json').read_text())
        if manifest.get('code_fingerprint') != fp:
            raise RuntimeError('Corrected dataset fingerprint changed')
        import numpy as np
        import zarr
        times = np.asarray(zarr.open(str(cache), mode='r')['t_ns'][:], dtype=np.int64)
        max_samples = int(np.searchsorted(times, np.datetime64('2025-01-01', 'ns').astype(np.int64)))-151
        state.update(status='running', cache=str(cache), max_samples=max_samples,
                     protocol='Same cache, features, seed, splits and evaluation; confidence loss .3 versus 0; no smoothing search.')
        checkpoints = {}
        env = os.environ.copy()
        env.update(PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', FOREX_PURE_BCE='1', FOREX_TRADE_FRACTION='0.05')
        for arm, cfg in configs.items():
            config = out/f'{arm}.yaml'
            if digest(config) != config_hashes[arm]:
                raise RuntimeError('Queued experiment configuration changed')
            command = [sys.executable, '-u', str(ROOT/'training/train_gpu.py'), '--config', str(config),
                       '--model', 'haelt', '--no-model-profile', '--no-promotion-gate', '--no-online-hard-mining',
                       '--no-pretrain', '--no-training-memory', '--no-per-pair-heads', '--direction-warmup-epochs', '0',
                       '--mt-w-conf', str(cfg['multitask']['w_conf']),
                       '--max-samples', str(max_samples), '--sentiment-mode', 'off']
            state.update(stage=arm, command=command, config_sha256=digest(config))
            save(status_path, state)
            with (out/f'{arm}.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            checkpoints[arm] = Path(cfg['paths']['checkpoint_dir'])/'haelt'/'haelt_best.pt'
            if not checkpoints[arm].is_file():
                raise RuntimeError(f'Missing completed checkpoint for {arm}')
            state['completed'].append(arm)
            save(status_path, state)
        state['stage'] = 'compare'
        save(status_path, state)
        compare(cache, checkpoints, max_samples, out/'comparison.json')
        state['status'] = 'complete'
    except Exception as exc:
        state.update(status='failed', error=str(exc))
        raise
    finally:
        save(status_path, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'research/sharpe_20261019')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wait-pid', type=int, default=0)
    parser.add_argument('--diagnose-checkpoint', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--start', default='2025-01-01')
    parser.add_argument('--end', default='2025-12-30')
    parser.add_argument('--max-rows', type=int, default=0)
    args = parser.parse_args()
    if args.diagnose_checkpoint:
        if not args.cache:
            parser.error('--cache is required for diagnostics')
        diagnose(args.cache, args.diagnose_checkpoint, args.out, args.start, args.end, args.max_rows)
    else:
        run(args.source.resolve(), args.out.resolve(), args.wait_pid)
