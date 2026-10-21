"""Isolated, chronological dataset/training experiment. Never deploys models.

Explicit cache paths only. All fitting and confidence thresholds use training
rows; the previously inspected 2025 tail is labelled diagnostic, not a lockbox.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler


def time_split(t_ns, start, end, history_years=None, seq_len=120, horizon=30):
    """Conservatively separate complete input/label windows by clock time.

    Validation starts with a fresh input window: no input or label bar from
    training can overlap it. A row gap is also required for irregular caches.
    """
    t = np.asarray(t_ns, dtype=np.int64)
    if np.any(np.diff(t) <= 0):
        raise ValueError("Timestamps must be strictly increasing")
    a = int(np.searchsorted(t, np.datetime64(start, 'ns').astype(np.int64)))
    b = int(np.searchsorted(t, np.datetime64(end, 'ns').astype(np.int64)))
    gap = seq_len + horizon + 1
    train_end = max(0, a - gap)
    tr = np.arange(train_end, dtype=np.int64)
    if history_years is not None and train_end:
        cutoff = t[train_end - 1] - int(history_years * 365.25 * 86400 * 1e9)
        tr = tr[t[tr] >= cutoff]
    te = np.arange(a, max(a, b - horizon - 1), dtype=np.int64)
    if len(tr) < 500 or len(te) < horizon * 2:
        raise ValueError("Insufficient data for the declared split")
    assert tr[-1] + horizon < te[0] - seq_len
    return tr, te


def score_policy(directions, indices, close, spread, t_ns, pip, horizon=30, cost_mult=1.0):
    """Nonoverlapping H-bar positions, daily mark-to-market equal-notional returns.

    Reject missing prices/spreads and trades spanning missing five-minute bars.
    Costs are charged at entry; gains accrue each bar, not only at exit.
    """
    idx = np.asarray(indices, dtype=np.int64)
    d = np.asarray(directions, dtype=np.float64)
    if len(idx) != len(d) or not len(idx):
        raise ValueError("Directions must align with explicit nonempty indices")
    days = np.asarray(t_ns, dtype='datetime64[ns]').astype('datetime64[D]')
    calendar = np.arange(days[idx[0]], days[idx[-1]] + np.timedelta64(1, 'D'))
    calendar = calendar[np.is_busday(calendar)]
    pnl = np.zeros(len(calendar))
    gross_total = cost_total = 0.0
    trades = gaps = 0
    next_free = -1
    for row, direction in zip(idx, d):
        if row < next_free or direction == 0 or row + horizon >= len(close):
            continue
        stop = row + horizon
        delta = np.diff(t_ns[row:stop + 1])
        if np.any(delta != 300_000_000_000):
            gaps += 1
            continue
        prices = close[row:stop + 1]
        if not np.isfinite(prices).all() or np.any(prices <= 0):
            raise ValueError("Invalid executed price")
        if not np.isfinite(spread[row]) or spread[row] < 0:
            raise ValueError("Invalid executed spread")
        costs = cost_mult * (spread[row] + 0.7 * pip) / prices[0]
        returns = direction * np.diff(prices) / prices[0]
        entry_day = np.searchsorted(calendar, days[row])
        if entry_day >= len(calendar) or calendar[entry_day] != days[row]:
            continue
        pnl[entry_day] -= costs
        for day, ret in zip(days[row + 1:stop + 1], returns):
            slot = np.searchsorted(calendar, day)
            if slot >= len(calendar) or calendar[slot] != day:
                raise ValueError("Trade crossed a non-business day")
            pnl[slot] += ret
        trades += 1
        gross_total += float(returns.sum())
        cost_total += float(costs)
        next_free = stop
    sd = pnl.std(ddof=1) if len(pnl) > 1 else 0.0
    sr = float(pnl.mean() / sd * np.sqrt(252)) if sd > 0 else None
    ci = [None, None]
    if len(pnl) >= 20 and sd > 0:
        rng = np.random.default_rng(1337)
        block = min(10, len(pnl))
        samples = []
        for _ in range(1000):
            starts = rng.integers(0, len(pnl) - block + 1, int(np.ceil(len(pnl) / block)))
            draw = np.concatenate([pnl[s:s + block] for s in starts])[:len(pnl)]
            if draw.std(ddof=1) > 0:
                samples.append(draw.mean() / draw.std(ddof=1) * np.sqrt(252))
        if samples:
            ci = np.quantile(samples, [.025, .975]).tolist()
    equity = 1 + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[1., equity])[1:]
    return {'sharpe_daily': sr, 'ci95_block10d': ci, 'n_trades': trades,
            'skipped_gap_entries': gaps, 'net_return_equal_notional': float(pnl.sum()),
            'gross_return_equal_notional': gross_total, 'cost_return': cost_total,
            'max_drawdown': float(np.max((peak - equity) / peak)),
            'n_business_days': len(pnl)}


def read_last_features(root, rows, batch=512):
    arr = root['X']
    out = np.empty((len(rows), arr.shape[-1]), dtype=np.float32)
    for s in range(0, len(rows), batch):
        ids = rows[s:s + batch]
        # Contiguous blocks avoid expensive per-row Zarr chunk decompression.
        block = np.asarray(arr[int(ids[0]):int(ids[-1]) + 1])
        out[s:s + len(ids)] = block[ids - ids[0], -1, :]
    return out


def run(cache, pair, out, legacy_checkpoint=None, candidate_checkpoint=None):
    root = zarr.open(str(cache), mode='r')
    manifest = json.loads(Path(str(cache) + '_manifest.json').read_text())
    from feature_store.fingerprint import feature_code_fingerprint
    if manifest.get('code_fingerprint') != feature_code_fingerprint():
        raise ValueError('Refusing a stale or unversioned feature/label cache')
    t = np.asarray(root['t_ns'][:], dtype=np.int64)
    close = np.asarray(root['close'][:], dtype=np.float64).reshape(-1)
    spread = np.asarray(root['spread'][:], dtype=np.float64).reshape(-1)
    y = np.asarray(root['y_cls'][:]).reshape(-1)
    if not set(np.unique(y)).issubset({-1, 0, 1}):
        raise ValueError('Expected explicit signed labels {-1,0,1}')
    pip = .01 if 'JPY' in pair else .0001
    out.mkdir(parents=True, exist_ok=True)
    result = {'cache':str(cache.resolve()), 'pair':pair,
              'cache_content_hash':manifest.get('content_hash'),
              'code_fingerprint':manifest['code_fingerprint'],
              'protocol': 'Predeclared 2023/2024 validation; 2025 is previously inspected diagnostic data. No promotion.',
              'results':[]}
    # Read once; transformations are fit separately inside every fold.
    rows = np.arange(len(t))
    features = read_last_features(root, rows)
    np.save(out/'last_features.npy', features)
    periods = [('2023-01-01','2024-01-01'), ('2024-01-01','2025-01-01'), ('2025-01-01','2025-12-30')]
    for start, end in periods:
        for history in (None, 2):
            tr, te = time_split(t, start, end, history)
            # Decimate training inputs to reduce repeated, overlapping outcomes.
            tr = tr[::30]
            xtr = features[tr]
            finite = np.isfinite(xtr).mean(axis=0) >= .99
            med = np.nanmedian(np.where(np.isfinite(xtr), xtr, np.nan), axis=0)
            keep = finite & np.isfinite(med) & (np.nanstd(xtr, axis=0) > 1e-8)
            if not keep.any():
                raise ValueError('No usable training features')
            def clean(x):
                return np.where(np.isfinite(x[:,keep]),x[:,keep],med[keep])
            scaler = RobustScaler(quantile_range=(5,95)).fit(clean(xtr))
            a = np.clip(scaler.transform(clean(xtr)), -8, 8)
            b = np.clip(scaler.transform(clean(features[te])), -8, 8)
            for name, model in [('logistic',LogisticRegression(C=1.0,max_iter=500,random_state=1337)),
                                ('small_tree',HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,
                                    min_samples_leaf=50,l2_regularization=10,early_stopping=False,random_state=1337))]:
                model.fit(a,y[tr])
                classes = model.classes_
                if -1 not in classes or 1 not in classes:
                    raise ValueError('Training data lacks both trade directions')
                neg, pos = int(np.where(classes == -1)[0][0]), int(np.where(classes == 1)[0][0])
                ptrain = model.predict_proba(a)
                threshold = float(np.quantile(np.maximum(ptrain[:,neg],ptrain[:,pos]), .95))
                p = model.predict_proba(b)
                confidence = np.maximum(p[:,neg],p[:,pos])
                direction = np.where(p[:,pos] >= p[:,neg],1.,-1.)
                direction[confidence < max(.52,threshold)] = 0
                metrics = score_policy(direction,te,close,spread,t,pip)
                stressed = score_policy(direction,te,close,spread,t,pip,cost_mult=1.5)
                entry = {'model':name,'start':start,'end':end,'history_years':history,
                         'train_rows':len(tr),'test_rows':len(te),'threshold':max(.52,threshold),
                         'features':int(keep.sum()),'metrics':metrics,'cost_stress_1_5x':stressed}
                result['results'].append(entry)
                import joblib
                joblib.dump({'model':model,'scaler':scaler,'keep':keep,'median':med,'threshold':max(.52,threshold),
                             'train_last_t_ns':int(t[tr[-1]]),'cache_hash':manifest.get('content_hash')},
                            out/f'{name}_{start}_{history or "expanding"}.joblib')
                (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
                print(pair,name,start,history,metrics,flush=True)
    for tag, checkpoint in [('legacy_diagnostic',legacy_checkpoint),('candidate_diagnostic',candidate_checkpoint)]:
        if checkpoint is None:
            continue
        # Legacy folds may have trained through earlier validation periods. Only
        # compare the previously seen 2025 diagnostic, never label it untouched.
        from inference.pytorch_inference import PyTorchInferenceEngine
        eng = PyTorchInferenceEngine(checkpoint_path=str(checkpoint),model_name='haelt',seq_len=120,device='cpu')
        _,te = time_split(t,'2025-01-01','2025-12-30')
        direction = np.zeros(len(te))
        for s in range(0,len(te),512):
            ids=te[s:s+512]
            windows=np.asarray(root['X'][int(ids[0]):int(ids[-1])+1])
            for j,row in enumerate(ids):
                k=s+j
                if k % 30:
                    continue
                p = eng.predict_proba(windows[j])
                if np.max(p) >= .52:
                    direction[k] = [-1.,0.,1.][int(np.argmax(p))]
        result[tag] = {'checkpoint':str(checkpoint),
            'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'limitation':'Previously inspected 2025 data; legacy training lineage not fully reconstructable. Fixed-horizon diagnostic, not live-zoo validation.',
            'metrics':score_policy(direction,te,close,spread,t,pip)}
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    os.environ.setdefault('OMP_NUM_THREADS','2')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--pair',choices=['EURUSD','USDCAD','USDJPY'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--legacy-checkpoint',type=Path)
    p.add_argument('--candidate-checkpoint',type=Path)
    a=p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        run(a.cache,a.pair,a.out,a.legacy_checkpoint,a.candidate_checkpoint)
