import numpy as np
import pandas as pd
import pytest

from scripts.research_sharpe_baseline import score_policy, time_split
from training.single_pair_time import timestamp_rows


def test_timestamp_rows_converts_units_and_drops_warmup_overlap():
    t = pd.date_range('2024-01-01', periods=5, freq='5min', tz='UTC').as_unit('us')
    ns = t.as_unit('ns').asi8
    actual, keep = timestamp_rows(t, 5, int(ns[1]))
    np.testing.assert_array_equal(actual, ns[2:])
    np.testing.assert_array_equal(np.arange(5)[keep], [2, 3, 4])


@pytest.mark.parametrize('times,n', [(['2024-01-01', '2024-01-01'],2),
                                    (['2024-01-02', '2024-01-01'],2),
                                    (['2024-01-01',None],2),
                                    (['2024-01-01'],2)])
def test_timestamp_rows_rejects_bad_alignment(times,n):
    with pytest.raises(ValueError):
        timestamp_rows(times,n)


def test_recent_and_expanding_use_identical_future_and_no_overlap():
    t = pd.date_range('2018-01-01','2025-02-01',freq='1h').asi8
    tr,te = time_split(t,'2024-01-01','2025-01-01')
    recent,te2 = time_split(t,'2024-01-01','2025-01-01',2)
    np.testing.assert_array_equal(te,te2)
    assert len(recent) < len(tr)
    assert tr[-1] + 30 < te[0] - 120
    assert t[recent[-1]] - t[recent[0]] <= 2 * 365.25 * 86400 * 1e9


def test_scoring_nonoverlap_costs_and_gap_rejection():
    t = pd.date_range('2024-01-02',periods=100,freq='5min').asi8
    close = np.linspace(1.,1.01,100)
    spread = np.full(100,.0001)
    idx = np.arange(60)
    d = np.ones(60)
    a = score_policy(d,idx,close,spread,t,.0001)
    b = score_policy(d,idx,close,spread,t,.0001,cost_mult=1.5)
    assert a['n_trades'] == 2
    assert b['net_return_equal_notional'] < a['net_return_equal_notional']
    t[20:] += 3600 * 1_000_000_000
    c = score_policy(d,idx,close,spread,t,.0001)
    assert c['skipped_gap_entries'] > 0


def test_invalid_spread_fails_instead_of_becoming_free_execution():
    t = pd.date_range('2024-01-02',periods=40,freq='5min').asi8
    with pytest.raises(ValueError,match='spread'):
        score_policy([1],[0],np.ones(40),np.full(40,np.nan),t,.0001)


def test_clean_training_cli_explicitly_disables_mining(monkeypatch):
    import sys
    from training.cli import parse_args
    monkeypatch.setattr(sys,'argv',['train','--config','config/run_eurusd.yaml',
                        '--no-online-hard-mining','--no-training-memory',
                        '--no-pretrain','--no-promotion-gate'])
    args=parse_args()
    assert args.online_hard_mining is False
    assert args.training_memory is False
    assert args.pretrain is False
    assert args.promotion_gate is False


def test_baseline_end_to_end_writes_all_history_comparisons(tmp_path):
    import json
    import zarr
    from feature_store.fingerprint import feature_code_fingerprint
    from scripts.research_sharpe_baseline import run
    times=pd.date_range('2018-01-01','2025-12-30',freq='1D').asi8
    n=len(times)
    rng=np.random.default_rng(41)
    cache=tmp_path/'tiny.zarr'
    root=zarr.open_group(str(cache),mode='w',zarr_format=2)
    arrays={'X':rng.normal(size=(n,2,3)).astype('float32'),
            't_ns':times,'close':np.full(n,1.2),'spread':np.full(n,.0001),
            'y_cls':rng.choice([-1.,0.,1.],size=n)}
    for name,values in arrays.items():
        root.create_array(name,data=values)
    (tmp_path/'tiny.zarr_manifest.json').write_text(json.dumps({
        'code_fingerprint':feature_code_fingerprint(),'content_hash':'test-only'}))
    out=tmp_path/'output'
    run(cache,'EURUSD',out)
    result=json.loads((out/'results.json').read_text())
    assert len(result['results']) == 12
    assert len(list(out.glob('*.joblib'))) == 12
    assert {r['history_years'] for r in result['results']} == {None,2}
    # Daily-spaced synthetic prices are intentionally not valid five-minute
    # trades; the experiment must report zero trades rather than false profit.
    assert all(r['metrics']['n_trades'] == 0 for r in result['results'])
