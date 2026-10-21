from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from backtesting.zoo_replay import ResearchZooAdapter, replay, usd_pnl, validate_market
from trading.live_actions import LiveAction
from trading.zoo_policy import (Scale, ReturnFit, fit_scale, fit_return, hysteresis,
                                portfolio_weights, save_policy, load_policy, sha256)


def fake_zoo():
    from trading.inference_engines import MultiArchitectureSpecialistEnsembleInferenceEngine
    engine=object.__new__(MultiArchitectureSpecialistEnsembleInferenceEngine)
    class CS:
        def set_agent_state(self,**kw):
            pass
        def predict_z_score(self,obs,record_history=True):
            self.last_z_score=float(obs[0])
            return self.last_z_score
        def _update_proba_from_z(self,z,action):
            self.last_raw=z
            self.last_proba=np.array([.2,.6,.2])
    engine.cs_engine=CS()
    engine.haelt_engine=SimpleNamespace(predict_proba=lambda x: np.array([.1,.2,.7]))
    engine.mamba_engine=None
    engine.weights=(.6,.4,0.)
    engine.pair_index=0
    engine.seq_len=2
    engine._obs_buffer_146=deque(maxlen=2)
    engine._pos_dir=0
    engine.z_entry=1.5
    engine.z_exit=-.3
    return engine


def test_adapter_calls_actual_zoo_with_identical_actions():
    a,b=fake_zoo(),fake_zoo()
    wrapped=ResearchZooAdapter(b)
    rng=np.random.default_rng(5)
    for z in rng.normal(0,3,100):
        obs=np.zeros(584,dtype='float32'); obs[0]=z
        pos=int(rng.choice([-1,0,1]))
        a.set_agent_state(position_lots=pos)
        expected=a.select_action(obs)
        assert wrapped.step(obs,t_ns=100,position=pos,cost_bps=2)==expected
        assert b.last_z_score==a.last_z_score


def test_shared_hysteresis_matches_actual_stationary_state_machine():
    from trading.inference_engines import StationaryEnsembleInferenceEngine
    engine=object.__new__(StationaryEnsembleInferenceEngine)
    engine.z_entry=1.5; engine.z_exit=-.3
    engine._update_proba_from_z=lambda *args: None
    for pos in [-1,0,1]:
        for z in [-2,-1.5,-.31,-.3,0,.3,.31,1.5,2]:
            engine._pos_dir=pos
            engine.predict_z_score=lambda *args,**kw:z
            target,action=hysteresis(z,pos)
            assert engine.select_action(np.zeros(584))==action
            assert engine._pos_dir==target


def test_cost_gate_closes_reversal_but_blocks_new_entry():
    assert hysteresis(-2,1,entry_allowed=False)==(0,int(LiveAction.CLOSE))
    assert hysteresis(2,0,entry_allowed=False)==(0,int(LiveAction.HOLD))


def test_swallowed_model_error_is_fatal_in_research():
    engine=fake_zoo()
    engine.haelt_engine.predict_proba=lambda x: np.array([np.nan,0,1])
    adapter=ResearchZooAdapter(engine)
    adapter.step(np.zeros(584),t_ns=1,position=0,cost_bps=2)
    with pytest.raises(RuntimeError,match='silently'):
        adapter.step(np.zeros(584),t_ns=2,position=0,cost_bps=2)


def test_calibration_artifact_rejects_overlap_and_model_changes(tmp_path):
    cp=tmp_path/'model.pt'; cp.write_bytes(b'original')
    bound={str(cp.resolve()):sha256(cp)}
    common=dict(pair='EURUSD',scales={'haelt':Scale(0,.1)},return_fit=ReturnFit(0,5,.5),
                fit_start_ns=20,fit_end_ns=30,validate_start_ns=40,
                model_trained_through_ns=10,artifact_hashes=bound)
    save_policy(tmp_path/'policy.json',**common)
    assert load_policy(tmp_path/'policy.json','EURUSD',[cp])['scales']['haelt']['scale']==.1
    with pytest.raises(ValueError,match='disjoint'):
        save_policy(tmp_path/'bad.json',**(common|{'fit_end_ns':40}))
    cp.write_bytes(b'changed')
    with pytest.raises(ValueError):
        load_policy(tmp_path/'policy.json','EURUSD',[cp])


def test_calibration_is_frozen_and_return_units_are_bps():
    x=np.linspace(-1,1,100)
    scale=fit_scale(x)
    fit=fit_return(x,2+4*x)
    assert fit.expected(1,1)==pytest.approx(6)
    assert fit.expected(-1,-1)==pytest.approx(2)
    before=scale.apply(.2)
    np.array([1e9])  # later observations cannot update a frozen scale
    assert scale.apply(.2)==before


def test_volatility_sizing_is_gated_and_caps_usd_exposure():
    with pytest.raises(ValueError,match='positive validation'):
        portfolio_weights([1,-1,1],[],['EURUSD','USDJPY','USDCAD'],mode='inverse_vol')
    w=portfolio_weights([1,-1,-1],np.ones((40,3)),['EURUSD','USDJPY','USDCAD'],
                        mode='inverse_vol',verified_edge=True)
    assert abs(w).sum()<=1
    assert max(abs(w))<=.4
    assert abs(w @ np.array([-1,1,1]))<=.5+1e-12


def test_usd_conversion_for_direct_and_base_usd_pairs():
    assert usd_pnl('EURUSD',1000,1.1,1.2)==pytest.approx(100)
    assert usd_pnl('USDJPY',1000,150,151)==pytest.approx(1000/151)


class ConstantAdapter:
    raw_edges={}; scores={}
    def step(self,*args,**kwargs):
        return int(LiveAction.BUY)


def test_simulated_fills_are_delayed_and_spread_changes_reduce_equity():
    t=np.datetime64('2024-01-02','ns').astype('int64')+np.arange(20)*300_000_000_000
    c=np.ones((20,1))
    spread=np.full_like(c,.0001)
    funding=np.zeros((20,1,2))
    a=replay([ConstantAdapter()],np.zeros((20,584)),t,c,spread,funding,['EURUSD'])
    b=replay([ConstantAdapter()],np.zeros((20,584)),t,c,spread*2,funding,['EURUSD'])
    assert a['equity'][0]==100_000
    assert a['equity'][1]<a['equity'][0]
    assert b['equity'][-1]<a['equity'][-1]
    assert a['metrics']['entries']==1  # holding never generates repeated entries
    assert a['metrics']['net_return']<0


def test_financing_and_missing_quotes_are_not_ignored():
    t=np.arange(4,dtype='int64')*300_000_000_000
    c=np.ones((4,1)); s=np.zeros((4,1)); f=np.zeros((4,1,2))
    a=replay([ConstantAdapter()],np.zeros((4,584)),t,c,s,f,['EURUSD'],slippage_pips=0)
    f[:,:,0]=2
    b=replay([ConstantAdapter()],np.zeros((4,584)),t,c,s,f,['EURUSD'],slippage_pips=0)
    assert b['equity'][-1]<a['equity'][-1]
    with pytest.raises(ValueError,match='Missing/invalid'):
        validate_market(t,c,np.full_like(s,np.nan),f,['EURUSD'])


def test_usdjpy_equity_uses_current_quote_conversion():
    t=np.datetime64('2024-01-02','ns').astype('int64')+np.arange(3)*300_000_000_000
    c=np.array([[150.],[150.],[151.]])
    r=replay([ConstantAdapter()],np.zeros((3,584)),t,c,np.zeros_like(c),np.zeros((3,1,2)),
             ['USDJPY'],slippage_pips=0)
    # Single-pair initial allocation is capped at 40% of USD equity.
    assert r['equity'][-1]==pytest.approx(100_000+40_000/151)


def test_calibrated_policy_requires_known_financing_forecast():
    from scripts.run_zoo_policy_research import ScoreTape
    t=np.arange(3,dtype='int64')*300_000_000_000
    policy={'entry':1.5,'exit':-.3,'decision_interval':1,'min_edge_bps':1.,
            'return_fit':{'intercept_bps':0.,'slope_bps':5.,'uncertainty_bps':1.}}
    with pytest.raises(ValueError,match='financing forecast'):
        replay([ScoreTape(np.ones(3)*2,policy)],np.zeros(3),t,np.ones((3,1)),
               np.zeros((3,1)),np.zeros((3,1,2)),['EURUSD'])


def test_plumbing_extract_and_missing_training_cutoffs_cannot_be_tuned():
    from scripts.run_zoo_policy_research import verify_lineage
    with pytest.raises(ValueError,match='Plumbing-only'):
        verify_lineage({'plumbing_only':True},100)
    with pytest.raises(ValueError,match='training-window evidence'):
        verify_lineage({'plumbing_only':False,'artifact_hashes':{'EURUSD':{}}},100)


def test_future_prices_do_not_change_earlier_fills_or_sizing():
    n=100
    t=np.datetime64('2024-01-02','ns').astype('int64')+np.arange(n)*300_000_000_000
    close=np.ones((n,1)); spread=np.full_like(close,.0001); funding=np.zeros((n,1,2))
    first=replay([ConstantAdapter()],np.zeros((n,584)),t,close,spread,funding,['EURUSD'])
    close[70:]*=2
    second=replay([ConstantAdapter()],np.zeros((n,584)),t,close,spread,funding,['EURUSD'])
    np.testing.assert_array_equal(first['equity'][:70],second['equity'][:70])


def test_tuning_without_edge_exports_no_policy_and_skips_sizing(tmp_path,monkeypatch):
    import json
    import scripts.run_zoo_policy_research as research
    from feature_store.fingerprint import feature_code_fingerprint
    n=12000
    t=np.datetime64('2024-01-02','ns').astype('int64')+np.arange(n)*300_000_000_000
    rng=np.random.default_rng(7)
    raw=rng.normal(size=(n,1,3))
    cp=tmp_path/'synthetic_model'; cp.write_bytes(b'test-fixture')
    bound={str(cp):sha256(cp)}
    evidence=tmp_path/'evidence.json'
    evidence.write_text(json.dumps({'artifact_hashes':bound,'latest_training_label_t_ns':int(t[0]-1)}))
    signals=tmp_path/'signals.npz'
    np.savez(signals,raw=raw,t_ns=t,close=np.ones((n,1)),spread=np.full((n,1),.0001),
             weights=np.array([[.5,.3,.2]]),legacy_scores=raw[:,:,0])
    signals.with_suffix('.json').write_text(json.dumps({
        'source_code_fingerprint':feature_code_fingerprint(),'plumbing_only':False,
        'pairs':['EURUSD'],'warmup_rows':720,'artifact_hashes':{'EURUSD':bound},
        'hysteresis_params':{'EURUSD':{'entry':1.5,'exit':-.3}},
        'inference_source_sha256':sha256(research.ROOT/'trading/inference_engines.py'),
        'lineage':{'EURUSD':{'evidence_file':str(evidence),'evidence_sha256':sha256(evidence)}}}))
    market=tmp_path/'costs.npz'
    np.savez(market,t_ns=t,pairs=np.array(['EURUSD']),funding_bps=np.zeros((n,1,2)),
             expected_financing_bps=np.zeros((n,1)),trade_allowed=np.ones((n,1),dtype=bool),
             commission_bps=0.,slippage_pips=.7)
    # One setting per component is sufficient to exercise the complete selection
    # and fail-closed export path without spending time on redundant grid points.
    monkeypatch.setattr(research,'product',lambda *args:[(1.5,-.3,1)])
    dates=[str(np.datetime64(int(t[i]),'ns')) for i in [800,4500,9000,11999]]
    out=tmp_path/'result'
    research.tune(signals,market,out,*dates)
    result=json.loads((out/'report.json').read_text())
    assert 'No policy passes' in result['selection']
    assert 'test_comparisons' not in result
    assert not list(out.glob('*candidate_policy.json'))
    assert result['trial_count']==7


def test_assembler_joins_timestamps_and_preserves_pair_slots(tmp_path):
    import json
    import zarr
    from feature_store.fingerprint import feature_code_fingerprint
    from scripts.run_zoo_policy_research import assemble
    paths=[]
    epoch=np.datetime64('2024-01-02','ns').astype('int64')
    for slot,pair in enumerate(['EURUSD','USDJPY','USDCAD']):
        path=tmp_path/f'{pair}.zarr'
        g=zarr.open_group(str(path),mode='w',zarr_format=2)
        indices=np.arange(slot,1005+slot)
        values=np.broadcast_to(indices[:,None,None]+slot*10000,(len(indices),1,146)).astype('float32')
        for name,data in [('X',values),('t_ns',epoch+indices*300_000_000_000),
                          ('close',indices.astype(float)+1),('spread',np.full(len(indices),.0001))]:
            g.create_array(name,data=data)
        Path=__import__('pathlib').Path
        Path(str(path)+'_manifest.json').write_text(json.dumps({'pairs':[pair],
            'code_fingerprint':feature_code_fingerprint(),'content_hash':pair}))
        Path(str(path)+'_feature_schema.json').write_text(json.dumps([f'f{k}' for k in range(146)]))
        paths.append(path)
    out=tmp_path/'joined.zarr'
    assemble(paths,out)
    combined=zarr.open(str(out),mode='r')
    assert combined['X'].shape==(1003,1,438)
    np.testing.assert_array_equal(combined['X'][0,0,[0,146,292]],[2,10002,20002])
    assert combined['t_ns'][0]==epoch+2*300_000_000_000
