"""Tests for remaining training-audit remediations."""

from __future__ import annotations

import pytest


def test_embargo_purge_from_config_matches_yaml_floor():
    from labeling.rl_reward_labeling import max_label_horizon_mult
    from training.cv_splits import embargo_purge_from_config

    emb, pur, meth = embargo_purge_from_config(
        {
            "training": {"seq_len": 80},
            "strategy": {"lookahead_bars": 30},
            "execution": {"delay_bars": 1},
            "validation": {"embargo_bars": 60, "purge_bars": 120, "method": "purged_embargo"},
        }
    )
    # Dynamic floor = 80 + int(30 * max_horizon_mult) + 1  (> yaml 60)
    eff_lh = int(30 * max_label_horizon_mult())
    assert emb == 80 + eff_lh + 1
    assert pur == 120
    assert meth == "purged_embargo"


def test_tabular_walk_forward_has_gap():
    from training.train_xgboost import walk_forward_splits

    cfg = {
        "training": {"seq_len": 20},
        "strategy": {"lookahead_bars": 5},
        "execution": {"delay_bars": 1},
        "validation": {"embargo_bars": 10, "purge_bars": 10, "method": "purged_embargo"},
    }
    splits = walk_forward_splits(2000, folds=3, cfg=cfg)
    assert splits
    for tr, va in splits:
        assert tr[-1] + 1 < va[0], "train must not abut val (need purge/embargo gap)"


def test_pretrain_guardrails_overlap_raises():
    from pretrain.guardrails import PretrainGuardrails

    with pytest.raises(RuntimeError, match="Leakage"):
        PretrainGuardrails().enforce_no_holdout_leakage((0, 100), (90, 120))
    assert PretrainGuardrails().enforce_no_holdout_leakage((0, 80), (80, 100))


def test_broker_bridge_fail_closed_when_disconnected():
    from execution.broker_bridge import BrokerBridge, BrokerNotConnectedError

    b = BrokerBridge(broker="MT5")
    assert not b.is_connected()
    with pytest.raises(BrokerNotConnectedError):
        b.get_positions()
    with pytest.raises(BrokerNotConnectedError):
        b.execute_order("EURUSD", "BUY", 0.01)
    with pytest.raises(BrokerNotConnectedError):
        b.modify_order(1)
    with pytest.raises(BrokerNotConnectedError):
        b.close_position(1)
    with pytest.raises(BrokerNotConnectedError):
        b.get_latency()


def test_broker_bridge_ibkr_requires_ib_insync(monkeypatch):
    import execution.broker_bridge as bb

    monkeypatch.setattr(bb, "_IB_INSYNC", False)
    b = bb.BrokerBridge(broker="IBKR")
    with pytest.raises(ImportError, match="ib_insync"):
        b.connect()


def test_broker_bridge_mt5_latency_from_ping(monkeypatch):
    import types

    import execution.broker_bridge as bb

    class _Info:
        ping = 42

    fake_mt5 = types.SimpleNamespace(
        terminal_info=lambda: _Info(),
        symbol_info_tick=lambda _s: types.SimpleNamespace(bid=1.0, ask=1.1),
    )
    monkeypatch.setattr(bb, "mt5", fake_mt5)
    b = bb.BrokerBridge(broker="MT5")
    b.connected = True
    assert b.get_latency() == 42


def test_stage_timer_accumulates():
    import time

    from training.train_gpu import _StageTimer

    t = _StageTimer()
    with t.stage("a"):
        time.sleep(0.01)
    with t.stage("a"):
        time.sleep(0.01)
    with t.stage("b"):
        time.sleep(0.01)
    s = t.summary()
    assert s["a"] >= 0.015
    assert s["b"] >= 0.005
    assert "a" in s and "b" in s
    g = t.gpu_summary()
    assert g["n_samples"] == 6  # enter+exit x 3 stages
    assert "gpu_util_pct_mean" in g
    assert "gpu_util_pct_max" in g
