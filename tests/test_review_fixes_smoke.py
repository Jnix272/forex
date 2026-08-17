"""Smoke tests for code-review fix items (no training runs)."""

from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_embargo_bars_includes_delay():
    from labeling.rl_reward_labeling import max_label_horizon_mult
    from training.train_gpu import _embargo_bars

    class A:
        seq_len = 60
        execution_delay_bars = 2
        lookahead_bars = 15

    e = _embargo_bars(A())
    # Embargo floor uses base_LH * max(LABEL_REGIME horizon mults), not bare LH.
    eff_lh = int(15 * max_label_horizon_mult())
    assert e == 60 + eff_lh + 2


def test_multitask_class_index_from_direction_labels():
    from training.train_gpu import _reward_to_class_index, labels_to_class_index

    y_dir = torch.tensor([-1.0, 0.0, 1.0])
    assert labels_to_class_index(y_dir).tolist() == [0, 1, 2]
    y_rew = torch.tensor([-2.0, 0.1, 3.0])
    assert _reward_to_class_index(y_rew).tolist() == [0, 1, 2]


def test_forced_logits_map_model_classes_to_live_actions():
    from inference.pytorch_inference import _logits_to_proba
    from trading.live_actions import LiveAction, model_class_to_live_action

    cases = [
        (np.array([9.0, 0.0, 0.0]), LiveAction.SELL),
        (np.array([0.0, 9.0, 0.0]), LiveAction.HOLD),
        (np.array([0.0, 0.0, 9.0]), LiveAction.BUY),
    ]
    for logits, expected in cases:
        model_class = int(np.argmax(logits))
        assert model_class_to_live_action(model_class) == int(expected)

    assert int(_logits_to_proba(np.array([-1.0])).argmax()) == 0
    assert int(_logits_to_proba(np.array([1.0])).argmax()) == 2


def test_rl_warmup_returns_live_hold_action():
    from inference.rl_inference import RLInferenceAgent
    from trading.live_actions import LiveAction

    agent = RLInferenceAgent.__new__(RLInferenceAgent)
    agent.seq_len = 3
    agent._feat_buffer = []

    assert agent.select_action(np.array([1.0, 2.0], dtype=np.float32)) == int(LiveAction.HOLD)


def test_live_feature_path_keeps_polars_for_realistic_bars():
    from features.feature_engineering import FeatureEngineer
    from trading.live_engine import _ensure_polars_frame

    ts = pd.date_range("2024-01-02", periods=90, freq="1min", tz="UTC")
    close = 1.0850 + np.sin(np.arange(len(ts)) / 12.0) * 0.0005
    bars = pl.DataFrame(
        {
            "timestamp_utc": ts,
            "open": close - 0.00005,
            "high": close + 0.00020,
            "low": close - 0.00020,
            "close": close,
            "volume": np.full(len(ts), 100),
            "bid_close": close - 0.00005,
            "ask_close": close + 0.00005,
            "spread_avg": np.full(len(ts), 0.00010),
        }
    )

    bars = _ensure_polars_frame(bars)
    feats = FeatureEngineer(atr_window=6, lag_windows=[5, 20, 60]).build(bars)

    assert isinstance(feats, pl.DataFrame)
    assert "timestamp_utc" in feats.columns
    assert "atr_6" in feats.columns
    assert len(feats) > 0


def test_polars_rl_labels_and_alignment():
    from labeling.rl_reward_labeling import align_labels_with_features, compute_rl_reward_labels

    ts = pd.date_range("2024-01-02", periods=30, freq="1min", tz="UTC")
    close = 1.1000 + np.arange(len(ts)) * 0.00005
    bars = pl.DataFrame(
        {
            "timestamp_utc": ts,
            "open": close,
            "high": close + 0.0001,
            "low": close - 0.0001,
            "close": close,
            "bid_close": close - 0.00005,
            "ask_close": close + 0.00005,
        }
    )
    feats = pl.DataFrame(
        {
            "timestamp_utc": ts,
            "atr_6": np.full(len(ts), 0.0002),
            "spread_pips": np.full(len(ts), 1.0),
            "feature_x": np.arange(len(ts), dtype=float),
        }
    )

    labels = compute_rl_reward_labels(bars, feats, lookahead_bars=3, execution_delay_bars=1)
    X, y, _sidecar = align_labels_with_features(labels, feats)

    assert isinstance(labels, pl.DataFrame)
    assert isinstance(X, pl.DataFrame)
    assert isinstance(y, pl.Series)
    assert "timestamp_utc" not in X.columns
    assert len(X) == len(y)


def test_alignment_keeps_direction_label_when_target_is_reward():
    from labeling.rl_reward_labeling import align_labels_with_features

    ts = pd.date_range("2024-01-02", periods=4, freq="1min", tz="UTC")
    feats = pd.DataFrame({"feature_x": np.arange(len(ts), dtype=float)}, index=ts)
    labels = pd.DataFrame(
        {
            "reward": [0.10, 0.05, -0.02, 0.00],
            "label": [1, -1, -1, 0],
            "path_quality": [0.9, 0.8, 0.7, 0.6],
        },
        index=ts,
    )

    X, y, sidecar = align_labels_with_features(labels, feats, target_col="reward")

    assert "label" not in X.columns
    assert y.tolist() == labels["reward"].tolist()
    assert sidecar["label"].tolist() == labels["label"].tolist()


def test_cache_integrity_rejects_missing_or_mismatched_direction_sidecars(monkeypatch):
    import training.cache_integrity as ci
    import training.train_gpu as tg

    class Args:
        label_method = "rl_reward"
        ignore_manifest = True

    monkeypatch.setattr(ci.Path, "is_dir", lambda self: False)

    monkeypatch.setattr(
        ci,
        "_cache_length_snapshot",
        lambda cache_path: {"npy_X": 10, "npy_y": 10, "npy_diff": 10},
    )
    ok, reason = tg._validate_cache_integrity("dummy_cache", Args())
    assert not ok
    assert "missing y_cls" in reason

    monkeypatch.setattr(
        ci,
        "_cache_length_snapshot",
        lambda cache_path: {"npy_X": 10, "npy_y": 10, "npy_y_cls": 10, "npy_diff": 9},
    )
    ok, reason = tg._validate_cache_integrity("dummy_cache", Args())
    assert not ok
    assert "npy_diff=9" in reason


def test_postprocess_cache_integrity_check_fails_after_bad_processing(monkeypatch):
    import pytest

    import training.cache_integrity as ci
    import training.train_gpu as tg

    class Args:
        label_method = "rl_reward"
        ignore_manifest = True

    monkeypatch.setattr(ci.Path, "is_dir", lambda self: False)
    monkeypatch.setattr(
        ci,
        "_cache_length_snapshot",
        lambda cache_path: {"npy_X": 10, "npy_y": 10, "npy_y_cls": 10, "npy_pq": 8},
    )

    with pytest.raises(RuntimeError, match="Post-processing cache integrity failed"):
        tg._postprocess_cache_integrity_check("dummy_cache", Args())


def test_feature_schema_mismatch_fails_fast():
    import inspect

    import training.train_gpu as tg

    source = inspect.getsource(tg._build_chunk)

    assert "Feature schema/order changed between chunks" in source
    assert "[FATAL DEBUG]" not in source
    assert "missing_cols.txt" not in source


def test_train_epoch_does_not_unconditionally_mix_labels():
    import inspect

    import training.train_gpu as tg

    source = inspect.getsource(tg.train_epoch)

    assert "torch.distributions.Beta" not in source
    assert "yb = lam * yb" not in source


def test_oanda_env_alias_and_net_short_exposure(monkeypatch):
    from trading.live_engine import OANDABroker

    monkeypatch.delenv("OANDA_API_TOKEN", raising=False)
    monkeypatch.setenv("OANDA_BEARER_TOKEN", "dummy-token")
    monkeypatch.setenv("OANDA_ACCOUNT_ID", "dummy-account")
    broker = OANDABroker()
    assert broker._token == "dummy-token"

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "positions": [
                        {"instrument": "EUR_USD", "long": {"units": "20000"}, "short": {"units": "5000"}},
                        {"instrument": "GBP_USD", "long": {"units": "0"}, "short": {"units": "-12000"}},
                    ]
                }
            ).encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: _Resp())

    positions = broker.get_positions()
    assert positions["EURUSD"] == 1.5
    assert positions["GBPUSD"] == -1.2


def test_backtest_net_pnl_matches_equity_delta():
    from backtesting.backtest import ForexScalingBacktest, ScalingAction

    idx = pd.date_range("2024-01-02", periods=3, freq="1min", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [1.0000, 1.0010, 1.0010],
            "high": [1.0002, 1.0012, 1.0012],
            "low": [0.9998, 1.0008, 1.0008],
            "close": [1.0000, 1.0010, 1.0010],
            "bid_close": [1.0000, 1.0010, 1.0010],
            "ask_close": [1.0000, 1.0010, 1.0010],
            "spread_avg": [0.0, 0.0, 0.0],
        },
        index=idx,
    )
    signals = pd.DataFrame(
        {
            "action": [ScalingAction.OPEN_LONG, ScalingAction.CLOSE_ALL, ScalingAction.HOLD],
            "lots": [1.0, 1.0, 1.0],
            "stop_loss": [0.5, 0.5, 0.5],
            "take_profit": [2.0, 2.0, 2.0],
        },
        index=idx,
    )

    bt = ForexScalingBacktest(
        bars,
        signals,
        initial_equity=10_000.0,
        commission_per_lot=3.5,
        slippage_pips=0.0,
        pip_value_per_lot=1.0,
        execution_delay_bars=0,
        apply_market_impact=False,
    )
    bt.run()
    metrics = bt.performance_metrics()

    assert metrics["gross_pnl_usd"] == pytest.approx(10.0)
    assert metrics["total_commission_usd"] == 7.0
    assert metrics["net_pnl_usd"] == bt.equity - bt.initial_equity
    assert metrics["net_pnl_usd"] == pytest.approx(3.0)


def test_nonfinite_training_targets_are_sanitized():
    from training.train_gpu import _sanitize_batch_tensors, labels_to_class_index

    xb = torch.tensor([[[float("nan"), float("inf"), -float("inf"), 2.0]]])
    yb = torch.tensor([float("nan")])
    y_cls = torch.tensor([float("inf")])
    y_conf = torch.tensor([-float("inf")])

    xb, yb, y_cls, y_conf, keep = _sanitize_batch_tensors(xb, yb, y_cls, y_conf)

    assert torch.isfinite(xb).all()
    # Targets are NOT silently zeroed - keep mask surfaces bad rows for the caller to drop.
    assert keep is not None
    assert bool(keep[0]) is False
    assert not torch.isfinite(yb).all()
    # Finite rows still clamp confidence into [0, 1] when present
    yb2 = torch.tensor([0.5])
    y_cls2 = torch.tensor([0.0])
    y_conf2 = torch.tensor([0.7])
    xb2 = torch.ones(1, 1, 4)
    xb2, yb2, y_cls2, y_conf2, keep2 = _sanitize_batch_tensors(xb2, yb2, y_cls2, y_conf2)
    assert bool(keep2.all())
    assert labels_to_class_index(y_cls2).tolist() == [1]
    assert y_conf2.tolist() == pytest.approx([0.7])


def test_train_epoch_without_teacher_model_does_not_name_error():
    from torch.utils.data import DataLoader, TensorDataset

    from training.train_gpu import train_epoch

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(8, 3),
    )
    loader = DataLoader(
        TensorDataset(
            torch.randn(4, 2, 4),
            torch.tensor([-1.0, 0.0, 1.0, 0.0]),
        ),
        batch_size=2,
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler(enabled=False)

    loss = train_epoch(
        model,
        loader,
        opt,
        torch.nn.CrossEntropyLoss(),
        scaler,
        torch.device("cpu"),
        False,
        True,
        thermal_limit=0,
    )

    assert isinstance(loss, float)
    assert np.isfinite(loss)


def test_haelt_sanitizes_nonfinite_inputs():
    from models.architectures import HAELTHybrid

    model = HAELTHybrid(
        input_size=8,
        seq_len=4,
        lstm_hidden=8,
        d_model=8,
        nhead=2,
        n_layers=1,
        dropout=0.0,
        num_classes=3,
    )
    x = torch.randn(2, 4, 8)
    x[0, 0, 0] = float("nan")
    x[1, 0, 0] = float("inf")

    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 3)
    assert torch.isfinite(out).all()


def test_triple_barrier_accepts_execution_delay():
    from labeling.triple_barrier_labeling import compute_triple_barrier_labels

    sig = inspect.signature(compute_triple_barrier_labels)
    assert "execution_delay_bars" in sig.parameters


def test_multitask_loss_confidence_target():
    from models.architectures import MultiTaskLoss

    crit = MultiTaskLoss()
    logits = torch.zeros(4, 3)
    ret = torch.randn(4)
    conf = torch.randn(4)
    y_cls = torch.tensor([0, 1, 2, 1])
    y_cont = torch.randn(4)
    y_pq = torch.tensor([0.2, 0.8, 0.5, 0.9])
    loss = crit(logits, ret, conf, y_cls, y_cont, y_pq)
    assert torch.isfinite(loss)


def test_multitask_class_balance_penalizes_prediction_collapse():
    from models.architectures import MultiTaskLoss

    y_cls = torch.tensor([0, 1, 2, 1, 2, 0])
    y_cont = torch.zeros(6)
    ret = torch.zeros(6)
    conf = torch.zeros(6)
    collapsed_logits = torch.tensor([[8.0, -4.0, -4.0]] * 6)
    balanced_logits = torch.zeros(6, 3)

    crit = MultiTaskLoss(class_balance_weight=1.0, entropy_weight=0.0, focal_gamma=0.0)

    collapsed = crit(collapsed_logits, ret, conf, y_cls, y_cont)
    balanced = crit(balanced_logits, ret, conf, y_cls, y_cont)

    assert collapsed > balanced


def test_multitask_class_balance_penalizes_missing_buy_class():
    from models.architectures import MultiTaskLoss

    y_cls = torch.tensor([0, 1, 2, 2, 2, 0])
    y_cont = torch.zeros(6)
    ret = torch.zeros(6)
    conf = torch.zeros(6)
    no_buy_logits = torch.tensor([[3.0, 3.0, -6.0]] * 6)
    label_dist_logits = torch.log(torch.tensor([[2 / 6, 1 / 6, 3 / 6]] * 6))

    crit = MultiTaskLoss(class_balance_weight=1.0, entropy_weight=0.0, focal_gamma=0.0)

    no_buy = crit(no_buy_logits, ret, conf, y_cls, y_cont)
    label_dist = crit(label_dist_logits, ret, conf, y_cls, y_cont)

    assert no_buy > label_dist


def test_multitask_class_balance_uses_fold_prior_target():
    from models.architectures import MultiTaskLoss

    y_cls = torch.tensor([1, 1, 1, 1])
    y_cont = torch.zeros(4)
    ret = torch.zeros(4)
    conf = torch.zeros(4)
    fold_prior = torch.tensor([0.37, 0.26, 0.37])
    hold_only_logits = torch.tensor([[-3.0, 4.0, -3.0]] * 4)
    prior_like_logits = torch.log(fold_prior).repeat(4, 1)

    crit = MultiTaskLoss(
        class_balance_weight=1.0,
        entropy_weight=0.0,
        focal_gamma=0.0,
        class_prior=fold_prior,
    )

    hold_only = crit(hold_only_logits, ret, conf, y_cls, y_cont)
    prior_like = crit(prior_like_logits, ret, conf, y_cls, y_cont)

    assert hold_only > prior_like


def test_multitask_balance_pushes_true_buy_logits_above_others():
    from models.architectures import MultiTaskLoss

    y_cls = torch.tensor([2, 2, 2, 0, 1, 1])
    y_cont = torch.zeros(6)
    ret = torch.zeros(6)
    conf = torch.zeros(6)
    buy_loses_logits = torch.tensor(
        [
            [2.0, 1.5, -1.0],
            [1.0, 2.0, -1.0],
            [2.0, 0.0, -0.5],
            [2.0, 0.0, -1.0],
            [0.0, 2.0, -1.0],
            [0.0, 2.0, -1.0],
        ]
    )
    buy_wins_logits = buy_loses_logits.clone()
    buy_wins_logits[:3] = torch.tensor(
        [
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ]
    )

    crit = MultiTaskLoss(class_balance_weight=1.0, entropy_weight=0.0, focal_gamma=0.0)

    buy_loses = crit(buy_loses_logits, ret, conf, y_cls, y_cont)
    buy_wins = crit(buy_wins_logits, ret, conf, y_cls, y_cont)

    assert buy_loses > buy_wins


def test_multitask_balance_keeps_rare_buy_from_being_ignored():
    from models.architectures import MultiTaskLoss

    y_cls = torch.tensor([0] * 12 + [1] * 12 + [2])
    y_cont = torch.zeros(25)
    ret = torch.zeros(25)
    conf = torch.zeros(25)
    logits_buy_loses = torch.zeros(25, 3)
    logits_buy_loses[:12, 0] = 2.0
    logits_buy_loses[12:24, 1] = 2.0
    logits_buy_loses[24] = torch.tensor([2.0, 1.0, -1.0])

    logits_buy_wins = logits_buy_loses.clone()
    logits_buy_wins[24] = torch.tensor([0.0, 0.0, 2.0])

    crit = MultiTaskLoss(
        class_balance_weight=1.0,
        entropy_weight=0.0,
        focal_gamma=0.0,
        class_prior=torch.tensor([0.37, 0.26, 0.37]),
    )

    buy_loses = crit(logits_buy_loses, ret, conf, y_cls, y_cont)
    buy_wins = crit(logits_buy_wins, ret, conf, y_cls, y_cont)

    assert buy_loses > buy_wins


def test_balanced_direction_indices_uses_all_classes(monkeypatch):
    import training.direction_control as dc
    import training.train_gpu as tg

    idx = np.arange(9, dtype=np.int64)
    y_cls = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.float32)
    monkeypatch.setattr(dc, "_read_y_cls_indices", lambda cache_path, indices: y_cls[indices])

    out = tg._balanced_direction_indices("cache", idx, total_samples=6, seed=7)
    out_labels = y_cls[out]

    assert set(out_labels.tolist()) == {-1.0, 0.0, 1.0}
    assert len(out) == 6


def test_direction_gate_detects_collapsed_buy_predictions():
    import training.train_gpu as tg

    class Args:
        direction_min_pred_class_share = 0.05
        direction_max_pred_class_share = 0.80
        direction_min_recall = 0.001

    diag = {
        "pred": [174235, 83284, 44],
        "true": [95758, 67782, 94023],
        "recall": [0.8, 0.4, 0.0],
    }

    failed, reason = tg._direction_gate_failed(diag, Args())

    assert failed
    assert "min_pred_share" in reason


def test_direction_gate_passes_balanced_predictions():
    import training.train_gpu as tg

    class Args:
        direction_min_pred_class_share = 0.05
        direction_max_pred_class_share = 0.80
        direction_min_recall = 0.001

    diag = {
        "pred": [90, 70, 85],
        "true": [95, 68, 82],
        "recall": [0.2, 0.15, 0.12],
    }

    failed, reason = tg._direction_gate_failed(diag, Args())

    assert not failed
    assert reason == "ok"


def test_rl_reward_validation_sharpe_resigns_best_side_reward():
    pred_cls = torch.tensor([0, 2, 1])  # sell, buy, hold
    direction = pred_cls.float() - 1.0
    best_side_reward = torch.tensor([2.0, 3.0, 5.0])
    y_cls = torch.tensor([-1.0, 1.0, 0.0])

    signed_reward = best_side_reward * y_cls.sign()
    pnl = direction * signed_reward

    assert pnl.tolist() == [2.0, 3.0, 0.0]


def test_ast_parse_edited_modules():
    paths = [
        ROOT / "training" / "train_gpu.py",
        ROOT / "data" / "historical_news.py",
        ROOT / "models" / "rl_agents.py",
        ROOT / "labeling" / "triple_barrier_labeling.py",
        ROOT / "models" / "architectures.py",
        ROOT / "infrastructure" / "ollama_helper.py",
        ROOT / "scripts" / "train.py",
        ROOT / "scripts" / "continuous_finetune.py",
        ROOT / "monitoring" / "train_logger.py",
        ROOT / "scripts" / "download_cot.py",
        ROOT / "tests" / "test_rl_market_arrays.py",
        ROOT / "trading" / "live_actions.py",
        ROOT / "trading" / "live_engine.py",
        ROOT / "trading" / "live_guards.py",
        ROOT / "inference" / "rl_inference.py",
        ROOT / "labeling" / "rl_reward_labeling.py",
        ROOT / "backtesting" / "backtest.py",
    ]
    for p in paths:
        ast.parse(p.read_text(encoding="utf-8"), filename=str(p))


if __name__ == "__main__":
    test_embargo_bars_includes_delay()
    test_multitask_class_index_from_direction_labels()
    test_forced_logits_map_model_classes_to_live_actions()
    test_nonfinite_training_targets_are_sanitized()
    test_train_epoch_without_teacher_model_does_not_name_error()
    test_haelt_sanitizes_nonfinite_inputs()
    test_triple_barrier_accepts_execution_delay()
    test_multitask_loss_confidence_target()
    test_multitask_class_balance_penalizes_prediction_collapse()
    test_rl_reward_validation_sharpe_resigns_best_side_reward()
    test_ast_parse_edited_modules()
    print("OK: all smoke tests passed")
