"""
tests/test_rl_inaction_and_backtest_gate.py
===========================================
Verification suite for:
  1. Temperature-aware / dynamic confidence thresholding in backtesting & promotion gates.
  2. Prevention of RL inaction collapse: reward scaling (overtrade=0.0005) & directional idle penalty.
  3. Multi-episode evaluation aggregation in evaluate_agent and RLEnsemble.evaluate.
  4. Hard quality gating in Stage 4 optimal roadmap certification.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from backtesting.backtest import ScalingAction
from models.architectures import TemperatureScaler
from models.rl_advanced import RLEnsemble
from models.rl_agents import ForexTradingEnv, PPOAgent, evaluate_agent
from scripts.auto_optimal_roadmap import run_stage_4_certification
from scripts.backtest_model import _extract_model_temperature, compute_effective_min_confidence


# ─────────────────────────────────────────────────────────────────────────────
# 1. Temperature-Aware Confidence Thresholding Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_effective_min_confidence_basic():
    """Verify effective confidence calculation across temperatures."""
    # T = 1.0 (uncalibrated / neutral) -> returns min_confidence exactly
    eff_1 = compute_effective_min_confidence(0.45, temperature=1.0, num_classes=3)
    assert np.isclose(eff_1, 0.45)

    # T = 1.41 (calibrated fold dampening logits) -> scales down towards 1/3
    # 1/3 + (0.45 - 1/3) / 1.41 = 0.333333 + 0.116667 / 1.41 = ~0.416
    eff_cal = compute_effective_min_confidence(0.45, temperature=1.41, num_classes=3)
    assert 0.41 < eff_cal < 0.42
    assert eff_cal < 0.45

    # T = 2.0 -> scaled further down towards 1/3
    eff_2 = compute_effective_min_confidence(0.45, temperature=2.0, num_classes=3)
    assert 0.38 < eff_2 < 0.40

    # Ensure effective threshold is always strictly greater than random chance (1/3 + 0.01)
    assert eff_2 > (1.0 / 3.0 + 0.01)


def test_compute_effective_min_confidence_starvation_protection():
    """Verify starvation protection when max confidence in window is below static threshold."""
    # Suppose a well-calibrated model produces max confidence of 0.38
    # Baseline threshold 0.45 would reject 100% of signals
    eff = compute_effective_min_confidence(0.45, temperature=1.0, num_classes=3, max_observed_conf=0.38)
    assert eff <= 0.38
    assert eff > 1.0 / 3.0


def test_extract_model_temperature():
    """Verify temperature extraction from plain models and TemperatureScaler wrappers."""
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 3)

        def forward(self, x):
            return self.fc(x)

    core = DummyModel()
    assert _extract_model_temperature(core) == 1.0

    scaler = TemperatureScaler(core)
    scaler.temperature.data.fill_(1.4086)
    assert np.isclose(_extract_model_temperature(scaler), 1.4086, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# 2. RL Reward Scaling & Inaction Trap Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_forex_env_reward_weights_default():
    """Verify that ForexTradingEnv defaults overtrade penalty to 0.0005, tx_cost to 1.0, and churn to 0.001."""
    prices = np.array([1.1000, 1.1005, 1.1010, 1.1008, 1.1015], dtype=np.float32)
    features = np.zeros((5, 10), dtype=np.float32)
    atr = np.full(5, 0.0010, dtype=np.float32)
    spreads = np.full(5, 0.0001, dtype=np.float32)

    env = ForexTradingEnv(prices=prices, features=features, atr=atr, spreads=spreads)
    assert env.rw["overtrade"] == pytest.approx(0.0005)
    assert env.rw["idle"] == pytest.approx(1.0)
    assert env.rw["tx_cost"] == pytest.approx(1.0)
    assert env.rw["churn"] == pytest.approx(0.001)
    assert env.enable_idle_penalty is False


def test_forex_env_idle_penalty_on_directional_signal():
    """Verify that choosing HOLD while flat during a strong signal incurs an idle penalty when enabled."""
    prices = np.array([1.1000, 1.1005, 1.1010, 1.1015, 1.1020], dtype=np.float32)
    # Put strong directional signal in first feature column (s = +0.30 > 0.15)
    features = np.zeros((5, 10), dtype=np.float32)
    features[:, 0] = 0.30
    atr = np.full(5, 0.0010, dtype=np.float32)
    spreads = np.full(5, 0.0001, dtype=np.float32)

    env = ForexTradingEnv(
        prices=prices,
        features=features,
        atr=atr,
        spreads=spreads,
        random_reset=False,
        enable_idle_penalty=True,
    )
    env.reset()

    # Step with HOLD (action 0) while position is 0
    obs, reward, done, info = env.step(ScalingAction.HOLD.value)
    # Expected idle penalty: 0.001 * 0.30 = 0.0003
    # Flat position means mtm_pnl = 0, cost = 0, dd = 0, overtrade = 0, churn = 0
    assert reward < 0.0, f"Expected negative reward from idle penalty, got {reward}"
    assert np.isclose(reward, -0.0003, atol=1e-5)


def test_forex_env_default_flat_inaction_no_penalty():
    """Verify that holding flat by default does NOT incur idle penalty (eliminating overtrading churn)."""
    prices = np.array([1.1000, 1.1005, 1.1010, 1.1015, 1.1020], dtype=np.float32)
    features = np.zeros((5, 10), dtype=np.float32)
    features[:, 0] = 0.30  # even with directional signal, idle penalty is False by default
    atr = np.full(5, 0.0010, dtype=np.float32)
    spreads = np.full(5, 0.0001, dtype=np.float32)

    env = ForexTradingEnv(prices=prices, features=features, atr=atr, spreads=spreads, random_reset=False)
    env.reset()

    obs, reward, done, info = env.step(ScalingAction.HOLD.value)
    assert np.isclose(reward, 0.0, atol=1e-6), f"Expected 0 penalty for flat HOLD, got {reward}"


def test_forex_env_churn_penalty_on_position_flip():
    """Verify that flipping position incurs churn penalty proportional to delta_pos."""
    prices = np.array([1.1000, 1.1000, 1.1000, 1.1000, 1.1000], dtype=np.float32)
    features = np.zeros((5, 10), dtype=np.float32)
    atr = np.full(5, 0.0010, dtype=np.float32)
    spreads = np.full(5, 0.0, dtype=np.float32)  # 0 spread to isolate churn

    env = ForexTradingEnv(
        prices=prices,
        features=features,
        atr=atr,
        spreads=spreads,
        commission_per_lot=0.0,
        slippage_pips=0.0,
        random_reset=False,
    )
    env.reset()

    # Step 1: Open Long (delta_pos = 1.0)
    _, r1, _, _ = env.step(ScalingAction.OPEN_LONG.value)
    # Expected: churn_penalty = 0.001 * 1.0 = 0.001, overtrade = 0.0005 * 4 = 0.002
    assert r1 < -0.0025

    # Step 2: Flip to Short (delta_pos = 2.0: from +1.0 to -1.0)
    _, r2, _, _ = env.step(ScalingAction.OPEN_SHORT.value)
    # churn_penalty = 0.001 * 2.0 = 0.002
    assert r2 < -0.0035


def test_forex_env_active_trading_incentive():
    """Verify that entering a trade has low entry penalty (0.0005) compared to old 0.20."""
    prices = np.array([1.1000, 1.1005, 1.1010, 1.1015, 1.1020], dtype=np.float32)
    features = np.zeros((5, 10), dtype=np.float32)
    features[:, 0] = 0.30
    atr = np.full(5, 0.0010, dtype=np.float32)
    spreads = np.full(5, 0.0001, dtype=np.float32)

    env = ForexTradingEnv(prices=prices, features=features, atr=atr, spreads=spreads, random_reset=False)
    env.reset()

    # Enter long on strong signal
    obs, reward, done, info = env.step(ScalingAction.OPEN_LONG.value)
    # Penalty is ~0.0005 overtrade + ATR-normalized execution cost (~0.40) + churn, not -1.0 or -0.20!
    assert reward > -0.50, f"Reward {reward} indicates excessive overtrading penalty"


def test_forex_env_atr_normalized_tx_cost():
    """Verify that transaction friction is normalized by bar ATR in USD."""
    prices = np.array([1.1000, 1.1000, 1.1000], dtype=np.float32)
    features = np.zeros((3, 10), dtype=np.float32)
    # ATR = 0.0010 (10 pips). On lot_size 10,000, bar ATR in USD = 0.0010 * 10,000 = $10.00
    atr = np.full(3, 0.0010, dtype=np.float32)
    spreads = np.zeros(3, dtype=np.float32)

    env = ForexTradingEnv(
        prices=prices,
        features=features,
        atr=atr,
        spreads=spreads,
        commission_per_lot=2.0,
        slippage_pips=0.0,
        lot_size=10_000.0,
        random_reset=False,
    )
    env.reset()

    # Open Long 1 lot: cost = 1.0 * 2.0 = $2.00
    # bar_atr_usd = 0.0010 * 10,000 = $10.00
    # tx_cost_norm = $2.00 / $10.00 = 0.20
    _, reward, _, _ = env.step(ScalingAction.OPEN_LONG.value)
    # Total reward components:
    # mtm_pnl = 0
    # tx_cost = -0.20
    # overtrade = -0.0005 * 4 = -0.002
    # churn = -0.001 * 1.0 = -0.001
    # total expected ~ -0.203
    assert np.isclose(reward, -0.203, atol=1e-3)


def test_forex_env_action_flip_penalty():
    """Verify that changing actions across consecutive steps incurs an action flip penalty."""
    prices = np.array([1.1000, 1.1000, 1.1000, 1.1000], dtype=np.float32)
    features = np.zeros((4, 10), dtype=np.float32)
    atr = np.full(4, 0.0010, dtype=np.float32)
    spreads = np.zeros(4, dtype=np.float32)

    env = ForexTradingEnv(
        prices=prices,
        features=features,
        atr=atr,
        spreads=spreads,
        commission_per_lot=0.0,
        slippage_pips=0.0,
        random_reset=False,
    )
    env.reset()

    # Step 1: HOLD (prev_action is initialized to None, no flip penalty on step 1)
    _, r1, _, _ = env.step(ScalingAction.HOLD.value)
    assert np.isclose(r1, 0.0, atol=1e-6)

    # Step 2: HOLD again (same action, no flip penalty)
    _, r2, _, _ = env.step(ScalingAction.HOLD.value)
    assert np.isclose(r2, 0.0, atol=1e-6)

    # Step 3: Switch action to OPEN_LONG (action 1 != action 0, incurs action_flip penalty = 0.002)
    _, r3, _, _ = env.step(ScalingAction.OPEN_LONG.value)
    # Expected flip penalty 0.002, churn 0.001, overtrade 0.002 -> reward ~ -0.005
    assert r3 < -0.004



# ─────────────────────────────────────────────────────────────────────────────
# 3. Multi-Episode Aggregate Evaluation Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_evaluate_agent_multi_episode_aggregation():
    """Verify evaluate_agent aggregates across all evaluation episodes."""
    np.random.seed(42)
    prices = 1.1000 + np.cumsum(np.random.randn(100) * 0.0002).astype(np.float32)
    features = np.random.randn(100, 8).astype(np.float32)
    features[:, 0] = 0.25  # strong signal
    atr = np.full(100, 0.0010, dtype=np.float32)
    spreads = np.full(100, 0.0001, dtype=np.float32)

    env = ForexTradingEnv(
        prices=prices,
        features=features,
        atr=atr,
        spreads=spreads,
        episode_len=20,
        random_reset=True,
    )

    agent = PPOAgent(obs_size=env.obs_size, n_actions=env.n_actions, device="cpu")
    returns, agg_summary = evaluate_agent(agent, env, n_episodes=3, greedy=True)

    assert len(returns) == 3
    assert "n_trades" in agg_summary
    assert "sharpe" in agg_summary
    assert "episode_summaries" in agg_summary
    assert len(agg_summary["episode_summaries"]) == 3
    assert agg_summary["n_trades"] == sum(s["n_trades"] for s in agg_summary["episode_summaries"])


def test_rl_ensemble_evaluate_multi_episode_aggregation():
    """Verify RLEnsemble.evaluate aggregates across all evaluation episodes."""
    np.random.seed(42)
    prices = 1.1000 + np.cumsum(np.random.randn(100) * 0.0002).astype(np.float32)
    features = np.random.randn(100, 8).astype(np.float32)
    atr = np.full(100, 0.0010, dtype=np.float32)
    spreads = np.full(100, 0.0001, dtype=np.float32)

    env = ForexTradingEnv(
        prices=prices,
        features=features,
        atr=atr,
        spreads=spreads,
        episode_len=20,
        random_reset=True,
    )

    ppo1 = PPOAgent(obs_size=env.obs_size, n_actions=env.n_actions, device="cpu")
    ppo2 = PPOAgent(obs_size=env.obs_size, n_actions=env.n_actions, device="cpu")
    ens = RLEnsemble([ppo1, ppo2], consensus_mode="soft_vote", obs_size=env.obs_size, device="cpu")

    returns, agg_summary, diag = ens.evaluate(env, n_episodes=4, greedy=True)

    assert len(returns) == 4
    assert "n_trades" in agg_summary
    assert "sharpe" in agg_summary
    assert "total_return_pct" in agg_summary
    assert len(agg_summary["episode_summaries"]) == 4
    assert agg_summary["n_trades"] == sum(s["n_trades"] for s in agg_summary["episode_summaries"])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Stage 4 Hard Quality Gate Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_stage_4_certification_quality_gate(monkeypatch):
    """Verify Stage 4 quality gate rejects zero trades and certified valid metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        import scripts.auto_optimal_roadmap as aor
        monkeypatch.setattr(aor, "ENSEMBLE_DIR", tmp_path)

        # Case 1: Zero trades -> FAILED_ZERO_TRADES
        rl_zero = {
            "ensemble_consensus": {
                "n_trades": 0,
                "sharpe": 0.0,
                "eval_return_pct": 0.0,
            }
        }
        (tmp_path / "rl_report.json").write_text(json.dumps(rl_zero), encoding="utf-8")
        cert1 = run_stage_4_certification()
        assert cert1["status"] == "FAILED_ZERO_TRADES"
        assert cert1["quality_gate_passed"] is False

        # Case 2: Insufficient trades (<10) -> REJECTED_INACTION_COLLAPSE
        rl_few = {
            "ensemble_consensus": {
                "n_trades": 5,
                "sharpe": 1.2,
                "eval_return_pct": 2.5,
            }
        }
        (tmp_path / "rl_report.json").write_text(json.dumps(rl_few), encoding="utf-8")
        cert2 = run_stage_4_certification()
        assert cert2["status"] == "REJECTED_INACTION_COLLAPSE"
        assert cert2["quality_gate_passed"] is False

        # Case 3: Negative return or Sharpe -> REJECTED_INACTION_COLLAPSE
        rl_neg = {
            "ensemble_consensus": {
                "n_trades": 45,
                "sharpe": -0.2,
                "eval_return_pct": -1.5,
            }
        }
        (tmp_path / "rl_report.json").write_text(json.dumps(rl_neg), encoding="utf-8")
        cert3 = run_stage_4_certification()
        assert cert3["status"] == "REJECTED_INACTION_COLLAPSE"
        assert cert3["quality_gate_passed"] is False

        # Isolate from real checkpoints/: the fold-consistency check globs them.
        import glob as _glob

        monkeypatch.setattr(_glob, "glob", lambda *a, **k: [])

        # Case 4: Valid metrics under gate v2 (>= MIN_TRADES, low DD, agents agree,
        # every agent profitable) -> CERTIFIED_READY_FOR_DEPLOYMENT
        good_agents = [
            {"agent_id": i, "eval_return_pct": 3.0, "max_drawdown_pct": 5.0} for i in range(3)
        ]
        rl_valid = {
            "individual_agents": good_agents,
            "ensemble_consensus": {
                "n_trades": 1500,
                "sharpe": 1.85,
                "eval_return_pct": 4.2,
                "max_drawdown_pct": 6.0,
                "conflict_rate": 0.2,
                "mean_agreement_score": 0.8,
            },
        }
        (tmp_path / "rl_report.json").write_text(json.dumps(rl_valid), encoding="utf-8")
        cert4 = run_stage_4_certification()
        assert cert4["status"] == "CERTIFIED_READY_FOR_DEPLOYMENT"
        assert cert4["quality_gate_passed"] is True
        assert cert4["gate_version"] >= 2

        # Case 5: consensus looks good but one agent lost money (2026-09-21 shape) -> rejected
        rl_bad_agent = dict(rl_valid)
        rl_bad_agent["individual_agents"] = [
            *good_agents[:2],
            {"agent_id": 2, "eval_return_pct": -122.5, "max_drawdown_pct": 262.0},
        ]
        (tmp_path / "rl_report.json").write_text(json.dumps(rl_bad_agent), encoding="utf-8")
        cert5 = run_stage_4_certification()
        assert cert5["quality_gate_passed"] is False
        assert any("Agent 2" in r for r in cert5["rejection_reasons"])

        # Case 6: 364 trades (the old certified run) is below MIN_TRADES -> rejected
        rl_few_v2 = dict(rl_valid)
        rl_few_v2["ensemble_consensus"] = {**rl_valid["ensemble_consensus"], "n_trades": 364}
        (tmp_path / "rl_report.json").write_text(json.dumps(rl_few_v2), encoding="utf-8")
        assert run_stage_4_certification()["quality_gate_passed"] is False
