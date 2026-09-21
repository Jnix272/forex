"""
tests/test_multi_rl.py
======================
Exhaustive verification suite for the Multi-RL (Multiple Reinforcement Learning Agents) system.

Verifies:
  a. RLEnsemble instantiation with 3 PPO agents and mixed PPO+DQN.
  b. Consensus action selection under all consensus modes ('soft_vote', 'majority', 'conservative').
  c. Agreement / disagreement / policy uncertainty metric calculation.
  d. MultiAgentCoordinator with RLEnsemble per pair.
  e. scripts/train_rl.py execution with --multi-agent --num-agents 2 --episodes 2 on CPU.
  f. Checkpoint save and load verification.
"""

from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtesting.backtest import ScalingAction
from models.rl_advanced import MultiAgentCoordinator, PolicyEnsemble, RLEnsemble
from models.rl_agents import DQNAgent, ForexTradingEnv, PPOAgent


def test_instantiation():
    print("\n--- Test 1: RLEnsemble Instantiation ---")
    obs_size = 16
    n_actions = 10
    device = "cpu"

    # 1a: 3 PPO agents
    ppo1 = PPOAgent(obs_size=obs_size, n_actions=n_actions, lr=3e-4, device=device)
    ppo2 = PPOAgent(obs_size=obs_size, n_actions=n_actions, lr=2.5e-4, device=device)
    ppo3 = PPOAgent(obs_size=obs_size, n_actions=n_actions, lr=3.5e-4, device=device)

    ens_ppo = RLEnsemble(
        agents=[ppo1, ppo2, ppo3],
        agent_types=["ppo", "ppo", "ppo"],
        consensus_mode="soft_vote",
        obs_size=obs_size,
        n_actions=n_actions,
        device=device,
    )
    assert len(ens_ppo.agents) == 3
    assert ens_ppo.agent_types == ["ppo", "ppo", "ppo"]
    assert len(ens_ppo.weights) == 3
    assert np.isclose(sum(ens_ppo.weights), 1.0)
    print("  [PASS] 3 PPO agents instantiated in RLEnsemble")

    # 1b: Mixed PPO + DQN
    dqn1 = DQNAgent(obs_size=obs_size, n_actions=n_actions, lr=1e-4, device=device)
    ens_mixed = RLEnsemble(
        agents=[ppo1, dqn1],
        agent_types=["ppo", "dqn"],
        consensus_mode="majority",
        obs_size=obs_size,
        n_actions=n_actions,
        device=device,
    )
    assert len(ens_mixed.agents) == 2
    assert ens_mixed.agent_types == ["ppo", "dqn"]
    print("  [PASS] Mixed PPO + DQN instantiated in RLEnsemble")

    # 1c: Alias verification
    assert PolicyEnsemble is RLEnsemble
    print("  [PASS] PolicyEnsemble alias verified")


def test_consensus_modes():
    print("\n--- Test 2: Consensus Action Selection & Conflict Handling ---")
    obs_size = 12
    n_actions = 10
    device = "cpu"

    ppo1 = PPOAgent(obs_size=obs_size, n_actions=n_actions, device=device)
    ppo2 = PPOAgent(obs_size=obs_size, n_actions=n_actions, device=device)
    ppo3 = PPOAgent(obs_size=obs_size, n_actions=n_actions, device=device)

    ens = RLEnsemble(
        agents=[ppo1, ppo2, ppo3],
        agent_types=["ppo", "ppo", "ppo"],
        consensus_mode="soft_vote",
        obs_size=obs_size,
        n_actions=n_actions,
        device=device,
    )

    obs = np.random.randn(obs_size).astype(np.float32)

    # Test soft_vote
    info_soft = ens.get_consensus_info(obs, greedy=True, mode_override="soft_vote")
    assert 0 <= info_soft["consensus_action"] < n_actions
    assert "mean_probabilities" in info_soft
    assert np.isclose(info_soft["mean_probabilities"].sum(), 1.0)
    print(f"  [PASS] soft_vote action: {info_soft['consensus_action']}, sum(p)={info_soft['mean_probabilities'].sum():.4f}")

    # Test mock agents with controlled distributions to verify conflict logic
    class MockAgent:
        def __init__(self, action: int, prob_val: float = 0.9):
            self.action = action
            self.prob_val = prob_val
            self.device = torch.device("cpu")

        def select_action(self, obs, mask=None):
            return self.action

    mock_buy = MockAgent(ScalingAction.OPEN_LONG.value)    # 1
    mock_sell = MockAgent(ScalingAction.OPEN_SHORT.value)  # 2
    mock_hold = MockAgent(ScalingAction.HOLD.value)        # 0
    mock_scale100 = MockAgent(ScalingAction.SCALE_IN_100.value)  # 5
    mock_scale25 = MockAgent(ScalingAction.SCALE_IN_25.value)    # 3

    # Conflict test in majority mode: BUY vs SELL conflict -> MUST fall back to HOLD (0)
    ens_conflict = RLEnsemble([mock_buy, mock_sell, mock_buy], consensus_mode="majority")
    info_conflict = ens_conflict.get_consensus_info(obs)
    assert info_conflict["conflict_detected"] is True
    assert info_conflict["consensus_action"] == ScalingAction.HOLD.value, (
        f"Expected HOLD (0) on BUY/SELL conflict, got {info_conflict['consensus_action']}"
    )
    print("  [PASS] majority mode safely fell back to HOLD on BUY vs SELL conflict")

    # Majority mode without conflict: BUY (1), BUY (1), HOLD (0) -> plurality is BUY (1)
    ens_maj = RLEnsemble([mock_buy, mock_buy, mock_hold], consensus_mode="majority")
    info_maj = ens_maj.get_consensus_info(obs)
    assert info_maj["conflict_detected"] is False
    assert info_maj["consensus_action"] == ScalingAction.OPEN_LONG.value
    print("  [PASS] majority mode correctly chose plurality BUY (1)")

    # Conservative mode test 1: ANY agent voting HOLD forces HOLD
    ens_cons_hold = RLEnsemble([mock_buy, mock_buy, mock_hold], consensus_mode="conservative")
    info_cons_hold = ens_cons_hold.get_consensus_info(obs)
    assert info_cons_hold["consensus_action"] == ScalingAction.HOLD.value
    print("  [PASS] conservative mode forced HOLD because one agent voted HOLD")

    # Conservative mode test 2: Opposite directions force HOLD
    ens_cons_opp = RLEnsemble([mock_buy, mock_sell], consensus_mode="conservative")
    info_cons_opp = ens_cons_opp.get_consensus_info(obs)
    assert info_cons_opp["consensus_action"] == ScalingAction.HOLD.value
    print("  [PASS] conservative mode forced HOLD on opposing directions")

    # Conservative mode test 3: Downsizing scale-in from SCALE_IN_100 to SCALE_IN_25
    ens_cons_scale = RLEnsemble([mock_scale100, mock_scale25], consensus_mode="conservative")
    info_cons_scale = ens_cons_scale.get_consensus_info(obs)
    assert info_cons_scale["consensus_action"] == ScalingAction.SCALE_IN_25.value, (
        f"Expected downsized SCALE_IN_25 (3), got {info_cons_scale['consensus_action']}"
    )
    print("  [PASS] conservative mode correctly downsized scale position to SCALE_IN_25 (3)")


def test_agreement_metrics():
    print("\n--- Test 3: Agreement, Disagreement & Policy Uncertainty Metrics ---")
    class FixedAgent:
        def __init__(self, action: int):
            self.action = action

        def select_action(self, obs, mask=None):
            return self.action

    obs = np.zeros(10, dtype=np.float32)

    # Complete agreement: all 3 vote BUY (1)
    ens_agree = RLEnsemble([FixedAgent(1), FixedAgent(1), FixedAgent(1)], consensus_mode="soft_vote")
    info_agree = ens_agree.get_consensus_info(obs)
    assert np.isclose(info_agree["agreement_score"], 1.0)
    assert np.isclose(info_agree["disagreement_score"], 0.0)
    assert np.isclose(info_agree["policy_uncertainty"], 0.0)
    assert info_agree["conflict_detected"] is False
    print("  [PASS] Complete agreement metrics: agreement=1.0, disagreement=0.0, uncertainty=0.0")

    # Partial disagreement: 2 vote BUY (1), 1 votes SELL (2)
    ens_disagree = RLEnsemble([FixedAgent(1), FixedAgent(1), FixedAgent(2)], consensus_mode="soft_vote")
    info_disagree = ens_disagree.get_consensus_info(obs)
    assert 0.0 < info_disagree["agreement_score"] < 1.0
    assert 0.0 < info_disagree["disagreement_score"] < 1.0
    assert np.isclose(info_agree["agreement_score"] + info_agree["disagreement_score"], 1.0)
    assert info_disagree["policy_uncertainty"] > 0.0
    assert info_disagree["conflict_detected"] is True
    print(
        f"  [PASS] Partial disagreement metrics: agreement={info_disagree['agreement_score']:.2f}, "
        f"disagreement={info_disagree['disagreement_score']:.2f}, "
        f"uncertainty={info_disagree['policy_uncertainty']:.3f}, conflict=True"
    )


def test_coordinator_integration():
    print("\n--- Test 4: MultiAgentCoordinator with RLEnsemble per pair ---")
    obs_size = 14 + 32  # pair_obs + global_context
    n_actions = 10
    device = "cpu"

    # Pair 1: EURUSD managed by RLEnsemble (2 PPO agents)
    ppo_eur1 = PPOAgent(obs_size=obs_size, n_actions=n_actions, device=device)
    ppo_eur2 = PPOAgent(obs_size=obs_size, n_actions=n_actions, device=device)
    ens_eur = RLEnsemble([ppo_eur1, ppo_eur2], consensus_mode="soft_vote", obs_size=obs_size, device=device)

    # Pair 2: GBPUSD managed by DQNAgent
    dqn_gbp = DQNAgent(obs_size=obs_size, n_actions=n_actions, device=device)

    coordinator = MultiAgentCoordinator(
        agents={"EURUSD": ens_eur, "GBPUSD": dqn_gbp},
        pairs=["EURUSD", "GBPUSD"],
        max_corr_exposure=1.5,
        global_feat_dim=20,
        context_dim=32,
        device=device,
    )

    observations = {
        "EURUSD": np.random.randn(14).astype(np.float32),
        "GBPUSD": np.random.randn(14).astype(np.float32),
    }
    global_features = np.random.randn(20).astype(np.float32)

    # Test basic action selection
    actions = coordinator.select_actions(observations, global_features=global_features)
    assert "EURUSD" in actions and "GBPUSD" in actions
    assert isinstance(actions["EURUSD"], int)
    assert isinstance(actions["GBPUSD"], int)
    print(f"  [PASS] MultiAgentCoordinator select_actions returned: {actions}")

    # Test action selection with consensus info
    actions, info = coordinator.select_actions(observations, global_features=global_features, return_info=True)
    assert "EURUSD" in info
    assert "agreement_score" in info["EURUSD"]
    print(f"  [PASS] select_actions with return_info=True: EURUSD agreement={info['EURUSD']['agreement_score']:.2f}")

    # Test dynamic agent registration
    ppo_usdjpy = PPOAgent(obs_size=obs_size, n_actions=n_actions, device=device)
    coordinator.set_agent("USDJPY", ppo_usdjpy)
    assert coordinator.get_agent("USDJPY") is ppo_usdjpy
    assert "USDJPY" in coordinator.pairs
    print("  [PASS] Dynamic set_agent / get_agent verified")


def test_checkpoint_roundtrip():
    print("\n--- Test 5: Checkpoint Save & Load Roundtrip ---")
    obs_size = 15
    n_actions = 10
    device = "cpu"

    ppo = PPOAgent(obs_size=obs_size, n_actions=n_actions, hidden=128, lr=3e-4, device=device)
    dqn = DQNAgent(obs_size=obs_size, n_actions=n_actions, hidden=128, lr=1e-4, device=device)

    ens_orig = RLEnsemble(
        agents=[ppo, dqn],
        agent_types=["ppo", "dqn"],
        consensus_mode="majority",
        weights=[0.6, 0.4],
        obs_size=obs_size,
        n_actions=n_actions,
        device=device,
    )

    obs = np.random.randn(obs_size).astype(np.float32)
    orig_info = ens_orig.get_consensus_info(obs, greedy=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "rl_ensemble_test.pt"
        ens_orig.save_checkpoint(ckpt_path, meta={"tag": "unit_test", "seed": 42})
        assert ckpt_path.exists()

        ens_loaded = RLEnsemble.load_checkpoint(ckpt_path, device=device)
        assert len(ens_loaded.agents) == 2
        assert ens_loaded.agent_types == ["ppo", "dqn"]
        assert ens_loaded.consensus_mode == "majority"
        assert np.allclose(ens_loaded.weights, [0.6, 0.4])

        loaded_info = ens_loaded.get_consensus_info(obs, greedy=True)
        assert orig_info["consensus_action"] == loaded_info["consensus_action"]
        assert np.allclose(orig_info["mean_probabilities"], loaded_info["mean_probabilities"], atol=1e-6)
        print("  [PASS] Checkpoint roundtrip preserved weights, configs, and predictions identically")


def test_train_rl_cli():
    print("\n--- Test 6: scripts/train_rl.py CLI Smoke Test (CPU) ---")
    python_exe = sys.executable
    script_path = ROOT / "scripts" / "train_rl.py"
    cache_path = ROOT / "test_rl.zarr"

    with tempfile.TemporaryDirectory() as tmp_ckpt:
        cmd = [
            python_exe,
            str(script_path),
            "--cache",
            str(cache_path),
            "--model-name",
            "dummy",
            "--multi-agent",
            "--num-agents",
            "2",
            "--ensemble-agents",
            "ppo,dqn",
            "--consensus-mode",
            "conservative",
            "--episodes",
            "2",
            "--checkpoint-dir",
            tmp_ckpt,
            "--device",
            "cpu",
        ]

        print(f"  Executing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"  Return code: {result.returncode}")
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            raise RuntimeError(f"scripts/train_rl.py failed with code {result.returncode}")

        out_dir = Path(tmp_ckpt) / "dummy"
        ens_ckpt = out_dir / "rl_ensemble_best.pt"
        rl_best = out_dir / "rl_best.pt"
        report_json = out_dir / "rl_report.json"

        assert ens_ckpt.exists(), f"Missing {ens_ckpt}"
        assert rl_best.exists(), f"Missing {rl_best}"
        assert report_json.exists(), f"Missing {report_json}"

        with open(report_json) as f:
            data = json.load(f)

        assert data["multi_agent"] is True
        assert data["num_agents"] == 2
        assert data["ensemble_agents"] == ["ppo", "dqn"]
        assert data["consensus_mode"] == "conservative"
        assert len(data["individual_agents"]) == 2
        assert "ensemble_consensus" in data
        assert "mean_agreement_score" in data["ensemble_consensus"]
        assert "mean_policy_uncertainty" in data["ensemble_consensus"]

        print("  [PASS] CLI generated rl_ensemble_best.pt, rl_best.pt, and extended rl_report.json")
        print(f"  Consensus Eval Return: {data['ensemble_consensus']['eval_return_pct']:+.2f}%")
        print(f"  Mean Agreement: {data['ensemble_consensus']['mean_agreement_score']:.1%}")
        print(f"  Mean Policy Uncertainty: {data['ensemble_consensus']['mean_policy_uncertainty']:.3f}")


def main():
    print("=" * 65)
    print("STARTING MULTI-RL EXHAUSTIVE VERIFICATION SUITE")
    print("=" * 65)

    test_instantiation()
    test_consensus_modes()
    test_agreement_metrics()
    test_coordinator_integration()
    test_checkpoint_roundtrip()
    test_train_rl_cli()

    print("\n" + "=" * 65)
    print("ALL 6 MULTI-RL VERIFICATION SUITES PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
