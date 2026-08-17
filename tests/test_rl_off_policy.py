"""
Tests for off-policy reward wiring (Improvement #5) in models.rl_agents.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.rl_agents import PPOAgent, _estimate_off_policy_rewards


@pytest.fixture
def ppo_agent():
    return PPOAgent(obs_size=16, n_actions=3, hidden=32, device="cpu")


def test_estimate_off_policy_rewards_ppo(ppo_agent):
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(40, 16))
    actions = rng.integers(0, 3, 40)
    rewards = rng.normal(0.05, 0.2, 40)
    est = _estimate_off_policy_rewards(ppo_agent, obs, actions, rewards, episode=0)
    assert est is not None
    assert est["episode"] == 0
    assert est["n_steps"] == 40
    assert isinstance(est["ips_value"], float)
    assert isinstance(est["dr_value"], float)


def test_estimate_off_policy_rewards_dqn_returns_none():
    from models.rl_agents import DQNAgent

    dqn = DQNAgent(obs_size=16, n_actions=3, device="cpu")
    rng = np.random.default_rng(1)
    obs = rng.normal(size=(20, 16))
    actions = rng.integers(0, 3, 20)
    rewards = rng.normal(size=20)
    # DQN exposes no net.actor -> returns None (graceful skip)
    est = _estimate_off_policy_rewards(dqn, obs, actions, rewards, episode=1)
    assert est is None


def test_estimate_off_policy_rewards_empty_actions(ppo_agent):
    est = _estimate_off_policy_rewards(ppo_agent, [], [], [], episode=2)
    assert est is None
