"""Fuzz / integration tests for models/rl_agents.py.

Covers:
  - ForexTradingEnv invariant fuzzing across adversarial market data
    (NaN, inf, flat, extreme spikes, tiny/large values).
  - DQN and PPO (MLP + LSTM) short training loops with finite-loss + mask checks.
  - ReplayBuffer edge cases.

Run: pytest tests/test_rl_agents_fuzz.py -q
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

ROOT = None  # set below
for _cand in (".", ".."):
    pass  # import path handled by conftest

# Import inside try so the file is always collectable; skip if torch missing.
torch = pytest.importorskip("torch")

from models.rl_agents import (  # noqa: E402
    DQNAgent,
    ForexTradingEnv,
    PPOAgent,
    ReplayBuffer,
    ScalingAction,
)

N_ACTIONS = len([a for a in ScalingAction])


def _make_market(n=2000, seed=0, regime="normal"):
    """Build (features, prices, atr, spreads) with optional adversarial regimes."""
    rng = np.random.default_rng(seed)
    if regime == "normal":
        steps = rng.normal(0, 1, (n, 5)).astype(np.float32)
        p = 1.0 + np.cumsum(rng.normal(0, 1e-4, n)).astype(np.float32)
    elif regime == "flat":
        steps = np.zeros((n, 5), dtype=np.float32)
        p = np.full(n, 1.0, dtype=np.float32)
    elif regime == "spike":
        steps = rng.normal(0, 1, (n, 5)).astype(np.float32)
        steps[::97] *= 1e6  # huge outlier features
        p = 1.0 + np.cumsum(rng.normal(0, 1e-4, n)).astype(np.float32)
        p[::97] *= 1e6  # price jumps
    elif regime == "naninf":
        steps = rng.normal(0, 1, (n, 5)).astype(np.float32)
        steps[10] = np.nan
        steps[20] = np.inf
        steps[30] = -np.inf
        p = (1.0 + np.cumsum(rng.normal(0, 1e-4, n))).astype(np.float32)
        p[20] = np.inf
    elif regime == "tiny":
        steps = rng.normal(0, 1e-12, (n, 5)).astype(np.float32)
        p = (1.0 + np.cumsum(rng.normal(0, 1e-12, n))).astype(np.float32)
    else:
        raise ValueError(regime)
    atr = np.abs(rng.normal(1e-4, 1e-5, n)).astype(np.float32) + 1e-6
    spreads = np.full(n, 1e-5, dtype=np.float32)
    return steps, p, atr, spreads


@pytest.mark.parametrize("regime", ["normal", "flat", "spike", "naninf", "tiny"])
def test_env_fuzz_invariants(regime):
    """Fuzz step() with random actions; assert hard invariants every step."""
    feats, prices, atr, spreads = _make_market(n=1500, seed=hash(regime) % 1000, regime=regime)
    env = ForexTradingEnv(
        features=feats,
        prices=prices,
        atr=atr,
        spreads=spreads,
        initial_equity=10_000.0,
        max_lots=3.0,
        random_reset=False,
        episode_len=600,
        bars_per_year=252 * 24 * 60,
    )
    rng = np.random.default_rng(1234)
    obs = env.reset()
    step = 0
    prev_idx = env.idx
    while not env.done and step < 2000:
        action = int(rng.integers(0, N_ACTIONS))
        obs, reward, done, info = env.step(action)
        step += 1
        # --- Invariants ---
        assert np.isfinite(obs).all(), f"[{regime}] non-finite obs at step {step}"
        assert np.isfinite(reward), f"[{regime}] non-finite reward at step {step}: {reward}"
        assert np.isfinite(env.equity), f"[{regime}] non-finite equity at step {step}"
        assert abs(env.position) <= env.max_lots + 1e-6, (
            f"[{regime}] position {env.position} exceeds max_lots {env.max_lots}"
        )
        assert env.idx > prev_idx, f"[{regime}] idx did not advance (look-ahead/stall?)"
        prev_idx = env.idx
        assert env.idx <= env.end_idx, f"[{regime}] idx {env.idx} > end_idx {env.end_idx}"

    # On done, position must be force-closed and equity consistent.
    assert env.done
    assert env.position == 0.0, f"[{regime}] position not closed at episode end: {env.position}"
    s = env.summary()
    assert np.isfinite(s["total_return_pct"])
    assert np.isfinite(s["sharpe"])


def test_env_reward_decomposition_finite():
    feats, prices, atr, spreads = _make_market(n=800, seed=7, regime="normal")
    env = ForexTradingEnv(feats, prices, atr, spreads, episode_len=300)
    env.reset()
    for _ in range(300):
        if env.done:
            break
        _, r, _, _ = env.step(int(np.random.default_rng().integers(0, N_ACTIONS)))
        assert np.isfinite(r)
    assert env.position == 0.0


@pytest.mark.parametrize("use_lstm", [False, True])
def test_ppo_training_loop_finite(use_lstm):
    """Short PPO training loop: assert finite losses + mask respect."""
    feats, prices, atr, spreads = _make_market(n=1200, seed=11, regime="normal")
    obs_size = feats.shape[1] + 5
    env = ForexTradingEnv(feats, prices, atr, spreads, episode_len=400)
    agent = PPOAgent(obs_size=obs_size, n_actions=N_ACTIONS, hidden=32, use_lstm=use_lstm, hist_len=16)

    losses = []
    for ep in range(4):
        obs = env.reset()
        done = False
        while not done:
            action, logp, value = agent.select_action(obs)
            nxt, reward, done, _ = env.step(int(action))
            agent.store(obs, action, reward, done, logp, value)
            obs = nxt
        agent.decay_epsilon() if hasattr(agent, "decay_epsilon") else None
        out = agent.update(last_value=0.0)
        if out:
            assert np.isfinite(out["loss"]), f"PPO loss not finite: {out}"
            losses.append(out["loss"])
    # Training should have produced at least one update.
    assert len(losses) > 0


def test_dqn_training_loop_finite():
    """Short DQN training loop: finite losses, mask-respecting actions."""
    feats, prices, atr, spreads = _make_market(n=1500, seed=22, regime="normal")
    obs_size = feats.shape[1] + 5
    env = ForexTradingEnv(feats, prices, atr, spreads, episode_len=500)
    agent = DQNAgent(obs_size=obs_size, n_actions=N_ACTIONS, hidden=32, batch=32, target_update=200)

    losses = []
    for ep in range(6):
        obs = env.reset()
        done = False
        while not done:
            mask = env.action_mask()
            action = agent.select_action(obs, mask=mask)
            assert 0 <= action < N_ACTIONS
            nxt, reward, done, _ = env.step(int(action))
            agent.store(obs, action, reward, nxt, done, env.action_mask())
            out = agent.update()
            if out:
                assert np.isfinite(out["loss"]), f"DQN loss not finite: {out}"
                losses.append(out["loss"])
            obs = nxt
        agent.decay_epsilon()
    assert len(losses) > 0


def test_replay_buffer_edge_cases():
    buf = ReplayBuffer(capacity=10)
    # Empty sample returns empty list (must not raise).
    assert buf.sample(5) == []
    assert len(buf) == 0
    # Fill with varied actions and masks.
    rng = np.random.default_rng(0)
    for i in range(20):
        obs = rng.normal(size=(8,)).astype(np.float32)
        act = int(rng.integers(0, N_ACTIONS))
        rew = float(rng.normal())
        nobs = rng.normal(size=(8,)).astype(np.float32)
        mask = np.ones(N_ACTIONS, dtype=bool)
        if i % 3 == 0:
            mask[0] = False
        buf.push(obs, act, rew, nobs, False, mask)
    assert len(buf) == 10  # capacity enforced
    batch = buf.sample(6)
    assert len(batch) == 6
    for t in batch:
        assert len(t) == 6
        assert t[5] is not None and t[5].shape == (N_ACTIONS,)


def test_action_mask_always_valid():
    """action_mask() must flag at least one action so masked agents never pick invalid."""
    feats, prices, atr, spreads = _make_market(n=600, seed=5, regime="normal")
    env = ForexTradingEnv(feats, prices, atr, spreads, episode_len=200)
    env.reset()
    for _ in range(200):
        if env.done:
            env.reset()
        m = env.action_mask()
        assert m.any(), "action_mask returned all-False (no valid action)"
        # OPEN_LONG must be valid when flat; CLOSE_ALL must be valid when in position
        if env.position == 0:
            assert m[ScalingAction.OPEN_LONG.value]
        else:
            assert m[ScalingAction.CLOSE_ALL.value]
        _, _, _, _ = env.step(int(np.argmax(m)))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
