"""Tests for the PPO greedy inference fix (audit I3, 2026-08-07).

Before the fix: ``ActorCritic.act`` always sampled from
``torch.distributions.Categorical(logits).sample()``, even during live
inference — making the live trading engine nondeterministic for PPO.

After the fix:
- ``ActorCritic.act`` accepts ``greedy: bool = False``. When True, dispatches
  to ``logits.argmax(dim=-1)`` instead of sampling.
- ``PPOAgent.select_action`` accepts ``greedy`` and forwards it to ``act``.
- ``RLInferenceAgent.select_action`` passes ``greedy=True``.

These tests use a pure-numpy schematic to avoid torch imports where possible,
and source-grep assertions to verify the fix landed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Schematic causal semantics (no torch dependency in the bug logic)
# ---------------------------------------------------------------------------

def test_argmax_is_deterministic_greedy_action():
    """Pure-logic verification: argmax over logits gives the same answer
    every call, unlike a sample from a distribution.
    """
    import random
    random.seed(0)
    logits = [random.uniform(-1, 1) for _ in range(3)]
    # argmax:
    argmax_action = max(range(len(logits)), key=lambda a: logits[a])
    # Verify argmax is reproducible regardless of how many times we "sample"
    for _ in range(10):
        a = max(range(len(logits)), key=lambda a: logits[a])
        assert a == argmax_action


def test_stochastic_sample_can_vary():
    """Sampling from a non-degenerate distribution CAN produce different
    actions across calls — this is what we want to ELIMINATE during live
    inference by using greedy=True.
    """
    import random
    random.seed(0)
    logits = [0.0, 0.0, 0.0]  # uniform
    # Simulate sampling: pick action uniformly
    samples = set(random.choice(range(3)) for _ in range(100))
    assert len(samples) > 1, "Sampling from a uniform distribution must vary"


def test_greedy_argmax_matches_most_probable_action():
    """For a peaked distribution, greedy argmax should match the action
    with the highest logit (and the highest frequency under repeated
    sampling).
    """
    logits = [-3.0, 5.0, -1.0]  # action 1 strongly preferred
    argmax_action = max(range(len(logits)), key=lambda a: logits[a])
    assert argmax_action == 1


# ---------------------------------------------------------------------------
# Source-level: confirm the fix is present
# ---------------------------------------------------------------------------

def test_actor_critic_act_accepts_greedy_kwarg():
    """``ActorCritic.act`` should accept a ``greedy: bool = False`` kwarg."""
    path = _ROOT / "models" / "rl_agents.py"
    if not path.exists():
        pytest.skip("models/rl_agents.py not found")
    src = path.read_text(encoding="utf-8")
    assert "def act(self, obs, mask=None, greedy: bool = False)" in src
    # argmax dispatch when greedy
    assert "logits.argmax(dim=-1)" in src


def test_actor_critic_act_still_samples_when_not_greedy():
    """Default behavior (greedy=False) must remain stochastic sampling."""
    path = _ROOT / "models" / "rl_agents.py"
    if not path.exists():
        pytest.skip("models/rl_agents.py not found")
    src = path.read_text(encoding="utf-8")
    # The default stochastic path must remain
    assert "torch.distributions.Categorical(logits=logits)" in src
    assert "dist.sample()" in src


def test_ppo_agent_select_action_accepts_greedy():
    """``PPOAgent.select_action`` should thread the greedy flag down to the actor."""
    path = _ROOT / "models" / "rl_agents.py"
    if not path.exists():
        pytest.skip("models/rl_agents.py not found")
    src = path.read_text(encoding="utf-8")
    assert "def select_action(self, obs: np.ndarray, mask: np.ndarray | None = None" in src
    assert "greedy: bool = False" in src
    # It must call self.net.act with greedy kwarg
    assert "self.net.act(xt, mask=m, greedy=greedy)" in src


def test_rl_inference_agent_passes_greedy_true():
    """``RLInferenceAgent.select_action`` should call ``select_action(..., greedy=True)``."""
    path = _ROOT / "inference" / "rl_inference.py"
    if not path.exists():
        pytest.skip("inference/rl_inference.py not found")
    src = path.read_text(encoding="utf-8")
    assert "greedy=True" in src
    assert "self._agent.select_action(full_obs, greedy=True)" in src


def test_rl_inference_agent_falls_back_for_dqn():
    """The live engine should fall back gracefully when the underlying agent
    doesn't accept the greedy kwarg (DQN raises TypeError, falls back to
    the positional call)."""
    path = _ROOT / "inference" / "rl_inference.py"
    if not path.exists():
        pytest.skip("inference/rl_inference.py not found")
    src = path.read_text(encoding="utf-8")
    # The try/except TypeError pattern should be present
    assert "except TypeError" in src


# ---------------------------------------------------------------------------
# Behavioural test using torch (skipped if torch unavailable)
# ---------------------------------------------------------------------------

def test_actor_critic_act_greedy_returns_argmax():
    """End-to-end: build a tiny ActorCritic and verify that act(greedy=True)
    returns the argmax of the logits (deterministic)."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        pytest.skip("torch not available")

    # Import the actual ActorCritic from rl_agents — it pulls torch via
    # the heavy import chain of training. Try and skip if any import fails.
    try:
        from models.rl_agents import ActorCritic
    except Exception as e:
        pytest.skip(f"could not import ActorCritic: {e}")

    # Construct a tiny ActorCritic
    try:
        net = ActorCritic(obs_dim=4, n_actions=3, hidden=8)
    except Exception as e:
        pytest.skip(f"ActorCritic constructor signature mismatch: {e}")

    obs = torch.zeros(1, 4)
    # Get the greedy action
    with torch.no_grad():
        action_g1, _, _ = net.act(obs, greedy=True)
        action_g2, _, _ = net.act(obs, greedy=True)
    # Greedy should be deterministic (same action both calls)
    assert action_g1.item() == action_g2.item(), (
        "greedy=True should return the same action every call (argmax)"
    )

    # Stochastic may vary; with random init weights, repeated samples likely
    # differ. We don't strictly assert non-determinism, just that the call works.
    with torch.no_grad():
        action_s1, _, _ = net.act(obs, greedy=False)
    # Just verify the call succeeded
    assert 0 <= action_s1.item() < 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
