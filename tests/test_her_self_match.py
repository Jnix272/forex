"""Tests for the HER self-match fix (audit RA2, 2026-08-07).

Before the fix: ``HERBuffer.end_episode`` with strategy="future" used
``random.randint(t_idx, n-1)`` INCLUSIVE of ``t_idx`` itself. For the
**last** transition ``t_idx == n-1 == future_idx`` was guaranteed, and
the relabels always self-matched (``dist=0``), giving the agent a
guaranteed ``+1.0`` reward for "free" via ``_hindsight_reward``.

After the fix:
- ``strategy="future"`` uses ``random.randint(t_idx + 1, n - 1)``
- The last transition (``t_idx == n-1``) skips HER relabel entirely (no
  future transitions to choose from)
- Self-match guards: even if future_idx coincides with t_idx by some
  other means, ``np.array_equal(her_goal, transition['achieved'])``
  skips the entry to avoid polluting the buffer.

Tests:
- Source-level: the new ``t_idx + 1`` form is present, ``random.randint(t_idx,``
  is gone (in the "future" branch).
- Behavioural: simulate a single-transition episode and verify no HER
  entries get added (since there are no future transitions).
- Behavioural: simulate a multi-transition episode and verify that NO
  HER relabel has ``her_goal == transition['achieved']`` (no self-match).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Source-level checks
# ---------------------------------------------------------------------------


def test_her_future_strategy_uses_t_idx_plus_1():
    path = _ROOT / "models" / "rl_advanced.py"
    if not path.exists():
        pytest.skip("models/rl_advanced.py not found")
    src = path.read_text(encoding="utf-8")
    # The fix: `future_idx = random.randint(t_idx + 1, n - 1)` in the "future" branch
    assert "future_idx = random.randint(t_idx + 1, n - 1)" in src, (
        "HER self-match fix missing: should sample t_idx+1..n-1 in 'future' mode"
    )


def test_her_future_strategy_does_not_use_inclusive_t_idx():
    """The OLD buggy form `random.randint(t_idx, n - 1)` should NOT appear
    in the "future" branch (only t_idx+1 is acceptable there).
    """
    path = _ROOT / "models" / "rl_advanced.py"
    if not path.exists():
        pytest.skip("models/rl_advanced.py not found")
    src = path.read_text(encoding="utf-8")
    # Strip comments
    code_lines = []
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("#") or s.startswith('"""') or s.startswith("'''"):
            continue
        code_lines.append(line)
    code = "\n".join(code_lines)
    # The form `random.randint(t_idx, n - 1)` (with t_idx, not t_idx+1)
    # should not appear in the future-strategy branch. Match the exact
    # pattern: `random.randint(t_idx,` - note the comma without +1.
    import re

    bad_pattern = re.compile(r"random\.randint\(t_idx\s*,")
    matches = bad_pattern.findall(code)
    assert not matches, (
        f"HER bug: `random.randint(t_idx, ...)` must not be in code (should use t_idx+1); matches={matches}"
    )


def test_her_self_match_guard_present():
    """The fix should include an explicit self-match guard via
    ``np.array_equal(her_goal, transition["achieved"])`` to skip degenerate
    relabels."""
    path = _ROOT / "models" / "rl_advanced.py"
    if not path.exists():
        pytest.skip("models/rl_advanced.py not found")
    src = path.read_text(encoding="utf-8")
    assert 'np.array_equal(her_goal, transition["achieved"])' in src, (
        "HER fix missing: should guard against her_goal == achieved"
    )


def test_her_skips_relabel_for_last_transition():
    """The fix should break/skip the HER loop when ``t_idx + 1 >= n``
    (i.e. there are no future transitions to choose from).
    """
    path = _ROOT / "models" / "rl_advanced.py"
    if not path.exists():
        pytest.skip("models/rl_advanced.py not found")
    src = path.read_text(encoding="utf-8")
    assert "t_idx + 1 >= n" in src, "HER fix missing: should skip relabel when t_idx+1 >= n (no future)"


# ---------------------------------------------------------------------------
# Behavioural tests (require numpy + the HERBuffer class)
# ---------------------------------------------------------------------------


def test_single_transition_episode_produces_no_her_relabels():
    """A 1-transition episode: ``t_idx == n-1 == 0``, t_idx+1=1 >= n=1 → skip.
    The buffer should contain ONLY the original transition, no HER entries.
    """
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy not available")
    sys.path.insert(0, str(_ROOT))
    try:
        from models.rl_advanced import HERBuffer
    except Exception as e:
        pytest.skip(f"could not import HERBuffer: {e}")

    her = HERBuffer(capacity=1000, k=4, strategy="future")
    obs = np.array([1.0, 2.0], dtype=np.float32)
    goal = np.array([1.5], dtype=np.float32)
    achieved = np.array([1.0], dtype=np.float32)
    her.store_transition(obs.copy(), 1, 0.0, obs.copy() * 1.01, False, goal.copy(), achieved.copy(), {})
    her.end_episode()
    # After end_episode: should have 1 original + 0 HER entries for "future" strategy
    # (No future to sample from for the single transition)
    assert len(her) == 1, f"Single-transition episode should produce 0 HER relabels, got {len(her) - 1}"


def test_multi_transition_episode_has_no_self_match_relabels():
    """In a multi-transition episode, NONE of the HER relabels should have
    ``her_goal == transition['achieved']`` (the self-match bug).
    """
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy not available")
    sys.path.insert(0, str(_ROOT))
    try:
        from models.rl_advanced import HERBuffer
    except Exception as e:
        pytest.skip(f"could not import HERBuffer: {e}")

    her = HERBuffer(capacity=1000, k=8, strategy="future")
    # 5 transitions with distinct achieved values
    base_obs = np.array([1.0, 2.0], dtype=np.float32)
    for i in range(5):
        achieved = np.array([1.0 + i * 0.01], dtype=np.float32)
        her.store_transition(
            obs=base_obs.copy(),
            action=i % 3,
            reward=float(i),
            next_obs=base_obs.copy(),
            done=(i == 4),
            goal=np.array([1.05], dtype=np.float32),  # original goal
            achieved=achieved,
            info={},
        )
    her.end_episode()
    # Original: 5 transitions. Relabel: at most 5*8=40 minus skips.
    # Each non-skipped relabel should have her_goal != transition['achieved'].
    items = list(her._buffer)
    # First 5 are original (info has no 'her' key)
    her_items = [t for t in items if t.get("info", {}).get("her")]
    for t in her_items:
        goal = np.asarray(t["goal"])
        achieved = np.asarray(t["achieved"])
        assert not np.array_equal(goal, achieved), f"HER self-match found: goal={goal}, achieved={achieved}"
    # And we should have at LEAST 1 HER entry (else the future strategy is broken)
    assert len(her_items) > 0, "Expected at least some HER relabels for the 5-step episode"


def test_random_strategy_also_avoids_self_match():
    """The 'random' strategy should also avoid self-matches via its
    deterministic self-skip path.
    """
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy not available")
    sys.path.insert(0, str(_ROOT))
    try:
        from models.rl_advanced import HERBuffer
    except Exception as e:
        pytest.skip(f"could not import HERBuffer: {e}")

    her = HERBuffer(capacity=1000, k=8, strategy="random")
    base_obs = np.array([1.0, 2.0], dtype=np.float32)
    for i in range(5):
        achieved = np.array([1.0 + i * 0.01], dtype=np.float32)
        her.store_transition(
            base_obs.copy(), i % 3, 0.0, base_obs.copy(), (i == 4), np.array([1.05], dtype=np.float32), achieved, {}
        )
    her.end_episode()
    items = list(her._buffer)
    her_items = [t for t in items if t.get("info", {}).get("her")]
    for t in her_items:
        assert not np.array_equal(t["goal"], t["achieved"]), "Random strategy HER relabel produced a self-match"


def test_two_step_episode_relabels_exist_and_are_distinct():
    """For a 2-transition episode, the first transition (t_idx=0) should get
    HER relabels picking from t_idx=1 only (the future transition). Its
    HER goals should all equal ep[1].achieved, which is != ep[0].achieved.
    """
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy not available")
    sys.path.insert(0, str(_ROOT))
    try:
        from models.rl_advanced import HERBuffer
    except Exception as e:
        pytest.skip(f"could not import HERBuffer: {e}")

    her = HERBuffer(capacity=1000, k=4, strategy="future")
    base_obs = np.array([0.0], dtype=np.float32)
    ep0_ach = np.array([1.00], dtype=np.float32)
    ep1_ach = np.array([1.05], dtype=np.float32)
    her.store_transition(
        base_obs.copy(), 0, 0.0, base_obs.copy(), False, np.array([1.10], dtype=np.float32), ep0_ach, {}
    )
    her.store_transition(
        base_obs.copy(), 1, 0.0, base_obs.copy(), True, np.array([1.10], dtype=np.float32), ep1_ach, {}
    )
    her.end_episode()
    items = list(her._buffer)
    her_items = [t for t in items if t.get("info", {}).get("her")]
    # 2 originals, the first transition has 4 HER entries (all sampling ep[1]),
    # the last transition is now skipped (no future) → 4 HER total
    assert len(her_items) == 4, (
        f"Expected 4 HER relabels for 2-step episode (4 from t_idx=0, 0 from last), got {len(her_items)}"
    )
    # All HER goals should be ep1_ach (the only allowed future transition)
    for t in her_items:
        assert np.array_equal(t["goal"], ep1_ach)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
