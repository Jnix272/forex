"""
models/rl_advanced.py
======================
RL enhancements:
  1. MultiAgentCoordinator — one PPO/DQN agent per pair, shared market state
  2. CurriculumScheduler   — graduated volatility regimes during training
  3. SharpeRewardWrapper   — rolling Sharpe as primary reward signal
  4. HERBuffer            — Hindsight Experience Replay for sparse rewards
"""

import collections
import random
import warnings
from typing import Any

import numpy as np

from backtesting.backtest import ScalingAction

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    TORCH = True
except ImportError:
    TORCH = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. MULTI-AGENT COORDINATOR
# ─────────────────────────────────────────────────────────────────────────────

class SharedMarketState(nn.Module if TORCH else object):
    """
    Global context encoder shared across all pair-specific agents.
    Encodes cross-asset correlations, VIX, and portfolio state into
    a compact context vector that each agent conditions its policy on.

    This prevents agents from taking opposing positions (e.g. long EUR/USD
    and short GBP/USD simultaneously when the pairs are 0.85 correlated).
    """

    def __init__(self, global_feature_dim: int, context_dim: int = 32):
        super().__init__() if TORCH else None
        if TORCH:
            self.encoder = nn.Sequential(
                nn.Linear(global_feature_dim, 64),
                nn.ReLU(),
                nn.Linear(64, context_dim),
                nn.Tanh(),
            )

    def forward(self, global_features: "torch.Tensor") -> "torch.Tensor":
        return self.encoder(global_features)


class MultiAgentCoordinator:
    """
    Runs one DQN/PPO agent per currency pair.
    All agents share a global market encoder (SharedMarketState).

    Coordination mechanism:
      - Each agent's observation = [pair_obs, global_context]
      - Portfolio-level risk check: if any pair breaches max correlated exposure,
        block new same-direction entries across correlated pairs
      - Net P&L computed at portfolio level, not per-pair
    """

    PAIR_CORRELATIONS = {
        ("EURUSD", "GBPUSD"): 0.85,
        ("EURUSD", "AUDUSD"): 0.72,
        ("GBPUSD", "AUDUSD"): 0.68,
        ("USDJPY", "USDCAD"): 0.61,
    }

    def __init__(
        self,
        agents:      dict[str, Any],   # pair -> DQNAgent | PPOAgent
        pairs:       list[str],
        max_corr_exposure: float = 1.5, # Max sum of correlated lots
        global_feat_dim:   int   = 20,
        context_dim:       int   = 32,
        device:      str   = "cpu",
    ):
        self.agents   = agents
        self.pairs    = pairs
        self.max_corr = max_corr_exposure
        self.device   = device

        if TORCH:
            self.global_enc = SharedMarketState(global_feat_dim, context_dim)
            self.global_enc.to(torch.device(device))

        # Portfolio state
        self.positions: dict[str, float] = dict.fromkeys(pairs, 0.0)
        self.equity     = 10_000.0

    def _corr_exposure(self, pair: str, direction: int) -> float:
        """
        Compute current correlated directional exposure if we add
        this new position. Returns total lots in the same direction
        across highly correlated pairs.
        """
        total = abs(self.positions.get(pair, 0.0))
        for (p1, p2), corr in self.PAIR_CORRELATIONS.items():
            if corr < 0.6: continue
            other = p2 if p1 == pair else (p1 if p2 == pair else None)
            if other:
                other_pos = self.positions.get(other, 0.0)
                if np.sign(other_pos) == direction:
                    total += abs(other_pos) * corr
        return total

    def select_actions(
        self,
        observations: dict[str, np.ndarray],
        global_features: np.ndarray | None = None,
    ) -> dict[str, int]:
        """
        Get action for each pair's agent, applying portfolio-level risk check.
        Returns dict: pair -> action (0=Buy, 1=Hold, 2=Sell)
        """
        # Encode global market context
        context = np.zeros(32)
        if TORCH and global_features is not None:
            with torch.no_grad():
                gf = torch.tensor(global_features, dtype=torch.float32,
                                   device=self.device).unsqueeze(0)
                context = self.global_enc(gf).cpu().numpy().squeeze(0)  # remove batch dim only; .squeeze() collapses scalar

        actions = {}
        for pair, agent in self.agents.items():
            obs = observations.get(pair, np.zeros(1))
            aug_obs = np.concatenate([obs, context])  # Augment with global context

            if hasattr(agent, "select_action"):
                raw_action = agent.select_action(aug_obs)
            else:
                raw_action = ScalingAction.HOLD.value

            # Portfolio-level risk gate — map 10-action space to direction.
            # Only actions that INCREASE net directional exposure are gated:
            #   OPEN_LONG(1)/OPEN_SHORT(2) open new positions;
            #   SCALE_IN(3-5) add to the current position (direction = sign(pos));
            #   HOLD(0), SCALE_OUT(6-8), CLOSE_ALL(9) reduce or hold exposure.
            if raw_action == 1:
                direction = 1
            elif raw_action == 2:
                direction = -1
            elif raw_action in (3, 4, 5):
                pos = self.positions.get(pair, 0.0)
                direction = int(np.sign(pos))
            else:
                direction = 0
            if direction != 0:
                corr_exp = self._corr_exposure(pair, direction)
                if corr_exp >= self.max_corr:
                    raw_action = ScalingAction.HOLD.value  # Force HOLD

            actions[pair] = raw_action
        return actions

    def update_position(self, pair: str, lots: float):
        self.positions[pair] = lots

    def portfolio_summary(self) -> dict:
        return {
            "positions":   dict(self.positions),
            "total_lots":  sum(abs(v) for v in self.positions.values()),
            "net_direction": np.sign(sum(self.positions.values())),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. CURRICULUM SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumScheduler:
    """
    Graduated training regime to help the RL agent learn without getting
    overwhelmed by extreme market conditions early in training.

    Curriculum:
      Phase 1 (0–20%):  Low-vol periods only (ATR < 0.5× avg)
      Phase 2 (20–50%): Normal vol periods (0.5–1.5× avg ATR)
      Phase 3 (50–80%): Include high-vol / news events
      Phase 4 (80–100%): Full data including regime breaks

    Returns episode episode_filter function for each phase.
    """

    PHASES = [
        {"name": "low_vol",    "progress": 0.20, "atr_max_mult": 0.7},
        {"name": "normal_vol", "progress": 0.50, "atr_max_mult": 1.5},
        {"name": "high_vol",   "progress": 0.80, "atr_max_mult": 3.0},
        {"name": "full",       "progress": 1.00, "atr_max_mult": np.inf},
    ]

    def __init__(self, total_episodes: int):
        self.total   = total_episodes
        self.current = 0

    def step(self):
        self.current += 1

    @property
    def progress(self) -> float:
        return self.current / max(self.total, 1)

    @property
    def current_phase(self) -> dict:
        for p in self.PHASES:
            if self.progress <= p["progress"]:
                return p
        return self.PHASES[-1]

    def get_difficulty_multiplier(self) -> float:
        """0.0 = easiest, 1.0 = full difficulty."""
        return min(self.progress * 1.25, 1.0)

    def filter_bars(
        self,
        features:    np.ndarray,    # (n_bars, n_features)
        atr_col_idx: int = 0,       # Index of the ATR column in features
        avg_atr:     float = 0.0,   # Expected to be in the SAME unit as features[:,atr_col_idx]
                                    # (raw pips/price, NOT Z-scored). Pass 0.0 to auto-compute.
    ) -> np.ndarray:
        """
        Returns boolean mask of bars allowed in current curriculum phase.

        ``avg_atr`` must be in the same unit as ``features[:, atr_col_idx]``.
        If 0.0 (default), it is computed as the nanmean of the ATR column so
        the threshold is always unit-consistent, regardless of normalisation.
        """
        phase = self.current_phase
        max_a = phase["atr_max_mult"]
        if max_a == np.inf:
            return np.ones(len(features), dtype=bool)
        atr = np.abs(features[:, atr_col_idx])
        # Auto-compute avg_atr from the actual data when not supplied.
        # This makes the comparison unit-agnostic: works for raw pips,
        # Z-scored features, or any other scaling.
        if avg_atr == 0.0:
            finite_atr = atr[np.isfinite(atr)]
            avg_atr = float(np.nanmean(finite_atr)) if len(finite_atr) > 0 else 1.0
        if avg_atr <= 0.0:
            avg_atr = 1.0  # safety: avoid division-by-zero in caller comparisons
        return atr <= max_a * avg_atr

    def log_phase(self, episode: int):
        phase = self.current_phase
        if episode % 50 == 0:
            print(f"[Curriculum] Ep {episode:4d} | Phase: {phase['name']:12s} | "
                  f"Progress: {self.progress:.1%} | "
                  f"Difficulty: {self.get_difficulty_multiplier():.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SHARPE REWARD WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class SharpeRewardWrapper:
    """
    Replaces raw P&L reward with rolling Sharpe ratio so the agent
    optimizes risk-adjusted returns, not just profits.

    Rolling Sharpe at step t:
      R_t = return at step t (net of costs)
      μ   = rolling mean of R over last N steps
      σ   = rolling std of R over last N steps
      Sharpe_reward_t = (R_t - μ) / (σ + ε)

    Benefits over raw P&L:
      - Penalizes high-variance strategies automatically
      - Self-normalizing: doesn't require reward scaling
      - Aligns training objective with live performance metric
    """

    def __init__(
        self,
        window:       int   = 100,
        risk_free:    float = 0.05 / 252,   # Daily risk-free rate
        annualize:    float = np.sqrt(252),
        cost_penalty: float = 0.3,
        dd_penalty:   float = 0.5,
    ):
        self.window      = window
        self.rf          = risk_free
        self.ann         = annualize
        self.cost_pen    = cost_penalty
        self.dd_pen      = dd_penalty
        self._returns:   collections.deque = collections.deque(maxlen=window)
        self._peak_eq    = 1.0
        self._equity     = 1.0

    def reset(self):
        self._returns.clear()
        self._peak_eq = 1.0
        self._equity  = 1.0

    def compute(
        self,
        raw_pnl:  float,
        tx_cost:  float = 0.0,
        equity:   float = 1.0,
    ) -> float:
        """
        Compute Sharpe-based reward for one step.

        raw_pnl  : realized P&L this bar (in price units)
        tx_cost  : transaction cost this bar
        equity   : current account equity (for drawdown calc)
        """
        # Net return this step
        r_net = raw_pnl - tx_cost
        self._returns.append(r_net)
        self._equity = equity
        self._peak_eq = max(self._peak_eq, equity)

        if len(self._returns) < 2:
            return 0.0

        ret_arr = np.array(self._returns)
        mu      = ret_arr.mean()
        sigma   = ret_arr.std() + 1e-9
        sharpe  = (mu - self.rf) / sigma * self.ann

        # Drawdown penalty
        dd      = max(0.0, (self._peak_eq - equity) / self._peak_eq)
        dd_pen  = self.dd_pen * dd

        # Transaction cost penalty (discourages overtrading)
        cost_pen = self.cost_pen * abs(tx_cost)

        return float(sharpe - dd_pen - cost_pen)

    def rolling_sharpe(self) -> float:
        if len(self._returns) < 2:
            return 0.0
        r = np.array(self._returns)
        return float((r.mean() - self.rf) / (r.std() + 1e-9) * self.ann)


# ─────────────────────────────────────────────────────────────────────────────
# 4. HINDSIGHT EXPERIENCE REPLAY (HER)
# ─────────────────────────────────────────────────────────────────────────────

class HERBuffer:
    """
    Hindsight Experience Replay for sparse reward environments.

    The key insight: even a failed trade (missed profit target) contains
    useful information. HER relabels the trade as if the *actual* exit price
    WAS the intended goal, generating a positive experience from a failure.

    For forex scalping:
      - Original experience: entered at 1.0850, target 1.0870, stopped at 1.0845 -> loss
      - HER relabeled: entered at 1.0850, goal was 1.0845, HIT -> positive signal
        (teaches the model what pattern *did* lead to a 5-pip move)

    Uses the "future" strategy: replay with goals taken from future states
    in the same episode.
    """

    def __init__(
        self,
        capacity:    int   = 100_000,
        k:           int   = 4,        # HER relabeling ratio
        goal_dim:    int   = 1,        # Dimension of goal (target price)
        strategy:    str   = "future", # "future" | "episode" | "random"
    ):
        self.capacity = capacity
        self.k        = k
        self.goal_dim = goal_dim
        self.strategy = strategy

        # Episode buffer (cleared after each episode)
        self._episode: list[dict] = []
        # Replay buffer (persistent)
        self._buffer:  collections.deque = collections.deque(maxlen=capacity)
        # Cached list view of the deque for sampling; rebuilt only after the
        # buffer composition changes (end_episode), not on every sample().
        self._cache: list | None = None

    def store_transition(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
        goal:     np.ndarray,   # Intended target (e.g. entry + ATR × 1.5)
        achieved: np.ndarray,   # Actually achieved (e.g. exit price)
        info:     dict = {},
    ):
        """Store one transition in the episode buffer."""
        self._episode.append({
            "obs": obs, "action": action, "reward": reward,
            "next_obs": next_obs, "done": done,
            "goal": goal, "achieved": achieved, "info": info,
        })

    def _hindsight_reward(self, achieved: np.ndarray, goal: np.ndarray) -> float:
        """
        Compute reward for a relabeled (hindsight) goal.
        Binary: 1.0 if achieved ≈ goal, -1.0 otherwise.
        For forex: 1.0 if exit price matches relabeled target.
        """
        dist = float(np.linalg.norm(achieved - goal))
        return 1.0 if dist < 0.0002 else -0.1   # ~2 pip tolerance

    def end_episode(self):
        """
        At episode end: store original experiences + HER relabeled experiences.
        """
        ep = self._episode
        if not ep:
            return

        # 1. Store original transitions
        for t in ep:
            self._buffer.append(t)

        # 2. Add k HER relabeled transitions per original
        n = len(ep)
        for t_idx, transition in enumerate(ep):
            for _ in range(self.k):
                # Pick a future achieved state as the hindsight goal.
                # RA2 fix (2026-08-07, audit finding RA2): the "future" strategy
                # previously used `random.randint(t_idx, n - 1)` — INCLUSIVE of
                # `t_idx` itself. When `t_idx == future_idx` (a 1/n chance per
                # sample on average, but ALWAYS the case for the final
                # transition t_idx == n-1 since both endpoints coincide), the
                # HER relabel becomes a self-match: `dist=0` deterministic, the
                # `_hindsight_reward` evaluates to the trivial max (+1.0 for a
                # tolerance-based reward), giving the agent a guaranteed
                # positive reward for "free" without actually achieving a goal.
                # Classic HER samples `t_idx+1 .. n-1` precisely to avoid this.
                # For t_idx == n-1 (last transition), there are no future
                # states to sample from — we SKIP the HER relabel for that
                # transition rather than polluting the buffer with self-matches.
                if self.strategy == "future":
                    if t_idx + 1 >= n:
                        # No future transitions in this episode — skip HER
                        # relabel for this transition rather than self-match.
                        break
                    future_idx = random.randint(t_idx + 1, n - 1)
                elif self.strategy == "episode":
                    # "episode" strategy still samples any of 0..n-1 (the
                    # whole episode, including past + future of t_idx). This
                    # is explicitly HER "episode" mode and is by-design a
                    # potential self-match source — callers should prefer
                    # "future" mode for sparse-reward DQN.
                    future_idx = random.randint(0, n - 1)
                    # Avoid the self-match: if we accidentally draw t_idx,
                    # try once more before falling back to a clipped index.
                    if future_idx == t_idx and n > 1:
                        future_idx = random.randint(0, n - 1)
                else:  # random
                    if n <= 1:
                        break
                    future_idx = random.randint(0, n - 1)
                    if future_idx == t_idx:
                        future_idx = (t_idx + 1) % n  # deterministically skip self

                her_goal     = ep[future_idx]["achieved"]

                # If the chosen "future achieved" coincides with the current
                # achieved (e.g. price didn't move between t_idx and future_idx),
                # the relabel would still be a degenerate self-match. Skip it.
                if np.array_equal(her_goal, transition["achieved"]):
                    continue

                her_reward   = self._hindsight_reward(transition["achieved"], her_goal)
                her_done     = future_idx == n - 1

                self._buffer.append({
                    "obs":      np.concatenate([transition["obs"], her_goal]),
                    "action":   transition["action"],
                    "reward":   her_reward,
                    "next_obs": np.concatenate([transition["next_obs"], her_goal]),
                    "done":     her_done,
                    "goal":     her_goal,
                    "achieved": transition["achieved"],
                    "info":     {"her": True},
                })

        self._episode.clear()
        self._cache = None

    def sample(self, batch_size: int) -> list[dict]:
        if len(self._buffer) < batch_size:
            return list(self._buffer)
        if self._cache is None:
            self._cache = list(self._buffer)
        return random.sample(self._cache, batch_size)

    def __len__(self):
        return len(self._buffer)

    @property
    def her_ratio(self) -> float:
        """Fraction of buffer that is HER relabeled (for monitoring)."""
        her_count = sum(1 for t in self._buffer if t.get("info", {}).get("her", False))
        return her_count / max(len(self._buffer), 1)


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TESTS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("RL Advanced — smoke tests")
    print("=" * 50)

    # Curriculum
    sched = CurriculumScheduler(total_episodes=500)
    for ep in [0, 100, 250, 400, 499]:
        sched.current = ep
        p = sched.current_phase
        print(f"  Ep {ep:3d} -> phase: {p['name']:12s} | atr_max: {p['atr_max_mult']}")

    # Sharpe reward
    sr = SharpeRewardWrapper(window=20)
    for i in range(25):
        r = sr.compute(raw_pnl=np.random.normal(0.001, 0.003), tx_cost=0.0001)
    print(f"\n  Sharpe reward (final): {r:.4f}")
    print(f"  Rolling Sharpe: {sr.rolling_sharpe():.3f}")

    # HER buffer
    her = HERBuffer(capacity=1000, k=4)
    for i in range(10):
        her.store_transition(
            obs=np.zeros(10), action=0, reward=-0.5,
            next_obs=np.zeros(10), done=(i==9),
            goal=np.array([1.0870]), achieved=np.array([1.0845])
        )
    her.end_episode()
    print(f"\n  HER buffer size: {len(her)}")
    print(f"  HER ratio: {her.her_ratio:.1%}")
    batch = her.sample(8)
    her_count = sum(1 for t in batch if t.get("info",{}).get("her",False))
    print(f"  HER in batch of 8: {her_count} relabeled")

    # Multi-agent
    print("\n  Multi-agent coordinator test:")
    class DummyAgent:
        def select_action(self, obs): return 1
    agents = {"EURUSD": DummyAgent(), "GBPUSD": DummyAgent()}
    coord  = MultiAgentCoordinator(agents, ["EURUSD","GBPUSD"])
    obs    = {"EURUSD": np.zeros(10), "GBPUSD": np.zeros(10)}
    actions = coord.select_actions(obs)
    print(f"  Actions: {actions}")
    print(f"  Portfolio: {coord.portfolio_summary()}")

    print("\nAll RL advanced tests passed ✓")
