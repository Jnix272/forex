"""
models/rl_advanced.py
======================
RL enhancements:
  1. MultiAgentCoordinator - one PPO/DQN agent per pair, shared market state
  2. CurriculumScheduler   - graduated volatility regimes during training
  3. SharpeRewardWrapper   - rolling Sharpe as primary reward signal
  4. HERBuffer            - Hindsight Experience Replay for sparse rewards
"""

import collections
import random
import warnings
from typing import Any

import numpy as np

from backtesting.backtest import ScalingAction

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

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
        agents: dict[str, Any],  # pair -> DQNAgent | PPOAgent
        pairs: list[str],
        max_corr_exposure: float = 1.5,  # Max sum of correlated lots
        global_feat_dim: int = 20,
        context_dim: int = 32,
        device: str = "cpu",
    ):
        self.agents = agents
        self.pairs = pairs
        self.max_corr = max_corr_exposure
        self.context_dim = int(context_dim)
        self.device = device

        if TORCH:
            self.global_enc = SharedMarketState(global_feat_dim, context_dim)
            self.global_enc.to(torch.device(device))

        # Portfolio state
        self.positions: dict[str, float] = dict.fromkeys(pairs, 0.0)
        self.equity = 10_000.0

    def _corr_exposure(self, pair: str, direction: int, new_lots: float = 1.0) -> float:
        """
        Compute current correlated directional exposure if we add
        this new position. Returns total lots in the same direction
        across highly correlated pairs.
        """
        curr = self.positions.get(pair, 0.0)
        projected = curr + direction * new_lots
        total = abs(projected) if np.sign(projected) == direction else 0.0
        for (p1, p2), corr in self.PAIR_CORRELATIONS.items():
            if corr < 0.6:
                continue
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
        return_info: bool = False,
    ) -> dict[str, int] | tuple[dict[str, int], dict[str, Any]]:
        """
        Get action for each pair's agent, applying portfolio-level risk check.
        Returns dict: pair -> action (0=Buy, 1=Hold, 2=Sell).
        If return_info=True, also returns consensus & diagnostic metrics per pair.
        """
        # Encode global market context
        context = np.zeros(self.context_dim, dtype=np.float32)
        if TORCH and global_features is not None:
            with torch.no_grad():
                gf = torch.tensor(global_features, dtype=torch.float32, device=self.device).unsqueeze(0)
                context = (
                    self.global_enc(gf).cpu().numpy().squeeze(0)
                )  # remove batch dim only; .squeeze() collapses scalar

        actions = {}
        consensus_reports = {}

        for pair, agent in self.agents.items():
            obs = observations.get(pair, np.zeros(1))
            aug_obs = np.concatenate([np.atleast_1d(obs), np.atleast_1d(context)])  # Augment with global context

            if hasattr(agent, "select_action"):
                raw = agent.select_action(aug_obs)
                raw_action = raw[0] if isinstance(raw, tuple) else raw
            else:
                raw_action = ScalingAction.HOLD.value

            # If agent has consensus information (e.g. RLEnsemble)
            if hasattr(agent, "last_consensus_info") and agent.last_consensus_info is not None:
                consensus_reports[pair] = agent.last_consensus_info
            elif hasattr(agent, "get_consensus_info"):
                consensus_reports[pair] = agent.get_consensus_info(aug_obs)

            # Portfolio-level risk gate - map 10-action space to direction.
            # Only actions that INCREASE net directional exposure are gated:
            #   OPEN_LONG(1)/OPEN_SHORT(2) open new positions;
            #   SCALE_IN(3-5) add to the current position (direction = sign(pos));
            #   HOLD(0), SCALE_OUT(6-8), CLOSE_ALL(9) reduce or hold exposure.
            if raw_action == 1:
                direction = 1
                new_lots = 1.0
            elif raw_action == 2:
                direction = -1
                new_lots = 1.0
            elif raw_action in (3, 4, 5):
                pos = self.positions.get(pair, 0.0)
                direction = int(np.sign(pos)) if pos != 0 else 0
                new_lots = {3: 0.25, 4: 0.50, 5: 1.0}[raw_action]
            else:
                direction = 0
                new_lots = 0.0
            if direction != 0:
                corr_exp = self._corr_exposure(pair, direction, new_lots=new_lots)
                if corr_exp >= self.max_corr:
                    raw_action = ScalingAction.HOLD.value  # Force HOLD

            actions[pair] = int(raw_action)

        if return_info:
            return actions, consensus_reports
        return actions

    def set_agent(self, pair: str, agent: Any):
        """Assign or replace an agent or RLEnsemble for a specific currency pair."""
        self.agents[pair] = agent
        if pair not in self.pairs:
            self.pairs.append(pair)
            self.positions[pair] = 0.0

    def get_agent(self, pair: str) -> Any:
        """Retrieve the agent or RLEnsemble for a pair."""
        return self.agents.get(pair)

    def update_position(self, pair: str, lots: float):
        self.positions[pair] = lots

    def portfolio_summary(self) -> dict:
        return {
            "positions": dict(self.positions),
            "total_lots": sum(abs(v) for v in self.positions.values()),
            "net_direction": np.sign(sum(self.positions.values())),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 1b. RL POLICY ENSEMBLE (Consensus & Multi-Agent Fusion)
# ─────────────────────────────────────────────────────────────────────────────


class RLEnsemble:
    """
    Ensemble of Reinforcement Learning policies with multi-policy consensus.

    Supports combining:
      - Multiple PPO agents (trained with varied seeds, volatility regimes, or hyperparameters)
      - Multiple DQN agents
      - Mixed heterogeneous ensembles (PPO + DQN)

    Consensus modes:
      1. 'soft_vote':
         Averages predicted action probability distributions across all agents.
         Argmax (or sampling) on the ensemble mean probability distribution.
      2. 'majority':
         Plurality vote across discrete actions chosen by individual agents.
         Safety fallback: If there is a direct conflict between BUY (OPEN_LONG)
         and SELL (OPEN_SHORT), it immediately falls back to HOLD (0).
      3. 'conservative':
         Ultra-safe capital preservation mode:
         - If ANY agent votes HOLD (0), the ensemble action is forced to HOLD (0).
         - If opposite directional conflict (BUY vs SELL) occurs, forced to HOLD (0).
         - If all agents agree on direction but choose different scale sizes, downsizes
           to the minimal/safest position size (e.g. SCALE_IN_25 instead of SCALE_IN_100).
    """

    CONSENSUS_MODES = ("soft_vote", "majority", "conservative")

    def __init__(
        self,
        agents: list[Any] | dict[str, Any],
        agent_types: list[str] | None = None,
        consensus_mode: str = "soft_vote",
        weights: list[float] | None = None,
        obs_size: int | None = None,
        n_actions: int = 10,
        device: str = "cpu",
        temperature: float = 1.0,
    ):
        if isinstance(agents, dict):
            self.agent_names = list(agents.keys())
            self.agents = list(agents.values())
        else:
            self.agents = list(agents)
            self.agent_names = [f"agent_{i}" for i in range(len(self.agents))]

        if len(self.agents) == 0:
            raise ValueError("RLEnsemble requires at least one sub-agent.")

        if consensus_mode not in self.CONSENSUS_MODES:
            raise ValueError(f"Unknown consensus_mode '{consensus_mode}'. Must be one of {self.CONSENSUS_MODES}")

        self.consensus_mode = consensus_mode
        self.device = str(device)
        self.temperature = max(float(temperature), 1e-4)
        self.n_actions = int(n_actions)

        # Infer or validate agent types
        if agent_types is not None:
            if len(agent_types) != len(self.agents):
                raise ValueError("Length of agent_types must match number of agents.")
            self.agent_types = [str(t).lower() for t in agent_types]
        else:
            self.agent_types = []
            for a in self.agents:
                cname = type(a).__name__.lower()
                if "ppo" in cname:
                    self.agent_types.append("ppo")
                elif "dqn" in cname:
                    self.agent_types.append("dqn")
                else:
                    self.agent_types.append("custom")

        # Weights
        if weights is not None:
            if len(weights) != len(self.agents):
                raise ValueError("Length of weights must match number of agents.")
            w_arr = np.array(weights, dtype=np.float32)
            w_sum = float(w_arr.sum())
            if w_sum <= 0:
                raise ValueError("Weights sum must be positive.")
            self.weights = (w_arr / w_sum).tolist()
        else:
            self.weights = [1.0 / len(self.agents)] * len(self.agents)

        # Observation size
        if obs_size is not None:
            self.obs_size = int(obs_size)
        else:
            self.obs_size = self._infer_obs_size()

        self.last_consensus_info: dict[str, Any] | None = None

    def _infer_obs_size(self) -> int:
        for a in self.agents:
            if hasattr(a, "obs_size") and a.obs_size is not None:
                return int(a.obs_size)
            if hasattr(a, "net"):
                if hasattr(a.net, "backbone") and len(a.net.backbone) > 0 and hasattr(a.net.backbone[0], "in_features"):
                    return int(a.net.backbone[0].in_features)
                if hasattr(a.net, "lstm") and hasattr(a.net.lstm, "input_size"):
                    return int(a.net.lstm.input_size)
            if hasattr(a, "policy_net"):
                if hasattr(a.policy_net, "net") and len(a.policy_net.net) > 0 and hasattr(a.policy_net.net[0], "in_features"):
                    return int(a.policy_net.net[0].in_features)
        return 0

    def get_agent_probabilities(
        self, agent: Any, obs: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, float]:
        """Extract action probability distribution and state-value estimate from an individual agent."""
        if not TORCH:
            action = agent.select_action(obs)
            if isinstance(action, tuple):
                action = action[0]
            probs = np.zeros(self.n_actions, dtype=np.float32)
            probs[int(action)] = 1.0
            return probs, 0.0

        # Custom agent with explicit probability extraction method
        if hasattr(agent, "get_action_probabilities"):
            raw = agent.get_action_probabilities(obs, mask=mask)
            if isinstance(raw, tuple):
                return np.asarray(raw[0], dtype=np.float32), float(raw[1])
            return np.asarray(raw, dtype=np.float32), 0.0

        dev = getattr(agent, "device", torch.device(self.device))

        # PPOAgent / ActorCritic
        if hasattr(agent, "net") and hasattr(agent.net, "actor"):
            if getattr(agent, "use_lstm", False) and hasattr(agent, "_preview_obs_seq"):
                x = agent._preview_obs_seq(obs)
                xt = torch.as_tensor(x, dtype=torch.float32, device=dev).unsqueeze(0)
            else:
                xt = torch.as_tensor(obs, dtype=torch.float32, device=dev)
                if xt.ndim == 1:
                    xt = xt.unsqueeze(0)

            with torch.no_grad():
                logits, value = agent.net(xt)
                if mask is not None:
                    m = torch.as_tensor(mask, dtype=torch.bool, device=dev)
                    if m.ndim == 1:
                        m = m.unsqueeze(0)
                    logits = logits.masked_fill(~m, -1e9)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                val = float(value.squeeze().cpu().item()) if value.numel() > 0 else 0.0
            return probs.astype(np.float32), val

        # DQNAgent / DQNetwork
        elif hasattr(agent, "policy_net"):
            xt = torch.as_tensor(obs, dtype=torch.float32, device=dev)
            if xt.ndim == 1:
                xt = xt.unsqueeze(0)

            with torch.no_grad():
                q = agent.policy_net(xt)
                if mask is not None:
                    m = torch.as_tensor(mask, dtype=torch.bool, device=dev)
                    if m.ndim == 1:
                        m = m.unsqueeze(0)
                    q = q.masked_fill(~m, -1e9)
                probs = F.softmax(q / self.temperature, dim=-1).squeeze(0).cpu().numpy()
                val = float(q.max(dim=-1)[0].squeeze().cpu().item())
            return probs.astype(np.float32), val

        # Generic / Fallback
        else:
            try:
                action = agent.select_action(obs, mask=mask)
            except TypeError:
                action = agent.select_action(obs)
            if isinstance(action, tuple):
                action = action[0]
            probs = np.zeros(self.n_actions, dtype=np.float32)
            probs[int(action)] = 1.0
            return probs, 0.0

    def get_consensus_info(
        self,
        obs: np.ndarray,
        mask: np.ndarray | None = None,
        greedy: bool = True,
        mode_override: str | None = None,
    ) -> dict[str, Any]:
        """Compute full multi-agent consensus metrics, action distributions, and final action."""
        mode = mode_override or self.consensus_mode
        n_agents = len(self.agents)

        probs_list: list[np.ndarray] = []
        values_list: list[float] = []
        actions_list: list[int] = []

        for agent in self.agents:
            p, v = self.get_agent_probabilities(agent, obs, mask=mask)
            probs_list.append(p)
            values_list.append(v)
            if greedy:
                actions_list.append(int(np.argmax(p)))
            else:
                p_sub = p.astype(np.float64)
                p_sub_sum = p_sub.sum()
                if p_sub_sum > 0:
                    p_sub /= p_sub_sum
                else:
                    p_sub = np.ones(self.n_actions, dtype=np.float64) / self.n_actions
                actions_list.append(int(np.random.choice(self.n_actions, p=p_sub)))

        # Weighted average probabilities
        probs_matrix = np.stack(probs_list, axis=0)  # (N, n_actions)
        weights_arr = np.array(self.weights, dtype=np.float32)[:, np.newaxis]  # (N, 1)
        mean_probs = np.sum(probs_matrix * weights_arr, axis=0)  # (n_actions,)
        p_sum = mean_probs.sum()
        if p_sum > 0:
            mean_probs /= p_sum
        else:
            if mask is not None and np.any(mask):
                mean_probs = np.zeros(self.n_actions, dtype=np.float32)
                mean_probs[mask] = 1.0 / np.sum(mask)
            else:
                mean_probs = np.ones(self.n_actions, dtype=np.float32) / self.n_actions

        mean_val = float(sum(w * v for w, v in zip(self.weights, values_list)))

        # BUY / SELL Conflict Detection
        # ScalingAction: HOLD=0, OPEN_LONG=1, OPEN_SHORT=2, SCALE_IN_25=3, 50=4, 100=5, SCALE_OUT_25=6, 50=7, 100=8, CLOSE_ALL=9
        has_long_entry = any(a == ScalingAction.OPEN_LONG.value for a in actions_list)
        has_short_entry = any(a == ScalingAction.OPEN_SHORT.value for a in actions_list)
        conflict_detected = bool(has_long_entry and has_short_entry)

        # Directional conflict
        long_actions = {1, 3, 4, 5}
        short_actions = {2}
        has_long = any(a in long_actions for a in actions_list)
        has_short = any(a in short_actions for a in actions_list)
        directional_conflict = bool(has_long and has_short)

        # ── Action Resolution according to Consensus Mode ──
        if mode == "soft_vote":
            if greedy:
                consensus_action = int(np.argmax(mean_probs))
            else:
                p_draw = mean_probs.astype(np.float64)
                p_draw_sum = p_draw.sum()
                if p_draw_sum > 0:
                    p_draw /= p_draw_sum
                else:
                    p_draw = np.ones(self.n_actions, dtype=np.float64) / self.n_actions
                consensus_action = int(np.random.choice(self.n_actions, p=p_draw))

        elif mode == "majority":
            if conflict_detected or directional_conflict:
                # Direct opposition: safety fall-back to HOLD
                consensus_action = ScalingAction.HOLD.value
            else:
                votes: dict[int, float] = collections.defaultdict(float)
                for a, w in zip(actions_list, self.weights):
                    votes[a] += w

                max_vote = max(votes.values())
                candidates = [a for a, v in votes.items() if abs(v - max_vote) < 1e-6]
                if len(candidates) == 1:
                    consensus_action = candidates[0]
                else:
                    # Tie-breaking: candidate with highest average probability
                    consensus_action = max(candidates, key=lambda c: mean_probs[c])

        elif mode == "conservative":
            # 1. Any agent voting HOLD forces HOLD
            if any(a == ScalingAction.HOLD.value for a in actions_list):
                consensus_action = ScalingAction.HOLD.value
            # 2. Opposite direction conflict forces HOLD
            elif conflict_detected or directional_conflict:
                consensus_action = ScalingAction.HOLD.value
            # 3. If all agents vote in same scale-in family, take the smallest / safest size
            elif all(a in (3, 4, 5) for a in actions_list):
                scale_in_rank = {3: 1, 4: 2, 5: 3}
                consensus_action = min(actions_list, key=lambda a: scale_in_rank.get(a, 99))
            elif all(a in (6, 7, 8) for a in actions_list):
                scale_out_rank = {6: 1, 7: 2, 8: 3}
                consensus_action = min(actions_list, key=lambda a: scale_out_rank.get(a, 99))
            # 4. If all agents unanimously agree on action
            elif len(set(actions_list)) == 1:
                consensus_action = actions_list[0]
            else:
                # Disagreement among active non-HOLD trades: conservative fall-back to HOLD
                consensus_action = ScalingAction.HOLD.value

        else:
            consensus_action = ScalingAction.HOLD.value

        # Mask sanity check: if consensus_action is invalid according to mask, pick best valid action
        if mask is not None and 0 <= consensus_action < len(mask) and not mask[consensus_action]:
            valid_actions = np.where(mask)[0]
            if len(valid_actions) > 0:
                consensus_action = int(valid_actions[np.argmax(mean_probs[valid_actions])])

        # ── Agreement / Disagreement / Policy Uncertainty Metrics ──
        if n_agents > 1:
            total_pairs = n_agents * (n_agents - 1) // 2
            matching_pairs = sum(
                1 for i in range(n_agents) for j in range(i + 1, n_agents) if actions_list[i] == actions_list[j]
            )
            action_agreement = matching_pairs / total_pairs
        else:
            action_agreement = 1.0

        def _get_dir(a: int) -> int:
            if a in (1, 3, 4, 5):
                return 1
            if a in (2,):
                return -1
            return 0

        if n_agents > 1:
            matching_dir_pairs = sum(
                1
                for i in range(n_agents)
                for j in range(i + 1, n_agents)
                if _get_dir(actions_list[i]) == _get_dir(actions_list[j])
            )
            direction_agreement = matching_dir_pairs / total_pairs
        else:
            direction_agreement = 1.0

        if n_agents > 1:
            tvd_sum = 0.0
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    tvd_sum += 0.5 * float(np.sum(np.abs(probs_list[i] - probs_list[j])))
            mean_tvd = tvd_sum / total_pairs
            distributional_agreement = float(np.clip(1.0 - mean_tvd, 0.0, 1.0))
        else:
            distributional_agreement = 1.0

        entropy = -float(np.sum(mean_probs * np.log(mean_probs + 1e-12)))
        max_entropy = float(np.log(self.n_actions))
        policy_uncertainty = float(np.clip(entropy / max_entropy, 0.0, 1.0))

        agreement_score = float(action_agreement)
        disagreement_score = float(1.0 - agreement_score)
        mean_log_prob = float(np.log(max(mean_probs[consensus_action], 1e-12)))

        info = {
            "consensus_action": int(consensus_action),
            "consensus_mode": mode,
            "greedy": greedy,
            "agreement_score": agreement_score,
            "disagreement_score": disagreement_score,
            "action_agreement": action_agreement,
            "direction_agreement": direction_agreement,
            "distributional_agreement": distributional_agreement,
            "policy_uncertainty": policy_uncertainty,
            "conflict_detected": conflict_detected,
            "directional_conflict": directional_conflict,
            "individual_actions": actions_list,
            "individual_values": values_list,
            "action_counts": dict(collections.Counter(actions_list)),
            "mean_probabilities": mean_probs,
            "mean_value": mean_val,
            "mean_log_prob": mean_log_prob,
        }
        return info

    def select_action(
        self,
        obs: np.ndarray,
        mask: np.ndarray | None = None,
        greedy: bool = True,
        return_tuple: bool = True,
    ) -> int | tuple[int, float, float]:
        """Select an action using the ensemble consensus policy."""
        info = self.get_consensus_info(obs, mask=mask, greedy=greedy)
        self.last_consensus_info = info
        if return_tuple:
            return info["consensus_action"], info["mean_log_prob"], info["mean_value"]
        return info["consensus_action"]

    def save_checkpoint(self, path: str | Path, meta: dict | None = None) -> Path:
        """Save full ensemble checkpoint with all sub-agent weights and meta configuration."""
        if not TORCH:
            raise RuntimeError("PyTorch is required to save RLEnsemble checkpoint.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        agent_states = []
        agent_configs = []

        for agent, atype in zip(self.agents, self.agent_types):
            cfg = {
                "type": atype,
                "obs_size": getattr(agent, "obs_size", self.obs_size),
                "n_actions": getattr(agent, "n_actions", self.n_actions),
                "hidden": getattr(agent, "hidden", 256),
                "use_lstm": getattr(agent, "use_lstm", False),
            }
            if atype == "ppo":
                state = agent.net.state_dict() if hasattr(agent, "net") else {}
                cfg.update({
                    "lr": getattr(agent, "lr", 3e-4),
                    "gamma": getattr(agent, "gamma", 0.99),
                    "lam": getattr(agent, "lam", 0.95),
                    "clip": getattr(agent, "clip", 0.2),
                    "entropy_coef": getattr(agent, "ent_c", 0.01),
                    "lstm_hidden": getattr(agent, "lstm_hidden", 128),
                    "hist_len": getattr(agent, "hist_len", 32),
                })
            elif atype == "dqn":
                state = agent.policy_net.state_dict() if hasattr(agent, "policy_net") else {}
                cfg.update({
                    "lr": getattr(agent, "lr", 1e-4),
                    "gamma": getattr(agent, "gamma", 0.99),
                    "batch": getattr(agent, "batch", 64),
                    "double": getattr(agent, "double", True),
                    "eps_start": getattr(agent, "eps", 1.0),
                    "eps_end": getattr(agent, "eps_end", 0.01),
                    "eps_decay": getattr(agent, "eps_decay", 0.995),
                })
            else:
                state = {}

            agent_states.append(state)
            agent_configs.append(cfg)

        payload = {
            "version": "1.0",
            "model_type": "RLEnsemble",
            "consensus_mode": self.consensus_mode,
            "weights": self.weights,
            "obs_size": self.obs_size,
            "n_actions": self.n_actions,
            "agent_types": self.agent_types,
            "agent_names": self.agent_names,
            "agent_configs": agent_configs,
            "agent_states": agent_states,
            "meta": meta or {},
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str = "cpu") -> "RLEnsemble":
        """Load RLEnsemble from checkpoint path."""
        if not TORCH:
            raise RuntimeError("PyTorch is required to load RLEnsemble checkpoint.")
        from models.rl_agents import DQNAgent, PPOAgent

        path = Path(path)
        payload = torch.load(path, map_location=device, weights_only=False)

        consensus_mode = payload.get("consensus_mode", "soft_vote")
        weights = payload.get("weights")
        obs_size = payload.get("obs_size")
        n_actions = payload.get("n_actions", 10)
        agent_types = payload.get("agent_types", [])
        agent_configs = payload.get("agent_configs", [])
        agent_states = payload.get("agent_states", [])

        agents = []
        for i, atype in enumerate(agent_types):
            cfg = agent_configs[i] if i < len(agent_configs) else {}
            state = agent_states[i] if i < len(agent_states) else {}
            sub_obs = cfg.get("obs_size", obs_size)
            sub_actions = cfg.get("n_actions", n_actions)

            if atype == "ppo":
                ag = PPOAgent(
                    obs_size=sub_obs,
                    n_actions=sub_actions,
                    hidden=cfg.get("hidden", 256),
                    lr=cfg.get("lr", 3e-4),
                    gamma=cfg.get("gamma", 0.99),
                    lam=cfg.get("lam", 0.95),
                    clip=cfg.get("clip", 0.2),
                    entropy_coef=cfg.get("entropy_coef", 0.01),
                    device=device,
                    use_lstm=cfg.get("use_lstm", False),
                    lstm_hidden=cfg.get("lstm_hidden", 128),
                    hist_len=cfg.get("hist_len", 32),
                )
                if state:
                    ag.net.load_state_dict(state)
            elif atype == "dqn":
                ag = DQNAgent(
                    obs_size=sub_obs,
                    n_actions=sub_actions,
                    hidden=cfg.get("hidden", 256),
                    lr=cfg.get("lr", 1e-4),
                    gamma=cfg.get("gamma", 0.99),
                    batch=cfg.get("batch", 64),
                    double_dqn=cfg.get("double", True),
                    eps_start=cfg.get("eps_start", 1.0),
                    eps_end=cfg.get("eps_end", 0.01),
                    eps_decay=cfg.get("eps_decay", 0.995),
                    device=device,
                    use_lstm=cfg.get("use_lstm", False),
                )
                if state:
                    ag.policy_net.load_state_dict(state)
                    ag.target_net.load_state_dict(state)
            else:
                raise ValueError(f"Unsupported agent type '{atype}' in checkpoint.")

            agents.append(ag)

        return cls(
            agents=agents,
            agent_types=agent_types,
            consensus_mode=consensus_mode,
            weights=weights,
            obs_size=obs_size,
            n_actions=n_actions,
            device=device,
        )

    def evaluate(
        self,
        env: Any,
        n_episodes: int = 5,
        greedy: bool = True,
    ) -> tuple[list[float], dict[str, Any], dict[str, float]]:
        """Run evaluation episodes on env using ensemble consensus policy.

        Returns (returns, agg_summary, ensemble_diagnostics).
        """
        returns = []
        episode_summaries = []
        all_pnls = []
        agreement_scores = []
        disagreement_scores = []
        uncertainties = []
        conflict_steps = 0
        total_steps = 0

        for _ in range(int(n_episodes)):
            obs = env.reset()
            while not env.done:
                mask = env.action_mask() if hasattr(env, "action_mask") else None
                info = self.get_consensus_info(obs, mask=mask, greedy=greedy)
                action = info["consensus_action"]
                obs, _, _, _ = env.step(action)

                agreement_scores.append(info["agreement_score"])
                disagreement_scores.append(info["disagreement_score"])
                uncertainties.append(info["policy_uncertainty"])
                if info["conflict_detected"]:
                    conflict_steps += 1
                total_steps += 1

            ep_summ = env.summary()
            episode_summaries.append(ep_summ)
            returns.append(float(ep_summ["total_return_pct"]))
            if hasattr(env, "episode_pnl"):
                all_pnls.extend(env.episode_pnl)

        all_pnls_arr = np.array(all_pnls)
        if len(all_pnls_arr) > 1 and all_pnls_arr.std(ddof=1) > 1e-12:
            agg_sharpe = float((all_pnls_arr.mean() / all_pnls_arr.std(ddof=1)) * np.sqrt(getattr(env, "bars_per_year", 75000)))
        else:
            agg_sharpe = float(np.mean([s["sharpe"] for s in episode_summaries])) if episode_summaries else 0.0

        agg_summary = {
            "total_return_pct": float(np.mean(returns)) if returns else 0.0,
            "sharpe": agg_sharpe,
            "n_trades": int(sum(s["n_trades"] for s in episode_summaries)),
            "total_costs": float(sum(s["total_costs"] for s in episode_summaries)),
            "max_dd_pct": float(max((s["max_dd_pct"] for s in episode_summaries), default=0.0)),
            "episode_summaries": episode_summaries,
        }

        diagnostics = {
            "mean_agreement_score": float(np.mean(agreement_scores)) if agreement_scores else 1.0,
            "mean_disagreement_score": float(np.mean(disagreement_scores)) if disagreement_scores else 0.0,
            "mean_policy_uncertainty": float(np.mean(uncertainties)) if uncertainties else 0.0,
            "conflict_rate": float(conflict_steps / max(total_steps, 1)),
            "total_steps": total_steps,
        }
        return returns, agg_summary, diagnostics


# Alias
PolicyEnsemble = RLEnsemble


# ─────────────────────────────────────────────────────────────────────────────
# 2. CURRICULUM SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────


class CurriculumScheduler:
    """
    Graduated training regime to help the RL agent learn without getting
    overwhelmed by extreme market conditions early in training.

    Curriculum:
      Phase 1 (0–20%):  Low-vol periods only (ATR < 0.5x avg)
      Phase 2 (20–50%): Normal vol periods (0.5–1.5x avg ATR)
      Phase 3 (50–80%): Include high-vol / news events
      Phase 4 (80–100%): Full data including regime breaks

    Returns episode episode_filter function for each phase.
    """  # noqa: RUF002

    PHASES = [
        {"name": "low_vol", "progress": 0.20, "atr_max_mult": 0.7},
        {"name": "normal_vol", "progress": 0.50, "atr_max_mult": 1.5},
        {"name": "high_vol", "progress": 0.80, "atr_max_mult": 3.0},
        {"name": "full", "progress": 1.00, "atr_max_mult": np.inf},
    ]

    def __init__(self, total_episodes: int):
        self.total = total_episodes
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
        features: np.ndarray,  # (n_bars, n_features)
        atr_col_idx: int = 0,  # Index of the ATR column in features
        avg_atr: float = 0.0,  # Expected to be in the SAME unit as features[:,atr_col_idx]
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
            print(
                f"[Curriculum] Ep {episode:4d} | Phase: {phase['name']:12s} | "
                f"Progress: {self.progress:.1%} | "
                f"Difficulty: {self.get_difficulty_multiplier():.2f}"
            )


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
    """  # noqa: RUF002

    def __init__(
        self,
        window: int = 100,
        risk_free: float = 0.05 / 252,  # Daily risk-free rate
        annualize: float = np.sqrt(252),
        cost_penalty: float = 0.3,
        dd_penalty: float = 0.5,
    ):
        self.window = window
        self.rf = risk_free
        self.ann = annualize
        self.cost_pen = cost_penalty
        self.dd_pen = dd_penalty
        self._returns: collections.deque = collections.deque(maxlen=window)
        self._peak_eq = 1.0
        self._equity = 1.0

    def reset(self):
        self._returns.clear()
        self._peak_eq = 1.0
        self._equity = 1.0

    def compute(
        self,
        raw_pnl: float,
        tx_cost: float = 0.0,
        equity: float = 1.0,
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
        mu = ret_arr.mean()
        sigma = ret_arr.std() + 1e-9
        sharpe = (mu - self.rf) / sigma * self.ann

        # Drawdown penalty
        dd = max(0.0, (self._peak_eq - equity) / self._peak_eq)
        dd_pen = self.dd_pen * dd

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
        capacity: int = 100_000,
        k: int = 4,  # HER relabeling ratio
        goal_dim: int = 1,  # Dimension of goal (target price)
        strategy: str = "future",  # "future" | "episode" | "random"
    ):
        self.capacity = capacity
        self.k = k
        self.goal_dim = goal_dim
        self.strategy = strategy

        # Episode buffer (cleared after each episode)
        self._episode: list[dict] = []
        # Replay buffer (persistent)
        self._buffer: collections.deque = collections.deque(maxlen=capacity)
        # Cached list view of the deque for sampling; rebuilt only after the
        # buffer composition changes (end_episode), not on every sample().
        self._cache: list | None = None

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        goal: np.ndarray,  # Intended target (e.g. entry + ATR x 1.5)
        achieved: np.ndarray,  # Actually achieved (e.g. exit price)
        info: dict | None = None,
    ):
        """Store one transition in the episode buffer."""
        if info is None:
            info = {}
        self._episode.append(
            {
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
                "goal": goal,
                "achieved": achieved,
                "info": info,
            }
        )

    def _hindsight_reward(self, achieved: np.ndarray, goal: np.ndarray) -> float:
        """
        Compute reward for a relabeled (hindsight) goal.
        Binary: 1.0 if achieved ≈ goal, -1.0 otherwise.
        For forex: 1.0 if exit price matches relabeled target.
        """
        dist = float(np.linalg.norm(achieved - goal))
        return 1.0 if dist < 0.0002 else -0.1  # ~2 pip tolerance

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
                # previously used `random.randint(t_idx, n - 1)` - INCLUSIVE of
                # `t_idx` itself. When `t_idx == future_idx` (a 1/n chance per
                # sample on average, but ALWAYS the case for the final
                # transition t_idx == n-1 since both endpoints coincide), the
                # HER relabel becomes a self-match: `dist=0` deterministic, the
                # `_hindsight_reward` evaluates to the trivial max (+1.0 for a
                # tolerance-based reward), giving the agent a guaranteed
                # positive reward for "free" without actually achieving a goal.
                # Classic HER samples `t_idx+1 .. n-1` precisely to avoid this.
                # For t_idx == n-1 (last transition), there are no future
                # states to sample from - we SKIP the HER relabel for that
                # transition rather than polluting the buffer with self-matches.
                if self.strategy == "future":
                    if t_idx + 1 >= n:
                        # No future transitions in this episode - skip HER
                        # relabel for this transition rather than self-match.
                        break
                    future_idx = random.randint(t_idx + 1, n - 1)
                elif self.strategy == "episode":
                    # "episode" strategy still samples any of 0..n-1 (the
                    # whole episode, including past + future of t_idx). This
                    # is explicitly HER "episode" mode and is by-design a
                    # potential self-match source - callers should prefer
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

                her_goal = ep[future_idx]["achieved"]

                # If the chosen "future achieved" coincides with the current
                # achieved (e.g. price didn't move between t_idx and future_idx),
                # the relabel would still be a degenerate self-match. Skip it.
                if np.array_equal(her_goal, transition["achieved"]):
                    continue

                her_reward = self._hindsight_reward(transition["achieved"], her_goal)
                her_done = future_idx == n - 1

                self._buffer.append(
                    {
                        "obs": np.concatenate([transition["obs"], her_goal]),
                        "action": transition["action"],
                        "reward": her_reward,
                        "next_obs": np.concatenate([transition["next_obs"], her_goal]),
                        "done": her_done,
                        "goal": her_goal,
                        "achieved": transition["achieved"],
                        "info": {"her": True},
                    }
                )

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
    print("RL Advanced - smoke tests")
    print("=" * 50)

    # Curriculum
    sched = CurriculumScheduler(total_episodes=500)
    for ep in [0, 100, 250, 400, 499]:
        sched.current = ep
        p = sched.current_phase
        print(f"  Ep {ep:3d} -> phase: {p['name']:12s} | atr_max: {p['atr_max_mult']}")

    # Sharpe reward
    sr = SharpeRewardWrapper(window=20)
    for i in range(25):  # noqa: B007
        r = sr.compute(raw_pnl=np.random.normal(0.001, 0.003), tx_cost=0.0001)
    print(f"\n  Sharpe reward (final): {r:.4f}")
    print(f"  Rolling Sharpe: {sr.rolling_sharpe():.3f}")

    # HER buffer
    her = HERBuffer(capacity=1000, k=4)
    for i in range(10):
        her.store_transition(
            obs=np.zeros(10),
            action=0,
            reward=-0.5,
            next_obs=np.zeros(10),
            done=(i == 9),
            goal=np.array([1.0870]),
            achieved=np.array([1.0845]),
        )
    her.end_episode()
    print(f"\n  HER buffer size: {len(her)}")
    print(f"  HER ratio: {her.her_ratio:.1%}")
    batch = her.sample(8)
    her_count = sum(1 for t in batch if t.get("info", {}).get("her", False))
    print(f"  HER in batch of 8: {her_count} relabeled")

    # Multi-agent
    print("\n  Multi-agent coordinator test:")

    class DummyAgent:
        def select_action(self, obs):
            return 1

    agents = {"EURUSD": DummyAgent(), "GBPUSD": DummyAgent()}
    coord = MultiAgentCoordinator(agents, ["EURUSD", "GBPUSD"])
    obs = {"EURUSD": np.zeros(10), "GBPUSD": np.zeros(10)}
    actions = coord.select_actions(obs)
    print(f"  Actions: {actions}")
    print(f"  Portfolio: {coord.portfolio_summary()}")

    print("\nAll RL advanced tests passed ")
