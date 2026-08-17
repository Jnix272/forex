"""
RL Adapter for CleanRL and Stable-Baselines3.

This module provides unified interfaces to popular RL frameworks and adapts
them to our Forex trading environment. It supports:

1. **CleanRL** (vwxyzjn/cleanrl) - Single-file, research-friendly implementations
2. **Stable-Baselines3** (DLR-RM/stable-baselines3) - Production-ready, modular implementations
3. **Custom RL** (existing PPO/DQN from rl_agents.py) - For backward compatibility

Usage:
    from training.rl_adapter import RLAdapter, create_rl_adapter, RLConfig

    # Using CleanRL PPO
    adapter = create_rl_adapter("cleanrl", "ppo", config)
    adapter.train(env, total_timesteps=100000)

    # Using Stable-Baselines3 PPO
    adapter = create_rl_adapter("sb3", "ppo", config)
    adapter.train(env, total_timesteps=100000)

    # Using custom RL (existing)
    adapter = create_rl_adapter("custom", "ppo", config)
    adapter.train(env, total_timesteps=100000)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


# ════════════════════════════════════════════════════════════════════════════
# Activation resolution helper
# ════════════════════════════════════════════════════════════════════════════

# Maps common lowercase activation names to their correct torch.nn class names.
# Using .capitalize() is wrong: it produces 'Relu' instead of 'ReLU', 'Gelu'
# instead of 'GELU', etc.  This lookup is case-insensitive and explicit.
_ACTIVATION_MAP: dict[str, str] = {
    "relu": "ReLU",
    "gelu": "GELU",
    "tanh": "Tanh",
    "sigmoid": "Sigmoid",
    "leakyrelu": "LeakyReLU",
    "leaky_relu": "LeakyReLU",
    "silu": "SiLU",
    "swish": "SiLU",  # alias
    "elu": "ELU",
    "selu": "SELU",
    "mish": "Mish",
    "softplus": "Softplus",
}


def _resolve_activation(name: str) -> "type[nn.Module]":
    """Return the torch.nn activation class for a given name string.

    Uses an explicit mapping table rather than ``.capitalize()`` which produces
    invalid names (e.g. ``'Relu'`` instead of ``'ReLU'``).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to resolve activation functions.")
    key = name.lower().replace("-", "_")
    cls_name = _ACTIVATION_MAP.get(key)
    if cls_name is None:
        # Fall back: try exact name first, then title-cased
        if hasattr(torch.nn, name):
            return getattr(torch.nn, name)
        raise ValueError(
            f"Unknown activation '{name}'. Supported: {sorted(_ACTIVATION_MAP)}. Or pass the exact torch.nn class name."
        )
    return getattr(torch.nn, cls_name)


# ════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class RLConfig:
    """Configuration for RL adapters."""

    # Algorithm
    algorithm: Literal["ppo", "dqn", "a2c", "sac", "td3"] = "ppo"

    # Network architecture
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: str = "relu"

    # Training
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # PPO specific
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10

    # DQN specific
    buffer_size: int = 100000
    learning_starts: int = 1000
    train_freq: int = 4
    target_update_interval: int = 1000
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05

    # General
    total_timesteps: int = 100000
    device: str | int = "auto"
    seed: int | None = None
    verbose: int = 1

    # Logging/Checkpointing
    log_dir: str | None = None
    save_freq: int = 10000
    save_path: str | None = None

    # Environment
    env_wrapper: Callable | None = None

    # Custom RL specific
    custom_reward_weights: dict | None = None


# ════════════════════════════════════════════════════════════════════════════
# Base Adapter Interface
# ════════════════════════════════════════════════════════════════════════════


class BaseRLAdapter(ABC):
    """Abstract base class for RL adapters."""

    def __init__(self, config: RLConfig):
        self.config = config
        self.model = None
        self.is_trained = False
        self.training_history = []

    @abstractmethod
    def train(
        self, env: Any, total_timesteps: int | None = None, callback: Callable | None = None, **kwargs
    ) -> dict[str, Any]:
        """Train the RL agent.

        Args:
            env: Trading environment (must implement reset, step, action_mask)
            total_timesteps: Override config total_timesteps
            callback: Optional callback function called during training

        Returns:
            Training metrics/history
        """
        pass

    @abstractmethod
    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        action_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict action for given observation.

        Args:
            obs: Observation array
            deterministic: Whether to use deterministic policy
            action_mask: Optional action mask for invalid actions

        Returns:
            (action, log_prob_or_none)
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        pass

    @abstractmethod
    def load(self, path: str, env: Any | None = None) -> None:
        """Load model checkpoint."""
        pass

    @abstractmethod
    def get_policy(self) -> Any:
        """Get underlying policy network for inspection/export."""
        pass

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self.config.seed = seed
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random

        random.seed(seed)


# ════════════════════════════════════════════════════════════════════════════
# CleanRL Adapter
# ════════════════════════════════════════════════════════════════════════════


class CleanRLAdapter(BaseRLAdapter):
    """
    Adapter for CleanRL: High-quality single-file RL implementations.

    Paper: https://arxiv.org/abs/2112.13621
    Repo: https://github.com/vwxyzjn/cleanrl

    Installation:
        pip install git+https://github.com/vwxyzjn/cleanrl.git
        # or install from source

    CleanRL provides single-file implementations, so we need to either:
    1. Import from installed package
    2. Copy the relevant algorithm file
    3. Use a local copy
    """

    def __init__(self, config: RLConfig):
        super().__init__(config)
        self._cleanrl_module = None
        self._agent = None
        self._env = None

    def _import_cleanrl(self, algorithm: str):
        """Import CleanRL algorithm module."""
        try:
            # Try to import from installed package
            if algorithm == "ppo":
                from cleanrl.ppo import Agent

                return Agent
            elif algorithm == "dqn":
                from cleanrl.dqn import Agent

                return Agent
            else:
                raise NotImplementedError(f"CleanRL {algorithm} not implemented in adapter")
        except ImportError:
            # Try to import from local cleanrl directory
            import sys

            cleanrl_path = os.path.join(os.path.dirname(__file__), "..", "cleanrl")
            if os.path.exists(cleanrl_path) and cleanrl_path not in sys.path:
                sys.path.insert(0, cleanrl_path)

            try:
                if algorithm == "ppo":
                    from cleanrl.ppo import Agent

                    return Agent
                elif algorithm == "dqn":
                    from cleanrl.dqn import Agent

                    return Agent
            except ImportError:
                pass

            raise ImportError(
                "CleanRL not installed. Install with: pip install git+https://github.com/vwxyzjn/cleanrl.git"
            )

    def _make_env_fn(self, env):
        """Create environment factory function for CleanRL."""

        def env_fn():
            return env

        return env_fn

    def train(
        self, env: Any, total_timesteps: int | None = None, callback: Callable | None = None, **kwargs
    ) -> dict[str, Any]:
        """Train using CleanRL."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for CleanRL adapter")

        total_timesteps = total_timesteps or self.config.total_timesteps
        algorithm = self.config.algorithm.lower()

        # Import CleanRL
        AgentClass = self._import_cleanrl(algorithm)

        # Get environment info
        # CleanRL expects gymnasium-style env with observation_space and action_space
        env.obs_size if hasattr(env, "obs_size") else env.observation_space.shape[0]
        env.n_actions if hasattr(env, "n_actions") else env.action_space.n

        # Create agent
        # Note: CleanRL agents expect specific arguments
        device = (
            self.config.device if self.config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if algorithm == "ppo":
            self._agent = AgentClass(
                envs=[env],  # CleanRL uses vectorized envs
                # Network
                hidden_sizes=self.config.hidden_dims,
                activation=_resolve_activation(self.config.activation),
                # PPO hyperparams
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                clip_coef=self.config.clip_range,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                # Training
                num_steps=self.config.n_steps,
                num_minibatches=self.config.n_steps // self.config.batch_size,
                update_epochs=self.config.n_epochs,
                # General
                total_timesteps=total_timesteps,
                device=device,
                seed=self.config.seed,
                capture_video=False,
            )
        elif algorithm == "dqn":
            self._agent = AgentClass(
                envs=[env],
                hidden_sizes=self.config.hidden_dims,
                activation=_resolve_activation(self.config.activation),
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                buffer_size=self.config.buffer_size,
                learning_starts=self.config.learning_starts,
                train_freq=self.config.train_freq,
                target_network_frequency=self.config.target_update_interval,
                exploration_fraction=self.config.exploration_fraction,
                exploration_initial_eps=self.config.exploration_initial_eps,
                exploration_final_eps=self.config.exploration_final_eps,
                total_timesteps=total_timesteps,
                device=device,
                seed=self.config.seed,
                capture_video=False,
            )
        else:
            raise NotImplementedError(f"Algorithm {algorithm} not supported")

        # Train
        self._agent.learn()
        self.is_trained = True

        # Extract training history if available
        if hasattr(self._agent, "losses"):
            self.training_history = self._agent.losses

        return {"total_timesteps": total_timesteps, "algorithm": algorithm}

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        action_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict using CleanRL agent."""
        if not self.is_trained:
            raise RuntimeError("Agent not trained. Call train() first.")

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required")

        # Convert obs to tensor
        obs_tensor = torch.from_numpy(obs).float().to(self._agent.device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        self._agent.eval()
        with torch.no_grad():
            if self.config.algorithm == "ppo":
                action, log_prob, _, _ = self._agent.get_action_and_value(obs_tensor)
                return action.cpu().numpy(), log_prob.cpu().numpy()
            elif self.config.algorithm == "dqn":
                q_values = self._agent.q_network(obs_tensor)
                if action_mask is not None:
                    mask_tensor = torch.from_numpy(action_mask).bool().to(self._agent.device)
                    q_values = q_values.masked_fill(~mask_tensor, -float("inf"))
                action = q_values.argmax(dim=-1)
                return action.cpu().numpy(), None

        return np.array([0]), None

    def save(self, path: str) -> None:
        """Save CleanRL agent."""
        if not self.is_trained:
            raise RuntimeError("Agent not trained.")

        if self.config.algorithm == "ppo":
            torch.save(
                {
                    "agent_state": self._agent.agent.state_dict(),
                    "optimizer_state": self._agent.optimizer.state_dict(),
                    "config": self.config,
                },
                path,
            )
        elif self.config.algorithm == "dqn":
            torch.save(
                {
                    "q_network_state": self._agent.q_network.state_dict(),
                    "optimizer_state": self._agent.optimizer.state_dict(),
                    "config": self.config,
                },
                path,
            )

    def load(self, path: str, env: Any | None = None) -> None:
        """Load CleanRL agent."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required")

        checkpoint = torch.load(path, map_location=self.config.device)

        # Recreate agent and load state
        self.train(env, total_timesteps=1)  # dummy train to initialize

        if self.config.algorithm == "ppo":
            self._agent.agent.load_state_dict(checkpoint["agent_state"])
            self._agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
        elif self.config.algorithm == "dqn":
            self._agent.q_network.load_state_dict(checkpoint["q_network_state"])
            self._agent.optimizer.load_state_dict(checkpoint["optimizer_state"])

        self.is_trained = True

    def get_policy(self) -> Any:
        """Get policy network."""
        if self.config.algorithm == "ppo":
            return self._agent.agent
        elif self.config.algorithm == "dqn":
            return self._agent.q_network
        return None


# ════════════════════════════════════════════════════════════════════════════
# Stable-Baselines3 Adapter
# ════════════════════════════════════════════════════════════════════════════


class SB3Adapter(BaseRLAdapter):
    """
    Adapter for Stable-Baselines3: Production-ready RL implementations.

    Repo: https://github.com/DLR-RM/stable-baselines3
    Docs: https://stable-baselines3.readthedocs.io/

    Installation:
        pip install stable-baselines3
        # For extra algorithms:
        pip install stable-baselines3[extra]
    """

    def __init__(self, config: RLConfig):
        super().__init__(config)
        self._sb3_model = None
        self._vec_env = None

    def _import_sb3(self):
        """Import Stable-Baselines3."""
        try:
            import stable_baselines3 as sb3

            return sb3
        except ImportError:
            raise ImportError("Stable-Baselines3 not installed. Install with: pip install stable-baselines3")

    def _wrap_env(self, env):
        """Wrap environment for SB3 (needs gymnasium interface)."""
        # GymEnvWrapper is defined in this same module - no import needed
        return GymEnvWrapper(env)

    def train(
        self, env: Any, total_timesteps: int | None = None, callback: Callable | None = None, **kwargs
    ) -> dict[str, Any]:
        """Train using Stable-Baselines3."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for SB3 adapter")

        sb3 = self._import_sb3()
        total_timesteps = total_timesteps or self.config.total_timesteps
        algorithm = self.config.algorithm.upper()

        # Wrap environment
        self._vec_env = self._wrap_env(env)

        # Create model
        device = (
            self.config.device if self.config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        policy_kwargs = {
            "net_arch": list(self.config.hidden_dims),
            "activation_fn": _resolve_activation(self.config.activation),
        }

        if algorithm == "PPO":
            self._sb3_model = sb3.PPO(
                "MlpPolicy",
                self._vec_env,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                clip_range=self.config.clip_range,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                policy_kwargs=policy_kwargs,
                device=device,
                seed=self.config.seed,
                verbose=self.config.verbose,
                tensorboard_log=self.config.log_dir,
            )
        elif algorithm == "DQN":
            self._sb3_model = sb3.DQN(
                "MlpPolicy",
                self._vec_env,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                buffer_size=self.config.buffer_size,
                learning_starts=self.config.learning_starts,
                train_freq=self.config.train_freq,
                target_update_interval=self.config.target_update_interval,
                exploration_fraction=self.config.exploration_fraction,
                exploration_initial_eps=self.config.exploration_initial_eps,
                exploration_final_eps=self.config.exploration_final_eps,
                policy_kwargs=policy_kwargs,
                device=device,
                seed=self.config.seed,
                verbose=self.config.verbose,
                tensorboard_log=self.config.log_dir,
            )
        elif algorithm == "A2C":
            self._sb3_model = sb3.A2C(
                "MlpPolicy",
                self._vec_env,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                n_steps=self.config.n_steps,
                policy_kwargs=policy_kwargs,
                device=device,
                seed=self.config.seed,
                verbose=self.config.verbose,
                tensorboard_log=self.config.log_dir,
            )
        else:
            raise NotImplementedError(f"SB3 algorithm {algorithm} not implemented in adapter")

        # Train
        if callback is not None:
            # Wrap callback for SB3
            from stable_baselines3.common.callbacks import BaseCallback

            class SB3Callback(BaseCallback):
                def __init__(self, callback_fn):
                    super().__init__()
                    self.callback_fn = callback_fn

                def _on_step(self) -> bool:
                    return self.callback_fn(self.locals, self.globals)

            sb3_callback = SB3Callback(callback)
            self._sb3_model.learn(total_timesteps=total_timesteps, callback=sb3_callback)
        else:
            self._sb3_model.learn(total_timesteps=total_timesteps)

        self.is_trained = True

        return {"total_timesteps": total_timesteps, "algorithm": algorithm}

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        action_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict using SB3 model."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        # SB3 predict returns (action, state) for recurrent policies
        action, _ = self._sb3_model.predict(obs, deterministic=deterministic, action_masks=action_mask)

        return action, None  # SB3 doesn't return log_probs by default

    def save(self, path: str) -> None:
        """Save SB3 model."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        self._sb3_model.save(path)

    def load(self, path: str, env: Any | None = None) -> None:
        """Load SB3 model."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required")

        sb3 = self._import_sb3()
        algorithm = self.config.algorithm.upper()

        if algorithm == "PPO":
            self._sb3_model = sb3.PPO.load(path, env=self._wrap_env(env) if env else None, device=self.config.device)
        elif algorithm == "DQN":
            self._sb3_model = sb3.DQN.load(path, env=self._wrap_env(env) if env else None, device=self.config.device)
        elif algorithm == "A2C":
            self._sb3_model = sb3.A2C.load(path, env=self._wrap_env(env) if env else None, device=self.config.device)

        self.is_trained = True

    def get_policy(self) -> Any:
        """Get SB3 policy."""
        return self._sb3_model.policy if self._sb3_model else None


# ════════════════════════════════════════════════════════════════════════════
# Custom RL Adapter (existing rl_agents.py)
# ════════════════════════════════════════════════════════════════════════════


class CustomRLAdapter(BaseRLAdapter):
    """
    Adapter for existing custom RL implementation (models/rl_agents.py).

    This wraps the ForexTradingEnv, PPOAgent, and DQNAgent from rl_agents.py
    to provide a unified interface.
    """

    def __init__(self, config: RLConfig):
        super().__init__(config)
        self._agent = None
        self._env = None

    def _import_rl_agents(self):
        """Import custom RL agents."""
        try:
            from models.rl_agents import (
                DQNAgent,
                ForexTradingEnv,
                PPOAgent,
            )
            from models.rl_agents import (
                train_agent as custom_train_agent,
            )

            return {
                "ForexTradingEnv": ForexTradingEnv,
                "PPOAgent": PPOAgent,
                "DQNAgent": DQNAgent,
                "train_agent": custom_train_agent,
            }
        except ImportError as e:
            raise ImportError(f"Custom RL agents not available: {e}")

    def train(
        self, env: Any, total_timesteps: int | None = None, callback: Callable | None = None, **kwargs
    ) -> dict[str, Any]:
        """Train using custom RL implementation."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for custom RL adapter")

        modules = self._import_rl_agents()
        algorithm = self.config.algorithm.lower()

        # The custom train_agent expects n_episodes, not total_timesteps
        # Convert approximately
        total_timesteps = kwargs.get("total_timesteps", 10000)
        n_episodes = kwargs.get("n_episodes", max(1, total_timesteps // 1000))

        if algorithm == "ppo":
            self._agent = modules["PPOAgent"](
                obs_size=env.obs_size,
                n_actions=env.n_actions,
                device=self.config.device
                if self.config.device != "auto"
                else ("cuda" if torch.cuda.is_available() else "cpu"),
                hidden_sizes=self.config.hidden_dims,
                lr=self.config.learning_rate,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                clip_range=self.config.clip_range,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
            )
        elif algorithm == "dqn":
            self._agent = modules["DQNAgent"](
                obs_size=env.obs_size,
                n_actions=env.n_actions,
                device=self.config.device
                if self.config.device != "auto"
                else ("cuda" if torch.cuda.is_available() else "cpu"),
                hidden_sizes=self.config.hidden_dims,
                lr=self.config.learning_rate,
                gamma=self.config.gamma,
                buffer_size=self.config.buffer_size,
                batch_size=self.config.batch_size,
                target_update=self.config.target_update_interval,
            )
        else:
            raise NotImplementedError(f"Custom RL algorithm {algorithm} not supported")

        self._env = env

        # Train
        returns = modules["train_agent"](
            self._agent,
            env,
            n_episodes=n_episodes,
            agent_type=algorithm,
        )

        self.is_trained = True
        self.training_history = returns

        return {"returns": returns, "n_episodes": n_episodes}

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        action_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict using custom RL agent."""
        if not self.is_trained:
            raise RuntimeError("Agent not trained. Call train() first.")

        if self.config.algorithm == "ppo":
            action, log_prob, _ = self._agent.select_action(obs, mask=action_mask, greedy=deterministic)
            return np.array([action]), np.array([log_prob])
        elif self.config.algorithm == "dqn":
            action = self._agent.select_action(obs, mask=action_mask)
            return np.array([action]), None

        return np.array([0]), None

    def save(self, path: str) -> None:
        """Save custom RL agent."""
        if not self.is_trained:
            raise RuntimeError("Agent not trained.")

        torch.save(
            {
                "agent_state": self._agent.state_dict(),
                "config": self.config,
                "algorithm": self.config.algorithm,
            },
            path,
        )

    def load(self, path: str, env: Any | None = None) -> None:
        """Load custom RL agent."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required")

        checkpoint = torch.load(path, map_location=self.config.device)

        # Recreate agent
        modules = self._import_rl_agents()
        algorithm = checkpoint.get("algorithm", self.config.algorithm)

        if algorithm == "ppo":
            self._agent = modules["PPOAgent"](
                obs_size=env.obs_size if env else self.config.hidden_dims[0],
                n_actions=env.n_actions if env else 10,
                device=self.config.device,
                hidden_sizes=self.config.hidden_dims,
            )
        elif algorithm == "dqn":
            self._agent = modules["DQNAgent"](
                obs_size=env.obs_size if env else self.config.hidden_dims[0],
                n_actions=env.n_actions if env else 10,
                device=self.config.device,
                hidden_sizes=self.config.hidden_dims,
            )

        self._agent.load_state_dict(checkpoint["agent_state"])
        self.is_trained = True

    def get_policy(self) -> Any:
        """Get custom policy network."""
        if self._agent:
            return self._agent.actor if hasattr(self._agent, "actor") else self._agent.q_network
        return None


# ════════════════════════════════════════════════════════════════════════════
# Environment Wrapper for Gymnasium Compatibility
# ════════════════════════════════════════════════════════════════════════════


class GymEnvWrapper:
    """
    Wrapper to make ForexTradingEnv compatible with Gymnasium/SB3.

    Our ForexTradingEnv has:
    - reset(valid_starts=None) -> obs
    - step(action) -> (obs, reward, done, info)
    - action_mask() -> np.ndarray
    - obs_size, n_actions properties

    Gymnasium expects:
    - reset(seed=None, options=None) -> (obs, info)
    - step(action) -> (obs, reward, terminated, truncated, info)
    - action_space, observation_space properties
    """

    def __init__(self, env):
        self.env = env

        # Define spaces
        import gymnasium as gym

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(env.obs_size,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(env.n_actions)

    def reset(self, seed=None, options=None):
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)

        valid_starts = options.get("valid_starts") if options else None
        obs = self.env.reset(valid_starts=valid_starts)
        info = {}
        return obs, info

    def step(self, action):
        """Step environment."""
        obs, reward, done, info = self.env.step(action)
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def action_mask(self):
        """Get action mask."""
        return self.env.action_mask()

    def render(self):
        """Render environment (not implemented)."""
        pass

    def close(self):
        """Close environment."""
        pass


# ════════════════════════════════════════════════════════════════════════════
# Factory Function
# ════════════════════════════════════════════════════════════════════════════


def create_rl_adapter(
    framework: Literal["cleanrl", "sb3", "stable_baselines3", "custom"],
    algorithm: Literal["ppo", "dqn", "a2c", "sac", "td3"] = "ppo",
    config: RLConfig | None = None,
    **kwargs,
) -> BaseRLAdapter:
    """
    Factory function to create RL adapter.

    Args:
        framework: "cleanrl", "sb3", "stable_baselines3", or "custom"
        algorithm: RL algorithm to use
        config: RLConfig (created from kwargs if not provided)
        **kwargs: Additional arguments

    Returns:
        RL adapter instance
    """
    if config is None:
        config = RLConfig(algorithm=algorithm, **kwargs)
    else:
        config.algorithm = algorithm
        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    framework = framework.lower()
    if framework in ("cleanrl",):
        return CleanRLAdapter(config)
    elif framework in ("sb3", "stable_baselines3", "stable-baselines3"):
        return SB3Adapter(config)
    elif framework in ("custom",):
        return CustomRLAdapter(config)
    else:
        raise ValueError(f"Unknown framework: {framework}. Choose from: cleanrl, sb3, stable_baselines3, custom")


# ════════════════════════════════════════════════════════════════════════════
# Integration with existing rl_runner
# ════════════════════════════════════════════════════════════════════════════


def run_rl_with_adapter(
    adapter: BaseRLAdapter,
    cache_path: str,
    train_indices: np.ndarray,
    prices: np.ndarray,
    atr: np.ndarray,
    spreads: np.ndarray,
    features: np.ndarray,
    **kwargs,
) -> dict[str, Any]:
    """
    Run RL training using adapter with Forex data.

    Integrates with existing data loading pipeline.
    """
    from models.rl_agents import ForexTradingEnv

    # Create environment
    env = ForexTradingEnv(
        features=features[train_indices],
        prices=prices[train_indices],
        atr=atr[train_indices],
        spreads=spreads[train_indices],
        **kwargs,
    )

    # Train adapter
    return adapter.train(env, **kwargs)


if __name__ == "__main__":
    # Demo
    print("RL Adapter Module")
    print("Available frameworks:")
    print("  - CleanRL (cleanrl): Single-file research implementations")
    print("  - Stable-Baselines3 (sb3): Production-ready modular implementations")
    print("  - Custom (custom): Existing rl_agents.py implementation")
    print()
    print("Usage:")
    print("  from training.rl_adapter import create_rl_adapter, RLConfig")
    print("  config = RLConfig(algorithm='ppo', total_timesteps=100000)")
    print("  adapter = create_rl_adapter('sb3', 'ppo', config)")
    print("  adapter.train(env)")
    print("  action, _ = adapter.predict(obs)")
