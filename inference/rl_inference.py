"""
inference/rl_inference.py
=========================
Load a trained DQN/PPO policy plus the frozen supervised encoder for live TIP fast path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import RL
from inference.onnx_inference import torch_load_safe
from inference._scaler_load import apply_inference_scaler
from models.rl_agents import DQNAgent, PPOAgent
from trading.inference_engines import BaseInferenceEngine
from trading.live_actions import LiveAction, scaling_action_to_live_action


def _resolve_rl_checkpoint(checkpoint_dir: Path, algo: str = "dqn") -> Path | None:
    algo = str(algo).lower()
    for name in (f"rl_{algo}_best.pt", f"rl_{algo}_last.pt"):
        p = checkpoint_dir / name
        if p.is_file():
            return p
    return None


class RLInferenceAgent(BaseInferenceEngine):
    """
    Fast-agent interface for LiveTradingEngine / TIPSearchManager.

    Expects the same 1-D feature row as the supervised engine; maintains a rolling
    window, encodes with the frozen backbone, and appends portfolio state (5 dims).
    """

    returns_live_actions = True

    def __init__(
        self,
        rl_checkpoint: str,
        supervised_checkpoint: str,
        model_name: str,
        seq_len: int = 60,
        n_features: int | None = None,
        algo: str = "dqn",
        device: Any | None = None,
        initial_equity: float = 10_000.0,
        max_lots: float = 3.0,
    ):
        import torch

        from inference.pytorch_inference import load_pytorch_model
        from training.train_gpu import _core_model

        self.algo = str(algo).lower()
        self.seq_len = int(seq_len)
        self.initial_equity = float(initial_equity)
        self.max_lots = float(max_lots)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model, self.n_features, self.seq_len, self.arch_name, self._scaler = load_pytorch_model(
            supervised_checkpoint,
            model_name,
            seq_len=self.seq_len,
            n_features=n_features,
            device=self.device,
        )
        core = _core_model(self._model)
        self._encoder = core.backbone if hasattr(core, "backbone") else core
        if hasattr(self._encoder, "head"):
            self._encoder.head = torch.nn.Identity()
        self._encoder.eval()

        ckpt = torch_load_safe(rl_checkpoint, map_location=self.device)
        meta_path = Path(rl_checkpoint).parent / f"rl_{self.algo}_best.json"
        obs_size = None
        n_actions = None
        if meta_path.is_file():
            import json
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            obs_size = int(meta.get("obs_size", 0)) or None
            n_actions = int(meta.get("n_actions", 0)) or None

        if obs_size is None:
            obs_size = self._infer_obs_size() + 5
        if n_actions is None:
            n_actions = self._infer_n_actions(ckpt)

        algo_kw = dict(RL.get(self.algo, {}))
        if self.algo == "dqn":
            self._agent = DQNAgent(obs_size=obs_size, n_actions=n_actions, device=str(self.device), **algo_kw)
            self._agent.policy_net.load_state_dict(ckpt, strict=False)
            self._agent.target_net.load_state_dict(self._agent.policy_net.state_dict())
            self._agent.eps = 0.0
        else:
            self._agent = PPOAgent(obs_size=obs_size, n_actions=n_actions, device=str(self.device), **algo_kw)
            self._agent.net.load_state_dict(ckpt, strict=False)

        from collections import deque
        self._feat_buffer: deque[np.ndarray] = deque(maxlen=self.seq_len)
        self._position = 0.0
        self._entry_price = 0.0
        self._equity = self.initial_equity
        self._holding = 0
        self._last_price = 0.0
        print(
            f"[RLInference] Loaded {Path(rl_checkpoint).name} | "
            f"encoder={self.arch_name} | obs={obs_size} | actions={n_actions}"
        )

    def _infer_obs_size(self) -> int:
        import torch
        dummy = torch.zeros(1, self.seq_len, self.n_features, device=self.device)
        with torch.no_grad():
            h = self._encoder(dummy)
            if h.ndim == 3:
                h = h[:, -1, :]
        return int(h.shape[-1])

    def _infer_n_actions(self, ckpt) -> int:
        if not isinstance(ckpt, dict) or not ckpt:
            return 10
        if self.algo == "dqn":
            for key, value in reversed(list(ckpt.items())):
                if getattr(value, "ndim", 0) in (1, 2) and (
                    key.endswith("net.4.bias") or key.endswith("net.4.weight")
                ):
                    return int(value.shape[0])
        else:
            for key, value in ckpt.items():
                if key.endswith("actor.weight") and getattr(value, "ndim", 0) == 2:
                    return int(value.shape[0])
        return 10

    def set_agent_state(
        self,
        position_lots: float = 0.0,
        entry_price: float = 0.0,
        equity: float | None = None,
        holding_bars: int = 0,
        current_price: float = 0.0,
    ) -> None:
        self._position = float(position_lots)
        self._entry_price = float(entry_price)
        if equity is not None:
            self._equity = float(equity)
        self._holding = int(holding_bars)
        self._last_price = float(current_price)

    def reset_buffer(self):
        self._feat_buffer.clear()

    def select_action(self, obs: np.ndarray) -> int:
        import torch

        row = np.asarray(obs, dtype=np.float32).reshape(-1)
        self._feat_buffer.append(row)
        if len(self._feat_buffer) < self.seq_len:
            return int(LiveAction.HOLD)

        window = np.stack(list(self._feat_buffer), axis=0)
        # Apply the training-time scaler (if available) — matches ZarrStreamDataset
        if self._scaler is not None:
            window = apply_inference_scaler(self._scaler, window)
        xb = torch.as_tensor(window[np.newaxis], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            h = self._encoder(xb)
            if h.ndim == 3:
                h = h[:, -1, :]
            emb = h.float().cpu().numpy().reshape(-1)

        price = float(self._last_price) if getattr(self, "_last_price", 0) else 0.0
        # Standard FX lot = 100,000 base currency units
        STANDARD_LOT_SIZE = 100_000.0
        upnl = (
            (price - self._entry_price) * self._position * STANDARD_LOT_SIZE
            if self._position != 0 and price > 0
            else 0.0
        )
        agent_state = np.array([
            np.clip(self._position / self.max_lots, -1, 1),
            np.clip(upnl / self.initial_equity, -0.5, 0.5),
            min(self._holding / 100.0, 1.0),
            np.clip((self._equity - self.initial_equity) / self.initial_equity, -0.5, 0.5),
            float(self._position != 0),
        ], dtype=np.float32)
        full_obs = np.concatenate([emb, agent_state]).astype(np.float32)

        # I3 fix (2026-08-07): live inference must be deterministic.
        # The PPO actor used to always sample, injecting stochasticity into
        # position-sizing decisions. Pass greedy=True for PPO (DQN already
        # has eps=0 set in __init__ at line 99, so it doesn't need this kwarg).
        try:
            # PPO exposes `greedy=` kwarg on select_action; DQN does not, so we
            # guard with try/except to stay backward-compatible with DQN agents.
            action_t = self._agent.select_action(full_obs, greedy=True)
            # PPO returns (action, log_prob, value); DQN-style returns scalar
            action_int = int(action_t[0]) if isinstance(action_t, tuple) else int(action_t)
        except TypeError:
            # DQN-style agent without greedy kwarg — fall through to default
            action_int = int(self._agent.select_action(full_obs))
        action = max(0, min(9, action_int))
        return scaling_action_to_live_action(action, position_lots=self._position)


def build_rl_fast_agent(
    checkpoint_dir: Path,
    model_name: str,
    seq_len: int = 60,
    n_features: int | None = None,
    algo: str = "dqn",
) -> RLInferenceAgent | None:
    """Return RLInferenceAgent if rl_* checkpoint exists, else None."""
    ckpt_dir = Path(checkpoint_dir)
    rl_path = _resolve_rl_checkpoint(ckpt_dir, algo=algo)
    if rl_path is None:
        return None
    sup = ckpt_dir / f"{model_name}_best.pt"
    if not sup.is_file():
        sup = ckpt_dir.parent / f"{model_name}_best.pt"
    if not sup.is_file():
        return None
    try:
        return RLInferenceAgent(
            rl_checkpoint=str(rl_path),
            supervised_checkpoint=str(sup),
            model_name=model_name,
            seq_len=seq_len,
            n_features=n_features,
            algo=algo,
        )
    except Exception as exc:
        print(f"[RLInference] Failed to load RL agent: {exc}")
        return None
