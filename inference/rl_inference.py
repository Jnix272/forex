"""
inference/rl_inference.py
=========================
Load a trained DQN/PPO policy plus the frozen supervised encoder for live TIP fast path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import RL
from inference._scaler_load import apply_inference_scaler
from inference.onnx_inference import torch_load_safe
from models.rl_agents import DQNAgent, PPOAgent
from trading.inference_engines import BaseInferenceEngine
from trading.live_actions import LiveAction, scaling_action_to_live_action


def _resolve_rl_checkpoint(checkpoint_dir: Path, algo: str = "dqn") -> Path | None:
    """Locate RL checkpoint with broad fallback across checkpoint tree.

    The active checkpoint dir (e.g. ``checkpoints/forex_4pair_2015_2025_haelt``)
    historically contained no ``rl_*`` files while TFT/ensemble sub-trees did.
    We now search in priority order:
      1. exact ``checkpoint_dir``
      2. ``checkpoint_dir / model_subdir`` and sibling model dirs
      3. ``checkpoints/ensemble`` (consensus policy)
      4. any ``checkpoints/*/rl_*`` recursively (last resort)
    """
    algo = str(algo).lower()
    ckpt_dir = Path(checkpoint_dir)
    # 1. Direct hits in the requested dir
    for name in (f"rl_{algo}_best.pt", f"rl_{algo}_last.pt", f"rl_{algo}.pt", f"{algo}_best.pt"):
        p = ckpt_dir / name
        if p.is_file():
            return p

    # 2. One level of sibling model subdirs (e.g. .../haelt vs .../tft)
    try:
        from config.settings import PROJECT_ROOT
    except Exception:
        PROJECT_ROOT = ckpt_dir.parent.parent if ckpt_dir.parent.name == "checkpoints" else Path("checkpoints")
    # Check parent's immediate children (sibling model dirs)
    parent = ckpt_dir.parent if ckpt_dir.parent else ckpt_dir
    for sibling in parent.iterdir() if parent.is_dir() else []:
        if not sibling.is_dir():
            continue
        for name in (f"rl_{algo}_best.pt", f"rl_{algo}_last.pt"):
            p = sibling / name
            if p.is_file():
                return p

    # 3. Canonical ensemble dir (holds PPO consensus)
    for ens_name in ("rl_ensemble_best.pt", f"rl_{algo}_best.pt", "rl_best.pt"):
        for cand in (
            PROJECT_ROOT / "checkpoints" / "ensemble" / ens_name,
            Path("checkpoints") / "ensemble" / ens_name,
        ):
            if cand.is_file():
                return cand

    # 4. Recursive glob over checkpoints (last resort, deterministic sort)
    try:
        roots = [PROJECT_ROOT / "checkpoints", Path("checkpoints")]
        for root in roots:
            if not root.is_dir():
                continue
            for p in sorted(root.rglob(f"rl_{algo}_best.pt")):
                if p.is_file():
                    return p
            # generic fallback: any rl_*_best.pt
            for p in sorted(root.rglob("rl_*_best.pt")):
                if p.is_file():
                    return p
    except Exception:
        pass
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
        lot_size: float = 10_000.0,
    ):
        import torch

        from inference.pytorch_inference import load_pytorch_model
        from models.factory import _core_model

        self.algo = str(algo).lower()
        self.seq_len = int(seq_len)
        self.initial_equity = float(initial_equity)
        self.max_lots = float(max_lots)
        self._lot_size = float(lot_size)  # must match ForexTradingEnv(lot_size=)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model, self.n_features, self.seq_len, self.arch_name, self._scaler = load_pytorch_model(
            supervised_checkpoint,
            model_name,
            seq_len=self.seq_len,
            n_features=n_features,
            device=self.device,
        )
        core_any = cast(Any, _core_model(self._model))
        self._encoder = getattr(core_any, "backbone", core_any)
        encoder_any = cast(Any, self._encoder)
        if hasattr(encoder_any, "head"):
            encoder_any.head = torch.nn.Identity()
        encoder_any.eval()
        self._encoder = encoder_any

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
        # Guard: degenerate obs (e.g. 6 = 1+5) indicates wrong encoder head (ensemble meta-learner)
        # Fall back to raw feature dim +5 to keep live in-distribution with training (which was raw when encoder failed).
        if obs_size is not None and obs_size < 20:
            _fallback = int(self.n_features) + 5 if self.n_features else 589
            print(f"[RLInference] WARN: degenerate obs_size {obs_size} (<20) from {Path(rl_checkpoint).name}; falling back to raw { _fallback} (n_feat {self.n_features}+5)")
            obs_size = _fallback
            # Force raw path
            self._encoder_obs = False
            self._expected_emb_dim = _fallback - 5
        # Ensure n_actions at least 3 (BUY/HOLD/SELL) even if checkpoint corrupt
        if n_actions is not None and n_actions < 3:
            n_actions = 10

        algo_kw = dict(RL.get(self.algo, {}))
        if self.algo == "ensemble" or (isinstance(ckpt, dict) and ckpt.get("model_type") == "RLEnsemble"):
            from models.rl_advanced import RLEnsemble

            self.algo = "ensemble"
            self._agent = RLEnsemble.load_checkpoint(rl_checkpoint, device=str(self.device))
        elif self.algo == "dqn":
            self._agent = DQNAgent(obs_size=obs_size, n_actions=n_actions, device=str(self.device), **algo_kw)
            agent_any = cast(Any, self._agent)
            agent_any.policy_net.load_state_dict(ckpt, strict=False)
            agent_any.target_net.load_state_dict(agent_any.policy_net.state_dict())
            agent_any.eps = 0.0
        else:
            self._agent = PPOAgent(obs_size=obs_size, n_actions=n_actions, device=str(self.device), **algo_kw)
            agent_any = cast(Any, self._agent)
            agent_any.net.load_state_dict(ckpt, strict=False)

        # ── Encoder vs raw contract detection ──────────────────────────────
        # Training with rl_encoder_obs=True produces obs = encoder_emb + 5.
        # If the persisted obs_size does not match encoder_emb+5, training fell
        # back to raw features (e.g. encoder load failed). We auto-detect and
        # feed raw observations at live time to stay in-distribution.
        try:
            _inferred_emb = self._infer_obs_size()
        except Exception:
            _inferred_emb = None
        if _inferred_emb is not None and obs_size is not None:
            self._encoder_obs = int(obs_size) == int(_inferred_emb) + 5
            self._expected_emb_dim = int(obs_size) - 5
            if not self._encoder_obs:
                print(
                    f"[RLInference] WARN: obs_size {obs_size} != encoder { _inferred_emb}+5 "
                    f"→ training used RAW features; live will feed raw (no encoder)."
                )
        else:
            self._encoder_obs = True
            self._expected_emb_dim = int(_inferred_emb) if _inferred_emb is not None else int(obs_size) - 5 if obs_size else 0

        from collections import deque

        self._feat_buffer: deque[np.ndarray] = deque(maxlen=self.seq_len)
        self._position = 0.0
        self._entry_price = 0.0
        self._equity = self.initial_equity
        self._holding = 0
        self._last_price = 0.0
        print(
            f"[RLInference] Loaded {Path(rl_checkpoint).name} | "
            f"encoder={self.arch_name} | obs={obs_size} | actions={n_actions} | "
            f"encoder_obs={self._encoder_obs}"
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
        if ckpt.get("model_type") == "RLEnsemble" and "n_actions" in ckpt:
            return int(ckpt["n_actions"])
        if self.algo == "dqn":
            for key, value in reversed(list(ckpt.items())):
                if getattr(value, "ndim", 0) in (1, 2) and (key.endswith("net.4.bias") or key.endswith("net.4.weight")):
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
        # Apply the training-time scaler (if available) - matches ZarrStreamDataset
        if self._scaler is not None:
            window = apply_inference_scaler(self._scaler, window)

        # Raw training fallback: feed the last scaled row directly (no encoder)
        if not getattr(self, "_encoder_obs", True):
            emb = np.asarray(window[-1], dtype=np.float32).reshape(-1)
            # Ensure emb dim matches agent's expected obs_size-5
            exp = int(getattr(self, "_expected_emb_dim", emb.shape[0]))
            if emb.shape[0] != exp:
                if emb.shape[0] < exp:
                    emb = np.pad(emb, (0, exp - emb.shape[0])).astype(np.float32)
                else:
                    emb = emb[:exp].astype(np.float32)
        else:
            xb = torch.as_tensor(window[np.newaxis], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                h = self._encoder(xb)
                if h.ndim == 3:
                    h = h[:, -1, :]
                emb = h.float().cpu().numpy().reshape(-1)

        price = float(self._last_price) if getattr(self, "_last_price", 0) else 0.0
        # Must match ForexTradingEnv default (10_000), NOT the standard FX lot (100_000).
        # A mismatch inflates the upnl feature 10x vs training, pushing obs out-of-distribution.
        _lot_size = getattr(self, "_lot_size", 10_000.0)
        upnl = (
            (price - self._entry_price) * self._position * _lot_size
            if self._position != 0 and price > 0
            else 0.0
        )
        agent_state = np.array(
            [
                np.clip(self._position / self.max_lots, -1, 1),
                np.clip(upnl / self.initial_equity, -0.5, 0.5),
                min(self._holding / 100.0, 1.0),
                np.clip((self._equity - self.initial_equity) / self.initial_equity, -0.5, 0.5),
                float(self._position != 0),
            ],
            dtype=np.float32,
        )
        full_obs = np.concatenate([emb, agent_state]).astype(np.float32)

        # Build action mask from current position state so the policy only selects
        # legal actions (PPO was trained with env.action_mask(); without it, illegal
        # actions like SCALE_IN while flat can be selected with highest logit).
        n_actions = getattr(self._agent, "n_actions", 10)
        mask = np.ones(n_actions, dtype=bool)
        pos = self._position
        if pos == 0:
            mask[3:9] = False  # can't scale/reduce/close when flat
        elif pos > 0:
            mask[2] = False    # can't open short while long
        elif pos < 0:
            mask[1] = False    # can't open long while short

        # I3 fix (2026-08-07): live inference must be deterministic.
        # Both agents now accept greedy+mask; fallback preserves mask even if
        # an agent signature is outdated.
        try:
            action_t = self._agent.select_action(full_obs, greedy=True, mask=mask)
            action_int = int(action_t[0]) if isinstance(action_t, tuple) else int(action_t)
        except TypeError:
            try:
                action_int = int(self._agent.select_action(full_obs, mask=mask))
            except TypeError:
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
    """Return RLInferenceAgent if rl_* checkpoint exists, else None.

    Supervised checkpoint is resolved via the canonical
    ``resolve_checkpoint_paths`` (handles nested ``haelt/haelt_best.pt`` etc.)
    with additional fallbacks for the historic double-nested layout.
    RL checkpoint falls back across sibling dirs / ensemble if the exact
    ``algo`` is missing (e.g. haelt requests dqn but only ensemble PPO exists).
    """
    ckpt_dir = Path(checkpoint_dir)
    rl_path = _resolve_rl_checkpoint(ckpt_dir, algo=algo)
    # Fallback algos: dqn↔ppo↔ensemble soft-vote
    if rl_path is None:
        for fallback in ("ensemble", "ppo", "dqn"):
            if fallback == str(algo).lower():
                continue
            rl_path = _resolve_rl_checkpoint(ckpt_dir, algo=fallback)
            if rl_path is not None:
                algo = fallback
                break
    if rl_path is None:
        return None
    # Resolve supervised checkpoint canonically (handles nested, production, etc.)
    sup: Path | None = None
    try:
        from config.settings import resolve_checkpoint_paths

        paths = resolve_checkpoint_paths(model_name, ckpt_dir)
        if paths.pt_path is not None and paths.pt_path.is_file():
            sup = paths.pt_path
    except Exception:
        pass
    if sup is None or not sup.is_file():
        # Legacy / double-nested fallbacks
        for cand in (
            ckpt_dir / f"{model_name}_best.pt",
            ckpt_dir / model_name / f"{model_name}_best.pt",
            ckpt_dir / model_name / model_name / f"{model_name}_best.pt",
            ckpt_dir.parent / f"{model_name}_best.pt",
            ckpt_dir.parent / model_name / f"{model_name}_best.pt",
        ):
            if cand.is_file():
                sup = cand
                break
    if sup is None or not sup.is_file():
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
