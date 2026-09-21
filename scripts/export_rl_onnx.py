"""
scripts/export_rl_onnx.py
=========================
Exports trained RL policies (single agent and multi-agent consensus ensemble)
to optimized ONNX graphs for low-latency live execution in C++ and Python.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rl_advanced import RLEnsemble
from models.rl_agents import PPOAgent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class SinglePPOExecutionModule(nn.Module):
    """Wraps PPO._ActorCritic to output action logits."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)
        h, _ = self.net.lstm(obs)
        h = self.net.proj(h[:, -1, :])
        logits = self.net.actor(h)
        return logits


class RLEnsembleExecutionModule(nn.Module):
    """Wraps the 3-agent PPO ensemble into a single differentiable ONNX graph."""

    def __init__(self, nets: list[nn.Module], weights: list[float]):
        super().__init__()
        self.nets = nn.ModuleList(nets)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)

        probs_list = []
        for net in self.nets:
            h, _ = net.lstm(obs)
            h = net.proj(h[:, -1, :])
            logits = net.actor(h)
            probs = F.softmax(logits, dim=-1)
            probs_list.append(probs)

        stacked_probs = torch.stack(probs_list, dim=0)
        w = self.weights.view(-1, 1, 1)
        consensus_probs = (stacked_probs * w).sum(dim=0)
        consensus_logits = torch.log(consensus_probs + 1e-12)
        return consensus_logits


def export_rl_models(
    rl_best_path: str = "checkpoints/ensemble/rl_best.pt",
    rl_ensemble_path: str = "checkpoints/ensemble/rl_ensemble_best.pt",
    out_dir: str = "checkpoints/ensemble",
    opset: int = 17,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Export rl_best.pt
    best_file = Path(rl_best_path)
    if best_file.exists():
        logging.info(f"Loading single PPO agent from {best_file}...")
        state_dict = torch.load(best_file, map_location="cpu", weights_only=False)
        obs_size = state_dict["lstm.weight_ih_l0"].shape[1]
        n_actions = state_dict["actor.weight"].shape[0]
        lstm_hidden = state_dict["lstm.weight_hh_l0"].shape[1]
        hidden = state_dict["actor.weight"].shape[1]
        logging.info(f"Detected PPO architecture: obs_size={obs_size}, n_actions={n_actions}, lstm_hidden={lstm_hidden}, hidden={hidden}")

        dummy_agent = PPOAgent(
            obs_size=obs_size,
            n_actions=n_actions,
            use_lstm=True,
            lstm_hidden=lstm_hidden,
            hidden=hidden,
            device="cpu",
        )
        dummy_agent.net.load_state_dict(state_dict)
        dummy_agent.net.eval()

        single_module = SinglePPOExecutionModule(dummy_agent.net)
        single_module.eval()

        dummy_obs = torch.randn(1, obs_size, dtype=torch.float32)
        single_onnx_out = out_path / "rl_best.onnx"
        logging.info(f"Exporting single agent to ONNX: {single_onnx_out}...")
        torch.onnx.export(
            single_module,
            dummy_obs,
            str(single_onnx_out),
            opset_version=opset,
            input_names=["observation"],
            output_names=["action_logits"],
            dynamic_axes={"observation": {0: "batch_size"}, "action_logits": {0: "batch_size"}},
        )
        logging.info(f"[SUCCESS] Saved single agent ONNX: {single_onnx_out}")

    # 2. Export rl_ensemble_best.pt
    ens_file = Path(rl_ensemble_path)
    if ens_file.exists():
        logging.info(f"Loading RLEnsemble from {ens_file}...")
        payload = torch.load(ens_file, map_location="cpu", weights_only=False)
        obs_size = int(payload.get("obs_size", 591))
        n_actions = int(payload.get("n_actions", 10))
        weights = list(payload.get("weights", [1.0 / 3.0] * 3))
        agent_states = payload.get("agent_states", [])
        agent_configs = payload.get("agent_configs", [])

        sub_nets = []
        for i, (cfg, st) in enumerate(zip(agent_configs, agent_states)):
            sub_obs = cfg.get("obs_size", obs_size)
            sub_act = cfg.get("n_actions", n_actions)
            sub_lhidden = cfg.get("lstm_hidden", 128)
            sub_hidden = cfg.get("hidden", 256)
            ag = PPOAgent(
                obs_size=sub_obs,
                n_actions=sub_act,
                use_lstm=True,
                lstm_hidden=sub_lhidden,
                hidden=sub_hidden,
                device="cpu",
            )
            ag.net.load_state_dict(st)
            ag.net.eval()
            sub_nets.append(ag.net)

        ens_module = RLEnsembleExecutionModule(sub_nets, weights)
        ens_module.eval()

        dummy_obs = torch.randn(1, obs_size, dtype=torch.float32)
        ens_onnx_out = out_path / "rl_ensemble_best.onnx"
        logging.info(f"Exporting 3-agent PPO consensus ensemble to ONNX: {ens_onnx_out}...")
        torch.onnx.export(
            ens_module,
            dummy_obs,
            str(ens_onnx_out),
            opset_version=opset,
            input_names=["observation"],
            output_names=["consensus_logits"],
            dynamic_axes={"observation": {0: "batch_size"}, "consensus_logits": {0: "batch_size"}},
        )
        logging.info(f"[SUCCESS] Saved RLEnsemble consensus ONNX: {ens_onnx_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export RL Execution Policies to ONNX")
    parser.add_argument("--rl-best", default="checkpoints/ensemble/rl_best.pt")
    parser.add_argument("--rl-ensemble", default="checkpoints/ensemble/rl_ensemble_best.pt")
    parser.add_argument("--out-dir", default="checkpoints/ensemble")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_rl_models(
        rl_best_path=args.rl_best,
        rl_ensemble_path=args.rl_ensemble,
        out_dir=args.out_dir,
        opset=args.opset,
    )
