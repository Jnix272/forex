"""
inference/onnx_inference.py
===========================
ONNX export and AMD DirectML inference for live trading.

Converts a trained PyTorch checkpoint -> ONNX, then runs it via
onnxruntime with the DirectML execution provider (AMD NPU/GPU on Windows).
Exposes a select_action() interface so it drops straight into LiveTradingEngine.

Usage — export:
    python inference/onnx_inference.py export \\
        --checkpoint checkpoints/haelt_best.pt \\
        --model haelt --seq-len 60

Usage — benchmark (compare CUDA / DirectML / CPU):
    python inference/onnx_inference.py benchmark \\
        --onnx checkpoints/haelt_best.onnx \\
        --seq-len 60 --n-feat 128 --runs 500

Usage — live inference only (drop-in for LiveTradingEngine):
    from inference.onnx_inference import DirectMLInferenceEngine
    engine = DirectMLInferenceEngine("checkpoints/haelt_best.onnx")
    action = engine.select_action(obs_array)   # obs: (n_features,) float32

Requirements:
    pip install onnx onnxruntime-directml
    (do NOT have onnxruntime and onnxruntime-directml installed at the same time)
"""

import argparse
from typing import Any
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.inference_engines import BaseInferenceEngine


def torch_load_safe(path, map_location=None):
    """Load a checkpoint with weights_only=True, falling back for legacy pickles."""
    import pickle

    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except (torch.SerializationWarning, pickle.UnpicklingError, AttributeError,
            TypeError, KeyError, ValueError, ModuleNotFoundError):
        return torch.load(path, map_location=map_location, weights_only=False)


def _checkpoint_state_dict(checkpoint):
    """Return a state_dict from either rich or raw training checkpoints."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model_state", "state_dict"):
            state = checkpoint.get(key)
            if isinstance(state, dict):
                return state
    return checkpoint


def _read_training_config(ckpt_path: Path, model_name: str) -> dict:
    """Load the sidecar config emitted by training/train_gpu.py when present."""
    # Handle fold suffix in checkpoint names (e.g., haelt_fold0_best.pt -> haelt_fold0_config.json)
    stem = ckpt_path.stem  # e.g., "haelt_fold0_best" or "haelt_best"
    candidates = [
        ckpt_path.with_name(stem.replace("_best", "") + "_config.json"),
        ckpt_path.with_name(stem + "_config.json"),
        ckpt_path.parent / f"{model_name}_config.json",
    ]
    if "_fold" not in ckpt_path.stem:
        candidates.extend(sorted(ckpt_path.parent.glob(f"{model_name}_fold*_config.json")))

    for path in candidates:
        if path.exists():
            import json
            cfg = json.loads(path.read_text(encoding="utf-8"))
            cfg["_config_path"] = str(path)
            return cfg
    return {}


def _infer_features_from_state(state: dict) -> int | None:
    for k, v in state.items():
        shape = getattr(v, "shape", None)
        if shape is None or len(shape) < 2:
            continue
        if "embed" in k or "input" in k or "proj_in" in k or "proj.weight" in k:
            return int(shape[-1])
    return None


def _make_training_args(
    model_name: str,
    cfg: dict,
    state_dict: dict,
    n_features: int,
    seq_len: int,
) -> SimpleNamespace:
    state_keys = tuple(state_dict.keys()) if isinstance(state_dict, dict) else ()
    has_pair_embed = any("pair_embed" in k for k in state_keys)
    has_multitask = any("mt_head" in k or ".mt_head." in k for k in state_keys)

    n_pairs = int(cfg.get("_n_pairs") or cfg.get("n_pairs") or 1)
    explicit_fpp = cfg.get("_f_per_pair") or cfg.get("f_per_pair")
    f_per_pair = (
        int(explicit_fpp)
        if explicit_fpp is not None
        else max(1, n_features // max(1, n_pairs))
    )
    if (
        n_pairs == 1
        and has_pair_embed
        and f_per_pair > 0
        and n_features > f_per_pair
        and n_features % f_per_pair == 0
    ):
        n_pairs = max(1, n_features // f_per_pair)
        if explicit_fpp is None:
            f_per_pair = max(1, n_features // n_pairs)

    return SimpleNamespace(
        model=model_name,
        hidden_size=int(cfg.get("hidden_size", 256)),
        d_model=int(cfg.get("d_model", 256)),
        nhead=int(cfg.get("nhead", 8)),
        num_layers=int(cfg.get("num_layers", 3)),
        dropout=float(cfg.get("dropout", 0.1)),
        seq_len=int(cfg.get("seq_len", seq_len)),
        multitask=bool(cfg.get("multitask", has_multitask)),
        pair_embed_dim=int(cfg.get("pair_embed_dim", 16 if has_pair_embed else 0)),
        loss=str(cfg.get("loss", "cross_entropy")),
        corr_window=int(cfg.get("corr_window", 20)),
        corr_window_long=int(cfg.get("corr_window_long", 60)),
        momentum_window=int(cfg.get("momentum_window", 20)),
        _n_pairs=n_pairs,
        _f_per_pair=f_per_pair,
    )


def _wrap_logits_output(model):
    """ONNX Runtime live inference only needs direction logits."""
    import torch

    class _DirectionLogits(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            if isinstance(out, (tuple, list)):
                return out[0]
            if isinstance(out, dict):
                return out.get("direction_logits", out.get("logits", out.get("direction")))
            return out

    return _DirectionLogits(model)


def _scalar_to_direction_logits(score, threshold: float = 0.15):
    """
    Convert a scalar return/edge score into C++-compatible [sell, hold, buy]
    logits. Hold wins inside [-threshold, +threshold].
    """
    import torch

    score = score.reshape(-1)
    hold = torch.zeros_like(score)
    sell = -score - float(threshold)
    buy = score - float(threshold)
    return torch.stack([sell, hold, buy], dim=-1)


def _policy_to_direction_logits(policy_out):
    """Collapse RL policy outputs to [sell, hold, buy] logits."""
    import torch

    if isinstance(policy_out, (tuple, list)):
        policy_out = policy_out[0]
    q = policy_out.reshape(policy_out.shape[0], -1)
    if q.shape[-1] == 3:
        return q
    if q.shape[-1] == 1:
        return _scalar_to_direction_logits(q[:, 0])
    if q.shape[-1] < 10:
        padded = torch.full((q.shape[0], 10), -1.0e6, dtype=q.dtype, device=q.device)
        padded[:, : q.shape[-1]] = q
        q = padded

    # ScalingAction mapping:
    # 0 HOLD, 1 OPEN_LONG, 2 OPEN_SHORT, 3/4/5 SCALE_IN, 6/7/8 SCALE_OUT, 9 CLOSE_ALL.
    # Without live position state, SCALE_IN cannot choose a side — treat as HOLD
    # (matches trading.live_actions when position is flat). Do NOT boost both
    # buy and sell (that made scale-in look bullish and bearish at once).
    hold = torch.maximum(q[:, 0], torch.maximum(q[:, 9], q[:, 6:9].amax(dim=1)))
    hold = torch.maximum(hold, q[:, 3:6].amax(dim=1))
    sell = q[:, 2]
    buy = q[:, 1]
    return torch.stack([sell, hold, buy], dim=-1)


def _wrap_ensemble_logits(model, threshold: float = 0.15):
    """Wrap EnsembleMetaLearner scalar output as 3-class direction logits."""
    import torch

    class _EnsembleDirectionLogits(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return _scalar_to_direction_logits(out, threshold=threshold)

    return _EnsembleDirectionLogits(model)


def _wrap_rl_policy_logits(encoder, policy, obs_size: int):
    """Wrap frozen supervised encoder + RL policy as 3-class direction logits.

    Degraded path: portfolio state dims are unknown at export time, so they are
    zeroed. Prefer ``export_rl_execution_to_onnx`` / ``_wrap_rl_execution_policy``
    for live RL where position/uPnL/holding are available.
    """
    import torch

    class _RLDirectionLogits(torch.nn.Module):
        def __init__(self, encoder_model, policy_model, policy_obs_size: int):
            super().__init__()
            self.encoder = encoder_model
            self.policy = policy_model
            self.policy_obs_size = int(policy_obs_size)

        def forward(self, x):
            h = self.encoder(x)
            if h.ndim == 3:
                h = h[:, -1, :]
            h = h.reshape(h.shape[0], -1)
            n_state = max(0, self.policy_obs_size - h.shape[-1])
            if n_state:
                # Flat/neutral agent state — direction-only export cannot see position.
                state = torch.zeros(h.shape[0], n_state, dtype=h.dtype, device=h.device)
                obs = torch.cat([h, state], dim=-1)
            else:
                obs = h[:, : self.policy_obs_size]
            return _policy_to_direction_logits(self.policy(obs))

    return _RLDirectionLogits(encoder, policy, obs_size)


def _wrap_rl_execution_policy(encoder, policy, obs_size: int):
    """Wrap frozen encoder + portfolio state as native RL execution logits."""
    import torch

    class _RLExecutionPolicy(torch.nn.Module):
        def __init__(self, encoder_model, policy_model, policy_obs_size: int):
            super().__init__()
            self.encoder = encoder_model
            self.policy = policy_model
            self.policy_obs_size = int(policy_obs_size)

        def forward(self, features, agent_state):
            h = self.encoder(features)
            if h.ndim == 3:
                h = h[:, -1, :]
            h = h.reshape(h.shape[0], -1)
            state = agent_state.reshape(agent_state.shape[0], -1).to(dtype=h.dtype, device=h.device)
            obs = torch.cat([h, state], dim=-1)
            if obs.shape[-1] < self.policy_obs_size:
                pad = torch.zeros(
                    obs.shape[0],
                    self.policy_obs_size - obs.shape[-1],
                    dtype=obs.dtype,
                    device=obs.device,
                )
                obs = torch.cat([obs, pad], dim=-1)
            elif obs.shape[-1] > self.policy_obs_size:
                obs = obs[:, : self.policy_obs_size]
            out = self.policy(obs)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out.reshape(out.shape[0], -1)

    return _RLExecutionPolicy(encoder, policy, obs_size)


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT:  PyTorch checkpoint -> ONNX
# ─────────────────────────────────────────────────────────────────────────────

def _export_onnx(model, dummy, output_path, opset, input_names, output_name,
                 dynamic_axes=None):
    """Export with the default exporter, falling back to the legacy one if the
    model fails to load back into onnxruntime (torch >=2.9 dynamo exporter can
    emit opset-18+ attributes like Split.num_outputs even at lower opsets)."""
    import torch

    if dynamic_axes is None:
        dynamic_axes = {
            input_names[0]: {0: "batch_size"},
            output_name: {0: "batch_size"},
        }

    def _export(**extra):
        torch.onnx.export(
            model,
            dummy,
            output_path,
            opset_version=opset,
            input_names=input_names,
            output_names=[output_name],
            dynamic_axes=dynamic_axes,
            export_params=True,
            do_constant_folding=True,
            **extra,
        )

    _export()
    try:
        import onnxruntime as ort
        ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        return
    except Exception:
        pass
    _export(dynamo=False)


def core_onnx_export(
    model: "torch.nn.Module",
    n_features: int,
    seq_len: int,
    output_path: str,
    opset: int = 17,
    batch_size: int = 1,
    output_name: str = "logits",
    scaler: Any = None,
) -> str:
    """Core logic to export a loaded PyTorch model to ONNX.
    
    If scaler is provided (StandardScaler with mean_ and scale_ attributes),
    the scaler is fused into the ONNX graph so the exported model accepts
    raw features and internally applies normalization.
    """
    import torch
    model.eval()
    try:
        export_device = next(model.parameters()).device
    except StopIteration:
        export_device = torch.device("cpu")

    # Create a wrapper that includes the scaler if provided
    if scaler is not None:
        from inference._scaler_load import apply_inference_scaler
        import numpy as np
        
        # Get scaler parameters
        mean = getattr(scaler, "mean_", None)
        scale = getattr(scaler, "scale_", None)
        
        if mean is not None and scale is not None:
            # Create scaler parameters as tensors
            mean_tensor = torch.from_numpy(np.asarray(mean, dtype=np.float32)).to(export_device)
            scale_tensor = torch.from_numpy(np.asarray(scale, dtype=np.float32)).to(export_device)
            
            # Avoid division by zero
            scale_tensor = torch.where(scale_tensor == 0, torch.ones_like(scale_tensor), scale_tensor)
            
            class ScaledModel(torch.nn.Module):
                def __init__(self, model, mean, scale):
                    super().__init__()
                    self.model = model
                    # Register as buffers so they're included in ONNX
                    self.register_buffer("mean", mean)
                    self.register_buffer("scale", scale)
                
                def forward(self, x):
                    # x: (batch, seq_len, n_features)
                    # NaN/Inf sanitization FIRST (same as training)
                    x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
                    # Apply z-score normalization: (x - mean) / scale
                    x = (x - self.mean) / self.scale
                    return self.model(x)
            
            model = ScaledModel(model, mean_tensor, scale_tensor)
            print(f"[Export] Fused StandardScaler into ONNX graph (n_features={len(mean)})")

    # Dummy input
    dummy = torch.randn(batch_size, seq_len, n_features, device=export_device)
    # Handle GNN models (different input signature)
    if hasattr(model, "n_nodes"):
        dummy = torch.randn(batch_size, 6, n_features // 6, device=export_device)

    _export_onnx(model, dummy, output_path, opset, ["features"], output_name)
    print(f"[Export] Saved ONNX model: {output_path}")
    print(f"         n_features={n_features} | seq_len={seq_len} | opset={opset}")
    _print_onnx_info(output_path)
    return output_path


def core_rl_execution_onnx_export(
    model: "torch.nn.Module",
    n_features: int,
    seq_len: int,
    output_path: str,
    opset: int = 17,
    batch_size: int = 1,
    agent_state_size: int = 5,
    scaler: Any = None,
) -> str:
    """Export a two-input execution policy: features + agent_state -> action logits.
    
    If scaler is provided (StandardScaler with mean_ and scale_ attributes),
    the scaler is fused into the ONNX graph so the exported model accepts
    raw features and internally applies normalization.
    """
    import torch

    model.eval()
    try:
        export_device = next(model.parameters()).device
    except StopIteration:
        export_device = torch.device("cpu")

    # Create a wrapper that includes the scaler if provided
    if scaler is not None:
        import numpy as np
        
        # Get scaler parameters
        mean = getattr(scaler, "mean_", None)
        scale = getattr(scaler, "scale_", None)
        
        if mean is not None and scale is not None:
            # Create scaler parameters as tensors
            mean_tensor = torch.from_numpy(np.asarray(mean, dtype=np.float32)).to(export_device)
            scale_tensor = torch.from_numpy(np.asarray(scale, dtype=np.float32)).to(export_device)
            
            # Avoid division by zero
            scale_tensor = torch.where(scale_tensor == 0, torch.ones_like(scale_tensor), scale_tensor)
            
            class ScaledRLModel(torch.nn.Module):
                def __init__(self, model, mean, scale):
                    super().__init__()
                    self.model = model
                    # Register as buffers so they're included in ONNX
                    self.register_buffer("mean", mean)
                    self.register_buffer("scale", scale)
                
                def forward(self, features, agent_state):
                    # features: (batch, seq_len, n_features)
                    # NaN/Inf sanitization FIRST (same as training)
                    features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
                    # Apply z-score normalization: (features - mean) / scale
                    features = (features - self.mean) / self.scale
                    return self.model(features, agent_state)
            
            model = ScaledRLModel(model, mean_tensor, scale_tensor)
            print(f"[Export] Fused StandardScaler into RL execution ONNX graph (n_features={len(mean)})")

    dummy_features = torch.randn(batch_size, seq_len, n_features, device=export_device)
    dummy_state = torch.zeros(batch_size, int(agent_state_size), device=export_device)
    _export_onnx(
        model,
        (dummy_features, dummy_state),
        output_path,
        opset,
        ["features", "agent_state"],
        "action_logits",
        dynamic_axes={
            "features": {0: "batch_size"},
            "agent_state": {0: "batch_size"},
            "action_logits": {0: "batch_size"},
        },
    )
    print(f"[Export] Saved RL execution ONNX model: {output_path}")
    print(
        f"         n_features={n_features} | seq_len={seq_len} | "
        f"agent_state={agent_state_size} | opset={opset}"
    )
    _print_onnx_info(output_path)
    return output_path


def export_to_onnx(
    checkpoint_path: str,
    model_name: str,
    seq_len: int,
    output_path: str | None = None,
    opset: int = 17,
    n_features: int | None = None,
) -> str:
    """
    Load a trained .pt checkpoint and export to ONNX.

    Args:
        checkpoint_path: Path to *_best.pt or *_calibrated.pt checkpoint.
        model_name:      Architecture name (haelt | tft | transformer | mamba | gnn | expert).
        seq_len:         Input sequence length used during training.
        output_path:     Destination .onnx path. Defaults to same dir as checkpoint.
        opset:           ONNX opset version (17 recommended for transformer ops).
        n_features:      Number of input features; required for some older checkpoints.

    Returns:
        Path to the exported .onnx file.
    """
    import torch

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load checkpoint
    ckpt = torch_load_safe(ckpt_path, map_location="cpu")

    # Reconstruct model. Prefer the training sidecar config because best.pt is
    # often a raw state_dict from the exact training builder.
    cfg = _read_training_config(ckpt_path, model_name)
    state_dict = _checkpoint_state_dict(ckpt)
    cfg_features = cfg.get("n_features")
    if cfg_features is not None:
        cfg_features = int(cfg_features)
        if n_features is not None and int(n_features) != cfg_features:
            print(
                f"[Export] WARN: --n-feat={n_features} differs from "
                f"{Path(cfg.get('_config_path', 'config')).name} n_features={cfg_features}; "
                "using checkpoint config."
            )
        n_features = cfg_features
    elif n_features is None and isinstance(ckpt, dict):
        n_features = ckpt.get("n_features", None)

    if n_features is None:
        # Try to infer from the first weight tensor
        n_features = _infer_features_from_state(state_dict)

    if n_features is None:
        config_path = ckpt_path.parent / f"{ckpt_path.stem.replace('_best','')}_config.json"
        if config_path.exists():
            import json
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            n_features = int(cfg.get("n_features")) if cfg.get("n_features") else None

    if n_features is None:
        raise ValueError(
            "Cannot infer n_features from checkpoint. "
            "Pass --n-feat explicitly or retrain with a newer checkpoint."
        )

    if cfg:
        from training.train_gpu import build_model as build_training_model
        model_args = _make_training_args(model_name, cfg, state_dict, int(n_features), seq_len)
        seq_len = int(model_args.seq_len)
        if model_name == "ensemble":
            import json as _json
            meta_json_path = ckpt_path.parent / "ensemble_meta_final.json"
            if meta_json_path.exists():
                meta_data = _json.loads(meta_json_path.read_text())
                base_names = meta_data.get("meta", {}).get("base_names", [])
            else:
                base_names = ["haelt", "tft", "transformer", "mamba", "gnn", "expert"]
            from models.ensemble import EnsembleMetaLearner
            loaded_bases = []
            for b_name in base_names:
                b_ckpt = next((p for p in [ckpt_path.parent.parent / b_name / f"{b_name}_best.pt", ckpt_path.parent.parent / f"{b_name}_best.pt"] if p.exists()), None)
                if b_ckpt:
                    b_model = build_training_model(b_name, int(n_features), model_args)
                    b_state = torch_load_safe(b_ckpt, map_location="cpu")
                    if isinstance(b_state, dict) and "model_state_dict" in b_state:
                        b_state = b_state["model_state_dict"]
                    elif isinstance(b_state, dict) and "state_dict" in b_state:
                        b_state = b_state["state_dict"]
                    b_model.load_state_dict(b_state, strict=False)
                    loaded_bases.append(b_model)
            if not loaded_bases:
                raise FileNotFoundError(
                    f"No ensemble base checkpoints found next to {ckpt_path.parent}"
                )
            model = EnsembleMetaLearner(
                loaded_bases,
                context_dim=32,
                hidden=64,
                base_names=base_names,
            )
        else:
            model = build_training_model(model_name, int(n_features), model_args)
    else:
        from models.architectures import build_model
        model = build_model(model_name, input_size=int(n_features), seq_len=seq_len)

    model.load_state_dict(state_dict, strict=False)
    model = _wrap_logits_output(model)
    model.eval()

    # Handle TemperatureScaler wrapper (calibrated checkpoints)
    if "temperature" in ckpt:
        from models.architectures import TemperatureScaler
        model = TemperatureScaler(model)
        model.temperature.data.fill_(float(ckpt["temperature"]))

    # Load scaler for ONNX fusion
    scaler = None
    cache_path = ckpt_path.parent
    try:
        from inference._scaler_load import load_inference_scaler
        scaler = load_inference_scaler(cache_path)
        if scaler is not None:
            print(f"[Export] Found scaler at {cache_path}/scaler.npz, fusing into ONNX")
        else:
            print(f"[Export] No scaler found at {cache_path}/scaler.npz, exporting without normalization")
    except Exception as e:
        print(f"[Export] Could not load scaler: {e}")

    if output_path is None:
        output_path = str(ckpt_path.with_suffix(".onnx"))

    return core_onnx_export(
        model=model,
        n_features=n_features,
        seq_len=seq_len,
        output_path=output_path,
        opset=opset,
        batch_size=1,
        output_name="logits",
        scaler=scaler,
    )


def export_ensemble_to_onnx(
    checkpoint_path: str,
    checkpoint_dir: str,
    seq_len: int,
    n_features: int,
    output_path: str | None = None,
    opset: int = 17,
    device: str = "cpu",
) -> str:
    """Export EnsembleMetaLearner as one C++-compatible 3-logit ONNX graph."""
    import json

    import torch

    from models.ensemble import EnsembleMetaLearner
    from scripts.train_ensemble_meta import load_base_model, resolve_checkpoint

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Ensemble checkpoint not found: {ckpt_path}")

    meta_path = ckpt_path.with_suffix(ckpt_path.suffix + ".json")
    meta_doc = {}
    if meta_path.exists():
        meta_doc = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = meta_doc.get("meta", meta_doc)
    base_names = list(meta.get("base_names") or [])
    if not base_names:
        raise ValueError(
            f"Could not determine ensemble base_names from {meta_path}. "
            "Retrain the ensemble meta-learner with current metadata."
        )

    dev = torch.device(device)
    bases = []
    cfgs = {}
    ckpt_dir = Path(checkpoint_dir)
    for name in base_names:
        base_ckpt = resolve_checkpoint(str(name), ckpt_dir)
        if base_ckpt is None:
            raise FileNotFoundError(f"Missing base checkpoint for ensemble member {name}")
        base, cfg = load_base_model(str(name), base_ckpt, int(n_features), int(seq_len), dev)
        base.eval()
        bases.append(base)
        cfgs[name] = cfg

    model = EnsembleMetaLearner(
        bases,
        context_dim=int(meta.get("context_dim", 32)),
        hidden=int(meta.get("hidden", 64)),
        base_names=base_names,
    ).to(dev)
    state = torch_load_safe(ckpt_path, map_location=dev)
    if isinstance(state, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            if isinstance(state.get(key), dict):
                state = state[key]
                break
    model.load_state_dict(state, strict=False)
    wrapped = _wrap_ensemble_logits(model).to(dev)
    wrapped.eval()

    # Load scaler for ONNX fusion
    scaler = None
    try:
        from inference._scaler_load import load_inference_scaler
        # Use checkpoint_dir as cache path
        scaler = load_inference_scaler(Path(checkpoint_dir))
        if scaler is not None:
            print(f"[Export] Found scaler, fusing into ONNX")
        else:
            print(f"[Export] No scaler found, exporting without normalization")
    except Exception as e:
        print(f"[Export] Could not load scaler: {e}")

    if output_path is None:
        output_path = str(ckpt_path.with_suffix(".onnx"))

    return core_onnx_export(
        model=wrapped,
        n_features=int(n_features),
        seq_len=int(seq_len),
        output_path=output_path,
        opset=opset,
        batch_size=1,
        output_name="logits",
        scaler=scaler,
    )


def export_rl_to_onnx(
    rl_checkpoint: str,
    supervised_checkpoint: str,
    model_name: str,
    seq_len: int,
    n_features: int,
    output_path: str | None = None,
    opset: int = 17,
    algo: str = "dqn",
    device: str = "cpu",
) -> str:
    """Export frozen supervised encoder + RL policy as a 3-logit ONNX graph.

    Direction-only / degraded: agent state is zeroed. For execution parity with
    training and ``RLInferenceAgent``, use ``export_rl_execution_to_onnx``.
    """
    import json

    import torch

    from config.settings import RL
    from inference.pytorch_inference import load_pytorch_model
    from models.rl_agents import DQNAgent, PPOAgent
    from training.train_gpu import _core_model

    rl_path = Path(rl_checkpoint)
    if not rl_path.exists():
        raise FileNotFoundError(f"RL checkpoint not found: {rl_path}")

    dev = torch.device(device)
    sup_model, n_features, seq_len, arch_name, _scaler = load_pytorch_model(
        supervised_checkpoint,
        model_name,
        seq_len=int(seq_len),
        n_features=int(n_features),
        device=dev,
    )
    core = _core_model(sup_model)
    inner = core.inner if hasattr(core, "inner") else core
    encoder = inner.backbone if hasattr(inner, "backbone") else inner
    if hasattr(encoder, "head"):
        encoder.head = torch.nn.Identity()
    encoder.eval()

    meta_path = rl_path.parent / f"rl_{str(algo).lower()}_best.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    obs_size = int(meta.get("obs_size", 0) or 0)
    if obs_size <= 0:
        with torch.no_grad():
            dummy = torch.zeros(1, int(seq_len), int(n_features), device=dev)
            h = encoder(dummy)
            if h.ndim == 3:
                h = h[:, -1, :]
            obs_size = int(h.reshape(1, -1).shape[-1]) + 5

    algo_name = str(algo).lower()
    algo_kw = dict(RL.get(algo_name, {}))
    state = torch_load_safe(rl_path, map_location=dev)
    if algo_name == "dqn":
        n_actions = int(meta.get("n_actions", 0) or 0)
        if n_actions <= 0:
            n_actions = int(next(reversed(state.values())).shape[0]) if isinstance(state, dict) and state else 3
        agent = DQNAgent(obs_size=obs_size, n_actions=n_actions, device=str(dev), **algo_kw)
        agent.policy_net.load_state_dict(state, strict=False)
        policy = agent.policy_net
    else:
        n_actions = int(meta.get("n_actions", 0) or 0) or 3
        if isinstance(state, dict):
            for key, value in state.items():
                if key.endswith("actor.weight") and getattr(value, "ndim", 0) == 2:
                    n_actions = int(value.shape[0])
                    break
        agent = PPOAgent(obs_size=obs_size, n_actions=n_actions, device=str(dev), **algo_kw)
        agent.net.load_state_dict(state, strict=False)
        policy = agent.net

    wrapped = _wrap_rl_policy_logits(encoder, policy, obs_size).to(dev)
    wrapped.eval()

    # Use the scaler from load_pytorch_model for ONNX fusion
    scaler = _scaler
    if scaler is not None:
        print(f"[Export] Found scaler, fusing into ONNX")
    else:
        print(f"[Export] No scaler found, exporting without normalization")

    if output_path is None:
        output_path = str(rl_path.with_suffix(".onnx"))

    return core_onnx_export(
        model=wrapped,
        n_features=int(n_features),
        seq_len=int(seq_len),
        output_path=output_path,
        opset=opset,
        batch_size=1,
        output_name="logits",
        scaler=scaler,
    )


def export_rl_execution_to_onnx(
    rl_checkpoint: str,
    supervised_checkpoint: str,
    model_name: str,
    seq_len: int,
    n_features: int,
    output_path: str | None = None,
    opset: int = 17,
    algo: str = "dqn",
    device: str = "cpu",
) -> str:
    """Export frozen supervised encoder + RL policy as native 10-action execution ONNX."""
    import json

    import torch

    from config.settings import RL
    from inference.pytorch_inference import load_pytorch_model
    from models.rl_agents import DQNAgent, PPOAgent
    from training.train_gpu import _core_model

    rl_path = Path(rl_checkpoint)
    if not rl_path.exists():
        raise FileNotFoundError(f"RL checkpoint not found: {rl_path}")

    dev = torch.device(device)
    sup_model, n_features, seq_len, arch_name, _scaler = load_pytorch_model(
        supervised_checkpoint,
        model_name,
        seq_len=int(seq_len),
        n_features=int(n_features),
        device=dev,
    )
    core = _core_model(sup_model)
    inner = core.inner if hasattr(core, "inner") else core
    encoder = inner.backbone if hasattr(inner, "backbone") else inner
    if hasattr(encoder, "head"):
        encoder.head = torch.nn.Identity()
    encoder.eval()

    meta_path = rl_path.parent / f"rl_{str(algo).lower()}_best.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    obs_size = int(meta.get("obs_size", 0) or 0)
    if obs_size <= 0:
        with torch.no_grad():
            dummy = torch.zeros(1, int(seq_len), int(n_features), device=dev)
            h = encoder(dummy)
            if h.ndim == 3:
                h = h[:, -1, :]
            obs_size = int(h.reshape(1, -1).shape[-1]) + 5

    algo_name = str(algo).lower()
    algo_kw = dict(RL.get(algo_name, {}))
    state = torch_load_safe(rl_path, map_location=dev)
    if algo_name == "dqn":
        n_actions = int(meta.get("n_actions", 0) or 0) or 10
        if int(meta.get("n_actions", 0) or 0) <= 0 and isinstance(state, dict):
            for key, value in reversed(list(state.items())):
                if getattr(value, "ndim", 0) in (1, 2) and (
                    key.endswith("net.4.bias") or key.endswith("net.4.weight")
                ):
                    n_actions = int(value.shape[0])
                    break
        agent = DQNAgent(obs_size=obs_size, n_actions=n_actions, device=str(dev), **algo_kw)
        agent.policy_net.load_state_dict(state, strict=False)
        policy = agent.policy_net
    else:
        n_actions = int(meta.get("n_actions", 0) or 0) or 10
        if int(meta.get("n_actions", 0) or 0) <= 0 and isinstance(state, dict):
            for key, value in state.items():
                if key.endswith("actor.weight") and getattr(value, "ndim", 0) == 2:
                    n_actions = int(value.shape[0])
                    break
        agent = PPOAgent(obs_size=obs_size, n_actions=n_actions, device=str(dev), **algo_kw)
        agent.net.load_state_dict(state, strict=False)
        policy = agent.net

    wrapped = _wrap_rl_execution_policy(encoder, policy, obs_size).to(dev)
    wrapped.eval()
    if output_path is None:
        output_path = str(rl_path.with_name(f"{rl_path.stem}_execution.onnx"))

    return core_rl_execution_onnx_export(
        model=wrapped,
        n_features=int(n_features),
        seq_len=int(seq_len),
        output_path=output_path,
        opset=opset,
        batch_size=1,
        agent_state_size=5,
        scaler=_scaler,
    )


def _print_onnx_info(onnx_path: str):
    """Print ONNX graph summary (input/output shapes, node count)."""
    try:
        import onnx
        m = onnx.load(onnx_path)
        onnx.checker.check_model(m)
        inputs  = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
                   for i in m.graph.input]
        outputs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
                   for o in m.graph.output]
        n_nodes = len(m.graph.node)
        print(f"         Inputs:  {inputs}")
        print(f"         Outputs: {outputs}")
        print(f"         Nodes:   {n_nodes}")
    except Exception as e:
        print(f"         (onnx info skipped: {e})")


def verify_onnx_scaler(
    checkpoint_path: str,
    model_name: str,
    seq_len: int,
    n_features: int | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    num_samples: int = 10,
) -> bool:
    """
    Verify that ONNX scaler fusion matches PyTorch ScaledModel output.
    
    Exports the model to ONNX with scaler fusion, then compares outputs
    between PyTorch (ScaledModel) and ONNX Runtime for random inputs.
    
    Args:
        checkpoint_path: Path to .pt checkpoint
        model_name: Model architecture name
        seq_len: Input sequence length
        n_features: Number of input features (inferred if None)
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        num_samples: Number of random test inputs
        
    Returns:
        True if all outputs match within tolerance, False otherwise
    """
    import torch
    import numpy as np
    import onnxruntime as ort
    
    print(f"[Verify] Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint and model (reuse existing logic)
    from inference.onnx_inference import export_to_onnx, core_onnx_export, _load_pytorch_model
    
    # Load model and scaler via existing function
    model, n_feat, seq_len_, arch_name, scaler = _load_pytorch_model(
        checkpoint_path, model_name, seq_len, n_features
    )
    
    if scaler is None:
        print("[Verify] No scaler found - skipping scaler fusion verification")
        return True
    
    # Create ScaledModel wrapper (same as core_onnx_export)
    from inference._scaler_load import apply_inference_scaler
    import numpy as np
    
    mean = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)
    
    if mean is None or scale is None:
        print("[Verify] Scaler missing mean_ or scale_")
        return False
    
    mean_tensor = torch.from_numpy(np.asarray(mean, dtype=np.float32))
    scale_tensor = torch.from_numpy(np.asarray(scale, dtype=np.float32))
    scale_tensor = torch.where(scale_tensor == 0, torch.ones_like(scale_tensor), scale_tensor)
    
    class ScaledModel(torch.nn.Module):
        def __init__(self, model, mean, scale):
            super().__init__()
            self.model = model
            self.register_buffer("mean", mean)
            self.register_buffer("scale", scale)
        
        def forward(self, x):
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            x = (x - self.mean) / self.scale
            return self.model(x)
    
    scaled_model = ScaledModel(model, mean_tensor, scale_tensor)
    scaled_model.eval()
    
    # Export to temporary ONNX
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        core_onnx_export(
            model=scaled_model,
            n_features=n_feat,
            seq_len=seq_len_,
            output_path=tmp_path,
            opset=17,
            batch_size=1,
            output_name="logits",
            scaler=None,  # Already fused in ScaledModel
        )
        
        # Create ONNX Runtime session
        sess = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        
        print(f"[Verify] Running {num_samples} random test samples...")
        max_diff = 0.0
        all_passed = True
        
        for i in range(num_samples):
            # Generate random input
            x_np = np.random.randn(1, seq_len_, n_feat).astype(np.float32)
            x_torch = torch.from_numpy(x_np)
            
            # PyTorch forward
            with torch.no_grad():
                pt_out = scaled_model(x_torch).numpy()
            
            # ONNX Runtime forward
            ort_out = sess.run([output_name], {input_name: x_np})[0]
            
            # Compare
            diff = np.abs(pt_out - ort_out).max()
            rel_diff = diff / (np.abs(pt_out).max() + 1e-8)
            max_diff = max(max_diff, diff)
            
            passed = np.allclose(pt_out, ort_out, rtol=rtol, atol=atol)
            if not passed:
                all_passed = False
                print(f"  Sample {i+1}: FAILED (max_diff={diff:.6f}, rel_diff={rel_diff:.6f})")
                print(f"    PT range: [{pt_out.min():.4f}, {pt_out.max():.4f}]")
                print(f"    ORT range: [{ort_out.min():.4f}, {ort_out.max():.4f}]")
            else:
                print(f"  Sample {i+1}: OK (max_diff={diff:.6f})")
        
        print(f"[Verify] Max absolute difference: {max_diff:.6f}")
        print(f"[Verify] Result: {'PASSED' if all_passed else 'FAILED'}")
        return all_passed
        
    finally:
        try:
            import os
            os.unlink(tmp_path)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE ENGINE:  ONNX Runtime + DirectML
# ─────────────────────────────────────────────────────────────────────────────

class DirectMLInferenceEngine(BaseInferenceEngine):
    """
    ONNX Runtime inference engine targeting AMD hardware via DirectML.

    Provider priority:
        1. DmlExecutionProvider  — AMD NPU / GPU (Windows, DirectML)
        2. CUDAExecutionProvider — fallback to NVIDIA if available
        3. CPUExecutionProvider  — final fallback

    Interface matches trained PyTorch models and LiveTradingEngine:
        engine.select_action(obs)  -> model class int (0=sell, 1=hold, 2=buy)
        engine.predict_proba(obs)  -> np.ndarray shape (3,)

    Args:
        onnx_path:     Path to exported .onnx file.
        seq_len:       Sequence length the model was exported with.
        device_id:     DirectML device index (0 = primary AMD adapter).
        prefer_cpu:    Force CPU execution (disables DirectML/CUDA).
        hold_threshold: Minimum max-class probability to act; below -> hold.
    """

    def __init__(
        self,
        onnx_path:       str,
        seq_len:         int  = 60,
        device_id:       int  = 0,
        prefer_cpu:      bool = False,
        hold_threshold:  float = 0.45,
    ):
        import onnxruntime as ort

        self.onnx_path      = str(onnx_path)
        self.seq_len        = seq_len
        self.hold_threshold = hold_threshold
        self._obs_buffer    = deque(maxlen=int(seq_len))   # rolling window of feature vectors

        # Build provider list
        available = ort.get_available_providers()
        if prefer_cpu:
            providers = ["CPUExecutionProvider"]
        else:
            providers = []
            if "DmlExecutionProvider" in available:
                providers.append(
                    ("DmlExecutionProvider", {"device_id": device_id})
                )
            if "CUDAExecutionProvider" in available:
                providers.append(
                    ("CUDAExecutionProvider", {"device_id": device_id})
                )
            providers.append("CPUExecutionProvider")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.intra_op_num_threads     = 4

        self.session   = ort.InferenceSession(self.onnx_path, opts, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Report active provider
        active = self.session.get_providers()[0]
        ep_label = {
            "DmlExecutionProvider":  "AMD DirectML (NPU/GPU)",
            "CUDAExecutionProvider": "NVIDIA CUDA",
            "CPUExecutionProvider":  "CPU",
        }.get(active, active)
        print(f"[DirectML] Session ready | EP: {ep_label} | {Path(onnx_path).name}")

        # Warm up (first call is slow due to JIT graph compilation)
        self._warmup()

    # ── public API ────────────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray) -> int:
        """
        Append obs to rolling buffer and return action when buffer is full.

        Args:
            obs: 1-D float32 array of length n_features (one bar's features).

        Returns:
            int: model class 0=sell, 1=hold, 2=buy.
                 Returns 1 (hold) until seq_len bars have been seen.
        """
        self._obs_buffer.append(obs.astype(np.float32))
        if len(self._obs_buffer) < self.seq_len:
            return 1  # hold until buffer is full

        proba = self.predict_proba(np.array(self._obs_buffer, dtype=np.float32))
        if proba.max() < self.hold_threshold:
            return 1
        return int(proba.argmax())

    def predict_proba(self, window: np.ndarray) -> np.ndarray:
        """
        Run ONNX inference on a (seq_len, n_features) window.

        Args:
            window: float32 array of shape (seq_len, n_features).

        Returns:
            Softmax probabilities, shape (3,) — [sell, hold, buy].
        """
        x = window[np.newaxis].astype(np.float32)  # (1, seq_len, n_feat)
        logits = self.session.run([self.output_name], {self.input_name: x})[0]
        logits = logits[0]  # (3,) or (1,)

        # Regression head: single value -> convert to buy/hold/sell proba
        if logits.shape[-1] == 1 or logits.ndim == 0:
            v = float(logits.flat[0])
            if v > 0.15:
                return np.array([0.1, 0.2, 0.7], dtype=np.float32)
            if v < -0.15:
                return np.array([0.7, 0.2, 0.1], dtype=np.float32)
            return np.array([0.1, 0.8, 0.1], dtype=np.float32)

        # Softmax
        logits = logits.astype(np.float64)
        e = np.exp(logits - logits.max())
        return (e / e.sum()).astype(np.float32)

    def reset_buffer(self):
        """Clear the rolling observation buffer (call on session restart)."""
        self._obs_buffer.clear()

    # ── internal ──────────────────────────────────────────────────────────────

    def _warmup(self):
        """Run one dummy inference to trigger JIT graph compilation."""
        # Infer n_features from model input shape
        shape = self.session.get_inputs()[0].shape
        n_feat = shape[2] if len(shape) >= 3 and isinstance(shape[2], int) else 64
        dummy  = np.zeros((1, self.seq_len, n_feat), dtype=np.float32)
        t0     = time.perf_counter()
        self.session.run([self.output_name], {self.input_name: dummy})
        ms = (time.perf_counter() - t0) * 1000
        print(f"[DirectML] Warm-up done in {ms:.1f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK:  CUDA vs DirectML vs CPU
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(
    onnx_path: str,
    seq_len:   int = 60,
    n_feat:    int = 128,
    runs:      int = 500,
):
    """
    Measure single-sample inference latency across all available execution providers.
    Prints a comparison table.
    """
    import onnxruntime as ort

    dummy = np.random.randn(1, seq_len, n_feat).astype(np.float32)

    available = ort.get_available_providers()
    candidates = [
        ("AMD DirectML",  "DmlExecutionProvider"),
        ("NVIDIA CUDA",   "CUDAExecutionProvider"),
        ("CPU",           "CPUExecutionProvider"),
    ]

    print(f"\n{'Provider':<22} {'Status':<12} {'Median ms':>10} {'P99 ms':>10} {'Throughput':>14}")
    print("─" * 72)

    for label, ep in candidates:
        if ep not in available and ep != "CPUExecutionProvider":
            print(f"{label:<22} {'not available':<12}")
            continue

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            sess  = ort.InferenceSession(onnx_path, opts, providers=[ep, "CPUExecutionProvider"])
            iname = sess.get_inputs()[0].name
            oname = sess.get_outputs()[0].name

            # Warm up
            for _ in range(10):
                sess.run([oname], {iname: dummy})

            # Timed runs
            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                sess.run([oname], {iname: dummy})
                times.append((time.perf_counter() - t0) * 1000)

            times   = sorted(times)
            median  = times[len(times) // 2]
            p99     = times[int(len(times) * 0.99)]
            tput    = 1000.0 / median
            print(f"{label:<22} {'ok':<12} {median:>10.3f} {p99:>10.3f} {tput:>12.0f}/s")

        except Exception as e:
            print(f"{label:<22} {'error':<12}  {e}")

    print()
    print("Note: for 1-min bar trading (1 inference/min) any provider is sufficient.")
    print("      DirectML wins on sustained power efficiency for 24/7 operation.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ONNX export + AMD DirectML inference")
    sub = ap.add_subparsers(dest="cmd")

    # export
    exp = sub.add_parser("export", help="Export .pt checkpoint to ONNX")
    exp.add_argument("--checkpoint", required=True, help="Path to *_best.pt")
    exp.add_argument("--model",      required=True,
                     choices=["haelt","tft","transformer","mamba","gnn","expert"])
    exp.add_argument("--seq-len",    type=int, default=60)
    exp.add_argument("--n-feat",     type=int, default=None,
                     help="Number of input features used during training")
    exp.add_argument("--output",     default=None, help="Output .onnx path")
    exp.add_argument("--opset",      type=int, default=17)

    ens = sub.add_parser("export-ensemble", help="Export EnsembleMetaLearner to one 3-logit ONNX")
    ens.add_argument("--checkpoint", required=True, help="Path to ensemble_meta_best.pt")
    ens.add_argument("--checkpoint-dir", default="checkpoints", help="Directory containing base checkpoints")
    ens.add_argument("--seq-len", type=int, default=60)
    ens.add_argument("--n-feat", type=int, required=True)
    ens.add_argument("--output", default=None)
    ens.add_argument("--opset", type=int, default=17)
    ens.add_argument("--device", default="cpu")

    rlx = sub.add_parser("export-rl", help="Export supervised encoder + RL policy to one 3-logit ONNX")
    rlx.add_argument("--rl-checkpoint", required=True, help="Path to rl_<algo>_best.pt")
    rlx.add_argument("--supervised-checkpoint", required=True, help="Path to supervised *_best.pt")
    rlx.add_argument("--model", required=True, choices=["haelt","tft","transformer","mamba","gnn","expert"])
    rlx.add_argument("--algo", default="dqn", choices=["dqn","ppo"])
    rlx.add_argument("--seq-len", type=int, default=60)
    rlx.add_argument("--n-feat", type=int, required=True)
    rlx.add_argument("--output", default=None)
    rlx.add_argument("--opset", type=int, default=17)
    rlx.add_argument("--device", default="cpu")

    rle = sub.add_parser("export-rl-execution", help="Export RL policy to native 10-action execution ONNX")
    rle.add_argument("--rl-checkpoint", required=True, help="Path to rl_<algo>_best.pt")
    rle.add_argument("--supervised-checkpoint", required=True, help="Path to supervised *_best.pt")
    rle.add_argument("--model", required=True, choices=["haelt","tft","transformer","mamba","gnn","expert"])
    rle.add_argument("--algo", default="dqn", choices=["dqn","ppo"])
    rle.add_argument("--seq-len", type=int, default=60)
    rle.add_argument("--n-feat", type=int, required=True)
    rle.add_argument("--output", default=None)
    rle.add_argument("--opset", type=int, default=17)
    rle.add_argument("--device", default="cpu")

    # benchmark
    bch = sub.add_parser("benchmark", help="Compare provider latencies")
    bch.add_argument("--onnx",    required=True)
    bch.add_argument("--seq-len", type=int, default=60)
    bch.add_argument("--n-feat",  type=int, default=128)
    bch.add_argument("--runs",    type=int, default=500)

    # test  — quick smoke test of the full pipeline
    tst = sub.add_parser("test", help="Smoke test: export + inference + benchmark")
    tst.add_argument("--checkpoint", required=True)
    tst.add_argument("--model",      required=True,
                     choices=["haelt","tft","transformer","mamba","gnn","expert"])
    tst.add_argument("--seq-len",    type=int, default=60)

    # verify — verify ONNX scaler fusion matches PyTorch
    vrf = sub.add_parser("verify", help="Verify ONNX scaler fusion matches PyTorch output")
    vrf.add_argument("--checkpoint", required=True, help="Path to *_best.pt")
    vrf.add_argument("--model",      required=True,
                     choices=["haelt","tft","transformer","mamba","gnn","expert"])
    vrf.add_argument("--seq-len",    type=int, default=60)
    vrf.add_argument("--n-feat",     type=int, default=None, help="Number of input features")
    vrf.add_argument("--rtol",       type=float, default=1e-4, help="Relative tolerance")
    vrf.add_argument("--atol",       type=float, default=1e-6, help="Absolute tolerance")
    vrf.add_argument("--samples",    type=int, default=10, help="Number of test samples")

    args = ap.parse_args()

    if args.cmd == "export":
        export_to_onnx(
            checkpoint_path=args.checkpoint,
            model_name=args.model,
            seq_len=args.seq_len,
            output_path=args.output,
            opset=args.opset,
            n_features=args.n_feat,
        )

    elif args.cmd == "export-ensemble":
        export_ensemble_to_onnx(
            checkpoint_path=args.checkpoint,
            checkpoint_dir=args.checkpoint_dir,
            seq_len=args.seq_len,
            n_features=args.n_feat,
            output_path=args.output,
            opset=args.opset,
            device=args.device,
        )

    elif args.cmd == "export-rl":
        export_rl_to_onnx(
            rl_checkpoint=args.rl_checkpoint,
            supervised_checkpoint=args.supervised_checkpoint,
            model_name=args.model,
            seq_len=args.seq_len,
            n_features=args.n_feat,
            output_path=args.output,
            opset=args.opset,
            algo=args.algo,
            device=args.device,
        )

    elif args.cmd == "export-rl-execution":
        export_rl_execution_to_onnx(
            rl_checkpoint=args.rl_checkpoint,
            supervised_checkpoint=args.supervised_checkpoint,
            model_name=args.model,
            seq_len=args.seq_len,
            n_features=args.n_feat,
            output_path=args.output,
            opset=args.opset,
            algo=args.algo,
            device=args.device,
        )

    elif args.cmd == "benchmark":
        benchmark(
            onnx_path=args.onnx,
            seq_len=args.seq_len,
            n_feat=args.n_feat,
            runs=args.runs,
        )

    elif args.cmd == "test":
        onnx_path = export_to_onnx(
            checkpoint_path=args.checkpoint,
            model_name=args.model,
            seq_len=args.seq_len,
        )
        engine = DirectMLInferenceEngine(onnx_path, seq_len=args.seq_len)
        shape  = engine.session.get_inputs()[0].shape
        n_feat = shape[2] if len(shape) >= 3 and isinstance(shape[2], int) else 64

        # Simulate seq_len bars of features
        print(f"\n[Test] Simulating {args.seq_len} bars of features (n_feat={n_feat})...")
        for i in range(args.seq_len):
            obs    = np.random.randn(n_feat).astype(np.float32)
            action = engine.select_action(obs)
        label = {0: "SELL", 1: "HOLD", 2: "BUY"}[action]
        print(f"[Test] Final action after {args.seq_len} bars: {label}")

        benchmark(onnx_path, seq_len=args.seq_len, n_feat=n_feat, runs=200)

    elif args.cmd == "verify":
        ok = verify_onnx_scaler(
            checkpoint_path=args.checkpoint,
            model_name=args.model,
            seq_len=args.seq_len,
            n_features=args.n_feat,
            rtol=args.rtol,
            atol=args.atol,
            num_samples=args.samples,
        )
        if not ok:
            raise SystemExit("[Verify] FAILED - scaler fusion mismatch detected")
        print("[Verify] SUCCESS - all outputs match within tolerance")

    else:
        ap.print_help()
