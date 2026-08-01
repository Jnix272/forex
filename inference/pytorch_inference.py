"""
inference/pytorch_inference.py
================================
PyTorch/CUDA live inference with the same select_action() interface as
DirectMLInferenceEngine and training checkpoints (production_best.pt / *_best.pt).
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.onnx_inference import (
    _checkpoint_state_dict,
    _infer_features_from_state,
    _make_training_args,
    _read_training_config,
    _wrap_logits_output,
)
from trading.inference_engines import BaseInferenceEngine


def _resolve_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _logits_to_proba(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if logits.size == 1:
        v = float(logits[0])
        if v > 0.15:
            return np.array([0.1, 0.2, 0.7], dtype=np.float32)
        if v < -0.15:
            return np.array([0.7, 0.2, 0.1], dtype=np.float32)
        return np.array([0.1, 0.8, 0.1], dtype=np.float32)
    e = np.exp(logits - logits.max())
    return (e / e.sum()).astype(np.float32)


def _load_ensemble_manifest(ckpt_path: Path) -> dict:
    manifest_path = ckpt_path.parent / "ensemble_manifest.json"
    if manifest_path.exists():
        try:
            import json

            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def load_pytorch_model(
    checkpoint_path: str,
    model_name: str,
    seq_len: int = 60,
    n_features: Optional[int] = None,
    device: Optional[Any] = None,
):
    """Rebuild and load weights using the same path as ONNX export."""
    import torch

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = _read_training_config(ckpt_path, model_name)
    sidecar_model = str(cfg.get("model") or model_name).lower().strip()
    ensemble_manifest = _load_ensemble_manifest(ckpt_path) if sidecar_model == "ensemble" else {}
    manifest_schema = ensemble_manifest.get("schema", {}) if isinstance(ensemble_manifest, dict) else {}
    if sidecar_model == "ensemble":
        n_features = int(manifest_schema.get("n_features", n_features or 0)) or n_features
        seq_len = int(manifest_schema.get("seq_len", seq_len))
    
    if sidecar_model == "xgboost":
        # Skip torch.load entirely for XGBoost JSON models
        from models.xgboost_model import XGBoostForecaster
        n_features = int(cfg.get("n_features", n_features or 64))
        seq_len = int(cfg.get("seq_len", seq_len))
        model = XGBoostForecaster(num_classes=1, flatten_sequence=False)
        model.load_model(str(ckpt_path))
        return model, n_features, seq_len, sidecar_model

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _checkpoint_state_dict(ckpt)

    cfg_features = cfg.get("n_features")
    if cfg_features is not None:
        n_features = int(cfg_features)
    elif n_features is None and isinstance(ckpt, dict):
        n_features = ckpt.get("n_features")
    if n_features is None:
        n_features = _infer_features_from_state(state_dict)
    if n_features is None:
        raise ValueError(
            f"Cannot infer n_features for {ckpt_path.name}. "
            "Pass --n-feat or retrain with sidecar config."
        )

    if sidecar_model == "ensemble":
        from models.ensemble import EnsembleMetaLearner
        from scripts.train_ensemble_meta import load_base_model, resolve_checkpoint

        manifest = ensemble_manifest
        meta = cfg.get("meta", cfg)
        base_models = manifest.get("base_models") or []
        base_names = [str(b.get("name")) for b in base_models if b.get("name")]
        if not base_names:
            base_names = list(meta.get("base_names") or ["haelt", "mamba", "gnn"])

        dev = device or _resolve_device()
        bases = []
        ckpt_root = ckpt_path.parent.parent
        for name in base_names:
            manifest_ckpt = next(
                (
                    Path(str(b.get("checkpoint")))
                    for b in base_models
                    if str(b.get("name")) == name and b.get("checkpoint")
                ),
                None,
            )
            if manifest_ckpt is not None and not manifest_ckpt.is_absolute():
                manifest_ckpt = (ckpt_path.parent / manifest_ckpt).resolve()
            base_ckpt = manifest_ckpt if manifest_ckpt is not None and manifest_ckpt.exists() else resolve_checkpoint(name, ckpt_root)
            if base_ckpt is None:
                raise FileNotFoundError(f"Missing base checkpoint for ensemble member {name}")
            base, _ = load_base_model(name, base_ckpt, int(n_features), int(seq_len), dev)
            base.eval()
            bases.append(base)

        model = EnsembleMetaLearner(
            bases,
            context_dim=int(meta.get("context_dim", 32)),
            hidden=int(meta.get("hidden", 64)),
            base_names=base_names,
        ).to(dev)
        model.load_state_dict(state_dict, strict=False)
        model = _wrap_logits_output(model)
        model.eval()
        return model, int(n_features), int(seq_len), sidecar_model

    if cfg:
        from training.train_gpu import build_model as build_training_model

        model_args = _make_training_args(
            sidecar_model, cfg, state_dict, int(n_features), seq_len
        )
        seq_len = int(model_args.seq_len)
        model = build_training_model(sidecar_model, int(n_features), model_args)
    else:
        from models.architectures import build_model

        model = build_model(sidecar_model, input_size=int(n_features), seq_len=seq_len)

    model.load_state_dict(state_dict, strict=False)
    model = _wrap_logits_output(model)

    if isinstance(ckpt, dict) and "temperature" in ckpt:
        from models.architectures import TemperatureScaler

        model = TemperatureScaler(model)
        model.temperature.data.fill_(float(ckpt["temperature"]))

    dev = device or _resolve_device()
    model.to(dev)
    model.eval()
    return model, int(n_features), int(seq_len), sidecar_model


class PyTorchInferenceEngine(BaseInferenceEngine):
    """Rolling-window PyTorch inference for LiveTradingEngine."""

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str,
        seq_len: int = 60,
        n_features: Optional[int] = None,
        hold_threshold: float = 0.45,
        device: Optional[Any] = None,
    ):
        import torch

        self.checkpoint_path = str(checkpoint_path)
        self.model_name = str(model_name).lower().strip()
        self.hold_threshold = float(hold_threshold)
        self._obs_buffer = deque(maxlen=int(seq_len))
        self.device = device or _resolve_device()

        self.model, self.n_features, self.seq_len, self.arch_name = load_pytorch_model(
            self.checkpoint_path,
            self.model_name,
            seq_len=seq_len,
            n_features=n_features,
            device=self.device,
        )
        self._obs_buffer = deque(self._obs_buffer, maxlen=int(self.seq_len))
        self._torch = torch
        print(
            f"[PyTorch] Loaded {Path(checkpoint_path).name} | "
            f"arch={self.arch_name} | seq={self.seq_len} | "
            f"n_feat={self.n_features} | device={self.device}"
        )
        self._warmup()

    def select_action(self, obs: np.ndarray) -> int:
        self._obs_buffer.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        if len(self._obs_buffer) < self.seq_len:
            return 1
        proba = self.predict_proba(np.stack(self._obs_buffer, axis=0))
        if proba.max() < self.hold_threshold:
            return 1
        return int(proba.argmax())

    def predict_proba(self, window: np.ndarray) -> np.ndarray:
        x = self._torch.as_tensor(
            window[np.newaxis], dtype=self._torch.float32, device=self.device
        )
        with self._torch.no_grad():
            logits = self.model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if hasattr(logits, "detach"):
                logits = logits.detach().cpu().numpy()
        return _logits_to_proba(np.asarray(logits)[0])

    def reset_buffer(self):
        self._obs_buffer.clear()

    def reload(
        self,
        checkpoint_path: Optional[str] = None,
        model_name: Optional[str] = None,
        seq_len: Optional[int] = None,
        n_features: Optional[int] = None,
    ) -> None:
        path = checkpoint_path or self.checkpoint_path
        name = model_name or self.model_name
        self.model, self.n_features, self.seq_len, self.arch_name = load_pytorch_model(
            path,
            name,
            seq_len=seq_len or self.seq_len,
            n_features=n_features or self.n_features,
            device=self.device,
        )
        self.checkpoint_path = str(path)
        self.model_name = name
        self.reset_buffer()
        print(f"[PyTorch] Hot-reloaded {Path(path).name}")

    def _warmup(self):
        dummy = self._torch.zeros(
            (1, self.seq_len, self.n_features),
            dtype=self._torch.float32,
            device=self.device,
        )
        t0 = time.perf_counter()
        with self._torch.no_grad():
            _ = self.model(dummy)
        ms = (time.perf_counter() - t0) * 1000
        print(f"[PyTorch] Warm-up done in {ms:.1f} ms")


def build_pytorch_engine(
    pt_path: Path,
    model_name: str,
    seq_len: int = 60,
    n_features: Optional[int] = None,
) -> PyTorchInferenceEngine:
    return PyTorchInferenceEngine(
        checkpoint_path=str(pt_path),
        model_name=model_name,
        seq_len=seq_len,
        n_features=n_features,
    )
