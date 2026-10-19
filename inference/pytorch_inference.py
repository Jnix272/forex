"""
inference/pytorch_inference.py
================================
PyTorch/CUDA live inference with the same select_action() interface as
DirectMLInferenceEngine and training checkpoints (production_best.pt / *_best.pt).
"""

from __future__ import annotations

import os
import warnings
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, cast

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference._scaler_load import (
    apply_inference_scaler,
    load_inference_scaler,
    scaler_feature_count,
)
from inference.onnx_inference import (
    _checkpoint_state_dict,
    _infer_features_from_state,
    _make_training_args,
    _read_training_config,
    _wrap_logits_output,
    torch_load_safe,
)
from trading.inference_engines import BaseInferenceEngine


def _resolve_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _logits_to_proba(logits: np.ndarray, threshold: float | None = None) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if logits.size == 1:
        v = float(logits[0])
        if threshold is None:
            try:
                threshold = float(os.getenv("PREDICTION_THRESHOLD", "0.35"))
            except Exception:
                threshold = 0.35
        sell = -v - float(threshold)
        hold = 0.0
        buy = v - float(threshold)
        logits = np.array([sell, hold, buy], dtype=np.float64)
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
    n_features: int | None = None,
    device: Any | None = None,
    cache_path: str | None = None,
):
    """Rebuild and load weights using the same path as ONNX export.

    Also loads the training-time ``StandardScaler`` (when discoverable) and
    returns it as the 5th tuple element.  This closes the train/live contract
    gap where the supervised pipeline z-scores features but inference fed raw
    features to the model (audit 2026-08-07, finding #1).

    The scaler is discovered via, in order:

    1. an explicit ``cache_path`` argument from the caller, or
    2. a ``cache_path`` key in the sidecar ``{checkpoint}_config.json`` file
       that the training loop writes alongside the checkpoint.

    Returns
    -------
    ``(model, n_features, seq_len, arch_name, scaler)`` - the scaler is
    ``None`` when no scaler file was found (RL encoder-only path / demo).
    """

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
        scaler = None  # FIX: scaler was undefined
        return model, n_features, seq_len, sidecar_model, scaler

    ckpt = torch_load_safe(ckpt_path, map_location="cpu")
    state_dict = _checkpoint_state_dict(ckpt)

    cfg_features = cfg.get("n_features")
    if cfg_features is not None:
        n_features = int(cfg_features)
    elif n_features is None and isinstance(ckpt, dict):
        n_features = ckpt.get("n_features")
    if n_features is None:
        n_features = _infer_features_from_state(state_dict)
    if n_features is None:
        raise ValueError(f"Cannot infer n_features for {ckpt_path.name}. Pass --n-feat or retrain with sidecar config.")

    # ── Train/live scaler + schema-hash parity ──────────────────────────────
    # Audit 2026-08-07, finding #1: the supervised pipeline z-scores features
    # but inference fed raw features to the model. We now load the persisted
    # StandardScaler (when discoverable) and assert that the scaler's
    # n_features matches the sidecar's n_features - schema-hash drift is a
    # hard failure.
    resolved_cache_path = (
        cache_path
        or cfg.get("cache_path")
        or (
            ensemble_manifest.get("training", {}).get("cache_path")
            if isinstance(ensemble_manifest, dict)
            else None
        )
    )
    # Prefer the train-only scaler saved beside the checkpoint (exact training transform).
    scaler = None
    _ckpt_scaler = ckpt_path.with_name(ckpt_path.stem + "_scaler.npz")
    if _ckpt_scaler.is_file():
        scaler = load_inference_scaler(_ckpt_scaler)
    if scaler is None:
        scaler = load_inference_scaler(resolved_cache_path)
    if scaler is not None:
        scaler_n = scaler_feature_count(scaler)
        if scaler_n is not None and int(scaler_n) != int(n_features):
            # No "tile" allowance: a single-pair scaler (e.g. 146 cols) on a 4-pair
            # model (584) would put EURUSD statistics on every pair's columns.
            raise RuntimeError(
                f"Inference scaler/schema contract mismatch: scaler.n_features_in_"
                f"={scaler_n} but checkpoint n_features={n_features}. The cache's "
                "feature-mask has drifted from when this checkpoint was trained - "
                "either re-train against the new cache or rebuild the cache with "
                "the original feature_mask."
            )

    if sidecar_model == "ensemble":
        from models.ensemble import EnsembleMetaLearner, load_base_scaler
        from scripts.train_ensemble_meta import load_base_model, resolve_checkpoint

        manifest = ensemble_manifest
        meta = cfg.get("meta", cfg)
        base_models = manifest.get("base_models") or []
        base_names = [str(b.get("name")) for b in base_models if b.get("name")]
        if not base_names:
            base_names = list(meta.get("base_names") or ["haelt", "mamba", "gnn"])

        dev = device or _resolve_device()
        bases = []
        base_scalers = []
        base_seq_lens = []
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
            base_ckpt = (
                manifest_ckpt
                if manifest_ckpt is not None and manifest_ckpt.exists()
                else resolve_checkpoint(name, ckpt_root)
            )
            if base_ckpt is None:
                raise FileNotFoundError(f"Missing base checkpoint for ensemble member {name}")
            base, base_cfg = load_base_model(name, base_ckpt, int(n_features), int(seq_len), dev)
            base.eval()
            bases.append(base)
            base_scalers.append(load_base_scaler(base_ckpt))
            base_seq_lens.append(int((base_cfg or {}).get("seq_len", seq_len)))

        ensemble_ctor = cast(Any, EnsembleMetaLearner)
        model = ensemble_ctor(
            bases,
            context_dim=int(meta.get("context_dim", 32)),
            hidden=int(meta.get("hidden", 64)),
            base_names=base_names,
            base_seq_lens=base_seq_lens,  # same per-base windows as in meta training
        ).to(dev)
        model.load_state_dict(state_dict, strict=False)
        if all(sc is not None for sc in base_scalers):
            # Each base scales its own input (as in meta training); the caller
            # must pass raw features, so no outer scaler.
            model.attach_base_scalers(base_scalers)
            model.to(dev)
            scaler = None
        else:
            warnings.warn(
                "Ensemble bases lack *_scaler.npz sidecars; bases will see inputs scaled with the "
                "cache scaler, which may differ from their training transform.",
                stacklevel=2,
            )
        model = _wrap_logits_output(model)
        model.eval()
        return model, int(n_features), int(seq_len), sidecar_model, scaler

    if cfg:
        from models.factory import build_model as build_training_model

        model_args = _make_training_args(sidecar_model, cfg, state_dict, int(n_features), seq_len)
        seq_len = int(model_args.seq_len)
        model = cast(
            Any,
            build_training_model(
                sidecar_model,
                int(n_features),
                seq_len=int(model_args.seq_len),
                **{k: v for k, v in vars(model_args).items() if k != "seq_len"},
            ),
        )
    else:
        from models.architectures import build_model

        model = cast(Any, build_model(sidecar_model, input_size=int(n_features), seq_len=seq_len))

    model.load_state_dict(state_dict, strict=False)
    model = _wrap_logits_output(model)

    if isinstance(ckpt, dict) and "temperature" in ckpt:
        from models.architectures import TemperatureScaler

        temp_ctor = cast(Any, TemperatureScaler)
        model = temp_ctor(model)
        model.temperature.data.fill_(float(ckpt["temperature"]))

    dev = device or _resolve_device()
    model = cast(Any, model).to(dev)
    model.eval()

    # Discover and attach feature mask from metadata sidecar if present
    meta_json = Path(checkpoint_path).with_suffix(Path(checkpoint_path).suffix + ".metadata.json")
    if not meta_json.is_file():
        meta_json = Path(checkpoint_path).with_name(Path(checkpoint_path).name + ".metadata.json")
    if not meta_json.is_file():
        for cand_dir in (Path(checkpoint_path).parent, Path(checkpoint_path).parent.parent):
            if cand_dir.is_dir():
                for pat in ("*refit*.metadata.json", "*best*.metadata.json", "*.metadata.json"):
                    metas = list(cand_dir.glob(pat))
                    if metas:
                        meta_json = metas[0]
                        break
            if meta_json.is_file():
                break
    if meta_json.is_file():
        try:
            import json
            import torch

            mdata = json.loads(meta_json.read_text(encoding="utf-8"))
            fmask_raw = mdata.get("feature_ablation", {}).get("mask")
            if fmask_raw:
                fmask_np = np.asarray(fmask_raw, dtype=np.float32)
                fmask_t = torch.from_numpy(fmask_np).to(dev)
                setattr(model, "feature_mask", fmask_t)
                if hasattr(model, "inner"):
                    setattr(model.inner, "feature_mask", fmask_t)
        except Exception:
            pass

    return model, int(n_features), int(seq_len), sidecar_model, scaler


class PyTorchInferenceEngine(BaseInferenceEngine):
    """Rolling-window PyTorch inference for LiveTradingEngine."""

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str,
        seq_len: int = 60,
        n_features: int | None = None,
        hold_threshold: float = 0.45,
        device: Any | None = None,
        cache_path: str | None = None,
        model: Any | None = None,
        scaler: Any | None = None,
    ):
        import torch

        self.checkpoint_path = str(checkpoint_path)
        self.model_name = str(model_name).lower().strip()
        self.hold_threshold = float(hold_threshold)
        self._obs_buffer = deque(maxlen=int(seq_len))
        self.device = device or _resolve_device()
        self._torch = torch

        if model is not None:
            self.model = _wrap_logits_output(model).to(self.device)
            self.model.eval()
            self.n_features = int(n_features or 146)
            self.seq_len = int(seq_len)
            self.arch_name = self.model_name
            self.scaler = scaler
            self.feature_mask = getattr(model, "feature_mask", None)
        else:
            self.model, self.n_features, self.seq_len, self.arch_name, self.scaler = load_pytorch_model(
                self.checkpoint_path,
                self.model_name,
                seq_len=seq_len,
                n_features=n_features,
                device=self.device,
                cache_path=cache_path,
            )
            self.feature_mask = getattr(self.model, "feature_mask", None)
        self._obs_buffer = deque(self._obs_buffer, maxlen=int(self.seq_len))
        self.pair = None
        self.pair_index = 0
        self.per_pair_heads = bool(
            int(self.n_features) == 146
            or any("MultiPair" in type(mod).__name__ for mod in self.model.modules())
        )
        print(
            f"[PyTorch] Loaded {Path(checkpoint_path).name} | "
            f"arch={self.arch_name} | seq={self.seq_len} | "
            f"n_feat={self.n_features} | scaler={'yes' if self.scaler is not None else 'no'}"
            f" | mask={'yes' if self.feature_mask is not None else 'no'}"
            f" | device={self.device}"
        )
        self._warmup()

    def _transform_window(self, window: np.ndarray) -> np.ndarray:
        """Apply the training-time StandardScaler and feature ablation mask to a (T, F) live window.

        Mirrors :func:`training.gpu_datasets._decompress_block` ordering:
        NaN/Inf cleanup → scaler.transform → feature_mask → float32 cast.
        """
        arr = np.asarray(window, dtype=np.float32)
        # Handle multi-pair window (584) fed to single-pair specialist (146)
        if arr.shape[-1] > self.n_features:
            if arr.shape[-1] == 584 and self.n_features == 146:
                p_idx = int(getattr(self, "pair_index", 0))
                arr = arr[..., p_idx * 146 : (p_idx + 1) * 146]
            else:
                arr = arr[..., : self.n_features]
        if self.scaler is not None:
            scaler_n = getattr(self.scaler, "n_features_in_", None)
            if scaler_n is not None and arr.shape[-1] == scaler_n * 4:
                blocks = []
                for p_idx in range(4):
                    blk = arr[..., p_idx * scaler_n : (p_idx + 1) * scaler_n]
                    blocks.append(apply_inference_scaler(self.scaler, blk))
                arr = np.concatenate(blocks, axis=-1)
            else:
                arr = apply_inference_scaler(self.scaler, arr)
        if self.feature_mask is not None:
            mask_np = self.feature_mask.detach().cpu().numpy() if hasattr(self.feature_mask, "detach") else np.asarray(self.feature_mask)
            if mask_np.shape[-1] == arr.shape[-1]:
                arr = arr * mask_np
        return arr

    def select_action(self, obs: np.ndarray) -> int:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if arr.shape[0] > self.n_features:
            if arr.shape[0] == 584 and self.n_features == 146:
                p_idx = int(getattr(self, "pair_index", 0))
                arr = arr[p_idx * 146 : (p_idx + 1) * 146]
            else:
                arr = arr[: self.n_features]
        self._obs_buffer.append(arr)
        if len(self._obs_buffer) < self.seq_len:
            return 1
        proba = self.predict_proba(np.stack(self._obs_buffer, axis=0))
        if proba.max() < self.hold_threshold:
            return 1
        return int(proba.argmax())

    def predict_proba(self, window: np.ndarray) -> np.ndarray:
        window = np.asarray(window, dtype=np.float32)
        if window.ndim == 1:
            window = np.repeat(window[np.newaxis, :], self.seq_len, axis=0)
        scaled = self._transform_window(window)
        x = self._torch.as_tensor(scaled[np.newaxis], dtype=self._torch.float32, device=self.device)
        core = getattr(self.model, "inner", self.model)
        with self._torch.no_grad():
            logits = core(x)
            if isinstance(logits, (tuple, list)) and len(logits) >= 3:
                # Multitask heads: same [SELL, HOLD, BUY] rule as validation and
                # the gate (training.decision); select_action then applies
                # hold_threshold to it.
                from models.ensemble import PRIMARY_PAIR_INDEX
                from training.decision import action_proba

                pr = action_proba(logits).detach().cpu().numpy()[0]
                idx = getattr(self, "pair_index", PRIMARY_PAIR_INDEX)
                if pr.ndim == 2:
                    idx = min(idx, pr.shape[0] - 1)
                    return pr[idx].astype(np.float32)
                return pr.astype(np.float32)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if hasattr(logits, "detach"):
                logits = logits.detach().cpu().numpy()
        row = logits
        if isinstance(row, (tuple, list)):
            row = row[0]
        if hasattr(row, "detach"):
            row = row.detach().cpu().numpy()
        row = np.asarray(row)
        if row.ndim == 0:
            row = row.reshape(1)
        elif row.ndim == 1 and row.size == 1:
            row = row
        elif row.ndim >= 2:
            row = row[0]
        if row.ndim == 1 and row.size > 1 and row.size != 3 and getattr(self.model, "_per_pair_output", True):
            # Per-pair heads: (P,) direction logits, one per pair. Trade the target pair.
            from models.ensemble import PRIMARY_PAIR_INDEX

            idx = getattr(self, "pair_index", PRIMARY_PAIR_INDEX)
            idx = min(idx, row.size - 1)
            row = row[idx : idx + 1]
        return _logits_to_proba(row, threshold=getattr(self, "threshold", None))

    def for_pair(self, pair: str) -> "PyTorchInferenceEngine":
        """Return an inference engine specialized for `pair`.

        Discovers dedicated single-pair specialist checkpoints for the active
        architecture (e.g. checkpoints/{arch}_{pair}_v1/...), falling back to
        production checkpoints or HAELT specialist checkpoints if available.
        Otherwise returns an isolated clone configured for the pair slot.
        """
        pair_clean = pair.upper().replace("/", "").replace("_", "")
        if getattr(self, "pair", None) == pair_clean:
            return self

        pair_order = ["EURUSD", "USDJPY", "GBPUSD", "USDCAD"]
        idx = pair_order.index(pair_clean) if pair_clean in pair_order else 0

        m_name = getattr(self, "model_name", "haelt").lower()
        candidates = [
            Path(f"checkpoints/{m_name}_{pair_clean.lower()}_v1/{m_name}/{m_name}_best.pt"),
            Path(f"checkpoints/{m_name}_{pair_clean.lower()}_v1/{m_name}/{m_name}/{m_name}_best.pt"),
            Path(f"checkpoints/{m_name}_{pair_clean.lower()}_v1/{m_name}_best.pt"),
            Path(f"checkpoints/{m_name}_{pair_clean.lower()}_v1/{m_name}_foldrefit_best.pt"),
            Path(f"checkpoints/{m_name}_{pair_clean.lower()}_v1/{m_name}_foldrefit_calibrated.pt"),
            Path(f"checkpoints/production_{m_name}_{pair_clean.lower()}.pt"),
            Path(f"checkpoints/{m_name}_{pair_clean.lower()}_best.pt"),
        ]
        if m_name != "haelt":
            candidates.extend([
                Path(f"checkpoints/haelt_{pair_clean.lower()}_v1/haelt/haelt_best.pt"),
                Path(f"checkpoints/haelt_{pair_clean.lower()}_v1/haelt/haelt/haelt_best.pt"),
                Path(f"checkpoints/haelt_{pair_clean.lower()}_v1/haelt/haelt_foldrefit_best.pt"),
                Path(f"checkpoints/haelt_{pair_clean.lower()}_v1/haelt/haelt_foldrefit_calibrated.pt"),
            ])
        candidates.extend([
            Path(f"checkpoints/production_{pair_clean.lower()}.pt"),
            Path(f"checkpoints/{pair_clean.lower()}_best.pt"),
        ])
        for cand in candidates:
            if cand.is_file():
                try:
                    cand_model = self.model_name
                    if "haelt_" in str(cand).lower() and m_name != "haelt":
                        cand_model = "haelt"
                    engine = PyTorchInferenceEngine(
                        checkpoint_path=str(cand),
                        model_name=cand_model,
                        seq_len=self.seq_len,
                        n_features=None,
                        hold_threshold=self.hold_threshold,
                        device=self.device,
                    )
                    engine.pair = pair_clean
                    engine.pair_index = idx
                    print(f"[SpecialistRouter] Successfully routed {pair_clean} -> Dedicated Specialist ({cand})")
                    return engine
                except Exception as exc:
                    print(f"[SpecialistRouter] Note: Candidate {cand} for {pair_clean} not loadable ({exc})")

        # Multi-pair or in-memory fallback: configure pair head index
        if self.checkpoint_path == ":memory:" or not Path(self.checkpoint_path).is_file():
            import copy
            cloned = copy.copy(self)
            cloned._obs_buffer = deque(maxlen=int(self.seq_len))
            cloned.pair = pair_clean
            cloned.pair_index = idx
            return cloned

        cloned = PyTorchInferenceEngine(
            checkpoint_path=self.checkpoint_path,
            model_name=self.model_name,
            seq_len=self.seq_len,
            n_features=self.n_features,
            hold_threshold=self.hold_threshold,
            device=self.device,
        )
        cloned.pair = pair_clean
        cloned.pair_index = idx
        return cloned

    def reset_buffer(self):
        self._obs_buffer.clear()

    def reload(
        self,
        checkpoint_path: str | None = None,
        model_name: str | None = None,
        seq_len: int | None = None,
        n_features: int | None = None,
        cache_path: str | None = None,
    ) -> None:
        path = checkpoint_path or self.checkpoint_path
        name = model_name or self.model_name
        self.model, self.n_features, self.seq_len, self.arch_name, self.scaler = load_pytorch_model(
            path,
            name,
            seq_len=seq_len or self.seq_len,
            n_features=n_features or self.n_features,
            device=self.device,
            cache_path=cache_path,
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
    n_features: int | None = None,
) -> PyTorchInferenceEngine:
    return PyTorchInferenceEngine(
        checkpoint_path=str(pt_path),
        model_name=model_name,
        seq_len=seq_len,
        n_features=n_features,
    )
