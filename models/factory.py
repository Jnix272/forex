"""Model construction helpers (shared by training and inference).

Moved here from ``training/model_factory.py`` to break the
training <-> inference import cycle; the training module is now a thin
re-export shim. See docs/CONTINUE.md.
"""

from __future__ import annotations

from models.architectures import build_model


def _core_model(model):
    """Unwrap DDP or torch.compile wrappers to get the core nn.Module."""
    if hasattr(model, "module"):
        return model.module
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model


def _strict_load_report(model, state_dict, context_name="Model", min_frac_loaded=0.6):
    """Return a dict summary of how much of a checkpoint loaded.

    The helper is called by training/pretrain/warm-start flows that expect a
    reporting object with ``frac_loaded`` and per-key counts, not a raw
    ``(missing, unexpected)`` tuple.
    """
    state_dict = dict(state_dict)
    target_state = model.state_dict()
    missing = [k for k in target_state if k not in state_dict]
    unexpected = [k for k in state_dict if k not in target_state]
    def _param_shape(p):
        if getattr(p, "_is_uninitialized", False) or p.__class__.__name__ == "UninitializedParameter":
            return None
        try:
            return tuple(p.shape)
        except Exception:
            return None

    def _safe_numel(p, fallback_tensor=None):
        if getattr(p, "_is_uninitialized", False) or p.__class__.__name__ == "UninitializedParameter":
            return fallback_tensor.numel() if fallback_tensor is not None else 0
        try:
            return p.numel()
        except Exception:
            return 0

    shape_mismatch = [
        k
        for k in target_state
        if k in state_dict
        and _param_shape(target_state[k]) is not None
        and tuple(state_dict[k].shape) != _param_shape(target_state[k])
    ]

    try:
        _missing, _unexpected = model.load_state_dict(state_dict, strict=False)
        missing = list(_missing)
        unexpected = list(_unexpected)
    except RuntimeError:
        # ``load_state_dict`` raises on shape mismatches under strict=False when
        # a key is present with the wrong tensor shape; record those separately.
        pass

    loaded_keys = {
        key
        for key in target_state
        if key in state_dict
        and (_param_shape(target_state[key]) is None or tuple(state_dict[key].shape) == _param_shape(target_state[key]))
    }
    n_target = sum(_safe_numel(t, state_dict.get(k)) for k, t in target_state.items())
    n_loaded = sum(_safe_numel(target_state[k], state_dict.get(k)) for k in loaded_keys)
    frac_loaded = float(n_loaded / n_target) if n_target else 1.0

    if missing or unexpected or shape_mismatch:
        print(
            f"[{context_name}] Load Warning: {len(missing)} missing, "
            f"{len(unexpected)} unexpected, {len(shape_mismatch)} shape-mismatch keys."
        )

    report = {
        "frac_loaded": frac_loaded,
        "n_loaded": n_loaded,
        "n_target": n_target,
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
        "min_frac_loaded": min_frac_loaded,
        "passed": frac_loaded >= float(min_frac_loaded),
    }
    return report


def _multitask_head_in(model) -> int:
    """Get the input dimension of the multitask head (if present)."""
    if hasattr(model, "multitask_head") and model.multitask_head is not None:
        return model.multitask_head[0].in_features
    return getattr(model, "hidden_size", 128)


__all__ = [
    "_core_model",
    "_multitask_head_in",
    "_strict_load_report",
    "build_model",
]
