"""Model construction helpers for GPU training.

See docs/CONTINUE.md."""
from __future__ import annotations

import torch.nn as nn

from models.architectures import (
    EXPERTEncoder,
    GNNFromSequence,
    HAELTHybrid,
    MambaScalper,
    MODEL_ROLES,
    MultiPairWrapper,
    MultiTaskWrapper,
    TFTScalper,
    iTransformerScalper,
    GLMBaseline,
)
from config.model_training_profile import (
    ModelTrainingProfile,
    get_training_profile,
    MODEL_PROFILES,
)
from training.cache_integrity import _resolve_pair_feat_indices

_HOST = None
_BOUND = False
_HOST_DEPS = (
    '_log_error',
    '_log_warn',
    '_log_info',
    '_get_pairs',
)


def bind_host(host_mod) -> None:
    global _HOST, _BOUND
    _HOST = host_mod
    g = globals()
    for name in _HOST_DEPS:
        if hasattr(host_mod, name):
            g[name] = getattr(host_mod, name)
    _BOUND = True


def _ensure_bound() -> None:
    import training.train_gpu as tg
    bind_host(tg)

def _core_model(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel / torch.compile wrappers before state_dict save/load."""
    m = model
    if hasattr(m, "module"):
        m = m.module
    # torch.compile wraps as OptimizedModule with weights under _orig_mod
    while hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m

def _multitask_head_in(model_name: str, args, n_features: int) -> int:
    """
    Return the dimensionality of each backbone's pre-head hidden state.
    This is the input size for MultiTaskWrapper / MultiTaskHead.

    Derived from each architecture's forward() ΓÇö the tensor fed into self.head
    before the MultiTaskWrapper replaces it with nn.Identity().
    """
    m = model_name.lower()
    if m == "tft":
        return args.hidden_size
    # iTransformer: mean-pool over variate tokens → d_model (not F*d_model)
    if m == "transformer":
        return int(args.d_model)
    # HAELT: cat([lstm_feat, trf_feat]) of size lstm_hidden + d_model
    if m == "haelt":
        return (args.hidden_size // 2) + (args.d_model // 2)
    if m == "mamba":
        return args.d_model
    # GNNFromSequence: h.reshape(B, hidden * n_nodes)
    if m == "gnn":
        return args.hidden_size * 6
    if m == "expert":
        return args.d_model
    if m == "glm":
        return getattr(args, "seq_len", 16) * n_features
    return args.hidden_size

def _format_param_count(model: nn.Module) -> str:
    total = 0
    skipped = 0
    for p in model.parameters():
        try:
            total += p.numel()
        except ValueError:
            skipped += 1
    suffix = "+" if skipped else ""
    return f"{total / 1e6:.2f}{suffix}M"

def _is_uninitialized_parameter(param: object) -> bool:
    try:
        from torch.nn.parameter import UninitializedParameter

        return isinstance(param, UninitializedParameter)
    except Exception:
        try:
            param.numel()
            return False
        except ValueError:
            return True

def build_model(name: str, n_features: int, args) -> nn.Module:
    # -- Multi-pair embedding expansion ----------------------------------------
    _ensure_bound()
    n_pairs     = getattr(args, "_n_pairs", 1)
    f_per_pair  = getattr(args, "_f_per_pair", n_features)
    embed_dim   = getattr(args, "pair_embed_dim", 0)
    use_pair_emb = n_pairs > 1 and embed_dim > 0

    # Backbone input width when MultiPairWrapper is active:
    #   pairs_flat  = n_pairs * (f_per_pair + embed_dim)
    #   n_cross     = n_pairs*(n_pairs-1)//2
    #   n_inter     = 3*n_cross (RelMom+ShortCorr+LongCorr) + n_pairs (VolShare) + 2 (Disp+Conf)
    # The wrapper concatenates pairs_flat and cross features before handing off to backbone,
    # so the backbone must be built with the combined width.
    if use_pair_emb:
        _n_cross       = n_pairs * (n_pairs - 1) // 2
        _n_interaction = 3 * _n_cross + n_pairs + 2
        backbone_input = n_pairs * (f_per_pair + embed_dim) + _n_interaction
    else:
        backbone_input = n_features

    # Multitask wrapper adds its own 3-class head; base always uses nc=1
    multitask = getattr(args, "multitask", False)
    nc = 1 if multitask else (3 if args.loss == "cross_entropy" else 1)
    core_name = name.replace("baseline_", "")
    builders = {
        "tft":         lambda: TFTScalper(
                           input_size=backbone_input, hidden=args.hidden_size,
                           heads=min(8,args.nhead), lstm_layers=args.num_layers,
                           dropout=args.dropout, num_classes=nc),
        "transformer": lambda: iTransformerScalper(
                           input_size=backbone_input, seq_len=args.seq_len,
                           d_model=args.d_model, nhead=args.nhead,
                           num_layers=args.num_layers,
                           dim_ff=getattr(args, "dim_ff", args.d_model * 2),
                           dropout=args.dropout,
                           num_classes=nc),
        "haelt":       lambda: HAELTHybrid(
                           input_size=backbone_input, seq_len=args.seq_len,
                           lstm_hidden=args.hidden_size//2, d_model=args.d_model//2,
                           nhead=max(2,args.nhead//2), n_layers=args.num_layers,
                           dropout=args.dropout, num_classes=nc),
        "mamba":       lambda: MambaScalper(
                           input_size=backbone_input, d_model=args.d_model,
                           num_layers=args.num_layers, dropout=args.dropout,
                           num_classes=nc),
        "gnn":         lambda: GNNFromSequence(
                           input_size=backbone_input, hidden=args.hidden_size,
                           num_layers=args.num_layers, dropout=args.dropout,
                           n_nodes=6, num_classes=nc, nhead=min(4, args.nhead)),
        "expert":      lambda: EXPERTEncoder(
                           input_size=backbone_input, d_model=args.d_model,
                           nhead=args.nhead,
                           num_layers=args.num_layers, dropout=args.dropout,
                           num_classes=nc),
        "glm":         lambda: GLMBaseline(
                           input_size=backbone_input, num_classes=nc,
                           seq_len=getattr(args, "seq_len", 16)),
    }

    if core_name not in builders:
        raise ValueError(f"Unknown model: {core_name}. Available: {list(builders.keys())}")

    m = builders[core_name]()

    # Pair embedding wrapper (only when embed_dim > 0 and training on multiple pairs)
    if use_pair_emb:
        _cw_short = getattr(args, "corr_window",      20)
        _cw_long  = getattr(args, "corr_window_long", 60)
        _mw       = getattr(args, "momentum_window",  20)
        _feat_names = getattr(args, "_feat_names", None)
        _ri, _ai = _resolve_pair_feat_indices(_feat_names, f_per_pair)
        m = MultiPairWrapper(
            m,
            n_pairs=n_pairs, f_per_pair=f_per_pair, embed_dim=embed_dim,
            corr_window=_cw_short, corr_window_long=_cw_long, momentum_window=_mw,
            return_idx=_ri, atr_idx=_ai,
        )
        print(f"[Model] {name.upper()} | MultiPair wrapper "
              f"({n_pairs}P ├ù {f_per_pair}F + {embed_dim}E | "
              f"corr={_cw_short}/{_cw_long}bar mom={_mw}bar) | "
              f"{_format_param_count(m)} parameters")
    elif n_pairs > 1:
        print(f"[Model] {name.upper()} | {n_pairs} pairs ├ù {f_per_pair}F concatenated | "
              f"{_format_param_count(m)} parameters")

    if multitask:
        head_in = _multitask_head_in(core_name, args, backbone_input)
        m = MultiTaskWrapper(
            m, head_in=head_in,
            hidden=64, dropout=args.dropout,
            proj_threshold=1024, proj_to=256,
            force_project=False,
        )
        print(f"[Model] {name.upper()} | MultiTask wrapper (head_in={head_in}) | "
              f"{_format_param_count(m)} parameters")
    elif n_pairs == 1:
        print(f"[Model] {name.upper()} | {_format_param_count(m)} parameters")
    return m

def _strict_load_report(target: nn.Module, state: dict, label: str,
                        min_frac_loaded: float = 0.5) -> dict:
    """Load `state` into `target` (strict=False) but capture and LOG the
    missing/unexpected/shape-mismatched keys, and FAIL LOUDLY when the load is
    effectively a no-op (A-H2 / A-C1).

    Only tensors whose name AND shape match are loaded; the rest are reported.
    Raises RuntimeError when fewer than `min_frac_loaded` of the target tensors
    were populated (i.e. the checkpoint didn't actually transfer).
    """
    _ensure_bound()
    target_sd = target.state_dict()
    # Tolerate leading wrapper prefixes from MultiTaskWrapper / torch.compile.
    if not any(k in target_sd for k in state) and state:
        if any(k.startswith("backbone.") for k in state):
            state = {k.replace("backbone.", "", 1): v for k, v in state.items()}
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    # Always strip _orig_mod. even when some keys already match (mixed ckpts).
    if any(k.startswith("_orig_mod.") for k in state):
        state = {
            (k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k): v
            for k, v in state.items()
        }
    filtered, mismatched = {}, []
    for k, v in state.items():
        if k in target_sd and hasattr(v, "shape") and target_sd[k].shape == v.shape:
            filtered[k] = v
        elif k in target_sd:
            mismatched.append(k)
    result = target.load_state_dict(filtered, strict=False)
    missing    = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    n_target = max(1, len(target_sd))
    n_loaded = n_target - len(missing)
    frac     = n_loaded / n_target
    print(f"[Load:{label}] {n_loaded}/{n_target} tensors ({frac:.1%}) | "
          f"missing={len(missing)} unexpected={len(unexpected)} "
          f"shape_mismatch={len(mismatched)}")
    if missing:
        print(f"  missing[:6]={missing[:6]}")
    if mismatched:
        print(f"  shape_mismatch[:6]={mismatched[:6]}")
    if frac < min_frac_loaded:
        raise RuntimeError(
            f"[Load:{label}] Only {frac:.1%} of target tensors were loaded "
            f"(< {min_frac_loaded:.0%}). The checkpoint did not transfer ΓÇö "
            "check architecture/seq_len/feature-count consistency."
        )
    return {"frac_loaded": frac, "missing": missing,
            "unexpected": unexpected, "shape_mismatch": mismatched}


def get_model_training_profile(model_name: str) -> ModelTrainingProfile:
    """Convenience function to get training profile from model factory."""
    return get_training_profile(model_name)


__all__ = [
    "build_model",
    "_strict_load_report",
    "_multitask_head_in",
    "_core_model",
    "get_model_training_profile",
    "ModelTrainingProfile",
    "MODEL_PROFILES",
]

