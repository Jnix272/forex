"""RL training + ONNX deploy helpers.

See docs/CONTINUE.md."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler, StandardScaler

from config.settings import BACKTEST, RISK, RL, SIZING
from models.rl_agents import DQNAgent, ForexTradingEnv, PPOAgent, evaluate_agent, train_agent
from training.cache_integrity import (
    _load_rl_market_from_cache,
    _on_disk_sequence_count,
    _require_rl_market_cache,
    _resolve_pair_feat_indices,
    _trainable_max_index,
)
from training.core import _TRAIN_LOGGER, WANDB, _safe_wandb_log, _safe_wandb_summary_update
from training.dataset_builder import _load_scaler_npz, _safe_save_json, _x_path, _y_path, _zarr_open_group
from training.feature_ablation import _atomic_copy
from training.gpu_cache_io import ZARR
from training.model_factory import _core_model, _strict_load_report, build_model
from training.post_train import _safe_save


def _rl_reward_weights(args):
    # Accept both spellings: gpu_cli maps YAML rl.reward -> args.rl_reward_weights,
    # while older code paths/tests may pass args.reward_weights directly.
    return getattr(args, "rl_reward_weights", None) or getattr(args, "reward_weights", {}) or {}


def _rl_train_val_slices(total, args):
    # Clamp to the promotion-holdout-safe trainable region so RL train/val
    # never touch the forward-holdout window reserved for PromotionGate.
    try:
        from training.cache_integrity import _trainable_max_index

        total = min(int(total), int(_trainable_max_index(int(total), args)))
    except Exception:
        pass
    val_split = getattr(args, "rl_val_split", getattr(args, "rl_val_frac", 0.1))
    val_n = int(total * val_split)
    return 0, total - val_n, total - val_n, val_n


def _rl_algo_kwargs(args, algo):
    # rl.algo_overrides dict (generic) + rl.<algo> per-algo sub-dict from YAML.
    overrides = getattr(args, "rl_algo_overrides", {}) or {}
    merged = dict(overrides.get(algo, {}))
    per_algo = getattr(args, f"rl_{algo}_overrides", None)
    if isinstance(per_algo, dict):
        merged.update(per_algo)
    return merged

# -----------------------------------------------------------------------------
# RL TRAINING
# -----------------------------------------------------------------------------


def _build_rl_market_arrays(y_labels, base_price: float = 1.085, base_spread: float = 0.00008):
    """DEPRECATED — label-leakage synthetic price path.

    Historically this integrated CPAR ``y_labels`` into ``prices = cumprod(1+ y/std*0.0004)``.
    That couples the tradable price to the prediction target → RL optimizes a
    non-tradable shape and inflates Sharpe. Replaced by hard failure: rebuild the
    Zarr cache so ``_load_rl_market_from_cache`` returns real OHLC.

    Kept only for backward compat with legacy unit tests that import the symbol;
    always raises unless ``allow_synthetic=True`` is explicitly passed.
    """
    raise RuntimeError(
        "[RL] _build_rl_market_arrays removed: synthetic price from y_labels causes "
        "label leakage (prices = cumprod(y/std)). Rebuild cache: "
        "python -m training.train_gpu --config config/run.yaml --build-only  "
        "so _load_rl_market_from_cache provides real OHLC. "
        "If you intentionally need the legacy path (tests), pass allow_synthetic=True."
    )


def _try_rl_market_from_features(
    X_last: np.ndarray,
    scaler: StandardScaler | RobustScaler | None,
    feat_names: list | None,
    f_per_pair: int,
    base_price: float = 1.085,
    base_spread: float = 0.00008,
) -> tuple | None:
    """Build RL price/ATR/spread arrays from denormalized feature columns."""
    if scaler is None:
        return None
    # Check scaler is fitted
    if isinstance(scaler, RobustScaler):
        if not hasattr(scaler, "center_") or scaler.center_ is None:
            return None
    else:
        if not hasattr(scaler, "mean_") or scaler.mean_ is None:
            return None
    X = np.asarray(X_last, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        return None
    sl = X[:, : min(f_per_pair, X.shape[1])]
    # Inverse transform: (x - center) / scale -> x * scale + center (RobustScaler)
    # or (x - mean) / scale -> x * scale + mean (StandardScaler)
    if isinstance(scaler, RobustScaler):
        center = np.asarray(scaler.center_, dtype=np.float64)[: sl.shape[1]]
        scale = np.asarray(scaler.scale_, dtype=np.float64)[: sl.shape[1]]
        scale = np.where(scale > 1e-12, scale, 1.0)
        raw = sl * scale + center
    else:
        mean = np.asarray(scaler.mean_, dtype=np.float64)[: sl.shape[1]]
        scale = np.asarray(scaler.scale_, dtype=np.float64)[: sl.shape[1]]
        scale = np.where(scale > 1e-12, scale, 1.0)
        raw = sl * scale + mean

    names = list(feat_names or [])[: sl.shape[1]]
    ri, ai = _resolve_pair_feat_indices(names, sl.shape[1])
    spread_i = next((names.index(c) for c in ("spread_pips", "spread_avg") if c in names), None)

    ret = np.nan_to_num(raw[:, ri], nan=0.0)
    if np.std(ret) < 1e-12:
        return None
    per_bar_ret = np.clip(ret, -0.01, 0.01)
    prices = (base_price * np.cumprod(1.0 + per_bar_ret)).astype(np.float32)

    atr_raw = np.abs(np.nan_to_num(raw[:, ai], nan=0.0))
    if float(np.median(atr_raw)) <= 0:
        abs_ret = np.abs(np.diff(prices, prepend=prices[0]))
        win = 14
        atr_raw = np.convolve(abs_ret, np.ones(win) / win, mode="same")
    atr = np.maximum(atr_raw.astype(np.float32), 1e-4)
    med_atr = float(np.median(atr)) or 1e-4

    if spread_i is not None:
        spr_pips = np.maximum(np.nan_to_num(raw[:, spread_i], nan=0.0), 0.0)
        spreads = np.clip(spr_pips * 0.0001, base_spread, 5 * base_spread).astype(np.float32)
    else:
        spreads = np.clip(base_spread * (atr / med_atr), base_spread, 5 * base_spread).astype(np.float32)
    return prices, atr, spreads


def _encode_rl_observations(cache_path, start: int, n_env: int, n_features: int, args, device, batch: int = 4096):
    """A-C3: run the frozen supervised encoder over the RL window's full
    sequences and return its pre-head embedding per bar as the RL observation.

    This is the connective tissue between the supervised/pretrained stage and RL:
    the policy observes the supervised representation instead of raw features,
    so RL fine-tunes ON TOP of the learned encoder. The encoder is frozen.
    """
    ckpt_dir = Path(args.checkpoint_dir)
    candidates = [ckpt_dir / args.model / f"{args.model}_best.pt", ckpt_dir / f"{args.model}_best.pt"]
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(f"no supervised checkpoint for {args.model} under {ckpt_dir}")
    model = build_model(args.model, n_features, args).to(device)
    core = _core_model(model)
    
    # Dry run to initialize Lazy modules before loading weights
    with torch.no_grad():
        _dummy = torch.zeros(1, getattr(args, "seq_len", 60), n_features, device=device)
        _ = core(_dummy)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    _strict_load_report(core, state, f"RLObsEncoder:{args.model}", min_frac_loaded=0.6)
    encoder = core.backbone if hasattr(core, "backbone") else core
    saved_head = None
    if hasattr(encoder, "head"):
        saved_head = encoder.head
        encoder.head = nn.Identity()
    core.eval()
    for p in core.parameters():
        p.requires_grad_(False)

    if ZARR and cache_path.endswith(".zarr") and Path(cache_path).is_dir():
        _z = _zarr_open_group(cache_path, mode="r")
        _Xs = _z["X"]

        def _read(a, b):
            return np.asarray(_Xs[start + a : start + b], dtype=np.float32)  # type: ignore[reportArgumentType]
    else:
        _Xm = np.load(_x_path(cache_path), mmap_mode="r")

        def _read(a, b):  # type: ignore[reportRedeclaration]
            return np.asarray(_Xm[start + a : start + b], dtype=np.float32)

    embs = []
    with torch.no_grad():
        for a in range(0, n_env, batch):
            b = min(a + batch, n_env)
            xb = torch.as_tensor(_read(a, b), dtype=torch.float32, device=device)
            xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
            h = encoder(xb)
            if h.ndim == 3:
                h = h[:, -1, :]
            if hasattr(core, "proj") and not isinstance(core.proj, nn.Identity):
                h = core.proj(h)
            embs.append(h.float().cpu().numpy())
    if saved_head is not None:
        encoder.head = saved_head
    return np.concatenate(embs, axis=0).astype(np.float32)


def _load_rl_slice(cache_path: str, start: int, n_bars: int) -> tuple[np.ndarray, np.ndarray]:
    """Load y and last-timestep features for an RL window."""
    if ZARR and cache_path.endswith(".zarr") and Path(cache_path).is_dir():
        _z = _zarr_open_group(cache_path, mode="r")
        y_env = np.asarray(_z["y"][start : start + n_bars], dtype=np.float32)  # type: ignore[reportArgumentType]
        X_last = np.asarray(_z["X"][start : start + n_bars, -1, :], dtype=np.float32)  # type: ignore[reportArgumentType]
    else:
        y_env = np.asarray(np.load(_y_path(cache_path), mmap_mode="r")[start : start + n_bars], dtype=np.float32)
        X_last = np.asarray(
            np.load(_x_path(cache_path), mmap_mode="r")[start : start + n_bars, -1, :],
            dtype=np.float32,
        )
    return y_env, X_last


def _build_rl_env(
    cache_path: str,
    start: int,
    n_bars: int,
    n_features: int,
    args,
    device,
) -> ForexTradingEnv:
    """Construct ForexTradingEnv for train or validation slice."""
    y_env, X_last = _load_rl_slice(cache_path, start, n_bars)
    n_bars = len(y_env)

    prices, atr, spreads = _load_rl_market_from_cache(cache_path, start, n_bars)
    _market_source = "cache"
    if float(np.std(prices)) < 1e-12:
        _rl_scaler = _load_scaler_npz(Path(cache_path))
        _fpp = int(getattr(args, "_f_per_pair", X_last.shape[1]) or X_last.shape[1])
        _fnames = getattr(args, "_feat_names", None)
        if _fnames is None and _rl_scaler is not None and hasattr(_rl_scaler, "feature_names_in_"):
            _fnames = list(_rl_scaler.feature_names_in_)
        _market = _try_rl_market_from_features(X_last, _rl_scaler, _fnames, _fpp)
        if _market is not None:
            prices, atr, spreads = _market
            _market_source = "features"
            print(f"[RL] WARN: market source={_market_source} -- rebuild cache for real OHLC")
        else:
            raise RuntimeError(
                "[RL] Market cache degenerate (std(prices) < 1e-12) and feature fallback unavailable. "
                "Synthetic price from y_labels is DISABLED to prevent label leakage. "
                "Rebuild cache: python -m training.train_gpu --config config/run.yaml --build-only"
            )

    obs_feats = None
    if bool(getattr(args, "rl_encoder_obs", True)):
        try:
            obs_feats = _encode_rl_observations(cache_path, start, n_bars, n_features, args, device)
        except Exception as _ee:
            print(f"[RL] Encoder-obs unavailable ({_ee}); falling back to raw features.")
    if obs_feats is None:
        obs_feats = X_last

    _ep_len = int(getattr(args, "rl_episode_len", 0) or 0) or None
    _enable_idle = not bool(getattr(args, "rl_encoder_obs", True))
    # When encoder obs is active, idle penalty would otherwise fire on a random
    # latent dimension (features[:,0] is not ret_5). Disable by default; if a
    # true directional signal is needed, pass supervised_signal explicitly.
    _sup_signal = None
    if _enable_idle:
        # raw mode: keep existing behaviour (features[:,0] is signal)
        pass
    # Annualize Sharpe correctly for bar frequency (5m → 252*288, not daily 252)
    _bf = str(getattr(args, "bar_freq", getattr(args, "strategy_bar_freq", "5m")))
    try:
        bf = _bf.strip().lower()
        if bf.endswith("m"):
            _bpy = 252 * 24 * 60 // int(bf[:-1])
        elif bf.endswith("h"):
            _bpy = 252 * 24 // int(bf[:-1])
        else:
            _bpy = 252 * 24 * 60 // 5
    except Exception:
        _bpy = 252 * 24 * 60 // 5
    return ForexTradingEnv(
        features=obs_feats,
        prices=prices,
        atr=atr,
        spreads=spreads,
        reward_weights=_rl_reward_weights(args),
        atr_sl_mult=RISK["atr_multiplier"],
        trail_activation_r=RISK["trail_activation_r"],
        breakeven_at_r=RISK["breakeven_at_r"],
        pyramid_pct=SIZING["pyramid_add_pct"],
        martingale_pct=SIZING["martingale_add_pct"],
        max_lots=SIZING["max_total_lots"],
        commission_per_lot=float(BACKTEST.get("commission_per_lot", 3.5)),
        slippage_pips=float(BACKTEST.get("slippage_pips", 0.7)),
        pip_size=0.0001,
        random_reset=True,
        episode_len=_ep_len,
        enable_idle_penalty=_enable_idle,
        supervised_signal=_sup_signal,
        bars_per_year=_bpy,
    )


def _save_rl_checkpoint(agent, ckpt_dir: Path, algo: str, tag: str) -> Path:
    path = ckpt_dir / f"rl_{algo}_{tag}.pt"
    if hasattr(agent, "policy_net"):
        sd = agent.policy_net.state_dict()
    elif hasattr(agent, "net"):
        sd = agent.net.state_dict()
    else:
        raise RuntimeError("[RL] Agent has no saveable policy weights")
    # Wrap in the same {"model_state": ...} envelope used by supervised checkpoints
    # so the ONNX exporter and supervised loader utilities can load RL checkpoints
    # with the same unwrapping idiom.
    _safe_save({"model_state": sd}, path)
    return path


def _production_onnx_paths(args) -> tuple[Path, Path]:
    try:
        from monitoring.demotion_monitor import PROD_CHECKPOINT as _prod  # type: ignore[reportAttributeAccessIssue]

        prod_onnx = Path(_prod).with_suffix(".onnx")
    except Exception:
        prod_onnx = Path(args.checkpoint_dir) / "production_best.onnx"
    prev_onnx = prod_onnx.with_name("production_prev.onnx")
    return prod_onnx, prev_onnx


def _write_feature_schema_for_onnx(schema_path: Path, args) -> None:
    _safe_save_json(_feature_schema_payload(args), schema_path)


def _feature_schema_payload(args, n_features: int | None = None) -> dict:

    import hashlib

    feature_names = list(getattr(args, "_feat_names", []) or [])
    n_feat = int(n_features or getattr(args, "_n_features", 0) or len(feature_names) or 0)

    schema_hash = hashlib.md5(json.dumps(feature_names, sort_keys=True).encode()).hexdigest()
    return {
        "feature_names": feature_names,
        "hash": schema_hash,
        "n_features": n_feat,
        "seq_len": int(getattr(args, "seq_len", 60)),
        "created_at": datetime.now(UTC).isoformat(),
    }


def _verify_onnx_schema_deployment(onnx_path: Path, schema_path: Path, args, *, n_features: int, seq_len: int) -> dict:
    """Verify ONNX artifact exists, schema matches training, and optional runtimes can load it."""
    result = {
        "onnx_path": str(onnx_path),
        "schema_path": str(schema_path),
        "status": "pass",
        "checks": {},
        "warnings": [],
        "errors": [],
    }

    try:
        if not onnx_path.exists() or onnx_path.stat().st_size <= 0:
            raise RuntimeError("onnx file missing or empty")

        result["checks"]["onnx_file"] = {"exists": True, "bytes": int(onnx_path.stat().st_size)}

        if not schema_path.exists():
            raise RuntimeError("schema json missing")

        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        expected = _feature_schema_payload(args, n_features=n_features)

        if int(schema.get("n_features", -1)) != int(n_features):
            raise RuntimeError(f"schema n_features mismatch: {schema.get('n_features')} != {n_features}")

        if int(schema.get("seq_len", -1)) != int(seq_len):
            raise RuntimeError(f"schema seq_len mismatch: {schema.get('seq_len')} != {seq_len}")

        if schema.get("hash") != expected.get("hash"):
            raise RuntimeError("schema feature hash mismatch")

        result["checks"]["schema"] = {
            "n_features": int(schema.get("n_features")),
            "seq_len": int(schema.get("seq_len")),
            "hash": schema.get("hash"),
        }

        try:
            import onnx

            model = onnx.load(str(onnx_path))

            onnx.checker.check_model(model)

            result["checks"]["onnx_checker"] = "pass"

        except ImportError:
            result["warnings"].append("onnx package not installed; checker skipped")

        except Exception as exc:
            raise RuntimeError(f"onnx checker failed: {exc}") from exc

        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

            input_name = sess.get_inputs()[0].name

            dummy = np.zeros((1, int(seq_len), int(n_features)), dtype=np.float32)

            outputs = sess.run(None, {input_name: dummy})

            if not outputs or not np.isfinite(np.asarray(outputs[0])).all():
                raise RuntimeError("onnxruntime smoke output missing or nonfinite")

            result["checks"]["onnxruntime_cpu"] = {
                "output_shape": list(np.asarray(outputs[0]).shape),
            }

        except ImportError:
            result["warnings"].append("onnxruntime not installed; CPU smoke test skipped")

    except Exception as exc:
        result["status"] = "fail"

        result["errors"].append(str(exc))

    return result


def _signal_cpp_server_reload(prod_onnx: Path) -> Path | None:
    import tempfile

    reload_flag = prod_onnx.parent / "reload_model.flag"
    fd, tmp_flag = tempfile.mkstemp(prefix=".reload.", suffix=".tmp", dir=str(prod_onnx.parent))
    os.close(fd)
    with open(tmp_flag, "w", encoding="utf-8") as f:
        f.write(f"reload {datetime.now(UTC).isoformat()}\n")
    os.replace(tmp_flag, reload_flag)
    return reload_flag


def _deploy_onnx_to_cpp_server(
    onnx_path: Path,
    args,
    model_name: str,
    artifact_dir: Path,
    source_checkpoint: Path | None = None,
) -> dict:
    """Atomically promote an exported ONNX graph to the C++ server path."""
    result = {
        "model_name": model_name,
        "source_checkpoint": str(source_checkpoint) if source_checkpoint else None,
        "source_onnx": str(onnx_path),
        "production_onnx": None,
        "schema_path": None,
        "reload_flag": None,
        "status": "skipped",
        "error": None,
    }
    try:
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX artifact does not exist: {onnx_path}")
        artifact_dir.mkdir(parents=True, exist_ok=True)

        source_schema = artifact_dir / f"{model_name}_onnx_schema.json"

        _safe_save_json(_feature_schema_payload(args), source_schema)

        verify = _verify_onnx_schema_deployment(
            onnx_path,
            source_schema,
            args,
            n_features=int(getattr(args, "_n_features", 0) or len(getattr(args, "_feat_names", []) or [])),
            seq_len=int(getattr(args, "seq_len", 60)),
        )

        result["verification"] = verify

        if verify.get("status") != "pass":
            raise RuntimeError(f"ONNX/schema verification failed: {verify.get('errors')}")

        prod_onnx, prev_onnx = _production_onnx_paths(args)
        prod_onnx.parent.mkdir(parents=True, exist_ok=True)
        if prod_onnx.exists():
            _atomic_copy(prod_onnx, prev_onnx)
        _atomic_copy(onnx_path, prod_onnx)
        schema_path = prod_onnx.with_suffix(".schema.json")
        _atomic_copy(source_schema, schema_path)

        reload_flag = _signal_cpp_server_reload(prod_onnx)
        result.update(
            {
                "production_onnx": str(prod_onnx),
                "schema_path": str(schema_path),
                "reload_flag": str(reload_flag),
                "status": "success",
            }
        )
        print(f"[Deploy] {model_name} ONNX promoted for C++ -> {prod_onnx}")
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        print(f"[Deploy] {model_name} ONNX promotion failed: {exc}")
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _safe_save_json(result, artifact_dir / "cpp_deployment.json")
    except Exception as exc:
        print(f"[Deploy] Could not write {model_name} cpp_deployment.json: {exc}")
    return result


def _run_rl_via_adapter(cache_path, n_features, args, device, framework="cleanrl", n_samples=None, run=None):
    """Opt-in RL fast path: route through ``create_rl_adapter``/``run_rl_with_adapter``.

    Used only when ``--rl-framework`` is explicitly non-"custom" (cleanrl / sb3).
    The adapter trains the external-framework policy and returns its metrics dict;
    the in-house PPO/DQN path (currency + val + ONNX export) is untouched.
    """
    print(f"\n[RL] framework={framework} | {args.rl_algo.upper()} | model={args.model}")
    _require_rl_market_cache(cache_path)

    total = int(n_samples or (_on_disk_sequence_count(cache_path) or 0))
    train_start, train_n, _val_start, _val_n = _rl_train_val_slices(total, args)
    if train_n < 256:
        raise RuntimeError(f"[RL] Insufficient trainable bars ({train_n}). Rebuild cache or reduce holdout.")

    train_env = _build_rl_env(cache_path, train_start, train_n, n_features, args, device)
    print(f"[RL] Train env | obs={train_env.obs_size} | market std={float(np.std(train_env.prices)):.6f}")

    from training.rl_adapter import RLConfig, create_rl_adapter, run_rl_with_adapter

    _algo = str(args.rl_algo).lower()
    _algo_cfg = RL.get(_algo, {})
    _n_episodes = int(args.rl_episodes)
    _n_timesteps = int(getattr(args, "rl_total_timesteps", 0) or max(10_000, _n_episodes * 1000))

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg = RLConfig(
        algorithm=_algo,  # type: ignore[reportArgumentType]
        learning_rate=float(_algo_cfg.get("lr", 3e-4)),
        gamma=float(_algo_cfg.get("gamma", 0.99)),
        n_steps=int(_algo_cfg.get("n_steps", 2048)),
        n_epochs=int(_algo_cfg.get("n_epochs", 10)),
        gae_lambda=float(_algo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(_algo_cfg.get("clip_epsilon", 0.2)),
        ent_coef=float(_algo_cfg.get("entropy_coeff", 0.01)),
        vf_coef=float(_algo_cfg.get("value_coeff", 0.5)),
        total_timesteps=_n_timesteps,
        device=str(device),
        seed=int(getattr(args, "seed", 1337)),
        log_dir=str(ckpt_dir / "logs"),
        save_path=str(ckpt_dir / f"rl_{_algo}_adapter.pt"),
    )
    adapter = create_rl_adapter(framework, _algo, cfg)  # type: ignore[reportArgumentType]

    metrics = run_rl_with_adapter(
        adapter,
        cache_path,
        np.arange(train_n),
        prices=np.asarray(train_env.prices, dtype=np.float32),
        atr=np.asarray(train_env.atr, dtype=np.float32),
        spreads=np.asarray(train_env.spreads, dtype=np.float32),
        features=np.asarray(train_env.features, dtype=np.float32),
        total_timesteps=_n_timesteps,
        n_episodes=_n_episodes,
    )

    _save_path = cfg.save_path
    if getattr(adapter, "model", None) is not None and hasattr(adapter, "save"):
        try:
            adapter.save(_save_path)  # type: ignore[reportArgumentType]
            print(f"[RL] Saved adapter policy -> {_save_path}")
        except Exception as _save_exc:
            print(f"[RL] Adapter checkpoint save skipped: {_save_exc}")

    result = {
        "framework": framework,
        "algorithm": _algo,
        "n_episodes": _n_episodes,
        "total_timesteps": _n_timesteps,
        "metrics": metrics,
        "checkpoint": _save_path,
        "status": "success",
    }
    try:
        if not (isinstance(metrics, dict) and "returns" in metrics):
            print(
                f"[RL] Adapter metrics keys: {sorted(metrics) if isinstance(metrics, dict) else type(metrics).__name__}"
            )
        if WANDB and run is not None:
            _safe_wandb_log(run, {f"rl_adapter/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
    except Exception as _rl_ae:
        print(f"[RL] Adapter post-run logging skipped: {_rl_ae}")
    print(f"[RL] Adapter run complete | framework={framework} algo={_algo}")
    return result


def run_rl(cache_path, n_features, args, device, n_samples=None, run=None):
    _rl_framework = str(getattr(args, "rl_framework", "custom") or "custom").lower()
    if _rl_framework != "custom":
        return _run_rl_via_adapter(
            cache_path,
            n_features,
            args,
            device,
            framework=_rl_framework,
            n_samples=n_samples,
            run=run,
        )
    print(f"\n[RL] {args.rl_algo.upper()} | {args.rl_episodes} episodes | model={args.model}")
    _require_rl_market_cache(cache_path)

    total = int(n_samples or (_on_disk_sequence_count(cache_path) or 0))
    train_start, train_n, val_start, val_n = _rl_train_val_slices(total, args)
    if train_n < 256:
        raise RuntimeError(f"[RL] Insufficient trainable bars ({train_n}). Rebuild cache or reduce holdout.")

    print(
        f"[RL] Holdout-safe window | train [{train_start}:{train_start + train_n}) "
        f"| val [{val_start}:{val_start + val_n}) | trainable_end={_trainable_max_index(total, args):,}"
    )

    train_env = _build_rl_env(cache_path, train_start, train_n, n_features, args, device)
    print(f"[RL] Train env | obs={train_env.obs_size} | market std={float(np.std(train_env.prices)):.6f}")

    # Skip RL for models whose profile marks rl_finetune=False (e.g. GNN, GLM).
    if not getattr(args, "rl_finetune", True):
        print(f"[RL] Skipping RL fine-tune for {args.model} (rl_finetune=False in profile).")
        return []

    dev = str(device)
    _algo = str(args.rl_algo).lower()
    _algo_kw = _rl_algo_kwargs(args, _algo)
    _use_lstm = bool(getattr(args, "rl_use_lstm", False))
    if _algo == "dqn":
        agent = DQNAgent(obs_size=train_env.obs_size, n_actions=train_env.n_actions, device=dev, use_lstm=_use_lstm, **_algo_kw)
    else:
        agent = PPOAgent(obs_size=train_env.obs_size, n_actions=train_env.n_actions, device=dev, use_lstm=_use_lstm, **_algo_kw)
    if _use_lstm:
        print(f"[RL] LSTM backbone enabled for {args.model}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _val_episodes = max(3, min(20, int(args.rl_episodes) // 25))
    _min_val_sharpe = float(getattr(args, "rl_min_val_sharpe", 0.0))
    _best_val_sharpe = float("-inf")
    _best_saved = False
    # Evaluate on val every this many training episodes
    _val_every = max(10, int(args.rl_episodes) // 10)

    # ── RL curriculum scheduler (graduated volatility exposure) ──────────
    _rl_curriculum = None
    if bool(getattr(args, "rl_curriculum", True)):
        try:
            from models.rl_advanced import CurriculumScheduler as _CS

            _rl_curriculum = _CS(total_episodes=args.rl_episodes)
            print(f"[RL] Curriculum: graduated volatility over {args.rl_episodes} episodes")
        except Exception as _rce:
            print(f"[RL] Curriculum unavailable: {_rce}")

    reward_sharpe = None
    if bool(getattr(args, "rl_use_sharpe_reward", False)):
        try:
            from models.rl_agents import _SharpeRewardAdapter

            _bpy = int(getattr(train_env, "bars_per_year", 0) or 0)
            if _bpy <= 0:
                # Derive from bar_freq (e.g. 5m → 252*24*60//5)
                _bf = str(getattr(args, "bar_freq", getattr(args, "strategy_bar_freq", "5m")))
                try:
                    bf = _bf.strip().lower()
                    if bf.endswith("m"):
                        _bpy = 252 * 24 * 60 // int(bf[:-1])
                    elif bf.endswith("h"):
                        _bpy = 252 * 24 // int(bf[:-1])
                    else:
                        _bpy = 252 * 24 * 60 // 5
                except Exception:
                    _bpy = 252 * 24 * 60 // 5
            reward_sharpe = _SharpeRewardAdapter(bars_per_year=_bpy)
            print(f"[RL] SharpeRewardWrapper enabled (bars_per_year={_bpy}, ann={_bpy**0.5:.1f})")
        except Exception as _sre:
            print(f"[RL] SharpeReward unavailable: {_sre}")

    her_buffer = None
    if bool(getattr(args, "rl_use_her", False)) and _algo == "dqn":
        try:
            from models.rl_advanced import HERBuffer

            her_buffer = HERBuffer(capacity=100_000, k=4)
            print("[RL] HERBuffer enabled for DQN hindsight replay")
        except Exception as _he:
            print(f"[RL] HER unavailable: {_he}")

    # Build per-episode val callback when a val window exists
    _ep_val_callback = None
    if val_n > 0:
        _ep_val_env_ref: list = []  # lazy-init so env is created once

        def _ep_val_callback(ep: int, _agent) -> None:
            nonlocal _best_val_sharpe, _best_saved
            if (ep + 1) % _val_every != 0:
                return
            if not _ep_val_env_ref:
                _ep_val_env_ref.append(
                    _build_rl_env(cache_path, val_start, val_n, n_features, args, device)
                )
            _ep_venv = _ep_val_env_ref[0]
            _, _ep_vs = evaluate_agent(_agent, _ep_venv, n_episodes=_val_episodes, agent_type=_algo, greedy=True)
            _ep_sharpe = float(_ep_vs.get("sharpe", 0.0))
            print(
                f"[RL] Ep {ep + 1} val | Sharpe={_ep_sharpe:.3f} "
                f"| Return={_ep_vs['total_return_pct']:+.2f}% | Trades={_ep_vs['n_trades']}"
            )
            # Save only when this episode is a genuine new best AND meets min bar.
            # Update _best_val_sharpe AFTER the save guard so the condition is not
            # a tautology (updating first makes _ep_sharpe >= _best_val_sharpe always True).
            if _ep_sharpe >= _min_val_sharpe and _ep_sharpe > _best_val_sharpe:
                _best_val_sharpe = _ep_sharpe
                _save_rl_checkpoint(_agent, ckpt_dir, _algo, "best")
                _best_saved = True

    returns = train_agent(
        agent,
        train_env,
        n_episodes=args.rl_episodes,
        agent_type=_algo,
        curriculum=_rl_curriculum,
        reward_sharpe=reward_sharpe,
        her_buffer=her_buffer,
        off_policy_rewards=bool(getattr(args, "off_policy_rewards", False)),
        episode_callback=_ep_val_callback,
    )

    if bool(getattr(args, "off_policy_rewards", False)):
        _op_est = getattr(agent, "off_policy_estimates", None)
        if _op_est:
            _dr = [e["dr_value"] for e in _op_est]
            _ips = [e["ips_value"] for e in _op_est]
            print(
                f"[RL] Off-policy rewards (Improvement #5, diagnostic only): "
                f"mean DR value={np.mean(_dr):.4f} | mean IPS value={np.mean(_ips):.4f} "
                f"over {len(_op_est)} episodes - does not train the policy"
            )

    if val_n > 0:
        val_env = _build_rl_env(cache_path, val_start, val_n, n_features, args, device)
        _, val_summary = evaluate_agent(agent, val_env, n_episodes=_val_episodes, agent_type=_algo, greedy=True)
        _val_sharpe = float(val_summary.get("sharpe", 0.0))
        print(
            f"[RL] Val | Sharpe={_val_sharpe:.3f} | Return={val_summary['total_return_pct']:+.2f}% "
            f"| Trades={val_summary['n_trades']}"
        )
        # Only overwrite the best checkpoint when this final evaluation is strictly
        # better than every per-episode checkpoint already saved mid-training.
        if _val_sharpe >= _min_val_sharpe and _val_sharpe > _best_val_sharpe:
            _best_val_sharpe = _val_sharpe
            _save_rl_checkpoint(agent, ckpt_dir, _algo, "best")
            meta = {
                "model": args.model,
                "algo": _algo,
                "val_sharpe": _val_sharpe,
                "val_return_pct": float(val_summary["total_return_pct"]),
                "obs_size": int(train_env.obs_size),
                "n_actions": int(train_env.n_actions),
                "encoder_obs": bool(getattr(args, "rl_encoder_obs", True)),
            }
            with (ckpt_dir / f"rl_{_algo}_best.json").open("w", encoding="utf-8") as _rl_meta_fp:
                json.dump(meta, _rl_meta_fp, indent=2)
            _best_saved = True
            try:
                from inference.onnx_inference import export_rl_execution_to_onnx, export_rl_to_onnx

                rl_best = ckpt_dir / f"rl_{_algo}_best.pt"
                sup_ckpt = ckpt_dir / args.model / f"{args.model}_best.pt"
                if not sup_ckpt.is_file():
                    sup_ckpt = ckpt_dir / f"{args.model}_best.pt"
                if not sup_ckpt.is_file():
                    sup_ckpt = ckpt_dir.parent / args.model / f"{args.model}_best.pt"
                rl_onnx = ckpt_dir / f"rl_{_algo}_best.onnx"
                export_rl_to_onnx(
                    rl_checkpoint=str(rl_best),
                    supervised_checkpoint=str(sup_ckpt),
                    model_name=str(args.model),
                    seq_len=int(getattr(args, "seq_len", 60)),
                    n_features=int(n_features),
                    output_path=str(rl_onnx),
                    algo=_algo,
                    device="cpu",
                )
                print(f"[RL] Exported ONNX -> {rl_onnx}")
                rl_exec_onnx = ckpt_dir / f"rl_{_algo}_execution.onnx"
                export_rl_execution_to_onnx(
                    rl_checkpoint=str(rl_best),
                    supervised_checkpoint=str(sup_ckpt),
                    model_name=str(args.model),
                    seq_len=int(getattr(args, "seq_len", 60)),
                    n_features=int(n_features),
                    output_path=str(rl_exec_onnx),
                    algo=_algo,
                    device="cpu",
                )
                _safe_save_json(
                    {
                        "model_name": f"rl_{_algo}_execution",
                        "source_checkpoint": str(rl_best),
                        "source_onnx": str(rl_exec_onnx),
                        "direction_model": "Use MODEL_PATH for the ensemble/supervised 3-logit direction model.",
                        "runtime_env": "Set EXECUTION_MODEL_PATH to this ONNX in the C++ server.",
                        "inputs": {
                            "features": [1, int(getattr(args, "seq_len", 60)), int(n_features)],
                            "agent_state": [1, 5],
                        },
                        "outputs": {"action_logits": [1, int(train_env.n_actions)]},
                        "actions": {
                            "0": "HOLD",
                            "1": "OPEN_LONG",
                            "2": "OPEN_SHORT",
                            "3": "SCALE_IN_25",
                            "4": "SCALE_IN_50",
                            "5": "SCALE_IN_100",
                            "6": "SCALE_OUT_25",
                            "7": "SCALE_OUT_50",
                            "8": "SCALE_OUT_100",
                            "9": "CLOSE_ALL",
                        },
                    },
                    ckpt_dir / f"rl_{_algo}_execution.json",
                )
                print(f"[RL] Exported execution ONNX -> {rl_exec_onnx}")
                if bool(getattr(args, "deploy_rl", False)):
                    args._n_features = int(n_features)
                    _deploy_onnx_to_cpp_server(
                        rl_onnx,
                        args,
                        model_name=f"rl_{_algo}",
                        artifact_dir=ckpt_dir,
                        source_checkpoint=rl_best,
                    )
            except Exception as exc:
                print(f"[RL] ONNX export/deploy skipped: {exc}")
            print(f"[RL] Saved best policy  {ckpt_dir / f'rl_{_algo}_best.pt'}")
        else:
            print(
                f"[RL] Val Sharpe {_val_sharpe:.3f} below min_val_sharpe {_min_val_sharpe:.3f} "
                "-- rl_*_best not updated"
            )

    _save_rl_checkpoint(agent, ckpt_dir, _algo, "last")

    s = train_env.summary()
    ret_arr = np.asarray(returns, dtype=np.float64)
    rl_stats = {
        "rl/total_return_pct": float(s["total_return_pct"]),
        "rl/sharpe": float(s["sharpe"]),
        "rl/n_trades": int(s["n_trades"]),
        "rl/episodes": int(args.rl_episodes),
        "rl/return_mean": float(ret_arr.mean()) if ret_arr.size else 0.0,
        "rl/return_std": float(ret_arr.std()) if ret_arr.size else 0.0,
        "rl/return_min": float(ret_arr.min()) if ret_arr.size else 0.0,
        "rl/return_max": float(ret_arr.max()) if ret_arr.size else 0.0,
        "rl/val_sharpe": float(_best_val_sharpe) if val_n > 0 else 0.0,
        "rl/best_saved": int(_best_saved),
    }
    print(
        f"[RL] Done | Train return: {s['total_return_pct']:+.2f}% | "
        f"Sharpe: {s['sharpe']:.3f} | Trades: {s['n_trades']} | "
        f"Ep mean: {rl_stats['rl/return_mean']:+.2f}%"
    )
    if _TRAIN_LOGGER is not None:
        _TRAIN_LOGGER.info(
            f"[RL] {args.rl_algo.upper()} complete -- "
            f"return={s['total_return_pct']:+.2f}% sharpe={s['sharpe']:.3f} "
            f"val_sharpe={rl_stats['rl/val_sharpe']:.3f}"
        )
        if hasattr(_TRAIN_LOGGER, "on_rl_complete"):
            _TRAIN_LOGGER.on_rl_complete(rl_stats)
    if WANDB and run is not None:
        _safe_wandb_log(run, rl_stats)
        _safe_wandb_summary_update(
            run,
            {
                k.replace("rl/", "best_rl_"): v
                for k, v in rl_stats.items()
                if k.startswith("rl/") and k != "rl/episodes"
            },
        )
    return returns
