"""
config/settings.py  —  Full System Specification
=================================================
Single source of truth for every component in the pipeline.
Covers all 25 architectural choices specified.
"""

import os
from dataclasses import dataclass
from datetime import time
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo


# Load .env from project root so WANDB_API_KEY, MLFLOW_TRACKING_URI, etc.
# are available as os.environ entries for every module that imports settings.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

# Repository root (parent of config/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(*parts: str) -> str:
    """Absolute path under the repo root (portable; works from any CWD)."""
    return str(PROJECT_ROOT.joinpath(*parts))


def active_checkpoint_dir(config_path: Path | None = None) -> Path:
    """Resolve the active run checkpoint directory.

    Priority: CHECKPOINT_RUN_DIR env → config paths.checkpoint_dir → checkpoints/.
    Used by demotion_monitor, continuous_finetune, and live deploy hooks so flags
    and production_best.pt align with training/train_gpu.py.
    """
    env = os.getenv("CHECKPOINT_RUN_DIR", "").strip()
    if env:
        return Path(env)
    cfg_path = config_path or (PROJECT_ROOT / "config" / "run.yaml")
    if cfg_path.is_file():
        try:
            import yaml
            with open(cfg_path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            rel = (data.get("paths") or {}).get("checkpoint_dir")
            if rel:
                p = Path(str(rel))
                return p if p.is_absolute() else PROJECT_ROOT / p
        except Exception:
            pass
    return PROJECT_ROOT / "checkpoints"


RELOAD_MODEL_FLAG = "reload_model.flag"
PRODUCTION_CHECKPOINT = "production_best.pt"


@dataclass(frozen=True)
class CheckpointPaths:
    """Resolved checkpoint artifacts under the active run directory."""

    checkpoint_dir: Path
    pt_path: Path | None
    onnx_path: Path | None
    source: str
    reload_flag: Path


def resolve_checkpoint_paths(
    model_name: str,
    checkpoint_dir: Path | None = None,
) -> CheckpointPaths:
    """Locate production or model-specific checkpoints for live inference.

    Priority (within ``active_checkpoint_dir()``):
      1. ``production_best.pt`` — canonical post-promotion artifact
      2. ``{model}_best.pt``
      3. ``{model}/{model}_best.pt``
      4. Legacy flat ``checkpoints/{model}_best.pt`` at repo root (deprecated)
    """
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else active_checkpoint_dir()
    model = str(model_name or "haelt").lower().strip()
    reload_flag = ckpt_dir / RELOAD_MODEL_FLAG

    candidates = [
        ("production", ckpt_dir / PRODUCTION_CHECKPOINT),
        ("model_best", ckpt_dir / f"{model}_best.pt"),
        ("nested", ckpt_dir / model / f"{model}_best.pt"),
        ("legacy_flat", PROJECT_ROOT / "checkpoints" / f"{model}_best.pt"),
    ]

    pt_path: Path | None = None
    source = "none"
    for label, path in candidates:
        if path.is_file():
            pt_path = path
            source = label
            break

    onnx_path: Path | None = None
    if pt_path is not None:
        sibling = pt_path.with_suffix(".onnx")
        if sibling.is_file():
            onnx_path = sibling
        elif source == "production":
            alt = ckpt_dir / f"{model}_best.onnx"
            if alt.is_file():
                onnx_path = alt

    return CheckpointPaths(
        checkpoint_dir=ckpt_dir,
        pt_path=pt_path,
        onnx_path=onnx_path,
        source=source,
        reload_flag=reload_flag,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ARTIFACT PATHS  —  separate top-level folders under the repo
# ─────────────────────────────────────────────────────────────────────────────
# checkpoints/   model weights, ONNX, lockbox markers
# exports/       ONNX and other export artifacts (monitoring ONNXExporter)
# logs/          training history, shadow mode, SHAP, reports, live engine
# data/          processed caches, embeddings, raw vendor feeds (incl. data/raw/*)
PATHS: dict[str, str] = {
    "checkpoints": project_path("checkpoints"),
    "exports": project_path("exports"),
    "logs": project_path("logs"),
    "data": project_path("data"),
    "data_processed": project_path("data", "processed"),
    "data_embeddings": project_path("data", "embeddings"),
    "data_raw_cot": project_path("data", "raw", "cot"),
    "logs_shadow": project_path("logs", "shadow"),
    "logs_shap": project_path("logs", "shap"),
    "logs_reports": project_path("logs", "reports"),
    "logs_live": project_path("logs", "live"),
    "file_contrastive_encoder": project_path("checkpoints", "contrastive_encoder.pt"),
    "file_lockbox_used": project_path("checkpoints", "lockbox_used.json"),
    "file_lockbox_log": project_path("checkpoints", "lockbox_log.json"),
}

BEST_CHECKPOINTS: dict[str, str] = {
    name: project_path("checkpoints", name, f"{name}_best.pt")
    for name in ("haelt", "mamba", "gnn", "tft", "transformer", "expert")
}

PATHS.update({
    "ensemble": project_path("checkpoints", "ensemble"),
    "ensemble_meta_best": project_path("checkpoints", "ensemble", "ensemble_meta_best.pt"),
    "ensemble_meta_latest": project_path("checkpoints", "ensemble", "ensemble_meta_best_latest.pt"),
    "ensemble_meta_final": project_path("checkpoints", "ensemble", "ensemble_meta_final.pt"),
    "file_haelt_best": BEST_CHECKPOINTS["haelt"],
    "file_mamba_best": BEST_CHECKPOINTS["mamba"],
    "file_gnn_best": BEST_CHECKPOINTS["gnn"],
})


# ─────────────────────────────────────────────────────────────────────────────
# DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────
DATA = {
    "resolution":      "tick",
    "storage_engine":  "timescaledb",
    "price_type":      "bid_ask",
    "pairs":           ["EURUSD", "GBPUSD", "USDJPY"],
    "primary_pair":    "EURUSD",
    "bar_freq":        "1min",
    "timescale": {
        "host":        "localhost",
        "port":        5432,
        "dbname":      "forex_ticks",
        "user":        "forex_user",
        "tick_table":  "tick_data",
        "bar_table":   "ohlcv_bars",
        "chunk_interval": "1 day",
    },
    "kafka": {
        "bootstrap_servers": "localhost:9092",
        "tick_topic":        "forex.ticks",
        "news_topic":        "forex.news",
        "cross_asset_topic": "forex.cross_asset",
        "consumer_group":    "forex_scaler",
        "batch_size":        1000,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = {
    "atr_window":           6,             # 6-period ATR (spec)
    "ofi_window":           20,
    "trade_arrival_window": 30,            # Trade Arrival Rate window
    "rsi_period":           14,
    "macd_fast":            12,
    "macd_slow":            26,
    "macd_signal":          9,
    "bollinger_window":     20,
    "bollinger_std":        2.0,
    "lag_windows":          [5, 20, 60],
    "sentiment_decay_lambda": 0.1,
    "buzz_window_minutes":  5,
}

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-ASSET INPUTS
# ─────────────────────────────────────────────────────────────────────────────
CROSS_ASSET = {
    "enabled": True,
    "assets": {
        "US10Y":    {"ticker": "^TNX",  "related_pair": "USDJPY",  "lag_bars": [1, 5]},
        "WTI":      {"ticker": "CL=F",  "related_pair": "all",     "lag_bars": [1, 5]},
        "COPPER":   {"ticker": "HG=F",  "related_pair": "all",     "lag_bars": [1, 5]},
        "IRON_ORE": {"ticker": "TIO=F", "related_pair": "all",     "lag_bars": [1, 5]},
        "VIX":      {"ticker": "^VIX",  "related_pair": "all",     "lag_bars": [1, 5]},
    },
    "correlation_window":  60,
    "gnn_update_freq":     5,
}

# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT / NEWS  — DUAL STREAM
# ─────────────────────────────────────────────────────────────────────────────
SENTIMENT = {
    "enabled":              True,
    "offline_model":        "ProsusAI/finbert",
    "embedding_dim":        768,
    "finbert_proj_dim":     8,  # WIRE-008: PCA projection dimension for FinBERT embeddings
    "embedding_cache":      PATHS["data_embeddings"],
    "online_model":         "mistralai/Mistral-7B-Instruct-v0.2",
    "slm_max_tokens":       64,
    "global_brain_update_sec": 60,
    "decay_lambda":         0.1,
    "buzz_window_min":      5,
    "eco_calendar_enabled": True,
    "high_impact_events":   ["NFP", "CPI", "FOMC", "GDP", "PMI", "ECB", "BOE"],
    "news_api":             "alpha_vantage",
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURES  (defined in config/models.py, re-exported here)
# ─────────────────────────────────────────────────────────────────────────────

TRAINING = {
    # Defaults mirrored from config/run.yaml (YAML wins when --config is used).
    "batch_size":       512,
    "epochs":           6,
    "patience":         3,   # must be < post-warmup epochs so early-stop can fire
    "loss":             "sharpe_huber",  # matches config/run.yaml
    "huber_delta":      1.0,
    "asymmetric_sign_weight": 2.0,
    "grad_clip":        0.75,
    "weight_decay":     0.001,
    "amp":              True,
    "val_split":        0.2,
    "seq_len":          80,              # matches config/run.yaml + curriculum
    "lr_warmup_epochs": 2,               # matches config/run.yaml training.lr_warmup_epochs
    "lr_schedule":      "warmup_cosine",
    "checkpoint_dir":   PATHS["checkpoints"],
    "walk_forward_folds": 6,
    "early_stop_metric": "sharpe",
    "sharpe_annualization_factor": 325.0,  # matches config/run.yaml (FX minute bars)
    "onecycle_pct_start": 0.1,
    "onecycle_max_lr_mult": 10.0,
    # Gradient accumulation: effective_batch = batch_size × grad_accum_steps
    "grad_accum_steps": 4,
    # Stochastic Weight Averaging: averages weights over last 25% of training
    "swa_enabled":      True,
    "swa_start_frac":   0.75,   # start at 75% of total epochs
    "swa_lr":           1e-5,   # constant LR during SWA phase
}

# Presets for local machines (use: python training/train_gpu.py --hardware-profile <name>)
# RTX 4060 = 8GB VRAM; pair with 16GB system RAM → keep workers/prefetch low to avoid host OOM.
HARDWARE_PROFILES = {
    "rtx_4060_16gb_ram": {
        "batch_size":         212,
        # ZarrStreamDataset fix applied: multiprocessing is now safe and fast on Windows.
        "num_workers":        4,
        "chunk_size":         250_000,
        "prefetch_factor":    4,
        "local_project_paths": True,
    },
    "rtx_4000_ada_cloud": {
        # RTX 4000 Ada Generation — 20GB VRAM, cloud pod (RunPod / Vast.ai)
        "batch_size":         8192,
        "num_workers":        2,
        "chunk_size":         500_000,
        "prefetch_factor":    2,
        "local_project_paths": False,
    },
    "a5000_24gb": {
        # RTX A5000 24GB — good cloud/workstation balance for long runs
        "batch_size":         1536,
        "num_workers":        8,
        "chunk_size":         500_000,
        "prefetch_factor":    4,
        "local_project_paths": False,
    },
    "a40_48gb": {
        # NVIDIA A40 48GB — high VRAM cloud profile tuned for stable throughput
        "batch_size":         384,
        "num_workers":        6,
        "chunk_size":         500_000,
        "prefetch_factor":    4,
        "local_project_paths": False,
    },
    # ── Ubuntu laptop profiles ────────────────────────────────────────────────
    "ubuntu_rtx_laptop": {
        # Generic Ubuntu laptop: RTX 30/40-series, 8–16 GB VRAM.
        # Workers capped at 4 to avoid RAM pressure from multiprocessing.spawn.
        # chunk_size kept small for thermal safety; raise to 400k if temps stay < 80 °C.
        # local_project_paths=False: honours paths.* from run_ubuntu.yaml so data
        # stored outside the repo (e.g. ~/forex_data/) is found correctly.
        "batch_size":         212,
        "num_workers":        4,
        "chunk_size":         250_000,
        "prefetch_factor":    4,
        "local_project_paths": False,
        # Linux local FS: lz4@1 for Zarr training-read throughput.
        "zarr_cname":         "lz4",
        "zarr_clevel":        1,
    },
    "ubuntu_rtx4070_laptop": {
        # RTX 4070 Laptop (12 GB VRAM, Ada Lovelace) — BF16 Tensor Cores kick in.
        # boost batch size since BF16 halves VRAM per activation.
        "batch_size":         1024,
        "num_workers":        6,
        "chunk_size":         350_000,
        "prefetch_factor":    2,
        "local_project_paths": True,
        "zarr_cname":         "lz4",
        "zarr_clevel":        1,
    },
    "ubuntu_rtx4090_desktop": {
        # RTX 4090 desktop on Ubuntu — full throughput, BF16, large batch.
        "batch_size":         2048,
        "num_workers":        8,
        "chunk_size":         500_000,
        "prefetch_factor":    4,
        "local_project_paths": True,
        "zarr_cname":         "lz4",
        "zarr_clevel":        1,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# CONTRASTIVE PRE-TRAINING
# ─────────────────────────────────────────────────────────────────────────────
PRETRAIN = {
    "enabled":          True,
    "method":           "byol",           # BYOL: no negatives, 1 grad pass, works on 8 GB VRAM
    "temperature":      0.5,              # used only if method="tscl"
    "projection_dim":   256,              # BYOL projector hidden/output dim
    "pred_dim":         128,              # BYOL predictor hidden dim
    "ema_decay":        0.996,            # target-network EMA decay (BYOL)
    "sample_windows":   "auto",
    "blocks_per_epoch": "auto",
    "read_windows":     64,               # Zarr span read chunk size (config/run.yaml pretrain.read_windows)
    "mask_prob":        0.20,             # used only if method="masked"
    "recon_hidden_dim": 512,              # masked / VAE / forecast decoder hidden size
    "latent_dim":       64,               # VAE latent size (method=vae)
    "vae_beta":         0.001,            # KL weight (method=vae)
    "n_clusters":       3,                # k-means clusters (method=cluster)
    "forecast_horizon": 5,                # future bars to predict (method=forecast)
    "drift_margin":     1.0,              # embedding distance target (method=drift)
    "augmentations": {
        "jitter_std":       0.01,
        "scaling_range":    (0.8, 1.2),
        "permutation_segs": 4,
        "dropout_prob":     0.1,
    },
    "pretrain_loss":    "huber",
    # YAML uses pretrain.epochs / min_epochs; keep pretrain_epochs as the CLI alias.
    "epochs":           3,                # matches config/run.yaml pretrain.epochs
    "min_epochs":       1,                # matches config/run.yaml pretrain.min_epochs
    "pretrain_epochs":  3,
    "pretrain_lr":      1e-4,
    "pretrain_batch":   256,
    "checkpoint":       PATHS["file_contrastive_encoder"],
}

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CACHE — slow-changing columns (sentiment, COT, macro, hurst)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_CACHE = {
    "enabled": True,
    "ofi_z_threshold": 2.0,
    "regime_window": 60,
    "slow_cols": [
        "sentiment_decayed",
        "eco_surprise",
        "hurst_exponent",
        "cot_net_hf",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL MATURITY — gates live promotion (dev → paper → production)
# ─────────────────────────────────────────────────────────────────────────────
MATURITY = {
    "stage": "paper",  # config/run.yaml maturity.stage
}

# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE DISTILLATION (SCALE MODEL)
# ─────────────────────────────────────────────────────────────────────────────
DISTILLATION = {
    "teacher_model":    "mamba",
    "teacher_ckpt":     BEST_CHECKPOINTS["mamba"],
    "student_model":    "haelt",
    "alpha":            0.5,
    "temperature":      2.0,
}

# Ensemble meta-learner defaults. config/run.yaml overrides these for full runs.
ENSEMBLE = {
    "enabled":              False,
    "epochs":               10,
    "div_weight":           0.1,
    "explicit_diversity":   False,
    "member_seed_offset":   997,
    "member_lr_jitter":     0.0,
    "member_dropout_jitter": 0.0,
    "checkpoint_best":      PATHS["ensemble_meta_best"],
    "checkpoint_latest":    PATHS["ensemble_meta_latest"],
    "checkpoint_final":     PATHS["ensemble_meta_final"],
}

BACKTEST = {
    "min_confidence":       0.45,
    "lots":                 0.1,
    "stop_pips":            12.0,
    "take_profit_pips":     18.0,
    "atr_stop_mult":        1.2,  # matches config/run.yaml
    "atr_take_profit_mult": 1.8,
    "slippage_pips":        0.7,
    "commission_per_lot":   3.5,
    "execution_delay_bars": 1,
    "mc_sims":              500,
}

# ─────────────────────────────────────────────────────────────────────────────
# RL AGENT  — PPO + DQN, 3-ACTION
# ─────────────────────────────────────────────────────────────────────────────
RL = {
    "algorithms":   ["PPO", "DQN"],
    "action_space": 3,                     # 0=Buy, 1=Hold, 2=Sell
    "ppo": {
        "gamma":        0.99,
        "lr":           3e-4,
        "clip_epsilon": 0.2,
        "entropy_coeff": 0.01,
        "value_coeff":  0.5,
        "n_steps":      2048,
        "n_epochs":     10,
        "gae_lambda":   0.95,
        # Optional temporal backbone: LSTM over the last `hist_len` obs.
        "use_lstm":     False,
        "lstm_hidden":  128,
        "hist_len":     32,
    },
    "dqn": {
        "gamma":         0.99,
        "lr":            1e-4,
        "eps_start":     1.0,
        "eps_end":       0.01,
        "eps_decay":     0.995,
        "buf_size":      1_000_000,
        "batch":         64,
        "target_update": 100,
        "double_dqn":    True,
    },
    "reward": {
        "pnl_weight":               1.0,
        "drawdown_penalty":         0.5,
        "transaction_cost_penalty": 0.3,
        "overtrade":                0.25,  # matches config/run.yaml
        "holding_cost":             0.01,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# LABELING — RL REWARD SIGNAL
# ─────────────────────────────────────────────────────────────────────────────
LABELING = {
    "method":               "rl_reward",
    "lookahead_bars":       30,  # synced with strategy.lookahead_bars in config/run.yaml

    # Defaults match config/run.yaml strategy.profit_target_atr / stop_loss_atr
    "profit_target_atr":    1.2,
    "stop_loss_atr":        0.8,
    "transaction_cost_pips": 1.5,
    "pip_size":             0.0001,
    # Triple-barrier: Numba-accelerated scans (parallel over bars; auto fallback if Numba missing)
    "tbm_numba":            True,
    "tbm_parallel":       True,
}

# ─────────────────────────────────────────────────────────────────────────────
# PIP SIZES — per-asset pip resolution lookup
# ─────────────────────────────────────────────────────────────────────────────
PIP_SIZES = {
    "default": 0.0001,
    "JPY":     0.01,     # All JPY pairs (USDJPY, EURJPY, GBPJPY, etc.)
    "XAU":     0.1,      # Gold
    "XAG":     0.001,    # Silver
    "BTC":     1.0,      # Bitcoin
    "ETH":     0.01,     # Ethereum
}

def get_pip_size(pair: str) -> float:
    """Look up pip size for a currency pair. Checks quote currency first, then base."""
    pair_upper = str(pair).upper().replace("/", "").replace("_", "")
    # Check quote currency (last 3 chars)
    quote = pair_upper[-3:] if len(pair_upper) >= 6 else pair_upper
    if quote in PIP_SIZES:
        return PIP_SIZES[quote]
    # Check base currency (first 3 chars)
    base = pair_upper[:3] if len(pair_upper) >= 6 else ""
    if base in PIP_SIZES:
        return PIP_SIZES[base]
    return PIP_SIZES["default"]


def price_to_pips(price_diff: float, pair: str) -> float:
    """Convert a raw price difference to pips for ``pair``."""
    pip = get_pip_size(pair)
    return float(price_diff) / pip if pip > 0 else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# SIZING + SCALING STRATEGY (BOTH COMBINED)
# ─────────────────────────────────────────────────────────────────────────────
SIZING = {
    "method":           "fractional_kelly",
    "kelly_fraction":   0.25,
    "max_position_pct": 0.05,
    "target_annual_vol": 0.10,
    "pip_risk":         20.0,
    "scaling_strategy": "both",            # pyramid winners + martingale losers
    "pyramid_add_pct":  0.25,
    "martingale_add_pct": 0.25,
    "max_total_lots":   3.0,
    "scale_out_targets": [0.25, 0.50, 0.75],
}

# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGEMENT — DYNAMIC STOP LOSS
# ─────────────────────────────────────────────────────────────────────────────
RISK = {
    "stop_type":            "dynamic",
    "atr_multiplier":       1.5,
    "trail_activation_r":   1.0,
    "breakeven_at_r":       0.5,
    "max_drawdown_halt":    0.10,
    "daily_loss_limit":     0.03,
    # ── RiskEngine (risk/risk_engine.py) pre-trade / post-trade limits ─────
    "max_notional_usd":            250_000.0,
    "max_order_freq_per_min":      10,
    "max_instrument_concentration": 0.50,
    "var_confidence":              0.99,
    "var_window":                  500,
    "cvar_multiplier":             1.5,
    "gap_move_threshold":          0.02,
    "require_approval_on_flatten": False,
}

# ─────────────────────────────────────────────────────────────────────────────
# TRADE FILTERS — BOTH ENABLED
# ─────────────────────────────────────────────────────────────────────────────
FILTERS = {
    "vol_filter_enabled":   True,
    "vol_multiplier":       3.0,
    "vol_lookback":         60,
    "news_filter_enabled":  True,
    "news_buffer_minutes":  15,
    "sentiment_threshold":  0.6,
    "dust_settle_bars":     3,
}

# ─────────────────────────────────────────────────────────────────────────────
# LATENCY — TIP-SEARCH
# ─────────────────────────────────────────────────────────────────────────────
LATENCY = {
    "strategy":             "tip_search",
    "fast_model":           "dqn",         # ~2ms
    "slow_model":           "haelt",       # ~5ms
    "fast_latency_ms":      2.0,
    "slow_latency_ms":      5.0,
    "switch_threshold_mult": 2.0,          # Use fast if ATR > 2× avg
    "max_acceptable_ms":    10.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION — EMBARGOING + PURGED K-FOLD
# GPU path maps validation.* from YAML via gpu_cli (_YAML_MAP) onto
# args.validation_* ; settings stubs are the no-YAML fallback. Keep synced to run.yaml.
# ─────────────────────────────────────────────────────────────────────────────
VALIDATION = {
    "method":               "purged_embargo",
    "n_splits":             7,
    "purge_bars":           120,
    "embargo_bars":         60,
    "min_train_size":       50_000,
    "train_window_bars":    50_000,
    "test_window_bars":     10_000,
}

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING + DRIFT DETECTION
# ─────────────────────────────────────────────────────────────────────────────
MONITORING = {
    "enabled":              True,
    "drift_window":         1000,
    "psi_threshold":        0.2,
    "ks_pvalue_threshold":  0.05,
    "sharpe_drop_threshold": 0.5,
    "check_freq_bars":      500,
    "retrain_trigger":      "auto",
    "retrain_cooldown_sec": 21600,  # 6h between auto-retrain spawns from live engine
    "wandb_project":        "forex-scaling-model",
}

# ─────────────────────────────────────────────────────────────────────────────
# RETRAINING — WALK-FORWARD ROLLING
# ─────────────────────────────────────────────────────────────────────────────
RETRAINING = {
    "strategy":             "walk_forward_rolling",
    "retrain_every_bars":   10_000,
    "rolling_window_bars":  50_000,
    "warm_start":           True,
    "auto_deploy":          True,
    "min_improvement":      0.05,
}

# ─────────────────────────────────────────────────────────────────────────────
# INFRASTRUCTURE — KAFKA + TIMESCALEDB
# ─────────────────────────────────────────────────────────────────────────────
INFRA = {
    "kafka_enabled":        True,
    "timescale_enabled":    True,
    "log_dir":              PATHS["logs"],
    "checkpoint_dir":       PATHS["checkpoints"],
    "data_dir":             PATHS["data"],
}

# ─────────────────────────────────────────────────────────────────────────────
# GPU / AMP / PRECISION
# ─────────────────────────────────────────────────────────────────────────────
# amp_dtype: "auto" | "bf16" | "fp16" | "fp32"
#   auto  → force BF16 on all Ampere+ (CC ≥ 8.0); FP16 on older GPUs; FP32 on CPU.
#           BF16 has full FP32 dynamic range and needs no GradScaler.
#   bf16  → BF16 Tensor Cores (falls back to FP16 if unsupported)
#   fp16  → FP16 Tensor Cores, GradScaler required to prevent underflow
#   fp32  → full precision (baseline / debugging)
#
# thermal_limit_celsius: pause training when GPU exceeds this temperature.
#   83 °C  is a safe ceiling for laptop GPUs (throttles start ~87 °C on most).
#   Set to 0 to disable thermal throttle (e.g. desktops with good cooling).
#
# torch_compile: enable torch.compile() by default for ~20–30 % extra
#   throughput on PyTorch ≥ 2.0 + CUDA + Triton. Adds ~60 s compile time on
#   first run. LSTM/GRU/RNN cells are left eager (torch.compiler.disable) so
#   the rest of the model can still use inductor / reduce-overhead.
#   Set to false to force eager mode.
GPU = {
    "amp_dtype":              "auto",
    "thermal_limit_celsius":  83,
    "torch_compile":          True,
    "torch_compile_mode":     "reduce-overhead",  # "default" | "reduce-overhead" | "max-autotune"
    "allow_tf32":             True,
    "cudnn_benchmark":        True,
}

# GOVERNANCE — used by live demotion monitor, deployment promotion gates,
# and capital-efficiency checks (see trading/live_engine.py, infrastructure/deployment.py).
GOVERNANCE = {
    'min_sharpe_promote':    1.5,
    'min_sharpe_emergency':  1.3,
    'min_profit_factor':     1.5,
    'max_drawdown_pct':      0.20,
    'min_trades':            500,
    'max_regime_concentration': 0.75,
    'max_cost_pct_gross_pnl':   0.30,
    'strict_psr':            False,
    'demotion_sharpe_floor': 0.5,
    'demotion_winrate_floor': 0.45,
    'demotion_window_trades': 300,
    'page_hinkley_delta':    0.005,
    'page_hinkley_lambda':   50.0,
    'mlflow_tracking_uri':   'http://localhost:5000',
    'mlflow_experiment':     'forex-scaling-model',
    # K: Capital efficiency gates (predict live survivability > raw Sharpe)
    'min_sharpe_per_turnover':  0.10,  # Sharpe / trades_per_day
    'min_sharpe_per_drawdown':  8.0,   # Sharpe / max_drawdown  (Calmar-flavoured)
    'min_sharpe_per_latency':   0.05,  # Sharpe / avg_latency_ms
}

# ALERTS — Discord webhook + Prometheus exporter for live_engine / demotion /
# promotion / retrain / TCA events (DISCORD_WEBHOOK_URL from env).
ALERTS = {
    "discord_webhook_url": os.getenv("DISCORD_WEBHOOK_URL", ""),
    "discord_min_interval_s": 300,
    "discord_environment": "production",
    "prometheus_port": 8000,
    "prometheus_enabled": True,
    "alert_on_circuit_breaker": True,
    "alert_on_drift": True,
    "alert_on_promotion": True,
    "alert_on_demotion": True,
    "alert_on_retrain": True,
    "alert_on_tca_breach": True,
}

# MACRO_DATA — FRED/AlphaVantage keys + tick-clean / gap policy consumed by
# features/macro_features.py, MacroMaterializer, and data cleaning paths.
MACRO_DATA = {
    "fred_api_key": os.getenv("FRED_API_KEY", ""),
    "yield_momentum_windows": [5, 20],
    "yield_vol_window": 20,
    "av_api_key": os.getenv("ALPHA_VANTAGE_API_KEY", ""),
    "eco_news_buffer_minutes": 15,
    "ollama_url": "http://localhost:11434",
    "ollama_model": "mistral",
    "bad_tick_z_thresh": 8.0,
    "bad_tick_window": 60,
    # PIPE-010: MAD-based robust tick cleaning (robust to non-Gaussian FX returns).
    "bad_tick_mad_z_thresh": 6.0,       # z = 0.6745*(x-median)/MAD above which a tick is an outlier
    "bad_tick_spread_ratio": 8.0,       # flag ticks whose spread > N x rolling median spread
    "bad_tick_spread_window": 120,      # rolling window for the median spread baseline
    # Gap detection / interpolation policy for bar frames.
    "gap_policy": "drop",               # "drop" | "ffill" | "interpolate"
    "gap_max_minutes": 5,               # only consider gaps up to this size for ffill/interpolate
}

# ─────────────────────────────────────────────────────────────────────────────
# MATURITY LADDER  —  Paper → Shadow → Live (small) → Scale
# Single source of truth for promotion / demotion thresholds per stage.
# ─────────────────────────────────────────────────────────────────────────────
MATURITY_LADDER = {
    # Stage definitions — each stage has its own risk envelope and promotion gate.
    "stages": ["paper", "shadow", "live_small", "scale"],

    "paper": {
        "description":        "Full diagnostics, pessimistic execution, no tuning after seeing results",
        "capital_pct":        0.0,        # no real capital
        "max_position_pct":   0.0,        # no real positions
        "kelly_fraction":     0.0,
        "circuit_breaker_dd": 1.0,        # disabled — paper runs to completion
        "daily_loss_limit":   1.0,
        # Promotion requires consistent edge before advancing
        "promote_min_sharpe": 1.5,
        "promote_min_trades": 500,
        "promote_min_profit_factor": 1.5,
        "promote_max_dd_pct": 0.15,
        "promote_regime_consistent": True,  # edge must hold across regime buckets
    },

    "shadow": {
        "description":        "Live data, no capital, promotion only on regime-consistent edge",
        "capital_pct":        0.0,
        "max_position_pct":   0.0,
        "kelly_fraction":     0.0,
        "circuit_breaker_dd": 1.0,
        "daily_loss_limit":   1.0,
        "promote_min_sharpe": 1.6,
        "promote_min_trades": 300,
        "promote_min_profit_factor": 1.6,
        "promote_max_dd_pct": 0.12,
        "promote_regime_consistent": True,
        # Shadow must match paper equity curve within tolerance
        "paper_tracking_error_max": 0.05,
    },

    "live_small": {
        "description":        "Minimal size, aggressive circuit breakers, daily reconciliation",
        "capital_pct":        0.10,       # 10 % of available capital
        "max_position_pct":   0.01,       # 1 % per trade
        "kelly_fraction":     0.10,       # very conservative
        "circuit_breaker_dd": 0.05,       # halt at 5 % drawdown
        "daily_loss_limit":   0.01,       # 1 % daily loss limit
        "max_total_lots":     0.5,
        "session_limits": {
            "asia":        {"max_lots": 0.2, "max_open_trades": 1},
            "london":      {"max_lots": 0.5, "max_open_trades": 2},
            "ny":          {"max_lots": 0.5, "max_open_trades": 2},
            "asia_london": {"max_lots": 0.3, "max_open_trades": 1},
            "london_ny":   {"max_lots": 0.5, "max_open_trades": 2},
            "off":         {"max_lots": 0.0, "max_open_trades": 0},  # no trading in off hours
        },
        "promote_min_sharpe": 1.4,
        "promote_min_trades": 200,
        "promote_stability_weeks": 4,   # must be stable for 4 weeks before scaling
        "promote_max_dd_pct": 0.04,
    },

    "scale": {
        "description":        "Only after stability; size increases slowly; no logic changes during scale-up",
        "capital_pct":        1.0,
        "max_position_pct":   0.05,       # up to 5 % per trade (matches SIZING)
        "kelly_fraction":     0.25,       # matches SIZING.kelly_fraction
        "circuit_breaker_dd": 0.10,       # matches RISK.max_drawdown_halt
        "daily_loss_limit":   0.03,       # matches RISK.daily_loss_limit
        "max_total_lots":     3.0,        # matches SIZING.max_total_lots
        "size_increase_pct_per_week": 0.10,  # max 10 % size increase per week
        "freeze_logic_during_scale": True,    # no code changes while scaling
        "session_limits": {
            "asia":        {"max_lots": 1.0, "max_open_trades": 3},
            "london":      {"max_lots": 3.0, "max_open_trades": 5},
            "ny":          {"max_lots": 3.0, "max_open_trades": 5},
            "asia_london": {"max_lots": 1.5, "max_open_trades": 3},
            "london_ny":   {"max_lots": 3.0, "max_open_trades": 5},
            "off":         {"max_lots": 0.5, "max_open_trades": 1},
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# LIVE_RISK  —  one source of truth shared by paper / backtest / shadow / live
# All environments import from here. Never hardcode risk parameters elsewhere.
# ─────────────────────────────────────────────────────────────────────────────
LIVE_RISK = {
    # ── Position sizing (mirrors SIZING for backward compat) ──────────────────
    "kelly_fraction":       0.25,         # fractional Kelly multiplier
    "max_position_pct":     0.05,         # max single-position size as fraction of equity
    "max_total_lots":       3.0,
    "target_annual_vol":    0.10,
    "pip_risk_default":     20.0,         # default pip stop (used when ATR unavailable)

    # ── Hard risk limits ──────────────────────────────────────────────────────
    "max_drawdown_halt":    0.10,         # halt trading at 10 % drawdown
    "soft_drawdown_reduce": 0.05,         # reduce size 50 % at 5 % drawdown
    "daily_loss_limit":     0.03,         # close all at 3 % daily loss
    "max_consecutive_losses": 5,          # reduce size after N consecutive losses
    "recovery_bars":        20,           # bars to wait after halt before resuming

    # Session exposure limits (applied universally: paper + live).
    # max_open_trades counts independent open position tickets/legs. Scale-in
    # and scale-out actions modify lots on an existing ticket and are governed
    # by max_lots plus execution-cost controls.
    # PIPE-005 / ISSUE-005: Session windows now use zoneinfo tz-aware boundaries
    # that respect DST, defined by local open and close times.
    "session_limits": {
        "asia":        {"max_lots": 1.0, "max_open_trades": 3, "hours_local": (time(9, 0), time(18, 0)),  "tz": ZoneInfo("Asia/Tokyo")},
        "london":      {"max_lots": 3.0, "max_open_trades": 5, "hours_local": (time(8, 0), time(16, 30)), "tz": ZoneInfo("Europe/London")},
        "ny":          {"max_lots": 3.0, "max_open_trades": 5, "hours_local": (time(9, 30), time(16, 0)), "tz": ZoneInfo("America/New_York")},
        # Overlap policy keys (DST SoT); no private "overlap" string.
        "asia_london": {"max_lots": 1.5, "max_open_trades": 3},
        "london_ny":   {"max_lots": 3.0, "max_open_trades": 5},
        "off":         {"max_lots": 0.5, "max_open_trades": 1, "hours_local": (time(18, 0), time(8, 0)), "tz": None},
    },

    # ── ATR stop parameters (mirrors RISK) ────────────────────────────────────
    "atr_multiplier":       1.5,
    "trail_activation_r":   1.0,
    "breakeven_at_r":       0.5,

    # ── Regime scaling limits ─────────────────────────────────────────────────
    "regime_scale": {
        "crisis":   0.50,   # 50 % of normal size in correlation crisis
        "trending": 1.20,   # 20 % bonus in trending regime
        "mean_rev": 0.75,   # 25 % reduction in mean-reversion regime
        "normal":   1.00,
    },
    "corr_crisis_threshold": 0.70,
    "hurst_trending":        0.60,
    "hurst_mean_rev":        0.40,
}

# ─────────────────────────────────────────────────────────────────────────────
# MULTI-SCALE FEATURES  —  windows used across feature engineering
# Short / medium / long triples; differences between scales carry the signal.
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_SCALES = {
    "atr_windows":     [6, 20, 60],       # short / medium / long ATR
    "vol_windows":     [6, 20, 60],       # rolling volatility scales
    "ofi_windows":     [5, 20, 60],       # order-flow imbalance scales
    "momentum_windows": [5, 20, 60],      # return momentum
    "vwap_window":     60,                # VWAP lookback for breakout pressure
    "chop_window":     14,                # Choppiness Index lookback
    "corr_window":     60,                # cross-asset correlation window
    "regime_window":   240,               # vol-regime median lookback
    "volatility_window": 120,             # trailing volatility / autocorr window
    "spread_window":   120,               # median spread for liquidity vacuum
    "ofi_z_short":     20,                # OFI fast window for Z-score numerator
    "ofi_z_long":      120,               # OFI slow window for Z-score denominator
    "eigenvalue_window": 60,              # rolling covariance window for eigenvalue concentration
    "vol_of_vol_window": 20,              # vol-of-vol rolling window
    "trend_stability_window": 20,         # Hurst / R/S window for trend stability
}

# ─────────────────────────────────────────────────────────────────────────────
# NO-TRADE ZONE  —  thresholds for explicitly labeling abstain regions
# ─────────────────────────────────────────────────────────────────────────────
NO_TRADE = {
    "vol_regime_low_pct":    0.25,   # bottom 25th percentile of rolling vol → low-vol abstain
    "ofi_z_neutral_band":    0.5,    # |OFI_Z| < 0.5 → no directional pressure
    "trend_stability_min":   0.45,   # Hurst < 0.45 → choppy, no trend
    "liquidity_vacuum_max":  2.0,    # spread > 2× median → thin liquidity, no trade
    "news_buffer_bars":      15,     # bars around high-impact events
    "min_atr_pips":          3.0,    # min ATR in pips; below = dead market
}

# ─────────────────────────────────────────────────────────────────────────────
# LABEL REGIME SCALING  —  wider barriers in high-vol, shorter horizons in MR
# ─────────────────────────────────────────────────────────────────────────────
LABEL_REGIME = {
    # Triple-barrier multipliers per regime
    "barrier_scale": {
        "high_vol":  {"tp_atr_mult": LABELING["profit_target_atr"] * 1.33, "sl_atr_mult": LABELING["stop_loss_atr"] * 1.66, "horizon_mult": 0.7},
        "normal":    {"tp_atr_mult": LABELING["profit_target_atr"], "sl_atr_mult": LABELING["stop_loss_atr"], "horizon_mult": 1.0},
        "low_vol":   {"tp_atr_mult": LABELING["profit_target_atr"] * 0.66, "sl_atr_mult": LABELING["stop_loss_atr"] * 0.88, "horizon_mult": 1.3},
        "mean_rev":  {"tp_atr_mult": LABELING["profit_target_atr"] * 0.66, "sl_atr_mult": LABELING["stop_loss_atr"] * 0.88, "horizon_mult": 0.5},
        "trending":  {"tp_atr_mult": LABELING["profit_target_atr"] * 1.33, "sl_atr_mult": LABELING["stop_loss_atr"] * 1.33, "horizon_mult": 1.5},
    },
    # Vol regime boundaries (rolling vol percentiles)
    "high_vol_pct": 0.75,
    "low_vol_pct":  0.25,
    # Session-specific transaction cost adjustments (cost first; horizon second).
    # Overlap keys used when DST-aware asia_london / london_ny flags are set.
    "session_cost_scale": {
        "asia":         1.2,   # wider spreads in Asia
        "london":       0.9,   # tighter spreads at London open
        "ny":           0.9,
        "asia_london":  1.0,   # Asia/London overlap — mid liquidity
        "london_ny":    0.85,  # most liquid overlap
        "off":          1.5,   # very wide off-hours
    },
    # Optional session → label-horizon gate (overlaps/liquid slightly shorter;
    # asia/off slightly longer). Multiplied with barrier_scale horizon_mult.
    "session_horizon_mult": {
        "asia":         1.15,
        "london":       1.0,
        "ny":           1.0,
        "asia_london":  0.95,
        "london_ny":    0.90,
        "off":          1.20,
    },
    # Wide spreads → shorter horizon (liquidity risk). Applied after session.
    "spread_horizon": {
        "z_wide":       1.5,   # spread_z above this → shorten
        "wide_mult":    0.75,
        "z_extreme":    2.5,   # very wide → much shorter
        "extreme_mult": 0.50,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# CURRICULUM TRAINING  —  variable sequence length + feature freeze schedule
#
# Schedule stubs must stay in sync with config/run.yaml (active). Feature name
# lists live only in YAML — settings carries epoch_unfreeze / always_on only.
# audit_settings_yaml_curriculum_drift() fails validate_run_config on drift.
# ─────────────────────────────────────────────────────────────────────────────
CURRICULUM = {
    # A3: Sequence-length schedule (bars per sample) — matches run.yaml
    "seq_schedule": [
        {"epoch_start": 0, "seq_len": 80},
    ],
    # A4: Feature group freeze schedule (unfreeze at epoch_unfreeze)
    # Canonical feature lists: config/run.yaml → curriculum.feature_groups
    "feature_groups": {
        "core":             {"epoch_unfreeze": 0,  "always_on": True},
        "microstructure":   {"epoch_unfreeze": 0,  "always_on": True},
        "momentum":         {"epoch_unfreeze": 0,  "always_on": True},
        "session":          {"epoch_unfreeze": 0,  "always_on": True},
        "execution_cost":   {"epoch_unfreeze": 1,  "always_on": False},
        "volatility":       {"epoch_unfreeze": 2,  "always_on": False},
        "cross_asset":      {"epoch_unfreeze": 3,  "always_on": False},
        "news":             {"epoch_unfreeze": 3,  "always_on": False},
        "macro":            {"epoch_unfreeze": 4,  "always_on": False},
        "market_regime":    {"epoch_unfreeze": 4,  "always_on": False},
        "higher_timeframe": {"epoch_unfreeze": 5,  "always_on": False},
        # label_quality: disabled in run.yaml (features not implemented)
    },
    # A2: Chunk-level early stopping
    "chunk_early_stop_patience": 3,     # abort if Sharpe drops for K chunks in a row
    "chunk_early_stop_min_batches": 50,  # minimum batches before evaluating chunk Sharpe

    # B: Difficulty curriculum — per-bar max of session/spread/news/vol signals.
    # Active run.yaml opens at max_difficulty=2 (no staged ramp).
    "difficulty_schedule": [
        {"epoch_start": 0, "max_difficulty": 2},
    ],
    # Two-tier spread thresholds (medium / hard):
    "difficulty_spread_threshold":      1.5,   # spread/median > this → medium (1)
    "difficulty_spread_threshold_hard": 2.0,   # spread/median > this → hard  (2)
}


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION  —  latency, slippage, and infrastructure awareness
# ─────────────────────────────────────────────────────────────────────────────
EXECUTION = {
    # F: Expected execution latency baseline.  Below this we have full edge;
    #    above it the cost of latency eats into the trade edge.
    "latency_baseline_ms":    50,       # ms — typical co-located gateway latency
    "latency_col":            "expected_latency_ms",
    # E: Slippage cost inflation factors.
    # effective_cost = base_cost × (1 + vol_alpha × max(0, vol_ratio - 1)
    #                               + liq_alpha × max(0, spread_ratio - 1))
    # where vol_ratio   = atr_t / rolling_median_atr
    #       spread_ratio = spread_t / median_spread
    "slippage_vol_alpha":     0.5,      # 50 % extra cost per unit excess vol
    "slippage_liq_alpha":     0.3,      # 30 % extra cost per unit excess spread
    "slippage_vol_window":    120,      # bars for rolling median ATR
    "slippage_spread_window": 120,      # bars for rolling median spread
}

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG PREFLIGHT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
try:
    from config.config_schema import LiveRiskSchema, SizingSchema, TrainingSchema

    # Parse dictionaries into validated models at runtime
    _ = TrainingSchema(**TRAINING)
    _ = SizingSchema(**SIZING)
    _ = LiveRiskSchema(**LIVE_RISK)
except ImportError:
    pass
