"""
training/cli -- CLI argument parsing sub-package.

Provides parse_args(), apply_hardware_profile, and all related helpers.
The public API mirrors training/gpu_cli.py (which is now a thin shim).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta

from config.settings import (
    CURRICULUM as SETTINGS_CURRICULUM,
)
from config.settings import (
    DISTILLATION,
    HARDWARE_PROFILES,
    MONITORING,
    PATHS,
    PRETRAIN,
    RL,
    TRAINING,
)
from config.settings import (
    ENSEMBLE as SETTINGS_ENSEMBLE,
)
from config.settings import (
    EXECUTION as SETTINGS_EXECUTION,
)
from config.strategy_profiles import STRATEGY_PROFILES, strategy_profile

# Sub-module re-exports (these are the split pieces)
from .hardware import apply_hardware_profile
from .helpers import (
    _apply_auto_run_dir,
    _build_auto_run_name,
    _collect_cli_profile_overrides,
    _set_global_seed,
    _slug_part,
)
from .profile import (
    _apply_model_profile,
    _apply_training_profile,
    _member_training_args,
    _model_build_args,
    _normalize_architecture_profile,
)
from .resume import (
    _baseline_ablation_completion_status,
    _latest_resumable_fold,
    _load_cv_fold_entry,
    _load_walk_forward_resume_history,
    _model_completion_status,
    _supervised_resume_status,
)
from .sync import (
    _apply_yaml_config,
    _apply_yaml_risk_to_live_risk,
    _resolve_seq_len,
    _sync_runtime_config,
)
from .yaml_map import _YAML_MAP


def parse_args():
    p = argparse.ArgumentParser(description="Forex Model -- 20M Tick GPU Trainer")
    p.set_defaults(curriculum=SETTINGS_CURRICULUM)
    p.set_defaults(execution=SETTINGS_EXECUTION)
    p.set_defaults(risk=None)
    p.set_defaults(sidecar=None)
    p.set_defaults(use_mixup=False)
    p.set_defaults(use_volatility_sampler=False)
    p.add_argument(
        "--config",
        type=str,
        default="config/run.yaml",
        help="Path to a YAML run config (e.g. config/run.yaml). "
        "Values are used as defaults; explicit CLI flags override them.",
    )

    # Strategy profile
    p.add_argument(
        "--strategy-mode",
        type=str,
        default="scalping",
        choices=sorted(STRATEGY_PROFILES.keys()),
        help="Trading horizon profile. scalping=1min fast trades; normal=1h slower trades.",
    )
    p.add_argument(
        "--bar-freq", type=str, default=None, help="Bar frequency for feature/label construction, e.g. 1min, 15min, 1h."
    )
    p.add_argument(
        "--lookahead-bars",
        type=int,
        default=None,
        help="Label forward horizon in bars. Defaults to the selected strategy profile.",
    )
    p.add_argument(
        "--profit-target-atr", type=float, default=None, help="ATR profit barrier for triple-barrier/normal labels."
    )
    p.add_argument(
        "--stop-loss-atr", type=float, default=None, help="ATR stop barrier for triple-barrier/normal labels."
    )

    # Scale
    p.add_argument("--n-ticks", type=int, default=20_000_000, help="Total tick count to train on (default: 20M)")
    p.add_argument("--chunk-size", type=int, default=500_000, help="Ticks per processing chunk (RAM safety valve)")
    p.add_argument(
        "--real-data-window-days",
        type=int,
        default=0,
        help="Days per real-data ingestion window. 0 = auto from --chunk-size.",
    )
    p.add_argument(
        "--window-batch-days",
        type=int,
        default=1,
        help="Group N consecutive date windows into one batch. "
        "Effective window = real_data_window_days * window_batch_days. "
        "Larger batches give features more lookback context (default: 1).",
    )

    # Data source
    p.add_argument(
        "--data-source",
        type=str,
        default="dukascopy",
        choices=["synthetic", "dukascopy", "tds", "lmax_historical", "auto", "databento"],
        help="Which data source to use",
    )
    p.add_argument("--data-start", type=str, default="2008-01-01")
    p.add_argument("--data-end", type=str, default=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"))
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Cap training to the first N samples of the processed cache "
        "(cache is time-ordered; e.g. half of 94,423 = 47,211 uses the "
        "earliest ~9 years). 0 = use all samples. Reuses the cache, no rebuild.",
    )
    p.add_argument("--pair", type=str, default="EURUSD")
    p.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Comma-separated pairs for joint multi-pair training, e.g. EURUSD,GBPUSD,USDJPY. "
        "Overrides --pair when set. Can also be a list in config/run.yaml under data.pairs.",
    )
    p.add_argument(
        "--pair-embed-dim",
        type=int,
        default=0,
        help="Learnable pair embedding size (int). Appended to each pair's features before "
        "the backbone. 0 = disabled (pairs are simply concatenated on the feature axis).",
    )
    p.add_argument(
        "--corr-window",
        type=int,
        default=20,
        help="Short rolling correlation window in bars for MultiPairWrapper cross-pair features. Default: 20.",
    )
    p.add_argument(
        "--corr-window-long",
        type=int,
        default=60,
        help="Long rolling correlation window in bars for MultiPairWrapper. Default: 60.",
    )
    p.add_argument(
        "--momentum-window",
        type=int,
        default=20,
        help="Windowed relative momentum lookback in bars for MultiPairWrapper. Default: 20.",
    )
    p.add_argument(
        "--pair-align",
        type=str,
        default="inner",
        choices=["inner", "outer"],
        help="Timestamp alignment across pairs: inner=common bars only (default), outer=fill missing bars with NaN.",
    )
    p.add_argument(
        "--full-day-data",
        action="store_true",
        help="Dukascopy: load all 24h (00-23 UTC). Default is session-only (07-17 UTC).",
    )

    # Model
    p.add_argument(
        "--model", type=str, default="haelt", choices=["tft", "transformer", "haelt", "mamba", "gnn", "expert", "glm"]
    )
    p.add_argument("--all-models", dest="all_models", action="store_true")

    p.add_argument(
        "--no-all-models",
        dest="all_models",
        action="store_false",
        help="Force a single-model run even if config model.all_models=true.",
    )
    # store_false's default would otherwise leave all_models=True with no flags.
    p.set_defaults(all_models=False)

    p.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model list for --all-models, e.g. transformer,expert. "
        "Empty means every registered supervised architecture.",
    )

    p.add_argument(
        "--div-weight",
        type=float,
        default=0.10,
        help="C: DiversityLoss weight during post-training diversity fine-tuning",
    )
    p.add_argument(
        "--same-role-mult",
        type=float,
        default=2.0,
        help="C: Extra diversity penalty multiplier for same-role model pairs",
    )

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--early-stop-patience", type=int, default=10, help="Stop training if val metric does not improve for this many epochs (0=disabled).")
    p.add_argument(
        "--batch-size", type=int, default=2048, help="Batch size -- 2048 optimal for 20M samples on RTX 4090"
    )
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument(
        "--lr-schedule",
        type=str,
        default="warmup_cosine",
        choices=["onecycle", "warmup_cosine"],
        help="Learning-rate schedule: onecycle (legacy default) or warmup_cosine.",
    )
    p.add_argument("--lr-warmup-epochs", type=int, default=3, help="Warmup epochs used by warmup_cosine schedule.")
    p.add_argument(
        "--lr-warmup-pct",
        type=float,
        default=0.1,
        help="Warmup fraction fallback for warmup_cosine when warmup_epochs <= 0.",
    )
    p.add_argument(
        "--lr-min-ratio",
        type=float,
        default=0.05,
        help="Final LR ratio for warmup_cosine (final_lr = lr * lr_min_ratio).",
    )
    p.add_argument("--onecycle-pct-start", type=float, default=0.1, help="OneCycleLR warmup fraction (legacy path).")
    p.add_argument(
        "--onecycle-max-lr-mult",
        type=float,
        default=10.0,
        help="OneCycleLR peak multiplier over base lr (legacy path).",
    )
    p.add_argument("--seq-len", type=str, default="120")
    p.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Global random seed (A-M3: seeded by default for reproducibility). "
        "Pass a different int to vary runs; threads into numpy/torch/augmenter RNGs.",
    )
    p.add_argument(
        "--deterministic",
        dest="deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="A-M3: force fully deterministic kernels (cudnn.deterministic=True, "
        "benchmark=False, torch.use_deterministic_algorithms). Slower but reproducible.",
    )
    p.add_argument("--val-split", type=float, default=0.1, help="Validation fraction (default 0.1)")
    p.add_argument(
        "--tune-split",
        type=float,
        default=0.05,
        help="Fraction reserved for auto-tune evaluation, separate from val (default 0.05). "
        "Set to 0 to disable three-way split (reverts to val reuse).",
    )
    p.add_argument(
        "--curriculum-gate-metric",
        type=str,
        default="train_loss",
        choices=["train_loss", "val_sharpe"],
        help="Metric used for curriculum progression gating. 'train_loss' (default) "
        "prevents val set leakage into curriculum decisions (SYS-005). "
        "'val_sharpe' restores legacy behavior.",
    )
    p.add_argument(
        "--curriculum-manager",
        action="store_true",
        default=False,
        help="Enable the unified CurriculumManager (difficulty + self-paced + "
        "loss-weighted + adaptive) as an extra per-epoch sample filter "
        "(Improvement #4). Default: off (existing curriculum unchanged).",
    )
    p.add_argument(
        "--curriculum-manager-mode",
        type=str,
        default="combined",
        choices=["difficulty", "self_paced", "loss_weighting", "adaptive", "combined"],
        help="CurriculumManager combination mode used with --curriculum-manager.",
    )
    p.add_argument(
        "--curriculum-callback",
        action="store_true",
        default=False,
        help="Build the per-epoch curriculum controller through the "
        "create_curriculum_callback() factory (CustomCurriculumAdapter) "
        "instead of create_curriculum_manager(). Requires --curriculum-manager.",
    )
    p.add_argument(
        "--curriculum-miner-feedback",
        action="store_true",
        help="Enable Online Miner -> Curriculum feedback (forgetting/easy ratios inform curriculum pace).",
    )
    p.add_argument(
        "--curriculum-miner-models",
        type=str,
        default="",
        help="Comma-separated list of models to enable miner feedback for (default: tft,transformer,haelt).",
    )
    p.add_argument(
        "--curriculum-forgetting-threshold",
        type=float,
        default=0.15,
        help="Forgetting rate threshold to freeze curriculum advancement (default: 0.15).",
    )
    p.add_argument(
        "--curriculum-easy-threshold",
        type=float,
        default=0.60,
        help="Easy sample ratio threshold to accelerate curriculum (default: 0.60).",
    )
    p.add_argument(
        "--curriculum-freeze-patience",
        type=int,
        default=1,
        help="Epochs to hold curriculum after freeze trigger (default: 1).",
    )
    p.add_argument(
        "--use-self-paced",
        action="store_true",
        help="Enable SelfPacedLearning curriculum (jointly optimizes model params and sample inclusion).",
    )
    p.add_argument(
        "--use-loss-weighting",
        action="store_true",
        help="Enable LossBasedWeighting curriculum (inverse/focal/threshold/softmax weighting).",
    )
    p.add_argument(
        "--self-paced-pace",
        type=str,
        default="linear",
        choices=["linear", "exponential", "cosine"],
        help="SelfPacedLearning pace schedule (default: linear).",
    )
    p.add_argument(
        "--self-paced-lambda", type=float, default=1.0, help="SelfPacedLearning lambda parameter (default: 1.0)."
    )
    p.add_argument(
        "--loss-weighting-scheme",
        type=str,
        default="focal",
        choices=["inverse", "focal", "threshold", "softmax"],
        help="LossBasedWeighting scheme (default: focal).",
    )
    p.add_argument(
        "--loss-weighting-focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma for LossBasedWeighting (default: 2.0).",
    )
    p.add_argument(
        "--self-paced-models",
        type=str,
        default="",
        help="Comma-separated list of models to enable self-paced for (default: tft,transformer,haelt).",
    )
    p.add_argument(
        "--loss-weighting-models",
        type=str,
        default="",
        help="Comma-separated list of models to enable loss weighting for (default: tft,transformer,haelt).",
    )

    p.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Enable AMP (automatic mixed precision) for faster training. Disabled by default to avoid NaNs.",
    )
    p.add_argument(
        "--no-amp",
        action="store_true",
        default=False,
        dest="no_amp",
        help="Disable AMP -- forces FP32 training. Eliminates NaN-grad skips on 2240-feature inputs at the cost of ~30% slower throughput.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help=(
            "AMP precision dtype. auto=force BF16 on all Ampere+ (CC>=8.0; "
            "full FP32 range, no GradScaler), FP16 on older GPUs. "
            "bf16: same as forced BF16 (falls back to FP16 if unsupported). "
            "fp16: needs GradScaler. fp32: no AMP (debug)."
        ),
    )
    p.add_argument(
        "--cross-asset-mode",
        type=str,
        default="auto",
        choices=["auto", "real", "synthetic", "off"],
        help="Cross-asset features source: auto=real for real FX data, synthetic for synthetic FX; "
        "real=attempt external commodities/yields download; synthetic/off disables external fetch",
    )
    p.add_argument(
        "--cross-asset-provider",
        type=str,
        default="auto",
        choices=["auto", "stooq", "yahoo", "fred", "eodhd"],
        help="Cross-asset data provider. Env CROSS_ASSET_SOURCE overrides this when set.",
    )
    p.add_argument(
        "--sentiment-mode",
        type=str,
        default="finbert",
        choices=["off", "finbert", "auto"],
        help="Sentiment feature mode: finbert=force FinBERT, auto=best available, off=disable sentiment feature columns",
    )
    p.add_argument(
        "--historical-news-mode",
        type=str,
        default="calendar",
        choices=["off", "calendar", "full"],
        help="Offline historical news mode: off=neutral, calendar=economic no-trade events, full=calendar + headline sentiment/counts.",
    )
    p.add_argument(
        "--historical-news-file",
        type=str,
        default=None,
        help="Optional CSV/JSON/JSONL historical headlines file. Defaults to data/raw/news/historical_news_combined.parquet or HISTORICAL_NEWS_FILE.",
    )
    p.add_argument(
        "--economic-calendar-file",
        type=str,
        default=None,
        help="Optional CSV/JSON/JSONL economic calendar file. Defaults to data/raw/eco_calendar/events.csv or ECONOMIC_CALENDAR_FILE.",
    )
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps; effective batch = batch_size * grad_accum_steps",
    )
    p.add_argument(
        "--swa-enabled",
        dest="swa_enabled",
        action=argparse.BooleanOptionalAction,
        default=bool(TRAINING.get("swa_enabled", False)),
        help="Enable Stochastic Weight Averaging over the final training phase.",
    )
    p.add_argument(
        "--swa-start-frac",
        type=float,
        default=float(TRAINING.get("swa_start_frac", 0.75)),
        help="Fraction of total epochs before SWA starts, e.g. 0.75.",
    )
    p.add_argument(
        "--swa-lr",
        type=float,
        default=float(TRAINING.get("swa_lr", 1e-5)),
        help="Constant learning rate used by the SWA scheduler.",
    )
    p.add_argument(
        "--sacs-enabled",
        dest="sacs_enabled",
        action=argparse.BooleanOptionalAction,
        default=bool(TRAINING.get("sacs_enabled", True)),
        help="Enable Sharpness-Aware Checkpoint Selection (SACS).",
    )
    p.add_argument(
        "--sacs-eps",
        dest="sacs_eps",
        type=float,
        default=float(TRAINING.get("sacs_eps", 0.005)),
        help="ε-ball radius for SACS weight perturbations.",
    )
    p.add_argument(
        "--sacs-n-samples",
        dest="sacs_n_samples",
        type=int,
        default=int(TRAINING.get("sacs_n_samples", 5)),
        help="Number of perturbation samples averaged for SACS sharpness estimate.",
    )
    p.add_argument(
        "--sacs-sharpness-weight",
        dest="sacs_sharpness_weight",
        type=float,
        default=float(TRAINING.get("sacs_sharpness_weight", 1.0)),
        help="λ penalty weight on sharpness term in SACS robust score.",
    )
    p.add_argument(
        "--live-feedback-path",
        dest="live_feedback_path",
        type=str,
        default=None,
        help="Path to live_feedback_hard_examples.json written by the retraining orchestrator. "
             "Seeds the online hard-example miner with priority weights from live trading.",
    )
    p.add_argument(
        "--training-framework",
        choices=["custom", "lightning", "composer"],
        default="custom",
        help="Training framework: custom (built-in loop), lightning (PyTorch Lightning), composer (Mosaic Composer).",
    )
    p.add_argument(
        "--rl-framework",
        choices=["custom", "cleanrl", "sb3"],
        default="custom",
        help="RL framework: custom (built-in), cleanrl, sb3 (Stable-Baselines3).",
    )
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--label-method",
        type=str,
        default="rl_reward",
        choices=["rl_reward", "triple_barrier", "cpar"],
        help="Supervised targets: rl_reward=regime-conditional P&L; cpar/triple_barrier=Continuous Path-Adjusted Reward regression",
    )
    p.add_argument(
        "--loss",
        type=str,
        default=None,
        choices=["huber", "asymmetric", "directional_huber", "sharpe_huber"],
        help="huber/asymmetric/directional_huber/sharpe_huber on scalar targets; "
    )
    p.add_argument(
        "--direction-weight",
        type=float,
        default=0.5,
        help="Extra wrong-direction penalty multiplier for directional_huber loss",
    )
    p.add_argument("--sharpe-weight", type=float, default=0.2, help="Sharpe proxy weight for sharpe_huber loss")
    p.add_argument(
        "--sharpe-annualization-factor",
        type=float,
        default=None,
        help="Override the auto-detected Sharpe annualization factor. "
        "Default: auto from bar_freq x lookahead_bars x TRADING_DAYS. "
        "Use only when you know the exact right value.",
    )
    p.add_argument(
        "--fx-full-day",
        action="store_true",
        default=False,
        help="Treat the data as full-day FX (24h) when computing the "
        "Sharpe annualization factor. Without this flag the "
        "factor assumes a 6.5h session profile.",
    )

    p.add_argument(
        "--guard-min-confidence",
        type=float,
        default=0.85,
        help="Minimum confidence required to execute a trade during validation (Disagreement Gating).",
    )
    p.add_argument("--num-workers", type=int, default=8, help="DataLoader workers -- 8 is sweet spot for H100/A100")
    p.add_argument(
        "--prefetch-factor", type=int, default=4, help="DataLoader prefetch (per worker); lower on 16GB RAM PCs"
    )
    p.add_argument(
        "--val-num-workers",
        type=int,
        default=None,
        help="Validation DataLoader workers. Default: auto from train workers.",
    )
    p.add_argument(
        "--val-prefetch-factor",
        type=int,
        default=None,
        help="Validation prefetch factor. Default: auto (lower than train).",
    )
    p.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        default=None,
        help="Force DataLoader pin_memory=True for train/val.",
    )
    p.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Force DataLoader pin_memory=False for train/val.",
    )
    p.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
        default=None,
        help="Force DataLoader persistent_workers=True when workers > 0.",
    )
    p.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="Force DataLoader persistent_workers=False.",
    )
    p.add_argument(
        "--thread-prefetch-batches",
        type=int,
        default=8,
        help="Background-thread prefetch queue depth for train/val loaders "
        "(overlaps Zarr decompress + H2D with GPU compute; only applied "
        "when num_workers==0 or --force-thread-prefetch is set).",
    )
    p.add_argument(
        "--force-thread-prefetch",
        dest="force_thread_prefetch",
        action="store_true",
        default=False,
        help="Layer the daemon-thread prefetch queue on top of DataLoader "
        "worker-process buffering even when num_workers>0. Useful for "
        "hiding GPU-step jitter on slow disks; trades ~2x pinned memory.",
    )
    p.add_argument(
        "--zarr-cname",
        type=str,
        default="auto",
        help="Blosc codec for training-cache Zarr writes "
        "(auto|lz4|zstd|zlib|...). auto = lz4@1 on Linux, zstd@3 elsewhere.",
    )
    p.add_argument(
        "--zarr-clevel",
        type=int,
        default=None,
        help="Blosc compression level (1-9). Default: platform auto (1 on Linux with lz4, 3 with zstd fallback).",
    )
    p.add_argument(
        "--dataset-build-workers",
        type=int,
        default=1,
        help="Parallel threads for loading date windows during dataset build. "
        "1 = sequential (safe default). 2-4 overlaps tick I/O across windows.",
    )
    p.add_argument(
        "--parallel-window-workers",
        type=int,
        default=1,
        help="Parallel processes for date-window feature engineering + labeling. "
        "1 = sequential (default). 2-4 parallelises CPU-heavy chunk builds "
        "across windows using ProcessPoolExecutor.",
    )
    p.add_argument(
        "--hardware-profile",
        type=str,
        default=None,
        choices=list(HARDWARE_PROFILES.keys()) if HARDWARE_PROFILES else None,
        help="Apply tuned defaults (batch/workers/chunk/prefetch/paths). "
        "rtx_4060_16gb_ram: RTX 4060 8GB VRAM + 16GB system RAM",
    )

    # Architecture
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument(
        "--fair-sweep",
        action="store_true",
        help="Architecture bake-off: identical hyperparams from run.yaml for every model "
        "(alias for --no-model-profile).",
    )
    p.add_argument(
        "--model-profile",
        dest="model_profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply per-architecture tuned defaults from config/models.py (default: on). "
        "Use --no-model-profile or --fair-sweep for a fair architecture comparison.",
    )
    p.add_argument(
        "--feature-ablation-name",
        type=str,
        default="",
        help="Name recorded in feature_ablation_report.json for this feature ablation run.",
    )

    p.add_argument(
        "--feature-ablation-drop-groups",
        type=str,
        default="",
        help="Comma-separated curriculum feature groups to zero for this run, e.g. news,cross_asset.",
    )

    p.add_argument(
        "--feature-ablation-keep-groups",
        type=str,
        default="",
        help="Comma-separated curriculum feature groups to keep; all other grouped features are zeroed.",
    )

    p.add_argument(
        "--feature-ablation-drop-features",
        type=str,
        default="",
        help="Comma-separated exact feature names to zero for this run.",
    )

    # Pre-training & Ablation
    p.add_argument("--pretrain", action="store_true", help="Enable contrastive pre-training")
    p.add_argument("--ablate-pretrain", action="store_true", help="Run ablation test on pretraining vs no-pretraining")

    # Pre-training
    p.add_argument(
        "--pretrain-method",
        choices=["byol", "tscl", "masked", "vae", "autoencoder", "cluster", "forecast", "drift", "jepa", "patch_mask", "cross_asset"],
        default=str(PRETRAIN.get("method", "byol")).lower(),
        help="Self-supervised pretrain: byol (default), tscl, masked, vae, cluster, forecast, drift, jepa, patch_mask, cross_asset",
    )
    p.add_argument(
        "--pretrain-framework",
        choices=["custom", "lightly", "solo"],
        default="custom",
        help="Pretraining framework: custom (built-in), lightly (lightly-ssl), solo (solo-learn).",
    )
    p.add_argument("--pretrain-epochs", type=int, default=30)
    p.add_argument(
        "--pretrain-max-epochs",
        type=int,
        default=0,
        help="Hard cap for pretraining epochs. 0 keeps pretrain_epochs unchanged.",
    )
    p.add_argument(
        "--pretrain-min-epochs",
        type=int,
        default=0,
        help="Minimum pretrain epochs before handoff checks can stop early.",
    )
    p.add_argument(
        "--pretrain-handoff-patience",
        type=int,
        default=0,
        help="Stop pretraining early after this many plateau epochs (0 disables).",
    )
    p.add_argument(
        "--pretrain-handoff-min-delta",
        type=float,
        default=0.0,
        help="Minimum pretrain loss improvement to reset handoff patience.",
    )
    p.add_argument(
        "--pretrain-handoff-loss",
        type=float,
        default=float("-inf"),
        help="Stop pretraining once loss <= threshold after min epochs. Disabled by default.",
    )
    p.add_argument(
        "--pretrain-regime",
        action="store_true",
        help="Use regime-aware TSCL: same-regime positives + cross-regime hard negatives",
    )
    p.add_argument(
        "--use-multi-task-pretrainer",
        action="store_true",
        default=False,
        help="Pretrain with the multi-task pretrainer (contrastive + masked recon + "
        "forecast + domain adaptation) from pretrain/multi_task.py instead of the "
        "built-in single-objective trainer (Improvement #3).",
    )
    p.add_argument(
        "--pretrain-lr",
        type=float,
        default=float(PRETRAIN.get("pretrain_lr", 1e-4)),
        help="Pretrain optimizer learning rate",
    )
    p.add_argument(
        "--pretrain-batch",
        type=int,
        default=int(PRETRAIN.get("pretrain_batch", 256)),
        help="Preferred pretrain batch size before VRAM safety cap",
    )
    p.add_argument(
        "--pretrain-projection-dim",
        type=int,
        default=int(PRETRAIN.get("projection_dim", 256)),
        help="Projection dimension for BYOL/TSCL heads",
    )
    p.add_argument(
        "--pretrain-pred-dim",
        type=int,
        default=int(PRETRAIN.get("pred_dim", 128)),
        help="BYOL predictor hidden dimension",
    )
    p.add_argument(
        "--pretrain-ema-decay",
        type=float,
        default=float(PRETRAIN.get("ema_decay", 0.996)),
        help="BYOL target-network EMA decay",
    )
    p.add_argument(
        "--pretrain-sample-windows",
        default="auto",
        help="Windows loaded per pretrain block, or 'auto' for RAM-based sizing",
    )
    p.add_argument(
        "--pretrain-blocks-per-epoch",
        default="auto",
        help="Fresh pretrain blocks per outer epoch, or 'auto' for effective sample volume",
    )
    p.add_argument(
        "--pretrain-mask-prob",
        type=float,
        default=float(PRETRAIN.get("mask_prob", 0.20)),
        help="Masked reconstruction probability when --pretrain-method masked",
    )
    p.add_argument(
        "--pretrain-recon-hidden-dim",
        type=int,
        default=int(PRETRAIN.get("recon_hidden_dim", 512)),
        help="Masked reconstruction decoder hidden size",
    )
    p.add_argument(
        "--pretrain-latent-dim",
        type=int,
        default=int(PRETRAIN.get("latent_dim", 64)),
        help="VAE latent dimension when --pretrain-method vae",
    )
    p.add_argument(
        "--pretrain-vae-beta",
        type=float,
        default=float(PRETRAIN.get("vae_beta", 0.001)),
        help="KL weight for VAE pretrain",
    )
    p.add_argument(
        "--pretrain-n-clusters",
        type=int,
        default=int(PRETRAIN.get("n_clusters", 3)),
        help="k-means clusters for cluster contrastive pretrain",
    )
    p.add_argument(
        "--pretrain-forecast-horizon",
        type=int,
        default=int(PRETRAIN.get("forecast_horizon", 5)),
        help="Future bars to predict in forecast pretext task",
    )
    p.add_argument(
        "--pretrain-drift-margin",
        type=float,
        default=float(PRETRAIN.get("drift_margin", 1.0)),
        help="Target L2 distance between clean and drift-augmented embeddings",
    )
    p.add_argument(
        "--force-pretrain",
        action="store_true",
        help="Delete existing contrastive encoder checkpoint and pretrain from scratch",
    )
    p.add_argument(
        "--multitask",
        action="store_true",
        help="Replace single prediction head with MultiTaskHead (direction CE + magnitude Huber + confidence BCE)",
    )
    p.add_argument(
        "--mt-w-ret", type=float, default=0.5, help="Multi-task loss weight for return_hat Huber term (default 0.5)"
    )
    p.add_argument(
        "--mt-w-conf", type=float, default=0.3, help="Multi-task loss weight for confidence BCE term (default 0.3)"
    )
    p.add_argument(
        "--direction-probe",
        dest="direction_probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a short balanced direction probe before full supervised training.",
    )
    p.add_argument("--direction-probe-epochs", type=int, default=2, help="Epochs for the pre-training direction probe.")
    p.add_argument(
        "--direction-probe-samples", type=int, default=4096, help="Total samples used by the balanced direction probe."
    )
    p.add_argument(
        "--direction-warmup-epochs",
        type=int,
        default=2,
        help="Initial epochs trained with balanced direction-only batches.",
    )
    p.add_argument(
        "--direction-min-true-class-share",
        type=float,
        default=0.15,
        help="Minimum train/val true class share required before training.",
    )
    p.add_argument(
        "--direction-min-pred-class-share",
        type=float,
        default=0.05,
        help="Minimum validation predicted share for each direction class.",
    )
    p.add_argument(
        "--direction-max-pred-class-share",
        type=float,
        default=0.80,
        help="Maximum validation predicted share for any one direction class.",
    )
    p.add_argument(
        "--direction-min-recall",
        type=float,
        default=0.001,
        help="Minimum per-class validation recall for direction readiness gates.",
    )
    p.add_argument(
        "--overconf-penalty",
        dest="overconf_penalty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply training-time overconfidence penalty for regression losses.",
    )
    p.add_argument("--overconf-weight", type=float, default=0.3, help="Weight for the overconfidence penalty.")
    p.add_argument(
        "--overconf-threshold",
        type=float,
        default=0.6,
        help="Absolute prediction threshold that triggers overconfidence checks.",
    )
    p.add_argument(
        "--calibrate",
        dest="calibrate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fit post-training temperature calibration on the validation set.",
    )
    p.add_argument(
        "--train-ensemble",
        action="store_true",
        default=bool(SETTINGS_ENSEMBLE.get("enabled", False)),
        help="After supervised training, train the EnsembleMetaLearner "
        "with diversity penalty across all trained base models",
    )
    p.add_argument(
        "--ensemble-epochs",
        type=int,
        default=int(SETTINGS_ENSEMBLE.get("epochs", 10)),
        help="Epochs to train the meta-learner (default 10)",
    )
    p.add_argument(
        "--ensemble-div-weight",
        type=float,
        default=float(SETTINGS_ENSEMBLE.get("div_weight", 0.1)),
        help="Diversity penalty weight for meta-learner training (default 0.1)",
    )
    p.add_argument(
        "--deploy-ensemble",
        action="store_true",
        default=bool(SETTINGS_ENSEMBLE.get("deploy", False)),
        help="After ensemble ONNX export, atomically promote it to production_best.onnx for the C++ server.",
    )
    p.add_argument(
        "--ensemble-explicit-diversity",
        action="store_true",
        default=bool(SETTINGS_ENSEMBLE.get("explicit_diversity", False)),
        help="Apply explicit per-member diversity controls during all-model training.",
    )
    p.add_argument(
        "--ensemble-member-seed-offset",
        type=int,
        default=int(SETTINGS_ENSEMBLE.get("member_seed_offset", 997)),
        help="Seed offset between ensemble members when explicit diversity is enabled.",
    )
    p.add_argument(
        "--ensemble-member-lr-jitter",
        type=float,
        default=float(SETTINGS_ENSEMBLE.get("member_lr_jitter", 0.0)),
        help="Relative LR jitter spread across members (e.g. 0.2 => +/-10%%).",
    )
    p.add_argument(
        "--ensemble-member-dropout-jitter",
        type=float,
        default=float(SETTINGS_ENSEMBLE.get("member_dropout_jitter", 0.0)),
        help="Absolute dropout jitter spread across members (clamped to [0,0.8]).",
    )

    # RL
    p.add_argument("--rl-train", action="store_true")
    p.add_argument("--rl-algo", type=str, default="dqn", choices=["dqn", "ppo"])
    p.add_argument("--rl-episodes", type=int, default=500)
    p.add_argument(
        "--rl-episode-len",
        type=int,
        default=2048,
        help="A-H1: bars per RL episode (sub-window sampled at a random offset "
        "each reset). 0 = full series each episode.",
    )
    p.add_argument(
        "--off-policy-rewards",
        action="store_true",
        default=False,
        help="Log IPS/DR OPE estimates during RL (diagnostic only - does not train).",
    )
    p.add_argument(
        "--rl-encoder-obs",
        dest="rl_encoder_obs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="A-C3: use the frozen supervised encoder embedding as the RL "
        "observation (connects supervised->RL). --no-rl-encoder-obs falls "
        "back to raw last-timestep features.",
    )
    p.add_argument(
        "--rl-val-frac", type=float, default=0.15, help="Fraction of the RL window held out for validation rollouts."
    )
    p.add_argument(
        "--rl-min-val-sharpe",
        type=float,
        default=-999.0,
        help="Minimum validation Sharpe required to save rl_*_best.pt.",
    )
    p.add_argument(
        "--rl-use-sharpe-reward",
        dest="rl_use_sharpe_reward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace env P&L reward with rolling SharpeRewardWrapper during RL training.",
    )
    p.add_argument(
        "--rl-use-her",
        dest="rl_use_her",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Hindsight Experience Replay (HERBuffer) for DQN training.",
    )
    p.add_argument(
        "--rl-all-models",
        action="store_true",
        default=False,
        help="With --all-models, run RL once per trained architecture subfolder.",
    )
    p.add_argument(
        "--deploy-rl",
        action="store_true",
        default=bool(RL.get("deploy", False)),
        help="After RL ONNX export, atomically promote it to production_best.onnx for the C++ server.",
    )
    p.add_argument(
        "--pretrain-temperature",
        type=float,
        default=float(PRETRAIN.get("temperature", 0.5)),
        help="Initial TSCL temperature (learnable during pretrain).",
    )
    p.add_argument(
        "--pretrain-read-windows",
        type=int,
        default=int(PRETRAIN.get("read_windows", 64)),
        help="Zarr span read chunk size during pretrain loading.",
    )
    p.add_argument(
        "--maturity-stage",
        type=str,
        default="paper",
        choices=["dev", "paper", "production"],
        help="Model maturity stage (gates live promotion expectations).",
    )
    p.add_argument(
        "--max-bad-frac",
        type=float,
        default=0.05,
        help="Max fraction of bad rows allowed inside a training sequence window.",
    )
    p.add_argument(
        "--max-zero-frac",
        type=float,
        default=0.80,
        help="Max fraction of all-zero feature rows allowed inside a sequence window.",
    )

    # Fine-tune / warm-start (B-C2)
    p.add_argument(
        "--finetune-warm-start",
        dest="finetune_warm_start",
        action="store_true",
        default=False,
        help="B-C2: load prior production/best weights then CONTINUE supervised "
        "training on the new window (distinct from --resume, which skips "
        "training when a best checkpoint already exists).",
    )
    p.add_argument(
        "--warm-start-from",
        type=str,
        default=None,
        help="Explicit checkpoint to warm-start from. Default: production_best.pt then the model's own _best.pt.",
    )

    # Promotion gate (B-C1)
    p.add_argument(
        "--promotion-gate",
        dest="promotion_gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="B-C1: after training, backtest the challenger on a held-out forward "
        "window and run PromotionGate to decide deployment (writes "
        "<model>_promotion.json). --no-promotion-gate disables.",
    )
    p.add_argument(
        "--force-promotion",
        action="store_true",
        help="Bypass the challenger vs production gate and force the promotion.",
    )
    p.add_argument(
        "--promote-forward-frac",
        type=float,
        default=0.1,
        help="Fraction of most-recent samples used as the held-out forward "
        "window for the promotion-gate backtest (B-C1).",
    )

    # HPO
    p.add_argument("--hparam-search", action="store_true")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument(
        "--auto-optuna",
        dest="auto_optuna",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overlay config/optuna/run_optuna_best_<model>_<metric>.yaml when present "
        "(default: optuna.auto_load in run.yaml, else true).",
    )

    # Tracking
    p.add_argument(
        "--legacy-config-globals",
        action="store_true",
        default=False,
        help="Skip immutable-config validation. Use only when migrating from the "
        "old dict-mutation config path. Removed once all callers use RuntimeConfig.",
    )
    p.add_argument("--wandb-project", type=str, default="forex-scaling-model")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument(
        "--auto-run-dir",
        action="store_true",
        default=False,
        help="Generate a descriptive checkpoint folder under checkpoints/runs "
        "from model, strategy, pairs, seq_len, folds, and RL/ensemble mode.",
    )
    p.add_argument(
        "--run-dir-root",
        type=str,
        default=None,
        help="Base directory for --auto-run-dir. Defaults to <checkpoint-dir>/runs.",
    )
    p.add_argument(
        "--auto-tune",
        dest="auto_tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write auto-tune proposal artifacts after each completed training phase. "
        "--no-auto-tune disables proposal generation and config nudges.",
    )

    p.add_argument(
        "--dry-tune",
        action="store_true",
        default=False,
        help="Write auto-tune proposals without mutating config/run.yaml.",
    )
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument(
        "--ollama-auto-tune",
        action="store_true",
        default=False,
        help="Allow Ollama to edit config and restart training after a run.",
    )
    p.add_argument("--save-every", type=int, default=5)

    # Paths
    p.add_argument("--checkpoint-dir", type=str, default=PATHS["checkpoints"])
    p.add_argument("--data-cache", type=str, default=PATHS["data_processed"])
    p.add_argument(
        "--resume",
        dest="resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume model/optimizer state from existing checkpoints. "
        "Use --no-resume for a clean supervised run even when config/run.yaml enables resume.",
    )

    p.add_argument(
        "--training-memory",
        dest="training_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply conservative hyperparameter nudges from logs/training_memory.json. "
        "Use --no-training-memory for a clean baseline/fresh run.",
    )

    p.add_argument(
        "--retrain-completed-models",
        action="store_true",
        default=False,
        help="With --all-models --resume, retrain models that already have "
        "completed artifacts instead of skipping to unfinished models.",
    )
    p.add_argument(
        "--force-rebuild",
        "--rebuild-cache",
        dest="force_rebuild",
        action="store_true",
        help="Ignore cached Zarr/NPY store and rebuild from scratch",
    )
    p.add_argument("--build-only", action="store_true", help="Only build the dataset pipeline and exit")
    p.add_argument(
        "--tabular-build-only",
        dest="tabular_build_only",
        action="store_true",
        default=False,
        help="Shorthand for --config config/run_tabular.yaml --build-only -- build the "
        "5M-tick triple_barrier tabular dataset (500 MB, ~2-3h) via the GPU pipeline "
        "without training. Equivalent to training.train_gpu with the tabular config.",
    )
    p.add_argument("--quick-mode", action="store_true", help="Fast sanity run: fewer folds/epochs, no ensemble or RL.")
    p.add_argument(
        "--drift-gate",
        dest="drift_gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run a pre-training input-distribution drift gate on cached features. "
        "Keep disabled for historical training runs.",
    )
    p.add_argument(
        "--drift-fail-open",
        dest="drift_fail_open",
        action="store_true",
        default=False,
        help="If drift gate check errors, continue training with a warning.",
    )
    p.add_argument(
        "--drift-baseline-samples",
        type=int,
        default=20_000,
        help="Baseline sample rows from cache start for drift gate.",
    )
    p.add_argument(
        "--drift-live-samples", type=int, default=5_000, help="Recent sample rows from cache end for drift gate."
    )
    p.add_argument(
        "--drift-psi-threshold",
        type=float,
        default=float(MONITORING.get("psi_threshold", 0.2)),
        help="PSI threshold for drift gate fail condition.",
    )
    p.add_argument(
        "--drift-ks-pvalue-threshold",
        type=float,
        default=float(MONITORING.get("ks_pvalue_threshold", 0.05)),
        help="KS p-value threshold for drift gate fail condition.",
    )
    p.add_argument(
        "--drift-ks-statistic-threshold",
        type=float,
        default=0.05,
        help="KS D-statistic (effect-size) floor for drift gate. "
        "KS only fails when BOTH p-value < threshold AND D-stat >= this value. "
        "Prevents false alarms on large datasets (default 0.05).",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Run torch.profiler for 3 warm-up + 5 active batches then exit. "
            "Outputs a Chrome trace to logs/profile_<model>_<run>.json. "
            "Open in chrome://tracing or https://ui.perfetto.dev. "
            "Reveals whether you are compute-bound, memory-bound, or input-bound. "
            "Recommended before tuning batch size or enabling torch.compile."
        ),
    )
    p.add_argument(
        "--integrity-gate",
        dest="integrity_gate",
        action="store_true",
        default=True,
        help="Fail fast when cached X/y lengths are inconsistent.",
    )
    p.add_argument(
        "--no-integrity-gate",
        dest="integrity_gate",
        action="store_false",
        help="Disable strict cache integrity gate (not recommended).",
    )
    p.add_argument(
        "--feature-schema-gate",
        dest="feature_schema_gate",
        action="store_true",
        default=None,
        help=(
            "Fail dataset build when curriculum features or required market columns "
            "are missing from the built schema. Default: follows --integrity-gate."
        ),
    )
    p.add_argument(
        "--no-feature-schema-gate",
        dest="feature_schema_gate",
        action="store_false",
        help="Disable the dataset feature-schema gate (not recommended).",
    )
    p.add_argument(
        "--min-pair-years",
        type=int,
        default=2,
        help="Minimum years of data required per pair for multi-pair training (default: 2).",
    )
    p.add_argument(
        "--expected-pair-years",
        type=int,
        default=18,
        help="Expected years of data per pair - warns if less (default: 18).",
    )
    p.add_argument(
        "--coverage-report", action="store_true", help="Generate data coverage report (data_coverage_report.json)."
    )
    p.add_argument(
        "--auto-rebuild-on-mismatch",
        action="store_true",
        help="If cache integrity fails, delete cache/sidecars and rebuild automatically.",
    )
    p.add_argument(
        "--pretrain-ablation",
        type=str,
        nargs="?",
        const="true",
        choices=["true", "false", "auto"],
        default="auto",
        help="If true or auto (for transformer/haelt), runs a full training baseline with NO PRETRAIN first.",
    )
    p.add_argument(
        "--pretrain-ablation-models",
        type=str,
        default="",
        help="Comma-separated model list used when --pretrain-ablation auto. "
        "Default/config recommendation: tft,transformer,haelt.",
    )

    p.add_argument(
        "--ignore-manifest",
        action="store_true",
        help="Bypass dataset_manifest.json checks and force load existing cache.",
    )
    p.add_argument(
        "--walk-forward-cv",
        action="store_true",
        help="Purged walk-forward CV (train past / val future, embargo=seq_len+lookahead+delay) instead of one split",
    )
    p.add_argument(
        "--walk-forward-folds",
        type=int,
        default=None,
        help="Number of walk-forward folds (default: TRAINING['walk_forward_folds'])",
    )
    p.add_argument(
        "--cv-strategy",
        type=str,
        default="legacy",
        choices=["legacy", "walk_forward", "comb", "online"],
        help="CV split strategy when --walk-forward-cv is on (Improvement #11): "
        "legacy = original walk_forward_splits (default); walk_forward = "
        "WalkForwardCV; comb = combinatorial purged CV; online = rolling window.",
    )
    p.add_argument(
        "--execution-delay-bars",
        type=int,
        default=1,
        help="Bars between model signal and executable entry; used for training labels and backtests.",
    )
    p.add_argument(
        "--data-quality-check",
        action="store_true",
        help="Run the data quality check script on the Zarr cache before training.",
    )
    p.add_argument(
        "--skip-training",
        action="store_true",
        help="Exit after data quality check (or dataset build) without training.",
    )
    p.add_argument(
        "--validate-config",
        action="store_true",
        help="Audit run.yaml/CLI for contradictions, estimate runtime, and exit without training.",
    )

    p.add_argument(
        "--teacher-model", type=str, default=None, help="Name of teacher model to distill from (e.g., haelt, ensemble)"
    )
    p.add_argument("--teacher-ckpt", type=str, default=None, help="Explicit teacher checkpoint path for distillation")
    p.add_argument(
        "--distill-weight", type=float, default=0.5, help="Weight of distillation loss relative to supervised loss"
    )
    p.add_argument(
        "--distill-temperature",
        type=float,
        default=float(DISTILLATION.get("temperature", 2.0)),
        help="Temperature for distillation soft targets (KL).",
    )

    # Advanced Training Mechanics (Phase 2)
    p.add_argument(
        "--enable-ewc",
        action="store_true",
        help="Enable Elastic Weight Consolidation to prevent catastrophic forgetting.",
    )
    p.add_argument("--ewc-lambda", type=float, default=1000.0, help="EWC penalty weight (default: 1000.0).")

    p.add_argument(
        "--enable-si", action="store_true", help="Enable Synaptic Intelligence (SI) to prevent catastrophic forgetting."
    )
    p.add_argument(
        "--si-lambda",
        type=float,
        default=1.0,
        help="SI penalty weight (default: 1.0). "
        "When the FeatureStabilityMonitor is active, this base lambda is "
        "scaled per epoch by 1/(1 + max_shift^2) as the SI dynamic lambda.",
    )

    p.add_argument("--enable-per", action="store_true", help="Enable Prioritized Experience Replay (PER).")

    p.add_argument(
        "--enable-adversarial",
        action="store_true",
        help="Enable adversarial training (PGD/FGSM/FreeLB) or legacy market shocks.",
    )
    p.add_argument(
        "--enable-irt",
        action="store_true",
        help="Enable Infinite Robust Training (IRT) orchestration."
    )
    p.add_argument(
        "--adversarial-method",
        type=str,
        default="pgd",
        choices=["pgd", "fgsm", "freelb", "market_shock", "graph_pgd"],
        help="Adversarial method: pgd (Projected Gradient Descent), fgsm (Fast Gradient Sign), "
        "freelb (Free Large-Batch), market_shock (legacy random shocks), "
        "graph_pgd (graph-aware PGD for GNNs; auto-selected for model_name 'gnn').",
    )
    p.add_argument(
        "--adversarial-prob",
        type=float,
        default=0.01,
        help="Probability of applying adversarial attack per batch (default: 0.01).",
    )
    p.add_argument(
        "--adversarial-eps", type=float, default=0.3, help="L-infinity perturbation budget epsilon (default: 0.3)."
    )
    p.add_argument("--adversarial-alpha", type=float, default=0.01, help="Step size for PGD/FreeLB (default: 0.01).")
    p.add_argument(
        "--adversarial-steps", type=int, default=7, help="Number of attack steps for PGD/FreeLB (default: 7)."
    )
    p.add_argument(
        "--adversarial-normalize-grad",
        action="store_true",
        help="L2 normalize gradients in PGD/Graph PGD (Madry best practice).",
    )
    p.add_argument(
        "--adversarial-warmup-steps",
        type=int,
        default=0,
        help="Gradually increase attack steps over this many training steps (0=disabled).",
    )
    p.add_argument(
        "--adversarial-eps-curriculum-scale",
        action="store_true",
        help="Scale adversarial epsilon with curriculum difficulty level (eps *= level/n_levels).",
    )
    p.add_argument(
        "--adversarial-models",
        type=str,
        default="",
        help="Comma-separated list of models to enable adversarial for (default: all except expert).",
    )

    # -- Risk engine (Improvement #1) - optional live/dry-run enforcement config --
    p.add_argument(
        "--risk-config",
        type=str,
        default=None,
        metavar="PATH",
        help="JSON/YAML file with a RiskEngine config (keys mirror "
        "config/settings.RISK) for live/dry-run enforcement during training.",
    )

    # -- Pre-parse to find --config, then apply YAML defaults before full parse --
    # --tabular-build-only is a shorthand for --config run_tabular.yaml --build-only
    # If present without an explicit --config, force the tabular config before YAML overlay.
    _has_tabular_flag = "--tabular-build-only" in sys.argv
    _has_explicit_config = any(a == "--config" or a.startswith("--config=") for a in sys.argv)
    if _has_tabular_flag and not _has_explicit_config:
        p.set_defaults(config="config/run_tabular.yaml")
    pre, _ = p.parse_known_args()
    if pre.config:
        _apply_yaml_config(p, pre.config)
        from training.optuna_config import apply_optuna_overlay_if_needed

        apply_optuna_overlay_if_needed(p, pre.config, getattr(pre, "auto_optuna", None), _apply_yaml_config)

    args = p.parse_args()
    # --tabular-build-only: imply --build-only and tabular config
    if getattr(args, "tabular_build_only", False):
        args.build_only = True
        if not _has_explicit_config:
            args.config = "config/run_tabular.yaml"
            print("[Config] --tabular-build-only: building tabular dataset (config/run_tabular.yaml, --build-only)")
        else:
            print(f"[Config] --tabular-build-only + --config {args.config}: --build-only enabled (explicit config takes precedence)")
    # --no-amp: force FP32 regardless of --dtype or hardware profile
    if getattr(args, "no_amp", False):
        args.dtype = "fp32"
        args.amp = False
    if args.val_split is None:
        args.val_split = float(TRAINING["val_split"])
    if args.loss is None:
        args.loss = str(TRAINING.get("loss", "huber"))
    if args.walk_forward_folds is None:
        args.walk_forward_folds = int(TRAINING.get("walk_forward_folds", 6))
    if args.grad_accum_steps is None:
        args.grad_accum_steps = int(TRAINING.get("grad_accum_steps", 1))
    prof = strategy_profile(args.strategy_mode)
    scalp = strategy_profile("scalping")
    if args.bar_freq is None:
        args.bar_freq = str(prof["bar_freq"])
    if args.strategy_mode != "scalping" and int(args.seq_len) == int(scalp["seq_len"]):
        args.seq_len = int(prof["seq_len"])
    if args.lookahead_bars is None:
        args.lookahead_bars = int(prof["lookahead_bars"])
    if args.profit_target_atr is None:
        args.profit_target_atr = float(prof["profit_target_atr"])
    if args.stop_loss_atr is None:
        args.stop_loss_atr = float(prof["stop_loss_atr"])
    if args.strategy_mode != "scalping" and int(args.execution_delay_bars) == int(scalp["execution_delay_bars"]):
        args.execution_delay_bars = int(prof["execution_delay_bars"])
    print(
        f"[Strategy] {args.strategy_mode} | bars={args.bar_freq} | seq_len={args.seq_len} | "
        f"lookahead={args.lookahead_bars}(dynamic:trending=20,ranging=12,volatile=6) | "
        f"TP/SL={args.profit_target_atr}/{args.stop_loss_atr} ATR | "
        f"early_stop=dynamic(patience={getattr(args, 'early_stop_patience', 10)},EMA+LR-halve+adaptive)"
    )
    if args.quick_mode:
        # Synthetic/quick smokes are too small for purged walk-forward
        # (embargo ≈ seq_len+lookahead often exceeds the sample count).
        if str(getattr(args, "data_source", "")).lower() == "synthetic":
            _lh = int(getattr(args, "lookahead_bars", 30) or 30)
            _delay = int(getattr(args, "execution_delay_bars", 1) or 1)
            _chunk_size = int(getattr(args, "chunk_size", 500000))
            if _chunk_size <= 50000:
                _cap = max(16, 48 - _lh // 2)
                if int(args.seq_len) > _cap:
                    print(f"[Quick] synthetic seq_len {args.seq_len} -> {_cap} (headroom for lookahead={_lh}+delay={_delay})")
                    args.seq_len = _cap
        cur = getattr(args, "curriculum", None)
        if isinstance(cur, dict):
            capped = []
            for entry in cur.get("seq_schedule") or []:
                if not isinstance(entry, dict):
                    continue
                e = dict(entry)
                if e.get("seq_len") is not None:
                    e["seq_len"] = min(int(e["seq_len"]), int(args.seq_len))
                capped.append(e)
            args.curriculum = {**cur, "seq_schedule": capped or [{"epoch_start": 0, "seq_len": int(args.seq_len)}]}
        print(
            f"[Quick] ON | folds={args.walk_forward_folds} | epochs={args.epochs} | "
            f"pretrain_epochs={args.pretrain_epochs} | ensemble=off | rl=off"
            f" | wf={'on' if args.walk_forward_cv else 'off'}"
        )
    # B-M1: a warm-start fine-tune on a short rolling window must NOT run k-fold
    # walk-forward CV (a 1-epoch fine-tune would attempt 5 folds then hit the
    # small-data fallback). Force the single-split + embargo path explicitly.
    if getattr(args, "finetune_warm_start", False) and args.walk_forward_cv:
        args.walk_forward_cv = False
        print("[FineTune] Warm-start mode: walk-forward CV disabled (single embargoed split).")
    if getattr(args, "fair_sweep", False):
        args.model_profile = False
    args._cli_profile_overrides = _collect_cli_profile_overrides()
    _sync_runtime_config(args)

    # -- Risk engine (Improvement #1): optional live/dry-run enforcement config. --
    if getattr(args, "risk_config", None):
        try:
            import json as _json
            from pathlib import Path as _Path

            rc_path = _Path(args.risk_config)
            if rc_path.is_file():
                text = rc_path.read_text()
                if rc_path.suffix.lower() in (".yaml", ".yml"):
                    import yaml as _yaml

                    rc_data = _yaml.safe_load(text) or {}
                else:
                    rc_data = _json.loads(text)
            elif args.risk_config.strip().startswith("{"):
                rc_data = _json.loads(args.risk_config)
            else:
                rc_data = {}
            from risk.risk_engine import RiskConfig, RiskEngine

            cfg = RiskConfig.from_dict(rc_data or {})
            args.risk_engine = RiskEngine(equity=float(getattr(args, "risk_equity", 10_000.0)), cfg=cfg)
            print(
                f"[Risk] Loaded risk config from {args.risk_config} | max_notional=${cfg.max_notional_usd:,.0f} "
                f"pos_cap={cfg.max_position_pct:.2%} dd_halt={cfg.max_drawdown_halt:.0%} "
                f"var={cfg.var_confidence:.0%}"
            )
        except Exception as e:
            print(f"[Risk] Failed to load --risk-config {args.risk_config}: {e}")
            args.risk_engine = None
    else:
        args.risk_engine = None

    # Time-anchored seq_len resolution
    _bf = getattr(args, "bar_freq", "5m")
    args.seq_len = _resolve_seq_len(args.seq_len, _bf)
    if hasattr(args, "curriculum") and isinstance(args.curriculum, dict):
        sched = args.curriculum.get("seq_schedule")
        if isinstance(sched, list):
            for entry in sched:
                if isinstance(entry, dict) and "seq_len" in entry:
                    entry["seq_len"] = _resolve_seq_len(entry["seq_len"], _bf)

    return args


__all__ = [
    # Core entry points
    "parse_args",
    "apply_hardware_profile",
    # YAML / config helpers
    "_YAML_MAP",
    "_apply_yaml_config",
    "_apply_yaml_risk_to_live_risk",
    "_sync_runtime_config",
    "_resolve_seq_len",
    # Seed / run-dir helpers
    "_set_global_seed",
    "_slug_part",
    "_build_auto_run_name",
    "_apply_auto_run_dir",
    "_collect_cli_profile_overrides",
    # Model profile helpers
    "_normalize_architecture_profile",
    "_apply_model_profile",
    "_apply_training_profile",
    "_member_training_args",
    "_model_build_args",
    # Resume helpers
    "_model_completion_status",
    "_baseline_ablation_completion_status",
    "_supervised_resume_status",
    "_latest_resumable_fold",
    "_load_cv_fold_entry",
    "_load_walk_forward_resume_history",
]
