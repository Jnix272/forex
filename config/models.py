"""
config/models.py  —  Supervised architecture hyperparameters
===========================================================
Defaults aligned with `training/train_gpu.py` builders and `models/architectures.py`.
Edit here to change per-architecture specs; `settings.MODELS` re-exports this dict.
"""

from typing import Any

# Keys accepted by train_gpu --model and build_model()
SUPPORTED_SUPERVISED: frozenset[str] = frozenset(
    {"tft", "transformer", "haelt", "mamba", "gnn", "expert"}
)

BENCHMARK_BASELINES: dict[str, dict[str, Any]] = {
    "xgboost": {
        "decision_role": "non_deep_baseline",
        "use_when": "Official non-deep benchmark every deep model must beat.",
        "default_use": "Baseline comparator for all deep models; also useful standalone for fast iteration.",
        "train_script": "training/train_xgboost.py",
        "checkpoint": "checkpoints/xgboost_best.json",
        "comparison_rule": "Deep models must beat XGBoost backtest Sharpe without worse max drawdown.",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "sequence_mode": "temporal",
        "task": "classification",
    }
}

MODELS: dict[str, dict[str, Any]] = {
    "tft": {
        "decision_role": "interpretability_exogenous",
        "use_when": "Need feature importance, interpretability, and many exogenous inputs.",
        "default_use": "Use as an interpretable context model, not the automatic production default.",
        "hidden_size": 128,
        "nhead": 4,
        "dropout": 0.1,
        "lstm_layers": 2,
        "seq_len": 60,
        "learning_rate": 1e-3,
    },
    "transformer": {
        "decision_role": "generic_long_range_baseline",
        "use_when": "Need a strong generic baseline for long-range multivariate sequence modeling.",
        "default_use": "Use as the benchmark that HAELT, Mamba, and TFT must beat.",
        "d_model": 128,
        "nhead": 8,
        "num_layers": 3,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "seq_len": 60,
        "learning_rate": 1e-4,
    },
    "haelt": {
        "decision_role": "flagship_production_candidate",
        "use_when": "Need the main production candidate after it wins downstream backtests.",
        "default_use": "Default production candidate for the current three-pair training run.",
        "lstm_hidden": 128,
        "d_model": 256,
        "nhead": 8,
        "num_layers": 3,
        "dropout": 0.25,
        "seq_len": 60,
        "learning_rate": 3e-4,
        "pretrain_epochs": 12,
        "pretrain_lr": 8e-5,
    },
    "mamba": {
        "decision_role": "long_sequence_efficiency",
        "use_when": "Need long-sequence efficiency or lower memory pressure than attention-heavy models.",
        "default_use": "Use for latency, memory, and distillation teacher/student experiments.",
        "d_model": 128,
        # d_state/d_conv/expand: MambaScalper uses simplified SSM blocks;
        # these params are reserved for future full-Mamba integration.
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "num_layers": 4,
        "dropout": 0.1,
        "seq_len": 60,
        "learning_rate": 1e-4,
        "pretrain_epochs": 14,
        "pretrain_lr": 1e-4,
    },
    "gnn": {
        "decision_role": "cross_asset_structure",
        "use_when": "Use only when cross-asset structure clearly matters and graph edges can be justified.",
        "default_use": "Reserve for correlation/spillover experiments and risk-modulation evidence.",
        "learning_rate": 1e-4,
        "seq_len": 80,
        "node_features": 32,
        "hidden_channels": 64,
        "num_layers": 3,
        "heads": 4,
        "dropout": 0.1,
    },
    "expert": {
        "decision_role": "specialist_ablation",
        "use_when": "Reserve for ablation or specialist experiments.",
        "default_use": "Do not use as a default production model.",
        "d_model": 128,
        "nhead": 8,
        "num_layers": 4,
        # use_conv_ffn/no_pos_encoding: hardcoded design decisions in EXPERTEncoder
        # (ConvFFN always used, positional encoding always omitted by design).
        "use_conv_ffn": True,
        "no_pos_encoding": True,
        "dropout": 0.1,
        "seq_len": 80,
        "learning_rate": 1e-4,
    },
}


def architecture_config(name: str) -> dict[str, Any]:
    """Return hyperparameter dict for a supervised architecture key."""
    key = name.lower().strip()
    if key not in MODELS:
        raise KeyError(
            f"Unknown architecture {name!r}; expected one of {sorted(SUPPORTED_SUPERVISED)}"
        )
    return dict(MODELS[key])
