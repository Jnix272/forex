"""Per-model training profiles derived from architecture properties."""

from dataclasses import dataclass, field
from typing import Optional, Literal

# Import at module level to avoid circular import risk
try:
    from models.architectures import build_model
except ImportError:
    build_model = None

import torch.nn as nn


@dataclass
class ModelTrainingProfile:
    """All training dimensions derived from model architecture."""

    model_name: str

    # Architecture properties (auto-detected)
    has_attention: bool = False
    has_lstm: bool = False
    has_conv: bool = False
    has_graph: bool = False
    has_positional_encoding: bool = True
    capacity: Literal["low", "medium", "high"] = "medium"
    inductive_bias: Literal["temporal", "relational", "transformer", "conv"] = "transformer"

    # Loss
    primary_loss: str = "sharpe_huber"
    secondary_loss: Optional[str] = None
    use_multitask: bool = True

    # Adversarial
    adversarial_enabled: bool = True
    adversarial_method: str = "pgd"
    adversarial_eps: float = 0.3
    adversarial_alpha: float = 0.01
    adversarial_steps: int = 7
    adversarial_prob: float = 0.01

    # Curriculum
    curriculum_mode: str = "combined"
    use_difficulty: bool = True
    use_self_paced: bool = True
    use_loss_weighting: bool = True
    difficulty_n_levels: int = 10
    difficulty_start_level: int = 1
    difficulty_advance_rate: float = 0.25
    difficulty_min_competence: float = 0.7
    self_paced_lambda: float = 1.0
    self_paced_min_fraction: float = 0.3
    loss_weighting_scheme: str = "focal"
    loss_weighting_temp: float = 1.0

    # Online Miner feedback
    miner_feedback: bool = True
    forgetting_threshold: float = 0.15
    easy_threshold: float = 0.60
    freeze_patience: int = 1

    # Pretraining
    pretrain_method: str = "masked"
    pretrain_framework: str = "custom"

    # SWA / EMA
    swa_enabled: bool = True
    swa_start_frac: float = 0.75
    swa_lr: float = 1e-5
    ema_decay: float = 0.99

    # Framework
    training_framework: str = "custom"

    # RL fine-tuning
    rl_finetune: bool = True
    rl_framework: str = "custom"
    rl_use_lstm: bool = False


# Registry: per-model profiles (auto-populated by get_training_profile if missing)
MODEL_PROFILES = {
    "haelt": ModelTrainingProfile(
        model_name="haelt",
        has_attention=True,
        has_lstm=True,
        capacity="high",
        inductive_bias="transformer",
        primary_loss="sharpe_huber",
        secondary_loss="directional_huber",
        adversarial_eps=0.3,
        adversarial_method="pgd",
        curriculum_mode="combined",
        use_self_paced=True,
        miner_feedback=True,
        pretrain_method="masked",
        pretrain_framework="custom",
        rl_use_lstm=True,
    ),
    "tft": ModelTrainingProfile(
        model_name="tft",
        has_attention=True,
        has_lstm=False,
        has_conv=False,
        capacity="high",
        inductive_bias="transformer",
        primary_loss="cross_entropy",
        secondary_loss="multitask",
        adversarial_eps=0.3,
        adversarial_method="pgd",
        curriculum_mode="combined",
        use_self_paced=True,
        miner_feedback=True,
        pretrain_method="masked",
        pretrain_framework="custom",
    ),
    "transformer": ModelTrainingProfile(
        model_name="transformer",
        has_attention=True,
        has_lstm=False,
        has_positional_encoding=False,
        capacity="high",
        inductive_bias="transformer",
        primary_loss="sharpe_huber",
        secondary_loss="cross_entropy",
        adversarial_eps=0.3,
        adversarial_method="pgd",
        curriculum_mode="combined",
        use_self_paced=True,
        miner_feedback=True,
        pretrain_method="byol_or_tscl",
        pretrain_framework="lightly",
    ),
    "mamba": ModelTrainingProfile(
        model_name="mamba",
        has_attention=False,
        has_conv=True,
        has_lstm=False,
        capacity="medium",
        inductive_bias="temporal",
        primary_loss="directional_huber",
        secondary_loss="sharpe_huber",
        adversarial_eps=0.3,
        adversarial_method="pgd",
        curriculum_mode="difficulty",
        use_self_paced=False,
        miner_feedback=False,
        pretrain_method="forecast",
        pretrain_framework="custom",
        swa_enabled=True,
    ),
    "gnn": ModelTrainingProfile(
        model_name="gnn",
        has_attention=False,
        has_graph=True,
        has_lstm=False,
        capacity="medium",
        inductive_bias="relational",
        primary_loss="sharpe_huber",
        secondary_loss="directional_huber",
        adversarial_eps=0.3,
        adversarial_method="graph_pgd",
        curriculum_mode="difficulty",
        use_self_paced=False,
        miner_feedback=False,
        pretrain_method="cluster",
        pretrain_framework="custom",
        rl_finetune=False,
    ),
    "expert": ModelTrainingProfile(
        model_name="expert",
        has_attention=True,
        has_conv=True,
        has_lstm=False,
        has_positional_encoding=False,
        capacity="low",
        inductive_bias="conv",
        primary_loss="directional_huber",
        secondary_loss="cross_entropy",
        adversarial_enabled=False,
        adversarial_eps=0.0,
        curriculum_mode="difficulty",
        use_self_paced=False,
        use_loss_weighting=False,
        miner_feedback=False,
        pretrain_method="tscl",
        pretrain_framework="custom",
        swa_enabled=False,
    ),
}


def get_training_profile(model_name: str) -> ModelTrainingProfile:
    """Get training profile for a model, with auto-detection fallback."""
    name = model_name.lower().strip()
    if name in MODEL_PROFILES:
        return MODEL_PROFILES[name]

    # Fallback: auto-detect from architecture
    return _auto_detect_profile(name)


def _auto_detect_profile(model_name: str) -> ModelTrainingProfile:
    """Auto-detect training profile from model architecture."""
    
    if build_model is None:
        raise ImportError("models.architectures.build_model not available for auto-detection")

    # Build dummy model to inspect architecture
    model = build_model(model_name, input_size=64, seq_len=128)

    profile = ModelTrainingProfile(model_name=model_name)

    # Detect architecture properties
    profile.has_attention = any(
        isinstance(m, (nn.MultiheadAttention, nn.TransformerEncoderLayer))
        for m in model.modules()
    )
    profile.has_lstm = any(
        isinstance(m, nn.LSTM) for m in model.modules()
    )
    profile.has_graph = hasattr(model, "n_nodes") or "GNN" in type(model).__name__.upper()
    profile.has_conv = any(
        isinstance(m, (nn.Conv1d, nn.Conv2d)) for m in model.modules()
    )
    profile.has_positional_encoding = not getattr(model, "no_pos_encoding", False)

    # Estimate capacity from parameter count
    total_params = sum(p.numel() for p in model.parameters())
    if total_params < 500_000:
        profile.capacity = "low"
    elif total_params < 2_000_000:
        profile.capacity = "medium"
    else:
        profile.capacity = "high"

    # Derive training config from properties
    _derive_training_config(profile)
    return profile


def _derive_training_config(profile: ModelTrainingProfile) -> None:
    """Rule-based derivation from architecture properties."""

    # Adversarial: disable for low capacity
    if profile.capacity == "low":
        profile.adversarial_enabled = False
        profile.adversarial_eps = 0.0

    # GNN gets graph_pgd
    if profile.has_graph:
        profile.adversarial_method = "graph_pgd"

    # Self-paced: disable for graph/temporal inductive bias
    profile.use_self_paced = not (profile.has_graph or profile.inductive_bias == "temporal")

    # Miner feedback: disable for graph/temporal
    profile.miner_feedback = not (profile.has_graph or profile.inductive_bias == "temporal")

    # Low capacity → simpler curriculum
    if profile.capacity == "low":
        profile.curriculum_mode = "difficulty"
        profile.use_loss_weighting = False
        profile.swa_enabled = False

    # Pretrain method from inductive bias
    if profile.has_graph:
        profile.pretrain_method = "cluster"
    elif profile.inductive_bias == "temporal":
        profile.pretrain_method = "forecast"
    elif profile.inductive_bias == "transformer":
        profile.pretrain_method = "byol_or_tscl"
    else:
        profile.pretrain_method = "masked"

    # RL LSTM for models with LSTM
    profile.rl_use_lstm = profile.has_lstm