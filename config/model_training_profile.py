"""Per-model training profiles derived from architecture properties."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch.nn as nn

# Dependency inversion: config never imports models. Instead,
# models/architectures registers its build_model factory here at import time
# via register_build_model(). The registry below covers all known models
# statically; this factory is only needed by the auto-detection fallback for
# unknown model names.
build_model: Callable[..., nn.Module | None] | None = None


def register_build_model(factory: "Callable[..., nn.Module]") -> None:
    """Called by models.architectures at import time to provide its factory."""
    global build_model
    build_model = factory


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
    secondary_loss: str | None = None
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

    # Continuous Learning
    enable_ewc: bool = False
    ewc_lambda: float = 1000.0
    enable_si: bool = False
    si_lambda: float = 1.0
    # Dynamic SI lambda - uses DynamicSILambdaConfig sigmoid schedule:
    # λ rises when loss is low (protect learned weights), falls when loss is
    # high (allow adaptation during regime shocks). See synaptic_intelligence.py.
    si_dynamic: bool = False
    si_lambda_min: float = 0.0
    si_lambda_max: float = 1.0
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
    # ── HAELT (Hybrid Attention-LSTM Transformer) ─────────────────────────────
    # Highest-capacity model: LSTM hidden state + multi-head attention.
    # The LSTM retains temporal memory across the sequence while attention
    # captures long-range dependencies. This dual structure makes it the most
    # prone to fold-to-fold forgetting (LSTM state reinitialises each fold but
    # the attention weights carry over). Both SI and EWC are used:
    #   - SI accumulates online importance during training (zero extra pass)
    #   - EWC adds a Fisher-diagonal anchor after each fold completes
    # EWC lambda is kept low (200) so the two don't fight each other.
    # Dynamic SI: relaxes lambda during regime shocks (high feature shift),
    # re-locks when the distribution stabilises.
    # Adversarial: 5 PGD steps (down from 7) — HAELT is slow; fewer steps
    # preserve throughput while still hardening the LSTM hidden state.
    # SWA starts at 65% of epochs: given best-epoch=0/1 pattern, averaging
    # weights early captures the brief high-Sharpe window.
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
        adversarial_steps=5,
        adversarial_prob=0.02,
        curriculum_mode="combined",
        use_self_paced=True,
        use_loss_weighting=True,
        miner_feedback=True,
        forgetting_threshold=0.05,
        easy_threshold=0.65,
        freeze_patience=2,
        enable_si=True,
        si_lambda=1.2,
        si_dynamic=True,
        si_lambda_min=0.15,
        si_lambda_max=3.0,
        enable_ewc=True,
        ewc_lambda=200.0,
        pretrain_method="masked",
        pretrain_framework="custom",
        swa_enabled=True,
        swa_start_frac=0.65,
        swa_lr=8e-6,
        ema_decay=0.995,
        rl_use_lstm=True,
        rl_finetune=True,
    ),
    # ── TFT (Temporal Fusion Transformer) ────────────────────────────────────
    # High-capacity: multi-head attention + variable selection networks (VSN) +
    # gated residuals. The VSN learns which input features matter per time step
    # — these weights are the most valuable to protect across folds, since the
    # feature importance ranking is the core of TFT's edge.
    # Both SI + EWC: EWC anchors VSN Fisher-important weights (they have clear
    # diagonal Fisher structure since VSN is a set of independent softmax gates),
    # while SI accumulates importance online throughout training.
    # Tighter forgetting threshold (0.06): VSN drift is subtle — a small shift
    # in feature salience scores causes large behavioural change.
    # Higher adversarial probability (0.02): TFT's gating is sensitive to
    # input perturbation, so slight adversarial noise improves robustness.
    # SWA 65%: same rationale as HAELT (best-epoch=0/1 pattern).
    "tft": ModelTrainingProfile(
        model_name="tft",
        has_attention=True,
        has_lstm=False,
        has_conv=False,
        capacity="high",
        inductive_bias="transformer",
        primary_loss="sharpe_huber",
        secondary_loss="directional_huber",
        adversarial_eps=0.25,
        adversarial_method="pgd",
        adversarial_steps=5,
        adversarial_prob=0.02,
        curriculum_mode="combined",
        use_self_paced=True,
        use_loss_weighting=True,
        miner_feedback=True,
        forgetting_threshold=0.06,
        easy_threshold=0.65,
        freeze_patience=2,
        enable_si=True,
        si_lambda=1.0,
        si_dynamic=True,
        si_lambda_min=0.10,
        si_lambda_max=2.0,
        enable_ewc=True,
        ewc_lambda=300.0,
        pretrain_method="masked",
        pretrain_framework="custom",
        swa_enabled=True,
        swa_start_frac=0.65,
        swa_lr=8e-6,
        ema_decay=0.995,
        rl_finetune=True,
    ),
    # ── iTransformer (inverted Transformer) ───────────────────────────────────
    # High-capacity: inverted attention over variates (features) rather than
    # time steps. Each token represents one feature's full time series, so
    # attention scores capture cross-feature dependencies (e.g. EUR/USD vs
    # USD/JPY spread). No positional encoding — relies purely on token identity.
    # No LSTM state reinitialisation problem, but variate-token representations
    # are prone to collapse (all tokens converge to similar embeddings) across
    # folds when distribution shifts. SI anchors the token interaction weights.
    # EWC adds Fisher-diagonal anchor for the attention projection matrices
    # which have the clearest per-parameter Fisher structure of any component.
    # Secondary loss: directional_huber (not cross_entropy) — sharpe_huber as
    # primary already implies a soft-direction signal; directional_huber
    # explicitly pushes the regression head toward correct sign prediction.
    # BYOL pretraining: directly pretrains the iTransformer backbone via
    # variate-level contrastive learning (random feature masking as augmentation).
    "transformer": ModelTrainingProfile(
        model_name="transformer",
        has_attention=True,
        has_lstm=False,
        has_positional_encoding=False,
        capacity="high",
        inductive_bias="transformer",
        primary_loss="sharpe_huber",
        secondary_loss="directional_huber",
        adversarial_eps=0.25,
        adversarial_method="pgd",
        adversarial_steps=5,
        adversarial_prob=0.02,
        curriculum_mode="combined",
        use_self_paced=True,
        use_loss_weighting=True,
        miner_feedback=True,
        forgetting_threshold=0.07,
        easy_threshold=0.65,
        freeze_patience=2,
        enable_si=True,
        si_lambda=1.0,
        si_dynamic=True,
        si_lambda_min=0.10,
        si_lambda_max=2.5,
        enable_ewc=True,
        ewc_lambda=350.0,
        pretrain_method="byol",
        pretrain_framework="custom",
        swa_enabled=True,
        swa_start_frac=0.65,
        swa_lr=8e-6,
        ema_decay=0.995,
        rl_finetune=True,
    ),
    # ── Mamba (Selective State Space Model) ──────────────────────────────────
    # Medium-capacity: pure SSM/conv architecture. Mamba's selective state
    # transitions (A, B, C, D matrices) are sequence-position-dependent and
    # don't have a natural Fisher diagonal structure — EWC is inappropriate.
    # SI fits well: it accumulates importance via the path integral of gradient
    # × Δθ, which naturally captures which SSM transition weights were most
    # active during learning, regardless of matrix structure.
    # Dynamic SI enabled: Mamba processes 5m bars with high intra-day regime
    # switching; the dynamic lambda allows the SSM to re-tune its transitions
    # during regime shocks and then re-lock when the distribution stabilises.
    # Miner feedback + self-paced: Mamba benefits from easy-first ordering
    # because its selective gating mechanism learns better on clear examples
    # before being exposed to ambiguous near-zero-label bars.
    # Curriculum mode "combined": Mamba is capable enough for the full stack.
    # Adversarial prob 0.015: slightly lower than attention models since SSM
    # is inherently more noise-robust via its low-pass filtering property.
    "mamba": ModelTrainingProfile(
        model_name="mamba",
        has_attention=False,
        has_conv=True,
        has_lstm=False,
        capacity="medium",
        inductive_bias="temporal",
        primary_loss="sharpe_huber",
        secondary_loss="directional_huber",
        adversarial_eps=0.25,
        adversarial_method="pgd",
        adversarial_steps=5,
        adversarial_prob=0.015,
        curriculum_mode="combined",
        use_self_paced=True,
        use_loss_weighting=True,
        miner_feedback=True,
        forgetting_threshold=0.10,
        easy_threshold=0.60,
        freeze_patience=1,
        enable_si=True,
        si_lambda=0.7,
        si_dynamic=True,
        si_lambda_min=0.05,
        si_lambda_max=1.5,
        pretrain_method="forecast",
        pretrain_framework="custom",
        swa_enabled=True,
        swa_start_frac=0.65,
        swa_lr=8e-6,
        ema_decay=0.99,
        rl_finetune=True,
    ),
    # ── GNN (Graph Neural Network) ────────────────────────────────────────────
    # Medium-capacity: message-passing over a 4-asset graph. The inter-asset
    # edge weights and node aggregation matrices have a clear Fisher structure:
    # each edge weight is either highly important (EURUSD-USDJPY correlation
    # regime) or near-zero (regime where the pair is decorrelated). EWC anchors
    # these learned topology weights across folds.
    # SI added as secondary protection with low lambda: even though EWC is
    # primary, SI's online accumulation catches parameter drift that EWC misses
    # between Fisher re-computations (EWC only runs at fold boundaries).
    # Miner feedback enabled: graph hard examples (bars where the cross-asset
    # signal conflicts — one pair trending while another mean-reverts) are the
    # most informative samples for the GNN to learn from.
    # Self-paced enabled: GNN benefits from easy (trend-aligned) graph states
    # before being exposed to conflicting-signal states.
    # Forgetting threshold tightened to 0.08: graph topology is slow to change
    # but when it does, it changes dramatically — catch drift early.
    # EWC lambda reduced to 600 (from 800): with SI now also active, combined
    # regularisation would over-constrain adaptation if EWC stays at 800.
    # Adversarial: graph_pgd perturbs node features along the graph gradient;
    # lower eps (0.2) since GNN is lower capacity and graph structure is fragile.
    "gnn": ModelTrainingProfile(
        model_name="gnn",
        has_attention=False,
        has_graph=True,
        has_lstm=False,
        capacity="medium",
        inductive_bias="relational",
        primary_loss="sharpe_huber",
        secondary_loss="directional_huber",
        adversarial_eps=0.20,
        adversarial_method="graph_pgd",
        adversarial_steps=3,
        adversarial_prob=0.015,
        curriculum_mode="combined",
        use_self_paced=True,
        use_loss_weighting=True,
        miner_feedback=True,
        forgetting_threshold=0.08,
        easy_threshold=0.60,
        freeze_patience=1,
        enable_ewc=True,
        ewc_lambda=600.0,
        enable_si=True,
        si_lambda=0.3,
        si_dynamic=False,
        si_lambda_min=0.0,
        si_lambda_max=0.5,
        pretrain_method="cluster",
        pretrain_framework="custom",
        swa_enabled=True,
        swa_start_frac=0.70,
        swa_lr=6e-6,
        ema_decay=0.99,
        rl_finetune=False,
    ),
    # ── Expert (lightweight conv MoE) ─────────────────────────────────────────
    # Low-capacity: conv feature extractor + mixture-of-experts routing.
    # No adversarial training (too few parameters to benefit — PGD would
    # simply drive all conv filters to the same direction).
    # Light SI only: stabilise the conv filter banks between folds without
    # over-constraining the already-small parameter space.
    # No EWC: Fisher diagonal is ill-conditioned for the MoE routing layer
    # (sparse activations mean most parameters have near-zero Fisher).
    # No SWA: low-capacity model plateaus quickly; weight averaging introduces
    # more variance than it reduces.
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
        forgetting_threshold=0.15,
        freeze_patience=1,
        enable_si=True,
        si_lambda=0.3,
        si_dynamic=False,
        si_lambda_min=0.0,
        si_lambda_max=0.3,
        pretrain_method="byol",
        pretrain_framework="custom",
        swa_enabled=False,
        rl_finetune=False,
    ),
    # ── PatchTST ──────────────────────────────────────────────────────────────
    # Medium-capacity: patch-based tokenisation via Conv1d + transformer.
    # Each patch covers ~5 bars (at 5m → 25min context per token). Patch-level
    # masking pretraining aligns with the tokenisation granularity.
    # Dynamic SI: patch token representations drift with volatility regime
    # changes; dynamic lambda allows re-learning patch boundaries during shocks.
    # Miner feedback: hard patches (high-vol, trend-reversal bars spanning a
    # patch boundary) are the most informative.
    "patchtst": ModelTrainingProfile(
        model_name="patchtst",
        has_attention=True,
        has_conv=True,
        has_lstm=False,
        capacity="medium",
        inductive_bias="transformer",
        primary_loss="sharpe_huber",
        secondary_loss="directional_huber",
        adversarial_eps=0.25,
        adversarial_method="pgd",
        adversarial_steps=5,
        adversarial_prob=0.015,
        curriculum_mode="combined",
        use_self_paced=True,
        use_loss_weighting=True,
        miner_feedback=True,
        forgetting_threshold=0.10,
        easy_threshold=0.60,
        freeze_patience=1,
        enable_si=True,
        si_lambda=0.6,
        si_dynamic=True,
        si_lambda_min=0.05,
        si_lambda_max=1.2,
        pretrain_method="patch_mask",
        pretrain_framework="custom",
        swa_enabled=True,
        swa_start_frac=0.65,
        swa_lr=8e-6,
        ema_decay=0.99,
        rl_finetune=True,
    ),
    # ── GLM (linear baseline) ─────────────────────────────────────────────────
    # No meaningful parameter topology to protect — weight decay handles
    # regularisation entirely. No pretraining, no RL, no SWA.
    "glm": ModelTrainingProfile(
        model_name="glm",
        capacity="low",
        has_attention=False,
        has_lstm=False,
        has_conv=False,
        has_graph=False,
        has_positional_encoding=False,
        primary_loss="sharpe_huber",
        use_multitask=False,
        adversarial_enabled=False,
        use_self_paced=False,
        use_loss_weighting=False,
        miner_feedback=False,
        forgetting_threshold=0.15,
        enable_si=False,
        enable_ewc=False,
        pretrain_method="none",
        pretrain_framework="none",
        swa_enabled=False,
        rl_finetune=False,
    ),
}


def get_training_profile(model_name: str) -> ModelTrainingProfile:
    """Get training profile for a model, with auto-detection fallback."""
    name = model_name.lower().strip()
    if name in MODEL_PROFILES:
        return MODEL_PROFILES[name]

    # Fallback: auto-detect from architecture
    return _auto_detect_profile(name)


def pretrain_method_for(model_name: str) -> str:
    """Get the deterministic pretraining method for a model, resolving aliases."""
    method = get_training_profile(model_name).pretrain_method
    aliases = {
        "masked_or_byol": "masked",
        "forecast_or_drift": "forecast",
        "byol_or_tscl": "byol",
    }
    return aliases.get(method, method)


def _auto_detect_profile(model_name: str) -> ModelTrainingProfile:
    """Auto-detect training profile from model architecture."""

    model_factory = build_model
    if model_factory is None:
        raise ImportError("models.architectures.build_model not available for auto-detection")

    # Build dummy model to inspect architecture
    model = model_factory(model_name, input_size=64, seq_len=128)
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module from build_model for {model_name!r}, got {type(model).__name__}")

    profile = ModelTrainingProfile(model_name=model_name)

    # Detect architecture properties
    profile.has_attention = any(
        isinstance(m, (nn.MultiheadAttention, nn.TransformerEncoderLayer)) for m in model.modules()
    )
    profile.has_lstm = any(isinstance(m, nn.LSTM) for m in model.modules())
    profile.has_graph = hasattr(model, "n_nodes") or "GNN" in type(model).__name__.upper()
    profile.has_conv = any(isinstance(m, (nn.Conv1d, nn.Conv2d)) for m in model.modules())
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
        profile.pretrain_method = "byol"
        profile.pretrain_framework = "custom"
    else:
        profile.pretrain_method = "masked"

    # RL LSTM for models with LSTM
    profile.rl_use_lstm = profile.has_lstm
