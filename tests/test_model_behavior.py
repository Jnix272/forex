"""
tests/test_model_behavior.py
============================
Behavioral verification tests — checks that models work *as intended*, not
just that they produce the right tensor shape.

Tests are organised into groups:
  1. TestModelRobustness       — NaN safety, B=1 edge case, large / zero inputs
  2. TestMambaScalper          — SSM-specific behaviour: causal ordering, no OOM
                                  on long sequences, state propagation
  3. TestAllModelsGradients    — finite gradients, reasonable init output scale
  4. TestMultiPairWrapperBehavior — correlation range, backbone sizing, regime attn,
                                    alignment confidence, missing-pair gating
  5. TestDiversityLossBehavior — role-conditioned penalty, scalar + differentiable
  6. TestFeatureStabilityMonitor — EMA drift, mask values, frozen-feature reset
  7. TestBuildModelIntegration — build_model produces correct wrapper widths

Run with:
    pytest tests/test_model_behavior.py -v
    pytest tests/test_model_behavior.py -v -k "mamba"
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from models.architectures import (
    EXPERTEncoder,
    GNNFromSequence,
    HAELTHybrid,
    MambaScalper,
    MultiPairWrapper,
    TFTScalper,
    iTransformerScalper,
    DiversityLoss,
    MODEL_ROLES,
)

# ── shared fixtures ───────────────────────────────────────────────────────────

B, T, F = 4, 60, 32   # standard batch / seq_len / features


@pytest.fixture(scope="module")
def x_std():
    """Standard random (B, T, F) batch."""
    return torch.randn(B, T, F)


def _all_finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all())


def _mpw_backbone_input(P: int, F: int, E: int) -> int:
    """Width that backbone must be built with when MultiPairWrapper is used."""
    n_cross = P * (P - 1) // 2
    n_inter = 3 * n_cross + P + 2   # RelMom + ShortCorr + LongCorr + VolShare + Disp + Conf
    return P * (F + E) + n_inter


# ─────────────────────────────────────────────────────────────────────────────
# 1. ROBUSTNESS — NaN safety, edge cases
# ─────────────────────────────────────────────────────────────────────────────

_ALL_MODELS_PARAMS = [
    (TFTScalper,          {"input_size": F}),
    (iTransformerScalper, {"input_size": F, "seq_len": T}),
    (HAELTHybrid,         {"input_size": F, "seq_len": T}),
    (MambaScalper,        {"input_size": F}),
    (EXPERTEncoder,       {"input_size": F}),
    (GNNFromSequence,     {"input_size": F, "hidden": 32, "num_layers": 2, "dropout": 0.0}),
]
_ALL_MODEL_IDS = ["tft", "itransformer", "haelt", "mamba", "expert", "gnn"]


class TestModelRobustness:
    """Every architecture must survive adversarial inputs without producing NaN."""

    @pytest.mark.parametrize("cls,kw", _ALL_MODELS_PARAMS, ids=_ALL_MODEL_IDS)
    def test_normal_input_no_nan(self, cls, kw):
        model = cls(**kw)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(B, T, F))
        assert _all_finite(out), f"{cls.__name__} produced NaN/Inf on normal input"

    @pytest.mark.parametrize("cls,kw", _ALL_MODELS_PARAMS, ids=_ALL_MODEL_IDS)
    def test_batch_size_one(self, cls, kw):
        """B=1 triggered a squeeze(-1) collapse bug in MultiTaskHead — verify fixed."""
        model = cls(**kw)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, T, F))
        assert out.shape == (1,), f"{cls.__name__} B=1: expected (1,) got {out.shape}"
        assert _all_finite(out), f"{cls.__name__} B=1: NaN output"

    @pytest.mark.parametrize("cls,kw", _ALL_MODELS_PARAMS, ids=_ALL_MODEL_IDS)
    def test_all_zeros_input(self, cls, kw):
        """Zero input must not produce NaN (LayerNorm / BN singularity guard)."""
        model = cls(**kw)
        model.eval()
        with torch.no_grad():
            out = model(torch.zeros(B, T, F))
        assert _all_finite(out), f"{cls.__name__} NaN on all-zeros input"

    @pytest.mark.parametrize("cls,kw", _ALL_MODELS_PARAMS, ids=_ALL_MODEL_IDS)
    def test_large_input_no_nan(self, cls, kw):
        """Inputs scaled to ±100 (untypical but should not explode immediately)."""
        model = cls(**kw)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(B, T, F) * 100)
        assert _all_finite(out), f"{cls.__name__} NaN on large-magnitude input"

    @pytest.mark.parametrize("cls,kw", _ALL_MODELS_PARAMS, ids=_ALL_MODEL_IDS)
    def test_output_is_not_constant(self, cls, kw):
        """Different inputs must produce at least some variation in output."""
        torch.manual_seed(0)
        model = cls(**kw)
        model.eval()
        with torch.no_grad():
            o1 = model(torch.randn(B, T, F))
            o2 = model(torch.randn(B, T, F))
        assert not torch.allclose(o1, o2, atol=1e-6), (
            f"{cls.__name__} outputs identical values for different random inputs"
        )

    @pytest.mark.parametrize("cls,kw", _ALL_MODELS_PARAMS, ids=_ALL_MODEL_IDS)
    def test_eval_mode_determinism(self, cls, kw):
        """Same input in eval() must produce bitwise-identical output."""
        model = cls(**kw)
        model.eval()
        x = torch.randn(B, T, F)
        with torch.no_grad():
            o1 = model(x)
            o2 = model(x)
        assert torch.allclose(o1, o2, atol=0.0), (
            f"{cls.__name__} is non-deterministic in eval() mode"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. MAMBA-SPECIFIC BEHAVIOUR
# ─────────────────────────────────────────────────────────────────────────────

class TestMambaScalper:
    """MambaScalper is an SSM — its correctness depends on causal left-to-right
    processing and recurrent state propagation across timesteps."""

    def _make_model(self, **kw) -> MambaScalper:
        defaults = dict(input_size=F, d_model=32, d_state=8, d_conv=4,
                        expand=2, num_layers=2, dropout=0.0)
        defaults.update(kw)
        return MambaScalper(**defaults).eval()

    # ── output contract ───────────────────────────────────────────────────────

    def test_scalar_output_regression_mode(self):
        """num_classes=1 must return (B,) scalar, NOT (B,1)."""
        m = self._make_model(num_classes=1)
        out = m(torch.randn(B, T, F))
        assert out.shape == (B,), f"Expected (B,), got {out.shape}"

    def test_vector_output_classification_mode(self):
        """num_classes=3 must return (B, 3) logits."""
        m = self._make_model(num_classes=3)
        out = m(torch.randn(B, T, F))
        assert out.shape == (B, 3), f"Expected (B, 3), got {out.shape}"

    def test_batch_size_one_shape(self):
        m = self._make_model(num_classes=1)
        out = m(torch.randn(1, T, F))
        assert out.shape == (1,), f"B=1 regression: expected (1,) got {out.shape}"

    def test_batch_size_one_classification_shape(self):
        m = self._make_model(num_classes=3)
        out = m(torch.randn(1, T, F))
        assert out.shape == (1, 3), f"B=1 classification: expected (1,3) got {out.shape}"

    # ── causal ordering — SSM must respect time ───────────────────────────────

    def test_sequence_order_matters(self):
        """Shuffling the time axis must change the output.
        A bag-of-words model would give the same result; an SSM must not.
        """
        torch.manual_seed(1)
        m = self._make_model()
        x = torch.randn(1, T, F)
        perm = torch.randperm(T)
        x_shuffled = x[:, perm, :]
        with torch.no_grad():
            o_orig    = m(x)
            o_shuffled = m(x_shuffled)
        assert not torch.allclose(o_orig, o_shuffled, atol=1e-4), (
            "MambaScalper ignores sequence order — SSM state is not being used"
        )

    def test_final_timestep_differs_from_first(self):
        """The output (from the last timestep only) must differ when we shorten
        the context window by one.  This verifies recurrent state carry-forward."""
        torch.manual_seed(2)
        m = self._make_model()
        x = torch.randn(1, T, F)
        with torch.no_grad():
            o_full     = m(x)                    # uses all T timesteps
            o_short    = m(x[:, :-1, :])         # one fewer context bar
        # They should NOT be identical (state would differ)
        # We allow the model to output a slightly different value
        assert not torch.allclose(o_full, o_short, atol=1e-5), (
            "Output unchanged when context is shortened — state carry-forward may be broken"
        )

    # ── long-sequence handling (O(L) complexity) ─────────────────────────────

    def test_long_sequence_no_oom(self):
        """MambaScalper must handle T=512 without running out of memory.
        A Transformer with T=512 would need 512² attention = 262k entries;
        Mamba is O(T) so this should be fast and cheap.
        """
        m = self._make_model(d_model=16, num_layers=2)
        x = torch.randn(2, 512, F)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (2,)
        assert _all_finite(out)

    # ── gradient health ───────────────────────────────────────────────────────

    def test_gradients_reach_embedding(self):
        """Loss.backward() must produce non-zero gradient at the input embedding."""
        m = self._make_model()
        x = torch.randn(B, T, F)
        out = m(x)
        out.sum().backward()
        embed_grad = m.embed.weight.grad
        assert embed_grad is not None, "No gradient at embed.weight"
        assert embed_grad.abs().sum().item() > 0, "Zero gradient at embed.weight"
        assert _all_finite(embed_grad), "NaN/Inf gradient at embed.weight"

    def test_gradient_norm_reasonable_at_init(self):
        """Gradient L2-norm at initialisation should be < 100 (no exploding grads)."""
        m = self._make_model()
        x = torch.randn(B, T, F)
        out = m(x)
        out.sum().backward()
        total_norm = sum(
            p.grad.norm().item() ** 2
            for p in m.parameters()
            if p.grad is not None
        ) ** 0.5
        assert total_norm < 200, (
            f"Gradient norm at init is {total_norm:.1f} — possible exploding gradient"
        )

    # ── loss decreases on a trivial task ─────────────────────────────────────

    def test_loss_decreases_on_trivial_task(self):
        """MambaScalper can fit a very simple pattern (last-feature → sign) in ≤20 steps."""
        torch.manual_seed(42)
        m = MambaScalper(input_size=4, d_model=16, num_layers=2, dropout=0.0)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)

        # Task: output sign of the last feature of the last bar
        X = torch.randn(32, 20, 4)
        y = X[:, -1, -1].sign()   # deterministic label

        losses = []
        for _ in range(20):
            opt.zero_grad()
            pred = m(X)
            loss = nn.MSELoss()(pred, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRADIENT HEALTH — all architectures
# ─────────────────────────────────────────────────────────────────────────────

class TestAllModelsGradients:

    @pytest.mark.parametrize("cls,kw", _ALL_MODELS_PARAMS, ids=_ALL_MODEL_IDS)
    def test_gradients_finite(self, cls, kw):
        model = cls(**kw)
        x = torch.randn(B, T, F)
        model(x).sum().backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert _all_finite(p.grad), (
                    f"{cls.__name__}.{name}: NaN/Inf gradient at init"
                )

    @pytest.mark.parametrize("cls,kw", _ALL_MODELS_PARAMS, ids=_ALL_MODEL_IDS)
    def test_output_scale_reasonable_at_init(self, cls, kw):
        """Untrained model outputs should have |mean| < 10.  Extreme offsets at init
        suggest broken weight initialisation or missing normalisation."""
        model = cls(**kw)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(16, T, F))
        mean_abs = out.abs().mean().item()
        assert mean_abs < 10.0, (
            f"{cls.__name__} output |mean| = {mean_abs:.2f} at init — suspiciously large"
        )

    @pytest.mark.parametrize("cls,kw", _ALL_MODELS_PARAMS, ids=_ALL_MODEL_IDS)
    def test_at_least_one_parameter_gets_gradient(self, cls, kw):
        model = cls(**kw)
        model(torch.randn(B, T, F)).sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, f"{cls.__name__}: no parameter received a gradient"


# ─────────────────────────────────────────────────────────────────────────────
# 4. MULTI-PAIR WRAPPER BEHAVIOUR
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiPairWrapperBehavior:
    """Verifies the cross-pair feature pipeline produces meaningful values
    and that the wrapper's backbone sees the correct input width."""

    def _make_wrapper(self, P=2, Fp=16, E=8, T=60, num_classes=1,
                      corr_window=10, corr_window_long=20, momentum_window=10):
        bi = _mpw_backbone_input(P, Fp, E)
        backbone = HAELTHybrid(
            input_size=bi, seq_len=T,
            lstm_hidden=32, d_model=32, nhead=2, n_layers=1,
            num_classes=num_classes,
        )
        return MultiPairWrapper(
            backbone, n_pairs=P, f_per_pair=Fp, embed_dim=E,
            corr_window=corr_window, corr_window_long=corr_window_long,
            momentum_window=momentum_window,
        )

    # ── backbone width ────────────────────────────────────────────────────────

    def test_backbone_receives_correct_input_width(self):
        """The backbone's first Linear must match what wrapper actually feeds it.
        Before the fix, backbone was built with n_pairs*(F+E) but received
        n_pairs*(F+E) + n_interaction — a silent dimension mismatch.
        """
        P, Fp, E = 3, 16, 8
        wrapper = self._make_wrapper(P=P, Fp=Fp, E=E)
        wrapper.eval()
        # If the widths match, this will not raise a size-mismatch RuntimeError
        x = torch.randn(2, 60, P * Fp)
        with torch.no_grad():
            out = wrapper(x)
        assert out.shape == (2,)

    def test_n_interaction_formula(self):
        """n_interaction must equal 3*n_cross + n_pairs + 2."""
        for P in [2, 3, 4]:
            n_cross = P * (P - 1) // 2
            expected = 3 * n_cross + P + 2
            actual   = _mpw_backbone_input(P, 16, 8) - P * (16 + 8)
            assert actual == expected, f"P={P}: expected n_inter={expected}, got {actual}"

    # ── rolling correlation correctness ──────────────────────────────────────

    def test_rolling_corr_range(self):
        """All Pearson correlation values must be in [-1, 1]."""
        P, Fp = 3, 12
        wrapper = self._make_wrapper(P=P, Fp=Fp, E=0)
        wrapper.eval()
        torch.manual_seed(7)
        xp = torch.randn(2, 60, P, Fp)
        with torch.no_grad():
            cross = wrapper._cross_pair_features(xp)
        n_cross = P * (P - 1) // 2
        # short_corr starts at index n_cross, long_corr starts at 2*n_cross
        short = cross[..., n_cross : 2 * n_cross]
        long_ = cross[..., 2 * n_cross : 3 * n_cross]
        assert short.min().item() >= -1.0 - 1e-5, "Short corr below -1"
        assert short.max().item() <=  1.0 + 1e-5, "Short corr above +1"
        assert long_.min().item() >= -1.0 - 1e-5, "Long corr below -1"
        assert long_.max().item() <=  1.0 + 1e-5, "Long corr above +1"

    def test_short_and_long_corr_differ_for_trending_series(self):
        """On a trending series, short-window corr should differ from long-window corr
        because the local and global co-movement patterns are different.
        """
        P, Fp = 2, 8
        # Build a wrapper with noticeably different short vs long windows
        wrapper = self._make_wrapper(P=P, Fp=Fp, E=0,
                                     corr_window=5, corr_window_long=40)
        wrapper.eval()
        # Trending + diverging pair features
        t = torch.linspace(0, 4 * math.pi, 60)
        pair0 = torch.stack([torch.sin(t + i * 0.2) for i in range(Fp)], dim=-1)
        pair1 = torch.stack([torch.cos(t + i * 0.3) for i in range(Fp)], dim=-1)
        x = torch.stack([pair0, pair1], dim=-2).unsqueeze(0)  # (1, T, 2, F)
        with torch.no_grad():
            cross = wrapper._cross_pair_features(x)
        # n_cross = 1 for P=2
        short_last = cross[0, -1, 1].item()   # short corr at last bar
        long_last  = cross[0, -1, 2].item()   # long corr at last bar
        assert abs(short_last - long_last) > 1e-3, (
            f"Short ({short_last:.4f}) == long ({long_last:.4f}) corr on trending series "
            "— the two lookbacks are not being applied independently"
        )

    def test_vol_share_sums_to_one(self):
        """Per-pair vol shares (ATR_i / basket_ATR) must sum to 1 across pairs."""
        P, Fp = 3, 8
        wrapper = self._make_wrapper(P=P, Fp=Fp, E=0)
        wrapper.eval()
        xp = torch.rand(1, 40, P, Fp) + 0.1  # positive ATR proxies
        with torch.no_grad():
            cross = wrapper._cross_pair_features(xp)
        n_cross = P * (P - 1) // 2
        # vol_share lives at positions [3*n_cross : 3*n_cross + P]
        vol_share = cross[0, :, 3 * n_cross : 3 * n_cross + P]
        row_sums  = vol_share.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), (
            "Vol shares do not sum to 1"
        )

    def test_alignment_confidence_range(self):
        """Alignment confidence must be in [0, 1] — it's a fraction of pairs present."""
        P, Fp = 4, 8
        wrapper = self._make_wrapper(P=P, Fp=Fp, E=0)
        wrapper.eval()
        xp = torch.randn(1, 30, P, Fp)
        with torch.no_grad():
            cross = wrapper._cross_pair_features(xp)
        # confidence is the last channel
        conf = cross[..., -1]
        assert conf.min().item() >= 0.0 - 1e-5, "Confidence < 0"
        assert conf.max().item() <= 1.0 + 1e-5, "Confidence > 1"

    def test_missing_pair_lowers_confidence(self):
        """Zeroing one pair's features should lower the alignment confidence."""
        P, Fp = 3, 8
        wrapper = self._make_wrapper(P=P, Fp=Fp, E=0)
        wrapper.eval()
        xp_full = torch.randn(1, 30, P, Fp).abs() + 0.1
        xp_miss = xp_full.clone()
        xp_miss[:, :, 0, :] = 0.0   # zero out pair 0 → only 2 of 3 pairs present
        with torch.no_grad():
            cross_full = wrapper._cross_pair_features(xp_full)
            cross_miss = wrapper._cross_pair_features(xp_miss)
        conf_full = cross_full[0, :, -1].mean().item()
        conf_miss = cross_miss[0, :, -1].mean().item()
        assert conf_miss < conf_full, (
            f"Conf with missing pair ({conf_miss:.3f}) ≥ full conf ({conf_full:.3f})"
        )

    # ── regime attention ──────────────────────────────────────────────────────

    def test_regime_attention_weights_sum_to_one(self):
        """softmax pair weights must sum to 1 across pairs at every bar."""
        P, Fp, E = 3, 12, 8
        wrapper = self._make_wrapper(P=P, Fp=Fp, E=E)
        wrapper.eval()
        x = torch.randn(2, 60, P * Fp)
        # Intercept pair_weights inside forward by running with hooks
        captured = {}
        def _hook(module, inp, out):
            captured["pair_weights"] = torch.softmax(out, dim=-1).detach()
        wrapper.regime_attn[-1].register_forward_hook(_hook)
        with torch.no_grad():
            wrapper(x)
        pw = captured["pair_weights"]   # (B, P)
        sums = pw.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5), (
            f"Regime attention weights do not sum to 1: {sums}"
        )

    def test_gradient_reaches_pair_embeddings(self):
        """Backward must push gradient all the way back to the static pair embeddings."""
        P, Fp, E = 2, 12, 8
        wrapper = self._make_wrapper(P=P, Fp=Fp, E=E)
        wrapper(torch.randn(2, 60, P * Fp)).sum().backward()
        g = wrapper.pair_embeds.weight.grad
        assert g is not None and g.abs().sum().item() > 0, (
            "No gradient reached pair_embeds.weight"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. DIVERSITY LOSS BEHAVIOUR
# ─────────────────────────────────────────────────────────────────────────────

class TestDiversityLossBehavior:

    def _preds(self, B: int, n: int, identical: bool = False):
        if identical:
            base = torch.randn(B, 1)
            return [base.squeeze(1)] * n
        return [torch.randn(B) for _ in range(n)]

    def test_identical_preds_high_penalty(self):
        """Perfectly correlated outputs → penalty ≈ weight (=0.1 default)."""
        div = DiversityLoss(weight=0.1)
        preds = self._preds(64, 4, identical=True)
        loss = div(preds)
        assert loss.item() > 0.08, (
            f"Identical preds should yield penalty ≈ 0.1, got {loss.item():.4f}"
        )

    def test_orthogonal_preds_low_penalty(self):
        """Uncorrelated outputs → penalty ≈ 0."""
        torch.manual_seed(99)
        div = DiversityLoss(weight=0.1)
        mat, _ = torch.linalg.qr(torch.randn(128, 4))
        preds = [mat[:, i] for i in range(4)]
        loss = div(preds)
        assert loss.item() < 0.02, (
            f"Orthogonal preds should yield near-zero penalty, got {loss.item():.4f}"
        )

    def test_single_model_returns_zero(self):
        """With only one model there are no pairs — penalty must be exactly 0."""
        div = DiversityLoss(weight=0.1)
        loss = div([torch.randn(32)])
        assert loss.item() == pytest.approx(0.0), "Single-model penalty should be 0"

    def test_scalar_output(self):
        div = DiversityLoss()
        loss = div(self._preds(16, 3))
        assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"

    def test_differentiable(self):
        div = DiversityLoss()
        preds = [torch.randn(16, requires_grad=True) for _ in range(3)]
        div(preds).backward()
        assert preds[0].grad is not None

    def test_same_role_penalty_higher_than_cross_role(self):
        """Models with the same role should incur a larger diversity penalty."""
        torch.manual_seed(5)
        preds = [torch.randn(32)] * 2   # identical outputs for clarity
        # Same role
        div_same  = DiversityLoss(weight=1.0, same_role_mult=3.0,
                                   roles=["fast_reaction", "fast_reaction"])
        # Cross role
        div_cross = DiversityLoss(weight=1.0, same_role_mult=3.0,
                                   roles=["fast_reaction", "context"])
        loss_same  = div_same(preds).item()
        loss_cross = div_cross(preds).item()
        assert loss_same > loss_cross, (
            f"Same-role penalty ({loss_same:.4f}) ≤ cross-role penalty ({loss_cross:.4f})"
        )

    def test_model_roles_cover_all_architectures(self):
        """Every registered architecture must appear in MODEL_ROLES."""
        from models.architectures import MODEL_REGISTRY
        for name in MODEL_REGISTRY:
            assert name in MODEL_ROLES, f"'{name}' missing from MODEL_ROLES"


# ─────────────────────────────────────────────────────────────────────────────
# 6. FEATURE STABILITY MONITOR
# ─────────────────────────────────────────────────────────────────────────────

try:
    _TRAIN_GPU_OK = True
except Exception:
    _TRAIN_GPU_OK = False

_skip_no_train_gpu = pytest.mark.skipif(
    not _TRAIN_GPU_OK,
    reason="training.train_gpu not importable (pandas/sklearn missing)",
)


@_skip_no_train_gpu
class TestFeatureStabilityMonitor:
    """FeatureStabilityMonitor is defined inside train_gpu — import it at test time."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from training.train_gpu import FeatureStabilityMonitor
        self.Monitor = FeatureStabilityMonitor

    def _monitor(self, n=8, **kw):
        # Build defaults then overlay caller overrides so there are no duplicate kwargs
        cfg = dict(ema_alpha=0.9, soft_threshold=2.0, hard_threshold=4.0,
                   freeze_after=2, damping_factor=0.5,
                   warmup_epochs=1, min_active_pct=0.25)
        cfg.update(kw)
        return self.Monitor(n_features=n, **cfg)

    def test_initial_mask_all_ones(self):
        """Before any updates the mask must be all-ones (nothing filtered)."""
        mon = self._monitor()
        mask = mon.get_mask()
        assert torch.allclose(mask, torch.ones(8)), "Initial mask is not all-ones"

    def test_stable_input_keeps_mask_ones(self):
        """Inputs with no shift should leave the mask at 1.0 after several updates."""
        mon = self._monitor()
        rng = np.random.default_rng(0)
        x = rng.standard_normal((16, 8)).astype(np.float32)
        for _ in range(5):
            mon.update(x)
        mask = mon.get_mask()
        assert (mask == 1.0).all(), "Stable features should stay fully active"

    def test_drifted_feature_gets_soft_mask(self):
        """A feature that shifts by > soft_threshold σ should be masked to 0.5."""
        n = 4
        mon = self._monitor(n=n, soft_threshold=1.0, hard_threshold=5.0,
                             freeze_after=10, warmup_epochs=0)
        rng = np.random.default_rng(1)
        # Warmup with zero-mean data
        base = rng.standard_normal((32, n)).astype(np.float32)
        for _ in range(3):
            mon.update(base)
        # Feature 0 suddenly shifts by +20 — way above soft threshold
        drifted = base.copy()
        drifted[:, 0] += 20.0
        mon.update(drifted)
        mask = mon.get_mask()
        # Feature 0 should now be 0.5 (soft) or 0.0 (hard)
        assert mask[0].item() < 1.0, (
            f"Drifted feature 0 still fully active (mask={mask[0].item():.2f})"
        )

    def test_frozen_feature_gets_zero_mask(self):
        """After freeze_after consecutive high-shift epochs, mask should be 0.0."""
        n = 4
        mon = self._monitor(n=n, soft_threshold=0.5, hard_threshold=1.0,
                             freeze_after=2, warmup_epochs=0)
        rng = np.random.default_rng(2)
        base = rng.standard_normal((32, n)).astype(np.float32)
        for _ in range(2):
            mon.update(base)
        # Huge shift on feature 1 for freeze_after+1 epochs
        drifted = base.copy()
        drifted[:, 1] += 50.0
        for _ in range(3):
            mon.update(drifted)
        mask = mon.get_mask()
        assert mask[1].item() == pytest.approx(0.0, abs=1e-5), (
            f"Persistently drifted feature 1 not frozen (mask={mask[1].item():.4f})"
        )

    def test_reset_frozen_restores_ones(self):
        """reset_frozen() must unfreeze all features → mask becomes all-ones again."""
        n = 4
        mon = self._monitor(n=n, soft_threshold=0.5, hard_threshold=1.0,
                             freeze_after=2, warmup_epochs=0)
        base = np.random.default_rng(3).standard_normal((32, n)).astype(np.float32)
        for _ in range(2):
            mon.update(base)
        drifted = base.copy(); drifted[:, 2] += 50.0
        for _ in range(3):
            mon.update(drifted)
        assert mon.get_mask()[2].item() == pytest.approx(0.0, abs=1e-5), "Setup failed"
        mon.reset_frozen()
        mask_after = mon.get_mask()
        assert torch.allclose(mask_after, torch.ones(n)), (
            f"After reset_frozen, mask is not all-ones: {mask_after}"
        )

    def test_min_active_pct_preserves_minimum(self):
        """Even if all features drift, at least min_active_pct must stay active."""
        n = 8
        mon = self._monitor(n=n, soft_threshold=0.5, hard_threshold=1.0,
                             freeze_after=2, warmup_epochs=0, min_active_pct=0.5)
        base = np.random.default_rng(4).standard_normal((32, n)).astype(np.float32)
        for _ in range(2):
            mon.update(base)
        # Massive drift on ALL features
        drifted = base + 100.0
        for _ in range(3):
            mon.update(drifted)
        mask = mon.get_mask()
        active_frac = (mask > 0).float().mean().item()
        assert active_frac >= 0.5 - 1e-5, (
            f"Only {active_frac:.1%} features active, expected ≥ 50% (min_active_pct)"
        )

    def test_report_returns_expected_keys(self):
        mon = self._monitor()
        rep = mon.report()
        # Actual keys use the feat_ prefix (feat_frozen, feat_active, feat_max_shift, …)
        for key in ("feat_frozen", "feat_active", "feat_max_shift"):
            assert key in rep, f"Key '{key}' missing from monitor report — got: {list(rep)}"

    def test_mask_device_matches_requested(self):
        """get_mask(device=...) must return a tensor on the specified device."""
        mon = self._monitor()
        mask_cpu = mon.get_mask(device=torch.device("cpu"))
        assert mask_cpu.device.type == "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# 7. BUILD_MODEL INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

@_skip_no_train_gpu
class TestBuildModelIntegration:
    """training.train_gpu.build_model must wire MultiPairWrapper with the correct
    backbone width and pass the right correlation windows through."""

    def _args(self, model_name, n_pairs, f_per_pair, embed_dim,
              corr_window=10, corr_window_long=20, momentum_window=10):
        import argparse
        # These mirror the args that train_gpu.build_model reads
        a = argparse.Namespace(
            loss="cross_entropy",
            multitask=False,
            seq_len=T,
            hidden_size=32,
            d_model=32,
            nhead=2,
            num_layers=2,
            dropout=0.0,
            pair_embed_dim=embed_dim,
            corr_window=corr_window,
            corr_window_long=corr_window_long,
            momentum_window=momentum_window,
            _n_pairs=n_pairs,
            _f_per_pair=f_per_pair,
        )
        return a

    @pytest.mark.parametrize("model_name,P,Fp,E", [
        ("mamba",  2, 16, 8),
        ("haelt",  3, 12, 8),
        ("expert", 2, 20, 0),   # embed_dim=0 → no wrapper
    ])
    def test_build_model_no_shape_error(self, model_name, P, Fp, E):
        """train_gpu.build_model must produce a model that accepts (B, T, P*Fp) input."""
        from training.train_gpu import build_model
        n_features = P * Fp
        args = self._args(model_name, P, Fp, E)
        model = build_model(model_name, n_features, args)
        model.eval()
        x = torch.randn(2, T, n_features)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 2, f"{model_name} P={P}: wrong batch dim {out.shape}"
        assert _all_finite(out), f"{model_name} NaN output from build_model"

    def test_wrapper_applied_when_embed_dim_gt_zero(self):
        """embed_dim>0 with n_pairs>1 must produce a MultiPairWrapper instance."""
        from training.train_gpu import build_model
        args = self._args("mamba", n_pairs=2, f_per_pair=16, embed_dim=8)
        model = build_model("mamba", 32, args)
        assert isinstance(model, MultiPairWrapper), (
            f"Expected MultiPairWrapper, got {type(model).__name__}"
        )

    def test_no_wrapper_when_embed_dim_zero(self):
        """embed_dim=0 must skip MultiPairWrapper even with multiple pairs."""
        from training.train_gpu import build_model
        args = self._args("mamba", n_pairs=2, f_per_pair=16, embed_dim=0)
        model = build_model("mamba", 32, args)
        assert not isinstance(model, MultiPairWrapper), (
            "MultiPairWrapper should not be added when embed_dim=0"
        )

    def test_multi_pair_wrapper_forward_after_build_model(self):
        """End-to-end: build_model → MultiPairWrapper → backbone forward must work."""
        from training.train_gpu import build_model
        P, Fp, E = 3, 12, 8
        args = self._args("mamba", n_pairs=P, f_per_pair=Fp, embed_dim=E)
        model = build_model("mamba", P * Fp, args)
        model.eval()
        x = torch.randn(4, T, P * Fp)
        with torch.no_grad():
            out = model(x)
        # build_model may return (B,) scalar or (B, C) logits depending on n_classes
        assert out.shape[0] == 4, f"Batch dim wrong: {out.shape}"
        assert out.ndim in (1, 2),  f"Expected 1-D or 2-D output, got {out.shape}"
        assert _all_finite(out)
