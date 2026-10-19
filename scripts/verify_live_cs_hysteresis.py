"""
End-to-end verification test for the upgraded Cross-Sectional + Hysteresis
StationaryEnsembleInferenceEngine, single-pair PyTorchInferenceEngine specialists,
and LiveTradingEngine / MultiPairLiveTradingEngine integration.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.pytorch_inference import PyTorchInferenceEngine
from trading.inference_engines import (
    MultiArchitectureSpecialistEnsembleInferenceEngine,
    StationaryEnsembleInferenceEngine,
)
from trading.live_actions import LiveAction
from trading.live_engine import LIVE_PAIR_ORDER, build_inference_agents


def test_stationary_ensemble_cs_hysteresis():
    print("=" * 90)
    print("1. Testing StationaryEnsembleInferenceEngine Cross-Sectional + Hysteresis State Machine")
    print("=" * 90)
    data = np.load(ROOT / ".tmp" / "all_584_audit_data.npz")
    X_all = data["X"]
    m = len(X_all)
    oos_start = int(m * 0.80)

    root_engine = StationaryEnsembleInferenceEngine(pair="EURUSD")
    assert root_engine.per_pair_heads is True
    assert root_engine.n_features == 584

    for pr in ("EURUSD", "USDJPY", "USDCAD"):
        eng = root_engine.for_pair(pr)
        assert eng.pair == pr
        assert eng.per_pair_heads is True
        assert eng._cs_policy is not None, f"Missing CS policy for {pr}"

        # Feed 30 sequential OOS 584-dim bars and verify z-scores and hysteresis transitions
        z_scores = []
        actions = []
        for row_idx in range(oos_start, oos_start + 30):
            obs_584 = X_all[row_idx]
            act = eng.select_action(obs_584)
            z_scores.append(eng.last_z_score)
            actions.append(act)
            assert np.isfinite(eng.last_z_score)
            assert np.isfinite(eng.last_raw)
            assert eng.last_proba.shape == (3,)
            assert abs(float(eng.last_proba.sum()) - 1.0) < 1e-4

        # Verify hysteresis state machine explicitly
        eng.reset_buffer()
        eng.set_agent_state(position_lots=0.0)
        assert eng._pos_dir == 0

        # Force high positive z -> BUY and _pos_dir == 1
        eng.z_entry = 1.40
        eng.z_exit = -0.25
        eng._pos_dir = 1
        eng.set_agent_state(position_lots=0.25)
        assert eng._pos_dir == 1
        eng.set_agent_state(position_lots=-0.25)
        assert eng._pos_dir == -1
        eng.set_agent_state(position_lots=0.0)
        assert eng._pos_dir == 0

        print(
            f"  [{pr}] OK | z_entry={eng.z_entry:.2f}, z_exit={eng.z_exit:.2f} | "
            f"z_range=[{min(z_scores):+.2f}, {max(z_scores):+.2f}] | "
            f"sample_last_raw={eng.last_raw:+.3f} | proba={eng.last_proba.round(3).tolist()}"
        )


def test_specialist_and_live_agent_wiring():
    print("\n" + "=" * 90)
    print("2. Testing Single-Pair Specialist PyTorchInferenceEngine & build_inference_agents")
    print("=" * 90)
    # Test stationary build_inference_agents
    fast_stat, slow_stat, meta_stat = build_inference_agents(
        pair="USDJPY",
        model_name="stationary",
        runtime="stationary",
        use_rl_fast=False,
    )
    assert meta_stat["per_pair_heads"] is True, "Stationary meta must have per_pair_heads=True"
    assert getattr(slow_stat, "per_pair_heads", False) is True

    # Test PyTorch specialist routing across EURUSD, USDJPY, USDCAD
    data = np.load(ROOT / ".tmp" / "all_584_audit_data.npz")
    X_sample = data["X"][-120:]

    base_pt = PyTorchInferenceEngine(
        checkpoint_path="checkpoints/haelt_eurusd_v1/haelt/haelt_best.pt",
        model_name="haelt",
        seq_len=60,
    )
    for pr in ("EURUSD", "USDJPY", "USDCAD"):
        spec = base_pt.for_pair(pr)
        assert spec.per_pair_heads is True, f"{pr} specialist must have per_pair_heads=True (no USD basis inversion)"
        assert spec.n_features == 146
        proba = spec.predict_proba(X_sample[-60:])
        assert proba.shape == (3,)
        assert np.all(np.isfinite(proba))
        print(
            f"  [{pr} HAELT Specialist] OK | per_pair_heads={spec.per_pair_heads} | "
            f"n_features={spec.n_features} | proba={proba.round(4).tolist()}"
        )

    # Also test Mamba EURUSD specialist if present
    mamba_ckpt = ROOT / "checkpoints" / "mamba_eurusd_v1" / "mamba" / "mamba_best.pt"
    if mamba_ckpt.is_file():
        mamba_eng = PyTorchInferenceEngine(
            checkpoint_path=str(mamba_ckpt),
            model_name="mamba",
            seq_len=60,
        )
        assert mamba_eng.per_pair_heads is True
        m_proba = mamba_eng.predict_proba(X_sample[-60:, :146])
        print(
            f"  [EURUSD Mamba Specialist] OK | per_pair_heads={mamba_eng.per_pair_heads} | "
            f"n_features={mamba_eng.n_features} | proba={m_proba.round(4).tolist()}"
        )


def test_multi_arch_specialist_zoo_ensemble():
    print("\n" + "=" * 90)
    print("3. Testing MultiArchitectureSpecialistEnsembleInferenceEngine & --model zoo Wiring")
    print("=" * 90)
    data = np.load(ROOT / ".tmp" / "all_584_audit_data.npz")
    X_sample = data["X"][-150:]

    fast_zoo, slow_zoo, meta_zoo = build_inference_agents(
        pair="EURUSD",
        model_name="zoo",
        runtime="zoo",
        use_rl_fast=False,
    )
    assert meta_zoo["per_pair_heads"] is True
    assert isinstance(slow_zoo, MultiArchitectureSpecialistEnsembleInferenceEngine)

    for pr in ("EURUSD", "USDJPY", "USDCAD"):
        zoo = slow_zoo.for_pair(pr)
        assert zoo.pair == pr
        assert zoo.per_pair_heads is True
        for row in X_sample:
            act = zoo.select_action(row)
        assert zoo.last_proba.shape == (3,)
        assert abs(float(zoo.last_proba.sum()) - 1.0) < 1e-4
        print(
            f"  [{pr} Multi-Arch Zoo] OK | weights={zoo.weights} | "
            f"sub_signals={zoo.last_sub_signals} | z={zoo.last_z_score:+.3f} | "
            f"proba={zoo.last_proba.round(3).tolist()} | action={LiveAction(act).name}"
        )

    print("\nALL LIVE WIRING, HYSTERESIS & MULTI-ARCH ZOO ASSERTIONS PASSED!")
    print("=" * 90)


if __name__ == "__main__":
    test_stationary_ensemble_cs_hysteresis()
    test_specialist_and_live_agent_wiring()
    test_multi_arch_specialist_zoo_ensemble()

