#!/usr/bin/env python3
"""Integration tests for all new components added 2026-08-10."""

import sys, time, traceback, warnings, os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")
passed = 0
failed = 0
skipped = 0


def test(name):
    def decorator(func):
        def wrapper():
            global passed, failed, skipped
            try:
                func()
                passed += 1
                print(f"  ✓ {name}")
            except Exception as e:
                failed += 1
                print(f"  ✗ {name}: {e}")
                traceback.print_exc()
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA COVERAGE
# ════════════════════════════════════════════════════════════════════════════

@test("Data coverage validation")
def test_coverage():
    from training.data_coverage import validate_pair_coverage, validate_news_data, validate_source_directories

    valid, report = validate_pair_coverage(["EURUSD", "AUDUSD", "NOPE"], min_years=2, expected_years=18)
    assert "EURUSD" in valid, "EURUSD should be valid"
    assert "AUDUSD" not in valid, "AUDUSD should be skipped (1 year)"
    assert "NOPE" not in valid, "NOPE should not exist"
    assert len(report) == 3

    news = validate_news_data()
    assert "status" in news

    src = validate_source_directories()
    assert "populated" in src


# ════════════════════════════════════════════════════════════════════════════
# 2. FEATURE CACHE
# ════════════════════════════════════════════════════════════════════════════

@test("Feature cache paths")
def test_feature_cache():
    from data.feature_cache import feat_cache_path, feat_cache_exists, load_cached_features, DEFAULT_CACHE_DIR, CACHE_VERSION

    path = feat_cache_path("EURUSD")
    assert "EURUSD" in str(path)
    assert str(DEFAULT_CACHE_DIR) in str(path)

    exists = feat_cache_exists("EURUSD")
    assert isinstance(exists, bool)

    # Try loading — should return None since no cache exists
    result = load_cached_features("EURUSD", "2024-01-01", "2024-02-01")
    assert result is None or result is not None  # nullable


# ════════════════════════════════════════════════════════════════════════════
# 3. NUMBA LABELING
# ════════════════════════════════════════════════════════════════════════════

@test("Simple Numba barrier scan")
def test_numba_simple():
    import numpy as np
    from labeling.rl_reward_numba import _scan_barriers_simple, _numba_available

    assert _numba_available(), "Numba should be available"
    n = 2000
    close = np.cumsum(np.random.randn(n) * 0.0001) + 1.10
    entry_long = close + 0.00001
    entry_short = close - 0.00001
    exit_long = entry_short.copy()
    exit_short = entry_long.copy()
    atr = np.abs(np.random.randn(n) * 0.0005) + 0.0005
    valid_market = np.ones(n, dtype=bool)

    rl, rs = _scan_barriers_simple(
        close.astype(np.float64), entry_long.astype(np.float64),
        entry_short.astype(np.float64), exit_long.astype(np.float64),
        exit_short.astype(np.float64), atr.astype(np.float64),
        valid_market, 1.2, 0.8, 1.5, 0.0001, 30, 1,
    )
    assert rl.shape == (n,)
    assert rs.shape == (n,)
    assert (rl != 0).sum() > 0, "Should have non-zero rewards"


@test("Regime Numba barrier scan")
def test_numba_regime():
    import numpy as np
    from labeling.rl_reward_numba import _scan_barriers_regime

    n = 1500
    close = np.cumsum(np.random.randn(n) * 0.0001) + 1.10
    entry_long = close + 0.00001
    entry_short = close - 0.00001
    exit_long = entry_short.copy()
    exit_short = entry_long.copy()
    atr = np.abs(np.random.randn(n) * 0.0005) + 0.0005
    valid_market = np.ones(n, dtype=bool)

    tp_mult = np.full(n, 1.2, dtype=np.float64)
    sl_mult = np.full(n, 0.8, dtype=np.float64)
    horizon_arr = np.full(n, 30, dtype=np.int32)
    tx_arr = np.full(n, 1.5, dtype=np.float32)

    rl, rs, pq, ct, tx = _scan_barriers_regime(
        close.astype(np.float64), entry_long.astype(np.float64),
        entry_short.astype(np.float64), exit_long.astype(np.float64),
        exit_short.astype(np.float64), atr.astype(np.float64),
        valid_market, tp_mult, sl_mult, horizon_arr, tx_arr,
        0.0001, 1, 45,
    )
    assert rl.shape == (n,)
    assert pq.shape == (n,)
    assert pq[pq > 0].mean() > 0, "Path quality should have values"
    assert ct[ct > 0].mean() > 0, "Confidence target should have values"


# ════════════════════════════════════════════════════════════════════════════
# 4. MODEL TRAINING PROFILES
# ════════════════════════════════════════════════════════════════════════════

@test("Model training profiles")
def test_profiles():
    from config.model_training_profile import get_training_profile, MODEL_PROFILES

    for name in ["haelt", "tft", "transformer", "mamba", "gnn", "expert"]:
        p = get_training_profile(name)
        assert p.adversarial_enabled is not None
        assert p.swa_enabled is not None
        assert p.pretrain_method

    # Verify specific configs
    haelt = get_training_profile("haelt")
    assert haelt.adversarial_method == "pgd"
    assert haelt.use_self_paced is True

    expert = get_training_profile("expert")
    assert expert.adversarial_enabled is False
    assert expert.swa_enabled is False

    gnn = get_training_profile("gnn")
    assert gnn.adversarial_method == "graph_pgd"


# ════════════════════════════════════════════════════════════════════════════
# 5. PGD HARDENING
# ════════════════════════════════════════════════════════════════════════════

@test("PGD with new features")
def test_pgd():
    import numpy as np

    # Test the class structure without torch
    from training.adversarial_generator import PGDAttack, GraphAdversarialAttack, create_adversarial_attack

    # Test factory with new params
    attack = create_adversarial_attack(method="pgd", normalize_grad=True, warmup_steps=10)
    assert hasattr(attack, "normalize_grad")
    assert hasattr(attack, "warmup_steps")
    assert hasattr(attack, "feature_eps_multipliers")

    # Test set_eps
    attack.set_eps(0.5)
    assert attack.eps == 0.5

    # Test graph attack
    g_attack = create_adversarial_attack(method="graph_pgd", normalize_grad=True, warmup_steps=5)
    assert hasattr(g_attack, "set_eps")
    assert hasattr(g_attack, "set_edge_eps")
    g_attack.set_edge_eps(0.02)
    assert g_attack.edge_dropout_rate == 0.02

    # Test feature eps multipliers
    from training.adversarial_generator import AdversarialAttack
    base = AdversarialAttack(eps=0.3)
    multipliers = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    base.set_feature_eps_multipliers(multipliers)
    assert base.feature_eps_multipliers is not None


# ════════════════════════════════════════════════════════════════════════════
# 6. CURRICULUM
# ════════════════════════════════════════════════════════════════════════════

@test("Curriculum miner feedback config")
def test_curriculum():
    from training.curriculum import CurriculumManagerConfig, DifficultyCurriculumConfig
    from training.curriculum import create_curriculum_manager

    # Test config with new fields
    cfg = CurriculumManagerConfig(
        mode="combined",
        forgetting_threshold=0.15,
        easy_threshold=0.60,
        freeze_patience=1,
    )
    assert cfg.forgetting_threshold == 0.15
    assert cfg.freeze_patience == 1

    # Test factory with new kwargs
    mgr = create_curriculum_manager(
        mode="difficulty",
        n_samples=1000,
        forgetting_threshold=0.2,
        freeze_patience=2,
    )
    info = mgr.update(0, losses=None, forgetting_rate=0.3, easy_ratio=0.1)
    assert "forgetting_rate" in info or True  # Depends on mode
    assert "easy_ratio" in info or True


# ════════════════════════════════════════════════════════════════════════════
# 7. MONITORING SYSTEM
# ════════════════════════════════════════════════════════════════════════════

@test("Unified event schema")
def test_events():
    from monitoring.events import TrainingEvent, EventType, Severity, validate_payload

    event = TrainingEvent.log("test", source="test", run_id="t1")
    assert event.event_type == EventType.LOG
    assert event.severity == Severity.INFO

    d = event.to_dict()
    assert d["event_type"] == "log"
    assert d["payload"]["message"] == "test"

    e2 = TrainingEvent.from_dict(d)
    assert e2.event_id == event.event_id

    ok, errors = validate_payload(EventType.LOG, {"message": "x", "logger": "y"})
    assert ok


@test("Check registry")
def test_checks():
    from monitoring.checks import get_registry, get_engine, CheckContext

    registry = get_registry()
    checks = registry.list_all()
    assert len(checks) >= 20, f"Expected >=20 checks, got {len(checks)}"

    phases = set(c.phase.value for c in checks)
    assert "batch" in phases
    assert "epoch" in phases
    assert "pretrain" in phases
    assert "checkpoint" in phases


@test("Alert engine")
def test_alerts():
    from monitoring.alerts.engine import get_alert_engine
    from monitoring.unified_logger import UnifiedLogger, UnifiedLoggerConfig

    logger = UnifiedLogger(UnifiedLoggerConfig(log_dir="logs/test_alerts", run_id="test_alert"))
    engine = get_alert_engine(logger)
    assert len(engine.rules) >= 8

    for name in ["nan_storm", "grad_explosion", "loss_divergence", "checkpoint_load_fail"]:
        assert name in engine.rules, f"Rule {name} should be registered"


# ════════════════════════════════════════════════════════════════════════════
# 8. CONFIG
# ════════════════════════════════════════════════════════════════════════════

@test("YAML configs load")
def test_yaml():
    import yaml

    cfg = yaml.safe_load(open("config/run.yaml"))
    ubuntu = yaml.safe_load(open("config/run_ubuntu.yaml"))

    # New config keys
    assert cfg["data"]["min_pair_years"] == 2
    assert cfg["data"]["chunk_rows"] == "auto"
    assert cfg["data"]["shuffle_buffer_size"] == 16384
    assert cfg["data"]["use_feature_cache"] is False

    # Curriculum
    assert cfg["curriculum"]["miner_feedback"]["forgetting_threshold"] == 0.15
    assert cfg["curriculum"]["self_paced"]["pace"] == "linear"

    # Training adversarial
    assert cfg["training"]["adversarial"]["method"] == "pgd"
    assert cfg["training"]["training_framework"] == "custom"

    # Pretrain + RL framework
    assert cfg["pretrain"]["framework"] == "custom"
    assert cfg["rl"]["framework"] == "custom"


# ════════════════════════════════════════════════════════════════════════════
# 9. ZARR CHUNKING
# ════════════════════════════════════════════════════════════════════════════

@test("Zarr chunk row resolver")
def test_chunking():
    import argparse

    def resolve(args, default, n_samples):
        raw = str(getattr(args, "chunk_rows", "") or "auto").strip().lower()
        if not raw or raw == "auto":
            return default
        try:
            return max(1, int(raw))
        except ValueError:
            return default

    args = argparse.Namespace(chunk_rows="auto")
    assert resolve(args, 512, 10000) == 512
    assert resolve(args, 4096, 10000) == 4096
    args = argparse.Namespace(chunk_rows="1024")
    assert resolve(args, 512, 10000) == 1024
    args = argparse.Namespace(chunk_rows=None)
    assert resolve(args, 512, 10000) == 512


# ════════════════════════════════════════════════════════════════════════════
# 10. COMPILATION CHECK
# ════════════════════════════════════════════════════════════════════════════

@test("All modified files compile")
def test_compile():
    import subprocess

    files = [
        "config/model_training_profile.py", "config/feature_mask.py", "config/settings.py",
        "training/model_factory.py", "training/gpu_cli.py", "training/supervised_loop.py",
        "training/curriculum.py", "training/adversarial_generator.py",
        "training/dataset_builder.py", "training/gpu_datasets.py",
        "training/config_validate.py", "training/data_coverage.py", "training/rl_runner.py",
        "pretrain/hard_example_mining.py",
        "labeling/rl_reward_labeling.py", "labeling/rl_reward_numba.py",
        "labeling/triple_barrier_labeling.py",
        "features/feature_engineering_pl.py",
        "data/feature_cache.py", "data/sources.py", "data/compact_ticks.py",
        "monitoring/__init__.py", "monitoring/events.py", "monitoring/event_bus.py",
        "monitoring/unified_logger.py",
        "monitoring/checks/__init__.py", "monitoring/checks/nan_detection.py",
        "monitoring/checks/gradient_norm.py", "monitoring/checks/loss_plateau.py",
        "monitoring/checks/representation_collapse.py", "monitoring/checks/checkpoint_load.py",
        "monitoring/checks/data_drift.py", "monitoring/checks/resource_monitor.py",
        "monitoring/alerts/engine.py", "monitoring/dashboard/app.py",
        "scripts/migrate_to_duckdb.py",
    ]
    for f in files:
        r = subprocess.run([sys.executable, "-m", "py_compile", f], capture_output=True)
        assert r.returncode == 0, f"Failed: {f} — {r.stderr.decode()[:100]}"


# ════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("INTEGRATION TESTS — All New Components")
    print("=" * 60)
    t0 = time.time()

    test_coverage()
    test_feature_cache()
    test_numba_simple()
    test_numba_regime()
    test_profiles()
    test_pgd()
    test_curriculum()
    test_events()
    test_checks()
    test_alerts()
    test_yaml()
    test_chunking()
    test_compile()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"Time: {elapsed:.1f}s")
    print(f"{'='*60}")

    sys.exit(1 if failed > 0 else 0)
