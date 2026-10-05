"""Regression tests for the live-risk / RL-deploy audit fixes (H1, H2, M1, M2, M7, low)."""

from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _live_risk_copy() -> dict:
    from config import settings

    return copy.deepcopy(settings.LIVE_RISK)


YAML_RISK = {
    "daily_loss_limit": 0.03,
    "kelly_fraction": 0.25,
    "max_drawdown_halt": 0.1,
    "max_position_pct": 0.05,
    "pip_value": 10.0,
    "regime_scale": {"ranging": 0.7, "trending": 1.0, "unknown": 0.3, "volatile": 0.5},
    "session_limits": {
        "london": {"max_lots": 2.0, "max_open_trades": 6},
        "ny": {"max_lots": 2.0, "max_open_trades": 6},
        "asia": {"max_lots": 1.0, "max_open_trades": 3},
        "asia_london": {"max_lots": 1.5, "max_open_trades": 4},
        "london_ny": {"max_lots": 2.0, "max_open_trades": 6},
        "off": {"max_lots": 0.0, "max_open_trades": 0},
    },
    "var_confidence": 0.99,
    "var_max_pct": 0.02,
}


# ── H1: run-YAML risk: block -> LIVE_RISK ────────────────────────────────────


def test_apply_yaml_risk_overrides_session_and_regime_limits():
    from risk.yaml_risk import apply_yaml_risk

    lr = _live_risk_copy()
    london_hours = lr["session_limits"]["london"].get("hours_local")
    changes = apply_yaml_risk(YAML_RISK, lr)

    sl = lr["session_limits"]
    assert sl["london"]["max_lots"] == 2.0 and sl["london"]["max_open_trades"] == 6
    assert sl["ny"]["max_lots"] == 2.0 and sl["ny"]["max_open_trades"] == 6
    assert sl["london_ny"]["max_lots"] == 2.0 and sl["london_ny"]["max_open_trades"] == 6
    assert sl["asia_london"]["max_open_trades"] == 4
    assert sl["off"]["max_lots"] == 0.0 and sl["off"]["max_open_trades"] == 0
    # tz / session hours are preserved from LIVE_RISK defaults
    assert sl["london"].get("hours_local") == london_hours

    rs = lr["regime_scale"]
    assert rs["trending"] == 1.0
    assert rs["mean_rev"] == 0.7
    assert rs["crisis"] == 0.5
    assert rs["unknown"] == 0.3
    assert "ranging" not in rs and "volatile" not in rs
    assert lr["var_max_pct"] == 0.02
    assert "session_limits.london.max_lots" in changes


def test_apply_yaml_risk_rejects_negative_values():
    from risk.yaml_risk import apply_yaml_risk

    with pytest.raises(ValueError):
        apply_yaml_risk({"daily_loss_limit": -0.1}, _live_risk_copy())


def test_load_yaml_risk_unparseable_file_raises(tmp_path):
    from risk.yaml_risk import load_yaml_risk

    bad = tmp_path / "run.yaml"
    bad.write_text("risk: [unclosed\n", encoding="utf-8")
    with pytest.raises(Exception):
        load_yaml_risk(bad)
    assert load_yaml_risk(tmp_path / "missing.yaml") is None


def test_apply_run_config_risk_uses_repo_run_yaml():
    from risk.yaml_risk import apply_run_config_risk

    lr = _live_risk_copy()
    res = apply_run_config_risk(ROOT / "config" / "run.yaml", live_risk=lr)
    assert res["applied"] is True
    assert res["source"].endswith("run.yaml")
    assert lr["session_limits"]["london"]["max_lots"] == 2.0
    assert lr["session_limits"]["off"]["max_open_trades"] == 0


def test_resolve_run_config_path_env_override(monkeypatch):
    from risk.yaml_risk import resolve_run_config_path

    monkeypatch.delenv("FOREX_CONFIG", raising=False)
    monkeypatch.setenv("FOREX_RUN_CONFIG", "config/run_deep.yaml")
    assert resolve_run_config_path("config/run.yaml") == Path("config/run_deep.yaml")
    # An explicit non-default path wins over the environment.
    assert resolve_run_config_path("custom.yaml") == Path("custom.yaml")
    monkeypatch.delenv("FOREX_RUN_CONFIG")
    assert resolve_run_config_path(None) == Path("config/run.yaml")


def test_risk_config_reads_live_risk_at_construction(monkeypatch):
    from risk import risk_engine

    monkeypatch.setitem(risk_engine._LR, "max_total_lots", 1.25)
    monkeypatch.setitem(risk_engine._LR, "daily_loss_limit", 0.015)
    cfg = risk_engine.RiskConfig()
    assert cfg.max_total_lots == 1.25
    assert cfg.max_daily_loss_pct == 0.015
    assert risk_engine.RiskConfig(max_total_lots=0.5).max_total_lots == 0.5


def test_session_enforcer_blocks_off_session_after_yaml():
    from risk.execution import SessionLimitsEnforcer
    from risk.yaml_risk import apply_yaml_risk

    lr = _live_risk_copy()
    apply_yaml_risk(YAML_RISK, lr)
    enf = SessionLimitsEnforcer(session_limits=lr["session_limits"])
    assert enf.check(open_lots=0.0, open_trades=0, session="off")["allowed"] is False
    london = enf.check(open_lots=0.0, open_trades=5, session="london")
    assert london["allowed"] is True and london["max_trades"] == 6


def test_sizer_uses_unknown_scale_for_missing_hurst(monkeypatch):
    from risk import execution

    monkeypatch.setitem(
        execution._LR, "regime_scale", {"crisis": 0.5, "trending": 1.0, "mean_rev": 0.7, "unknown": 0.3}
    )
    sizer = execution.RegimePositionSizer(base_kelly=0.25)
    assert sizer.unknown_scale == 0.3
    assert math.isclose(sizer._regime_scale(corr_avg=0.0, hurst=float("nan")), 0.3)
    assert math.isclose(sizer._regime_scale(corr_avg=0.0, hurst=None), 0.3)
    assert math.isclose(sizer._regime_scale(corr_avg=0.0, hurst=0.7), 1.0)
    out = sizer.size(10_000.0, 0.6, 1.5, [0.0] * 5, 0.0010, hurst=float("nan"))
    assert out["regime"] == "unknown"


def test_sizer_unknown_scale_falls_back_to_min_scale(monkeypatch):
    from risk import execution

    monkeypatch.setitem(execution._LR, "regime_scale", {"crisis": 0.5, "trending": 1.2, "mean_rev": 0.75})
    sizer = execution.RegimePositionSizer(base_kelly=0.25)
    assert sizer.unknown_scale == 0.5


def test_portfolio_var_limit_from_live_risk(monkeypatch):
    from risk import execution

    monkeypatch.setitem(execution._LR, "var_max_pct", 0.0125)
    monkeypatch.setitem(execution._LR, "var_confidence", 0.975)
    pv = execution.PortfolioVaR()
    assert pv.max_var == 0.0125
    assert pv.conf == 0.975
    assert execution.PortfolioVaR(max_var_pct=0.05).max_var == 0.05


def test_live_main_applies_yaml_risk_before_engine_and_preflight():
    src = (ROOT / "trading" / "live_engine.py").read_text(encoding="utf-8")
    main_src = src[src.index("args = p.parse_args()"):]
    i_apply = main_src.index("apply_run_config_risk(args.pairs_config)")
    assert i_apply < main_src.index("MultiPairLiveTradingEngine(")
    assert i_apply < main_src.index("_run_live_preflight(")
    assert i_apply < main_src.index("build_inference_agents(")


# ── M7: LiveSafetyConfig from LIVE_RISK ──────────────────────────────────────


def test_live_safety_config_from_live_risk():
    from trading.live_engine import LiveSafetyConfig

    cfg = LiveSafetyConfig.from_live_risk(max_spread_pips=1.7, live_risk={"daily_loss_limit": 0.03, "max_drawdown_halt": 0.10})
    assert cfg.max_spread_pips == 1.7
    assert cfg.max_daily_loss_pct == 0.03
    assert cfg.max_total_drawdown_pct == 0.10
    src = (ROOT / "trading" / "live_engine.py").read_text(encoding="utf-8")
    assert "LiveSafetyConfig.from_live_risk(max_spread_pips=max_spread_pips)" in src


# ── M1: RL agent kwargs mapping ──────────────────────────────────────────────


def test_filter_agent_kwargs_maps_and_drops_ppo_keys(capsys):
    from config.settings import RL
    from models.rl_agents import PPOAgent, filter_agent_kwargs

    kw = filter_agent_kwargs(PPOAgent, dict(RL["ppo"]))
    assert "n_steps" not in kw
    assert kw["clip"] == RL["ppo"]["clip_epsilon"]
    assert kw["entropy_coef"] == RL["ppo"]["entropy_coeff"]
    assert kw["value_coef"] == RL["ppo"]["value_coeff"]
    assert kw["lam"] == RL["ppo"]["gae_lambda"]
    assert "n_steps" in capsys.readouterr().out
    agent = PPOAgent(obs_size=8, n_actions=10, device="cpu", **kw)
    assert agent.clip == RL["ppo"]["clip_epsilon"]


def test_rl_runner_yaml_ppo_kwargs_construct_agent():
    from models.rl_agents import PPOAgent, filter_agent_kwargs
    from training.rl_runner import _agent_constructor_kwargs, _rl_algo_kwargs

    args = SimpleNamespace(rl_ppo_overrides={"gamma": 0.99, "lr": 3e-4, "n_steps": 2048})
    kw = filter_agent_kwargs(PPOAgent, _rl_algo_kwargs(args, "ppo"))
    agent = PPOAgent(obs_size=8, n_actions=10, device="cpu", use_lstm=True, hist_len=4, **kw)
    saved = _agent_constructor_kwargs(agent, kw)
    assert saved["use_lstm"] is True and saved["hist_len"] == 4 and "n_steps" not in saved
    json.dumps(saved)


def test_rl_inference_agent_kwargs_prefers_saved_meta():
    from inference.rl_inference import RLInferenceAgent
    from models.rl_agents import PPOAgent

    stub = SimpleNamespace(algo="ppo")
    kw = RLInferenceAgent._agent_kwargs(stub, PPOAgent, {"use_lstm": True, "hidden": 64, "n_steps": 2048})
    assert kw == {"use_lstm": True, "hidden": 64}
    fallback = RLInferenceAgent._agent_kwargs(stub, PPOAgent, None)
    assert "n_steps" not in fallback and "clip" in fallback
    PPOAgent(obs_size=8, n_actions=10, device="cpu", **fallback)


def test_dqn_agent_does_not_claim_lstm(capsys):
    from models.rl_agents import DQNAgent

    agent = DQNAgent(obs_size=8, n_actions=10, device="cpu", use_lstm=True, buf_size=10)
    assert agent.use_lstm is False
    assert "use_lstm=True ignored" in capsys.readouterr().out


# ── M2: best metadata / ONNX consistency ─────────────────────────────────────


def test_write_rl_best_meta_includes_agent_kwargs(tmp_path):
    from training.rl_runner import _write_rl_best_meta

    env = SimpleNamespace(obs_size=21, n_actions=10)
    args = SimpleNamespace(model="haelt", rl_encoder_obs=True)
    p = _write_rl_best_meta(
        tmp_path, "ppo", args, env, {"use_lstm": True}, {"sharpe": 0.8, "total_return_pct": 1.5, "n_trades": 12},
        source="episode_50",
    )
    meta = json.loads(p.read_text(encoding="utf-8"))
    assert p.name == "rl_ppo_best.json"
    assert meta["agent_kwargs"] == {"use_lstm": True}
    assert meta["val_sharpe"] == 0.8 and meta["obs_size"] == 21 and meta["saved_by"] == "episode_50"


def test_mid_training_best_writes_meta_and_onnx():
    src = (ROOT / "training" / "rl_runner.py").read_text(encoding="utf-8")
    cb = src[src.index("def _ep_val_callback"):src.index("returns = train_agent(")]
    assert '_save_rl_checkpoint(_agent, ckpt_dir, _algo, "best")' in cb
    assert "_write_rl_best_meta(" in cb
    assert "_export_rl_best_onnx(" in cb


def test_failed_onnx_export_removes_stale_graph(tmp_path, monkeypatch):
    import inference.onnx_inference as oi
    from training.rl_runner import _export_rl_best_onnx

    stale = tmp_path / "rl_dqn_best.onnx"
    stale.write_bytes(b"old")

    def _boom(**_kw):
        raise RuntimeError("export failed")

    monkeypatch.setattr(oi, "export_rl_to_onnx", _boom)
    out = _export_rl_best_onnx(tmp_path, "dqn", SimpleNamespace(model="haelt", seq_len=60), 10, 10)
    assert out is None
    assert not stale.exists()


# ── H2: deploy requires a checkpoint-bound promotion certificate ─────────────


def _deploy_fixture(tmp_path, monkeypatch):
    import training.rl_runner as rr

    ckpt = tmp_path / "rl_dqn_best.pt"
    ckpt.write_bytes(b"weights")
    onnx = tmp_path / "rl_dqn_best.onnx"
    onnx.write_bytes(b"graph")
    prod = tmp_path / "prod" / "production_best.onnx"
    monkeypatch.setattr(rr, "_production_onnx_paths", lambda _a: (prod, prod.with_name("production_prev.onnx")))
    args = SimpleNamespace(checkpoint_dir=str(tmp_path), seq_len=60, _n_features=4, _feat_names=["a", "b", "c", "d"])
    return rr, ckpt, onnx, prod, args


def test_deploy_refused_without_promotion_gate(tmp_path, monkeypatch):
    rr, ckpt, onnx, prod, args = _deploy_fixture(tmp_path, monkeypatch)
    res = rr._deploy_onnx_to_cpp_server(onnx, args, "rl_dqn", tmp_path, source_checkpoint=ckpt)
    assert res["status"] == "refused"
    assert not prod.exists()
    written = json.loads((tmp_path / "cpp_deployment.json").read_text(encoding="utf-8"))
    assert written["status"] == "refused"


def test_deploy_refused_when_certificate_covers_other_checkpoint(tmp_path, monkeypatch):
    from validation.gate_policy import certificate_from_gate

    rr, ckpt, onnx, prod, args = _deploy_fixture(tmp_path, monkeypatch)
    other = tmp_path / "haelt_best.pt"
    other.write_bytes(b"other")
    cert = certificate_from_gate({"promoted": True, "gates": {"sharpe": True}}, [other])
    (tmp_path / "promotion_gate.json").write_text(json.dumps(cert), encoding="utf-8")
    res = rr._deploy_onnx_to_cpp_server(onnx, args, "rl_dqn", tmp_path, source_checkpoint=ckpt)
    assert res["status"] == "refused"
    assert "does not cover" in res["error"]
    assert not prod.exists()


def test_deploy_refused_with_rejected_certificate(tmp_path, monkeypatch):
    from validation.gate_policy import certificate_from_gate

    rr, ckpt, onnx, prod, args = _deploy_fixture(tmp_path, monkeypatch)
    cert = certificate_from_gate({"promoted": False, "gates": {"sharpe": False}}, [ckpt])
    (tmp_path / "promotion_gate.json").write_text(json.dumps(cert), encoding="utf-8")
    res = rr._deploy_onnx_to_cpp_server(onnx, args, "rl_dqn", tmp_path, source_checkpoint=ckpt)
    assert res["status"] == "refused"
    assert not prod.exists()


def test_deploy_gate_passes_with_bound_certificate(tmp_path):
    from training.rl_runner import _deploy_promotion_check
    from validation.gate_policy import certificate_from_gate

    ckpt = tmp_path / "rl_dqn_best.pt"
    ckpt.write_bytes(b"weights")
    cert = certificate_from_gate({"promoted": True, "gates": {"sharpe": True}}, [ckpt])
    (tmp_path / "promotion_gate.json").write_text(json.dumps(cert), encoding="utf-8")
    ok, why = _deploy_promotion_check(tmp_path, ckpt)
    assert ok, why
    ckpt.write_bytes(b"retrained")  # certificate no longer matches the weights
    ok, _ = _deploy_promotion_check(tmp_path, ckpt)
    assert not ok


def test_run_deep_yaml_rl_deploy_disabled():
    import yaml

    cfg = yaml.safe_load((ROOT / "config" / "run_deep.yaml").read_text(encoding="utf-8"))
    assert cfg["rl"]["deploy"] is False
    assert float(cfg["rl"]["min_val_sharpe"]) >= 0.5


# ── Low: sidecar retention ───────────────────────────────────────────────────


def test_sidecar_prunes_old_logs_but_keeps_current_run(tmp_path):
    from monitoring.sidecar import prune_old_sidecar_logs

    old = tmp_path / "sidecar_old.jsonl"
    old_rot = tmp_path / "sidecar_old.log.1"
    cur = tmp_path / "sidecar_cur.jsonl"
    fresh = tmp_path / "sidecar_new.log"
    other = tmp_path / "unrelated.jsonl"
    for p in (old, old_rot, cur, fresh, other):
        p.write_text("x", encoding="utf-8")
    past = time.time() - 40 * 86400
    for p in (old, old_rot, cur, other):
        os.utime(p, (past, past))
    removed = prune_old_sidecar_logs(tmp_path, 30, keep_run="cur")
    assert set(removed) == {old, old_rot}
    assert cur.exists() and fresh.exists() and other.exists()
    assert prune_old_sidecar_logs(tmp_path, 0) == []
