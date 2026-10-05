"""Stale-cache detection, window feature cache, and feature_store config wiring.

All data is tiny and synthetic; nothing under data/processed is touched.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

import feature_store.fingerprint as fpmod
from data import feature_cache as fc
from training import cache_integrity as ci

# ── helpers ────────────────────────────────────────────────────────────────


def _args(**kw):
    base = dict(
        pair="EURUSD",
        pairs=None,
        seq_len=60,
        bar_freq="5min",
        lookahead_bars=30,
        label_method="cpar",
        historical_news_mode="off",
        auto_rebuild_on_mismatch=True,
        ignore_manifest=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def fake_code(monkeypatch):
    state = {"fp": "code-v1"}
    monkeypatch.setattr(fpmod, "feature_code_fingerprint", lambda: state["fp"])
    return state


@pytest.fixture
def cot_file(tmp_path, monkeypatch):
    p = tmp_path / "cot.parquet"
    p.write_bytes(b"cot-v1")
    monkeypatch.setattr(ci, "_COT_PARQUET_PATH", p)
    monkeypatch.setattr(fc, "_COT_PARQUET", p)
    return p


def _bump_mtime(p):
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))


def _write_manifest(cache_path, args):
    manifest = {"n_features": 3, **ci._manifest_fingerprint_fields(args)}
    (cache_path.parent / (cache_path.name + "_manifest.json")).write_text(json.dumps(manifest))


def _bars(n=400, seed=0, start=datetime(2024, 1, 2, tzinfo=UTC)):
    rng = np.random.default_rng(seed)
    ts = [start + timedelta(minutes=5 * i) for i in range(n)]
    close = 1.085 + np.cumsum(rng.normal(0, 2e-4, n))
    return pl.DataFrame(
        {
            "timestamp_utc": ts,
            "open": close - 5e-5,
            "high": close + 2e-4,
            "low": close - 2e-4,
            "close": close,
            "volume": rng.integers(50, 500, n).astype(float),
            "bid_close": close - 5e-5,
            "ask_close": close + 5e-5,
            "spread_avg": np.full(n, 1e-4),
        }
    ).with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))


# ── code fingerprint ───────────────────────────────────────────────────────


def test_code_fingerprint_tracks_source_edits_not_line_endings(tmp_path):
    (tmp_path / "features").mkdir()
    src = tmp_path / "features" / "mod.py"
    src.write_bytes(b"x = 1\n")
    fp1 = fpmod.code_fingerprint(["features/*.py"], tmp_path)
    src.write_bytes(b"x = 1\r\n")
    _bump_mtime(src)
    assert fpmod.code_fingerprint(["features/*.py"], tmp_path) == fp1
    src.write_bytes(b"x = 2\n")
    _bump_mtime(src)
    assert fpmod.code_fingerprint(["features/*.py"], tmp_path) != fp1


# ── dataset manifest ───────────────────────────────────────────────────────


def test_manifest_matches_when_nothing_changed(tmp_path, fake_code, cot_file):
    cache = tmp_path / "ds.zarr"
    _write_manifest(cache, _args())
    ok, msg = ci._validate_cache_integrity(str(cache), _args())
    assert "content hash" not in msg and "content_hash" not in msg, msg


def test_manifest_mismatch_on_feature_code_change(tmp_path, fake_code, cot_file):
    cache = tmp_path / "ds.zarr"
    _write_manifest(cache, _args())
    fake_code["fp"] = "code-v2"
    ok, msg = ci._validate_cache_integrity(str(cache), _args())
    assert not ok
    assert "content hash mismatch" in msg and "source code" in msg


def test_manifest_mismatch_on_cot_mtime_change(tmp_path, fake_code, cot_file):
    cache = tmp_path / "ds.zarr"
    _write_manifest(cache, _args())
    _bump_mtime(cot_file)
    ok, msg = ci._validate_cache_integrity(str(cache), _args())
    assert not ok
    assert "raw side inputs" in msg


def test_manifest_without_content_hash_is_stale_when_auto_rebuild(tmp_path, fake_code, cot_file):
    cache = tmp_path / "ds.zarr"
    (tmp_path / "ds.zarr_manifest.json").write_text(json.dumps({"n_features": 3}))
    ok, msg = ci._validate_cache_integrity(str(cache), _args(auto_rebuild_on_mismatch=True))
    assert not ok and "no content_hash" in msg
    ok2, msg2 = ci._validate_cache_integrity(str(cache), _args(auto_rebuild_on_mismatch=False))
    assert "no content_hash" not in msg2


def test_pinned_fingerprint_survives_mid_build_edit(tmp_path, fake_code, cot_file):
    args = _args()
    pinned = ci._pin_build_fingerprint(args)
    fake_code["fp"] = "edited-mid-build"
    assert ci._manifest_fingerprint_fields(args)["content_hash"] == pinned["content_hash"]


# ── resume state ───────────────────────────────────────────────────────────


def test_resume_state_roundtrip_and_stale_is_deleted(tmp_path):
    cache = tmp_path / "ds.zarr"
    ci._write_resume_state(cache, 7, "hashA")
    assert ci._read_resume_idx(cache, "hashA") == 7
    assert ci._read_resume_idx(cache, "hashB") == -1
    assert not ci._resume_path(cache).exists()


def test_legacy_resume_without_hash_is_ignored(tmp_path):
    cache = tmp_path / "ds.zarr"
    ci._resume_path(cache).write_text(json.dumps({"last_completed_window_idx": 1314}))
    assert ci._read_resume_idx(cache, "hashA") == -1
    assert not ci._resume_path(cache).exists()


# ── feature cache config ───────────────────────────────────────────────────


def test_feature_cache_requires_both_switches():
    root = {"feature_cache": {"enabled": True}, "data": {"use_feature_cache": False}}
    assert fc.resolve_feature_cache_config(yaml_root=root)["enabled"] is False
    root["data"]["use_feature_cache"] = True
    cfg = fc.resolve_feature_cache_config(yaml_root=root)
    assert cfg["enabled"] is True
    args = SimpleNamespace(use_feature_cache=False, feature_cache=None, feature_cache_dir=None, config=None)
    assert fc.resolve_feature_cache_config(args, yaml_root=root)["enabled"] is False


def test_feature_cache_reads_yaml_from_args_config(tmp_path, monkeypatch):
    monkeypatch.delenv("FOREX_CONFIG", raising=False)
    monkeypatch.delenv("FOREX_RUN_CONFIG", raising=False)
    y = tmp_path / "run.yaml"
    y.write_text(
        "feature_cache:\n  enabled: true\n  slow_cols: [hurst]\n"
        f"data:\n  use_feature_cache: true\n  feature_cache_dir: {tmp_path.as_posix()}/fc\n"
    )
    cfg = fc.resolve_feature_cache_config(SimpleNamespace(config=str(y)))
    assert cfg["enabled"] and cfg["slow_cols"] == ["hurst_exponent"]
    assert cfg["cache_dir"].endswith("/fc")


# ── window feature cache ───────────────────────────────────────────────────


class _CountingFE:
    """Deterministic stand-in for FeatureEngineer with public settings."""

    def __init__(self, window=5):
        self.window = window
        self._calls = 0

    def build(self, bars, **kw):
        self._calls += 1
        c = pl.col("close")
        return bars.select(
            "timestamp_utc",
            c.rolling_mean(self.window).fill_null(0.0).alias("sma"),
            c.pct_change().fill_null(0.0).alias("sentiment_decayed"),
        )


def _get(cache, fe, bars, kwargs=None):
    kwargs = kwargs if kwargs is not None else {"sentiment": None, "pair": "EURUSD"}
    return cache.get_or_build(
        lambda: fe.build(bars, **kwargs),
        pair="EURUSD",
        bars=bars,
        fe=fe,
        fe_kwargs=kwargs,
        win_start=None,
        bar_freq="5min",
        seq_len=60,
    )


def test_window_cache_hit_on_identical_inputs(tmp_path, fake_code, cot_file):
    cache = fc.WindowFeatureCache(tmp_path / "fc", build_version="t")
    fe, bars = _CountingFE(), _bars(120)
    a = _get(cache, fe, bars)
    b = _get(cache, fe, bars)
    assert fe._calls == 1 and cache.stats["hits"] == 1
    assert a.equals(b)
    assert list((tmp_path / "fc" / "windows" / "EURUSD").rglob("*.parquet"))
    assert not list((tmp_path / "fc").rglob("*.tmp"))


def test_window_cache_miss_on_code_change(tmp_path, fake_code, cot_file):
    cache = fc.WindowFeatureCache(tmp_path / "fc", build_version="t")
    fe, bars = _CountingFE(), _bars(120)
    _get(cache, fe, bars)
    fake_code["fp"] = "code-v2"
    _get(cache, fe, bars)
    assert fe._calls == 2 and cache.stats["hits"] == 0


def test_window_cache_miss_on_cot_mtime_change(tmp_path, fake_code, cot_file):
    cache = fc.WindowFeatureCache(tmp_path / "fc", build_version="t")
    fe, bars = _CountingFE(), _bars(120)
    _get(cache, fe, bars)
    _bump_mtime(cot_file)
    _get(cache, fe, bars)
    assert fe._calls == 2


def test_window_cache_miss_on_features_window_change(tmp_path, fake_code, cot_file, monkeypatch):
    cache = fc.WindowFeatureCache(tmp_path / "fc", build_version="t")
    bars = _bars(120)
    fe = _CountingFE(window=5)
    _get(cache, fe, bars)
    fe.window = 7
    _get(cache, fe, bars)
    assert fe._calls == 2
    # features.* YAML windows (effective_feature_scales) are part of the key too.
    monkeypatch.setattr(fpmod, "effective_feature_scales", lambda *a, **k: {"atr_windows": [6, 99]})
    _get(cache, fe, bars)
    assert fe._calls == 3


def test_window_cache_miss_on_side_input_change(tmp_path, fake_code, cot_file):
    cache = fc.WindowFeatureCache(tmp_path / "fc", build_version="t")
    fe, bars = _CountingFE(), _bars(120)
    s1 = pl.DataFrame({"timestamp_utc": bars["timestamp_utc"], "sentiment": np.zeros(120)})
    s2 = s1.with_columns(pl.lit(0.5).alias("sentiment"))
    _get(cache, fe, bars, {"sentiment": s1, "pair": "EURUSD"})
    _get(cache, fe, bars, {"sentiment": s2, "pair": "EURUSD"})
    assert fe._calls == 2


def test_window_cache_falls_back_on_undigestable_input(tmp_path, fake_code, cot_file):
    cache = fc.WindowFeatureCache(tmp_path / "fc", build_version="t")
    fe, bars = _CountingFE(), _bars(120)
    out = _get(cache, fe, bars, {"weird": object()})
    assert fe._calls == 1 and cache.stats["errors"] == 1 and out.height == 120


def test_corrupt_cache_entry_is_recomputed(tmp_path, fake_code, cot_file):
    cache = fc.WindowFeatureCache(tmp_path / "fc", build_version="t")
    fe, bars = _CountingFE(), _bars(120)
    _get(cache, fe, bars)
    (entry,) = list((tmp_path / "fc" / "windows").rglob("*.parquet"))
    entry.write_bytes(b"garbage")
    out = _get(cache, fe, bars)
    assert fe._calls == 2 and out.height == 120


def test_slow_cols_reused_when_only_unrelated_inputs_change(tmp_path, fake_code, cot_file):
    cache = fc.WindowFeatureCache(tmp_path / "fc", slow_cols=["sentiment_decayed"], build_version="t")
    fe, bars = _CountingFE(), _bars(120)
    ca1 = pl.DataFrame({"timestamp_utc": bars["timestamp_utc"], "dxy": np.zeros(120)})
    ca2 = ca1.with_columns(pl.lit(1.0).alias("dxy"))
    first = _get(cache, fe, bars, {"sentiment": None, "news_events": None, "cross_asset": ca1, "pair": "EURUSD"})
    assert cache.stats["slow_writes"] == 1
    second = _get(cache, fe, bars, {"sentiment": None, "news_events": None, "cross_asset": ca2, "pair": "EURUSD"})
    assert fe._calls == 2 and cache.stats["slow_hits"] == 1
    assert first["sentiment_decayed"].equals(second["sentiment_decayed"])


def test_cache_on_equals_cache_off_with_real_feature_engineer(tmp_path, fake_code, cot_file):
    from features.feature_engineering import FeatureEngineer

    bars = _bars(300, seed=3)
    off = FeatureEngineer(atr_window=6, lag_windows=[5, 20, 60]).build(bars)

    cache = fc.WindowFeatureCache(tmp_path / "fc", slow_cols=["hurst_exponent", "sentiment_decayed"], build_version="t")
    fe = FeatureEngineer(atr_window=6, lag_windows=[5, 20, 60])
    kwargs = {"pair": "EURUSD"}
    miss = cache.get_or_build(
        lambda: fe.build(bars, **kwargs), pair="EURUSD", bars=bars, fe=fe, fe_kwargs=kwargs,
        win_start=None, bar_freq="5min", seq_len=60,
    )
    fe2 = FeatureEngineer(atr_window=6, lag_windows=[5, 20, 60])
    hit = cache.get_or_build(
        lambda: fe2.build(bars, **kwargs), pair="EURUSD", bars=bars, fe=fe2, fe_kwargs=kwargs,
        win_start=None, bar_freq="5min", seq_len=60,
    )
    assert cache.stats["errors"] == 0, cache.stats
    assert cache.stats["hits"] == 1
    for got in (miss, hit):
        assert got.columns == off.columns
        assert got.schema == off.schema
        assert got.equals(off, null_equal=True)


def test_configure_window_cache_disabled_is_none(tmp_path):
    from training import dataset_builder as db

    db._configure_feature_window_cache(cfg={"enabled": False})
    assert db._FEATURE_WINDOW_CACHE is None


# ── feature store ──────────────────────────────────────────────────────────


def _mat_df(start, end):
    return pl.DataFrame({"timestamp_utc": [start, end], "close": [1.1, 1.1005]})


def test_feature_store_lookup_misses_on_code_version_change(tmp_path):
    from data.feature_definitions import MaterializationStrategy
    from feature_store.polars_store import FeatureStore

    start, end = datetime(2024, 1, 1, 8, tzinfo=UTC), datetime(2024, 1, 1, 9, tzinfo=UTC)
    s1 = FeatureStore(root=tmp_path / "fs", code_version="v1")
    s1._store_materialization("close", _mat_df(start, end), start, end, MaterializationStrategy.EAGER_BATCH)
    assert s1.is_materialized("close", start, end)
    s2 = FeatureStore(root=tmp_path / "fs", code_version="v2")
    assert not s2.is_materialized("close", start, end)
    assert s2.get_materialized_ranges("close") == []
    assert FeatureStore(root=tmp_path / "fs", code_version="v1").is_materialized("close", start, end)


def test_feature_store_config_keys_honored(tmp_path):
    import pyarrow.parquet as pq

    from data.feature_definitions import MaterializationStrategy
    from feature_store.config import FeatureStoreSettings, open_feature_store

    settings = FeatureStoreSettings.from_dict(
        {
            "enabled": True,
            "path": str(tmp_path / "fs"),
            "registry_db": "reg.sqlite",
            "data_root": "parq",
            "compression": "snappy",
            "default_strategy": "incremental",
            "incremental_lookback_bars": 42,
            "job_queue": {"enabled": True, "max_workers": 3, "poll_interval_sec": 5},
        }
    )
    assert settings.root == str(tmp_path / "fs")
    assert (settings.job_queue_max_workers, settings.job_queue_poll_interval_sec) == (3, 5.0)
    store = open_feature_store(settings)
    assert (tmp_path / "fs" / "reg.sqlite").exists()
    assert store.default_strategy is MaterializationStrategy.INCREMENTAL
    assert store.incremental_lookback_bars == 42

    start, end = datetime(2024, 1, 1, 8, tzinfo=UTC), datetime(2024, 1, 1, 9, tzinfo=UTC)
    store._store_materialization("close", _mat_df(start, end), start, end, MaterializationStrategy.EAGER_BATCH)
    files = list((tmp_path / "fs" / "parq").rglob("*.parquet"))
    assert files, "materialization must land under data_root"
    assert pq.ParquetFile(files[0]).metadata.row_group(0).column(0).compression == "SNAPPY"


def test_feature_store_root_wins_over_path_alias():
    from feature_store.config import FeatureStoreSettings

    s = FeatureStoreSettings.from_dict({"root": "a", "path": "b"})
    assert s.root == "a"


def test_job_queue_processes_pending_jobs(tmp_path):
    from feature_store.polars_store import FeatureStore

    bars = _bars(200)
    store = FeatureStore(root=tmp_path / "fs", code_version="v1")
    store.set_bars(bars)
    start = bars["timestamp_utc"][0]
    end = bars["timestamp_utc"][-1]
    store.enqueue_materialization("atr_6", start, end)
    assert store.get_pending_jobs()
    out = store.process_pending_jobs(max_workers=2)
    assert out["done"] == 1, out
    assert not store.get_pending_jobs()


def _yaml(tmp_path, enabled=True, auto=True):
    y = tmp_path / "run.yaml"
    y.write_text(
        "feature_store:\n"
        f"  enabled: {str(enabled).lower()}\n  auto_materialize: {str(auto).lower()}\n"
        f"  root: {(tmp_path / 'fs').as_posix()}\n"
        "drift_detection:\n  baseline_window_days: 1\n  live_window_days: 1\n"
        "  monitored_features: [close, atr_6, ret_5]\n"
    )
    return y


def test_auto_materialize_hook_materializes_after_build(tmp_path, monkeypatch, capsys):
    import feature_store.config as fsc
    from feature_store.config import auto_materialize_after_build
    from feature_store.polars_store import FeatureStore

    fsc._WARNED.clear()
    monkeypatch.delenv("FOREX_CONFIG", raising=False)
    monkeypatch.delenv("FOREX_RUN_CONFIG", raising=False)
    bars = _bars(600, start=datetime(2024, 1, 1, tzinfo=UTC))
    calls = []

    def loader(pair, args, start, end):
        calls.append((pair, start, end))
        return bars

    args = SimpleNamespace(config=str(_yaml(tmp_path)), pairs=["USDJPY", "EURUSD"], data_end="2024-01-03", data_source="dukascopy")
    out = auto_materialize_after_build(args, "ignored", bars_loader=loader)
    assert calls and calls[0][0] == "USDJPY"
    assert out is not None and "atr_6" in out
    assert "close" not in out  # raw price level excluded from monitored features
    err = capsys.readouterr().out
    assert "price-level" in err and "ret_5" in err
    store = FeatureStore(root=tmp_path / "fs")
    assert store.get_materialized_ranges("atr_6")


@pytest.mark.parametrize("enabled,auto", [(False, True), (True, False)])
def test_auto_materialize_hook_noop_when_disabled(tmp_path, monkeypatch, enabled, auto):
    from feature_store.config import auto_materialize_after_build

    monkeypatch.delenv("FOREX_CONFIG", raising=False)
    monkeypatch.delenv("FOREX_RUN_CONFIG", raising=False)
    called = []
    args = SimpleNamespace(config=str(_yaml(tmp_path, enabled, auto)), pair="EURUSD", data_end="2024-01-03")
    assert auto_materialize_after_build(args, "x", bars_loader=lambda *a: called.append(1)) is None
    assert not called
    assert not (tmp_path / "fs").exists()


def test_auto_materialize_hook_never_raises(tmp_path, monkeypatch):
    from feature_store.config import auto_materialize_after_build

    monkeypatch.delenv("FOREX_CONFIG", raising=False)

    def boom(*a):
        raise RuntimeError("no data")

    args = SimpleNamespace(config=str(_yaml(tmp_path)), pair="EURUSD", data_end="2024-01-03")
    assert auto_materialize_after_build(args, "x", bars_loader=boom) is None


# ── retraining pipeline: drift thresholds from config ──────────────────────


def test_retraining_config_reads_all_drift_and_store_keys(tmp_path, capsys):
    import feature_store.config as fsc
    from retraining.pipeline import load_config_from_yaml

    fsc._WARNED.clear()
    y = tmp_path / "run.yaml"
    y.write_text(
        "feature_store:\n  path: custom/fs\n  compression: lz4\n  registry_db: r.db\n  data_root: d\n"
        "  default_strategy: incremental\n  incremental_lookback_bars: 7\n  enabled: true\n"
        "  job_queue: {enabled: true, max_workers: 4, poll_interval_sec: 9}\n"
        "drift_detection:\n  enabled: false\n  psi_threshold: 0.33\n  psi_bins: 17\n"
        "  ks_pvalue_threshold: 0.01\n  ks_stat_threshold: 0.2\n  monitored_features: [close, atr_6]\n"
        "retraining:\n  drift_feature_frac: 0.5\n  min_interval_days: 3\n  schedule_interval_days: 11\n"
        "  promote_on_complete: false\n  model_root: models/x\n"
    )
    cfg = load_config_from_yaml(str(y))
    assert cfg.feature_store_root == "custom/fs"
    assert (cfg.feature_store_compression, cfg.feature_store_registry_db, cfg.feature_store_data_root) == ("lz4", "r.db", "d")
    assert cfg.feature_store_default_strategy == "incremental" and cfg.feature_store_incremental_lookback_bars == 7
    assert (cfg.feature_store_job_queue_enabled, cfg.feature_store_job_queue_max_workers) == (True, 4)
    assert cfg.drift_enabled is False
    assert (cfg.drift_psi_threshold, cfg.drift_psi_bins) == (0.33, 17)
    assert (cfg.drift_ks_pvalue_threshold, cfg.drift_ks_stat_threshold) == (0.01, 0.2)
    assert cfg.drift_monitored_features == ["atr_6"]
    assert "price-level" in capsys.readouterr().out
    assert (cfg.retrain_drift_feature_frac, cfg.retrain_min_interval_days, cfg.retrain_schedule_interval_days) == (0.5, 3, 11)
    assert cfg.retrain_promote_on_complete is False and cfg.retrain_model_root == "models/x"


def test_full_pipeline_uses_configured_thresholds(tmp_path, monkeypatch):
    from pathlib import Path

    import retraining.pipeline as rp

    cfg = rp.PipelineConfig(
        feature_store_root=str(tmp_path / "fs"),
        drift_psi_threshold=0.4,
        drift_psi_bins=13,
        drift_ks_pvalue_threshold=0.02,
        drift_ks_stat_threshold=0.3,
        drift_monitored_features=["close", "atr_6"],
        retrain_model_root=str(tmp_path / "models"),
        retrain_drift_feature_frac=0.6,
        log_every_step=False,
    )
    pipe = rp.FullPipeline(cfg)
    oc = pipe.orchestrator.config
    assert oc.psi_threshold == 0.4 and oc.drift_feature_frac == 0.6
    assert Path(oc.model_root) == tmp_path / "models"

    seen = {}

    def fake_check(store, feature_names, **kw):
        seen["names"] = feature_names
        seen.update(kw)
        return rp.DriftReport([], 0.0, 1.0, 0.0, 0, 0, rp.DriftSeverity.NONE, False, "", "")

    monkeypatch.setattr(rp, "schedule_drift_check", fake_check)
    pipe.check_drift(as_of=datetime(2024, 1, 10, tzinfo=UTC))
    assert seen["names"] == ["atr_6"]
    assert (seen["psi_bins"], seen["psi_threshold"]) == (13, 0.4)
    assert (seen["ks_pvalue_threshold"], seen["ks_stat_threshold"]) == (0.02, 0.3)


def test_drift_threshold_reapplied_before_persist(tmp_path):
    import retraining.pipeline as rp
    from feature_store.polars_store import FeatureStore

    store = FeatureStore(root=tmp_path / "fs", code_version="v1")
    from data.feature_definitions import MaterializationStrategy

    as_of = datetime(2024, 3, 1, tzinfo=UTC)
    rng = np.random.default_rng(0)

    def put(start, end, loc):
        n = int((end - start).total_seconds() // 3600) + 1
        ts = [start + timedelta(hours=i) for i in range(n)]
        df = pl.DataFrame({"timestamp_utc": ts, "atr_6": rng.normal(loc, 1.0, n)})
        store._store_materialization("atr_6", df, start, end, MaterializationStrategy.EAGER_BATCH)

    # baseline window = [as_of-38d, as_of-8d], live window = [as_of-7d, as_of]
    put(as_of - timedelta(days=39), as_of - timedelta(days=8), 0.0)
    put(as_of - timedelta(days=7), as_of, 0.6)
    strict = rp.schedule_drift_check(
        store, ["atr_6"], 30, 7, as_of=as_of, psi_threshold=0.01, ks_pvalue_threshold=0.05, ks_stat_threshold=0.05
    )
    lax = rp.schedule_drift_check(
        store, ["atr_6"], 30, 7, as_of=as_of, psi_threshold=100.0, ks_pvalue_threshold=1e-300, ks_stat_threshold=1.0
    )
    assert strict.drift_detected and not lax.drift_detected


def test_sync_maps_feature_cache_and_store_keys(tmp_path):
    import argparse

    from training.gpu_cli import _apply_yaml_config

    y = tmp_path / "run.yaml"
    y.write_text(
        "data:\n  use_feature_cache: true\n  feature_cache_dir: data/features\n"
        "feature_store:\n  enabled: true\n  root: data/feature_store\n"
    )
    p = argparse.ArgumentParser()
    _apply_yaml_config(p, str(y))
    ns = p.parse_args([])
    assert ns.use_feature_cache is True and ns.feature_cache_dir == "data/features"
    assert ns.feature_store["root"] == "data/feature_store"


def test_unimplemented_keys_warn_once(capsys):
    fc._WARNED.discard("ofi_z_threshold")
    fc._WARNED.discard("fc_regime_window")
    fc.warn_unimplemented_feature_cache_keys({"ofi_z_threshold": 2.0, "regime_window": 60})
    fc.warn_unimplemented_feature_cache_keys({"ofi_z_threshold": 2.0, "regime_window": 60})
    out = capsys.readouterr().out
    assert out.count("ofi_z_threshold") == 1 and out.count("regime_window is accepted") == 1
