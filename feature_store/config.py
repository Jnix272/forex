"""run.yaml ``feature_store:`` settings and the post-dataset-build materialization hook.

Every key in the section is consumed here:

- enabled                   master switch; false = no store is opened by the hooks
- root (alias: path)        store directory
- registry_db / data_root   SQLite registry and parquet directory (relative to root)
- compression               parquet codec for materialized features
- default_strategy          MaterializationStrategy used when a caller passes none
- incremental_lookback_bars overlap used by FeatureStore.needs_incremental_update
- auto_materialize          materialize monitored features after a dataset build
- job_queue.enabled         enqueue jobs + process them via the queue instead of direct calls
- job_queue.max_workers     threads used to process queued jobs
- job_queue.poll_interval_sec  sleep between polls in FeatureStore.serve_job_queue
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

_KNOWN_KEYS = {
    "enabled",
    "root",
    "path",
    "registry_db",
    "data_root",
    "compression",
    "default_strategy",
    "incremental_lookback_bars",
    "auto_materialize",
    "job_queue",
}
_KNOWN_JOB_KEYS = {"enabled", "max_workers", "poll_interval_sec"}
_WARNED: set[str] = set()

# Raw price levels always drift over a multi-month window; never monitor them.
PRICE_LEVEL_COLUMNS = frozenset(
    {"close", "open", "high", "low", "bid", "ask", "mid", "mid_close", "bid_close", "ask_close", "mid_open"}
)
DEFAULT_MONITORED_FEATURES = ("ret_5", "ret_20", "atr_6", "atr_20", "vol_20", "ofi")


def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"[FeatureStore] WARN: {msg}", flush=True)


@dataclass
class FeatureStoreSettings:
    enabled: bool = False
    root: str = "data/feature_store"
    registry_db: str = "registry.db"
    data_root: str = "features"
    compression: str = "zstd"
    default_strategy: str = "eager_batch"
    incremental_lookback_bars: int = 100
    auto_materialize: bool = False
    job_queue_enabled: bool = False
    job_queue_max_workers: int = 1
    job_queue_poll_interval_sec: float = 60.0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, section: dict | None) -> FeatureStoreSettings:
        sec = dict(section or {})
        unknown = sorted(set(sec) - _KNOWN_KEYS)
        if unknown:
            _warn_once("unknown:" + ",".join(unknown), f"unknown feature_store keys ignored: {unknown}")
        root, path = sec.get("root"), sec.get("path")
        if root and path and str(root) != str(path):
            _warn_once("root_path", f"feature_store.root={root!r} and legacy path={path!r} differ; using root")
        jq = sec.get("job_queue") or {}
        if isinstance(jq, dict):
            unknown_jq = sorted(set(jq) - _KNOWN_JOB_KEYS)
            if unknown_jq:
                _warn_once("unknown_jq", f"unknown feature_store.job_queue keys ignored: {unknown_jq}")
        else:
            jq = {}
        return cls(
            enabled=bool(sec.get("enabled", False)),
            root=str(root or path or "data/feature_store"),
            registry_db=str(sec.get("registry_db") or "registry.db"),
            data_root=str(sec.get("data_root") or "features"),
            compression=str(sec.get("compression") or "zstd"),
            default_strategy=str(sec.get("default_strategy") or "eager_batch"),
            incremental_lookback_bars=int(sec.get("incremental_lookback_bars", 100) or 100),
            auto_materialize=bool(sec.get("auto_materialize", False)),
            job_queue_enabled=bool(jq.get("enabled", False)),
            job_queue_max_workers=int(jq.get("max_workers", 1) or 1),
            job_queue_poll_interval_sec=float(jq.get("poll_interval_sec", 60) or 60),
            extra={k: sec[k] for k in unknown},
        )


def _yaml_root(config_path: str | None = None) -> dict:
    cfg_path = config_path or os.environ.get("FOREX_CONFIG") or os.environ.get("FOREX_RUN_CONFIG")
    if not cfg_path or not os.path.exists(cfg_path):
        return {}
    try:
        import yaml

        with open(cfg_path, encoding="utf-8-sig") as f:
            root = yaml.safe_load(f) or {}
        return root if isinstance(root, dict) else {}
    except Exception:
        return {}


def load_feature_store_settings(args=None, config_path: str | None = None, yaml_root: dict | None = None) -> FeatureStoreSettings:
    """YAML ``feature_store:`` (FOREX_CONFIG or ``config_path``), overridden by ``args.feature_store``."""
    root = yaml_root if yaml_root is not None else _yaml_root(config_path or getattr(args, "config", None))
    sec = dict(root.get("feature_store") or {})
    a_fs = getattr(args, "feature_store", None) if args is not None else None
    if isinstance(a_fs, dict):
        sec.update(a_fs)
    return FeatureStoreSettings.from_dict(sec)


def open_feature_store(settings: FeatureStoreSettings, **kwargs):
    from feature_store.polars_store import FeatureStore

    return FeatureStore(
        root=settings.root,
        registry_db=settings.registry_db,
        data_root=settings.data_root,
        compression=settings.compression,
        default_strategy=settings.default_strategy,
        incremental_lookback_bars=settings.incremental_lookback_bars,
        **kwargs,
    )


def monitored_features(configured, *, source: str = "drift_detection.monitored_features") -> list[str]:
    """Configured monitored features minus raw price levels (warned once)."""
    names = [str(c) for c in (configured or DEFAULT_MONITORED_FEATURES)]
    dropped = [c for c in names if c.lower() in PRICE_LEVEL_COLUMNS]
    if dropped:
        _warn_once(
            f"price_levels:{source}",
            f"{source} lists raw price-level columns {dropped}; they always drift and are excluded.",
        )
    return [c for c in names if c.lower() not in PRICE_LEVEL_COLUMNS]


def materialize_with_settings(store, settings: FeatureStoreSettings, feature_names, bars, start, end) -> dict:
    """Materialize via the job queue when job_queue.enabled, else directly."""
    from data.feature_materializers import materialize_feature_set

    if not settings.job_queue_enabled:
        return materialize_feature_set(store, list(feature_names), bars, start, end)
    store.set_bars(bars)
    for name in feature_names:
        store.enqueue_materialization(name, start, end)
    return store.process_pending_jobs(max_workers=settings.job_queue_max_workers)


def _load_recent_bars(pair: str, args, start: datetime, end: datetime):
    from data.data_ingestion import ForexDataPipeline
    from data.sources import ForexDataManager

    ticks = ForexDataManager(verbose=False).load(
        pair=pair,
        source=getattr(args, "data_source", "dukascopy"),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        session_only=not getattr(args, "full_day_data", False),
    )
    if ticks is None or len(ticks) == 0:
        return None
    return ForexDataPipeline(
        bar_freq=str(getattr(args, "bar_freq", "5min") or "5min"),
        session_filter=False,
        apply_frac_diff=False,
        session_mode="dst",
        add_session_label=True,
        spread_cap_multiplier=3.0,
    ).run(ticks, pair=pair)


def auto_materialize_after_build(args, cache_path: str | None = None, *, bars_loader=None) -> dict | None:
    """After a successful dataset build, materialize the drift-monitored features for
    the primary pair over the drift baseline+live window ending at ``data_end``.

    No-op unless feature_store.enabled and feature_store.auto_materialize. Never
    raises: failures are reported and the build result stands.
    """
    settings = load_feature_store_settings(args)
    if not (settings.enabled and settings.auto_materialize):
        return None
    if str(getattr(args, "data_source", "")) == "synthetic" and bars_loader is None:
        return None
    try:
        root = _yaml_root(getattr(args, "config", None))
        drift = root.get("drift_detection") or {}
        names = monitored_features(drift.get("monitored_features"))
        days = int(drift.get("baseline_window_days", 90)) + int(drift.get("live_window_days", 7)) + 1
        end_s = getattr(args, "data_end", None)
        end = datetime.fromisoformat(str(end_s)).replace(tzinfo=UTC) if end_s else datetime.now(UTC)
        start = end - timedelta(days=days)
        pairs = getattr(args, "pairs", None) or [getattr(args, "pair", "EURUSD")]
        if isinstance(pairs, str):
            pairs = [p.strip() for p in pairs.split(",") if p.strip()]
        pair = str(pairs[0]).upper()
        bars = (bars_loader or _load_recent_bars)(pair, args, start, end)
        if bars is None or len(bars) == 0:
            print(f"[FeatureStore] auto-materialize: no bars for {pair} {start:%Y-%m-%d}..{end:%Y-%m-%d}")
            return None
        import polars as pl

        if not isinstance(bars, pl.DataFrame):
            bars = pl.from_pandas(bars.reset_index() if "timestamp_utc" not in bars.columns else bars)
        store = open_feature_store(settings)
        unregistered = [n for n in names if store.get_feature(n) is None]
        if unregistered:
            _warn_once(
                "unregistered:" + ",".join(unregistered),
                f"monitored features not in the feature-store registry are skipped: {unregistered}",
            )
        names = [n for n in names if n not in unregistered]
        if not names:
            return None
        out = materialize_with_settings(store, settings, names, bars, start, end)
        print(
            f"[FeatureStore] auto-materialized {len(names)} feature(s) for {pair} "
            f"{start:%Y-%m-%d}..{end:%Y-%m-%d} -> {store.root} (code {store.code_version})"
        )
        return out if isinstance(out, dict) else {"result": out}
    except Exception as e:
        print(f"[FeatureStore] WARN: auto-materialize failed ({e}); dataset build unaffected")
        return None
