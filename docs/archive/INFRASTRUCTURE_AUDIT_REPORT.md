# Infrastructure & Reliability Audit Report
**Date:** 2026-08-01  
**Project:** `/run/media/jamie/jamie/forex-main`  
**Analyst:** Antigravity Infrastructure & Reliability Audit  
**Areas Audited:** 6 — Leakage Guardrails, Sidecar Storage, Event Logging, Structured Readiness Reporting, Memory Auditability, Self-Improving Training  
**Total Issues Found:** 14

---

## Executive Summary

The infrastructure layer is partially built but has two **critical gaps** that make live deployment unsafe: there is no persistent audit trail for live trade decisions, and there is no pre-flight readiness check before the live engine starts trading. The self-improving feedback loop exists architecturally but is disconnected from live reality — the model learns from price data but never learns from its own execution mistakes. Several medium-severity gaps exist in sidecar storage and data auditability.

---

## Area 1 — Leakage Guardrails

### What Currently EXISTS ✅
- Temporal split enforcement via `_embargo_bars()` and `_purge_bars()` — `training/train_gpu.py:6538`
- Statistical correlation check against forward returns — `DatasetManifest.check_future_leak` at `training/train_gpu.py:5649`
- True walk-forward backtester — `scripts/backtest_true_walk_forward.py`
- Lockbox pattern for clean data isolation — `backtesting/improvements.py:437`

### What is BROKEN ⚠️

**INF-001 — Sequence Cropping is Silent (No Assertion)**  
| Severity | File | Line |
|----------|------|------|
| 🟡 Medium | `training/train_gpu.py` | 359 |

`_crop_to_seq_len()` silently truncates tensors with no log or assertion. If sequence windows mismatch the model's `seq_len`, data is lost invisibly.

### What is COMPLETELY MISSING 🚫

**INF-002 — No Mechanical Look-Ahead Block at Feature Engineering Level**  
| Severity | Area |
|----------|------|
| 🔴 High | Feature Engineering / Leakage |

The existing leakage check (`DatasetManifest.check_future_leak`) is a purely **statistical heuristic** (`|correlation| > 0.30`). This means:
- A subtle `shift(-1)` that leaks only a small amount of future information will pass undetected
- There are no AST-level or timestamp-index-level mechanical guards that structurally prevent future data access

There are also no strict fold isolation assertions verifying `test_time > train_time + embargo_bars` before the training loop begins.

**Recommended Fix:**
```python
# Add to feature materialization pipeline:
def assert_no_lookahead(df: pl.DataFrame, feature_cols: list[str]):
    """Mechanically verify no feature has a timestamp after its bar close."""
    assert df["timestamp_utc"].is_sorted(), "Data must be sorted before feature check"
    for col in feature_cols:
        # verify no NaN-filled future values snuck in via shift(-n)
        if df[col].shift(-1).corr(df["close"].shift(-1)) > 0.95:
            raise ValueError(f"Feature {col} shows strong future correlation — possible shift(-1) leak")
```

---

## Area 2 — Sidecar Storage

### What Currently EXISTS ✅
- Dataset metadata manifests — `data/dataset_manifest.py:12`
- Feature schema SHA-256 hashes — `data/feature_schema.py:28`
- Atomic JSON sidecars for checkpoints via `_safe_save`
- CatBoost/XGBoost feature importance JSONs — `training/train_catboost.py:521`

### What is BROKEN ⚠️

**INF-003 — Neural Model Feature Importance is Not Stored**  
| Severity | File | Line |
|----------|------|------|
| 🟡 Medium | `training/train_gpu.py` | ~10000 |

Feature importance extraction only runs for tree-based models (CatBoost, XGBoost). Deep sequence models (HAELT, Mamba) do not emit attention weight sidecars or gradient-based feature importance. You cannot inspect which features are driving HAELT predictions.

### What is COMPLETELY MISSING 🚫

**INF-004 — No Calibration Metadata Sidecar**  
| Severity | File | Line |
|----------|------|------|
| 🟡 Medium | `training/train_gpu.py` | 10046 |

The temperature scaler is fitted but no structured report is emitted. Missing:
- Expected Calibration Error (ECE)
- Brier Score
- Reliability diagram data
- Temperature parameter value

**INF-005 — No Raw Data Lineage Mapping**  
| Severity | Area |
|----------|------|
| 🟢 Low | Data Pipeline |

There is no record mapping raw source files → processed cache → model checkpoint. If a data file is updated or corrupted, there is no way to know which trained model was affected.

**Recommended Fix:**
```python
# Emit after calibration fitting:
calibration_report = {
    "temperature": float(scaler.temperature.item()),
    "ece": compute_ece(val_probs, val_labels),
    "brier_score": brier_score_loss(val_labels, val_probs),
    "calibration_set_rows": len(cal_dataset),
    "timestamp": datetime.utcnow().isoformat()
}
with open(checkpoint_dir / "calibration_report.json", "w") as f:
    json.dump(calibration_report, f, indent=2)
```

---

## Area 3 — Event Logging

### What Currently EXISTS ✅
- Standard Python `logging` and `loguru` throughout
- Discord semantic alerts — `monitoring/discord_alerts.py`
- C++ ONNX server JSONL shadow journaling — `cpp/src/onnx_runner.cpp:7`

### What is BROKEN ⚠️

**INF-006 — BrokerBridge Logs Fake Executions**  
| Severity | File | Line |
|----------|------|------|
| 🚨 Critical | `execution/broker_bridge.py` | 47 |

The stub raises `NotImplementedError` but still logs to standard text output as if execution succeeded. Fake execution logs mixed with real system logs makes the audit trail completely unreliable.

**INF-007 — OrderManager State is Memory-Only**  
| Severity | File | Line |
|----------|------|------|
| 🔴 High | `execution/order_manager.py` | 24 |

`OrderManager` tracks active orders in a Python dict in memory. Every state transition (placed → filled → closed) is lost on process restart. There is no persistent record of what orders were live at any point.

### What is COMPLETELY MISSING 🚫

**INF-008 — No Persistent Trade Decision Audit Trail**  
| Severity | Area |
|----------|------|
| 🚨 Critical | Live Execution |

There is no structured, persistent JSONL log capturing the full chain:
```
Signal generated → Order placed → Fill received / Rejection
```
Without this, it is impossible to:
- Debug why a live trade happened
- Reconstruct the sequence of events after a loss
- Audit compliance or regulatory requirements
- Feed live trading mistakes back into retraining

**Recommended Fix — `execution/execution_logger.py`:**
```python
import structlog
import json
from pathlib import Path
from datetime import datetime, timezone

log = structlog.get_logger()

class ExecutionLogger:
    def __init__(self, log_dir: str = "logs/execution"):
        self.path = Path(log_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.audit_file = self.path / f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"

    def log_signal(self, pair, direction, confidence, features_hash):
        self._write({"event": "SIGNAL", "pair": pair, "direction": direction,
                     "confidence": confidence, "features_hash": features_hash})

    def log_order(self, order_id, pair, side, size, price, sl, tp):
        self._write({"event": "ORDER_PLACED", "order_id": order_id, "pair": pair,
                     "side": side, "size": size, "price": price, "sl": sl, "tp": tp})

    def log_fill(self, order_id, fill_price, slippage_pips):
        self._write({"event": "ORDER_FILLED", "order_id": order_id,
                     "fill_price": fill_price, "slippage_pips": slippage_pips})

    def log_rejection(self, order_id, reason):
        self._write({"event": "ORDER_REJECTED", "order_id": order_id, "reason": reason})

    def _write(self, record: dict):
        record["ts"] = datetime.now(timezone.utc).isoformat()
        with open(self.audit_file, "a") as f:
            f.write(json.dumps(record) + "\n")
```

---

## Area 4 — Structured Readiness Reporting

### What Currently EXISTS ✅
- Data readiness gate logging NaN rates and drop reasons — `data/pair_readiness.py:59`
- Feature readiness schema hash check before training — `data/feature_readiness.py:105`
- Model promotion gate — `validation/promotion_gate.py`

### What is BROKEN ⚠️

**INF-009 — All Readiness Checks are Training-Phase Only**  
| Severity | Area |
|----------|------|
| 🔴 High | Live Trading |

Existing readiness checks validate the *training* pipeline. The *live trading* pipeline has no equivalent checks. The system can start trading with a stale data feed, a disconnected broker, or an unwarmed ONNX model.

### What is COMPLETELY MISSING 🚫

**INF-010 — No Live Trading Pre-Flight Check**  
| Severity | Area |
|----------|------|
| 🚨 Critical | Live Execution |

There is no `preflight_check.py` or equivalent. Before `live_engine.py` begins trading, it should verify:

| Check | Currently Verified? |
|-------|-------------------|
| Data feed connection live | ❌ No |
| Last tick timestamp < 5 seconds ago | ❌ No |
| Broker connection authenticated | ❌ No |
| Open positions reconciled with broker | ❌ No |
| Risk limits loaded and valid | ❌ No |
| ONNX model loaded and warmed up | ❌ No |
| Feature schema matches live data | ✅ Yes |
| Model calibration loaded | ❌ No |

**Recommended Fix — `trading/preflight_check.py`:**
```python
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json

@dataclass
class ReadinessReport:
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    feed_connected: bool = False
    feed_latency_ms: float = float('inf')
    broker_connected: bool = False
    positions_reconciled: bool = False
    risk_limits_loaded: bool = False
    model_warmed_up: bool = False
    calibration_loaded: bool = False
    schema_matched: bool = False
    errors: list = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return all([self.feed_connected, self.broker_connected,
                    self.model_warmed_up, self.risk_limits_loaded,
                    self.schema_matched])

    def save(self, path="logs/live_readiness_report.json"):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

def run_preflight(config) -> ReadinessReport:
    report = ReadinessReport()
    # ... run each check, populate report fields ...
    report.save()
    if not report.is_ready:
        raise RuntimeError(f"Pre-flight failed: {report.errors}")
    return report
```

---

## Area 5 — Memory Auditability

### What Currently EXISTS ✅
- Feature schema SHA-256 versioning — `data/feature_schema.py:18`
- Models crash if live feature schema mismatches trained schema (strong safety feature ✅)
- Dataset manifests validate config parameters against caches — `data/dataset_manifest.py:122`

### What is BROKEN ⚠️

**INF-011 — Raw Data Payload is Not Fingerprinted**  
| Severity | File | Line |
|----------|------|------|
| 🟡 Medium | `data/dataset_manifest.py` | 122 |

Column schema is hashed but the actual data content is not. You can verify that the *structure* matches, but you cannot verify that the model was trained on the correct *time window* or that the source data hasn't been silently modified.

**Recommended Fix:**
```python
import hashlib, polars as pl

def fingerprint_dataset(df: pl.DataFrame) -> str:
    """Hash the timestamp index + shape to detect data substitution."""
    index_bytes = df["timestamp_utc"].cast(pl.Utf8).to_frame().write_csv().encode()
    shape_str = f"{df.shape[0]}x{df.shape[1]}".encode()
    return hashlib.sha256(index_bytes + shape_str).hexdigest()[:16]

# Store in _manifest.json:
manifest["dataset_fingerprint"] = fingerprint_dataset(df)
manifest["row_count"] = len(df)
manifest["date_range"] = [str(df["timestamp_utc"].min()), str(df["timestamp_utc"].max())]
```

---

## Area 6 — Self-Improving Training

### What Currently EXISTS ✅
- Retrain orchestrator with drift and demotion triggers — `retraining/orchestrator.py:313`
- Subprocess retrain dispatch from live engine — `trading/live_engine.py:1215`
- Hard example miner wiring for supervised training — in training pipeline

### What is BROKEN ⚠️

**INF-012 — Feedback Loop Ignores Live Execution Reality**  
| Severity | File | Line |
|----------|------|------|
| 🔴 High | `trading/live_engine.py` | 1215 |

`live_engine.py` triggers retraining when new price data arrives. It does **not** feed back:
- Which signals resulted in stop-loss hits
- Fill slippage measurements
- Order rejections
- Drawdown episodes

The model retrains on the same type of data it was already trained on, learning nothing from its own failures.

### What is COMPLETELY MISSING 🚫

**INF-013 — No Live Error Hard Example Mining**  
| Severity | Area |
|----------|------|
| 🔴 High | Self-Improving Training |

The hard example miner is wired for offline supervised training only. Live trading mistakes are never injected as prioritised training samples.

**INF-014 — No Live → Training Feedback Database**  
| Severity | Area |
|----------|------|
| 🔴 High | Self-Improving Training |

There is no database or store connecting live trade outcomes back to the training pipeline.

**Recommended Fix — Live Feedback Pipeline:**
```python
# In execution_logger.py — add trade outcome logging
def log_trade_close(self, order_id, entry_price, exit_price, pnl_pips,
                    hit_sl: bool, slippage_pips: float):
    record = {
        "event": "TRADE_CLOSED",
        "order_id": order_id,
        "pnl_pips": pnl_pips,
        "hit_sl": hit_sl,
        "slippage_pips": slippage_pips,
        "is_hard_example": hit_sl or slippage_pips > 2.0
    }
    self._write(record)
    if record["is_hard_example"]:
        self._flag_for_retraining(order_id)

# In retraining/orchestrator.py — add live error ingestion
def ingest_live_errors(self, audit_jsonl_path: str):
    """Load hard examples from live trading and oversample them in next retrain."""
    hard_examples = []
    with open(audit_jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            if record.get("is_hard_example"):
                hard_examples.append(record["order_id"])
    self.hard_example_ids = hard_examples  # used by curriculum in next epoch
```

---

## Summary Table

| ID | Severity | Area | Issue |
|----|----------|------|-------|
| INF-001 | 🟡 Medium | Leakage | Silent sequence crop — no assertion |
| INF-002 | 🔴 High | Leakage | No mechanical AST/timestamp look-ahead block |
| INF-003 | 🟡 Medium | Sidecar | Neural model attention weights not stored |
| INF-004 | 🟡 Medium | Sidecar | No calibration ECE/Brier sidecar emitted |
| INF-005 | 🟢 Low | Sidecar | No raw data lineage mapping |
| INF-006 | 🚨 Critical | Logging | BrokerBridge logs fake executions |
| INF-007 | 🔴 High | Logging | OrderManager state is memory-only (lost on restart) |
| INF-008 | 🚨 Critical | Logging | No persistent Signal→Order→Fill audit trail |
| INF-009 | 🔴 High | Readiness | All readiness checks are training-phase only |
| INF-010 | 🚨 Critical | Readiness | No live trading pre-flight check exists |
| INF-011 | 🟡 Medium | Auditability | Raw data payload not fingerprinted |
| INF-012 | 🔴 High | Self-Improve | Retraining ignores live execution outcomes |
| INF-013 | 🔴 High | Self-Improve | No live error hard example mining |
| INF-014 | 🔴 High | Self-Improve | No live→training feedback database |

---

## Recommended Build Order

| Priority | Item | Effort | Unblocks |
|----------|------|--------|---------|
| 1 | INF-008: `ExecutionLogger` — persistent JSONL audit trail | Low | All live debugging |
| 2 | INF-010: `preflight_check.py` — live trading pre-flight | Medium | Safe live start |
| 3 | INF-007: Persist OrderManager state to disk | Low | Restart recovery |
| 4 | INF-002: Mechanical timestamp look-ahead block | Medium | True leakage safety |
| 5 | INF-014: Live feedback database | Medium | Self-improvement loop |
| 6 | INF-012/013: Live error hard example mining | Medium | Model learns from mistakes |
| 7 | INF-004: Calibration sidecar report | Low | Model transparency |
| 8 | INF-003: Neural attention weight sidecars | Medium | Interpretability |
| 9 | INF-011: Dataset fingerprinting | Low | Data integrity |
| 10 | INF-005: Raw data lineage mapping | Low | Audit compliance |

---

---

# Appendix — Cross-Report Master Issue Index

This section consolidates all findings from all four audit reports into a single prioritised master list.

| ID | Severity | Report | Area | Issue | File |
|----|----------|--------|------|-------|------|
| SYS-001 | 🚨 Critical | Systems | Backtesting | Sharpe inflated by closed-trade-only equity | `backtesting/backtest.py:441` |
| INF-006 | 🚨 Critical | Infrastructure | Logging | BrokerBridge logs fake executions | `execution/broker_bridge.py:47` |
| INF-008 | 🚨 Critical | Infrastructure | Logging | No persistent Signal→Order→Fill audit trail | `execution/` |
| INF-010 | 🚨 Critical | Infrastructure | Readiness | No live trading pre-flight check | `trading/` |
| PIPE-001 | 🚨 Critical | Pipeline | Execution | Broker bridge is an unimplemented stub | `execution/broker_bridge.py:47` |
| PIPE-002 | 🚨 Critical | Pipeline | Risk | Hardcoded pip value — USDJPY sized 1000× too large | `features/feature_engineering_pl.py:260` |
| DS-001 | 🚨 Critical | Dataset | Labels | Exit barriers use mid-price (zero exit spread) | `labeling/triple_barrier_labeling.py:76` |
| DS-002 | 🚨 Critical | Dataset | Leakage | EMA look-ahead bias at split boundaries | `features/feature_engineering_pl.py:464` |
| INF-002 | 🔴 High | Infrastructure | Leakage | No mechanical look-ahead block at feature level | `features/` |
| INF-007 | 🔴 High | Infrastructure | Logging | OrderManager state is memory-only (lost on restart) | `execution/order_manager.py:24` |
| INF-009 | 🔴 High | Infrastructure | Readiness | All readiness checks are training-phase only | `data/` |
| INF-012 | 🔴 High | Infrastructure | Self-Improve | Retraining ignores live execution outcomes | `trading/live_engine.py:1215` |
| INF-013 | 🔴 High | Infrastructure | Self-Improve | No live error hard example mining | `retraining/` |
| INF-014 | 🔴 High | Infrastructure | Self-Improve | No live→training feedback database | `retraining/` |
| PIPE-003 | 🔴 High | Pipeline | NLP | News embeddings return all zeros (dataset unused) | `features/feature_engineering_pl.py:273` |
| PIPE-004 | 🔴 High | Pipeline | Data | News parquet eagerly loaded into RAM | `data/historical_news.py:142` |
| PIPE-005 | 🔴 High | Pipeline | Risk | Session windows ignore Daylight Saving Time | `config/settings.py:748` |
| SYS-002 | 🔴 High | Systems | Auto-Tune | Val set reused for hyperparameter decisions | `training/train_gpu.py:12585` |
| SYS-003 | 🔴 High | Systems | Calibration | Temperature scaling uses same val as early stop | `models/architectures.py:1047` |
| SYS-004 | 🔴 High | Systems | Config | Preflight skips all risk parameter validation | `training/config_validate.py:228` |
| DS-003 | 🔴 High | Dataset | Schema | Timezone-naive timestamps cause silent misalignment | `data/databento_loader.py:41` |
| DS-004 | 🔴 High | Dataset | NLP | Bag-of-words sentiment fallback inverts signals | `data/historical_news.py:168` |
| DS-005 | 🔴 High | Dataset | Memory | Feature engineering OOM on full dataset | `features/feature_engineering_pl.py` |
| INF-001 | 🟡 Medium | Infrastructure | Leakage | Silent sequence crop — no assertion or log | `training/train_gpu.py:359` |
| INF-003 | 🟡 Medium | Infrastructure | Sidecar | Neural model attention weights not stored | `training/train_gpu.py:~10000` |
| INF-004 | 🟡 Medium | Infrastructure | Sidecar | No calibration ECE/Brier sidecar emitted | `training/train_gpu.py:10046` |
| INF-011 | 🟡 Medium | Infrastructure | Auditability | Raw data payload not fingerprinted | `data/dataset_manifest.py:122` |
| PIPE-006 | 🟡 Medium | Pipeline | Model | Extreme clipping destroys news-event signals | `models/architectures.py:758` |
| PIPE-007 | 🟡 Medium | Pipeline | Config | API keys hardcoded as empty strings | `config/settings.py` |
| PIPE-008 | 🟡 Medium | Pipeline | Features | Candlestick thresholds are untunable magic numbers | `features/feature_engineering_pl.py:539` |
| PIPE-009 | 🟡 Medium | Pipeline | Backtest | SL/TP ffill bleeds values between unrelated trades | `backtesting/backtest.py:124` |
| SYS-005 | 🟡 Medium | Systems | Curriculum | Val Sharpe gates curriculum progression | `training/curriculum_controller.py:131` |
| SYS-006 | 🟡 Medium | Systems | News | Economic surprise has 1-bar look-ahead | `data/economic_calendar.py:275` |
| SYS-007 | 🟡 Medium | Systems | Curriculum | .bfill() leaks future spread into difficulty scores | `training/train_gpu.py:3760` |
| DS-006 | 🟡 Medium | Dataset | Quality | Holiday filter only covers Dec 25 and Jan 1 | `data/data_ingestion.py:250` |
| DS-007 | 🟡 Medium | Dataset | Features | "HMM" regime feature is just a volatility bucket | `features/feature_engineering_pl.py:156` |
| DS-008 | 🟡 Medium | Dataset | Memory | Deduplication OOM on large news CSV | `scripts/download_historical_news.py:665` |
| PIPE-010 | 🟢 Low | Pipeline | Data | Intermediate merge output is uncompressed CSV | `scripts/merge_datasets.py:66` |
| PIPE-011 | 🟢 Low | Pipeline | Training | Silent sequence crop hides data loss | `training/train_gpu.py:359` |
| SYS-008 | 🟢 Low | Systems | Training | NaN batch mid-accumulation leaves partial grads | `training/train_gpu.py:7940` |
| DS-009 | 🟢 Low | Dataset | Features | Simulated OBI ignores real L2 Databento data | `features/feature_engineering_pl.py:96` |
| DS-010 | 🟢 Low | Dataset | Memory | Float64/Float32 precision ping-pong wastes memory | `labeling/rl_reward_labeling.py:100` |
| INF-005 | 🟢 Low | Infrastructure | Sidecar | No raw data lineage mapping | `data/` |

---

## Totals by Severity

| Severity | Count |
|----------|-------|
| 🚨 Critical | 8 |
| 🔴 High | 15 |
| 🟡 Medium | 14 |
| 🟢 Low | 6 |
| **Total** | **43** |

## Totals by Report

| Report | Issues |
|--------|--------|
| `PIPELINE_IMPROVEMENT_REPORT.md` | 11 |
| `DATASET_IMPROVEMENT_REPORT.md` | 10 |
| `SYSTEMS_AUDIT_REPORT.md` | 8 |
| `INFRASTRUCTURE_AUDIT_REPORT.md` | 14 |
| `CODEBASE_HEALTH_AUDIT.md` | 18 new (see below) |
| **Grand Total** | **61** |

---

## New Issues from `CODEBASE_HEALTH_AUDIT.md` (2026-08-01, second pass)

| ID | Severity | Area | Issue | File |
|----|----------|------|-------|------|
| BUG-001 | 🚨 Critical | Live Trading | OANDA net position `abs(short_u)` fragile | `trading/live_engine.py:676` |
| BUG-002 | 🚨 Critical | Live Trading | LiveSafetyGate never resets → permanent halt | `trading/live_engine.py:482` |
| BUG-003 | 🚨 Critical | Live Trading | Stale equity on broker failure → wrong sizing | `trading/live_engine.py:978` |
| BUG-004 | 🔴 High | Live Trading | Position reversal doesn't close existing position | `trading/live_engine.py:1109` |
| BUG-005 | 🔴 High | Risk | `DrawdownAwareExitManager.new_day()` never called | `risk/execution.py:211` |
| BUG-006 | 🔴 High | API/Risk | Kelly division by zero when `win_loss_ratio=0` | `sizing/kelly_criterion.py:18` |
| BUG-007 | 🔴 High | Backtesting | GPU backtester look-ahead bias (missing signal shift) | `backtesting/gpu_backtester.py:39` |
| BUG-008 | 🔴 High | RL Training | Reward normalizer divides by std without subtracting mean | `models/rl_agents.py:634` |
| BUG-010 | 🟡 Medium | Live Trading | Drift detection uses random labels (concept drift undetectable) | `trading/live_engine.py:1199` |
| BUG-011 | 🟡 Medium | Risk | VaR zero-pads short return histories for new pairs | `risk/execution.py:232` |
| BUG-012 | 🟡 Medium | RL Training | Replay buffer O(N) sampling bottleneck (1M elements/step) | `models/rl_agents.py:499` |
| WIRE-001 | 🚨 Critical | Config | `label_quality` 5 features referenced but never computed | `config/run.yaml`, `config/feature_mask.py` |
| WIRE-002 | 🟡 Medium | Config | `dim_feedforward` config key silently ignored | `config/models.py:56` |
| WIRE-004 | 🟡 Medium | Config | `data_cache` path Windows-only on Linux system | `config/run.yaml:374` |
| WIRE-009 | 🟢 Low | Tests | `test_dashboard.py` referenced but missing | `run_e2e_tests.py` |
| DEAD-001 | 🟢 Low | Dead Code | 19 dead root-level scripts | root `*.py` |
| DEAD-002 | 🟡 Medium | Dead Code | `sizing/` package disconnected from live pipeline | `sizing/` |
| DEAD-003 | 🟡 Medium | Dead Code | UTF-8 BOM in 5 files (potential SyntaxError) | `models/xgboost_model.py` + 4 others |
