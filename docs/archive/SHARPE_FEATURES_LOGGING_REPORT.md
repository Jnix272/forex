# Sharpe Math, Feature Pipeline & Logging Audit Report
**Date:** 2026-08-01  
**Project:** `/run/media/jamie/jamie/forex-main`  
**Analyst:** Antigravity Training & Feature Pipeline Audit  
**Total Issues Found:** 13

---

## Executive Summary

Three interconnected problems are identified. The Sharpe ratio is being calculated incorrectly in **both** the training loss function and the backtester — through different bugs but with the same effect of inflating values. The feature pipeline has a confirmed critical wiring bug where FinBERT embeddings are silently discarded and replaced with zeros regardless of what is passed in. And logging relies almost entirely on bare `print()` statements, with exceptions swallowed without stack traces — making the entire system nearly impossible to debug in production.

---

# AREA 1 — Sharpe Math in Training

## SH-001 — `SharpeProxyLoss` Applies Square Root Twice (Double-Sqrt Bug)

| Attribute | Detail |
|-----------|--------|
| **Files** | `training/train_gpu.py:430`, `models/architectures.py:209` |
| **Severity** | 🔴 High |

**Description:**  
`config/run.yaml` supplies `sharpe_annualization_factor: 325.0`. The value `325` is already a pre-computed square root of the annual bar count (`√(252 × 420) ≈ 325`). However, inside the loss function, the code applies the square root **again**:

```python
# Current — wrong: applies sqrt a second time
self._ann_sqrt = float(ann) ** 0.5  # → √325 ≈ 18
```

This means the actual annualisation factor used in the loss is `~18` instead of the intended `325`. The Sharpe gradient is nearly **18× too weak**, so the Huber baseline dominates training and the Sharpe loss provides almost no signal.

**Recommended Fix:**  
Pick one convention and stick to it:
```python
# Option A — pass raw period count in config (e.g. 252*420 = 105840), apply sqrt once:
self._ann_sqrt = float(ann) ** 0.5  # sqrt(105840) ≈ 325 ✅

# Option B — pass pre-computed sqrt in config (325.0), don't apply sqrt again:
self._ann_sqrt = float(ann)  # use directly ✅
```

---

## SH-002 — `SharpeProxyLoss` Uses Biased Standard Deviation (ddof=0)

| Attribute | Detail |
|-----------|--------|
| **File** | `training/train_gpu.py:443` |
| **Severity** | 🟡 Medium |

**Description:**  
```python
std = returns.std(unbiased=False)  # ddof=0 — biased estimator
```
Using `unbiased=False` on small training batches (e.g. 128 samples) gives a standard deviation that is systematically too small. This artificially inflates the Sharpe ratio during training, causing the model to think its signals are cleaner than they actually are.

**Recommended Fix:**
```python
std = returns.std(unbiased=True)  # ddof=1 — correct sample std ✅
```

---

## SH-003 — Backtest Sharpe Uses Closed-Trade-Only Equity (Confirmed)

| Attribute | Detail |
|-----------|--------|
| **File** | `backtesting/backtest.py:404, 442` |
| **Severity** | 🚨 Critical |

**Description:**  
```python
# backtesting/backtest.py:442
returns = self.results_df["equity"].pct_change()
```
The `equity` column is only updated when a trade **closes**. All bars during open holding periods show flat equity (0% return). This deflates the return standard deviation toward zero and massively inflates Sharpe.

This was already identified as **SYS-001** — confirmed here from a second independent code path analysis.

**Recommended Fix:**
```python
returns = self.results_df["total_value"].pct_change()  # mark-to-market every bar ✅
sharpe = (returns.mean() - rf_per_bar) / returns.std(ddof=1) * ann_sqrt
```

---

## SH-004 — Risk-Free Rate Never Subtracted

| Attribute | Detail |
|-----------|--------|
| **Files** | `training/train_gpu.py`, `backtesting/backtest.py` |
| **Severity** | 🟢 Low |

**Description:**  
All Sharpe calculations in the codebase compute raw `mean / std` with no risk-free rate deduction. With current rates at ~5%, this overstates Sharpe by a small but non-trivial amount on strategies with low annualised return.

**Recommended Fix:**
```python
ANNUAL_RF = 0.05  # load from config
rf_per_bar = ANNUAL_RF / bars_per_year
excess_returns = returns - rf_per_bar
sharpe = excess_returns.mean() / excess_returns.std(ddof=1) * ann_sqrt
```

---

## SH-005 — Cross-Validation Sharpe Averaged Across Folds (Mathematically Incorrect)

| Attribute | Detail |
|-----------|--------|
| **File** | `validation/purged_cv.py:92` |
| **Severity** | 🟢 Low |

**Description:**  
```python
regime_consistency = np.mean(val_sharpes)  # average of per-fold Sharpes
```
Averaging Sharpe ratios across folds breaks mathematical consistency. Because Sharpe is non-linear (ratio of mean to std), a fold with high volatility and good returns weighs equally to a low-volatility fold with mediocre returns. The correct approach is to concatenate all fold predictions into one series and compute a single overall Sharpe.

**Recommended Fix:**
```python
# Concatenate all fold returns first, then compute one Sharpe
all_returns = np.concatenate([fold["returns"] for fold in fold_results])
overall_sharpe = all_returns.mean() / all_returns.std(ddof=1) * ann_sqrt
```

---

# AREA 2 — Feature Pipeline Correctness

## FP-001 — FinBERT Embeddings Hardwired to Zero (Confirmed Critical)

| Attribute | Detail |
|-----------|--------|
| **File** | `features/feature_engineering_pl.py:1334, 1458, 1611` |
| **Severity** | 🚨 Critical |

**Description:**  
This is the definitive root cause of **PIPE-003**. The `build()` method accepts a `finbert_embs` parameter, but the implementation **unconditionally overwrites it with zeros** regardless of what was passed in:

```python
# features/feature_engineering_pl.py — current (broken):
finbert_embs = ...       # parameter passed in (line 1334, assigned but discarded)

# Line 1458 — ignores finbert_embs entirely, always creates zero columns:
F = F.with_columns([
    pl.lit(0.0).alias("fb_0"),
    pl.lit(0.0).alias("fb_1"),
    ...
    pl.lit(0.0).alias("fb_7"),
])
```

The variable is assigned at line 1334 (which is why vulture flagged it as "assigned but never used"), and the actual embedding join at line 1458 replaces it with literal zeros. The model has **never received a real news embedding in any training run**.

**Recommended Fix:**
```python
# If finbert_embs is provided, join it to the feature frame on timestamp:
if finbert_embs is not None and len(finbert_embs) > 0:
    emb_df = pl.DataFrame(finbert_embs)  # columns: timestamp_utc, fb_0..fb_7
    F = F.join_asof(emb_df.sort("timestamp_utc"), on="timestamp_utc", strategy="backward").with_columns(
        [pl.col(f"fb_{i}").fill_null(0.0) for i in range(8)]
    )
else:
    # Fallback to zeros only when no embeddings provided
    F = F.with_columns([pl.lit(0.0).alias(f"fb_{i}") for i in range(8)])
```

---

## FP-002 — `pre_news` Flag Leaks Future Information for Unscheduled Events

| Attribute | Detail |
|-----------|--------|
| **File** | `features/feature_engineering_pl.py:1515–1518` |
| **Severity** | 🔴 High |

**Description:**  
The pipeline computes a `pre_news` window flag: bars within a configurable buffer period **before** a news event are marked as `pre_news=1`. This is valid only for **scheduled calendar events** (e.g. NFP at 08:30 every first Friday) where the timing is publicly known in advance.

However, if `news_events` is populated with any **unscheduled breaking news** (e.g. flash crashes, surprise central bank statements), knowing you are in a "pre-event window" before an unscheduled event is a direct future leak — the model is told something is about to happen before it happens.

**Recommended Fix:**
```python
# Tag each event with its type at source
scheduled_events = [e for e in news_events if e["type"] == "calendar"]
unscheduled_events = [e for e in news_events if e["type"] == "breaking"]

# Only apply pre_news window to scheduled events
mask_pre = compute_pre_window(scheduled_events)

# For unscheduled events, only flag post_news (after the fact)
mask_post_unscheduled = compute_post_window(unscheduled_events)
```

---

## FP-003 — Column Name Mismatch: `time_sin` vs `hour_sin`

| Attribute | Detail |
|-----------|--------|
| **File** | `features/feature_engineering_pl.py:333, 1622` |
| **Severity** | 🟡 Medium |

**Description:**  
The feature builder produces columns named `time_sin`, `time_cos`, `day_sin`, `day_cos`. However, the ONNX export and some config references still use `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`. Any downstream code expecting `hour_sin` will silently receive NaN or raise a `ColumnNotFoundError`.

This was already identified as a known issue (fixed in `onnx_compatible_features()`) but the fix is not consistent across all consumers.

**Recommended Fix:**  
Establish one canonical name in `config/settings.py` and use it everywhere:
```python
TEMPORAL_FEATURE_NAMES = {
    "hour_sin": "time_sin",  # maps old names → current names
    "hour_cos": "time_cos",
    "dow_sin": "day_sin",
    "dow_cos": "day_cos",
}
```

---

## FP-004 — Features Not Explicitly Cast to Float32

| Attribute | Detail |
|-----------|--------|
| **File** | `features/feature_engineering_pl.py:1657` |
| **Severity** | 🟢 Low |

**Description:**  
`sanitize_frame()` handles NaN/Inf bounds but does not explicitly cast all numeric columns to `Float32`. Polars may keep intermediate calculations in `Float64`, leading to silent dtype mismatch when PyTorch loads the data (which expects `float32` tensors).

**Recommended Fix:**
```python
# At the end of sanitize_frame():
float_cols = [c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64)]
df = df.with_columns([pl.col(c).cast(pl.Float32) for c in float_cols])
```

---

# AREA 3 — Pipeline Logging

## LOG-001 — Exceptions Swallowed Without Stack Traces

| Attribute | Detail |
|-----------|--------|
| **File** | `training/train_gpu.py:281, 1605, 3056, 4388` (and many more) |
| **Severity** | 🔴 High |

**Description:**  
Throughout `train_gpu.py`, exceptions are caught and printed as bare strings:
```python
# Current — stack trace lost forever:
except Exception as exc:
    print(f"[Training] step failed: {exc}")
```
The full stack trace (file, line number, call chain) is discarded. In a 14,000-line training file, this makes it nearly impossible to diagnose production failures.

**Recommended Fix:**
```python
import logging, traceback
logger = logging.getLogger(__name__)

except Exception as exc:
    logger.exception(f"[Training] step failed: {exc}")
    # OR:
    logger.error(f"[Training] step failed: {exc}\n{traceback.format_exc()}")
```

---

## LOG-002 — Critical Events Logged via `print()` Not Structured Logger

| Attribute | Detail |
|-----------|--------|
| **Files** | `training/train_gpu.py`, `monitoring/discord_alerts.py`, `monitoring/demotion_monitor.py` |
| **Severity** | 🟡 Medium |

**Description:**  
Operational events that should be routed to monitoring systems are emitted as bare `print()` calls:
```python
print(f"[Preflight] FATAL: model not loaded")
print(f"[DemotionMonitor] ⬇️ DEMOTION TRIGGERED for {model_id}")
```
`print()` output cannot be filtered by level, routed to log files, or consumed by log aggregators (Splunk, Datadog, Grafana Loki).

**Recommended Fix:**  
Replace all operational `print()` with structured `logging` calls:
```python
import logging

logger = logging.getLogger("forex.training")

logger.critical("[Preflight] FATAL: model not loaded")
logger.warning("[DemotionMonitor] DEMOTION TRIGGERED for %s", model_id)
```
Configure a `RotatingFileHandler` and a `StreamHandler` at startup in `main.py`.

---

## LOG-003 — API Has Zero Internal Logging

| Attribute | Detail |
|-----------|--------|
| **File** | `api/main.py` |
| **Severity** | 🔴 High |

**Description:**  
The FastAPI application has no server-side logging. When a position sizing request fails or receives an anomalous payload, the API raises an HTTP 400 error but leaves no trace on the server. There is no way to investigate why API calls are failing without reproducing the issue.

**Recommended Fix:**
```python
import logging

logger = logging.getLogger("forex.api")


@app.post("/position-size")
async def position_size(payload: PositionPayload):
    logger.info("position_size called: pair=%s risk_pct=%.4f", payload.pair, payload.risk_pct)
    try:
        result = compute_position(payload)
        logger.info("position_size result: size=%.2f", result.size)
        return result
    except Exception as e:
        logger.exception("position_size failed for payload %s", payload)
        raise HTTPException(status_code=400, detail=str(e))
```

---

## LOG-004 — Log Files Have No Rotation Policy

| Attribute | Detail |
|-----------|--------|
| **File** | `monitoring/demotion_monitor.py:363`, `training/train_gpu.py` |
| **Severity** | 🟡 Medium |

**Description:**  
Demotion alert JSONs (`logs/demotions/demotion_*.json`) are written indefinitely with no cleanup. Training JSONL logs have no rotation. On long-running deployments, this will silently fill available disk space and can crash the process when a write fails.

**Recommended Fix:**
```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    "logs/training.log",
    maxBytes=50 * 1024 * 1024,  # 50 MB per file
    backupCount=5,  # keep 5 rotated files
)
```
Add a cron or startup cleanup for old demotion JSON files:
```bash
find logs/demotions/ -name "*.json" -mtime +30 -delete
```

---

## Summary Table

| ID | Severity | Area | Issue | File |
|----|----------|------|-------|------|
| SH-001 | 🔴 High | Sharpe Math | Double-sqrt annualisation in SharpeProxyLoss | `training/train_gpu.py:430` |
| SH-002 | 🟡 Medium | Sharpe Math | Biased std (ddof=0) in SharpeProxyLoss | `training/train_gpu.py:443` |
| SH-003 | 🚨 Critical | Sharpe Math | Backtest uses closed-trade-only equity | `backtesting/backtest.py:442` |
| SH-004 | 🟢 Low | Sharpe Math | Risk-free rate never subtracted anywhere | Multiple files |
| SH-005 | 🟢 Low | Sharpe Math | CV Sharpe averaged across folds (wrong) | `validation/purged_cv.py:92` |
| FP-001 | 🚨 Critical | Features | FinBERT embeddings hardwired to zero | `features/feature_engineering_pl.py:1458` |
| FP-002 | 🔴 High | Features | pre_news flag leaks future data for unscheduled events | `features/feature_engineering_pl.py:1515` |
| FP-003 | 🟡 Medium | Features | Column name mismatch time_sin vs hour_sin | `features/feature_engineering_pl.py:333` |
| FP-004 | 🟢 Low | Features | Numeric cols not cast to Float32 | `features/feature_engineering_pl.py:1657` |
| LOG-001 | 🔴 High | Logging | Exceptions swallowed without stack traces | `training/train_gpu.py:281+` |
| LOG-002 | 🟡 Medium | Logging | Critical events use print() not structured logger | Multiple files |
| LOG-003 | 🔴 High | Logging | API has zero internal logging | `api/main.py` |
| LOG-004 | 🟡 Medium | Logging | No log rotation — disk exhaustion risk | `monitoring/demotion_monitor.py:363` |

---

## Recommended Fix Order

| Priority | Fix | Effort |
|----------|-----|--------|
| 1 | FP-001: Wire real FinBERT embeddings into pipeline | Medium |
| 2 | SH-003: Switch backtest equity to mark-to-market | Low |
| 3 | SH-001: Fix double-sqrt annualisation bug | Low |
| 4 | LOG-001: Add traceback logging to all except blocks in train_gpu.py | Medium |
| 5 | LOG-003: Add structured logging to api/main.py | Low |
| 6 | SH-002: Change ddof=0 → ddof=1 in SharpeProxyLoss | Low |
| 7 | FP-002: Separate scheduled vs unscheduled events for pre_news | Medium |
| 8 | LOG-002: Replace print() with logging throughout | Medium |
| 9 | FP-003: Standardise column name across all consumers | Low |
| 10 | LOG-004: Add RotatingFileHandler + demotion log cleanup | Low |
| 11 | SH-004: Add risk-free rate to all Sharpe calculations | Low |
| 12 | SH-005: Concatenate fold returns before computing CV Sharpe | Low |
| 13 | FP-004: Explicit Float32 cast at end of sanitize_frame | Low |
