# Forex ML Pipeline — Improvement Report
**Date:** 2026-08-01  
**Project:** `/run/media/jamie/jamie/forex-main`  
**Analyst:** Antigravity Pipeline Audit  
**Total Issues Found:** 11

---

## Executive Summary

A full audit of the trading pipeline identified **11 actionable issues** spanning live execution, risk management, NLP signal integration, data efficiency, model architecture, backtesting correctness, and configuration hygiene. Two issues are classified as **Critical** and represent hard blockers for any live trading deployment. Three are **High** priority and directly limit model performance. The remaining six are Medium or Low and represent correctness, maintainability, and optimisation opportunities.

---

## 🔴 Critical Issues (2)
*Must be resolved before any live deployment.*

---

### ISSUE-001 — Broker Bridge is an Unimplemented Stub

| Attribute | Detail |
|-----------|--------|
| **File** | `execution/broker_bridge.py` |
| **Line** | 47 |
| **Area** | Live Execution |
| **Severity** | Critical |

**Description:**  
The `execute_order` method in `BrokerBridge` raises a `NotImplementedError("BrokerBridge is currently a stub...")`. The system has no capability to place, modify, or close real trades.

**Impact:**  
Any attempt to run the system in live mode will raise an unhandled exception on the very first order. The entire live trading stack is non-functional.

**Recommended Fix:**  
Implement the broker bridge for at least one execution target (MT5 via `MetaTrader5` Python package, or IBKR via `ib_insync`). Include order placement, modification, closure, and position querying methods. Add connection health-check and reconnect logic.

---

### ISSUE-002 — Hardcoded Pip Value Causes Catastrophic Position Sizing for JPY Pairs

| Attribute | Detail |
|-----------|--------|
| **File** | `features/feature_engineering_pl.py` |
| **Line** | 260 |
| **Area** | Risk / Position Sizing |
| **Severity** | Critical |

**Description:**  
The position size calculation hardcodes `pip_value = pl.lit(10.0)`. This value is correct only for USD-quoted pairs (EURUSD, GBPUSD) where 1 pip = $10 per standard lot. For USDJPY, the pip value is approximately `$0.009 × account_equity × exchange_rate`, which is roughly **1,000× smaller**.

**Impact:**  
USDJPY positions will be sized approximately 1,000× too large relative to the intended risk percentage. A single trade could exceed the entire account balance.

**Recommended Fix:**  
Compute `pip_value` dynamically based on the quote currency. For JPY-quoted pairs (e.g., USDJPY), `pip_value = (lot_size × 0.01) / exchange_rate`. For USD-quoted pairs, `pip_value = lot_size × 0.0001`. This lookup should be driven by a currency metadata table in `config/settings.py`.

---

## 🟠 High Priority Issues (3)
*Significant impact on model performance or system stability.*

---

### ISSUE-003 — NLP/Sentiment Embeddings Return All Zeros

| Attribute | Detail |
|-----------|--------|
| **File** | `features/feature_engineering_pl.py` |
| **Line** | 273 |
| **Area** | NLP / Feature Engineering |
| **Severity** | High |

**Description:**  
The `embedding_placeholders()` function returns columns filled with `0.0` instead of real FinBERT or sentence-transformer embeddings. The entire `historical_news_combined.parquet` dataset (255,707 high-quality macroeconomic news events, 2008–2026) produces **zero signal** into the model.

**Impact:**  
The model has no awareness of central bank announcements, NFP releases, inflation prints, or geopolitical events. This is the most impactful missing signal in the current pipeline.

**Recommended Fix:**  
1. Load `historical_news_combined.parquet` at feature build time using `pl.scan_parquet()`.
2. Align news rows to OHLCV bars using an `asof` join on `timestamp_utc`.
3. Pass the `headline` column through a pre-trained `ProsusAI/finbert` model to generate 768-dim sentiment embeddings.
4. Reduce via PCA to 32–64 components and store as feature columns.
5. Pass these features to the HAELT model's cross-attention news encoder.

---

### ISSUE-004 — News Parquet Eagerly Loaded Into RAM

| Attribute | Detail |
|-----------|--------|
| **File** | `data/historical_news.py` |
| **Lines** | 142, 238 |
| **Area** | Data Loading |
| **Severity** | High |

**Description:**  
`_read_table()` uses `pl.read_parquet(path)` to eagerly load the entire news dataset into RAM at once. As the dataset grows, this will trigger an OOM crash identical to the one experienced during the GDELT resume scan.

**Recommended Fix:**  
Replace all instances of `pl.read_parquet(path)` with `pl.scan_parquet(path)` and push all filtering predicates into the lazy evaluation graph before calling `.collect()`.

---

### ISSUE-005 — Session Windows Ignore Daylight Saving Time (DST)

| Attribute | Detail |
|-----------|--------|
| **File** | `config/settings.py` |
| **Lines** | 748–752 |
| **Area** | Risk / Configuration |
| **Severity** | High |

**Description:**  
London and NY session start/end hours are hardcoded as fixed UTC integers. During DST transitions (March and November), the true session start shifts by ±1 hour relative to UTC. The system applies the wrong session context during those periods.

**Recommended Fix:**  
Use `zoneinfo` (`Europe/London`, `America/New_York`) to compute UTC session boundaries at runtime. Cache the result daily.

---

## 🟡 Medium Priority Issues (4)

---

### ISSUE-006 — Extreme Input Clipping Destroys News-Event Volatility Signals

| Attribute | Detail |
|-----------|--------|
| **File** | `models/architectures.py` |
| **Lines** | 758, 766 |
| **Severity** | Medium |

The `HAELTHybrid` model maps NaN inputs to `±10.0` and clamps all values to `[-10.0, 10.0]`. High-impact news events (FOMC, NFP, Brexit) routinely produce normalised measurements outside this range — silently destroyed before the model sees them.

**Fix:** Replace hard clipping with a learned `tanh` gate or `torch.nn.functional.hardtanh` with configurable bounds. Move NaN imputation upstream to the feature builder.

---

### ISSUE-007 — API Keys Hardcoded as Empty Strings

| Attribute | Detail |
|-----------|--------|
| **File** | `config/settings.py` |
| **Severity** | Medium |

`fred_api_key`, `av_api_key`, and `discord_webhook_url` are set to `""`. Risk of secrets accidentally being committed to version control.

**Fix:** Use `os.getenv('FRED_API_KEY', '')`. Add `.env.example` and add `config/secrets.py` to `.gitignore`.

---

### ISSUE-008 — Candlestick Pattern Thresholds Are Untunable Magic Numbers

| Attribute | Detail |
|-----------|--------|
| **File** | `features/feature_engineering_pl.py` |
| **Lines** | 539, 548 |
| **Severity** | Medium |

`is_hammer` and similar patterns use hardcoded ratios (e.g., `body/range <= 0.3`) with no configuration exposure. Optuna cannot tune these values during hyperparameter search.

**Fix:** Move thresholds into a `CandlestickConfig` dataclass inside `config/settings.py`.

---

### ISSUE-009 — SL/TP Forward-Fill Can Bleed Values Between Unrelated Trades

| Attribute | Detail |
|-----------|--------|
| **File** | `backtesting/backtest.py` |
| **Line** | 124 |
| **Severity** | Medium |

`.ffill()` is applied across the entire bar dataframe. A new trade with null SL/TP inherits the previous trade's stale values. Backtest results are silently incorrect — trades appear to have tighter risk management than they actually do.

**Fix:** Reset SL/TP to `NaN` whenever position direction changes. Only forward-fill within a continuous open-position window.

---

## 🟢 Low Priority Issues (2)

---

### ISSUE-010 — Intermediate Merge Output is an Uncompressed Flat CSV

| Attribute | Detail |
|-----------|--------|
| **File** | `scripts/merge_datasets.py` |
| **Line** | 66 |
| **Severity** | Low |

Output sinks to `historical_news_master.csv` — flat, unpartitioned, uncompressed. Downstream scans are slow.

**Fix:** Sink to partitioned Parquet (`partition_by='year'`) for 10–50× faster downstream reads.

---

### ISSUE-011 — Silent Sequence Cropping Hides Data Loss During Training

| Attribute | Detail |
|-----------|--------|
| **File** | `training/train_gpu.py` |
| **Line** | 359 |
| **Severity** | Low |

`_crop_to_seq_len()` silently truncates tensors without logging a warning or asserting. Sequence window mismatches are invisible.

**Fix:** Add `logging.warning(...)` whenever a crop is applied. Add a `strict_seq_len` config flag that raises `ValueError` in test mode.

---

## Recommended Fix Order

| # | Priority | Issue | Effort | Impact |
|---|----------|-------|--------|--------|
| 1 | Critical | ISSUE-002: Pip value per currency | Low | Catastrophic if unresolved |
| 2 | Critical | ISSUE-001: Implement broker bridge | High | System non-functional without it |
| 3 | High | ISSUE-003: Wire in real NLP embeddings | Medium | Biggest model alpha improvement |
| 4 | High | ISSUE-004: Lazy news parquet loading | Low | Prevents future OOM crashes |
| 5 | High | ISSUE-005: DST-aware session windows | Low | Correctness during volatile periods |
| 6 | Medium | ISSUE-009: SL/TP ffill bug in backtest | Low | Silent correctness issue |
| 7 | Medium | ISSUE-007: API keys from env vars | Low | Security hygiene |
| 8 | Medium | ISSUE-006: Soft-clip extreme values | Medium | Better news-event response |
| 9 | Medium | ISSUE-008: Candlestick config exposure | Low | Enables Optuna tuning |
| 10 | Low | ISSUE-010: Partitioned Parquet output | Low | Speed improvement |
| 11 | Low | ISSUE-011: Sequence crop warning | Low | Debugging quality |
