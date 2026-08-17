# Dataset Building Process — Improvement Report
**Date:** 2026-08-01  
**Project:** `/run/media/jamie/jamie/forex-main`  
**Analyst:** Antigravity Dataset Pipeline Audit  
**Total Issues Found:** 10

---

## Executive Summary

The dataset building pipeline is architecturally sophisticated — particularly the RL reward labeling and fractional differentiation implementations. However, several **critical label construction flaws and data leakage risks** were identified that will produce artificially optimistic backtest results and degrade live performance. Two issues directly corrupt label quality. One introduces subtle EMA look-ahead bias across train/validation splits. The remaining issues cover schema correctness, signal quality, performance, and robustness.

---

## 🚨 Critical Issues (2)
*Directly corrupt label quality and produce overly optimistic P&L.*

---

### DS-001 — Exit Barriers Evaluated at Mid-Price (Zero Spread on Exit)

| Attribute | Detail |
|-----------|--------|
| **Files** | `labeling/triple_barrier_labeling.py` (L76–86, L137–151) |
| | `labeling/rl_reward_labeling.py` (L155, L161, L169) |
| **Area** | Label Construction / Data Leakage |
| **Severity** | 🚨 Critical |

**Description:**  
Both the Triple Barrier Method (TBM) and RL Reward labelers correctly use the spread-adjusted price for the **entry** (`entry_long`, `entry_short`). However, they evaluate the **forward barrier path** — the TP and SL hits — using the raw `close` (mid-price).

Correct exit execution should be:
- **Long exit (TP/SL hit):** triggered and executed at the **Bid** price
- **Short exit (TP/SL hit):** triggered and executed at the **Ask** price

**Impact:**  
By using mid-price for exit evaluation, the pipeline assumes **zero spread on every exit**. For EURUSD with a 1-pip spread, this understates exit costs by ~10% per trade on a 10-pip TP. The labels become systematically over-optimistic, causing the model to learn strategies that appear profitable in training but lose money live.

**Recommended Fix:**
```python
# In triple_barrier_labeling.py
# Replace:
hit_tp = future_path >= tp_price  # using close
# With:
hit_tp_long = future_bid_path >= tp_price  # exit long → bid
hit_tp_short = future_ask_path <= tp_price  # exit short → ask
```
Load bid/ask columns from the tick data loader and build separate `bid_path` and `ask_path` forward sequences alongside the existing `close_path`.

---

### DS-002 — EMA / Rolling Stats Suffer Look-Ahead Bias at Split Boundaries

| Attribute | Detail |
|-----------|--------|
| **File** | `features/feature_engineering_pl.py` (L464) |
| **Area** | Feature Engineering / Data Leakage |
| **Severity** | 🚨 Critical |

**Description:**  
For MACD and all other EWM (Exponentially Weighted Mean) indicators, Polars' `ewm_mean(adjust=False)` seeds the EMA from the **first value of the input series**. When features are generated independently per split chunk (train/val/test), each chunk initialises its EMA from a different starting point.

**Impact:**  
The EMA values for the first `3 × slow_window` bars of any test or validation chunk are mathematically inconsistent with what they would have been if the model had been running continuously from the beginning. This is a form of future look-ahead that breaks walk-forward consistency and makes validation metrics unreliable.

**Recommended Fix:**  
When building features for any non-training split, always **prepend at least `3 × slow_window` bars of historical context** before the split boundary, generate features on the combined sequence, and then slice off the context prefix before returning the validation/test features:
```python
WARMUP_BARS = 3 * slow_window
df_with_context = df_historical[-WARMUP_BARS:].append(df_split)
features = build_features(df_with_context)
return features[WARMUP_BARS:]  # drop the warm-up prefix
```

---

## 🔴 High Priority Issues (3)
*Significant impact on data correctness or training stability.*

---

### DS-003 — Timezone-Naive Parquet Timestamps Cause Silent Misalignment

| Attribute | Detail |
|-----------|--------|
| **File** | `data/databento_loader.py` (L41, L45) |
| **Area** | Schema / Type Safety |
| **Severity** | 🔴 High |

**Description:**  
The loader parses `start`/`end` strings with `datetime.strptime(...)` and attaches `tzinfo=timezone.utc`. However, if the underlying Parquet file's `ts_event` column was written as **timezone-naive**, Polars will either crash or silently misalign the filter (`pl.col('ts_event') >= start_dt`), since comparing naive and timezone-aware datetimes produces unexpected behaviour.

**Recommended Fix:**  
After loading the lazy frame, explicitly enforce UTC timezone on the timestamp column before filtering:
```python
lf = (
    pl.scan_parquet(path)
    .with_columns(pl.col("ts_event").dt.replace_time_zone("UTC"))
    .filter(pl.col("ts_event") >= start_dt)
)
```

---

### DS-004 — Crude Bag-of-Words Sentiment Fallback Corrupts News Signal

| Attribute | Detail |
|-----------|--------|
| **File** | `data/historical_news.py` (L168–173, L346–358) |
| **Area** | NLP / Data Quality |
| **Severity** | 🔴 High |

**Description:**  
When a pre-computed sentiment score is missing, the code falls back to `_sentiment_score()` — a rudimentary bag-of-words counter against `_POSITIVE_WORDS` and `_NEGATIVE_WORDS`. In financial contexts, words like **"fall"** or **"weak"** can be bullish for the counter-currency (e.g., "weak dollar" is bullish for EURUSD). The bag-of-words approach inverts such signals.

**Impact:**  
Historical backfill data processed through this fallback will have systematically wrong sentiment polarity for important macro events. This directly pollutes the training signal for the cross-attention news encoder.

**Recommended Fix:**  
Remove the bag-of-words fallback entirely. If a FinBERT score is unavailable, fill with `0.0` (neutral) rather than guessing:
```python
sentiment = row.get("sentiment_score") or 0.0  # neutral, not guessed
```

---

### DS-005 — Feature Engineering Materialises All 2,240 Columns Into RAM at Once

| Attribute | Detail |
|-----------|--------|
| **File** | `features/feature_engineering_pl.py` (multiple) |
| **Area** | Performance / Memory |
| **Severity** | 🔴 High |

**Description:**  
The `FeatureEngineer.build()` call chains hundreds of rolling window computations across 2,240 features on the full dataset simultaneously. Window functions in Polars force full materialisation into RAM. On a 14GB system with training data spanning 2008–2026, this will exhaust available memory.

**Recommended Fix:**  
Process the dataset in **overlapping chronological chunks** to bound peak RAM usage:
```python
CHUNK_BARS = 50_000
OVERLAP_BARS = 3 * slow_window  # warm-up context

for chunk_start in range(0, len(df), CHUNK_BARS):
    chunk = df[max(0, chunk_start - OVERLAP_BARS) : chunk_start + CHUNK_BARS]
    features = build_features(chunk)
    save_chunk(features[OVERLAP_BARS:])  # drop warm-up
```
Use `sink_parquet()` with year-based partitioning for incremental, resumable output.

---

## 🟡 Medium Priority Issues (3)

---

### DS-006 — Holiday Filtering Only Covers Dec 25 and Jan 1

| Attribute | Detail |
|-----------|--------|
| **File** | `data/data_ingestion.py` (L250–253) |
| **Area** | Data Quality / Robustness |
| **Severity** | 🟡 Medium |

**Description:**  
The holiday filter drops only December 25 and January 1. Numerous "floating" market holidays are unhandled:
- **EUR/GBP:** Good Friday, Easter Monday
- **USD:** Thanksgiving, Columbus Day, Presidents Day
- **JPY:** Japanese Bank Holidays (9+ per year)

**Impact:**  
The pipeline trains on dead/thin-liquidity bars during unhandled holidays. The RL reward labeler's `liquidity_vacuum` slippage multiplier will skyrocket on these days, generating misleading negative reward labels from what is really just illiquid noise.

**Recommended Fix:**  
Integrate `pandas_market_calendars` to build a proper holiday mask per currency:
```python
import pandas_market_calendars as mcal

calendar = mcal.get_calendar("FOREX")
valid_days = calendar.valid_days(start_date, end_date)
df = df.filter(pl.col("date").is_in(valid_days))
```

---

### DS-007 — "HMM Regime Probabilities" is Not an HMM

| Attribute | Detail |
|-----------|--------|
| **File** | `features/feature_engineering_pl.py` (L156–179) |
| **Area** | Feature Completeness / Correctness |
| **Severity** | 🟡 Medium |

**Description:**  
The `hmm_regime_probs` function claims to generate Hidden Markov Model regime probabilities but simply bins rolling standard deviation into static terciles (33rd / 66th quantiles). This is a volatility bucket, not an HMM. The misnaming creates confusion and the feature is weaker than a true regime classifier.

**Recommended Fix (Option A — Rename):**  
Rename to `vol_regime_bucket` to accurately describe what it computes.

**Recommended Fix (Option B — Implement):**  
Train a Gaussian HMM offline using `hmmlearn` on the full historical volatility series and merge the posterior regime probabilities as a precomputed feature column:
```python
from hmmlearn.hmm import GaussianHMM

model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
model.fit(vol_series.reshape(-1, 1))
regime_probs = model.predict_proba(vol_series.reshape(-1, 1))
```

---

### DS-008 — In-Memory Deduplication Set Will OOM on Large News CSV

| Attribute | Detail |
|-----------|--------|
| **File** | `scripts/download_historical_news.py` (L665–737) |
| **Area** | Pipeline Robustness / Memory |
| **Severity** | 🟡 Medium |

**Description:**  
`_IncrementalCSVWriter` initialises `self._seen` by reading every existing row from the CSV file into a Python `set`. Each entry is a tuple of multiple string columns. With millions of rows, this set will consume several GB of RAM — the same class of OOM bug we already fixed in the GDELT downloader.

**Recommended Fix:**  
Replace the in-memory set with a DuckDB or Polars on-disk deduplication at write time:
```python
import duckdb

con = duckdb.connect()
con.execute(f"""
    INSERT INTO output SELECT * FROM new_rows
    WHERE (timestamp_utc, headline) NOT IN (SELECT timestamp_utc, headline FROM output)
""")
```

---

## 🟢 Low Priority Issues (2)

---

### DS-009 — Order Book Imbalance is Simulated, Not Real

| Attribute | Detail |
|-----------|--------|
| **File** | `features/feature_engineering_pl.py` (L96–115) |
| **Area** | Feature Completeness |
| **Severity** | 🟢 Low |

**Description:**  
`multi_level_obi` simulates L2 depth by synthetically slicing the High-Low range. The `databento_loader.py` already pulls **MBP-10** (Market by Price, 10-level) data with real `bid_sz_01..10` and `ask_sz_01..10` columns — but those are ignored in feature engineering.

**Recommended Fix:**  
Expose the true L2 bid/ask size columns from `databento_loader.py` and use them to build genuine Order Book Imbalance:
```python
true_obi = (bid_sz_01 - ask_sz_01) / (bid_sz_01 + ask_sz_01 + 1e-9)
```
This will be a dramatically stronger microstructure signal than the synthetic OHLC-derived version.

---

### DS-010 — Float64/Float32 Precision Ping-Pong Wastes Memory

| Attribute | Detail |
|-----------|--------|
| **File** | `labeling/rl_reward_labeling.py` (L100–114, L140) |
| **Area** | Performance / Memory |
| **Severity** | 🟢 Low |

**Description:**  
Market arrays are cast to `float64`, operated on, and then cast back to `float32` for rewards. This ping-ponging wastes memory — for standard forex pricing (5 decimal places), `float32` provides more than sufficient pip-level accuracy throughout the entire labeling loop.

**Recommended Fix:**  
Standardise on `float32` throughout. Remove the `float64` cast and operate natively in `float32`:
```python
prices = np.asarray(prices, dtype=np.float32)  # never upcast to float64
```

---

## Recommended Fix Order

| # | Priority | Issue | Effort | Impact |
|---|----------|-------|--------|--------|
| 1 | 🚨 Critical | DS-001: Exit barriers use mid-price (zero exit spread) | Medium | Labels systematically over-optimistic |
| 2 | 🚨 Critical | DS-002: EMA look-ahead bias at split boundaries | Low | Walk-forward validation unreliable |
| 3 | 🔴 High | DS-004: Bag-of-words sentiment fallback inverts signals | Low | Corrupts news training signal |
| 4 | 🔴 High | DS-003: Timezone-naive timestamps cause silent misalignment | Low | Silent data misalignment |
| 5 | 🔴 High | DS-005: Feature engineering OOM on full dataset | Medium | Training crashes on large windows |
| 6 | 🟡 Medium | DS-006: Holiday filtering misses floating bank holidays | Low | Noisy RL reward labels |
| 7 | 🟡 Medium | DS-007: HMM feature is just a vol bucket | Medium | Weak regime signal |
| 8 | 🟡 Medium | DS-008: Deduplication OOM on large news CSV | Low | Pipeline crash risk |
| 9 | 🟢 Low | DS-009: Simulated OBI ignores real L2 Databento data | Medium | Missed microstructure alpha |
| 10 | 🟢 Low | DS-010: Float64/Float32 ping-pong wastes memory | Low | Minor memory waste |
