# Data Quality Issues & Fixes

## Overview
This document summarizes the known data quality issues in the forex dataset and the fixes applied to mitigate their impact on model training.

---

## 🔴 CRITICAL - GBPUSD 39-Day Hole

**Issue**: GBPUSD has a ~39-day data gap from **2018-11-23 to 2019-01-02** (951 hours / 39.6 days missing). Any training sequence crossing this boundary will see a massive artificial price jump.

**Root Cause**: Missing Dukascopy data for this period.

**Fix Applied**: Added explicit embargo in `data/data_ingestion.py`:
```python
KNOWN_DATA_GAPS = {
    "GBPUSD": [
        ("2018-11-23", "2019-01-02", "39-day hole: missing Dukascopy data Nov 2018 - Jan 2019"),
    ],
    ...
}
```

The `filter_embargo_gaps()` function is called in `ForexDataPipeline.run()` before gap detection/interpolation, dropping all bars within the known gap range.

**Alternative**: Re-download the missing period from Dukascopy.

---

## 🟠 HIGH - USDJPY 6.9-Day Hole

**Issue**: USDJPY has a ~6.9-day gap in **June 2009** (165 hours missing).

**Fix Applied**: Added embargo in `KNOWN_DATA_GAPS`:
```python
"USDJPY": [
    ("2009-06-01", "2009-06-30", "6.9-day hole: missing Dukascopy data June 2009"),
],
```

**Alternative**: Re-download the missing week from Dukascopy.

---

## 🟠 HIGH - Asymmetric Pair Coverage

**Issue**: 7 pairs only have 2024 data (1 year), while 3 major pairs have 18 years (2008-2025):

| Pair | Years | Date Range |
|------|-------|------------|
| EURUSD | 18 | 2008-2025 ✓ |
| GBPUSD | 18 | 2008-2025 ✓ |
| USDJPY | 18 | 2008-2025 ✓ |
| **AUDUSD** | **1** | **2024 only** |
| **EURGBP** | **1** | **2024 only** |
| **EURJPY** | **1** | **2024 only** |
| **GBPJPY** | **1** | **2024 only** |
| **NZDUSD** | **1** | **2024 only** |
| **USDCAD** | **1** | **2024 only** |
| **USDCHF** | **1** | **2024 only** |

**Impact**: Multi-pair training is heavily asymmetric. Models trained on all pairs will be dominated by the 3 pairs with deep history.

**Recommendations**:
1. **Download historical data** for the 7 pairs from Dukascopy (2008-2023)
2. **Use pair-specific models** instead of joint multi-pair training
3. **Weight sampling** by data availability during training

---

## 🟡 LOW - GBPJPY/EURJPY Extreme Spreads

**Issue**: JPY pairs (GBPJPY, EURJPY) have many extreme-spread ticks during BoJ/Brexit events that distort OHLCV bars.

**Fix Applied**: Added spread cap in `resample_to_bars()`:
```python
def resample_to_bars(df: pl.DataFrame, freq: str = "1min", pair: str = None, spread_cap_multiplier: float = 3.0):
    if pair is not None and pair.endswith("JPY") and "spread" in df.columns:
        median_spread = df["spread"].median()
        if median_spread is not None and median_spread > 0:
            cap = median_spread * spread_cap_multiplier
            df = df.with_columns(pl.col("spread").clip(upper_bound=cap).alias("spread"))
```

Configurable via `ForexDataPipeline(spread_cap_multiplier=3.0)` (default 3x median spread).

---

## 🟡 LOW - News Sentiment Nulls

**Issue**: 
- `historical_news_fnspid_full.csv`: **100% null** sentiment_score (28.2M rows)
- `historical_news_combined.parquet`: **29.5% null** sentiment_score (8.4M of 28.5M rows)

**Fix Applied**: 
1. In `data/historical_news.py`: Drop null sentiment rows before aggregation, and ensure neutral (0.0) sentiment frame exists even when no scored headlines
2. In `features/feature_engineering_pl.py`: `sentiment_tiers()` now fills null sentiment with 0.0 before computing decayed sentiment

```python
# In sentiment_tiers():
df = df.with_columns([pl.col("sentiment").fill_null(0.0).alias("sentiment_raw")])
```

This ensures the model's news features treat missing sentiment as neutral (0) rather than NaN.

---

## 🟡 LOW - ForexLive News Stub

**Issue**: `historical_news_forexlive.csv` has only **10 rows** (header + 9 data rows), all from 2026 with timestamps of `1970-01-01T00:00:00Z`.

**Status**: This appears to be a scraper stub/test file. The main news data comes from:
- `historical_news_combined.parquet` (28.5M rows, 29.5% null sentiment)
- `historical_news_fnspid_full.csv` (28.2M rows, 100% null sentiment)
- `historical_news_augmented.csv` (smaller augmented set)

**Recommendation**: Remove forexlive from the pipeline or fix the scraper.

---

## Files Modified

| File | Changes |
|------|---------|
| `data/data_ingestion.py` | Added `KNOWN_DATA_GAPS`, `filter_embargo_gaps()`, spread cap in `resample_to_bars()`, `ForexDataPipeline.spread_cap_multiplier` parameter, pair-aware `run()` method |
| `training/dataset_builder.py` | Pass `pair` to `pipeline.run()` and enable `spread_cap_multiplier` |
| `data/historical_news.py` | Ensure neutral sentiment frame when no scored headlines exist |
| `features/feature_engineering_pl.py` | Fill null sentiment with 0.0 in `sentiment_tiers()` and `sentiment_decay()` |

---

## Verification

To verify the fixes work:

```python
from data.data_ingestion import ForexDataPipeline, filter_embargo_gaps
import polars as pl

# Test embargo filtering
df = pl.DataFrame(
    {
        "timestamp_utc": pl.datetime_range(start="2018-11-20", end="2019-01-05", interval="1h", time_zone="UTC"),
        "open": 1.3,
        "high": 1.31,
        "low": 1.29,
        "close": 1.305,
        "volume": 1000,
        "spread_avg": 0.0001,
    }
)

filtered = filter_embargo_gaps(df, "GBPUSD")
# Should have gap from Nov 23 - Jan 2 removed

# Test spread cap
pipeline = ForexDataPipeline(spread_cap_multiplier=3.0)
bars = pipeline.run(jpy_tick_data, pair="GBPJPY")
# JPY spread capped at 3x median
```