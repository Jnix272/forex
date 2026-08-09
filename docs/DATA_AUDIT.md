# Data Audit — 2026-08-07

Cross-checks every raw data source on disk against what the training pipeline actually ingests, plus a logging audit of which loads emit signal vs which fail silently.

Findings sourced from two forensic subagent passes + on-disk verification.

---

## 1. On-disk inventory (47.4 GB / 189,077 files)

| Subdir | Size | Notes |
|---|---|---|
| `data/raw/news/` | 24 GB | `historical_news_combined.parquet` 1.7G; raw CSVs (`historical_news_2021_2025.csv` 14G, `historical_news_fnspid_full.csv` 5.6G, `historical_news_2026.csv` 1.5G, `Hugging_Face.csv` 191M); `.pre_step2.bak` 1.6G; `.2026_append.parquet` 254M (redundant sidecar, safe to delete); `historical_news_augmented.csv` 15M (auto-loaded); `historical_news_synthetic_fill.csv` 352K; `historical_news_forexlive.csv` 2.7K |
| `data/compact/dukascopy/` | 10.7 GB | Hive-partitioned re-materialization of raw ticks (derived cache, not raw input) — `dukascopy_ticks.duckdb` + `pair=*/year=*/month=*/day=*/ticks.parquet` |
| `data/raw/dukascopy/` | 9.0 GB | Per-hour tick cache, 173K files, 10 pairs. **Coverage asymmetry**: 7 minors (AUDUSD, EURGBP, EURJPY, GBPJPY, NZDUSD, USDCAD, USDCHF) have **only 2024**; 3 majors (EURUSD, GBPUSD, USDJPY) have **2008–2025** |
| `data/embeddings/` | 855 MB | `sentiment_cache.pkl` (FinBERT/Ollama/VADER cache — not embeddings despite the dir name) |
| `data/processed/cross_asset/` | ~3 MB | CSV caches `<asset>_<provider>_<sym>.csv` from Stooq/FRED/Yahoo (mostly Yahoo per audit) |
| `data/raw/cot/` | 244 KB | `cot_financials_cleaned.parquet` |
| `data/raw/eco_calendar/` | 12 MB | `events.csv` |
| Placeholder/empty | — | `data/raw/eodhd/`, `data/raw/lmax/`, `data/raw/myfxbook/` (loaders exist in code, no data files present) |

### Suspicious files flagged for review

- `historical_news_combined.parquet.pre_step2.bak` — 1.6G backup from step 2 fix; only 19 MB diff from the live file. Safe to drop after we trust the live file.
- `historical_news_2026_append.parquet` 254M — **redundant**: combined parquet already contains all 2026 rows (verified: 8,409,141 rows in combined, exactly matching 8,409,141 rows in append).
- `historical_news_augmented.csv` — auto-loaded silently, 69,203 augmented headlines reach every training run today
- `historical_news_synthetic_fill.csv` 352K, `historical_news_forexlive.csv` 2.7K — likely test fixtures, not real srcapes
- `Hugging_Face.csv` — contains 1969-12-31 sentinel-timestamp rows
- 7 runner files (`*.failure`, `*.progress`, dump JSONs) named in `scripts/` defaults no longer exist on disk

---

## 2. What the pipeline actually ingests

Confirmed by reading both yaml configs (`config/run.yaml`, `config/run_ubuntu.yaml`), the CLI entrypoint (`training/gpu_cli.py`), and every loader module (`data/{historical_news,cross_asset,sources,data_ingestion}.py`, `training/dataset_builder.py`, `features/{feature_engineering_pl,finbert_sentiment,macro_features,cot_features}.py`).

### Sources wired and verified working

| Source | File / endpoint | Wired at | Logs visible? |
|---|---|---|---|
| **Dukascopy ticks** (majors 2008–2025) | `data/raw/dukascopy/<PAIR>/...` | `ForexDataManager.load(source="dukascopy")` via `--data-source` | yes (verbose-gated `print()`) |
| **Dukascopy compact DuckDB view** | `data/compact/dukascopy/dukascopy_ticks.duckdb` | `query_dukascopy_duckdb` (preferred fast path) | per-file errors only; successful reads silent |
| **Historical news headlines** | `data/raw/news/historical_news_combined.parquet` | `load_historical_news_bundle(news_file=…)` `mode=full` | **NO** — file 1.7 GB opened silently, no row-count log |
| **Augmented news** | `data/raw/news/historical_news_augmented.csv` | **auto-appended** in `_load_events` lines 198–200 when default parquet is in path list | **NO** — silently concatenated |
| **Economic calendar** | `data/raw/eco_calendar/events.csv` | `load_historical_news_bundle(calendar_file=…)` | **NO** — silent; missing file skipped |
| **COT financials** | `data/raw/cot/cot_financials_cleaned.parquet` | `dataset_builder.py:2506` (main path) + `:2437` (parallel worker) — **hard-coded path** | partial — main path logs `Loaded COT data (N rows)`; parallel worker **silent** |
| **Cross-asset panel** (yields, equities, VIX, DXY, oil, gold) | `data/processed/cross_asset/<asset>_<provider>_<sym>.csv` ← live fetch fallbacks (Stooq/Yahoo/FRED/EODHD) | `load_cross_asset_panel(start, end, cache_dir, source)` via `--cross-asset-mode` + `--cross-asset-provider` | partial — main path prints `Loaded external assets: N`; per-asset success is silent; only total failure emits `logging.warning` |
| **FRED yields** (10Y/2Y for 8 currencies) | `fredapi.Fred.get_series` (needs `FRED_API_KEY`), fallback to synthetic | `features/macro_features.py::MacroYieldFeatureBuilder.build` — **independent** FRED path inside FeatureEngineer | yes — `[MacroFeatures] FRED {name}: {N} obs` or `… failed ({e}) — using synthetic` |
| **FinBERT sentiment cache** | `data/embeddings/sentiment_cache.pkl` 855 MB | `_load_cache()` on `SentimentPipeline.__init__` | **NO** on canonical load (only stale-merge prints). Corrupted cache → empty dict silently → every headline re-scored live |
| **Live sentiment override** (finbert/ollama/vader) | Ollama `localhost:11434` → HF `ProsusAI/finbert` → VADER | `_build_chunk:1626–1656` (newly fixed — see entry in CHANGELOG) | partial — backend detection prints; per-chunk override silent on success / warns only on except |

### Sources on disk but NOT wired into training

| Source | Status |
|---|---|
| `historical_news_2026_append.parquet` | redundant — its 8.4M rows are already in the combined parquet |
| `data/raw/eodhd/`, `data/raw/lmax/`, `data/raw/myfxbook/` | empty placeholder dirs; the loaders exist in code but no data was ever downloaded |
| `historical_news_2021_2025.csv`, `historical_news_fnspid_full.csv`, `historical_news_2026.csv` | raw source CSVs from prior ingestion steps; **not loaded at training time** (the combined parquet supersedes them) |
| `Hugging_Face.csv` | training-aid dataset; not loaded at training time |
| 7 minor dukascopy pair dirs with only 2024 coverage | loaders CAN read them, but the YAML config only requests EURUSD/USDJPY/GBPUSD as pairs — so they are not actually ingested in the current setup |
| `finbert_embs` field on `HistoricalNewsBundle` | **dead code**: never assigned by `load_historical_news_bundle`, so FeatureEngineer always hits the zero-placeholder branch + prints warning |

---

## 3. Logging audit

### Failures with no log at all (silent `try/except`)

Critical silent failure sites where the loader returns empty / None and the FeatureEngineer zero-fills:

1. `data/historical_news.py:120–122` — missing file → empty df returned silently
2. `data/historical_news.py:296–297` — empty df → `empty_news_bundle()` returned silently *(highest-impact silent failure — zero news/COT/eco features with no top-level warning)*
3. `data/historical_news.py:215–228` — DuckDB slice falls back to polars read with no log
4. `data/cross_asset.py:125, 162, 200, 215, 225, 264` — Stooq / Yahoo / FRED / EODHD / cache read/write all return `None` silently on any error
5. `features/finbert_sentiment.py:84–85, 106–107, 119–120` — cache load (`855 MB`) failure → empty dict; cache save → silently skipped
6. `training/dataset_builder.py:2437–2441` — COT load in parallel worker is silent (main path logs)
7. `training/dataset_builder.py:2700–2705` — Zarr resume-state json read silent
8. `features/feature_engineering_pl.py:2018, 2043–2045, 2084–2087, 2188–2192, 2206, 2219` — every "None" input silently zero-fills + at most a stderr warning, no structured log
9. `features/macro_features.py:132–134, 216–217` — FRED exceptions are logged but synthetic-fallback is **invisible** to FeatureEngineer (FE only sees `macro_df` came back, never whether it was real or synthetic)
10. `data/data_ingestion.py:523, 701` — pandas_market_calendars / Lomb-Scargle silent
11. `data/sources.py:490, 1548` — per-hour `to_parquet` cache write / per-file parquet read silent

### Logged (mostly via bare `print()`, not structured)

| Step | What's logged | Row counts? | Format |
|---|---|---|---|
| COT load (main path) | `[MultiPair] Loaded COT data (N rows)` | YES | `print()` |
| Cross-asset panel success | `[MultiPair] Loaded external assets: N (source=…, cache=…)` | count, not rows | `print()` |
| Cross-asset per-asset failure | `logging.warning: CrossAsset: all providers failed for X (tried [...])` | NO | `logging.warning` |
| FRED yields (macro path) | `[MacroFeatures] FRED {name}: {N} obs` | YES | `print()` |
| Sentiment prefetch | `[Sentiment] Prefetch: M misses of N unique headlines` (only when misses > 0) | YES | `print()` |
| Sentiment backend detection | `[Sentiment] Ollama not reachable` / `Falling back to VADER` | n/a | `print()` |
| Dukascopy tick load | per-hour `[Dukascopy] Async Loading …`, %  progress, missing hours | YES (hours) | `print()` verbose-gated |
| Tick → bar pipeline | `[Pipeline] Raw tick rows: N`, `Bars after resampling: N`, `Final bar count after cleaning: N` | YES | `print()` |
| FeatureEngineer.build warnings | `WARNING: no FinBERT embeddings`, `COT features build failed`, `Macro features build failed`, `multi-modal sentiment failed` | NO | `print()` (stderr-style) |
| RL labeling | `[RLLabelingRegime] N labels | Long/Short/Hold/No-trade: … | path_quality: …` | YES | `print()` |
| Data quality gate | `[DataQuality] Chunk k: flagged N rows`, `dropped N sequences` | YES | `print()` |
| Pair readiness | `[Pair Readiness] <pair> PASS seq=N dropped=N nan_pct=…%` | YES | `print()` |
| FeatureSchema gate | OK / `FAIL — mismatches …` | count | `print()` |

### What is missing in logs

- **No row count** for: news parquet load, eco-calendar load, augmented-news load, COT in parallel worker, cross-asset cache hits, FinBERT cache load (855 MB pickle)
- **No log on success** for: cross-asset cache reads, FinBERT cache load, sentiment override success per chunk
- **No `from log alone` distinguishability** for: real FRED vs synthetic yields (only the sub-print `[MacroFeatures] FRED …` reveals it), zero-news-bundle vs "news file path was wrong"
- **Two COT load paths with different logging** — if you run `--parallel-window-workers > 1` you'll see no COT log line at all even though COT *was* loaded

---

## 4. Cross-cutting issues worth action

1. **Silent dataframe zero-fill on `sentiment` is the deep bug** (flagged in cont. conversation as `historical_news.py:347 fill_null(0.0)`): converts the 8.4M NULL URL-mode rows into 0.0 (neutral) per timestamp before aggregation. This dilutes the real-news signal at every news timestamp because `mean()` averages URL-rows (0.0) with real-headline scores (e.g. ±0.6).
2. **Two FRED/yield paths run independently** — `cross_asset.py::load_cross_asset_panel` (gate via `--cross-asset-mode`) and `features/macro_features.py::MacroYieldFeatureBuilder.build` (always-on inside FeatureEngineer). They both call FRED, cache to different places, and log with different formats. The macro path will silently produce synthetic yields if FRED_API_KEY is missing, while cross-asset path silently returns None.
3. **`finbert_embeddings` field on `HistoricalNewsBundle` is dead code** — always None today. Either wire it (load `data/embeddings/*.npy` files into the bundle) or remove the field and the FE placeholder branch.
4. **COT load has divergent logging** between main path (logged) and parallel worker (silent). Should be unified in a `load_cot(path)` helper.
5. **`historical_news_combined.parquet.timestamp_utc` is `VARCHAR`, not `TIMESTAMP`** — Adds a cast cost and means DuckDB-side date filtering is more expensive than necessary. Worth a one-shot cast during the next news ingestion.
6. **No structured `logging`** — entire data pipeline uses bare `print()`. A small `infrastructure/logging_utils.py::log_data_load(source, path, n_rows, status, t0, exc=None)` helper emitting `logging.getLogger("data").info(...)` would convert the 20k-line log into a grep-able record.

---

## 5. Suggested next actions (after audit decisions)

Priority order, smallest-bug-impact first:

| # | Action | Cost | Why before Zarr rebuild |
|---|---|---|---|
| A | Fix `historical_news.py:347` `fill_null(0.0)` — skip NULLs in mean (don't fill 0) | 2-line patch | Rebuilding Zarr over diluted sentiment is wasted compute |
| B | Delete redundant `historical_news_2026_append.parquet` and `pre_step2.bak` | `rm` call | Free 1.9 GB, prevent confusion |
| C | Add `load_cot` helper unifying main + parallel COT load + log | 10-line patch | Same code path, same log line |
| D | Add `log_data_load` helper + wire into 6 highest-impact load sites (news parquet, eco-calendar, augmented, FinBERT cache, cross-asset per-asset) | 1 small module + ~30 LOC at sites | Before a 60h rebuild, you want to verify from the first log line that every source actually loaded |
| E | Re-smoke tiny `--data-end 2008-01-15` to exercise (A)+(C)+(D) end-to-end | 2 min run | Confirm fixes land before committing to big rebuild |
| F | Then full Zarr rebuild via `run_ubuntu.yaml --skip-training` | 60-70h | Real rebuild |
| G | Drop `finbert_embeddings` dead code path or wire it | optional cleanup | Reduces operator confusion |

The audit does **not** block any immediate user-facing decision — but the Zarr rebuild shouldn't run until at least (A) and (D) land, since (A) affects every sentiment-per-timestamp value and (D) gives us the visibility the audit just exposed is missing.
