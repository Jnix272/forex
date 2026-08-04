# Forex ML Pipeline — Improvement Plan
**Date:** 2026-08-03
**Project:** `/run/media/jamie/jamie/forex-main`
**Scope:** Personal-use FX research + automated trading (solo developer, no regulatory constraints)
**Hardware:** RTX 4060 8 GB + 16 GB RAM, Linux

> This document is the forward-looking **roadmap**. Historical audits live in
> [`archive/`](archive/) (start with [`archive/FIXES_APPLIED.md`](archive/FIXES_APPLIED.md)).
> Config / curriculum / dataset schema gates: [`CONFIG_CONSISTENCY.md`](CONFIG_CONSISTENCY.md).
> Day-to-day tracker: [`CONTINUE.md`](CONTINUE.md). Docs index: [`README.md`](README.md).

---

## 1. TL;DR — Biggest Wins First

| Priority | Work | Why now | Effort |
|---|---|---|---|
| 🟥 P0 | Fix critical bugs (broker stub, JPY pip sizing, exit at mid-price, label leakage) | Hard blockers for any real money / meaningful backtest | Small |
| 🟧 P1 | Diagnose the actual bottleneck (GPU idle? data-bound? overfitting?) | Prevents optimising the wrong thing | Medium |
| 🟨 P2 | Stage 1 data-ingestion upgrades (✅ done) + real news sentiment | Biggest data-quality lift per unit effort | Medium |
| 🟩 P3 | Streaming pipeline, auto-retrain, portfolio backtester | Foundation for "set-and-forget" operation | Large |

---

## 2. Where the Pipeline Is Today

- **Data:** Dukascopy ticks 9.4 GB (10 pairs, 2008–2025, daily parquet), eco calendar `events.csv`, COT parquet, ~20 GB of historical news CSV, empty dirs for EODHD / LMAX / Myfxbook.
- **Ingestion:** `data/data_ingestion.py` — Polars-native load/clean/resample, MAD bad-tick cleaning, time + tick/volume/dollar information bars, DST-aware sessions, holiday calendar, gap detection/fill, Lomb-Scargle sampling analysis, lazy parquet reads. **Stage 1 (this plan §3) is implemented and tested.**
- **Features/Labels:** Polars feature engineering, FFD fractional diff, TBM + RL reward labeling, spread-aware entry.
- **Training:** sklearn LightGBM-style + deep models on a single RTX 4060 8 GB; RL training loops exist.
- **Execution:** `BrokerBridge` is a stub — no live orders possible.

---

## 3. Stage 1 — Data Ingestion Upgrades (DONE)

Implemented in `data/data_ingestion.py`, wired through `config/settings.py`, covered by `tests/test_data_ingestion.py` (21 tests).

| Feature | What it does | Config |
|---|---|---|
| Timestamp detection | Recognises Dukascopy `__index_level_0__` / `ts_event` / `time` columns | — |
| MAD bad-tick cleaning | Robust z-score vs. median absolute deviation + spread sanity; preserves legit wide-spread (news) ticks | `bad_tick_mad_z_thresh`, `bad_tick_spread_ratio`, `bad_tick_spread_window` |
| Information bars | Tick / volume / dollar bars close on event count, not wall clock | `bar_type`, `info_bar_threshold` |
| DST-aware sessions | Asia/London/NY windows defined in local tz so UTC shifts with DST | `session_mode`, `add_session_label` |
| Market holidays | `pandas-market-calendars` FOREX calendar + fixed-date fallback + thin-liquidity day guard | — |
| Gap detection/fill | `detect_bar_gaps` + `fill_gaps` (drop / ffill / interpolate), never bridges weekends | `gap_policy`, `gap_max_minutes` |
| Lazy loading | `scan_parquet`/`scan_csv` with start/end filter pushdown (avoids OOM on 16 GB RAM) | `load_tick_data(..., start, end)` |
| Sampling analysis | Lomb-Scargle periodogram on inter-arrival times → dominant period + regularity | — |

**Remaining for this stage (optional):** CLI `--start/--end`, per-pair cache index, schema drift guard.

---

## 4. Stage 2 — Diagnose the Real Bottleneck (Before Anything Else)

We do not yet know whether the model is **compute-bound, data-bound, or accuracy-bound**. Measure first:

1. **GPU utilisation** while training (`nvidia-smi dmon`). If < 80% idle on the 4060 → data loading / feature pipeline is the bottleneck.
2. **Wall-clock split** of one full training run: ingestion → feature build → label build → fit → eval.
3. **Learning curves** (train vs. val loss/Sharpe by epoch). If train ≫ val → capacity/regularisation; if both plateau early → feature quality.
4. **Walk-forward P&L sanity** on a holdout year vs. random baseline.

Only then choose: buy more data (Databento), stream ticks, or train harder.

---

## 5. Stage 3 — Quick Wins (Data Quality)

These are cheap and have outsized impact, roughly in order:

1. **Real news sentiment** — the 14 GB `historical_news_2021_2025.csv` + `fnspid_full.csv` are currently barely used. Wire the existing `data/historical_news.py` bundle into features (sentiment delta around high-impact events).
2. **Real order-book / L2 OBI features** — for Dukascopy raw ticks, compute order-book-imbalance proxies; feeds the RL state. (Full depth needs Databento.)
3. **Point-in-time discipline** — align all eco/COT/news to their *available* timestamps (`data/point_in_time.py` exists) so backtests don't leak future releases.
4. **Fix DS-001 exit-at-mid** (see audit) so labels reflect spread on exits.
5. **Chunked feature materialisation** — the 16 GB RAM box can't hold 17 years × 10 pairs of 1s ticks in memory; materialise per-pair/per-year parquet feature stores (`data/feature_store.py` exists) and stream fold windows.

---

## 6. Stage 4 — Major Projects

1. **Streaming tick pipeline** — poll Dukascopy/Databento, upsert compacted parquet (`data/sources.py` compaction exists), keep a rolling in-memory window for live inference.
2. **Automated retrain loop** — weekly walk-forward retrain behind promotion gates (the repo already has promotion/readiness gates); add drift triggers (model + data drift detectors exist).
3. **Portfolio backtester** — multi-pair, spread + slippage + margin aware, co-integration-aware position sizing; replaces single-pair eval.
4. **Live execution** — implement `BrokerBridge` for MT5 or IBKR (ISSUE-001) with health-check/reconnect and dry-run→paper→live promotion.

---

## 7. Data Roadmap

| Source | Status | Verdict |
|---|---|---|
| Dukascopy ticks (9.4 GB, 2008–2025) | ✅ present | Good coverage; keep as raw archive |
| Eco calendar `events.csv` | ✅ present | Use for event-window features |
| COT | ✅ present | Weekly; low-frequency regime features |
| Historical news CSVs | ✅ present | Parsed but under-used → Stage 3 |
| EODHD / LMAX / Myfxbook | empty dirs | Skip unless a specific feature requires them |
| **Databento** | not subscribed | 🎯 Best next purchase: consolidated L2 order book + MBO, clean schema, streaming API |

> Databento (via `data/databento_loader.py`, already scaffolded) is the single highest-leverage data upgrade: real book, funding, per-tick timestamps, no download scraping.

---

## 8. Hardware-Aware Training Config

- 8 GB VRAM → keep models ≤ ~1M params / batch ≤ 256 / fp16 or int8 quant for inference; RL replay buffers on disk not GPU.
- 16 GB RAM → never load multi-pair tick sets in one frame; use lazy parquet + per-fold streaming (`load_tick_data(start=..., end=...)`).
- Disk is plentiful → cache everything as parquet (feature store, compacted cache); avoid CSV round-trips.
- Train on 1 pair at a time or a curated 3–4 pair basket; full 10-pair ensembles only for final promotion, overnight.

---

## 9. Risk & Correctness Checklist (from audit, condensed)

- [ ] ISSUE-001 Broker bridge stub → implement or gate live mode
- [ ] ISSUE-002 JPY pip sizing hardcoded → currency metadata table in `config/settings.py`
- [ ] DS-001 exits at mid-price → use bid/ask exit paths
- [ ] DS-002 label construction flaws → fix before retraining
- [ ] EMA look-ahead bias across train/validation splits → re-tune split
- [ ] Gap/weekend/holiday handling now enforced at ingest (Stage 1) — keep enabled

---

## 10. Suggested Phased Timeline

| Phase | Weeks | Deliverable |
|---|---|---|
| 0 | — | Stage 1 data-ingestion upgrades (✅ done) |
| 1 | 1–2 | Measure bottleneck; run learning-curve + walk-forward baseline |
| 2 | 2–4 | News-sentiment features + point-in-time alignment; fix label bugs |
| 3 | 2–3 | Databento L2/OBI + feature-store chunking; stream ingest spike |
| 4 | 3–6 | Auto-retrain loop + portfolio backtester behind promotion gates |
| 5 | 2–4 | Broker bridge + paper trading, then live with small size |

Next concrete step: **run the Stage 2 bottleneck measurement** (§4) — instrument one training run and record GPU util, wall-clock splits, and learning curves before choosing the next investment of time/money.
