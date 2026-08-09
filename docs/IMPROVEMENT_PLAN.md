# Forex ML Pipeline — Improvement Plan (roadmap)

**Date:** 2026-08-03 · **Pointer updated:** 2026-08-06  
**Project:** `/run/media/jamie/jamie/forex-main`  
**Scope:** Personal-use FX research + automated trading (solo developer)  
**Hardware:** RTX 4060 8 GB + 16 GB RAM, Linux

> **Status of what’s done vs open:** [`IMPROVEMENTS.md`](IMPROVEMENTS.md) (canonical).  
> Day-to-day next steps: [`CONTINUE.md`](CONTINUE.md).  
> Config / schema gates: [`CONFIG_CONSISTENCY.md`](CONFIG_CONSISTENCY.md).  
> Historical audits: [`archive/`](archive/).

This file is the **longer roadmap** (data purchases, bottleneck diagnosis, phased timeline). Do not treat §9 checkboxes below as a live backlog — those items were remediated; see IMPROVEMENTS Done.

---

## 1. TL;DR — Biggest remaining bets

| Priority | Work | Why | Effort |
|---|---|---|---|
| Near-term | Session P3/P1/P4 (live limits, SoT, spread names) | Correctness for paper/live | Small–medium |
| Diagnose | GPU util / wall-clock / learning curves / WF P&L | Avoid optimizing the wrong layer | Medium |
| Data | Real news sentiment depth + PIT discipline | Biggest quality lift still available | Medium |
| Scale | Streaming + auto-retrain + portfolio BT | Set-and-forget foundation | Large |

---

## 2. Where the pipeline is today

- **Data:** Dukascopy ticks ~9.4 GB (10 pairs, 2008–2025), eco calendar, COT, historical news CSVs; empty EODHD / LMAX / Myfxbook dirs.
- **Ingestion:** Stage 1 done (MAD cleaning, info bars, DST sessions, holidays, gaps, lazy parquet) — see IMPROVEMENTS Done.
- **Features/Labels:** Polars FE, FFD, TBM + RL rewards; dynamic LH + DST overlap keys; bid/ask exits (DS-001 fixed).
- **Training:** Deep + tabular + RL on RTX 4060; `train_gpu` split; config/curriculum/schema gates live.
- **Execution:** BrokerBridge MT5/IBKR fail-closed via `--broker`; LMAX FIX still open (REST pricing only).

---

## 3. Stage 1 — Data ingestion (DONE)

See [`IMPROVEMENTS.md`](IMPROVEMENTS.md) Done. Optional polish: CLI `--start/--end`, per-pair cache index, schema drift guard.

---

## 4. Stage 2 — Diagnose the real bottleneck

Measure before buying data or training harder:

1. GPU utilisation while training (`nvidia-smi dmon`). If &lt; 80% → data/feature bound.
2. Wall-clock split: ingest → features → labels → fit → eval.
3. Learning curves (train vs val loss/Sharpe).
4. Walk-forward P&L vs random baseline on a holdout year.

---

## 5. Stage 3 — Data quality upsides

1. **Real news sentiment** — wire historical news CSVs deeper into features (delta around high-impact events).
2. **L2 / OBI proxies** — Dukascopy tick imbalance; full depth needs Databento.
3. **Point-in-time** — eco/COT/news available-timestamps (`data/point_in_time.py`).
4. **Chunked feature materialisation** — per-pair/year parquet + stream fold windows (feature store exists).

---

## 6. Stage 4 — Major projects

1. Streaming tick pipeline (poll + compacted parquet + rolling live window).
2. Automated retrain loop with promotion/drift gates.
3. Portfolio backtester (multi-pair, margin, co-integration sizing).
4. Harden live: health-check/reconnect; dry-run → paper → live. LMAX FIX still open.

---

## 7. Data roadmap

| Source | Status | Verdict |
|---|---|---|
| Dukascopy ticks | ✅ present | Keep as raw archive |
| Eco / COT / news CSVs | ✅ present | News still under-used → Stage 3 |
| EODHD / LMAX / Myfxbook | empty dirs | Skip unless needed |
| **Databento** | not subscribed | Best next purchase: L2/MBO + streaming |

---

## 8. Hardware-aware training

- 8 GB VRAM → ≤ ~1M params / batch ≤ 256 / fp16–int8 inference; RL replay on disk.
- 16 GB RAM → lazy parquet + per-fold streaming; never load all pairs of 1s ticks at once.
- Disk plentiful → parquet caches; Linux Zarr FP16 + lz4@1 preferred.
- Train 1 pair or a 3–4 pair basket; full 10-pair ensembles for promotion overnight.

---

## 9. Correctness (superseded checklist)

Historical ISSUE/DS IDs from early audits are **closed or superseded**. Live open work is only in [`IMPROVEMENTS.md`](IMPROVEMENTS.md) Open (session P1/P3/P4 + ops). Archive: [`archive/FIXES_APPLIED.md`](archive/FIXES_APPLIED.md), [`archive/FULL_AUDIT_REPORT.md`](archive/FULL_AUDIT_REPORT.md).

---

## 10. Phased timeline

| Phase | Deliverable |
|---|---|
| 0 | Stage 1 ingestion ✅ |
| 1 | Bottleneck measurement (§4) |
| 2 | News + PIT; label bugs ✅ largely done |
| 3 | Databento / feature-store chunking / stream spike |
| 4 | Auto-retrain + portfolio BT |
| 5 | Paper → small live |

**Phase 3 (2026-08-08) — Architectural Replacements: COMPLETE**
- P3-1: AdversarialGenerator → PGD/FGSM/FreeLB ✅
- P3-2: Curriculum → Composer/Lightning callbacks ✅
- P3-3: Pretraining → lightly-ssl/Solo-learn adapters ✅
- P3-4: RL → CleanRL/SB3 adapters ✅
- P3-5: ONNX scaler fusion (single artifact) ✅

**Next concrete engineering:** session P3 (live limits), then P1/P4 — see [`CONTINUE.md`](CONTINUE.md).
