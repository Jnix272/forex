# Full Codebase Audit Report

**Date:** 2026-08-04
**Repo:** `/run/media/jamie/jamie/forex-main` (branch `main`, HEAD `b02ce9c`)
**Version:** 6.5.0
**Scope:** Static analysis, test-suite baseline, core financial-logic correctness, security scan, maintainability.
**Tools:** `ruff 0.16.1`, `uv run pytest`, `git`, grep.

---

## 1. Executive Summary

The pipeline is a broad, production-aware tick-to-trade forex ML stack (data ingestion → features → labels → multi-model
training → RL → backtesting → live execution → promotion gates → risk/drift/alerting → audit manifests). The bulk of
the code is present and the majority of the test suite is green, but the audit surfaced **4 concrete correctness
bugs** in critical financial paths, **2 broken test-collection imports**, and a large body of static-analysis debt
(4041 lint findings) plus structural concerns (a 15,352-line monolithic `train_gpu.py`).

| Area | Status |
|---|---|
| Secrets / credentials | OK — all sourced from env vars |
| Lint (ruff) | 4041 errors, 2576 auto-fixable |
| Type-checking | None configured (no mypy / pyright) |
| Test suite | 1157 passed, 7 failed, 7 errored, 28 skipped (+2 collection errors) |
| Critical correctness bugs | 4 found (see §3) |
| Broken test imports | 2 modules (see §4) |

---

## 2. Static Analysis (ruff)

`ruff check .` against the source tree (excluding `.venv`/`.git`/`__pycache__`):

- **Total:** 4041 errors; **2576 auto-fixable** with `ruff --fix`.
- No ruff config block exists in `pyproject.toml`, so the **default rule set** (E, F, W, I, UP, SIM, PL, B, C, BLE,
  RUF, S, FURB, PIE, TRY, RET, PERF) is applied everywhere — including test files and legacy scripts — producing a
  high-noise baseline.

Top findings by category:

| Count | Code | Note |
|---|---|---|
| 896 | UP006 | Use PEP 585 generics (`dict`/`tuple`/`list`) instead of `Dict`/`Tuple`/`List` |
| 643 | UP045 | Use PEP 604 `X | None` instead of `Optional[X]` |
| 549 | BLE001 | Blind `except Exception` / `except:` (also: **277 broad-except lines** incl. bare `except:`) |
| 329 | I001 | Unsorted / unformatted imports |
| 115 | S110 | `try/except-pass` — silent error swallowing |
| 97  | F401 | Unused imports |
| 92  | F841 | Unused local variables |
| 90  | F541 | f-string without placeholders |
| 45  | C408  | Unnecessary `dict()`/list calls |
| 36  | FURB167 | `regex flag` alias usage |
| 31  | F811  | Redefined-while-unused (likely import duplicates) |

**Type safety:** No `mypy`/`pyright` is installed or wired into `pyproject.toml` / CI. The project relies on
runtime checks (the `sanitize_array`/`sanitize_frame` FiniteGuard) rather than static typing — so latent type
mismatches (e.g. string vs float columns) surface only at runtime (see §3.1).

---

## 3. Critical Correctness Bugs

### 3.1 `sanitize_array` crashes on empty-string / non-numeric data — **HIGH**

`infrastructure/numerics.py:42`
```python
def sanitize_array(arr, *, fill_value=0.0, context="array", clip_range=(-20.0, 20.0)):
    clean = np.asarray(arr, dtype=np.float32).copy()   # ValueError on ''
    bad_mask = ~np.isfinite(clean)
    ...
```
The function's stated purpose is to sanitize NaN/Inf, yet `np.asarray(arr, dtype=np.float32)` **raises
`ValueError: could not convert string to float: ''`** when the input contains empty-string missing values (common in
mixed-type columns from real Dukascopy data). This crashes the training pipeline. It is called in
`training/train_gpu.py:4478, 4526, 4535, 7056` and reproduced in:

- `tests/test_e2e_real_data.py::test_full_pipeline_with_real_data` — `ValueError` at `numerics.py:42`.
- `tests/test_model_full_data_flow.py` — 7 setup errors, same root cause.
- `tests/test_training_smoke.py` — 3 failures, same root cause.

**Fix:** coerce safely before the ndarray cast, e.g.
`arr = pd.to_numeric(pd.Series(arr).ravel(), errors="coerce").to_numpy()` then proceed; or guard with a try/except
that converts non-finite/non-numeric to `fill_value`.

### 3.2 `clean_bad_ticks` fails to cap injected spikes — **HIGH**

`data/data_ingestion.py:167`. Test `test_all.py::test_bad_tick_cleaning_caps_spike` injects a `+0.5` spike at row 100
of a 200-row synthetic stream and expects `clean_bad_ticks(..., z_thresh=5.0)` to replace it with the rolling median
(`abs(out["mid"][100] - out["mid"][99]) < 0.01`). It does not: `mid[100]` survives at `1.5849` vs `1.0849`.

The MAD path (`mad_z_score`) should flag such a deviation, but with `rolling_*` `min_samples=window=60` and a single
outlier in the window the MAD scale collapses toward the (tiny) normal dispersion, which *should* inflate the score.
The spike surviving indicates the outlier condition is not matching — likely the `rolling_median`/MAD expressions
return null at that index (window edge / `min_samples`) or the `spread_outlier` exclusion disqualifies it.

**Impact:** bad-tick cleaning is a data-quality gate for every downstream feature; undetected spikes propagate into
labels and training.

**Fix:** inspect `clean_bad_ticks` with the failing fixture; ensure rolling stats have adequate `min_samples` and the
outlier condition isn't short-circuited by the spread exclusion.

### 3.3 OANDA broker mis-netting of long/short positions — **HIGH**

`trading/live_engine.py:684` (`OANDABroker.get_positions`)
```python
net = long_u + short_u  # OANDA reports short units as negative
```
The code assumes short units are always returned as **negative** strings (e.g. `"-12000"`), which is the real OANDA
convention. `test_review_fixes_smoke.py::test_oanda_env_alias_and_net_short_exposure` provides short units as a
**positive** `"5000"` and expects net `1.5` lots (`20000 - 5000 = 15000`), but the broker computes
`20000 + 5000 = 25000` → `2.5`. The GBP/USD case (short `"-12000"`) passes only because the units happen to be
negative.

**Impact:** If any feed/fixture or OANDA response returns short units as positive, net exposure is reported as
`long + short` instead of `long - short`, **doubling the short-side exposure** — a direct risk-management / position
netting defect.

**Fix:** net robustly: `net = long_u - abs(short_u)`.

### 3.4 `RegimeCV` purge does not enforce fold adjacency — **MEDIUM**

`tests/test_cv.py::test_regime_cv_purge`. `RegimeCV(n_splits=3, regime_labels=..., purge=10)` produces a train sample
at index `353` immediately adjacent to a val fold starting at `354` (gap of `1 < 5` required). The purge is documented
as removing train samples adjacent to the val fold *within each regime*, but the assertion checks global adjacency and
fails, indicating either the purge is applied only intra-regime (leaving cross-regime leakage) or the purge distance
is not honoured at fold boundaries.

**Impact:** potential train→val leakage in regime cross-validation, inflating CV Sharpe estimates.

---

## 4. Broken Test Collection (2 modules)

`pytest --co` fails to import two test modules (documented in `docs/CONTINUE.md` as "pre-existing, unrelated" — they
are not, they are stale API references):

1. `tests/test_system.py:4-6`
   ```python
   from config.models import MODELS, SUPPORTED_SUPERVISED, architecture_config
   from config.settings import MODELS as SETTINGS_MODELS, PATHS, PROJECT_ROOT, project_path
   ```
   `config.settings` no longer exports `MODELS` (it lives in `config.models`, which *is* imported on the line above).
   The second import is a stale re-export reference left over from a refactor.

2. `tests/test_streaming_pipeline.py:15` imports `FeatureState, SymbolState, FeatureSink, RedisFeatureSink,
   PostgresFeatureSink, ConsoleSink, create_sink` from `features.streaming_pipeline`. The **rewritten** pipeline
   (CHANGELOG 2026-08-04) only exports `StreamConfig, MarketTick, ComputedFeatures, parse_tick, tick_to_keyed,
   build_feature_pipeline, create_test_dataflow, StreamingFeaturePipeline`. The tests target the old Bytewax
   sink-based API that the rewrite removed.

Both prevent `pytest tests/` from collecting those modules (collection is **interrupted**, skipping all their tests).

---

## 5. Test Suite Baseline (excluding the 2 broken-collection modules)

`uv run pytest tests/ -q --ignore=tests/test_system.py --ignore=tests/test_streaming_pipeline.py`

```
7 failed, 1157 passed, 28 skipped, 7 errors  (185.5s)
```
Breakdown of the 7 failures + 7 errors:

| Test | Result | Root cause |
|---|---|---|
| `test_all.py::test_bad_tick_cleaning_caps_spike` | FAIL | §3.2 bad-tick spike not capped |
| `test_cv.py::test_regime_cv_purge` | FAIL | §3.4 RegimeCV purge |
| `test_e2e_real_data.py::test_full_pipeline_with_real_data` | FAIL | §3.1 sanitize_array on `''` |
| `test_review_fixes_smoke.py::test_oanda_env_alias_and_net_short_exposure` | FAIL | §3.3 OANDA netting |
| `test_training_smoke.py::test_mini_supervised_smoke_test` | FAIL | §3.1 sanitize_array on `''` |
| `test_training_smoke.py::test_mock_reject_promotion_test` | FAIL | §3.1 sanitize_array on `''` |
| `test_training_smoke.py::test_mock_pass_promotion_test` | FAIL | §3.1 sanitize_array on `''` |
| `test_model_full_data_flow.py` (7 setup errors) | ERROR | §3.1 sanitize_array on `''` |

The 28 skips are data/network-gated (FRED_API_KEY, Stooq rate limits, real-download guards) — expected offline.

**Note:** `tests/test_all.py` is not listed in `pyproject.toml`'s `[tool.pytest.ini_options].markers`, and several
modules (`test_dashboard`, `test_visualizer`, etc.) are legacy integration scripts with network dependencies.

---

## 6. Security

- **No hardcoded secrets / credentials.** A repository-wide grep for credential-like assignments
  (`api_key/apikey/secret/password/bearer/token = "..."`) outside env-var access patterns returned **0 results**.
  `config/settings.py` loads `.env` (`python-dotenv`) and all services read via `os.environ`/`os.getenv`.
- **Risk:** the **277 broad/bare `except` lines** (BLE001/S110) silently swallow errors, which could mask
  security-relevant failures (e.g. broker auth/position-fetch errors returning empty lists instead of raising). This
  is visible in the training logs: `[Sentiment] WARN: init failed (...) — disabling sentiment features` swallowed
  during normal runs.

---

## 7. Maintainability & Architecture

- **Monolith:** `training/train_gpu.py` is **15,352 lines** — the single largest file by a huge margin (next is
  `features/feature_engineering_pl.py` at 1,986). It embeds dataset chunking, training, the auto-tuner, CV, HPO,
  curriculum, multi-task, and promotion-gate logic. It is essentially un-reviewable in one pass.
- **Dual `sanitize_frame` inconsistency:**
  - `infrastructure/numerics.py::sanitize_frame` clips **all** numeric columns to `[-20, 20]` by default.
  - `features/feature_engineering_pl.py::sanitize_frame` has a `_SANITIZE_NO_CLIP` exclusion set for price/volume/
    spread/latency/COT columns (the fix for "ISSUE-002" in the changelog).
  These are two different functions with the same name and different contracts. `labeling/rl_reward_labeling.py:621`
  imports the numerics version — risking clipping of raw price data to `[-20, 20]`.
- **Error swallowing:** 115 `try/except-pass` + 549 blind-except = ~664 sites where failures are hidden, making
  debugging and root-cause analysis hard.
- **Technical debt markers:** 13 `TODO`/`FIXME`/`XXX`/`HACK` markers in non-test source.

---

## 8. Positive Findings (working well)

- **Audit & reproducibility** (`audit/lineage.py`, `audit/manifest.py`) is solid and well-tested
  (`tests/test_audit.py`, 17 tests green): data lineage chain, model-registry records, decision trail, and
  `manifest.json` self-hash + artifact-sha256 verification with tamper detection (`validate_priority2_promotion`
  in `validation/promotion_audit.py` consumes these). This is the strongest architectural subsystem.
- **Risk engine** (`risk/risk_engine.py`, `fx_greeks.py`, `portfolio_monitor.py`) — 45 tests green; comprehensive
  pre/post-trade checks, FX Greeks via Black-76, portfolio aggregation.
- **No-trade-zone, meta-labeling, cross-asset factors, sentiment fusion, multi-task pretraining, curriculum,
  streaming pipeline, execution engine** — all newly added and covered (352+ tests green across the wired-module
  suites in `docs/CONTINUE.md`).
- **Data ingestion** (Stage 1: MAD bad-tick cleaning, info bars, DST sessions, holiday calendar, gap handling, lazy
  parquet) is implemented and tested.
- **4041 lint findings are overwhelmingly style/cosmetic** (PEP 585/604 annotations, unsorted imports, unused
  variables) — not security or correctness — and 2576 are auto-fixable.

---

## 9. Priority Ranked Recommendations

| Prio | Action | Effort |
|---|---|---|
| P0 | Fix `sanitize_array` empty-string crash (§3.1) — blocks real-data training for all models | Small |
| P0 | Fix OANDA position netting `long_u - abs(short_u)` (§3.3) — risk misreporting | Small |
| P1 | Fix `clean_bad_ticks` spike not flagged/capped (§3.2) — data quality | Small |
| P1 | Fix `RegimeCV` purge adjacency guard (§3.4) — CV leakage | Small |
| P2 | Repair the 2 broken test-collection imports (§4) — restore `test_system.py` (16 tests) and `test_streaming_pipeline.py` (28 tests) | Small |
| P2 | Consolidate the two `sanitize_frame` implementations; ensure the numerics path respects `_SANITIZE_NO_CLIP` | Small |
| P3 | Add ruff config to `pyproject.toml` (select/stabilize rule set; exclude legacy scripts) so lint is meaningful | Small |
| P3 | `ruff check --fix` to auto-apply 2576 fixes | Trivial |
| P3 | Wire a type-checker (mypy/pyright) — none configured today | Medium |
| P3 | Split `train_gpu.py` (15,352 lines) into focused modules | Large |

---

*End of report. All findings are reproducible via `uv run pytest` and `uv run ruff check .` from the repo root.*
