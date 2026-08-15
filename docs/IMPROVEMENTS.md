# Improvements — Done, Open, Other

**Updated:** 2026-08-09  
**Role:** Single source of truth for improvement backlog + completed work.  
**Not here:** append-only session log → [`SESSION_REPORT.md`](SESSION_REPORT.md); user-facing history → [`../CHANGELOG.md`](../CHANGELOG.md); config gates → [`CONFIG_CONSISTENCY.md`](CONFIG_CONSISTENCY.md).

| Need | Doc |
|------|-----|
| What’s next (short) | [`CONTINUE.md`](CONTINUE.md) |
| Session/DST technical detail | [`SESSION_AUDIT.md`](SESSION_AUDIT.md) |
| Training stage audit | [`TRAINING_PIPELINE_AUDIT.md`](TRAINING_PIPELINE_AUDIT.md) |
| Longer roadmap (data/HW/phases) | [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md) |
| Historical audits | [`archive/`](archive/) |

---

## Done / Fixed

Verified against living docs + spot-checks (`SessionLimitsEnforcer` wired in `live_engine`; slippage uses production session keys via `session_utils`).

### P0 — Critical audit remediations (2026-08-06)

| ID | Fix |
|----|-----|
| CFG-001 | `quick.enabled: false` by default; quick mode warns when it forces ensemble/RL off |
| CFG-002 | YAML `risk:` → `LIVE_RISK` deep-merge; regime keys `crisis/trending/mean_rev/normal` |
| CFG-003 | `use_mixup` / `use_volatility_sampler` mapped + applied in `supervised_loop` |
| LIVE-001 | Missing checkpoint raises (no silent DemoAgent unless `--demo`) |
| LIVE-002 | Base/LMAX `market_order` raise — no fake fills |
| DATA-001/002 | MacroMaterializer + FeatureStore compute wired to cross-asset/materializers |
| FEAT-001 | HMM fit failure logs explicit uniform fallback |
| BT-001 | Numba core handles `SCALE_IN_100` / `SCALE_OUT_100` (actions 5/8) |
| RL-001/002 | PPO LSTM history appends correctly; SL/force-close use avg entry + reward |
| INF-001 | ONNX RL scale-in → HOLD without position; prefer execution ONNX |
| Wire | BrokerBridge → live `--broker mt5\|ibkr`; sidecar YAML → args; audio FinBERT; domain-adapt `fine_tune`/`mmd`/`coral` |

### Phase 3 — Architectural Replacements (2026-08-08)

| ID | Fix |
|----|-----|
| **P3-1** | **Replace AdversarialGenerator with PGD/FGSM/FreeLB** (`training/adversarial_generator.py`): implemented proper gradient-based adversarial attacks (`FGSMAttack`, `PGDAttack`, `FreeLBAttack`); legacy `MarketShockGenerator` retained for backward compatibility; factory `create_adversarial_attack()` with CLI args `--adversarial-method`, `--adversarial-eps`, `--adversarial-alpha`, `--adversarial-steps`; tests in `tests/test_adversarial_generator.py`. ✅ **Wired (2026-08-09, Task E):** `supervised_loop.py` now calls `create_adversarial_attack(...)` (replacing the dead `AdversarialGenerator` init); auto-selects `graph_pgd` for `model_name == "gnn"`; `_prepare_train_batch` dispatches gradient-based attacks vs legacy `market_shock`; `graph_pgd` added to `--adversarial-method` CLI choices. The `supervised_loop.py` epoch loop is also repaired (Task A — see below). |
| **P3-2** | **Migrate Curriculum to Composer/Lightning callbacks** (`training/curriculum_callbacks.py`): framework-agnostic `BaseCurriculum` with pace functions (linear, exp, sqrt, step, log, root); `PLCurriculumCallback` for PyTorch Lightning; `ComposerCurriculumCallback` for MosaicML Composer; `CustomCurriculumAdapter` wraps existing `DifficultyCurriculum`/`SelfPacedLearning`/`CombinedCurriculum`; factory `create_curriculum_callback()`; tests in `tests/test_curriculum_callbacks.py` |
| **P3-3** | **Migrate Pretraining to lightly-ssl / Solo-learn** (`training/pretrain_adapter.py`): unified `BasePretrainAdapter`; `TS2VecAdapter` (time-series SSL); `TNCAdapter` (Temporal Neighborhood Coding); `CustomPretrainAdapter` wraps existing BYOL/Masked/VAE; `LightlySoloAdapter` adapts vision SSL frameworks (lightly-ssl, solo-learn) for 1D time series via 1D CNN backbones; factory `create_pretrain_adapter()`; tests in `tests/test_pretrain_adapter.py` |
| **P3-4** | **Migrate RL to CleanRL / Stable-Baselines3** (`training/rl_adapter.py`): unified `BaseRLAdapter`; `CleanRLAdapter` (PPO/DQN); `SB3Adapter` (PPO/DQN/A2C with Gymnasium); `CustomRLAdapter` wraps existing `PPOAgent`/`DQNAgent`; `GymEnvWrapper` makes `ForexTradingEnv` SB3-compatible; factory `create_rl_adapter()`; tests in `tests/test_rl_adapter.py` |
| **P3-5** | **Export scaler in ONNX graph (single artifact)** (`inference/onnx_inference.py`): `core_onnx_export` and `core_rl_execution_onnx_export` accept optional `scaler` (StandardScaler); `ScaledModel`/`ScaledRLModel` wrappers register mean/scale as ONNX buffers; exported models accept raw features and internally apply z-score normalization + NaN/Inf sanitization; all export paths updated (`export_to_onnx`, `export_ensemble_to_onnx`, `export_rl_to_onnx`, `export_rl_execution_to_onnx`) |

### P3-6 — Per-Model Training Profiles (2026-08-09)

| ID | Fix |
|----|-----|
| **P3-6** | **Auto-applied per-model training profiles** (`config/model_training_profile.py`): central `ModelTrainingProfile` registry defines 12 training dimensions per architecture (adversarial, curriculum, miner feedback, pretraining, SWA/EMA, RL, framework). Auto-detection fallback inspects architecture for unknown models. Applied in `_apply_model_profile()` via `training/model_factory.py::get_model_training_profile()`. Wired into `supervised_loop.py` for adversarial gating (`adversarial_models`), curriculum (self-paced/loss-weighting/miner feedback gating), miner init gating, and SWA. New CLI flags: `--adversarial-models`, `--curriculum-miner-feedback`, `--curriculum-miner-models`, `--curriculum-forgetting-threshold`, `--curriculum-easy-threshold`, `--use-self-paced`, `--use-loss-weighting`, `--self-paced-models`, `--loss-weighting-models`, `--training-framework`, `--pretrain-framework`, `--rl-framework`. Model-specific configs: haelt/tft/transformer get full adversarial+self-paced+miner; mamba gets adversarial+difficulty only; gnn gets graph_pgd+difficulty; expert gets no adversarial/swa/miner. All 6 models compile and validated. |

### P3-7 — Adversarial + Curriculum Coordination (2026-08-09)

| ID | Fix |
|----|-----|
| **P3-7** | **Adversarial epsilon scales with curriculum difficulty** (`training/supervised_loop.py`): after `CurriculumManager.update()`, `difficulty_level / max_level` ratio scales `PGDAttack.eps` via new `set_eps()` method. `--adversarial-eps-curriculum-scale` flag enables. GNN's `GraphAdversarialAttack` gets dual `set_eps()` / `set_edge_eps()` for node features vs edge dropout. |

### P3-8 — Pretrain Hard Examples → Adversarial (2026-08-09)

| ID | Fix |
|----|-----|
| **P3-8** | **Feature vulnerability from pretraining hard examples** (`pretrain/hard_example_mining.py`): `compute_feature_vulnerability()` computes per-dimension gradient-norm scores from hard indices, saves to `logs/hard_feature_dims.json`. `AdversarialAttack` base class gets `feature_eps_multipliers` (via `_get_effective_eps()`), `PGDAttack` and `GraphAdversarialAttack` apply per-dimension eps scaling. Supervised training loads vulnerability and passes to attack constructor. |

### P3-9 — PGD Hardening (2026-08-09)

| ID | Fix |
|----|-----|
| **P3-9** | **PGD improvements** (`training/adversarial_generator.py`): L2 gradient normalization (`normalize_grad=True`), gradual attack step warmup (`warmup_steps`), per-dimension epsilon multipliers (`feature_eps_multipliers`). Same hardening applied to `GraphAdversarialAttack`. Factory `create_adversarial_attack()` accepts new params. CLI: `--adversarial-normalize-grad`, `--adversarial-warmup-steps`. |

### P3-10 — ONNX Scaler Verification (2026-08-09)

| ID | Fix |
|----|-----|
| **P3-10** | **ONNX scaler fusion validation** (`inference/onnx_inference.py`): `verify_onnx_scaler()` exports model with fused scaler, runs identical random inputs through PyTorch `ScaledModel` and ONNX Runtime, asserts `allclose(rtol=1e-4)`. CLI subcommand: `python inference/onnx_inference.py verify --checkpoint ... --model ... --seq-len 60`. |

### P3-11 — Unified Monitoring System (2026-08-09)

| ID | Fix |
|----|-----|
| **P3-11** | **Unified monitoring system** (`monitoring/`): Complete observability stack replacing fragmented logging/checking. **Components:** `events.py` (unified event schema with 7 types), `event_bus.py` (async priority queue, deduplication, SQLite persistence, backpressure), `unified_logger.py` (single entry point replacing train_logger/sidecar/logging_utils, supports LOG/CHECK/ALERT/METRIC/CHECKPOINT/HEARTBEAT/PROGRESS), `checks/` (24 built-in: NaN/Inf, grad norm, loss plateau, embedding collapse, checkpoint validation, data drift PSI/KS, GPU/CPU/disk resources), `alerts/engine.py` (10 rules: NaN storm, OOM storm, grad explosion, loss divergence, val Sharpe collapse, LR too low, data drift, checkpoint load fail, circuit breaker, model demoted; rate limiting + multi-channel dispatch), `dashboard/app.py` (FastAPI + WebSocket live dashboard with Chart.js metrics, real-time event log, check results, system resources). **Integration:** `monitoring/__init__.py` single import; replaces `train_logger.py`, `sidecar.py`, `logging_utils.py`, `health_check.py`; 24 checks registered across 7 phases; alert rules with rate limiting + multi-channel (console, Discord, email, PagerDuty); web dashboard at `python -m monitoring.dashboard.app` with Chart.js metrics visualization, real-time event log, check results, system resources. All modules compile clean, integration tests pass. |

### P3-12 — Data Pipeline Audit & Fixes (2026-08-09)

| ID | Fix |
|----|-----|
| **P3-12** | **Data pipeline audit & fixes** (`training/dataset_builder.py`, `training/gpu_datasets.py`, `labeling/rl_reward_labeling.py`, `labeling/triple_barrier_labeling.py`, `config/feature_mask.py`): Deep audit of 14 files found 21 issues (2 Critical, 4 High, 10 Medium, 5 Low). 9 fixed: (C1) Multi-pair Zarr arrays created at full shape → changed to `shape=(0,)+dims` for safe `.append()`. (C2) `DataQualityReporter` class never instantiated → wired into `build_dataset_chunked()` to generate `data_quality_report.json`. (H2) Hardcoded 1.5 pip direction label threshold ignores per-bar session cost → now uses `tx_pips_arr` computed per-entry from session/slippage multipliers. (H3) `sanitize_array` hard-clips features to [-20,20] arbitrarily → disabled clip_range for features (NaN/Inf handled by col_medians), labels clip to [-50,50]. (H4) No scaler shape validation at DataLoader load time → added `n_features_in_` check before `.transform()` in `ZarrStreamDataset._decompress_block`. (M7) Feature mask allowlist missing `mid`, `spread`, `asia_london`, `london_ny`, `time_idx` → added 5 columns. (L1) Triple barrier sequential fallback passes `close` instead of `exit_long_path`/`exit_short_path` → fixed. (L4) Dead expression `tx_cost_pips * pip_size` → removed. H1 (scaler identity passthrough) confirmed intentional — per-fold scaling happens in `_fit_fold_scaler()`.

### P3-13 — Type/Shape/Logic Bug Fixes (2026-08-10)

| ID | Fix |
|----|-----|
| **BUG-001** | **`gradient_norm` avg_norm calculation fixed** (`monitoring/checks/gradient_norm.py:51`): Changed from meaningless `total_norm / param_count` to mean of individual parameter gradient norms. Updated threshold keys from `warn`/`crit`/`vanish` to `grad_norm_warn`/`grad_norm_crit`/`grad_norm_vanish` for config consistency. |
| **BUG-002** | **`data_drift` scipy import moved to module level** (`monitoring/checks/data_drift.py:14-19`): Moved `from scipy import stats` from inside `ks_statistic()` function to module level with `try/except` and `SCIPY_AVAILABLE` flag for graceful fallback. |
| **BUG-003** | **`checkpoint_load` optimizer state check fixed** (`monitoring/checks/checkpoint_load.py:156-169`): Added optimizer-type-aware validation (Adam → `exp_avg`/`exp_avg_sq`, SGD → `momentum_buffer`, RMSprop → `square_avg`/`momentum_buffer`). Eliminates false positives where SGD was flagged for missing `exp_avg`. | |

### P1 — Unwired / mismatch remediations (2026-08-06)

| ID | Fix |
|----|-----|
| CFG | LABELING ATR synced 1.2/0.8; feature windows wired; `max_bad_frac`/`max_zero_frac` → sequence quality; `feature_cache.hurst`→`hurst_exponent`; pretrain/maturity synced |
| ENS-001 | CE `(B,3)` logits → buy−sell scalar |
| BT-002 | MC Sharpe: default unannualized for per-trade P&L; percentiles wired |
| BT-003 | `AdvancedBacktestEngine.run` executes via LOB + market fills |
| INF-002 | Inference `torch_load_safe` fallback; live `--seq-len` defaults to training 80 |
| FEAT-002 | FinBERT placeholder / prefetch errors logged; Kafka news uses real FinBERT CLS pool |
| TRAIN | `--rl-use-sharpe-reward` / `--rl-use-her`; HyperBand/BOHB `suggest_params` samples search space |

### P2 — Stubs / docs / rare paths (2026-08-06)

| ID | Fix |
|----|-----|
| DOC-001 | GOVERNANCE / ALERTS / MACRO_DATA comments match live usage |
| BT-003 leftover | Real `compute_slippage` + `SlippageDecomposition.from_execution` |
| ARCH-001 | MambaBlock documented as gated residual (`mamba_gated_v2`) |
| CFG-007 | TRAINING stubs synced; `patience: 3` |
| LIVE rare | LMAX REST login + order book for pricing (orders still FIX-gated) |
| TRAIN docs | `--off-policy-rewards` diagnostic-only |

### Dataset / labeling mismatches (2026-08-06)

- Column names: `session_label`, `regime_class`, `expected_latency_ms` (+ FEATURE_MASK)
- Integer→string regime mapping for barrier scale
- Inline `mean_rev` horizon_mult aligned to `LABEL_REGIME` (0.5)
- `enable_no_trade_zones` default True

### ZarrStreamDataset refactor + builder/reader contract (2026-08-07)

Nine fix points in `training/gpu_datasets.py::ZarrStreamDataset` plus a new dataset-builder/reader contract validator wired into the post-build integrity check. Closes the silent staleness that allowed a bad writer to ship a cache the reader would paper over with synthetic values.

| ID | Fix |
|----|-----|
| ZS-1 | Yield batched sub-blocks (one `rng.permutation` walk per block, no per-row `float(...)` scalar copies) |
| ZS-2 | Cross-chunk shuffle buffer (`shuffle_buffer_size` default 8192) breaks temporal autocorrelation on FX |
| ZS-6 | `pq` fallback corrected to `1.0` (matches `dataset_builder._sidecar_or_default`); legacy `min(1,\|y\|)` conflated path-quality with absolute return |
| ZS-8 | `np.array_split` for worker sharding — no silent tail-worker empty yields, `len(loader)` no longer over-counts |
| ZS-9 | `shuffle=True` guard: rely on PyTorch's native `IterableDataset` ValueError (already informative) |
| ZS-10 | `wrap_loader_prefetch` skips daemon-thread overlay when `num_workers > 0` (was double-buffering ~50 GB pinned); opt-back-in via `--force-thread-prefetch` / YAML `hardware.force_thread_prefetch: true` |
| ZS-11 | Zarr handle dict keyed by `(worker_id, cache_path)` — no cross-epoch leak on `persistent_workers=True` |
| ZS-12 | Confirmed `y_cls`/`pq` already written into the zarr group by `dataset_builder._append_chunk`; NPY fallback branch marked legacy |
| ZS-13 | Per-worker `np.random.default_rng` via `SeedSequence(shuffle_seed, spawn_key=(worker_id,))`; legacy `np.random` was fork-inherited → identical shuffles in every worker |
| RC-* | New `_validate_dataset_builder_reader_contract` (6 value/shape checks) wired into `_postprocess_cache_integrity_check` so bad writes fail-stop at build time |

Tests: `tests/test_zarr_stream_dataset.py` (14), `tests/test_dataset_builder_reader_contract.py` (10), updated `tests/test_zarr_prefetch.py` (split the "always wraps" test into single-process / workers-present / force).

### Dynamic LH + DST sessions — labeling P2 (2026-08-06)

| Area | Change |
|------|--------|
| Label horizon | `resolve_horizon` / `max_label_horizon_mult` — base LH × regime × `session_horizon_mult` × spread_z |
| Session cost | `resolve_session_cost` / `resolve_session_key` prefer DST flags `asia_london` / `london_ny` |
| Embargo | CV floor uses `base_LH * max_label_horizon_mult()` |
| Ingestion / FE | DST `_apply_dst_sessions` overlap floats; FE prefers bars `london_ny` |
| Cache | LABEL_REGIME digest `lr*` in cache tag |
| Labels | Defaults LH=30 / TP=1.2 / SL=0.8 (settings ↔ YAML ↔ scalping) |
| Ubuntu | `run_ubuntu.yaml` = synchronized with `run.yaml` for production scale + HW overlays |
| Backtest | Numba path ≥50k bars; equity-curve metrics when trades empty |
| Fix-all | `numpy<2.5` (Numba); smoke CUDA isolation |

Closest checks: `test_purged_cv_embargo` + `test_curriculum_audit` + `test_config_consistency` + DST ingestion → **73 passed**.

### Session SoT + live limits + fill mult (P1 / P3 / P4) (2026-08-06)

| ID | Fix |
|----|-----|
| **P1** | `trading/session_utils.classify_session` DST SoT; risk uses `london_ny` (not `"overlap"`); aliases normalize |
| **P3** | `LiveTradingEngine` calls `SessionLimitsEnforcer.check` before orders; `asia_london`/`london_ny` in `session_limits` |
| **P4** | Shared `session_spread_mult` (= `LABEL_REGIME.session_cost_scale`); `SlippageCalibrator` production keys; backtest synthetic fill path uses mult |

Focused: `tests/test_session_sot_p1_p3_p4.py` + `TestSessionLimitsEnforcer`.

### LMAX FIX Integration (2026-08-06)

| ID | Fix |
|----|-----|
| **EXEC-001** | `quickfix` added to requirements; `execution/lmax_fix_app.py` created for Logon/Logout/ExecutionReports. |
| **EXEC-002** | `LMAXBroker` natively establishes `quickfix` connection via `LMAX_FIX_CONFIG` and formats `NewOrderSingle` FIX payloads for market orders. |

### Performance (2026-08-05)

1. Training: removed per-batch GPU sync; batch 512 / accum 4 / workers 8
2. Regime features: vectorized vol + Numba hurst/fractal
3. Backtest: Numba core (~15–20× on 100K bars)
4. Streaming: Redpanda `--smp 2` / `--memory 2G`

### Earlier closed (summary)
- **Training pipeline audit** top-5 + follow-ups (DS-001 bid/ask exits, CatBoost shell, embargo floor, promotion fail-closed, tabular purged CV, pretrain guardrails, stage timings/GPU util, BrokerBridge fail-closed + latency) — see [`TRAINING_PIPELINE_AUDIT.md`](TRAINING_PIPELINE_AUDIT.md)
- **`train_gpu` split** (~15k → ~2.2k lines) + config/curriculum/schema gates
- **Cache/loop perf:** Polars chunk path, FP16 Zarr X, fused AdamW, Linux lz4@1
- **Model/RL/pretrain batch:** critical bugs, architecture upgrades, HER/DQN masks, domain adapt, volume profile + vol clock features
- **Stage 1 data ingestion** (MAD ticks, info bars, DST sessions, holidays, gaps, lazy load) — done
- **Full suite (2026-08-04):** 1247 passed, 28 skipped (env), 0 failed
- Historical audit dumps → [`archive/`](archive/) (incl. `FIXES_APPLIED.md`, `FULL_AUDIT_REPORT.md`)

---

## Data Quality & Observability (2026-08-09)

| ID | Fix |
|----|-----|
| **OBS-001** | **Structured logging infrastructure** (`infrastructure/logging_utils.py`): `log_data_load()`, `log_feature_build()`, `log_training_step()`, `timed_load()` context manager. Grep-able format: `[timestamp] forex.data LEVEL \| key=value ...` |
| **OBS-002** | **Cross-asset per-asset logging** (`data/cross_asset.py`): Every asset now logs provider attempted, row count, status. Logs fallback chain (Stooq → Yahoo → FRED → EODHD) with `cross_asset_provider` + `cross_asset` entries. |
| **OBS-003** | **FRED dual-path visibility** (`features/macro_features.py`): `MacroYieldFeatureBuilder.load_yields()` logs `status=success` with `note=real=N, synthetic=M` or `status=fallback_synthetic` with `note=no FRED_API_KEY`. Previously two independent silent FRED calls. |
| **OBS-004** | **COT load unification** (`training/dataset_builder.py::load_cot()`): Single helper used by main path + parallel workers. Logs row count + status on every call (was: main logged, worker silent). |
| **OBS-005** | **Regime detection structured logging** (`features/feature_engineering_pl.py`): `detect_regimes_polars` (HMM + Hurst + fractal) logs success vs fallback to vol-bucket with `log_feature_build()`. |
| **OBS-006** | **Pipeline standardization fix** (`data/data_ingestion.py`): `ForexDataPipeline.run()` now calls `_standardize_dataframe()` before resampling, ensuring `mid`/`spread` derived columns exist. Fixes `ColumnNotFoundError: unable to find column "mid"`. |
| **OBS-007** | **Resampler robustness** (`data/data_ingestion.py::resample_to_bars()`): Ensures `mid`/`spread` exist before group_by_dynamic; JPY spread cap (3× median) applied before aggregation. |
| **OBS-008** | **News sentiment null handling** (already in code): `historical_news.py` drops null sentiment rows before aggregation (no `fill_null(0.0)` dilution); `feature_engineering_pl.py` fills null with 0.0 (neutral) before decay. |

### Verification
End-to-end pipeline test on compact daily EURUSD 2008-01-01..2008-01-08:
- Loaded 448,234 ticks from compact cache
- Pipeline: 3,300 bars, 4,885 bad ticks cleaned (MAD z>6.0)
- FeatureEngineer: **248 features** built in 1.02s
  - Regime: 18 cols (HMM state probs 3, Hurst R/S & DFA, fractal dim, regime_label/class, persistence)
  - Sentiment: 4 cols (raw, decayed, missing, staleness)
  - Macro: 30+ cols (9 spreads, 9 carries, yield curve, momentum, vol, gate weights)
- Structured logs emitted for:
  - `cross_asset_provider` (25 assets × provider chain)
  - `fred_yields` fallback_synthetic (no FRED_API_KEY)
  - `regime_detection` success (HMM + Hurst + fractal)

---

## In progress / just landing

**2026-08-09 — Curriculum / Adversarial / EMA consolidation (Improvements #1–4) — PARTIAL.** Do not re-verify as done; loop repair comes first.

| Item | Status |
|------|--------|
| `CurriculumManager` as the curriculum authority in `supervised_loop.py` (`_curriculum_mgr` via `create_curriculum_manager`) | ✅ **Done (Task A)** — legacy adaptive-curriculum loop body fully removed (schedule floors, stage gating, Sharpe-EMA window, all history/TB curriculum logs, resume-ckpt `curriculum_state`, control-report `adaptation_config`); advancement now flows entirely through `CurriculumManager.update(ep, losses)`. Per-sample weights **applied to the loss** via `_apply_curriculum_weights` (Task F). |
| `OnlineHardExampleMiner` as the in-loop miner | ✅ **Done (Task B)** — offline `HardExampleMiner` class removed; docstring refs + dead re-export name cleaned. |
| Graph-aware adversarial training (`GraphAdversarialAttack`, "graph_pgd") | ✅ **Done (Task E)** — `supervised_loop.py` auto-selects `graph_pgd` for `model_name == "gnn"` via `create_adversarial_attack`; `_prepare_train_batch` dispatches gradient-based vs legacy `market_shock`; `graph_pgd` in CLI choices. |
| Per-model `pretrain_method` in `_member_training_args` (Improvement #1) | Done |
| **`training/supervised_loop.py` epoch loop** | ✅ **Repaired (Task A)** — all legacy adaptive-curriculum init/body refs stripped; loop runs through `CurriculumManager`. `OneCycleLR` switched to `total_steps` mode (Task G) so curriculum-filtered epochs don't desync the LR cycle. Training smoke + 232 tests pass. |

Nothing else mid-flight — P1/P3/P4 landed (see Done above).

---

## Open / Other

### Optional / ops

1. Rebuild training Zarr on Linux (FP16 X + lz4@1) before long runs — `lr*` / mask digests invalidate on LABEL_REGIME or FEATURE_MASK edits
2. Paper + promote a model; non-paper live needs `promotion_gate.json`
3. Set `FRED_API_KEY` (or fix Stooq) to exercise ~14 yield/cross-asset skips
4. Exercise `--broker mt5` / `ibkr` against a paper terminal

### Longer roadmap (not immediate backlog)

Measure bottleneck → news sentiment depth → Databento L2 → streaming retrain → portfolio backtester → harden live. Full narrative: [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md).

Residual noted in training audit: hub KD ignores `distill_temperature` (MSE). Optional Stage-1 ingestion polish: CLI `--start/--end`, per-pair cache index, schema drift guard.

---

## Suggested next (copy into CONTINUE)

1. ✅ **Repair `training/supervised_loop.py` epoch loop** — DONE (Task A); training smoke + 232 tests pass.
2. ✅ **Finish the partial consolidation** — DONE: legacy adaptive-curriculum loop body removed (Task A); `HardExampleMiner` docstring/dead-ref cleaned (Task B); `graph_pgd` auto-select wired (Task E); sample weights applied to loss (Task F); `OneCycleLR` `total_steps` mode (Task G).
3. **§9.2 P1 design-gaps** (see [`FIXES.md`](FIXES.md)): backtest execution realism (#16-18), conformal `main_logits` (#19), macro leakage test (#20), numerics clip config (#22), news pipeline init-load (#23), per-fold feature-quality (#25), portfolio session limits (#27), kill-switch design (#28), grad checkpointing (#30), TrainingController crash-ckpt (#31), scaler checksum (#32), miner temporal-separation test (#35).
4. **§8 remaining factory wiring** (opt-in): `create_curriculum_callback` / `create_pretrain_adapter` / `create_rl_adapter` exist + unit-tested but not called in production (P3-2/P3-3/P3-4). `create_adversarial_attack` IS wired (P3-1).
5. Rebuild Zarr + paper-promote when ready to train/trade
6. Optional: Numba backtest path honor `session_spread_mult`
7. Exercise `--broker mt5` / `ibkr` against a paper terminal
\n- **Pretraining Method Unification**: Centralized dual-alias resolution (e.g., `masked_or_byol`) into a single `pretrain_method_for` resolver to eliminate silent fallbacks to BYOL.
