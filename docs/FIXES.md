# FIXES — verified 2026-08-09 code-vs-docs audit

Every item below was confirmed against the working tree. Line numbers = `training/supervised_loop.py` unless noted.

## 1. 🔴 CRITICAL — repair `supervised_loop.py` epoch loop (broken)

The adaptive-curriculum init blocks were deleted but the loop body still references them → `NameError` on the first training epoch.

**Used-before-def names (first reference):**
| Name | First ref | Purpose |
|---|---|---|
| `_sched_floor_seq` | :2363 | sequence-length schedule floor (call) |
| `_active_seq_len` | :2364 | current seq-length stage gate |
| `_seq_frozen` | :2364 | stage frozen flag |
| `_sched_floor_diff` | :2374 | difficulty-stage schedule floor (call) |
| `_active_diff_stage` | :2375 | current difficulty stage gate |
| `_last_logged_seq_len` | :2387 | logging bookkeeping |
| `_difficulty_stage_for_epoch` | :2536 | deleted helper — still called |
| `_rolling_sharpes` | :2691 | Sharpe-EMA window |
| `_adapt_ema_alpha` | :2880 | EMA alpha (its init was deleted) |
| `_adapt_recovery_window` | :2911 | EMA recovery window |
| `_adapt_min_epochs_per_stage` | :2929 | per-stage min epochs |

**Chosen fix (recommended in `SESSION_REPORT.md`):** strip the loop body's remaining legacy references and route fully through `CurriculumManager` (already initialized at :2335-2353 via `create_curriculum_manager`). Do NOT `git restore` the whole file — the tree carries large unrelated uncommitted changes.

**After repair, legacy vars to stop writing to history/TB:** `_sharpe_ema`, `_epochs_since_advance`, `_post_stall_stable_count`, `_curriculum_stalls`.

## 2. Wire `graph_pgd` auto-select (Improvement #3, P3-1)

- `GraphAdversarialAttack` exists in `training/adversarial_generator.py` (factory key `"graph_pgd"`; node-feature PGD + edge dropout rate 0.1). `GNNFromSequence.forward(x, adj=None)` (:1097) and `GNNCrossAsset.forward(x, adj=None)` (:1041) accept `adj`.
- **Not wired:** `supervised_loop.py` has zero hits for `graph_pgd` / `"gnn"` / `adversarial_method`; loop still inits legacy `AdversarialGenerator` at :2306-2318.
- Add: `model_name == "gnn"` → `adversarial_method = "graph_pgd"`, and pass `adj` through to the attack/model.

## 3. Apply `CurriculumManager` sample weights to the loss

- Inclusion masks ARE applied (:2480-2498). Sample weights are computed and logged to history (:2503) but **never applied** to the loss/batches.
- Decide: apply weights in the loss, or drop the misleading "weights" logging.

## 4. OneCycleLR stale `steps_per_epoch` with curriculum

- Scheduler built once at :1882 from initial `len(train_dl)`. Curriculum rebuilds the DataLoader with filtered indices each epoch → cycle length desyncs from actual optimizer steps. No length-change detection/recreation exists.
- Fix: rebuild `OneCycleLR` (or switch to `WarmupCosine`) whenever `len(train_dl)` changes.

## 5. Clean leftover `HardExampleMiner` references (cosmetic)

Offline class is removed (online-only achieved). Remaining leftovers:
- `training/hard_example_miner.py:20` and `:153` — docstring examples referencing `HardExampleMiner`.
- `training/supervised_loop.py:126` — dead `'HardExampleMiner'` name in `_HOST_DEPS` re-export.

## 6. Docs that no longer match code (already annotated in CHANGELOG/SESSION_REPORT)

- M4/S1: `validate_epoch` returns numeric `0.0`, **no** `None` sentinel; `best_sharpe` inits `-inf`, **no** `early_stop_min_sharpe` arg. Early-stop logic is plain `v_sh > best_sharpe + min_delta` (:2845).
- OneCycleLR length-change fix and OHEM `'wrong' in locals()` fix: documented but **not in the code**.

## 7. Integration gaps between training subsystems

Verified gaps where separate mechanisms don't coordinate. (Items 1, 4, 7.2 are genuine design gaps; 7.3 is a factual description of independent mechanisms.)

### 7.1 Adversarial + Curriculum — no coordination
- Perturbation is applied **uniformly** per batch in `_prepare_train_batch` (`supervised_loop.py:943-948`) with the same stochastic gate (`AdversarialAttack.forward`, `adversarial_generator.py:65-73`). The curriculum difficulty stage (sample inclusion, seq-len, feature masks) never reaches the adversarial module → same attack strength in easy and hard stages.
- Fix option: make attack strength/`probability` stage-aware (scale `eps`/`alpha` by the difficulty stage or feed the current stage into the attack).
- **Note:** currently dead — `_adversarial` is always `None` (dead `AdversarialGenerator` name, item 2).

### 7.2 Hard examples → adversarial generation (and pretraining) — missing
- `PretrainHardExampleMiner` (`pretrain/hard_example_mining.py:12`) is **dead code**: `ingest_validation_failures` / `sample_batch_indices` have zero callers. Hard examples are NOT actually injected into pretraining either.
- `OnlineHardExampleMiner` only resamples the supervised dataloader (`get_oversampled_indices`, `supervised_loop.py:2537-2538`); the adversarial step sees no hard/forgotten info to focus its budget on.
- Fix option: wire the pretrain miner into the pretraining pipeline; optionally pass the miner's hard/forgotten masks to the adversarial module.

### 7.3 SWA + EMA — no shared state (fact, not necessarily a bug)
Three independent mechanisms touching different objects:
- SWA averages model **weights** (`AveragedModel`/`SWALR`, `supervised_loop.py:2651-2668`, BN-fix :3106-3116).
- Feature-stat EMA (decay 0.90) tracks **input feature** mean/std → drives curriculum feature-mask zeroing (:244-303, sampled :2420).
- Sharpe EMA (`_sharpe_ema` / `_adapt_ema_alpha`) tracks **validation signal** for curriculum advancement (:2880-2900).

None reads another's state (e.g., drift stats unused at SWA start; SWA start doesn't reset the EMAs). Decide whether coordination is desired. **Note:** currently moot — `_adapt_ema_alpha` init is deleted (item 1).

### 7.4 Online miner → curriculum — forgetting doesn't inform advancement
- The miner's output feeds **only** the next epoch's dataloader sampling (:2537-2538). `ForgettingTracker` / `get_forgotten_mask` (`hard_example_miner.py:193, 230-232`) are never read for curriculum purposes.
- Curriculum advancement is driven solely by Sharpe-EMA / rolling-sharpe plateau logic (`_rolling_sharpes`, `_difficulty_stage_for_epoch`, `_sharpe_ema`) → a model that starts forgetting won't slow/pause/revert the curriculum.
- Fix option: feed miner forgetting rate (or hard-sample ratio) into the curriculum as a stall/pause signal.

## 8. Phase 3 "Complete" — structurally true, but NOT wired (verified 2026-08-09)

Claimed five "Phase 3 Complete" tasks were checked against the tree. Verdict: **all files/classes/factories/tests exist and the 4 Phase-3 test suites pass (58 passed, 2 skipped — `lightly` not installed)**. The word **"Complete" is overstated**: none of the four factories is called anywhere in the production pipeline — they exist only as standalone modules exercised by their own unit tests. (`create_*` factories have zero callers outside their defining module and tests.)

| Claim | Verified reality |
|---|---|
| 1. Adversarial → PGD/FGSM/FreeLB (`training/adversarial_generator.py`; `create_adversarial_attack`) | ✅ Classes/factory/tests exist. ⚠️ **Not wired**: `--adversarial-method` CLI arg exists (`gpu_cli.py:1297`) but `supervised_loop.py` never calls the factory — it still inits the dead `AdversarialGenerator` name (`supervised_loop.py:2322`). `GraphAdversarialAttack` also untested. |
| 2. Curriculum → Composer/Lightning callbacks (`training/curriculum_callbacks.py`; `create_curriculum_callback`) | ✅ All classes/factory/tests exist. ⚠️ Factory **not used** in the training pipeline. |
| 3. Pretraining → lightly-ssl/Solo-learn (`training/pretrain_adapter.py`; `create_pretrain_adapter`) | ✅ All adapters/factory/tests exist (2 tests skip w/o `lightly`). ⚠️ Factory **not used**; `PretrainHardExampleMiner` also dead (§7.2). |
| 4. RL → CleanRL/SB3 (`training/rl_adapter.py`; `create_rl_adapter`) | ✅ All adapters/`GymEnvWrapper`/factory/tests exist. ⚠️ Factory **not used** in the pipeline. |
| 5. Scaler fused into ONNX (`inference/onnx_inference.py`) | ✅ **Fully wired.** `ScaledModel`/`ScaledRLModel` register `mean`/`scale` buffers + `nan_to_num` in `forward`; `core_onnx_export`/`core_rl_execution_onnx_export` accept `scaler`; `export_to_onnx`/`export_ensemble_to_onnx`/`export_rl_to_onnx`/`export_rl_execution_to_onnx` load it from `scaler.npz` via `load_inference_scaler` and pass it through. |

Fix options: wire the four factories into the pipeline (supervised_loop adversarial path → `create_adversarial_attack`; curriculum callbacks; pretrain runner → `create_pretrain_adapter`; RL training → `create_rl_adapter`), or downgrade the "Complete" claim to "modules created + unit-tested, integration pending".

---

## Verification after fixes

1. Smoke test the training pipeline (loop must reach epoch 2+ without `NameError`).
2. Re-verify Phase 3 wiring: P3-1 `graph_pgd` for `model_name == "gnn"`, P3-2 curriculum callbacks, P3-3 pretraining adapters (§8).
3. `py_compile` all of `training/`; run `pytest tests/ -q` subset touched by the loop.
4. Update `CHANGELOG.md` / `SESSION_REPORT.md` / `IMPROVEMENTS.md` statuses and refresh line numbers.
5. Re-assess the §7 integration gaps only after the loop is repaired (several are currently masked by the epoch-1 `NameError` and the dead `AdversarialGenerator`).

---

## 9. DEEP CODEBASE AUDIT — Additional Critical & High Issues (2026-08-09)

Comprehensive audit of 50+ files across 10 subsystems. **15 Critical (P0), 20 High (P1), 15 Medium (P2)** issues found.

### 9.1 P0 — Critical (Must Fix Immediately)

| # | Issue | File:Line | Description |
|---|-------|-----------|-------------|
| 1 | **GPU Backtester 2-Bar Look-Ahead Bias** | `backtesting/gpu_backtester.py:39-42` | `d_positions = d_signals[:-2]` paired with `d_returns = d_returns[1:]` — signal[i] uses return[i+1→i+2] (2-bar forward). Fix: `d_signals[:-1]` / `d_returns[1:]`. |
| 2 | **Meta-Labeler In-Sample Leakage** | `labeling/triple_barrier_meta.py:285-297` | Optuna trials call `primary_pred_fn(bars, features)` on FULL data, then run TBM. Meta-model trains on predictions with full future access. Fix: temporal split first, fit primary on train only. |
| 3 | **RL Reward Tail-Cut Mismatch** | `labeling/rl_reward_labeling.py:315 vs 746` | Base cuts `lookahead_bars + delay`; regime function uses `max(lookahead_bars, lookahead_bars * mult) + delay` → ~23 fabricated zero rows/chunk. |
| 4 | **HMM Look-Ahead in Regime Detection** | `features/regime_detection.py:347-352, 409-415` | `state_probs` property computes on FULL series using future data despite causal warm-start. Fix: decode with frozen params only. |
| 5 | **LMAX close_position Unimplemented** | `trading/live_engine.py:677` | `raise NotImplementedError("LMAXBroker.close_position requires position tracking")` — cannot flatten positions in emergency. |
| 6 | **XGBoost Scaler UnboundLocalError** | `inference/pytorch_inference.py:116` | Returns `scaler` in xgboost branch but `scaler` never defined. Fix: `scaler = None` or load properly. |
| 7 | **RL UPnL Wrong Lot Size (10k vs 100k)** | `inference/rl_inference.py:179-183` | `upnl = (price - entry_price) * position * 10_000.0` — standard FX lot is 100,000. |
| 8 | **RiskEngine Position Size Inconsistency** | `risk/risk_engine.py:168-170` | Uses caller `position_size_pct` for one check but computes `total_notional/equity` for another — inconsistent reference. |
| 9 | **PortfolioVaR Standalone VaR for Sizing** | `risk/execution.py:366` | Uses single-asset VaR for position sizing instead of portfolio contribution VaR. |
| 10 | **OneCycleLR Stepped Per-Optimizer-Step** | `training/supervised_loop.py:1894-1901, 904` | Scheduler stepped every optimizer step but OneCycleLR expects per-batch. LR cycle completes too fast. |
| 11 | **DataParallel Used Instead of DDP** | `training/supervised_loop.py:1841` | `nn.DataParallel` deprecated, GIL bottleneck. Migrate to `DistributedDataParallel`. |
| 12 | **RedPanda Single-Node + Overprovisioned** | `infrastructure/docker-compose.streaming.yml:10-13` | `--memory 2G --overprovisioned` defeats limits, no replication — data loss on crash. |
| 13 | **CombCV Embargo Logic Inverted** | `validation/cv.py:63-74` | Keeps indices OUTSIDE embargo window instead of removing them. |
| 14 | **sys.stdout Hijacking Breaks Libraries** | `monitoring/train_logger.py:317-318` | Global `sys.stdout = StreamToLogger(...)` breaks tqdm, other libs. |
| 15 | **ONNX Scaler Fusion Wrong Order** | `inference/onnx_inference.py:389-394` | Applies `nan_to_num` AFTER scaling — should sanitize BEFORE scaling. |

### 9.2 P1 — High Priority (Fix Soon)

| # | Issue | File:Line | Description |
|---|-------|-----------|-------------|
| 16 | Stop-Loss Closes Entire Position Ignoring Partial TP | `backtesting/backtest.py:480-535` | Numba path doesn't track partially closed state. |
| 17 | Stop Execution Ignores Spread Crossing | `backtesting/backtest.py:397-420` | Uses fixed slippage pips, ignores spread on stop exit. |
| 18 | Market Impact Applied to Limit Orders | `backtesting/backtest.py:267-271` | Impact added for both entry AND exit, doesn't distinguish order type. |
| 19 | Conformal Calibration Leakage | `features/no_trade_zones.py:473-493` | Uses `val_logits` for calibration, applies to main data. |
| 20 | Macro Features Forward-Fill Leakage Risk | `features/macro_features.py:277-285` | Daily yields forward-filled to minute bars — verify no future yields. |
| 21 | TemperatureScaler After Logit Wrapping | `inference/pytorch_inference.py:212-216` | Should apply to raw logits before direction extraction. |
| 22 | Sanitize Clip Range Clips Macro Features | `infrastructure/numerics.py:29` | Hardcoded `(-20, 20)` clips valid yields/spreads. |
| 23 | Ollama Hardcoded, FinBERT Loads Per-Call | `infrastructure/news_pipeline.py:39, 120` | Model name hardcoded; FinBERT loads on first call (blocks). |
| 24 | Promotion Gate gross_pnl Undocumented | `validation/promotion_gate.py:204` | Required but callers may not know they must pass gross profit. |
| 25 | Feature Quality Monitor on Full Dataset | `features/feature_engineering_pl.py:2306` | Not per-fold — leakage risk. |
| 26 | Kelly Formula Incorrect | `risk/execution.py:68` | `win_prob - q/max(win_loss_r, 0.01)` — standard is `p - q/b`. |
| 27 | Session Limits Per-Pair Not Per-Portfolio | `trading/live_engine.py:1462-1479` | Aggregate caps may be exceeded. |
| 28 | No Emergency Kill Switch | `trading/live_engine.py` | Missing external signal handler for immediate flattening. |
| 29 | Daily Loss Limit Wrong Equity Reference | `risk/risk_engine.py:199-202, 305-306` | Uses `self.equity` vs `daily_start_equity` — can diverge. |
| 30 | No Gradient Checkpointing | `training/supervised_loop.py:1847` | Large models (TFT, HAELT, Mamba) lack checkpointing. |
| 31 | TrainingController No Crash Checkpointing | `training/training_controller.py` | State not persisted on crash. |
| 32 | No Scaler Version/Checksum Validation | `inference/_scaler_load.py` | No way to verify scaler.npz matches model checkpoint. |
| 33 | allow_pickle=True Security Risk | `inference/_scaler_load.py:57-59` | `np.load(path, allow_pickle=True)` for feature_names. |
| 34 | Feature Schema Audit First Chunk Only | `training/dataset_builder.py:457` | Schema drift in later chunks not caught. |
| 35 | Hard Example Mining Leakage Risk | `training/hard_example_miner.py` | Validation failures may leak into training. |

### 9.3 P2 — Medium Priority (Technical Debt)

| # | Issue | File | Description |
|---|-------|------|-------------|
| 36 | Numba Code Duplication (700+ lines) | `backtesting/backtest.py:426-828` | Maintenance burden, high bug surface. |
| 37 | GPU Backtester Prototype-Only | `backtesting/gpu_backtester.py` | No SL/TP, scaling, realistic execution. |
| 38 | SlippageCalibrator Heuristic Multipliers | `backtesting/improvements.py:340-365` | Session/urgency multipliers not calibrated. |
| 39 | LMAX FIX App Non-Functional Stub | `execution/lmax_fix_app.py` | Only logs, no order routing/market data. |
| 40 | FinBERT Not Thread-Safe | `features/finbert_sentiment.py:393-405` | ThreadPoolExecutor with `max_workers>1` crashes. |
| 41 | Audio Sentiment No Timeout/Streaming | `features/audio_sentiment.py:54-60` | Whisper transcription unbounded. |
| 42 | Mixed Print/Log Calls | Multiple | Inconsistent logging. |
| 43 | Hardcoded Demo Values in Main | `main.py:72-73, 93` | Synthetic data only, not production entry. |
| 44 | No CLI Argument Parsing | `main.py` | No config/model/data path args. |
| 45 | Sydney Mapped to Asia Session | `trading/session_utils.py:51` | Different liquidity profile than Tokyo. |
| 46 | Hardcoded 1.5× Off-Hours Spread | `trading/session_utils.py:64` | Not calibrated to actual spreads. |
| 47 | Subprocess Without Timeout | `scripts/train.py:241` | Can hang indefinitely. |
| 48 | Missing run_feature_engineering.py | `scripts/run_pipeline.py:75` | Referenced script doesn't exist. |
| 49 | Heartbeat Timeout 30 min | `monitoring/train_logger.py:71` | Too long for frozen training detection. |
| 50 | No Log Rotation for JSONL | `monitoring/train_logger.py:321` | Single file grows unbounded. |
| 51 | No Healthchecks in Docker | `infrastructure/docker-compose.streaming.yml` | Orchestration can't detect unhealthy containers. |
| 52 | No Resource Limits on RisingWave | `infrastructure/docker-compose.streaming.yml:22-30` | Can OOM host. |
| 53 | Train/Inference Scaler Fusion Inconsistency | Multiple | ONNX fuses, PyTorch applies at runtime, RL applies — mismatch risk. |
| 54 | Lot/Notional/Pip Convention Inconsistency | Multiple | Risk: 100k, RL: 10k, Portfolio: 100k with overrides. |
| 55 | Circuit Breaker Logic Divergence | `risk/risk_engine.py`, `risk/execution.py` | Two halt mechanisms track equity separately. |

### 9.4 Recommended Implementation Order

**Phase 1: Critical Safety & Correctness (Week 1)** — All 15 P0 items
**Phase 2: High-Impact Correctness (Week 2)** — All 20 P1 items  
**Phase 3: Technical Debt & Robustness (Week 3)** — All 15 P2 items

### 9.5 Validation Strategy

- Add unit tests for each fix (especially look-ahead bias, leakage fixes)
- Run existing test suite: `pytest tests/ -x -v`
- Run backtest comparison before/after fixes
- Integration test live engine with paper broker
- Verify promotion gate with real model outputs
