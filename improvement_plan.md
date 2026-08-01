# Forex Scaling Model: Improvement Plan

This document tracks high-priority future features and architectural upgrades to be implemented after the current training sequence completes.

## ✅ Recently Implemented (Verified)
- [x] **XGBoost Baseline Improvements:** Upgraded the non-deep XGBoost baseline to utilize full historical sequences, proper train/validation splits, and feature importance extraction.
- [x] **Sharpe Ratio Math Audit:** Fixed 6 bugs across 40+ files related to Sharpe ratio calculations (including fixing incorrect annualization factors for 1-minute bars).
- [x] **14 Baseline Training Improvements:** Tuned hyperparameters in `config/run.yaml`, including critical learning rate fixes, applying BYOL pretraining, Gaussian noise, and temporal Mixup augmentations.
- [x] **Heuristic Auto-Tune Improvements:** Expanded the automated hyperparameter tuning logic.
- [x] **Training Control & Memory:** Persisting per-model lessons in `training_memory.json` and tracking best epoch, fold, Sharpe, and failure patterns.
- [x] **Model-Specific Recipes:** Custom default LR, warmup, and sequence schedules mapped per architecture (e.g., Mamba vs. TFT).
- [x] **Overfitting Detection:** Real-time tracking of train-validation gap and Sharpe collapse, generating auto-tune proposals (e.g., increasing dropout automatically via `HEURISTIC 1`).
- [x] **Training Control Report:** Generating detailed `[model_name]_training_control_report.json` files logging curriculum events, early stops, and overfit warnings.
- [x] **Multitask Calibration Fix:** Post-training `TemperatureScaler` now handles multitask models by calibrating direction logits as classification output, preventing the `[N,3]` vs `[N]` calibration shape failure.
- [x] **Training Memory Gate Accounting Fix:** Passing promotion-gate checks are no longer recorded as failures in `training_memory.json`, keeping future auto-tune recommendations cleaner.
- [x] **Sharpe-Focused Feature Batch:** Added execution-cost context, market-regime/chop features, completed-candle 5m/15m/1h context, and leakage-safe trailing label-quality proxies. Requires a full feature-cache rebuild before training can use the new schema.
- [x] **Adaptive Curriculum Wiring Fix:** Feature-group masking now uses the cache's ordered feature schema when available, the new feature batch is assigned to curriculum groups, and Sharpe-collapse stalls now block the next curriculum advance instead of only updating an unused counter.
- [x] **Economic Prior & News Category Features:** Wired `eco_prior` and `news_cats` features into the feature pipeline.
- [x] **Config Preflight Validation:** Added comprehensive range validation for hyperparameters, model names, and loss functions before training starts.
- [x] **Batch FinBERT Prefetch & Parallel Window Loading:** Implemented upfront headline cache-warming and multi-threaded parallel dataset loading to eliminate I/O bottlenecks during dataset build.
- [x] **True Walk-Forward Out-Of-Sample Backtester:** Implemented a rigorous backtester eliminating in-sample leakage for walk-forward trained models.
- [x] **Curriculum-Aware Optuna Tuning:** Optuna now tunes the *shape and dynamics* of the curriculum (8 parameters) rather than just static properties.
- [x] **Execution-Aware Promotion:** Promotion gate now runs true OHLC holdout backtesting instead of relying on label-proxy PnL.

## 1. High Priority: OANDA Retail Sentiment & Order Book Integration

**Goal:** Provide the AI with "orthogonal" (uncorrelated) data by exposing retail trader psychology, allowing the model to learn institutional stop-hunting mechanics and retail trap patterns.

**Current Status:** A first-pass OANDA fetch/feature spike exists in `data/fetch_oanda_sentiment.py`, and `FeatureEngineer.build(..., oanda_data=...)` can consume basic OANDA-derived columns through `scripts/run_feature_engineering.py`. This is not yet production-ready: it currently writes a simple CSV, can silently fall back to mock data, and does not yet produce the required sidecar parquet cache, coverage report, leakage/staleness checks, or ablation evidence.

**Next Implementation Pass:**
1. Harden `data/fetch_oanda_sentiment.py` into a sidecar collector that writes both raw snapshots and normalized features under `data/oanda_sentiment/`.
2. Add explicit `source_status` / `is_mock` fields so mock or failed API responses can never enter production training by accident.
3. Generate `oanda_sentiment_report.json` before any feature-cache rebuild, with coverage, freshness, null-rate, cadence, and pair-level pass/warn/fail status.
4. Add leakage-safe joins in feature engineering: backward-only as-of alignment, capped forward-fill, `sentiment_staleness_minutes`, and hard failure when coverage is too sparse.
5. Run an ablation before enabling the columns by default: same folds, same cache window, baseline features versus baseline plus OANDA features.
6. Only after the ablation passes, rebuild the production cache and update feature schema/version/hash artifacts.

**Implementation Steps:**
- [ ] **Endpoint Availability Check:** Confirm OANDA Practice account access, target instrument coverage, update frequency, and response shape for the required sentiment/order-book endpoints before wiring anything into training.
- [ ] **API Authentication:** Configure a free OANDA Practice account and store the token in `.env` using a dedicated variable such as `OANDA_API_TOKEN`.
- [x] **Prototype Fetching Script:** Initial `data/fetch_oanda_sentiment.py` can fetch or mock OANDA `orderBook` and `positionBook` responses and engineer basic features for local experiments.
- [ ] **Production Sidecar Fetching Script:** Upgrade the prototype so it writes raw and normalized snapshots to a sidecar cache first, not directly into the training feature cache:
  - `/v3/instruments/{instrument}/orderBook` (Pending limit orders / stop-losses)
  - `/v3/instruments/{instrument}/positionBook` (Live open positions and ratios)
- [ ] **Sidecar Storage:** Save snapshots under a path such as `data/oanda_sentiment/{instrument}_{date}.parquet`, with timestamp, instrument, source endpoint, raw snapshot time, and normalized feature columns.
- [ ] **Coverage & Freshness Report:** Generate `oanda_sentiment_report.json` before model integration. Include pairs covered, timestamp range, missing minutes, stale snapshot count, null rate per feature, update cadence, and pass/warn/fail usability status per pair.
- [ ] **Leakage Guardrails:** Align snapshots using only data available at or before each bar timestamp. Cap forward-fill windows, mark stale snapshots explicitly, and fail the feature join if sentiment data is too sparse.
- [x] **Prototype Feature Engineering:** Basic numerical features are available: `retail_long_ratio`, `stop_loss_cluster_dist_long`, `stop_loss_cluster_dist_short`, and `order_imbalance`.
- [ ] **Production Feature Engineering:** Expand and validate the OANDA feature set, instead of passing raw JSON buckets to the model:
  - `retail_long_ratio`: Percentage of retail traders currently holding long positions.
  - `retail_long_ratio_zscore`: Recent z-score of retail long exposure.
  - `retail_position_flip_1h`: Change in long/short positioning over the last hour.
  - `stop_loss_cluster_dist_long`: Pip distance to the largest cluster of sell-stop orders below the current price.
  - `stop_loss_cluster_dist_short`: Pip distance to the largest cluster of buy-stop orders above the current price.
  - `nearest_stop_cluster_side`: Which side has the nearest meaningful stop cluster.
  - `nearest_stop_cluster_distance_pips`: Pip distance to that nearest meaningful stop cluster.
  - `order_imbalance`: Ratio of pending limit buy orders vs. limit sell orders.
  - `order_book_imbalance_zscore`: Recent z-score of pending order imbalance.
  - `sentiment_staleness_minutes`: Minutes since the last valid sentiment/order-book snapshot.
- [ ] **Mock Data Safety:** Add an explicit `is_mock` / `source_status` marker and fail production feature builds when OANDA falls back to mock data.
- [ ] **Ablation Before Default Use:** Compare baseline features against baseline plus OANDA features on the same folds and forward holdout. Require improvement in validation Sharpe, forward Sharpe, drawdown, profit factor, and trade count before enabling by default.
- [ ] **Dataset Integration:** Merge these columns into the training feature set only after sidecar coverage, leakage checks, and ablation pass. Rebuild the cache, update feature schema/version/hash artifacts, and treat the model feature count as `current_features + N` rather than assuming automatic safety.

## 2. Adaptive Curriculum Upgrades
Current curriculum transitions (difficulty stage, sequence length) are purely static (epoch-based). This upgrade will make them reactive to model performance:
- [ ] **Performance Gating:** Advance difficulty stages only if the rolling validation Sharpe ratio is positive/stable for `N` epochs.
- [ ] **Seq_len Protection:** Delay or temporarily revert sequence length increases if a sudden collapse in `val_sharpe` is detected.
- [ ] **Regime-Aware LR Tuning:** Automatically reduce the learning rate by 15-20% when unfreezing complex features or transitioning from easy -> hard market regimes.
- [ ] **Event Logging:** Log curriculum stage transitions as first-class events to W&B and console for easier debugging.

## 3. Deployment, Gating, & Testing Upgrades
Currently, promotion gating is relatively basic (label-proxy PnL, console-based pair readiness) and lacks rigorous metadata safety. This phase will harden the deployment pipeline:
- [ ] **Confidence-Aware DisagreementGate:** Pass real model confidence through inference instead of `confidence=None`.
- [ ] **Dynamic Threshold Tuning:** Add confidence-threshold tuning per model/fold and persist it to `threshold_tuning.json` (or bake it into the promoted artifact).
- [ ] **Execution-Aware Promotion:** Upgrade promotion gates from simple label-proxy PnL to execution-aware forward holdout (incorporating OHLC, spread, slippage, and latency).
- [ ] **Structured Readiness Reporting:** Upgrade the console pair-readiness gate into a structured `pair_readiness_report.json` with timestamp-alignment diagnostics.
- [ ] **Metadata Verification:** Strengthen checkpoint validation to strictly require model name, feature count, seq_len, feature schema hash, and ONNX sample inference before promotion.
- [ ] **Discord Monitoring Cleanup:** Clean out stale `model_promoted` docs/examples from `monitoring/discord_alerts.py`.
- [ ] **Audit & Safety Tests:** Add rigorous test coverage for auto-tune proposal safety, deployment failure transaction logs, and live confidence gating.
- [ ] **Memory Auditability:** Route all `training_memory.json` suggestions through formal, auditable proposal files before they can affect the next run.
- [ ] **Advanced Latency Telemetry:** Upgrade the existing `prometheus_exporter.py` to track full `tick-to-trade` latency. Dump p50/p95/p99 latency boundaries into a JSONL structure for deep Grafana historical analysis.

## 4. Future Pipeline Upgrades


---

# Extended Project Improvement Plan

# Project Improvement Plan

Last reviewed: 2026-06-18

This document lists high-impact improvements for the Forex Scaling Model project, based on the current training, checkpointing, Discord alerting, promotion, validation, and execution pipeline.

## Priority 0: Correctness And Trust

### Make Artifact Saving Verifiable (MOSTLY COMPLETED)

Current status:

- `_safe_save()` now writes checkpoint files atomically with `torch.save()` and verifies that the temporary checkpoint can be loaded before replacement.
- `_safe_save(..., metadata=...)` now writes an atomic `.metadata.json` sidecar and verifies expected metadata fields for best, periodic, and resume checkpoints.
- `_safe_save_json()` now writes JSON artifacts atomically.
- Per-model `manifest.json`, `promotion_gate.json`, `fold_selection.json`, and `deployment.json` are now written by the training path.
- Remaining work: extend metadata sidecars to specialized artifacts such as pretrain, RL, ensemble, ONNX, and deployment copies.

- Add a run-level `manifest.json` for every training run.
- Include model name, run name, fold id, start/end timestamps, best epoch, best metric, checkpoint paths, promotion result, deploy result, git hash, config path, warnings, and errors.
- Use atomic writes for all critical artifacts:
  - `*_best.pt`
  - `*_last.pt`
  - `*_promotion.json`
  - `production_best.pt`
  - `production_best.onnx`
  - `reload_model.flag`
- After saving a checkpoint, immediately verify:
  - file exists
  - file size is greater than zero
  - `torch.load` succeeds
  - checkpoint metadata matches the expected model, feature count, and sequence length

### Fix Best-Fold Promotion Path Handling (COMPLETED)

- Update `_promote_best_fold()` so it detects both checkpoint layouts:
  - `<checkpoint_dir>/<model>_fold0_best.pt`
  - `<checkpoint_dir>/<model>/<model>_fold0_best.pt`
- Write a clear `fold_selection.json` with all candidate folds and the selected fold.
- Fail loudly if the best fold cannot be copied to the canonical model checkpoint.
- Use tie-breakers instead of peak Sharpe only:
  - highest validation Sharpe
  - lower validation loss
  - smaller train-validation gap
  - lower fold volatility
  - lower drawdown if available

### Separate Training Completion From Production Deployment (MOSTLY COMPLETED)

Current status:

- Training completion now sends `training_completed`.
- Promotion gate success and production deployment success are separate code paths.
- `production_deploy_completed` and `production_deploy_failed` alert methods exist.
- Remaining work: clean stale `model_promoted` examples/docs in `monitoring/discord_alerts.py` so the documented alert vocabulary matches the pipeline.

- Keep `training_completed` as a blue informational Discord alert.
- Send `model_promoted` only after production deployment succeeds.
- Remove or rename any training-completion code path that still sends `model_promoted`.
- Add a `production_deploy_failed` alert for failed checkpoint copy, ONNX export, or reload flag creation.

## Priority 1: Self-Improving Training

### Add Training Memory (IMPLEMENTED, NEEDS AUDITABLE PROPOSALS)

Current status:

- `training/training_memory.py` exists and writes persistent memory under `logs/training_memory.json`.
- Training memory is loaded at startup and applies conservative nudges to training arguments.
- The training loop updates memory after a run.
- Per-model history and failure counts are tracked.
- Remaining work: route memory-driven nudges through the same auditable proposal path as auto-tune, so no memory change is applied without a reviewable reason.

Create a persistent `training_memory.json` that summarizes lessons from previous runs:

```json
{
  "best_model": "haelt",
  "best_sharpe": 0.5608,
  "best_epoch_pattern": "early_peak",
  "common_failure": "validation_sharpe_collapse_after_lr_peak",
  "recommended_lr": 0.00005,
  "recommended_dropout": 0.30,
  "recommended_patience": 6,
  "recommended_max_epochs": 24
}
```

Use this file to adjust the next run conservatively.

### Make `_auto_tune_next_run` Auditable

Current status:

- `_auto_tune_next_run()` now accepts `run_name` and `dry_tune`.
- `--dry-tune` is a real CLI option.
- Auto-tune proposals are written to `logs/auto_tune/<run>_proposal.json`.
- Remaining work: add tests that prove high-risk fields cannot be mutated and that proposal JSON is always written.

- Save every automatic config change to `logs/auto_tune/<run>_proposal.json`.
- Include:
  - observed issue
  - previous value
  - proposed value
  - reason
  - confidence
- Add a dry-run mode so config suggestions can be reviewed before mutation.
- Never let auto-tuning silently change high-risk settings like data range, label method, or production thresholds.

### Build An Automation Ladder

Make the project increasingly automatic, but split automation into safe tiers:

```text
observe automatically        -> always safe
propose automatically        -> safe, reviewable
apply low-risk tuning        -> allowed with audit trail
promote/deploy automatically -> only after strict gates pass
live-size automatically      -> only in approved maturity stages
```

Automate run observation:

- collect train loss, validation loss, validation Sharpe, train-validation gap, fold volatility, NaN skips, OOM skips, GPU memory, GPU temperature, and runtime
- write one run summary per model/fold
- send Discord alerts for start, epoch, fold complete, promotion gate, deployment, and failure states
- attach artifact paths to alerts so every automated decision can be inspected

Automate safe tuning:

- tune `min_confidence` per model/fold from validation and backtest sweeps
- tune patience, max epochs, dropout, weight decay, and LR within bounded ranges
- tune curriculum pace when Sharpe collapses after difficulty or sequence-length increases
- tune execution filters such as max spread, min liquidity, and session-specific sizing from backtest results
- tune RL reward weights only as proposals unless a shadow-mode comparison proves improvement

Automate model selection:

- train base models with model-specific recipes
- compare each model on the same forward holdout
- compare single best model vs fold ensemble vs multi-architecture ensemble
- promote only if the candidate beats production on execution-adjusted Sharpe, drawdown, profit factor, and minimum trade count
- save the reason the winner was selected in `fold_selection.json`, `ensemble_selection.json`, or `promotion_gate.json`

Automate deployment only after gates:

- export ONNX automatically after promotion passes
- verify feature schema hash, checkpoint metadata, ONNX load, and sample inference
- atomically copy `production_best.pt` and `production_best.onnx`
- write `deployment.json` with every step and failure reason
- trigger reload only after the production artifacts pass verification

Keep these changes proposal-only unless explicitly approved or promoted:

- data date range
- label method
- pair list
- live maturity stage
- production threshold floors
- risk limits
- real-money order sizing

Required automation artifacts:

- `logs/auto_tune/<run>_proposal.json`
- `training_memory.json`
- `threshold_tuning.json`
- `training_control_report.json`
- `execution_report.json`
- `promotion_gate.json`
- `deployment.json`

### Learn From Sharpe Collapse

When validation Sharpe peaks early and then drops:

- reduce max LR or warmup peak
- stop earlier on future runs
- increase dropout or weight decay slightly
- hold the sequence curriculum stage longer
- restore the best epoch before continuing

For the observed run, the model peaked around epoch 8 and degraded after the curriculum/difficulty shift. That should become an explicit training-memory signal.

### Add Hard-Example Mining

After validation, save difficult samples:

- high-confidence wrong direction predictions
- large missed reward opportunities
- wrong predictions during high-impact news windows
- wrong predictions during spread/liquidity stress
- regime-transition failures

Then lightly oversample these examples in the next run.

### Compare Challenger Against Production

Promotion should require the challenger to beat:

- current `production_best.pt`
- XGBoost baseline
- previous best fold
- risk-adjusted forward holdout

Save challenger-vs-production metrics in `promotion_gate.json`.

## Priority 2: Validation And Promotion

### Strengthen Promotion Reports

Write separate files:

- `train_summary.json`
- `fold_selection.json`
- `promotion_gate.json`
- `deployment.json`
- `manifest.json`

Avoid mixing training completion, gate result, and deployment status in one ambiguous artifact.

### Improve Gate Inputs

Current status:

- Promotion now runs a real `PromotionGate` on a chronological forward holdout.
- The current gate still uses reward-label proxy PnL from the feature cache, not a full OHLC execution-aware backtest.
- Remaining work: when OHLC, spread, slippage, latency, and execution filters are available, run the gate on realistic forward backtest results instead of proxy PnL.

The promotion gate should use a realistic forward holdout whenever possible:

- true OHLC backtest
- spread/slippage model
- latency penalty
- turnover
- max drawdown
- profit factor
- regime concentration
- PSR/DSR

If only label-proxy PnL is used, mark the gate result as proxy-based.

### Add Deployment Transaction State

Deploy should write:

```json
{
  "status": "success",
  "source_checkpoint": "haelt_best.pt",
  "production_checkpoint": "production_best.pt",
  "previous_checkpoint": "production_prev.pt",
  "onnx_export": "success",
  "reload_flag": "written",
  "deployed_at": "2026-06-10T20:00:00Z"
}
```

On failure, include the exact failed step and exception.

## Priority 3: Discord And Observability

### Improve Discord Alert Semantics

Current status:

- Most desired alert methods exist.
- Remaining work: remove or clearly deprecate the stale `model_promoted` documentation/example in `monitoring/discord_alerts.py`.

Use these alert types:

- `training_started`
- `training_epoch`
- `training_completed`
- `fold_selected`
- `promotion_gate_passed`
- `promotion_gate_failed`
- `production_deploy_completed`
- `production_deploy_failed`
- `model_demoted`
- `retrain_started`

Each alert title should mean exactly one thing.

### Add Artifact References To Alerts

Include:

- run name
- fold id
- best checkpoint
- production checkpoint
- promotion JSON
- log path
- curve image path
- ONNX status
- reload flag status

### Improve Rate Limiting

- Rate-limit epoch alerts by run and fold, not just alert type.
- Always allow critical alerts:
  - crash
  - gate failed
  - deploy failed
  - demotion
- Avoid suppressing important events because another alert of the same type was sent recently.

## Priority 4: Data And Features

### Add Dataset Manifest

Every processed cache should include:

- source
- pairs
- start/end
- bar frequency
- news mode
- feature count
- label method
- sequence length
- cache creation time
- feature schema hash

Training should refuse mismatched caches unless explicitly overridden.

### Add Feature Schema Versioning (PARTIALLY COMPLETED)

Current status:

- Training writes cache feature-schema sidecars.
- ONNX deployment writes a production schema file with ordered feature names and a schema hash.
- Live inference reads production schema metadata for feature count in some paths.
- Remaining work: enforce schema-hash equality before live mode, include schema hash in all promotion/deployment artifacts, and fail fast when live feature order differs from training.

- Save the ordered feature names used during training.
- Save the same schema next to ONNX exports.
- Make live inference fail fast when feature order or feature count differs.

### Expand Data Quality Reports

Add reports for:

- missing bars by pair
- spread outliers
- zero-volume periods
- weekend/holiday filtering impact
- feature NaN rate
- label class balance
- reward distribution
- per-regime sample counts

### Add Pair Readiness Gate Before Training (MOSTLY COMPLETED)

Current status:

- A pair readiness console gate exists in `training/train_gpu.py`.
- It fails when a required pair produces zero usable sequences or excessive NaNs.
- Multi-pair timestamp alignment now uses an explicit inner join over real timestamps.
- Difficulty sidecar values are computed with `_compute_difficulty_scores()` and combined across pairs.
- Structured `<cache>_pair_readiness_report.json` artifacts are now written for single-pair and multi-pair cache builds with per-pair pass/warn/fail status, raw tick/schema checks, usable sequence counts, nonfinite rates, label distribution, difficulty distribution, spread/ATR summaries, and timestamp-alignment loss.
- Remaining work: add finer-grained drop-reason buckets for weekend, holiday, dead-bar, spread, ATR, news, and label filters, plus explicit missing-hour counts before resampling.

The project already cleans ticks, filters bad bars, checks labels, drops low-quality sequences, and validates cache lengths. The missing piece is one clear pre-training gate that proves every selected pair is clean, aligned, and usable before the model starts.

Write a `pair_readiness_report.json` before training begins.

For each pair, include:

- requested source, pair, start date, end date, and bar frequency
- raw tick count
- timestamp range loaded
- duplicate timestamp count
- missing hour or missing bar count
- bars after resampling
- bars dropped by weekend, holiday, dead-bar, spread, ATR, news, and label filters
- feature NaN, positive infinity, and negative infinity counts
- label class balance or reward distribution
- valid sequence count after quality filtering
- spread median, spread p95, and spread max
- ATR median, ATR p95, and ATR max
- readiness status: `pass`, `warn`, or `fail`
- exact fail reason when a pair is not usable

Training should fail before the first epoch if:

- any required pair has zero usable sequences
- timestamps are not UTC-normalized
- `bid`, `ask`, or timestamp columns are missing
- `ask <= bid` remains after cleaning
- feature or label nonfinite counts remain above a strict threshold
- label classes are badly collapsed
- cache sidecar lengths do not match
- RL market arrays `close`, `atr`, and `spread` are missing when RL or execution simulation needs them

Fix multi-pair timestamp alignment:

- Enforce `pair_align: inner` using real bar timestamps, not just shortest sequence count.
- Save the aligned timestamp range and common-bar count.
- Report per-pair bars dropped by the inner join.
- Fail or warn when one pair loses too many bars compared with the others.

Repair curriculum difficulty sidecar:

- Use `_compute_difficulty_scores()` when building `diff_seq`.
- Save difficulty distribution by pair and by final joined dataset.
- Fail or warn when all difficulty values are zero on real data, because that disables the easy/medium/hard curriculum.

Add a required pre-training console summary:

```text
Pair readiness:
  EURUSD pass  ticks=... bars=... seq=... dropped=...
  USDJPY pass  ticks=... bars=... seq=... dropped=...
  GBPUSD warn  missing_hours=... seq=...
```

This makes pair cleanliness visible before training spends GPU time.

## Priority 5: Model Training

### Add Model Cards

For each trained model, save a small model card:

- architecture
- data window
- pairs
- features
- label method
- validation results
- forward holdout results
- known weaknesses
- promotion status

### Improve Curriculum Adaptation

Current curriculum is static. Make it adaptive:

- advance difficulty only if validation Sharpe is stable
- delay `seq_len` increases when Sharpe collapses
- reduce LR when moving from easy to medium/hard regimes
- log curriculum transitions as first-class events

### Add Training Control, Memory, And Overfit Response

Training should become a managed loop instead of a fixed epoch run. Add a training controller that watches every epoch and adjusts the next decision based on validation behavior.

Model-specific training recipes:

- Give each architecture its own default LR, warmup, dropout, patience, curriculum pace, sequence schedule, loss, and feature set.
- Start HAELT more conservatively with slower warmup and longer easy-curriculum stages.
- Let Mamba test shorter and longer sequence windows earlier because it is useful for fast reaction and temporal state.
- Let TFT focus more on interpretable context, regime features, and multitask direction/return/confidence heads.
- Let GNN focus on cross-pair, cross-asset, and correlation-stress windows.
- Let expert models specialize by regime, session, volatility state, or confirmation role instead of training them like general-purpose models.

Training memory:

- Persist per-model lessons in `training_memory.json`.
- Track best epoch, best fold, best validation Sharpe, forward-holdout result, train-validation gap, curriculum stage at peak, LR at peak, and failure pattern.
- Store repeated warnings such as Sharpe collapse after hard curriculum, overfitting after LR peak, weak fold behavior, NaN skips, OOM skips, or unstable validation loss.
- Use memory only as a conservative suggestion source for the next run; do not silently mutate high-risk config values.
- Save every memory-driven suggestion to an auditable auto-tune proposal.

Overfitting detection:

- Detect when train loss keeps improving while validation loss rises.
- Detect when validation Sharpe spikes once and then collapses.
- Detect when directional accuracy stays high but forward-holdout PnL, drawdown, or profit factor gets worse.
- Detect when confidence rises but high-confidence wrong trades increase.
- Respond by lowering LR, increasing dropout or weight decay slightly, holding curriculum difficulty, restoring the best epoch, or stopping early.

Save a compact training-control report:

```text
training_control_report.json
```

The report should include:

- model recipe used
- memory suggestions applied or skipped
- overfitting signals detected
- curriculum changes
- LR changes
- best epoch and restore decision
- final promotion recommendation

### Use Stability-Aware Fold Selection (COMPLETED)

Avoid selecting a fold based only on a single spike. Score folds with:

```text
score = val_sharpe
        - train_val_gap_penalty
        - fold_volatility_penalty
        - drawdown_penalty
```

Keep the old simple ranking as a fallback.

### Train Diverse Model Recipes

Do not train every model in exactly the same way. Build a controlled set of diverse training recipes so the ensemble has genuinely different opinions.

Recommended first recipe set:

```text
HAELT       full features, sharpe_huber, seq_len 60
Mamba       fast reaction, directional_huber, seq_len 30 or 45
TFT         context/interpretable, cross_entropy or multitask direction
XGBoost     last-bar tabular baseline
```

Expanded recipe set:

```text
Transformer generic long-range baseline
GNN         cross-asset/risk structure
Expert      specialist confirmation model
```

Useful diversity axes:

- architecture: HAELT, Mamba, TFT, Transformer, GNN, Expert, XGBoost
- loss function: `sharpe_huber`, `directional_huber`, `cross_entropy`, `asymmetric`
- sequence length: 30, 45, 60, 120
- feature set: full, no news, no cross-asset, price-only, volatility-only
- regime focus: scalping, normal, high-volatility, session-only, news-window
- random seed: multiple seeds for the same recipe only after the core recipes work

Each recipe should write:

- recipe name
- model architecture
- loss
- feature set
- sequence length
- fold metrics
- forward holdout metrics
- checkpoint path
- promotion status

### Improve Pretraining And Prove It Helps

The project already supports several pretraining paths: BYOL, TSCL, regime-aware TSCL, masked reconstruction, VAE, cluster, forecast, and drift pretraining. The next step is to make pretraining measurable, model-specific, and leakage-safe.

Status: **MOSTLY COMPLETED**. The core trainer now writes a real per-model `pretrain_report.json` that records the selected method, recommended method family, leakage-safe pretrain/supervised/holdout index windows, sample counts, epochs completed, average/final loss, embedding diagnostics, quality-gate result, checkpoint path, hard-example reuse count, and whether the pretrained encoder actually loaded into supervised training. The old placeholder report was replaced with a merge/update so these diagnostics are not overwritten after supervised training.

`pretrain_ablation.json` now includes baseline and pretrained fold summaries, metric deltas, and a simple verdict (`pretrain_helped`, `pretrain_hurt`, `mixed`, or `unknown`) using the same folds and seed path. The wrapper script writes the same verdict shape for two-pass ablation runs.

Still remaining: feed forward-holdout Sharpe, max drawdown, and profit factor from the promotion gate back into `pretrain_ablation.json`, and separate pretraining onto older broad history from supervised training onto newer target history when the data window configuration supports that split.

Add a no-pretrain baseline for every serious training run:

```text
same model + same fold + no pretrain
same model + same fold + selected pretrain method
```

Promotion should know whether pretraining actually improved:

- validation Sharpe
- validation loss
- directional accuracy
- forward-holdout Sharpe
- max drawdown
- profit factor
- train-validation gap

Save the result as:

```text
pretrain_ablation.json
```

Recommended pretrain method map:

```text
HAELT       masked or BYOL
Mamba       forecast or drift contrastive
TFT         masked + regime-aware context
GNN         cluster or cross-asset regime pretrain
Transformer BYOL or TSCL baseline
Expert      regime-aware TSCL confirmation
```

Separate pretrain and supervised windows where possible:

```text
older broad history        -> pretrain
newer target history       -> supervised training
untouched forward history  -> promotion gate
```

This keeps pretraining as market-structure learning instead of letting it memorize the supervised validation period.

Add a `pretrain_report.json` for every model:

- method
- pretrain window
- supervised window
- epochs completed
- average pretrain loss
- final embedding standard deviation
- alignment and uniformity when available
- quality-gate result
- checkpoint path
- whether pretrained weights loaded into supervised training
- downstream metric delta versus no-pretrain baseline

Improve handoff logic so pretraining stops only when representation quality is acceptable:

- loss plateau is not enough by itself
- embedding std must not indicate collapse
- uniformity must be acceptable for contrastive methods
- first supervised warm-start epochs should not increase the train-validation gap
- bad pretrain should be discarded automatically and logged

Reuse hard-example mining for pretraining:

- oversample confident-wrong validation windows in the next pretrain run
- include high-reward missed opportunities
- include regime-transition failures
- include high-spread or high-volatility mistakes
- cap hard-example influence so pretraining does not overfit rare noise

Guardrails:

- Do not use the forward promotion holdout in pretraining.
- Do not choose the pretrain method using the same validation slice used for production promotion.
- Keep method comparisons on the same folds and same seed when possible.
- Prefer the simplest pretrain method that improves forward metrics, not the one with the lowest pretrain loss.

### Use Ensemble Learning As A Risk Filter

Use the ensemble not only to improve prediction quality, but also to measure disagreement.

Simple policy:

```text
all/most models agree      -> normal trade size
weak agreement             -> reduced trade size
strong disagreement        -> hold
best model says buy/sell
ensemble disagrees         -> skip or shadow-only
```

Recommended progression:

1. Train HAELT, Mamba, TFT, and XGBoost.
2. Compare each model against the same walk-forward and forward-holdout windows.
3. Train the existing `EnsembleMetaLearner` on a separate meta-training split.
4. Backtest ensemble against:
   - best single model
   - XGBoost baseline
   - current `production_best.pt`
5. Run ensemble in shadow mode before allowing production promotion.

Guardrails:

- Do not train the meta-learner on the same validation slice used to choose base checkpoints.
- Keep a separate meta-validation or forward-holdout segment.
- Log model weights from the meta-learner so one model cannot silently dominate.
- Penalize highly correlated base predictions.
- Promote the ensemble only if it improves risk-adjusted metrics, not just raw Sharpe.

Artifacts to add:

```text
ensemble_recipe_manifest.json
ensemble_weights_summary.json
ensemble_backtest.json
ensemble_promotion_gate.json
```

### Use Reinforcement Learning As The Execution Layer

Do not use reinforcement learning as the primary market direction predictor. Keep supervised models and ensembles responsible for signal quality, then use RL to decide how to execute that signal safely.

Recommended architecture:

```text
supervised model / ensemble signal
        |
        v
risk gate: spread, drawdown, regime, volatility
        |
        v
RL execution policy
        |
        v
position size / scale in / scale out / close / hold
```

RL should decide:

- whether to hold or trade
- position size
- scale-in amount
- scale-out amount
- early close
- whether to skip during high spread or bad regime

RL should not be allowed to:

- bypass max position limits
- bypass daily or weekly drawdown limits
- override hard risk stops
- trade during blocked market regimes
- promote itself without supervised/backtest comparison

Compare three execution modes before promotion:

```text
supervised-only execution
supervised + rule-based risk execution
supervised + RL execution
```

Add an `rl_report.json` after every RL run:

- algorithm: DQN or PPO
- observation mode: raw features or frozen encoder embedding
- train return
- validation Sharpe
- validation return
- max drawdown
- profit factor
- trade count
- turnover
- average hold time
- action distribution
- reward component breakdown
- whether `rl_*_best.pt` was updated
- comparison against supervised-only execution

Improve reward diagnostics by logging each component separately:

- PnL reward
- drawdown penalty
- transaction cost penalty
- overtrade penalty
- spread penalty
- regime penalty

Add anti-overtrading guardrails:

- penalize repeated long/short flips
- penalize scaling in during high spread
- penalize too many trades per session
- penalize holding through blocked news/regime windows
- cap daily RL action count

Train RL only after the supervised model is stable:

- folds are stable
- promotion gate is close to passing or already passing
- feature schema is locked
- cache includes real `close`, `atr`, and `spread`
- backtest has enough trades to evaluate execution behavior

Promotion should require RL to beat the non-RL execution baseline on risk-adjusted metrics, not just raw return.

Run RL in shadow mode before live use and record every RL action decision:

```text
signal_before_rl
risk_gate_result
rl_action
position_before
position_after
spread
atr
regime
reward_components
```

### Expand Distillation

The config has teacher/student settings. Improve this by:

- verifying teacher checkpoint exists before training
- logging teacher metrics
- saving student-vs-teacher agreement
- tuning temperature by validation performance
- using production model as optional teacher for stability

## Priority 6: Execution And C++ Runtime

### Phase Ensemble Deployment To Avoid Premature C++ Complexity

If the production system moves from one model to an ensemble, the C++ runtime may eventually need to support multiple model sessions. Do not start there.

Recommended deployment path:

```text
Phase 1: Train ensemble in Python and run it in shadow/backtest mode.
Phase 2: Export one combined ensemble ONNX if possible.
Phase 3: Keep ensemble direction as one production_best.onnx.
Phase 4: Optionally load a second RL execution ONNX through EXECUTION_MODEL_PATH for 10-action execution.
Phase 5: Add multi-model ensemble member sessions only if the ensemble proves it needs separate runtime sessions.
```

Preferred short-term architecture:

```text
HAELT / Mamba / TFT / XGBoost
        |
        v
Python ensemble wrapper
        |
        v
single direction production_best.onnx
        |
        v
optional rl_<algo>_execution.onnx
        |
        v
C++ runtime runs direction first, then optional execution policy
```

Only update the C++ runtime for multi-model inference after the ensemble has passed shadow-mode and forward-holdout validation.

If multi-model C++ inference becomes necessary, add:

- multiple ONNX Runtime sessions
- per-model schema validation
- per-model output logging
- ensemble-weight logging
- fallback behavior when one model fails
- latency budget checks for the full ensemble
- config-driven model paths
- Discord/JSONL alerts when ensemble members disagree

Safer intermediate option:

```text
primary model signal + ensemble confirmation
```

Policy example:

```text
primary says BUY and ensemble agrees      -> normal size
primary says BUY and ensemble is mixed    -> reduced size
primary says BUY and ensemble disagrees   -> hold
```

### Update Code Paths For Ensemble Readiness

The codebase already has several ensemble building blocks, but the training, export, live inference, and C++ runtime paths need clearer boundaries before ensemble deployment.

Completed before expanding ensemble work:

- `training/train_gpu.py`: `_safe_save()` writes atomically with `torch.save()` and verifies loadability.
- `training/train_gpu.py`: `_auto_tune_next_run()` call/signature mismatch is fixed and `--dry-tune` is a real CLI option.
- `monitoring/discord_alerts.py`: production deploy and promotion gate alert methods exist.
- `training/train_gpu.py`: `_promote_best_fold()` handles nested and flat checkpoint layouts for checkpoints and config files.

Remaining before expanding ensemble work:

- `monitoring/discord_alerts.py`: clean stale `model_promoted` documentation/example.
- `training/train_gpu.py`: make fold metadata copying/verification explicit when nested and flat layouts are both present.
- `tests/`: add coverage for auto-tune safety, deploy transaction failure states, and manifest contents.

Python training updates:

- `models/ensemble.py`: add a production wrapper around `EnsembleMetaLearner` that exposes one stable prediction output and optional model-weight diagnostics.
- `scripts/train_ensemble_meta.py`: save a full ensemble manifest with base model names, fold ids, checkpoint paths, feature schema, recipe names, metrics, and meta-validation window.
- `training/train_gpu.py`: compare best single fold, best single model, and ensemble candidate before production promotion.
- `training/train_gpu.py`: keep ensemble promotion separate from normal training completion so the Discord alert only says production when deployment really succeeded.
- `scripts/backtest_model.py`: support ensemble checkpoint and manifest inputs so ensemble results can be compared against the current `production_best.pt`.

ONNX and Python live-runtime updates:

- `inference/onnx_inference.py`: export the ensemble as one combined ONNX model first, if possible.
- `inference/onnx_inference.py`: handle ensemble wrappers that may return prediction plus diagnostics, while keeping the production output stable.
- `inference/pytorch_inference.py`: add loading support for ensemble meta-checkpoints and their base model checkpoints.
- `trading/live_engine.py`: load/export the single combined ensemble ONNX path for live inference.
- `trading/live_engine.py`: log model disagreement, ensemble weights, reduced-size decisions, and skipped trades caused by ensemble disagreement.

C++ runtime updates only if separate runtime sessions are required:

- `infrastructure/cpp_server/src/main.cpp`: move from a single `MODEL_PATH` to a config-driven model list.
- `infrastructure/cpp_server/src/ModelRunner.cpp`: keep the current single-session runner for single ONNX deployment, but add a separate `EnsembleRunner` if multiple ONNX sessions are needed.
- `infrastructure/cpp_server/src/ModelRunner.cpp`: validate input shape and feature schema per model before allowing live inference.
- `infrastructure/cpp_server`: add latency and failure handling for multi-model inference so one failed member cannot create an unsafe live decision.

Recommended order:

1. Fix the training correctness issues.
2. Train diverse recipes in Python.
3. Backtest every model separately.
4. Train and validate the ensemble meta-learner.
5. Export one combined ensemble ONNX.
6. Run ensemble in Python shadow mode.
7. Keep C++ loading one direction `production_best.onnx`.
8. Add optional `EXECUTION_MODEL_PATH` for the native 10-action RL execution policy.
9. Add multi-model ensemble member sessions only after the ensemble proves it needs separate runtime sessions.

### Close Python/C++ Feature Parity

- Make feature parity checks mandatory before live deployment.
- Save parity reports next to ONNX exports.
- Include feature schema hash in both Python and C++ logs.

### Harden Shadow Mode

- Ensure live mode cannot run unless parity passes.
- Keep partial feature mode restricted to shadow mode.
- Add explicit Discord alert when shadow mode detects feature mismatch.

### Improve Execution Quality And Cost Control

Execution quality should be treated as part of model performance. A model should not be promoted unless it still performs after realistic spread, slippage, latency, and session constraints.

Add hard execution filters:

- maximum spread
- maximum estimated slippage
- minimum liquidity
- maximum volatility shock
- blocked rollover window
- blocked major-news window
- blocked stale-data window

Use confidence-aware sizing:

```text
high confidence + low spread       -> normal size
medium confidence                  -> reduced size
low confidence                     -> hold
ensemble disagreement              -> hold
high spread or poor liquidity      -> hold or shadow only
```

Tune confidence thresholds per model and fold:

- Do not rely on one fixed `0.45` threshold for every architecture, pair set, and fold.
- During validation/backtest, sweep candidate thresholds such as `0.35`, `0.40`, `0.45`, `0.50`, `0.55`, and `0.60`.
- Pick the threshold that maximizes execution-adjusted validation Sharpe while keeping enough trades for a reliable sample.
- Save the selected threshold in the fold/model artifact, for example `fold_selection.json`, `train_summary.json`, or a dedicated `threshold_tuning.json`.
- Promotion should compare the model using its tuned threshold, not the default threshold.

Pass real confidence into live guards:

Current status:

- Inference engines expose `predict_proba()` and can compute max-class confidence.
- `DisagreementGate` already supports a confidence argument and can block `low_confidence`.
- Still open: `trading/live_engine.py` passes `confidence=None` into `DisagreementGate`, so the live guard is not using model confidence yet.
- Threshold tuning is still missing; current thresholds are fixed/config/profile values rather than per-model/fold tuned artifacts.

- Ensure the inference path returns both action and confidence/probability.
- Pass the real model confidence into `DisagreementGate` and live guard checks instead of `confidence=None`.
- If confidence is below the tuned model/fold threshold, force `HOLD`.
- Log skipped trades with `blocked_by: low_confidence`, the threshold used, and the raw confidence.
- Keep the live threshold synchronized with the promoted model artifact so backtest, shadow mode, and live mode use the same decision rule.

Add an `execution_report.json` for backtest, shadow, and live runs:

- signal timestamp
- expected entry price
- actual or simulated fill price
- spread at entry
- estimated slippage
- order size
- stop loss
- take profit
- exit reason
- realized PnL
- execution cost
- latency breakdown
- session
- regime
- model confidence
- ensemble disagreement score

Improve stop and exit logic:

- ATR-based initial stop
- reward-ratio take profit
- trailing stop after favorable movement
- breakeven move after partial profit
- time stop for stale trades
- max adverse excursion stop
- forced exit before blocked session/news windows

Add session-specific execution rules:

- reduce size during thin Asian-session liquidity
- prefer normal sizing during London/New York overlap
- block rollover
- block or shadow-only around high-impact news
- track per-session win rate and cost

Compare execution policies before production:

```text
fixed size
confidence-based size
rule-based risk execution
RL execution
ensemble-gated execution
```

Promotion should include execution-adjusted metrics:

- net Sharpe after spread/slippage
- profit factor after costs
- max drawdown after costs
- average slippage
- cost as percent of gross PnL
- trade count after execution filters
- rejected-trade count and reasons

### Improve Latency Reporting

Track:

- tick-to-feature latency
- feature-to-inference latency
- inference-to-order latency
- total tick-to-trade latency
- p50/p95/p99 latency

Write latency summaries to JSONL and expose them to Grafana/Prometheus.

## Priority 7: Testing

### Add Artifact Tests

Test that:

- best fold is selected correctly
- nested checkpoint directory is supported
- canonical `haelt_best.pt` is written
- promotion JSON is written
- production checkpoint copy is atomic
- Discord alert names match actual pipeline state

### Add Training Smoke Tests

Keep tiny tests for:

- one mini supervised run
- one fold-selection run
- one promotion-gate reject
- one promotion-gate pass with mocked metrics
- one deploy transaction with mocked ONNX export

### Add Config Tests

Validate that:

- `config/run.yaml` parses
- required sections exist
- active pairs match settings defaults
- checkpoint directory resolves consistently
- data window matches selected cache

## Recommended Next Work

1. Pass real model confidence through inference and into `DisagreementGate` instead of `confidence=None`.
2. Add confidence-threshold tuning per model/fold and persist the selected threshold in `threshold_tuning.json` or the promoted model artifact.
3. Upgrade the existing pair readiness console gate into a structured `pair_readiness_report.json` with per-pair pass/warn/fail status and timestamp-alignment diagnostics.
4. Strengthen checkpoint metadata verification: model name, feature count, sequence length, feature schema hash, expected artifact path, and ONNX sample inference.
5. Clean stale `model_promoted` docs/examples from `monitoring/discord_alerts.py`.
6. Add tests for auto-tune proposal safety, deployment failure transaction records, manifest contents, and live confidence gating.
7. Upgrade the promotion gate from label-proxy PnL to execution-aware forward holdout wherever OHLC, spread, slippage, and latency data are available.
8. Route training-memory suggestions through auditable proposal files before applying them to the next run.


---

# Project Structure Improvement Plan

# Project Structure Improvement Plan

Last updated: 2026-06-12

This plan keeps the trading/training code stable while cleaning the repo so the root only contains entry points, setup files, and project metadata.

## Current Structure Problems

The code packages are mostly organized, but the repository root has drifted back into a mixed workspace:

- Root patch scripts: `patch.py`, `patch_argparse.py`, `patch_live_engine.py`, `patch_p3.py`, `patch_p4.py`, `patch_p4_part2.py`.
- Root debug/fix scripts: `find_terms.py`, `fix_duplicate.py`, `test.py`.
- Root generated artifacts: `promotion_20260611_200939.json`, `promotion_20260612_000210.json`, `parity_features.csv`, `parity_ticks.csv`.
- Two archive locations: `patch_archive/` and `scripts/archive/`.
- Scratch utilities under `scratch/` that should be clearly marked temporary.
- Possible overlap between `compare_models/` and `scripts/compare_models.py`.

## Target Root Layout

Keep the root boring and predictable:

```text
forex_scaling_model/
  README.md
  CHANGELOG.md
  SETUP.md
  pyproject.toml
  requirements*.txt
  run.ps1
  main.py
  setup_ubuntu.sh
  .env
  .gitignore
  .dockerignore
  config/
  data/
  docs/
  scripts/
  training/
  features/
  models/
  inference/
  trading/
  backtesting/
  risk/
  sizing/
  validation/
  monitoring/
  infrastructure/
  tests/
  artifacts/
  logs/
  checkpoints/
  wandb/
```

Root should not contain generated CSVs, promotion JSONs, one-off patch scripts, or temporary debug scripts.

## Recommended Moves

| Current path | Target path | Reason |
|---|---|---|
| `patch.py` | `scripts/archive/patches/patch.py` | Historical patch helper, not a root entry point |
| `patch_argparse.py` | `scripts/archive/patches/patch_argparse.py` | Historical patch helper |
| `patch_live_engine.py` | `scripts/archive/patches/patch_live_engine.py` | Historical patch helper |
| `patch_p3.py` | `scripts/archive/patches/patch_p3.py` | Historical patch helper |
| `patch_p4.py` | `scripts/archive/patches/patch_p4.py` | Historical patch helper |
| `patch_p4_part2.py` | `scripts/archive/patches/patch_p4_part2.py` | Historical patch helper |
| `patch_archive/` | `scripts/archive/patch_archive/` | Consolidate archive locations |
| `find_terms.py` | `scripts/archive/root_cleanup/find_terms.py` | One-off inspection helper |
| `fix_duplicate.py` | `scripts/archive/root_cleanup/fix_duplicate.py` | One-off repair helper |
| `test.py` | `scripts/archive/root_cleanup/test.py` | Ambiguous root test file; real tests live in `tests/` |
| `promotion_*.json` | `logs/mlflow_fallback/` or `artifacts/promotions/` | Generated promotion artifacts |
| `parity_features.csv` | `artifacts/parity/parity_features.csv` | Generated parity check artifact |
| `parity_ticks.csv` | `artifacts/parity/parity_ticks.csv` | Generated parity check artifact |
| `scratch/*` | `logs/scratch/` or keep under ignored `scratch/` | Temporary local experiments |

Do these moves in a separate cleanup commit after confirming none of the patch scripts are still needed as active tools.

## Package Boundaries

Use these package roles consistently:

- `training/`: training loops, dataset build/cache logic, fold selection, promotion orchestration.
- `models/`: model definitions, ensemble modules, distillation helpers.
- `pretrain/`: self-supervised and transfer pretraining code.
- `features/`: feature engineering and schema logic.
- `labeling/`: label generation and target construction.
- `validation/`: walk-forward validation, promotion metrics, MLflow logging.
- `inference/`: ONNX/PyTorch inference wrappers.
- `trading/`: live/shadow trading engine and execution decision flow.
- `risk/`, `sizing/`: portfolio, sizing, and risk controls.
- `monitoring/`: Discord, health reports, alerts.
- `scripts/`: command-line tools that orchestrate the packages.
- `scripts/archive/`: old helpers kept for reference but not part of the active workflow.

## Config Rules

- Keep active run configs in `config/`.
- Keep vendor-only extras in `configs/`.
- Do not duplicate run profiles across both folders.
- If a script takes `--config`, default to `config/run.yaml`.

## Artifact Rules

- Checkpoints: `checkpoints/`.
- Training reports and local runtime logs: `logs/`.
- MLflow fallback promotion JSON: `logs/mlflow_fallback/`.
- Human-readable generated reports or comparison outputs: `artifacts/`.
- Parity files: `artifacts/parity/`.
- Temporary debug output: `logs/scratch/` or ignored `scratch/`.

## Import Safety Checklist

Before moving active code, check:

1. Search imports and subprocess calls for the filename.
2. Move one category at a time.
3. Run AST parsing or py_compile for touched Python modules.
4. Run smoke tests once dependencies are available.
5. Update docs and changelog in the same cleanup.

Suggested checks:

```powershell
rg "patch_live_engine|patch_p4|fix_duplicate|find_terms|parity_features|promotion_" .
python -m py_compile training/train_gpu.py monitoring/discord_alerts.py trading/live_engine.py inference/onnx_inference.py models/ensemble.py
```

## Implementation Phases

### Phase 1: Guardrails

- Update `.gitignore` so new root patch/debug/artifact files are ignored.
- Keep this plan and `docs/FOLDER_STRUCTURE.md` aligned.

### Phase 2: Safe Moves

- Move generated parity CSVs into `artifacts/parity/`.
- Move root promotion JSONs into `logs/mlflow_fallback/` or `artifacts/promotions/`.
- Move one-off root patch/debug scripts into `scripts/archive/`.

### Phase 3: Consolidation

- Merge `patch_archive/` into `scripts/archive/patch_archive/`.
- Decide whether `compare_models/` is generated output or an active package.
- If generated output, move it under `artifacts/compare_models/`.
- If active code, document its public entry point and add tests.

### Phase 4: Enforcement

- Add a lightweight structure check script that fails when new root clutter appears.
- Add the check to the smoke-test workflow once test dependencies are available.

Structure check:

```powershell
python scripts/check_project_structure.py
```

## Next Best Cleanup

The highest-value next cleanup is moving the root patch/debug files into `scripts/archive/patches/` and `scripts/archive/root_cleanup/`. That makes the repo easier to scan without deleting your repair history.
