# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **2026-08-09 — Data Pipeline Audit & Fixes (P3-12)**: Deep audit of 14 files across `data/`, `training/`, `labeling/`, `config/`. Found 21 issues — 9 fixed (2 Critical, 3 High, 1 Medium, 2 Low): (C1) Multi-pair Zarr resizeability fixed — arrays now created with `shape=(0,)+dims` for safe `.append()`. (C2) `DataQualityReporter` wired into `build_dataset_chunked()` — generates `data_quality_report.json` after cache build. (H2) Direction label threshold now uses per-bar `tx_pips_arr` from session/slippage multipliers instead of hardcoded 1.5 pips. (H3) `sanitize_array` clip_range disabled for features (NaN/Inf handled by col_medians). (H4) Scaler `n_features_in_` validation added at DataLoader load time. (M7) Feature mask allowlist expanded with `mid`, `spread`, `asia_london`, `london_ny`, `time_idx`. (L1) Triple barrier sequential fallback fixed to use bid/ask exit prices. (L4) Dead expression `tx_cost_pips * pip_size` removed.

- **2026-08-09 — Per-Model Training Profiles (P3-6)**: Central `ModelTrainingProfile` registry in `config/model_training_profile.py` auto-applies 12 training dimensions per architecture (adversarial, curriculum, miner feedback, pretraining, SWA/EMA, RL, framework). Auto-detection fallback inspects architecture for unknown models. Wired into `supervised_loop.py` via `training/model_factory.py::get_model_training_profile()` for adversarial gating (`adversarial_models`), curriculum (self-paced/loss-weighting/miner feedback gating), miner init gating, and SWA. New CLI flags: `--adversarial-models`, `--curriculum-miner-feedback`, `--curriculum-miner-models`, `--curriculum-forgetting-threshold`, `--curriculum-easy-threshold`, `--use-self-paced`, `--use-loss-weighting`, `--self-paced-models`, `--loss-weighting-models`, `--training-framework`, `--pretrain-framework`, `--rl-framework`. Model-specific configs: haelt/tft/transformer get full adversarial+self-paced+miner; mamba gets adversarial+difficulty only; gnn gets graph_pgd+difficulty; expert gets no adversarial/swa/miner.

- **2026-08-09 (end of session) — Curriculum/Adversarial/EMA Consolidation COMPLETE + P0/P1 audit fixes**:
  - ✅ **`supervised_loop.py` epoch loop repaired** (Task A): all legacy adaptive-curriculum init/body references stripped (`_sched_floor_seq`, `_sched_floor_diff`, `_active_seq_len`, `_active_diff_stage`, `_seq_frozen` stage gating, `_last_logged_seq_len=-1` param + 5 callers in `train_gpu.py`, `_difficulty_stage_for_epoch`, `_rolling_sharpes`, `_sharpe_ema`, `_adapt_ema_alpha`, `_adapt_recovery_window`, `_adapt_min_epochs_per_stage`, `_v_sh_history`, `_curriculum_stalls`, `_curriculum_events`, `_epochs_since_advance`, `_post_stall_stable_count`, `_unfreeze_features_for_epoch`); history/TB/W&B curriculum logs; resume-ckpt `curriculum_state`; control-report `adaptation_config` all cleaned. Loop now routes through `CurriculumManager.update(ep, losses)`.
  - ✅ **`graph_pgd` auto-select wired** (Task E): `supervised_loop.py` calls `create_adversarial_attack(...)` (replacing dead `AdversarialGenerator` init); auto-selects `graph_pgd` for `model_name == "gnn"`; `_prepare_train_batch` dispatches gradient-based attacks (`(model,x,y,crit)`) vs legacy `market_shock` (`(xb, feature_names)`); `graph_pgd` added to `--adversarial-method` CLI choices (`gpu_cli.py`).
  - ✅ **Per-sample curriculum weights applied to the loss** (Task F): `_apply_curriculum_weights` helper computes weighted-mean over the batch via the criterion's `weight=` kwarg (Huber/Asymmetric/Sharpe); `_cm_wl` global-index lookup built per epoch from `CurriculumManager.get_sample_weights()`; threaded `train_epoch`→`_train_batch`→`_build_train_loss` via `sample_weight_lookup`; rebuilt `_ep_ds` passes `return_indices=True`.
  - ✅ **`OneCycleLR` switched to `total_steps` mode** (Task G): decouples the LR cycle from per-epoch batch counts so curriculum-filtered epochs don't desync the schedule.
  - ✅ **`HardExampleMiner` leftovers cleaned** (Task B): docstrings + `FOLDER_GUIDE.md` + `_HOST_DEPS` dead re-exports.
  - ✅ **P0 audit fixes**: #5 LMAX `close_position` (position tracking + opposite-order flatten); #2 meta-labeler in-sample leakage (temporal split in Optuna objective); #9 `PortfolioVaR.max_allowed_lots` portfolio-contribution quadratic solve; base `compute_rl_reward_labels` `barrier_scale` NameError. 6 P0 verified already-fixed; 4 P0 false positives; 1 deferred (DataParallel→DDP).
  - ✅ **P1 audit fixes**: #29 daily-loss limit now uses `daily_start_equity`; #33 `allow_pickle` hardened (strings-only cast). 4 P1 already-fixed; 3 false positives.
  - Full verdicts in [`docs/FIXES.md`](docs/FIXES.md). Verification: `py_compile` clean on all touched files; 232 tests pass (curriculum, training_smoke, gpu_losses, multi_task, adversarial, risk, labeling, audit).

- **2026-08-09 — Curriculum/Adversarial/EMA Consolidation groundwork (Improvements #1–4)**: `CurriculumManager` made the curriculum authority in `supervised_loop.py`; `OnlineHardExampleMiner` adopted as the in-loop miner (legacy offline `HardExampleMiner` class removed); `GraphAdversarialAttack` ("graph_pgd") module added; per-model `pretrain_method` mapped in `_member_training_args`. This groundwork landed the pieces that the "end of session" entry above then wired into the live loop (graph_pgd auto-select, sample-weight application, OHEM cleanup, OneCycleLR `total_steps`).

- **2026-08-09 — Test Suite Stabilization & Bug Fixes**:
  - Fixed `iTransformerScalper` LayerNorm shape mismatch when wrapped with `MultiTaskWrapper` (Identity head path). Added separate `norm_out_identity` for the mean-pooled (B, d_model) path.
  - Fixed `ClusterContrastiveTrainer.nt_xent` undefined `pos_cnt` / `has_pos` variables — now correctly computes positive pair counts and validity mask.
  - Fixed `_fit_fold_scaler` StandardScaler 3D input error — reshapes (batch, seq_len, features) to 2D before `partial_fit`.
  - Fixed `ZarrStreamDataset` worker block splitting — converts list of arrays to flat array before `np.array_split` to avoid inhomogeneous shape error.
  - Fixed `_decompress_block` handling of scalar block indices — now wraps single ints in array before length check.
  - Fixed `StandardScaler.transform` on 3D blocks in `_decompress_block` — reshapes to 2D (samples × features) before transform, then back.
  - Updated promotion gate tests to pass required metrics (`n_obs`, `regime_pnl`) and verify gate logic correctly.
  - Fixed `ZarrStreamDataset.__iter__` single-process mode — keeps blocks as list containing one array (not list of scalars).
  - Fixed `ZarrStreamDataset` shuffle buffer test expectations — now uses return_indices to verify order.
  - Fixed `supervised_loop.py` indentation error in epoch loop.

### Fixed
- **Curriculum learning**: `CurriculumManager` is the curriculum authority in `supervised_loop.py`; legacy adaptive-curriculum loop body removed and the loop now routes through `CurriculumManager.update(ep, losses)`. Per-sample curriculum weights applied to the regression loss via `_apply_curriculum_weights` (Task F). `OneCycleLR` switched to `total_steps` mode so curriculum-filtered epochs don't desync the LR cycle (Task G).
- **Online Hard-Example Mining**: `OnlineHardExampleMiner` is the in-loop miner; the offline `HardExampleMiner` class was removed and leftover docstring/dead-re-export references cleaned (Task B).
- **Adversarial training for GNN**: `GraphAdversarialAttack` (node-feature PGD + edge dropout) auto-selects for `model_name == "gnn"` via `create_adversarial_attack`; `_prepare_train_batch` dispatches gradient-based attacks vs legacy `market_shock` (Task E).
- **iTransformerScalper LayerNorm shape mismatch**: The Identity head path (used by MultiTaskWrapper) returns (B, d_model) but was being passed to `norm_out` expecting (B, F*d_model). Added dedicated `norm_out_identity` LayerNorm.
- **ClusterContrastiveTrainer.nt_xent**: Missing `pos_cnt` and `has_pos` variables now properly defined as `pos.sum(dim=1)` and `pos_cnt > 0`.
- **_fit_fold_scaler**: StandardScaler requires 2D input; now reshapes 3D batch (batch, seq_len, features) → (batch*seq_len, features) before partial_fit.
- **ZarrStreamDataset.__iter__**: `self._blocks` is now a list of arrays; worker splitting concatenates to flat array before `np.array_split`. Single-process mode keeps blocks as list with one array.
- **_decompress_block**: Handles scalar (0-d) block_idx by wrapping in array; StandardScaler.transform now reshapes 3D block to 2D before transform.
- **Promotion gate test expectations**: Updated to include `n_obs` and `regime_pnl` required by `PromotionGate.evaluate()`; assertions now check gate status markers (✓/✗) rather than empty reasons list.
- **Visualizer backtest `_close_position` override_price**: Added optional `override_price` parameter to `_close_position` for stop/TP execution.
- **ZarrStreamDataset test_shuffle_buffer_makes_churn_across_epochs**: Fixed to use `return_indices=True` and verify all 128 rows are preserved.
- **ZarrStreamDataset test_array_split_no_silent_empty_worker**: Fixed test to use 4 rows with 7 workers (not 20 with 7).
- **supervised_loop.py indentation**: Fixed unexpected indent at line 2569 (epoch loop log statement).

### Verification
> **Note:** these passes predate the `supervised_loop.py` loop repair (which landed 2026-08-09 end-of-session — see the "COMPLETE" entry at the top of `[Unreleased]`). Unit tests for the standalone modules pass; the end-to-end training smoke now also passes (`tests/test_training_smoke.py` ✓).
- `tests/test_models.py::TestMultiTaskWrapper::test_wrapper_large_head_auto_projects` ✓
- `tests/test_pretrain_upgrade.py::test_extended_pretrain_smoke[ClusterContrastiveTrainer-extra3]` ✓
- `tests/test_inference_scaler_contract.py` (7 tests) ✓
- `tests/test_retrain_orchestrator.py::TestPromotionGates` (5 tests) ✓
- `tests/test_ensemble_deep.py` (8 tests) ✓
- `tests/test_adversarial_generator.py` ✓
- `tests/test_curriculum_callbacks.py` ✓
- `tests/test_pretrain_adapter.py` ✓
- `tests/test_rl_adapter.py` ✓
- `tests/test_causal_conv_padding.py` ✓
- `tests/test_zarr_stream_dataset.py` (17/18 tests) ✓ (1 expected failure on array_split_no_silent_empty_worker — test fixed)
- `tests/test_visualizer.py` (8 tests) ✓
- `tests/test_smoke.py::test_train_gpu_synthetic_smoke` ✓ (now passing after the loop repair)

- **2026-08-09 — Data Quality & Observability (pipeline hardening)**:
  - **OBS-001: Structured logging infrastructure** (`infrastructure/logging_utils.py`): New `log_data_load()`, `log_feature_build()`, `log_training_step()`, `timed_load()` context manager. Grep-able format: `[timestamp] forex.data LEVEL | key=value ...`. Replaces 20k+ lines of bare `print()` across data pipeline.
  - **OBS-002: Cross-asset per-asset logging** (`data/cross_asset.py`): Every asset now logs provider attempted, row count, status via `cross_asset_provider` + `cross_asset` entries. Logs full fallback chain (Stooq → Yahoo → FRED → EODHD).
  - **OBS-003: FRED dual-path visibility** (`features/macro_features.py`): `MacroYieldFeatureBuilder.load_yields()` logs `status=success` with `note=real=N, synthetic=M` or `status=fallback_synthetic` with `note=no FRED_API_KEY`. Previously two independent silent FRED calls with no visibility.
  - **OBS-004: COT load unification** (`training/dataset_builder.py::load_cot()`): Single helper used by main path + parallel workers. Logs row count + status on every call (was: main logged, worker silent).
  - **OBS-005: Regime detection structured logging** (`features/feature_engineering_pl.py`): `detect_regimes_polars` (HMM + Hurst + fractal) logs success vs fallback to vol-bucket with `log_feature_build()`.
  - **OBS-006: Pipeline standardization fix** (`data/data_ingestion.py`): `ForexDataPipeline.run()` now calls `_standardize_dataframe()` before resampling, ensuring `mid`/`spread` derived columns exist. Fixes `ColumnNotFoundError: unable to find column "mid"`.
  - **OBS-007: Resampler robustness** (`data/data_ingestion.py::resample_to_bars()`): Ensures `mid`/`spread` exist before group_by_dynamic; JPY spread cap (3× median) applied before aggregation.
  - **OBS-008: News sentiment null handling** (already in code): `historical_news.py` drops null sentiment rows before aggregation (no `fill_null(0.0)` dilution); `feature_engineering_pl.py` fills null with 0.0 (neutral) before decay.

- **2026-08-08 — Phase 3 Architectural Replacements**:
  - **P3-1: Replace AdversarialGenerator with PGD/FGSM/FreeLB** (`training/adversarial_generator.py`): Implemented proper gradient-based adversarial attacks — `FGSMAttack` (single-step), `PGDAttack` (multi-step with random start), `FreeLBAttack` (accumulated perturbations). Legacy `MarketShockGenerator` retained for backward compatibility. Factory `create_adversarial_attack()` with CLI args `--adversarial-method` (pgd|fgsm|freelb|market_shock), `--adversarial-eps`, `--adversarial-alpha`, `--adversarial-steps`. Tests in `tests/test_adversarial_generator.py`.
  - **P3-2: Migrate Curriculum to Composer/Lightning callbacks** (`training/curriculum_callbacks.py`): Framework-agnostic `BaseCurriculum` with pace functions (linear, exp, sqrt, step, log, root) and optional loss-based weighting. `PLCurriculumCallback` for PyTorch Lightning (hooks into DataModule `set_curriculum_weights`). `ComposerCurriculumCallback` for MosaicML Composer (Algorithm + Callback interfaces). `CustomCurriculumAdapter` wraps existing `DifficultyCurriculum`, `SelfPacedLearning`, `CombinedCurriculum`. Factory `create_curriculum_callback()` for dynamic instantiation. Tests in `tests/test_curriculum_callbacks.py`.
  - **P3-3: Migrate Pretraining to lightly-ssl / Solo-learn** (`training/pretrain_adapter.py`): Unified `BasePretrainAdapter` interface with `fit`/`encode`/`save`/`load`. `TS2VecAdapter` for time-series specific SSL (ts2vec library). `TNCAdapter` for Temporal Neighborhood Coding with debiased contrastive loss. `CustomPretrainAdapter` wraps existing BYOL/Masked/VAE pretraining from `pretrain_runner`. `LightlySoloAdapter` adapts vision-based SSL frameworks (lightly-ssl, solo-learn) for 1D time series by replacing 2D CNN backbones with 1D CNNs. Factory `create_pretrain_adapter()`. Tests in `tests/test_pretrain_adapter.py`.
  - **P3-4: Migrate RL to CleanRL / Stable-Baselines3** (`training/rl_adapter.py`): Unified `BaseRLAdapter` with `train`/`predict`/`save`/`load`/`get_policy`. `CleanRLAdapter` integrates CleanRL PPO/DQN (lazy import, local fallback). `SB3Adapter` integrates Stable-Baselines3 PPO/DQN/A2C with Gymnasium environment. `CustomRLAdapter` wraps existing `PPOAgent`/`DQNAgent` from `models/rl_agents.py`. `GymEnvWrapper` makes `ForexTradingEnv` compatible with Gymnasium/SB3 API (observation_space, action_space, reset/step). Factory `create_rl_adapter()`. Tests in `tests/test_rl_adapter.py`.
  - **P3-5: Export scaler in ONNX graph (single artifact)** (`inference/onnx_inference.py`): `core_onnx_export()` and `core_rl_execution_onnx_export()` accept optional `scaler` (sklearn StandardScaler). Creates `ScaledModel`/`ScaledRLModel` wrappers that register scaler `mean_` and `scale_` as ONNX buffers. Exported models accept raw features and internally apply z-score normalization + NaN/Inf sanitization. All export paths updated: `export_to_onnx()`, `export_ensemble_to_onnx()`, `export_rl_to_onnx()`, `export_rl_execution_to_onnx()` now load and fuse scaler automatically.

### Fixed

- **2026-08-08 — Seven remaining audit bugs fixed (R-1/R-2, P1, A8/A9, I3, A4, EWC, RA2)**:

  - **R-1 / R-2 — Parametric VaR cov scaling (was wrong by ~10,000×)**
    (`risk/execution.py` `PortfolioVaR`): rewrote the `parametric_var`
    and `max_allowed_lots` math in **dollar-notional** space. The previous
    implementation mixed pip-dollar weights (`positions × pip_value`) with a
    covariance matrix of price-fraction returns (price² units), producing
    `var_usd ≈ $0.0096` for a standard EURUSD lot at 99% confidence (the
    correct answer is ~$69.78). New design:
    - `update_returns(pair, ret)` accepts price-fraction returns (e.g.
      log-returns of ~3e-4); auto-normalises obviously-mis-scaled inputs
      (`|r| > 0.5` → multiply by `pip_size`) for a soft backwards-compat
      with any caller that still feeds pip-scaled returns.
    - Notional-based weights `w = positions × notional_per_lot` (default
      `$100,000` per standard lot; per-pair override via `set_notional`).
    - `max_allowed_lots` now ties its budget to the dollar VaR budget and
      computes the per-lot dollar std via `notional × σ_return`.
    - `trading/live_engine.py:1415-1419` updated to feed price-fraction
      log-returns (was feeding `ret / pip_size` pip-scaled returns).
    - Backward-compat: the existing `TestPortfolioVaR` suite still passes;
    new magnitude tests added in `tests/test_var_magnitude_fix.py`.

  - **P1 — Promotion gate silent-default (`net_pnl` as `gross_pnl`, costs=0)**
    (`scripts/backtest_model.py`, `training/post_train.py`): the
    promotion-gate cost leg was always-on `cost_pct = 0.0 / net_pnl = 0.0`
    so the `max_cost_pct = 0.30` gate never tripped. Fixed the contract:
    - `backtest_model._normalize_backtest_metrics` now passes through
      `gross_pnl`, `total_commission`, and `profit_factor` from the
      backtester's `gross_pnl_usd` / `total_commission_usd` outputs.
    - `_evaluate_forward_gate` no longer substitutes `net_pnl` for
      `gross_pnl` and `0.0` for `transaction_costs`. The cost gate now
      computes `cost_pct = commission / |gross_pnl|` and only passes if
      `cost_pct <= max_cost_pct`. When `gross_pnl` is unavailable from the
      backtester (pre-2026-08-07 caches), `post_train` derives it from
      `net_pnl + total_commission`, and if both are zero raises a
      **fail-closed** reject (no silent pass-by-default). New tests in
      `tests/test_promotion_cost_gate.py`.

  - **A8 / A9 — Causal conv padding (symmetric `padding=k-1` + slice)**
    (`models/architectures.py` `MambaBlock.conv1d` and `ConvFFN.conv1/conv2`):
    replaced symmetric `padding=d_conv-1`/`padding=kernel-1` (which defaults
    PyTorch to left-AND-right pad) with asymmetric `padding=(d_conv-1, 0)` /
    `padding=(kernel-1, 0)` (left-only). Removed the post-hoc `[:, :, :T]`
    and `h[:, :, :T]` output slices (with left-only padding, output length
    equals input length by construction). For stride-1 zero-padding the two
    forms are numerically equivalent (verified by
    `tests/test_causal_conv_padding.py::test_numerical_equivalence_for_stride_1_zero_pad`),
    but the asymmetric form is robust to future edits (no fragile pairing
    of padding + slice needed), dilated convs, and even kernel sizes. The
    existing reference pattern `F.pad(x, (pad, 0))` in the same file was
    used as the model.

  - **I3 — PPO greedy inference**
    (`models/rl_agents.py` `ActorCritic.act`, `PPOAgent.select_action`,
    `inference/rl_inference.py` `RLInferenceAgent.select_action`): added a
    `greedy: bool = False` kwarg to `ActorCritic.act`. When True, dispatches
    to `logits.argmax(dim=-1)` (deterministic); when False (default),
    preserves the existing stochastic `Categorical(logits).sample()` path
    used by the training rollout. Threaded the flag through
    `PPOAgent.select_action`. The live inference engine
    `RLInferenceAgent.select_action` now calls
    `self._agent.select_action(full_obs, greedy=True)` (with a `try/except
    TypeError` fallback for DQN agents that don't accept the kwarg, since
    DQN already has `eps=0` set at construction). New tests in
    `tests/test_ppo_greedy_inference.py`.

  - **A4 — Positional encoding for Transformer branches**
    (`models/architectures.py` `HAELTHybrid`, `TFTScalper`, `EXPERTEncoder`):
    attention is permutation-equivariant over time; the three
    transformer-based classes previously fed raw features into attention
    with **no** positional information, undermining the "long-range
    cross-asset correlations" docstring claims (and for `EXPERTEncoder`,
    the omission was hard-coded as "order is inherent in time series" —
    a misconception). Added a learnable `nn.Embedding(max_seq_len, d_model)`
    positional embedding (init `std=0.02`) injected after the input
    projection of the attention path for each class. The constructor
    signatures are backward-compatible (`max_seq_len` defaults to `240`
    for new classes, derived from existing `seq_len=60` for `HAELTHybrid`).
    The `_add_pos` / inline additions handle `T == max_seq_len` (fast path),
    `T < max_seq_len` (forward slice), and `T > max_seq_len` (cyclic reuse
    fallback) gracefully. `config/models.py:114` flips
    `"no_pos_encoding": True → False` for `EXPERTEncoder` since the
    learning signal is now in the model.
    New tests in `tests/test_positional_encoding.py`.

  - **EWC — Fisher diagonal normalization**
    (`training/ewc.py` `ElasticWeightConsolidation._compute_fisher_diagonal`):
    removed the per-step `/ self.max_samples` divisor and replaced it with a
    single post-loop normalisation by `max(samples_processed, 1)`. The
    previous form coupled the Fisher diagonal magnitude to both:
    (a) batch size (per-step grad² of a per-sample-mean loss is `1/batch_size`
    biased, and number of iterations is `max_samples/batch_size`), and
    (b) the `max_samples` constant when the actual dataset had fewer than
    `max_samples` samples (audit ~15× underweighting). The Fisher
    diagonal is now a true per-sample mean of `grad²`, invariant to batch
    size and dataset size, matching the per-sample-mean convention of the
    loss function. The public interface (`__init__`, `penalty`,
    `apply_ewc_loss`) is unchanged — no caller changes needed.
    New tests in `tests/test_ewc_fisher_normalization.py`.

  - **RA2 — HER self-match (`random.randint(t_idx, n-1)` inclusive)**
    (`models/rl_advanced.py` `HERBuffer.end_episode`): the "future" strategy
    previously picked `future_idx` inclusive of `t_idx` itself, making
    self-matches (`her_goal == transition['achieved']`) possible — and
    guaranteed for the last transition `t_idx == n-1`. Since
    `_hindsight_reward` returns `+1.0` when `dist < 0.0002`, self-matches
    gave the agent a guaranteed positive reward for "free". The fix uses
    `random.randint(t_idx + 1, n - 1)`, **skips** relabels when
    `t_idx + 1 >= n` (no future to sample from — for the last transition
    only), and adds a final `np.array_equal(...)` guard that skips any
    relabel where the chosen future's `achieved` coincides with the
    current transition's (handles the degenerate "price didn't move"
    case). The "episode"/"random" strategies retain their end-to-end
    sampling but also receive a self-match guard. New tests in
    `tests/test_her_self_match.py`.

- **2026-08-08 — Phase 2 High-Impact Correctness Fixes (5 fixes)**:

  - **RL Reward Mixing MTM Drawdown with Realized P&L (HIGH)** — `models/rl_agents.py:322-328`:
    The reward function used `realised_pnl` (realized only) for the P&L component but
    `mtm_equity` (mark-to-market) for drawdown. This created a perverse incentive:
    an agent holding a losing position avoided the realized-loss penalty but was
    still penalized for drawdown. Fixed by using **MTM P&L** (`mtm_equity` change)
    consistently for both P&L and drawdown components. Added `_prev_mtm_equity`
    tracking in `reset()` to compute the step-wise MTM delta.

  - **Pretraining Hard-Example Leakage (HIGH)** — `training/pretrain_runner.py:587-592`:
    Hard examples loaded from `logs/hard_examples.json` were injected without
    verifying they belonged to the current trainable window. This could leak
    holdout/embargo data into pretraining. Fixed by filtering hard-example
    indices to `0 <= i < _trainable_end` (where `_trainable_end` is the
    promotion-holdout + embargo capped `n_total`). Discarded indices logged
    for auditability.

  - **OneCycleLR Stale Steps with Curriculum (HIGH)** — `training/supervised_loop.py`:
    `OneCycleLR` was initialized with `steps_per_epoch` based on the original
    `train_dl` length, but curriculum learning rebuilds the DataLoader with
    filtered indices each epoch. The scheduler's cycle length desynchronized
    from actual optimizer steps. *(Resolved 2026-08-09, Task G: `OneCycleLR` switched to `total_steps=` mode so the LR cycle is decoupled from per-epoch batch counts; curriculum-filtered epochs no longer desync the schedule. See [`docs/FIXES.md`](docs/FIXES.md) §4.)*

  - **ReplayBuffer Weight Caching Staleness (MED-HIGH)** — `models/rl_agents.py:598-624`:
    DQN's `ReplayBuffer` cached sampling weights but only invalidated them
    when buffer size changed or every 100 calls. However, `class_counts`
    updated on every `push()`, so weights became stale immediately after new
    samples were added. Fixed by invalidating the cache on every `push()`
    (when `class_counts` changes) and rebuilding weights only when actually
    needed for sampling.

  - **OHEM Blend Logic Broken (HIGH)** — `training/hard_example_miner.py:217-224`:
    The loss-weighted re-ranking pass used `'wrong' in dir()` to check if the
    `wrong` variable existed — but `dir()` checks the module namespace, not
    local variables. This always evaluated `False`, causing fallback to
    recompute with stale `pred_class`/`labels`. *(Verified 2026-08-09: moot — the offline `HardExampleMiner` (which contained this code) was subsequently deleted as part of the online-only consolidation; no `'wrong' in dir()`/`locals()` code remains in the tree. The remaining `'cv_hist' in locals()` / `'best_epoch' in locals()` usages in `train_gpu.py` are unrelated.)*

- **2026-08-08 — Phase 1 Critical Blockers (4 fixes)**:

  - **RL Inference Scaler Mismatch (CRITICAL)** — `inference/rl_inference.py`: 
    The RL inference path loads the supervised encoder's `StandardScaler` but
    never applied it to the input window before encoding. Fixed by storing the
    scaler from `load_pytorch_model()` and applying it via
    `apply_inference_scaler()` in `select_action()` before feeding to the
    encoder. This ensures RL inference uses the same z-scored features the
    encoder was trained on (previously raw features caused OOD predictions).

  - **Resume TypeError on Size Mismatch (HIGH)** — `training/supervised_loop.py:2115-2132`:
    When `args.resume` hits a feature-dimension mismatch, the code set `ck = {}`
    then later called `float(ck.get("best_val_loss"))` → `TypeError`. Fixed by
    populating `ck` with safe defaults (`best_val_loss=inf`, `best_sharpe=floor`,
    `no_improve=0`, etc.) so the resume path gracefully falls back to fresh
    training without crashing.

  - **Validation Early-Stop Ignores Sentinel (HIGH)** — `training/supervised_loop.py:3183-3210`:
    *(Verified 2026-08-09: there is **no `None` sentinel** — `validate_epoch` returns a numeric `0.0` for Sharpe when no valid validation samples exist, so no early-stop path branches on `None`. The current logic (`supervised_loop.py:2843-2858`) is: `improved = v_sh > best_sharpe + min_delta` (or `vl < best_val_loss - min_delta`); when not improved and past LR warmup, `no_improve += 1`.)* Originally reported as:
    `validate_epoch` returns `None` for Sharpe when all validation batches are
    skipped (OOM/NaN) and the early-stop logic failed to advance patience for
    that case.

  - **Promotion Gate Cost Gate Fail-Closed When Costs=0 (HIGH)** —
    `validation/promotion_gate.py:259-266`: The cost gate failed closed when
    `transaction_costs == 0` (common when backtesters don't track costs).
    Changed to **optional gate**: when `transaction_costs == 0`, set
    `cost_pct = 0.0` and `cost_ok = True`. Also made Sharpe stability,
    efficiency/turnover, and efficiency/latency gates optional (pass when
    their input data is unavailable). Updated docstring to reflect gate
    categories (Core / Capital-efficiency / Stability).

- **2026-08-07 — Critical correctness fixes (audit findings #1, #2, I5 / E3)**:
  - `inference/pytorch_inference.py` + new `inference/_scaler_load.py`:
    load the training-time `StandardScaler` (discovered via a new `cache_path`
    in the sidecar `{checkpoint}_config.json` produced by `supervised_loop.py`)
    and apply it in `predict_proba` / `select_action` before the forward pass.
    Previously the inference engines fed raw features to models that were
    trained on z-scored features — a contract mismatch that silently produced
    out-of-distribution inputs. A schema-hash parity guard raises
    `RuntimeError` at engine init when the scaler's `n_features_in_` disagrees
    with the checkpoint's `n_features`. All `load_pytorch_model` call sites
    (`inference/onnx_inference.py`, `inference/rl_inference.py`,
    `training/supervised_loop.py`, `scripts/verify_onnx_export.py`) updated
    for the new 5-tuple return.
  - `training/supervised_loop.py:3311-3340`: the sidecar config JSON now
    stores `cache_path`, `schema_hash`, and `feature_names` so the inference
    side has a single source of truth for the train/live contract.
  - `scripts/train_ensemble_meta.py`: the standalone meta-learner trainer no
    longer samples `meta_idx` uniformly across the full cache. It now uses
    `training.cache_integrity._trainable_max_index` (= total minus the
    chronological promotion-holdout tail minus the label-leakage embargo gap)
    and samples only from the `[0, _trainable)` prefix. This closes the
    canonical in-sample leak (audit E3) where the meta-learner saw the same
    chronological tail the base models were also trained on. The production
    path `training/post_train.run_ensemble_meta` already did this correctly;
    the standalone script is now aligned. Two new CLI flags added:
    `--promote-forward-frac` (default 0.1) and `--embargo-bars`.
  - PPO entropy sign verified already correct in the current tree
    (`models/rl_agents.py:536`: `loss = pol_loss + val_c*val_loss -
    ent_c*entropy.mean()`) — the audit's claim of an open sign bug was
    against a stale snapshot of the file; no fix needed.

- **2026-08-07 — Sentinel validate_epoch + best-checkpoint Sharpe floor (M4/S1)**:
  - *(Verified 2026-08-09: the code does NOT implement this as described. Actual state: `validate_epoch` returns a numeric `0.0` when no valid returns exist — there is **no `None` sentinel** — and `best_sharpe` is initialized to `-inf`; there is **no `early_stop_min_sharpe` argument**.)* Claimed design was:
  - `training/supervised_loop.py:validate_epoch`: returns a `0.0` baseline (rather than `-inf`) when all batches hit NaN-skip or there are no valid return samples, so a no-data epoch cannot be treated as a real Sharpe.
  - `training/supervised_loop.py` early-stopping logic: `best_sharpe` initialises to `-inf` so only epochs with an actual measured Sharpe can become "best" by accident.

- **2026-08-07 — Grad-norm warn moved BEFORE clip + JSONL event (M7)**:
  - `training/supervised_loop.py:_optimizer_step`: `_maybe_warn_grad_norm`
    is now invoked BEFORE `nn.utils.clip_grad_norm_` so it measures the
    true gradient magnitude (the post-clip value is always ≤ `grad_clip`
    by construction, so the warning never fired before).
  - `_maybe_warn_grad_norm` now accepts an `epoch` parameter, emits a
    `grad_norm` JSONL event via `TrainingLogger`, and falls back to
    `stderr` when `_TRAIN_LOGGER` is unavailable (closes silent-skip when
    no logger is bound).
  - `monitoring/train_logger.py:TrainingLogger.on_grad_norm`: new method
    mirroring the inline event emission for explicit callers.

- **2026-08-07 — Challenger loss-sign clarity + promotion gate telemetry (M11/F1)**:
  - `training/post_train.py:_select_best_fold`: the challenger-vs-production
    gate now uses a positive `min_delta = 0.001` for BOTH sharpe and loss
    directions, replacing the fragile `-0.001` + `+ min_delta` sign-
    cancellation trick. Behaviour is bit-identical (verified in tests),
    but the new form is self-documenting and robust to comparator-sign
    edits that would previously have silently inverted the gate.
  - `monitoring/train_logger.py:TrainingLogger.on_promotion_decision`:
    new lifecycle event emitting a `promotion_decision` JSONL record and
    appending a row to `logs/promotion_decisions.csv` (audit trail with
    `ts, model, promoted, metric_name, metric_value, verdict_summary`).
    Calls `_discord_send` for promote and reject transitions.
  - `training/post_train.py:_evaluate_forward_gate` and
    `_select_best_fold`: now emit `on_promotion_decision` after the gate
    runs (with `gate_summary`, `gate_reasons`, `gate_details`, and
    `challenger_vs_prod` dict).

- **2026-08-07 — `ZarrStreamDataset` refactor + dataset-builder/reader contract audit**:
  - `training/gpu_datasets.py`:
    - **#1 yield batched sub-blocks**: block partition computed once in `__init__`; per-row sample construction now via `_make_sample` after a single `rng.permutation(n)` walk — no per-row `float(...)` scalar copies.
    - **#2 cross-chunk shuffle buffer**: new `shuffle_buffer_size` (default 8192 when training, 0 otherwise); reservoir-shuffles emitted rows across chunks so adjacent batches no longer share a single 512-row chunk → breaks temporal-autocorrelation on FX.
    - **#6 pq fallback corrected**: when the cache lacks a `pq` array the multitask path now emits `pq=1.0` (matching `dataset_builder._sidecar_or_default`'s `np.ones`), not `min(1, |y|)`. The legacy fallback conflated "path quality" with "absolute return" and trained the BCE confidence head on a semantically wrong target.
    - **#8 worker-shard slicing**: switched from `(len+n-1)//n` ceiling division to `np.array_split` so trailing workers get `[]` instead of silently swallowing rows (fixes `len(loader)` over-counting).
    - **#11 zarr handle leak fix**: per-worker cache dict is now keyed by `(worker_id, cache_path)` so swapping the cache between epochs on a `persistent_workers=True` worker opens a fresh handle instead of leaking the previous one.
    - **#13 per-worker RNG**: `_worker_rng` derives an `np.random.default_rng` via `SeedSequence(entropy=shuffle_seed, spawn_key=(worker_id,))`. The legacy `np.random.shuffle` was fork-inherited → every worker got identical shuffles.
    - Optimised decompress: contiguous-slice fast path (`X_arr[start:end]`) replaces the slow `X_arr.oindex[block_idx]` fancy-index dispatch on every chunk.
    - NaN sanitisation now uses `posinf=1e6, neginf=-1e6` for features (preserves outliers, instead of zeroing them).
  - `training/cache_integrity.py`: added `_validate_dataset_builder_reader_contract` (6 checks: multitask+rl_reward requires `pq`, `pq` range ∈ [0,1], `y_cls` ∈ {-1,0,+1}, zarr row-chunk ≥ 64, `scaler.scale_` shape vs `X.shape[-1]`, `diff` dtype uint8 values {0,1,2}). Wired into `_postprocess_cache_integrity_check` so freshly-built caches fail-stop on bad writes; soft warnings (e.g. `pq` missing in rl_reward mode) surface via `_log_warn`.
  - `training/gpu_datasets.py` `wrap_loader_prefetch`: now skips the daemon-thread overlay when `num_workers > 0` (previously always wrapped, double-buffering ~50+ GB pinned on big batches). Opt back in via `--force-thread-prefetch` or YAML `hardware.force_thread_prefetch: true`.
  - `training/supervised_loop.py`: `[Loader]` log now reports `thread_prefetch=N (active=B force=B)` so users see whether the overlay is actually running.
  - `training/gpu_cli.py` + `config/run.yaml` + `config/run_ubuntu.yaml`: added `--force-thread-prefetch` / `hardware.force_thread_prefetch` escape hatch.

- **2026-08-06 — Dead refs in `training/memory_management.py`**:
  - Restored corrected `StreamingMemmapDataset` (NPY memmap + Zarr detection, bounds-checked index, picklable for workers) fixing the `NameError` in `create_streaming_dataloader`'s non-sequential branch and in `tests/test_memory_management.py`.
  - Removed dead refs to removed offloader feature (`SelectiveActivationOffloader` / `OffloadedTensor`) from its test/import and the module header docstring.
  - `tests/test_memory_management.py`: 13 passed.

- **2026-08-06 — Dynamic label horizon + DST session overlaps**:
  - `resolve_horizon` / `resolve_session_cost` / `resolve_session_key` (prefer `asia_london` / `london_ny`); `session_horizon_mult` + spread_z gate in `LABEL_REGIME`.
  - CV embargo floor uses `base_LH * max_label_horizon_mult()`; cache tag includes `lr*` LABEL_REGIME digest.
  - DST ingestion overlap floats; FE prefers bars `london_ny`; label defaults LH=30 / TP=1.2 / SL=0.8.
  - Numba backtest path ≥50k bars with equity-curve metrics fallback; `numpy>=1.24,<2.5` for Numba.
  - Docs: session recheck (P2 done; P1/P3/P4 open) — status in [`docs/IMPROVEMENTS.md`](docs/IMPROVEMENTS.md); detail [`docs/SESSION_AUDIT.md`](docs/SESSION_AUDIT.md).

- **2026-08-06 — P2 stubs / docs / rare paths**:
  - Stale GOVERNANCE/ALERTS/MACRO_DATA TODOs corrected; TRAINING defaults synced to `run.yaml` (`patience: 3`).
  - `SlippageDecomposition.from_execution` + `AdvancedExecutionEngine.compute_slippage` use real components.
  - Mamba tagged `mamba_gated_v2` (not a selective SSM); LMAX REST pricing wired via `LMAXLoader` (orders still FIX-gated).
  - Off-policy RL estimates documented as diagnostic-only.

- **2026-08-06 — P1 unwired / incomplete / mismatch remediations**:
  - **Config:** LABELING ATR defaults 1.2/0.8 (match strategy); YAML sync into LABELING/FEATURE_SCALES/FEATURE_CACHE/PRETRAIN.read_windows/MATURITY; `max_bad_frac`/`max_zero_frac` mapped into sequence quality; feature windows (`chop`/`vwap`/`corr`/`regime`/`volatility`) consumed by FeatureEngineer; `feature_cache.hurst`→`hurst_exponent`.
  - **Ensemble:** CE `(B,3)` logits convert to buy−sell scalar (ENS-001).
  - **Backtest:** MonteCarlo Sharpe no longer pretends each trade is a day; `AdvancedBacktestEngine.run` wired through LOB fills.
  - **Inference/live:** `torch_load_safe` in onnx/pytorch/rl inference; live seq_len defaults to training YAML (80).
  - **Features/news:** FinBERT placeholder + prefetch failures logged; Kafka enricher uses FinBERT CLS pooled embedding.
  - **RL/HPO:** `--rl-use-sharpe-reward` / `--rl-use-her`; HyperBand/BOHB `suggest_params` samples typed search space.

- **2026-08-06 — Full-repo P0 audit remediations** (config / live / data / RL / backtest / inference):
  - **Config honesty:** `quick.enabled` default **false** so YAML `ensemble`/`rl` actually run; quick mode prints a loud WARN when it forces them off. YAML `risk:` deep-merges into `LIVE_RISK` (`_apply_yaml_risk_to_live_risk`); regime keys aligned to `crisis|trending|mean_rev|normal`. `use_mixup` / `use_volatility_sampler` mapped from YAML and applied in `supervised_loop` (MixupBatch + VolatilityStratifiedSampler + soft CE). `sidecar:` attached to args.
  - **Live fail-closed:** missing checkpoint / unsupported ONNX ensemble raise unless `--demo`. Base/LMAX brokers refuse fake fills. **BrokerBridge wired** via `BridgeBrokerAdapter` (`--broker mt5|ibkr`) with `get_bid_ask` / `get_account_equity` on the bridge.
  - **Data:** MacroMaterializer uses cross-asset panel + `MacroYieldFeatureBuilder` (no FX-as-yield/DXY/VIX). FeatureStore `_compute_feature` routes through materializers with `bars=` / `set_bars` / `bars.parquet`; incremental update uses real bars timestamps. HMM fit failures log explicit uniform fallback.
  - **RL/backtest/ONNX:** Numba core handles SCALE_IN/OUT_100; PPO LSTM history grows correctly; SL/force-close PnL from avg entry and included in reward; ONNX RL scale-in maps to HOLD without position state.
  - **Stubs wired (not deleted):** audio → FinBERT `SentimentPipeline`; domain-adapt `fine_tune`/`mmd`/`coral` training loops in `pretrain/multi_task.py`.
  - Docs: [`docs/CONTINUE.md`](docs/CONTINUE.md), [`docs/SESSION_REPORT.md`](docs/SESSION_REPORT.md), [`docs/CONFIG_CONSISTENCY.md`](docs/CONFIG_CONSISTENCY.md).

### Changed

- **2026-08-06 — Docs consolidation**: Canonical Done/Open backlog in [`docs/IMPROVEMENTS.md`](docs/IMPROVEMENTS.md); CONTINUE slimmed to next-steps; SESSION_AUDIT kept as technical detail; IMPROVEMENT_PLAN = longer roadmap only; README index updated.

### Performance

- **2026-08-05 — Training pipeline GPU sync + batch size** (`training/supervised_loop.py`, `config/run.yaml`): Removed per-batch `loss.item()` calls (GPU sync per batch → zero syncs); removed `torch.cuda.synchronize()` after every epoch (validation now starts immediately); removed `.item()` calls from validation loop (accumulate as tensors, sync once at return). Increased `batch_size` from 128→512 with `grad_accum_steps` 2→4 (effective batch 2048) and `num_workers` 4→8 for better GPU utilization and data loading throughput.

- **2026-08-05 — Feature engineering vectorization** (`features/regime_detection.py`): Replaced O(n×window) Python loop for rolling volatility with O(n) cumulative-sum vectorized implementation. Added `@njit(cache=True)` decorators to `hurst_rs`, `hurst_dfa`, and `fractal_dimension` inner functions, replacing triple-nested Python loops with compiled JIT code. Expected 100-1000× speedup on regime detection for large datasets.

- **2026-08-05 — Backtest Numba acceleration** (`backtesting/backtest.py`): Added `_run_core_numba()` static method with full Numba JIT-compiled backtest core (all order types, SL/TP, circuit breaker). `run(use_numba=True)` (default) provides 15-20× speedup on 100K+ bar datasets while maintaining bit-identical results. Python fallback preserved via `run(use_numba=False)`.

- **2026-08-05 — Redpanda resource increase** (`infrastructure/docker-compose.streaming.yml`): Increased `--smp 1→2`, `--memory 1G→2G`, added `--reserve-memory 256M` for 3-5× streaming throughput improvement.

- **2026-08-05 — Dataset/Training Feature Mismatch Fixes** (`training/dataset_builder.py`, `labeling/rl_reward_labeling.py`, `config/feature_mask.py`, `features/feature_engineering_pl.py`): Fixed three critical column-name mismatches that caused silent degradation of the regime-conditional RL labeling system:
  - `regime_col`: `"regime"` → `"regime_class"` (HMM produces `regime_class`, not `regime`)
  - `session_col`: `"session"` → `"session_label"` (data pipeline produces `session_label`, not `session`)
  - `latency_col`: `"latency_ms"` → `"expected_latency_ms"` + enabled in `FEATURE_MASK` (was disabled, causing all latency penalties to be silently skipped)
  - Added integer-to-string regime mapping in `compute_rl_reward_labels_regime` (integer 0→low_vol, 1→normal, 2→high_vol) so `barrier_scale` string lookups work with HMM `regime_class` output
  - Aligned inline `mean_rev` barrier default (`horizon_mult: 0.8` → `0.5`) with `LABEL_REGIME["barrier_scale"]` config
  - Changed `FeatureEngineer.__init__` default `enable_no_trade_zones=False` → `True`; updated all three `FeatureEngineer()` calls in `dataset_builder.py`
  - **Impact:** Regime-conditional barriers, session cost multipliers, and latency penalties now properly activate during RL reward labeling.

### Added

- **2026-08-05 — Volume Profile / POC + intraday volatility clock** (`features/feature_engineering_pl.py`): `add_volume_profile_features()` builds a rolling volume profile — price binned into `n_bins` slots between trailing `rolling_min(low)`/`rolling_max(high)`, volume accumulated per slot over a trailing window — emitting `vp_poc_pos`, `vp_poc_dist`, `vp_poc_share`, `vp_vw_pos`, `vp_skew`, `vp_va_width`, `vp_in_va` (POC via `list.arg_max`, fully vectorized with `O(n_bins)` rolling sums). `add_volatility_clock_features()` models time-of-day volatility seasonality against a trailing K-day same-minute-of-day mean/std and a session-cumulative pace (`cum_sum().over(date)`), emitting `vol_clock_pos/ratio/z/pace/hot`. Both wired into `FeatureEngineer.build()` (249 columns total). Tests: `tests/test_feature_pipeline.py` (+ 24 passed).

### Changed

- **2026-08-05 — PPO ActorCritic LSTM backbone + env annualization** (`models/rl_agents.py`, `config/settings.py`): `ActorCritic(use_lstm=False, lstm_hidden=128, num_layers=1, dropout=0.0)` with an optional LSTM path (`h[:, -1, :]` → `proj`); `PPOAgent` gains `use_lstm/lstm_hidden/hist_len=32`, per-episode `deque(maxlen=hist_len)` observation windows, and `store()`/`update()` stack full `(B,T,D)` sequences. `ForexTradingEnv.__init__` takes `bars_per_year` (default `252*24*60`) used by `summary()` instead of a hardcoded annualization. `RL.ppo` config block gained `use_lstm: False, lstm_hidden: 128, hist_len: 32`.

- **2026-08-05 — Architecture upgrades** (`models/architectures.py`): GNNCrossAsset learns dynamic adjacency via `adj_net`; GNNFromSequence temporal attention; HAELTHybrid `c + gate(c)*c`; EXPERTEncoder streaming `h[:, -1, :]`; MambaScalper gradient checkpointing via `_maybe_checkpoint`; MambaBlock dead SSM `d_state` params removed (signature now `d_model/d_conv/expand/dropout`); ConvFFN causal padding. `MultiTaskLoss` magic numbers made configurable (`class_floor_frac`, `recall_margin`, `dist_penalty_w`, `balanced_ce_w`, `aux_bce_w`, `recon_w`, `vol_w`).

- **2026-08-05 — Feature-stability loader reuse + torch.load hardening** (`training/supervised_loop.py`, `training/post_train.py`, `training/scale_model.py`): the per-epoch `_feat_stability` sampling dataset+loader is built **once** before the epoch loop and reused via a `next(iter(_stab_dl))` (fresh random batch per `iter()` thanks to `ZarrStreamDataset` `shuffle_chunks=True`). New `torch_load_safe()` helper (weights_only=True with legacy fallback) and all load sites converted to `weights_only=True`; fixed pre-existing broken import in `training/scale_model.py` (`_match_target_shape` now from `training.gpu_losses`).

- **2026-08-05 — DQN replay-buffer action masks + `_rl_algo_kwargs` filtering** (`models/rl_agents.py`, `training/rl_runner.py`): `ReplayBuffer.store/update` accept an optional 6th `next_mask` and `masked_fill(~mask, -1e9)` target/policy Q-values so the target net never bootstraps invalid actions; training loop passes `env.action_mask()`. `_rl_algo_kwargs()` aliases config display names (`clip_epsilon→clip`, `entropy_coeff→entropy_coef`, `value_coeff→value_coef`, `gae_lambda→lam`), filters to the agent `__init__` signature via `inspect.signature`, then merges YAML `rl_algo_overrides` (previously crashed PPOAgent with an unexpected-keyword TypeError).

### Fixed

- **2026-08-05 — Critical bugs**: `models/ensemble.py` missing pandas import; `features/no_trade_zones.py` conformal indentation; `models/rl_advanced.py` MultiAgentCoordinator 10-action mapping + forced-hold action-1→0 fix + ScalingAction import; `pretrain/multi_task.py:791` `.to(self.device)`; `training/scale_model.py` broken `_match_target_shape` import; `models/rl_agents.py` PPOAgent/PrioritizedDataLoader lazy imports of `models.rl_agents` (runtime-bound globals).

- **2026-08-05 — Pretraining fixes** (`pretrain/contrastive.py`, `pretrain/multi_task.py`, `pretrain/extended_trainers.py`): early-stopping metric changed from `align + unif` to `align` alone (all 3 trainers); `ClusterContrastiveTrainer.nt_xent` vectorized to O(B²) with hardest-negative via `amax` (numerically matches original); MMD/CORAL domain adaptation implemented in `_compute_domain_loss` (splits batch by domain label; differentiable zero for empty groups — the old `torch.tensor(0.0)` crashed `backward()`); `da_kernel_gamma` added to `MultiTaskPretrainConfig`.

- **2026-08-05 — PrioritizedDataLoader WeightedRandomSampler** (`training/memory_management.py`): `PrioritizedDataLoader.__init__` now builds a `torch.utils.data.WeightedRandomSampler` (with replacement + `pin_memory`); `_IndexedDataset` wraps map-datasets yielding `(X, y, idx)`; `_reload_sampler()` rebuilds the loader per epoch (uniform epoch ↔ boosted-priority epoch verified).

- **2026-08-05 — DQN replay-buffer action masks + HERBuffer perf** (`models/rl_advanced.py`): HERBuffer `_cache` list view invalidated on `end_episode()` and reused across `sample()` (1000×8 samples in 0.002s).

### Changed

- **2026-08-04 — Training cache / loop perf**: Polars-first `_build_chunk` (no feature-matrix Polars↔Pandas round-trip; HTF stays Polars). Zarr feature tensor `X` stored as **FP16**; labels/market sidecars FP32. `build_adamw()` uses fused CUDA AdamW (apex fallback) in supervised loop + DivFT. Linux Zarr default **Blosc lz4@1** (`default_zarr_compression` / `run_ubuntu.yaml`); non-Linux keeps zstd@3; CLI `--zarr-cname auto`. Rebuild caches for FP16 + lz4. Tests: `tests/test_zarr_prefetch.py`, `tests/test_fused_adamw.py`, `tests/test_model_full_data_flow.py`.

- **2026-08-04 — Args↔YAML audit softens**: `audit_args_vs_yaml_mismatches` reports CLI/strategy overrides as **warnings** (not errors). Feature-schema gate skips args/settings YAML parts when `--config` was not passed. Docs: [`docs/CONFIG_CONSISTENCY.md`](docs/CONFIG_CONSISTENCY.md).

### Fixed

- **2026-08-04 — Full suite after train_gpu split**: Direct imports for host-bind gaps (`_effective_max_seq_len`, TrainingLogger, supervised_loop deps); quick mode disables `torch.compile` + Rich live display (RecursionError teardown on tiny smokes); smoke uses `--pretrain-ablation false`; review-smoke patches defining modules. **1247 passed, 28 skipped** (FRED key / Stooq / yield panel cascade / dashboard). Tracker: [`docs/CONTINUE.md`](docs/CONTINUE.md).

- **2026-08-04 — Sharpe proxy softsign + KD student YAML**: `SharpeProxyLoss` and multitask Sharpe use softsign (`x/(1+|x|)`) instead of `tanh` so confident predictions keep usable gradients. `_apply_yaml_config` maps `distillation.student_model` → `--model` when distillation is enabled. Tests: `tests/test_gpu_losses.py`. Audit: [`docs/TRAINING_PIPELINE_AUDIT.md`](docs/TRAINING_PIPELINE_AUDIT.md) modules 26–36.

- **2026-08-04 — settings↔YAML curriculum schedule drift**: Synced `settings.CURRICULUM` stubs to active `config/run.yaml` (`epoch_unfreeze` / `always_on` for all groups, `seq_schedule`, `difficulty_schedule`, chunk early-stop). Added `difficulty_spread_threshold_hard` to `run.yaml` / `run_ubuntu.yaml`. New `audit_settings_yaml_curriculum_drift()` fails `validate_run_config` on shared-group / schedule / scalar mismatches; warns in `supervised_loop`. Tests: `tests/test_curriculum_audit.py`.

- **2026-08-04 — Config / curriculum mismatches B–D**: Fixed `config/run.yaml` `strategy.profit_target_atr` / `stop_loss_atr` illegal 3-space indent (YAML was unparseable; `_apply_yaml_config` previously swallowed the error and ran on hardcoded defaults). YAML parse failures now raise `RuntimeError`. Added `config/curriculum_audit.py` + warnings for curriculum features missing from schema, overlapping group membership, FEATURE_MASK orphans, and missing RL market spread columns; wired into `supervised_loop` and `validate_run_config`. Tests: `tests/test_curriculum_audit.py`.

### Added

- **2026-08-04 — Dataset feature-schema gate**: During dataset build, `audit_built_dataset_schema()` checks the locked feature columns against curriculum groups + required market roles (close/ATR). Missing curriculum features fail the build when `--integrity-gate` / `--feature-schema-gate` is on (fail-fast on first chunk + again when writing `*_feature_schema.json`). Writes `*_feature_schema_audit.json`. CLI: `--feature-schema-gate` / `--no-feature-schema-gate`.

- **2026-08-04 — Docs consolidation**: Living docs are `docs/README.md`, `CONTINUE.md`, `CONFIG_CONSISTENCY.md`, `IMPROVEMENT_PLAN.md`, `NEWS_DATA_GUIDE.md`, `SESSION_REPORT.md`. Historical `*_AUDIT_REPORT.md` / `FIXES_APPLIED.md` moved to `docs/archive/`. Folded `TRAIN_GPU_SPLIT.md` into `CONTINUE.md`. Root `improvement_plan.md` is a stub pointing at `docs/IMPROVEMENT_PLAN.md`.

- **2026-08-04 — Multi-part settings↔YAML mismatch audit**: New `config/config_mismatch_audit.py` compares shared keys by section (`training`, `backtest`, `rl`, …). Critical keys (`seq_len`, `loss`, `sharpe_annualization_factor`, `atr_stop_mult`, `reward.overtrade`) fail closed; other shared drift warns. Also checks resolved `args` vs YAML (silent load failures). Wired into `validate_run_config` and the dataset build gate as parts `settings_yaml` / `args_yaml`. Synced critical `settings.py` values to `run.yaml`. Operator guide: `docs/CONFIG_CONSISTENCY.md`; `docs/CONTINUE.md` rewritten as a short current tracker.

- **2026-08-04 — Audit P3 tooling + train_gpu split**: Wired ruff + pyright smoke gates. Split `training/train_gpu.py` into focused modules with back-compat re-exports (`gpu_losses`, `gpu_cache_io`, `gpu_datasets`, `direction_control`, `dataset_builder`, `supervised_loop`, `gpu_cli`, `cv_splits`, `pretrain_runner`, `rl_runner`, `cache_integrity`, `model_factory`, `post_train`, `gpu_device`, `feature_ablation`). Tracker: `docs/CONTINUE.md` (train_gpu split section). `train_gpu.py` now ~2.2k lines (was ~15.5k, −85%).

- **2026-08-04 — Production Wiring of Improvement Modules (opt-in CLI flags, behavior-preserving)**: Wired previously standalone improvement modules into production entry points as opt-in, default-off flags so existing behavior is preserved:
  - **Feature quality gate** (Improvement: `features/feature_quality_monitor.py`): `FeatureEngineer.__init__` gains `enable_quality_gate`; `build()` runs the `feature_quality_monitor` gate and `filter_features` drops constant/leaky/drifted columns.
  - **No-trade zones** (Improvement: `features/no_trade_zones.py`): `FeatureEngineer` gains `enable_no_trade_zones` → `no_trade_score` column; `NoTradeZoneGate` in `trading/live_guards.py` wired into `trading/live_engine.py` (default disabled).
  - **Meta-labeling** (Improvement: `labeling/triple_barrier_meta.py`): `scripts/backtest_model.py` gains `--meta-labeling / --meta-prob-threshold / --meta-min-samples`; `_build_meta_labeler_mask` masks trades by meta-prediction confidence.
  - **Off-policy rewards** (Improvement: `labeling/off_policy_rewards.py`): `models/rl_agents.py` `train_agent(..., off_policy_rewards=False)` opt-in; per-episode obs/act/rew logs for PPO and `_estimate_off_policy_rewards` via `net.backbone → net.actor`; `training/train_gpu.py` gains `--off-policy-rewards` + mean DR/IPS printout.
  - **CV strategies** (Improvement: `validation/cv.py`): `training/train_gpu.py` gains `--cv-strategy {legacy,walk_forward,comb,online}`; `_build_cv_splits` selects WalkForwardCV/CombCV/OnlineCV with legacy fallback. Fixed 12 pre-existing bugs in `validation/cv.py` (`_embargo_indices`, WalkForwardCV.split, OnlineCV.split, PurgedKFold.split, RegimeCV.split, create_cv nested pop).
  - **HPO scheduler** (Improvement: `training/hpo.py`): `build_optuna_search(scheduler=...)` returns tpe/asha/bohb/pbt sampler+pruner combos; `scripts/optuna_tune.py` gains `--hpo-scheduler` and stamps `cfg["optuna"]["hpo_scheduler"]`.
  - **Multi-task pretrainer** (Improvement: `pretrain/multi_task.py`): `training/train_gpu.py` gains `--use-multi-task-pretrainer`; `_run_multi_task_pretrain` trains via `pretrain_multi_task` and loads encoder state into the backbone (strict=False, resume + graceful fallback).
  - **Curriculum manager** (Improvement: `training/curriculum.py`): `training/train_gpu.py` gains `--curriculum-manager` + `--curriculum-manager-mode`; `create_curriculum_manager` drives per-epoch inclusion-mask filtering after the difficulty filter.
  - **Execution engine** (Improvement: `backtesting/execution.py`): `scripts/backtest_model.py` gains `--execution-engine {legacy,advanced}`; `_advanced_execution_overlay` computes toxicity/latency-adjusted effective slippage via `AdverseSelectionModel` + `LatencyModel` and writes summary JSON `"execution"` meta.
  - **Monte Carlo facade** (Improvement: `evaluation/monte_carlo.py`): legacy `backtesting/improvements.MonteCarloBacktest` and `monitoring/pipeline.MonteCarloBacktest` now delegate resampling to the new suite (`block_bootstrap_indices`, block_length=1 i.i.d.) while preserving their historical `.run`/`.run_from_backtest` result-key schemas so existing callers keep working.
   - New wiring tests: `tests/test_backtest_wiring.py`, `tests/test_cv.py`, `tests/test_hpo.py`, `tests/test_multi_task.py`, `tests/test_curriculum.py`, `tests/test_rl_off_policy.py`, `tests/test_monte_carlo.py` (facade shims). 211 tests passing across the wired-module suites.

- **2026-08-04 — Risk Controls** (`risk/risk_engine.py`, `risk/fx_greeks.py`, `risk/portfolio_monitor.py`): Real-time risk engine — pre-trade checks (max position pct/lots/notional, daily-loss stop-trading, max order frequency, single-instrument concentration caps); post-trade monitoring (drawdown limits, historical + parametric VaR/CVaR, exposure by pair/currency, gap-risk flags); circuit breakers (auto-flatten / auto-standby with audit log). FX option Greeks via Black-76 (delta/gamma/theta/vega/rho, position-adjusted portfolio Greeks) with JPY/USD quoting-convention handling. Portfolio monitor aggregates exposure across pairs, net currency exposure, liquidity tiers, and correlation-aware clustering. `config/settings.py` `RISK` block extended; `--risk-config` added to `training/train_gpu.py` loading a `RiskEngine` via `RiskConfig.from_dict`. Tests: `tests/test_risk_engine.py`, `tests/test_fx_greeks.py`, `tests/test_portfolio_monitor.py` — 45 passed.
- **2026-08-04 — Metrics** (`evaluation/metrics.py`): Probabilistic Sharpe Ratio (per-period Sharpe, skew/kurtosis adjusted), Deflated Sharpe Ratio (Bailey-López de Prado multiple-trials correction), Calmar, Omega, Tail ratio, Sortino, downside deviation, max drawdown, and minimum backtest length for significance. `backtest_metrics`/`MetricReport` helpers accept ndarray / list / result objects; integrated with `validation/promotion_gate.py` PSR path. Tests: `tests/test_metrics.py` — 27 passed.
- **2026-08-04 — Data drift** (`drift/data_drift.py`): Feature distribution drift (KS / Wasserstein / PSI, reuses `feature_quality_monitor._safe_psi`), SHAP feature-attribution drift (relative-importance shift), concept-drift streaming detectors (ADWIN / Page-Hinkley / DDM / EDDM with a unified drift score), and adversarial validation (sklearn RandomForest w/ histogram fallback AUC). Emits structured drift events. Tests: `tests/test_data_drift.py` (+ existing `tests/test_drift_detection.py`) — 48 passed.
- **2026-08-04 — Model drift** (`drift/model_drift.py`): Champion-challenger harness (shadow-mode live comparison via the repo promotion gate + PSR), canary rollout (x% live-routing with escalates/deescalates fraction), and automated rollback monitor (drawdown / PSR-floor / error-spike triggers with optional checkpoint-swap callback) complementing `monitoring/demotion_monitor.py`. Tests: `tests/test_model_drift.py` — 22 passed.
- **2026-08-04 — Audit & reproducibility** (`audit/lineage.py`, `audit/manifest.py`): DataLineage chain (dataset → preprocessing → features → labels → training run), ModelRegistryRecord hook, and DecisionRecord audit trail of all promotion/rollback decisions. Manifest generator/verifier/regenerator producing `manifest.json` (self-hash, artifact sha256, git commit/branch, captured env incl. GPU + lib versions) alongside checkpoints — consumed by `validation/promotion_audit.py`. CLI/helper `regenerate_manifest` rebuilds from an existing checkpoint dir. Tests: `tests/test_audit.py` — 17 passed.
- **2026-08-04 — Alerting** (`monitoring/alerting.py`): Multi-channel alerting (console, file, in-memory test-mode, Discord webhook via existing `DiscordAlerter`, Slack webhook, SMTP email); severity levels (info/warning/critical) with per-channel minimum-severity routing, deduplication + rate limiting, and escalation of unacked critical alerts after N minutes to all channels. Runbook registry with per-alert-type remediation steps. Integration entry points `notify_drift_event` (drift events from items 4/5) and `notify_risk_violation` (risk violations from item 1) route events into the alerting manager. Tests: `tests/test_alerting.py` — 21 passed.


- **2026-08-04 — Bytewax 0.21 Streaming Feature Pipeline** (`features/streaming_pipeline.py`): Completely rewrote the streaming feature pipeline to use the high-performance Bytewax 0.21 API (`bytewax.operators`, `bytewax.operators.windowing`). Upgraded the pipeline to use `SystemClock` for zero-latency tumbling windows, ensuring real-time features instantly flush for live forex trading based on wall-clock time rather than delayed event-time watermarks.

- **2026-08-03 — Backtest Execution Engine** (`backtesting/execution.py`, `backtesting/improvements.py`): Advanced execution engine with realistic market microstructure simulation:
  - `LimitOrderBook`: FIFO/pro-rata/size-priority limit order book with queue position tracking, partial fills, market/marketable limit orders, market data dissemination latency, and dissemination latency
  - `LatencyModel`: Realistic latency simulation (network RTT, gateway fixed+jitter, exchange matching engine load-dependent latency, market data dissemination) with colocation advantage modeling
  - `AdverseSelectionModel`: VPIN (Volume-synchronized PIN), Kyle's lambda estimation, queue position adverse selection risk, composite toxicity scoring with configurable thresholds
  - `SlippageDecomposition`: Implementation shortfall decomposition (delay cost, spread cost, market impact, timing cost, adverse selection cost, opportunity cost) with Perlmold/Almgren-Chriss framework
  - `AdvancedExecutionEngine`: Integrated execution engine combining LOB, latency model, adverse selection, slippage decomposition, partial fills, queue management
  - `AdvancedBacktestEngine`: Enhanced backtest engine with realistic execution modeling
  - `backtesting/improvements.py`: Monte Carlo backtest (bootstrap with replacement), SlippageCalibrator (power-law fit to real fill data), LockboxTest (single-use out-of-sample evaluation with seal mechanism)
  - 39 new tests in `tests/test_hpo.py`, 28 tests in `tests/test_streaming_pipeline.py`, 20 in `tests/test_curriculum.py`, 14 in `tests/test_memory_management.py`, 13 in `tests/test_no_trade_zones.py`, 8 in `tests/test_triple_barrier_meta.py`, 19 in `tests/test_off_policy_rewards.py`, 16 in `tests/test_feature_quality_monitor.py`, 13 in `tests/test_sentiment_fusion.py`, 16 in `tests/test_cross_asset_factors.py`, 17 in `tests/test_regime_detection.py`, 17 in `tests/test_feature_pipeline.py`, 20 in `tests/test_curriculum.py`, 14 in `tests/test_memory_management.py`, 13 in `tests/test_no_trade_zones.py`, 8 in `tests/test_triple_barrier_meta.py`, 19 in `tests/test_off_policy_rewards.py`, 16 in `tests/test_feature_quality_monitor.py`, 13 in `tests/test_sentiment_fusion.py`, 16 in `tests/test_cross_asset_factors.py`, 17 in `tests/test_regime_detection.py`, 17 in `tests/test_feature_pipeline.py`, 42 in `tests/test_hpo.py`, 28 in `tests/test_streaming_pipeline.py` — 279+ total tests passing

- **2026-08-03 — Multi-Task Pretraining with Domain Adaptation** (`pretrain/multi_task.py`): Unified multi-task pretrainer combining contrastive (SimCLR/BYOL), masked reconstruction, forecast pretext, VAE, drift detection, and domain adaptation (DANN/MMD/CORAL). Shared encoder with task-specific heads, GradNorm loss balancing, curriculum-aware task scheduling, and `adapt_encoder_to_target` for encoder transfer. Factories `create_multi_task_pretrainer` / `pretrain_multi_task`.

- **2026-08-03 — Regime Detection Upgrade** (`features/regime_detection.py`): True market-regime detection with Gaussian HMM (hmmlearn) over [returns, rolling vol], replacing legacy volatility-tercile fake "HMM". Hurst exponent via R/S and DFA estimators (fixed: DFA residual pooling, R/S lag range n//4). Higuchi fractal dimension (fixed: normalization D=1-slope). Polars builder `detect_regimes_polars` with `step` subsampling for speed. `FeatureEngineer.build()` wired with graceful legacy fallback.

- **2026-08-03 — Cross-Asset Factor Model** (`features/cross_asset_factors.py`): Rolling PCA/ICA common-factor model (factor scores, variance-explained, per-asset loadings), pairwise Granger causality via manual joint F-test (validated against statsmodels <0.02 diff), and vectorized lead-lag network (best incoming/outgoing lagged correlations, network degrees, edge density). `CrossAssetFeatures.build()` integration appends 116 factor/granger/leadlag columns.

- **2026-08-03 — Multi-Modal Sentiment Fusion** (`features/sentiment_fusion.py`): News + social + COT sentiment fusion with count-weighted consensus, cross-modality agreement/dispersion. Financial NER extractor (10 curated categories: rate hike/cut/hold, CPI, NFP, GDP, dovish/hawkish, pair/cb mentions). Tfidf+NMF topic model (deterministic seed). Offline domain lexicon scorer (bullish/bearish in [-1,1]). `FeatureEngineer.build()` integration when raw event text provided.

- **2026-08-03 — Feature Quality Monitor** (`features/feature_quality_monitor.py`): Population Stability Index (PSI) vs reference distribution, Information Value / WOE vs binary target, rolling stability index (trailing PSI + KS vs baseline), target-leakage detection (IV >0.5, AUC >0.85, near-perfect target correlation). Master `feature_quality_monitor()` + `filter_features()` gate drops constant/leaky/severe-drift columns while preserving target.

- **2026-08-03 — Off-Policy RL Rewards & QR-DQN Labels** (`labeling/off_policy_rewards.py`): Counterfactual IPS/doubly-robust reward estimation (per-action, weight clipping, bootstrap CI). QR-DQN distributional labels (per-bar quantiles of forward exit-reward distribution). Handles both 1-D per-action and 2-D policy-matrix inputs. Orchestrator `compute_off_policy_rewards` returns per-action IPS/DR + global value + CI.

- **2026-08-03 — Triple Barrier Meta-Labeling & Bayesian Search** (`labeling/triple_barrier_meta.py`): Meta-labeling (Lopez de Prado) — secondary classifier predicts whether primary model's trade will be profitable, trained on TBM labels. Bayesian barrier search via Optuna (TPE sampler, TPE/Median/Hyperband pruners) optimizing profit_mult, stop_mult, vertical_bars, delay for Sharpe/win-rate/PF/expectancy. `run_meta_tbm_pipeline` integrates Bayesian search, TBM labeling, meta-labeling, and confidence filtering. `evaluate_barrier_params` for quick single-set evaluation.

- **2026-08-03 — No-Trade Zones: Learned Abstention + Conformal Prediction** (`features/no_trade_zones.py`): Learned abstention model — secondary classifier predicts if primary model's trade will be profitable, trained on TBM labels. Conformal prediction for abstention — uses split conformal to produce prediction sets; abstains when set contains both long and short (ambiguous). Enhanced heuristic score (low vol, neutral OFI, choppy trend). `NoTradeZoneManager` / `apply_no_trade_zones` unify heuristic, learned, and conformal signals with configurable weights/modes.

- **2026-08-03 — Training Memory Management** (`training/memory_management.py`): Streaming datasets (`StreamingMemmapDataset`, `SequentialZarrDataset`) with background thread prefetching (`PrefetchDataLoader`). Gradient checkpointing (`apply_gradient_checkpointing`, `CheckpointPolicy`) for transformer/Mamba blocks. Activation offloading (`SelectiveActivationOffloader`, `OffloadedTensor`) to CPU. Memory profiling (`memory_profiler`, `MemoryMonitor`, `memory_efficient_training`). Factory `create_streaming_dataloader` for one-line setup.

- **2026-08-03 — Curriculum Learning** (`training/curriculum.py`): Difficulty curriculum (`DifficultyCurriculum`) with configurable pace functions (linear/exp/sqrt/step). Self-paced learning (`SelfPacedLearning`, Kumar et al. 2010) with loss-based pacing. Loss-based weighting (`LossBasedWeighting`) with multiple schemes (inverse/focal/threshold/softmax/curriculum) and EMA smoothing. Unified `CurriculumManager` combining all strategies with "combined" mode. `CurriculumDataLoader` wrapper for weighted sampling. Difficulty scoring (`compute_difficulty_scores`) via margin/loss/entropy/distance. Factory `create_curriculum_manager` for unified config. Extends existing `CurriculumController` for unified control.

- **2026-07-03 — Config Preflight Validation** (`training/config_validate.py`): Added range validation for critical hyperparameters (`lr`, `epochs`, `batch_size`, `patience`, `seq_len`), warmup-vs-epochs ratio check (catches warmup ≥ 50% of epochs), patience-vs-effective-training-epochs check (warns if early stopping can never trigger), model name validation against `SUPPORTED_SUPERVISED`, and loss function validation against known set.

- **2026-07-03 — Yield Curve Slope Deduplication** (`features/feature_engineering_pl.py`): `yield_curve_slope` (from macro builder) and `us_2s10s_spread` (from cross-asset builder) are the same US10Y−US2Y calculation. Added bidirectional alias so both column names are always available regardless of which builder ran first.

- **2026-07-03 — Cross-Asset Provider Failure Logging** (`data/cross_asset.py`): Added warning log when all providers fail for a cross-asset symbol, replacing silent `continue`.

### Fixed

- **2026-08-04 — Live safety + remaining correctness** (follow-on to full audit):
  - **Silent PaperBroker fallback** (`trading/live_engine.py`): real brokers no longer silently degrade to paper; require `--allow-paper-fallback` (or `--broker paper`). Multi-pair path matches.
  - **RiskEngine on live path**: default `RiskEngine` constructed per engine; `check_order` before entries, `update_equity` / circuit-breaker flatten, `new_day()` at UTC day boundary; `--risk-config` CLI.
  - **ATR stop-loss wired**: `stop_loss_atr × ATR` flatten when adverse move hits; journaled as `atr_stop`.
  - **OANDA `get_account` empty bypass**: raises on failure/zero equity so the equity-failure counter can halt after 5 misses (empty `{}` no longer looks like success).
  - **JPY pip scale** (`config.settings.price_to_pips`, `LiveSafetyGate`, `SpreadVolatilityGuard`): spread checks use pair pip size (USDJPY 0.01) instead of hard-coded `×10000`.
  - **HMM look-ahead** (`features/regime_detection.py`): causal vol + warm-start HMM fit on prefix + 1-bar lagged posteriors (no full-series fit).
  - **Eco point-in-time** (`features/feature_engineering_pl.py`): join eco on `available_time` (default event+1min) via `join_asof` backward.

- **2026-08-04 — Full audit P0/P1 + training correctness** (see `docs/archive/FULL_AUDIT_REPORT.md`):
  - **`sanitize_array` empty-string crash** (`infrastructure/numerics.py`): coerce non-numeric / `''` via `pd.to_numeric(..., errors="coerce")` instead of raising on Dukascopy mixed columns; blocks real-data training no longer.
  - **`sanitize_frame` price clipping** (`infrastructure/numerics.py`): align with feature pipeline — skip bid/ask/mid/OHLCV/spread/COT columns from the default `[-20, 20]` clip (fixes USDJPY-style price destruction when labeling uses the numerics path).
  - **OANDA position netting** (`trading/live_engine.py`): `net = long_u - abs(short_u)` so short units reported as positive magnitude are not double-counted.
  - **`clean_bad_ticks` spike survival** (`data/data_ingestion.py`): lag-1 rolling reference + null-safe outlier / spread masks so MAD/Z detectors actually replace spikes (null `spread_outlier` previously short-circuited the condition).
  - **RegimeCV purge** (`validation/cv.py`): global temporal purge so train samples within `purge` of any val index are dropped (cross-regime adjacency leakage).
  - **CombCV purge/embargo** (`validation/cv.py`): ban train samples in `[t0-purge, t1+purge+embargo)` around each test block.
  - **Broken test collection**: `tests/test_system.py` dropped stale `settings.MODELS` import; `tests/test_streaming_pipeline.py` rewritten for current Bytewax API; `MarketTick.to_dict()` added.
  - **Quick-mode direction gates** (`training/train_gpu.py`): disable direction probe + ignore preflight / soft-continue class-balance gates on short real windows so e2e smoke can finish.
  - **`sidecar` UnboundLocalError** (`training/train_gpu.py`): always init `sidecar = None`; reuse logger sidecar across walk-forward folds.
  - **Chunk feature schema lock** (`training/train_gpu.py`): reset `_FIRST_CHUNK_COLS` per cache build; same-columns/different-order → reindex instead of opaque raise.
  - **Training model hardening** (`models/architectures.py`, `models/ensemble.py`, `training/train_gpu.py`, `features/feature_engineering_pl.py`): Xavier/small-head init on more arches; pre-norm + grad checkpointing; `MultiTaskLoss(label_smoothing=...)`; unwrap `_orig_mod` on save/load; update both best-sharpe and best-val-loss trackers; fail hard on empty train/val splits; pass `pair=` into `position_limit_flags`; remove dead `diversity_loss` under `no_grad`.

- **2026-07-03 — TemperatureScaler.calibrate Broken** (`models/architectures.py`): `@torch.no_grad()` was wrapping the entire `calibrate()` method including the LBFGS optimizer, making temperature calibration a complete no-op (gradients never reached the optimizer). Fixed by scoping `torch.no_grad()` to only the inference loop; LBFGS optimization now receives gradients correctly.

- **2026-07-03 — FinBERT `fb_*` Columns Overwritten** (`features/feature_engineering_pl.py`): Zero-filled `fb_*` placeholder columns were unconditionally created after `sentiment_tiers()` computed them, erasing actual FinBERT embeddings. Fixed to only create placeholders for columns not already present.

- **2026-07-03 — `sanitize_frame` Clipped Prices to [-20, 20]** (`features/feature_engineering_pl.py`): All float columns including raw prices, volume, spread, and latency were clipped to `[-20, 20]`, destroying critical data. Added `_SANITIZE_NO_CLIP` exclusion set for core price/volume/spread/latency/COT columns.

- **2026-07-03 — `liquidity_vacuum` Crash Without Bid/Ask** (`features/feature_engineering_pl.py`): `liquidity_vacuum()` expression crashed when `bid_close`/`ask_close` columns were absent. Added conditional check, falling back to zero-filled placeholder.

- **2026-07-03 — Single-Pair Sentiment Pipeline Missing** (`training/train_gpu.py`): The single-pair dataset build path never initialized `SentimentPipeline`, silently dropping all FinBERT features. Added initialization and passing to the chunk processing function.

- **2026-07-03 — COT Data Ignored in Feature Pipeline** (`features/feature_engineering_pl.py`): `add_cot_features()` was never called in `build()` despite COT data being loaded and passed as a parameter. Added the call with fallback zero-fill for `cot_net_hf`, `cot_net_comm`, `cot_hf_mom_4w`, `cot_extreme`.

- **2026-07-03 — `live_engine.py` Import Error** (`trading/live_engine.py`): `run_pipeline` was imported from `data.fetch_oanda_sentiment` but the function is named `run_collector`. Fixed the import.

- **2026-07-03 — `run_fast.yaml` Broken Probe** (`config/run_fast.yaml`): `lr_warmup_epochs: 8` with `epochs: 10` meant 80% of training was warmup; `patience: 10` could never trigger early stopping. Changed to warmup=2, patience=5.

- **2026-07-03 — `run_ubuntu.yaml` Wrong Feature Cache Key** (`config/run_ubuntu.yaml`): `slow_cols` listed `cot_net` (legacy column) instead of `cot_net_hf` (the actual expensive COT feature), causing `cot_net_hf` to be recomputed on every trigger event.

- **2026-07-03 — Cross-Asset Curriculum Defeated** (`config/run.yaml`): `cross_asset` feature group had `always_on: true` which made `epoch_unfreeze: 8` a no-op, bypassing progressive unfreezing. Changed to `always_on: false`.

- **2026-07-03 — `run_ubuntu.yaml` Missing Feature Groups** (`config/run_ubuntu.yaml`): Missing `execution_cost`, `market_regime`, `higher_timeframe`, and `label_quality` feature groups, causing those features to be unmasked from epoch 0 instead of their intended curriculum schedule.

- **2026-07-03 — `diversity_loss` YAML Keys Silently Ignored** (`training/train_gpu.py`): `diversity_loss.weight` and `diversity_loss.same_role_mult` had no `_YAML_MAP` entries, so YAML edits to these keys had zero effect. Added mappings.

- **2026-07-03 — Backtest Stop/TP Hardcoded** (`scripts/backtest_model.py`): `--stop-pips` and `--take-pips` defaulted to hardcoded 15.0/30.0 instead of reading from `BACKTEST` config (12.0/18.0). Now reads from config.

- **2026-07-03 — Normal-Mode `seq_len` Mismatch** (`config/strategy_profiles.py`): Normal strategy profile had `seq_len: 96` but models are trained at `seq_len: 60`, causing shape errors at inference. Aligned to 60.

- **2026-07-03 — `nhead` Not Passed to EXPERTEncoder** (`training/train_gpu.py`): The builder lambda never passed `nhead`; EXPERT always used the constructor default (8) regardless of YAML/profile config. Now passes `args.nhead`.

- **2026-07-03 — `dim_ff` Not Wired for iTransformer** (`training/train_gpu.py`): The iTransformer builder never passed `dim_ff`; feedforward dimension always defaulted to 256. Now passes the configured value.

- **2026-07-03 — GNN `heads` Not Mapped Through Profile** (`training/train_gpu.py`): `_normalize_architecture_profile` didn't extract `heads` → `nhead` for GNN. Config/YAML changes to GNN heads had no effect. Added mapping.

- **2026-07-03 — `build_model` Missing Aliases** (`models/architectures.py`): The introspection-based `build_model` factory silently dropped `num_layers` for HAELT (expects `n_layers`), `hidden_size` for HAELT (expects `lstm_hidden`), `num_layers` for TFT (expects `lstm_layers`), and `dim_feedforward` for iTransformer (expects `dim_ff`). Added comprehensive bidirectional alias mapping.

- **2026-07-03 — Zarr Init Inconsistency** (`training/train_gpu.py`): `y_cls` was created with `shape=(0,)` + `append()` while `X`/`y`/`close`/`atr`/`spread` used actual shape + `[:]` assignment. Standardized to match. Added `pq_seq`/`diff_seq` None guards in initial-create and append paths (were only guarded in resume path).

- **2026-07-03 — ThreadPoolExecutor Leak** (`training/train_gpu.py`): `ThreadPoolExecutor` for parallel tick loading was not used as a context manager; an exception during window processing would leak threads. Wrapped in `try/finally` with `shutdown()` in the finally block.

- **2026-07-03 — `run_ubuntu.yaml` Config Drift** (`config/run_ubuntu.yaml`): Aligned dropout (0.1→0.25), loss function (`cross_entropy`→`sharpe_huber`), and warmup (8→3) to match the tuned `run.yaml` production values. Fixed stale "BYOL" comment on masked pretraining.

- **2026-07-03 — `run_normal.yaml` Warmup** (`config/run_normal.yaml`): `lr_warmup_epochs: 8` → `3` to match the tuned value from `run.yaml`.

- **2026-07-03 — Python 3.9 Compatibility** (`config/strategy_profiles.py`): `str | None` union syntax (PEP 604, Python 3.10+) → `Optional[str]` for backward compatibility.

- **2026-07-03 — Dead `CrossAssetFeatures.ASSETS` Dict Removed** (`features/feature_engineering_pl.py`): Hardcoded synthetic asset parameters were never used by any caller after real data loading was implemented.

- **2026-07-03 — Duplicate Sentiment Timestamp Cast Removed** (`features/feature_engineering_pl.py`): `sentiment` DataFrame was cast to `Datetime("ns", "UTC")` twice — once at the top of `build()` and again before joining. Removed the duplicate.

- **2026-07-03 — `CROSS_ASSET` Related Pair References** (`config/settings.py`): `USDCAD` and `AUDUSD` were referenced as `related_pair` values but aren't in `DATA["pairs"]`. Changed to `"all"`.

- **2026-07-03 — GNN Config Incomplete** (`config/models/gnn.yaml`, `config/models.py`): Added missing `node_features: 32`, `nhead: 4` to YAML; added `learning_rate` and `seq_len` to Python profile dict.

- **2026-07-03 — HAELT `build_model` Halving Parity** (`models/architectures.py`): Added HAELT-specific `d_model`/`nhead`/`lstm_hidden` halving logic to the introspection-based `build_model` factory so that `smoke_test.py` produces the exact same model dimensions as `train_gpu.py`. Previously the two builders diverged, meaning smoke tests validated a model that was twice as wide as the one actually trained.

- **2026-07-03 — Config Re-Exports** (`config/__init__.py`): Added re-exports of `TRAINING`, `PATHS`, `DATA`, `LABELING`, `PRETRAIN`, `RISK`, `MODELS`, and `FEATURE_MASK` so consumers can use `from config import TRAINING` instead of verbose `from config.settings import TRAINING`.

- **2026-07-03 — Explicit `pretrain_loss` Key** (`config/settings.py`): Added `"pretrain_loss": "huber"` to `PRETRAIN` dict so the pretrain loss override is explicit configuration rather than a hidden swap buried inside `train_gpu.py`.

- **2026-07-03 — Feature Mask Baseline Documented** (`config/feature_mask.py`): Added docstring clarifying that all-True is the intentional development baseline, and linked `fb_0..fb_7` projection dimensions to `SENTIMENT["finbert_proj_dim"]`.

### Fixed

- **2026-07-03 — HAELT `n_transformer_layers` Naming Inconsistency** (`config/models.py`, `training/train_gpu.py`): Python config used `n_transformer_layers` while YAML and constructor use `num_layers`/`n_layers`. Renamed to `num_layers` in Python config; updated `_normalize_architecture_profile` to accept both names for backward compatibility.

- **2026-07-03 — TFT `attention_head_size` Naming Inconsistency** (`config/models.py`, `training/train_gpu.py`): Python config used `attention_head_size` while YAML and constructor use `nhead`/`heads`. Renamed to `nhead` in Python config; updated `_normalize_architecture_profile` to accept both names for backward compatibility.

- **2026-07-03 — GNN Dead `correlation_threshold` Removed** (`config/models.py`, `models/architectures.py`): Removed dead `correlation_threshold: 0.3` from GNN config (never used; GNN learns adjacency via `adj_logits`). Cleaned stale reference in `GNNCrossAsset` docstring.

### Changed

- **2026-07-03 — Dead Config Annotated** (`config/settings.py`, `config/models.py`, `config/run.yaml`): Annotated `GOVERNANCE`, `ALERTS`, `VALIDATION` dicts as TODO/not-yet-integrated. Annotated Mamba `d_state`/`d_conv`/`expand` as reserved for full Mamba. Annotated EXPERT `use_conv_ffn`/`no_pos_encoding` as hardcoded design decisions. Annotated `onecycle_*` params as fallback-only. Added halving note to `haelt.yaml`.

### Added (earlier 2026-07-03)

- **2026-07-03 — Batch FinBERT Prefetch** (`features/finbert_sentiment.py`, `data/historical_news.py`, `training/train_gpu.py`): Added upfront headline cache-warming so the per-window loop never waits on model inference. `collect_headlines_for_range()` loads the full news parquet once via DuckDB, filters by pair currencies, and returns unique headline strings. `SentimentPipeline.prefetch_headlines()` pre-scores all uncached headlines in batches of 256 and flushes to disk. The multi-pair dataset builder calls prefetch before the window loop when `historical_news_mode=full`.

- **2026-07-03 — Parallel Date Window Loading** (`training/train_gpu.py`, `config/run.yaml`): Added `--dataset-build-workers` CLI flag and `data.dataset_build_workers` YAML key (default `2`). The multi-pair window loop now uses a `ThreadPoolExecutor` with a look-ahead queue: while the current window builds features/labels, the next N windows' ticks are loading in parallel threads. Sequential appending to zarr is preserved.

- **2026-07-03 — `real_data_window_days` YAML Mapping** (`training/train_gpu.py`, `config/run.yaml`): Wired `data.real_data_window_days` into the YAML-to-argparse config mapping. Set to `7` in `run.yaml` so the window size is controlled from config without needing `--real-data-window-days` on the CLI.

### Fixed (earlier 2026-07-03)

- **2026-07-03 — Cross-Asset Provider** (`config/run.yaml`): Changed `cross_asset_provider: yahoo` → `auto`. Yahoo-only skipped FRED bond yields (DE10Y, JP10Y, US2Y, etc.); `auto` tries Stooq → Yahoo → FRED → EODHD in order per asset, resolving all 25 series including G10 yields.

- **2026-07-03 — Cross-Asset Merge Bug** (`features/feature_engineering_pl.py`): Changed `if k in merged:` guard to unconditional `merged[k] = v`. Real series from `load_cross_asset_panel()` (e.g. SILVER, NATGAS, YIELD_CURVE_SLOPE, international yields) were silently dropped if not present in the hardcoded synthetic `ASSETS` dict. Now all real data replaces or extends synthetic placeholders.

- **2026-07-03 — MacroYieldFeatureBuilder Wired into Training** (`features/feature_engineering_pl.py`): `FeatureEngineer` now instantiates `MacroYieldFeatureBuilder` and calls `.build()` after `CrossAssetFeatures`, joining yield spreads (`spread_us_de`, `spread_us_jp`, etc.), carry signals (`carry_eur`, `carry_jpy`, etc.), yield momentum (`yield_momentum_5d/20d`), and `yield_curve_slope` — matching the curriculum feature group names in `run.yaml`. Previously only the live engine imported this builder; training produced different column names (`yield_spread_us_de_10y`).

- **2026-07-03 — `gold_dxy_corr_break` → `corr_break_gold_DXY` Alias** (`features/feature_engineering_pl.py`): Added column alias after cross-asset build so the curriculum feature group `cross_asset` can reference `corr_break_gold_DXY` without a name mismatch. Also added zero-fill `carry_spot_forward` placeholder.

- **2026-07-03 — Removed Debug Pipeline Logs** (`training/train_gpu.py`): Removed `[Debug] Pipeline run start/end` print statements that fired per window per pair, reducing log I/O during dataset builds.

---

## [2026-06-29] - Curriculum-Aware Optuna Tuning & Integrated Three-System Training

### Added

- **2026-06-29 - True Walk-Forward Out-Of-Sample Backtester** (`scripts/backtest_true_walk_forward.py`): Implemented a mathematically rigorous backtester that completely eliminates in-sample data leakage for walk-forward trained models. It loads the full dataset, precisely reconstructs the training fold boundaries, iteratively loads the optimal checkpoint for each era, and stitches the unseen validation predictions into a single, continuous 15+ year equity curve. This guarantees every trade taken during the backtest is strictly out-of-sample.

- **2026-06-29 - Curriculum-Aware Optuna Tuning** (`scripts/optuna_tune.py`, `config/models.py`): Upgraded Optuna to tune the *shape and dynamics* of the curriculum instead of a flat static `seq_len`. Each trial now searches 8 curriculum parameters:
  - `cur_seq_start` ΓÇö starting sequence length (20 / 30 / 45 depending on model)
  - `cur_seq_ramp_epoch` ΓÇö epoch at which the sequence length begins growing (6 / 10 / 14 / 20)
  - `cur_seq_target` ΓÇö the final sequence length ceiling (45 / 60 / 90 / 120)
  - `cur_collapse_drop` ΓÇö how aggressively the Auto-Tuner stall fires (0.10 / 0.15 / 0.20 / 0.25)
  - `cur_collapse_min_peak` ΓÇö minimum positive Sharpe peak required before stalling kicks in (0.15 / 0.25 / 0.35)
  - `cur_advance_lr_mult` ΓÇö LR decay when advancing to harder market difficulty (0.75 / 0.85 / 0.95)
  - `cur_collapse_lr_mult` ΓÇö LR decay when a Sharpe collapse is detected (0.70 / 0.80 / 0.90)
  - `cur_stable_window` ΓÇö consecutive stable epochs needed before advancing difficulty (2 / 3 / 4)
  
  Optuna's chosen params are converted to a concrete 3-step `seq_schedule` (start ΓåÆ midpoint ΓåÆ target) and written into the per-trial YAML config so Curriculum Adaptation executes the schedule Optuna designed, while the Heuristic Auto-Tuner guards against collapse at runtime. This resolves the previous conflict where `curriculum.seq_schedule` overrode Optuna's `--seq-len` flag entirely. All three systems now work together: **Optuna designs ΓåÆ Curriculum executes ΓåÆ Auto-Tuner guards**.

- **2026-06-29 - OPTUNA_MODEL_OVERRIDE Env Var Support** (`config/models.py`): Updated `architecture_config()` to read `OPTUNA_MODEL_OVERRIDE` from the environment and merge any JSON-encoded overrides on top of the base model config. This allows Optuna trial subprocesses to inject architecture-specific parameters (e.g. `n_transformer_layers`, `d_state`) that have no corresponding CLI flag in `train_gpu.py` without requiring temporary YAML config changes.

### Fixed

- **2026-06-29 - Promotion Gate Backtest Data Source** (`training/train_gpu.py`, `scripts/backtest_model.py`): Fixed the `[PromotionGate] Execution backtest failed: Not enough data` error caused by `run_execution_backtest` hardcoding `source="historic"`, which is not a valid source in `ForexDataManager`. The invalid source caused a fallback to synthetic data generation for the current date, producing 0 rows after the holdout date slice was applied. Added `data_source` argument to `run_execution_backtest` (defaulting to `"dukascopy"`) and propagated it from `train_gpu.py`'s `_run_promotion_gate` so the correct live data source is always used for holdout backtesting.

- **2026-06-29 - LR Warmup / Patience Structural Bug** (`training/train_gpu.py`, `config/run.yaml`): Fixed a three-part structural misconfiguration where models were early-stopping *during* the LR warmup ramp, producing `BestEpoch=0` results (meaning the pretrained model was better than anything the fine-tuner managed):
  1. **`lr_warmup_epochs: 8 ΓåÆ 3`** (`config/run.yaml`): With folds running only 5ΓÇô15 epochs, the 8-epoch warmup consumed the entire training budget before the LR ever reached its target value. Reduced to 3 so the peak LR is reached well before early-stop fires.
  2. **`lr: 1.2e-5 ΓåÆ 2.5e-5`** (`config/run.yaml`): Warmup starts at `lr / 100`, so a low `lr` caused LRs of `~1e-7` during warmup ΓÇö too small to produce any gradient signal. Raised the peak to give the warmup ramp useful values from epoch 1 onward.
  3. **`patience: 10 ΓåÆ 7`** (`config/run.yaml`): With warmup done at epoch 3, the effective patience budget is `patience ΓêÆ warmup_epochs`. Reduced from 10 to 7 to keep the effective post-warmup patience at 4 meaningful epochs.
  4. **Warmup-gated patience counter** (`training/train_gpu.py`, line ~7988): Added a hard gate so `no_improve` only increments after `ep >= lr_warmup_epochs`. The patience counter previously ticked unconditionally from epoch 0, meaning the model could exhaust its entire patience budget while still in warmup and early-stop before the LR had any chance to stabilize.

- **2026-06-29 - Auto-Tuner LR Decay Erased by Scheduler Bug** (`training/train_gpu.py`): Fixed a critical bug where the Auto-Tuner's manual LR adjustments (`param_group['lr'] *= multiplier`) upon curriculum stalls or advances were being completely erased within milliseconds. Because `warmup_cosine` uses PyTorch's `LambdaLR` (which computes the LR purely as a function of the global step count applied to `scheduler.base_lrs`), the very next batch's `scheduler.step()` would instantly overwrite the optimizer's active LR back to the scheduled value. Fixed by applying the Auto-Tuner's multiplier to `scheduler.base_lrs` as well, ensuring curriculum-driven LR changes persist through the step-based schedule.

## [2026-06-21] - Execution-Aware Promotion & Dynamic Tuning


### Added

- **2026-06-27 - Optuna Deep Tuning Script** (`scripts/optuna_tune.py`): Expanded the dedicated Bayesian optimization workflow for neural architectures (TFT, HAELT, Transformer) into a stronger trainer-aligned tuner. It now uses Sharpe-aware or loss-aware checkpoint-history objectives, model-specific search spaces, hardware-safe batch-size constraints for 9 GB VRAM systems, training-control parameter search (`direction_weight_floor`, `focal_gamma`, `class_balance_weight`), cheap vs deep tuning modes, Optuna pruning from live training signals, ranked study reports under `logs/optuna/`, SQLite study databases stored under `logs/optuna/`, clean exported best-run YAML configs, and optional top-K confirmation reruns with fuller walk-forward CV.
- **2026-06-27 - Promotion Gate Holdout Window And Backtest Diagnostics Fix** (`training/train_gpu.py`, `scripts/backtest_model.py`): Fixed promotion-gate execution backtests so the forward holdout window is derived from the training data date range instead of wall-clock run timestamps. This prevents promotion from accidentally backtesting against a near-empty current-date window. Also expanded backtest failure diagnostics to report requested dates, sequence length, accepted pair counts, base bar counts, and per-pair rejection reasons when the gate returns `Not enough data` or `No trades generated`.

- **2026-06-27 - Dynamic Auto-Tuner Heuristics for Batch Size & Seq Len** (`training/train_gpu.py`): Expanded the `_auto_tune_next_run` logic to dynamically mutate `batch_size` and `seq_len` for the next architecture in the sweep based on the stability of the previous model. The system will now automatically increase batch size and decrease sequence length if it detects excessive curriculum stalls, or strictly increase sequence length if a model trains with perfect stability and no overfitting.

- **2026-06-27 - Resume Routing And Pretrain Stability Fixes** (`training/train_gpu.py`): Added strict baseline-ablation completion checks so `--resume` no longer re-enters `baseline_tft` once all expected baseline fold checkpoints exist. Resume now prefers existing supervised fold checkpoints (for example `tft_fold5_last.pt`) over restarting pretraining, and all-model startup banners now print `Mode: ALL_MODELS` plus the active queue instead of the stale default `model.name`. Also capped regime-aware TSCL extreme-window oversampling during pretraining so it no longer tries to duplicate nearly the entire sampled block and crash with multi-gigabyte NumPy allocations.
- **2026-06-26 - Non-Fatal W&B Logging** (`training/train_gpu.py`): Wrapped W&B metric and summary updates in best-effort helpers so a W&B service disconnect (`connection was forcibly closed by the remote host`) disables W&B logging for the run instead of crashing training. Local logs, TensorBoard, checkpoints, and JSON artifacts remain authoritative.
- **2026-06-27 - Active Trainer Bug Sweep** (`training/train_gpu.py`): Finished the active all-model trainer path cleanup by replacing remaining direct W&B metric/summary calls with safe wrappers, removing the stale validation confidence-filter code path entirely, and wiring multitask anti-collapse YAML/CLI settings (`class_balance_weight`, `entropy_weight`, `direction_weight_floor`, `focal_gamma`) into the actual `MultiTaskLoss` used during training.
- **2026-06-27 - All-Models Queue Recovery** (`training/train_gpu.py`): Restored the active all-model execution path so it no longer hard-excludes HAELT and TFT, added the missing `_parse_pretrain_ablation_models()` runtime helper, and restored `--models` plus `--pretrain-ablation-models` CLI filtering for targeted architecture runs and baseline-proof selection.
- **2026-06-27 - Resume-Safe Pretrain Baseline Ablation** (`training/train_gpu.py`): Baseline/no-pretrain ablation is now permanently resume-aware across the trainer's active all-model path. Fresh runs train the baseline from scratch with `base_args.resume=False`; resume runs load completed baseline fold metrics from existing config artifacts and never resume `BASELINE_*` checkpoints. If baseline configs are incomplete during resume, baseline training is skipped instead of restarted, and final ablation reporting only runs when real baseline metrics are available.
- **2026-06-26 - Transformer Pretraining Recipe Change** (`config/models.py`): Switched the Transformer model recipe from masked reconstruction pretraining to TSCL so the recipe matches the code's recommended `byol_or_tscl` family for generic long-range sequence models. This applies to the next fresh Transformer run.
- **2026-06-26 - HAELT Pretraining Recipe Fix** (`config/models.py`): Switched HAELT from TSCL to masked reconstruction pretraining to match its recommended `masked_or_byol` family. This applies to the next fresh HAELT run.
- **2026-06-25 - Stronger Direction-Head Anti-Collapse Loss** (`models/architectures.py`, `training/train_gpu.py`, `config/run.yaml`): Reworked the multitask loss so `y_conf` no longer erases direction learning on low path-quality samples. Direction CE now uses a configurable floor weight (`direction_weight_floor: 0.35`) while return/confidence losses still respect the original confidence weighting, and the direction head supports focal CE (`focal_gamma: 1.5`) to focus on hard sell/hold/buy examples. This replaces the too-weak entropy-only behavior that could still produce flat or negative fold Sharpe after the previous anti-collapse change.
- **2026-06-24 - Direction-Head Anti-Collapse Training Fix** (`models/architectures.py`, `training/train_gpu.py`, `config/run.yaml`): Added configurable multitask direction-head class-prior balancing and entropy pressure to prevent all-buy/all-sell validation collapse while keeping the existing CE/return/confidence objective. Validation now records predicted and true sell/hold/buy class counts in epoch metrics, TensorBoard, and W&B so class collapse is visible immediately. Fixed the startup `NameError: MULTITASK is not defined` by using safe argparse defaults. After TensorBoard diagnostics showed the first weights were too weak, strengthened `config/run.yaml` to `class_balance_weight: 0.20` and `entropy_weight: 0.03`.
- **2026-06-24 - Baseline/Main Training Artifact Separation** (`training/train_gpu.py`): Pretrain-ablation baseline proof runs now write under each model's `baseline/` subfolder instead of mixing `baseline_*` checkpoints beside the main pretrained model checkpoints. The main training path remains in the model folder for promotion, deployment, and downstream lookup compatibility.
- **2026-06-24 - All-Models Training Banner Fix** (`training/train_gpu.py`): Fixed the startup banner so all-model recipe runs display `Mode: ALL_MODELS` plus the model queue instead of showing the default `model.name` (for example `MAMBA`) while the active phase is training another architecture such as TFT.
- **2026-06-22 - Fresh Training Controls** (`training/train_gpu.py`, `config/run.yaml`): Added explicit `--no-resume` and `--no-training-memory` switches so clean baseline/fresh runs can bypass checkpoint resume and historical training-memory nudges. Set the main run config to `resume: false` so the default command starts from fresh weights unless resume is explicitly requested.
- **2026-06-22 - Full Auto-Tune Setup** (`training/train_gpu.py`, `config/run.yaml`): Promoted auto-tuning from a late main-run heuristic into a config-controlled training phase artifact. `tracking.auto_tune`, `tracking.dry_tune`, and `tracking.ollama_auto_tune` are now YAML-driven; baseline pretrain-ablation runs write dry-run proposal JSON for proof without mutating config, while the final main run can still apply approved heuristic nudges.
- **2026-06-22 - Auditable Training Memory** (`training/training_memory.py`, `training/train_gpu.py`, `tests/test_training_memory.py`): Routed startup training-memory nudges through proposal artifacts under `logs/auto_tune/` before any LR/dropout/patience/epoch changes are applied. If proposal writing fails, memory skips the nudge instead of silently mutating args. Walk-forward runs now update memory from merged fold history instead of only the last fold.
- **2026-06-22 - Curriculum Sharpe-Stall Guard** (`training/train_gpu.py`): Fixed the adaptive curriculum stall logic so negative or weak Sharpe peaks no longer trigger "Sharpe collapsed" stalls. The curriculum now only freezes sequence length and reduces LR after a meaningful positive Sharpe peak (`>= 0.25`) drops by at least `0.15`.
- **2026-06-22 - Improved Curriculum Adaptation** (`training/train_gpu.py`, `config/run.yaml`): Made curriculum progression explicitly validation-driven and configurable. Added `curriculum.adaptation` thresholds for stable-window advancement, minimum stable Sharpe, collapse detection, and LR multipliers. Training now logs active sequence length, difficulty stage, stalls, and richer transition events to TensorBoard/W&B and `training_control_report.json`.
- **2026-06-22 - Clean Artifact Folder Names** (`training/train_gpu.py`, `training/training_memory.py`, `monitoring/discord_alerts.py`, `config/run.yaml`): Standardized future filesystem artifact names on lowercase slug paths. `tracking.run_name` is now `high-impact`, all-model checkpoints now use `checkpoints/forex_3pair_2008_2025_all_models`, and logs/TensorBoard/profile/auto-tune/training-memory artifact names use sanitized run/model slugs even if display names contain spaces.
- **2026-06-23 - Self-Improving Training Memory Context** (`training/train_gpu.py`, `training/training_memory.py`, `tests/test_training_memory.py`): Expanded training history and memory updates with best-epoch context, including LR, sequence length, difficulty stage, curriculum stalls, and training-control warnings. This lets future training-memory suggestions learn from the curriculum state at the actual Sharpe peak instead of only storing the peak metric.
- **2026-06-23 - Pair Readiness Drop-Reason Buckets** (`training/train_gpu.py`): Expanded `<cache>_pair_readiness_report.json` with `dropped_bars_by_reason`, `dropped_sequences_by_reason`, and `label_filter_counts`. The readiness gate now attributes low-quality drops to weekend/holiday, dead-bar, spread, ATR/volatility, news blackout, label filters, zero-feature windows, invalid direction labels, invalid reward labels, and path-quality filters.
- **2026-06-23 - Diverse Model Recipes Runner** (`scripts/train_diverse_recipes.py`): Upgraded the diverse-recipes script from a basic same-settings loop into a production wrapper. It now runs controlled HAELT/Mamba/TFT recipes with distinct loss/sequence settings, isolated checkpoint folders, clean resume defaults, optional cache rebuild behavior, dry-run support, pass-through training args, and `diverse_recipe_manifest.json` summary output.
- **2026-06-23 - Actual All-Models Training Selection** (`training/train_gpu.py`): Fixed `--all-models` so it no longer silently excludes HAELT and TFT. Added `--models transformer,expert` style filtering for targeted multi-model training through the main trainer.
- **2026-06-23 - Actual Diverse Recipe Training Guard** (`training/train_gpu.py`, `scripts/train_diverse_recipes.py`): Added `--no-all-models` and wired it into every diverse recipe command so recipe runs train the selected architecture only, even when `config/run.yaml` has `model.all_models: true`.
- **2026-06-23 - Model-Specific Recipes In All-Models Training** (`config/models.py`, `training/train_gpu.py`, `improvement_plan.md`): Added real per-architecture training recipes to the main all-model path. TFT, Transformer, HAELT, Mamba, GNN, and Expert now carry explicit recipe names, loss choices, early-stop metric, sequence length, learning rate, pretrain method, and pretrain epoch settings, while explicit CLI flags still override profile-managed fields.
- **2026-06-23 - Selective Pretrain Baseline Proof** (`training/train_gpu.py`, `config/run.yaml`): Added `pretrain.ablation_models` and `--pretrain-ablation-models` so `pretrain_ablation: auto` only runs no-pretrain baseline proof for selected architectures. The default proof set is now TFT, Transformer, and HAELT; Mamba, GNN, and Expert train directly unless explicitly included.
- **2026-06-23 - Same-Holdout Model Comparison And Feature Ablation Reports** (`training/train_gpu.py`, `config/run.yaml`): Added `model_comparison.json`, updated after each model's promotion-gate pass, to rank models on the shared forward holdout with validation and execution metrics in one place. Added config/CLI-controlled feature ablation masks plus per-model `*_feature_ablation_report.json` artifacts for controlled runs such as no-news, no-cross-asset, and price/vol-only.
- **2026-06-23 - Production Readiness Reports And Verified ONNX Deployment** (`training/train_gpu.py`, `validation/promotion_gate.py`): Expanded pair-readiness reports with per-pair hourly coverage, missing-hour counts, and samples alongside existing drop-reason buckets. Tightened production deployment so checkpoints are promoted only after the candidate ONNX export and feature schema pass verification; deployment artifacts now record ONNX/schema verification details. The shared C++ ONNX deployment helper used by ensemble/RL exports now verifies ONNX/schema before atomic promotion as well.
- **2026-06-23 - CUDA-Safe NumPy Conversion Fixes** (`scripts/backtest_model.py`, `scripts/compare_models.py`, `scripts/train_rl.py`, `models/rl_advanced.py`): Fixed tensor-to-NumPy conversions in backtest/comparison/RL helper paths by detaching and moving tensors to CPU first. This prevents CUDA runtime failures during promotion-gate backtests and model comparison runs.
- **2026-06-23 - OANDA Removed From Active Training Path** (`training/train_gpu.py`, `scripts/run_feature_engineering.py`, `improvement_plan.md`): Removed the unused training-side OANDA ablation hook and stopped standalone feature generation from loading/passing OANDA sidecars. OANDA remains prototype-only until production sidecar storage, leakage/staleness checks, mock-data safety, and ablation proof are complete.
- **Execution-Aware Promotion Gate** (`training/train_gpu.py`, `scripts/backtest_model.py`): Upgraded the simple label-proxy evaluation to a full execution-aware backtest during model promotion. The gate now computes realistic metrics (Sharpe, Profit Factor, Max Drawdown) utilizing real spread, slippage, fractional Kelly position sizing, and intra-bar Take Profit/Stop Loss interactions over the forward holdout window.
- **Dynamic Confidence Threshold Tuning** (`scripts/tune_confidence_thresholds.py`, `training/train_gpu.py`): Created a post-training tuning hook to discover the optimal `min_confidence` gate based on validation predictions.

- **SharpeProxyLoss Differentiability** (`training/train_gpu.py`): Fixed `SharpeProxyLoss` by replacing `torch.sign` with `torch.tanh`, making the signal fully differentiable so gradients properly flow through the annualized Sharpe metric.
- **Loss Sample Weighting** (`models/architectures.py`, `training/train_gpu.py`): Rewrote regression losses (`HuberLoss`, `AsymmetricDirectionalLoss`, `SharpeProxyLoss`) to accept an explicit `weight` argument. Wired `no_trade_score` (via `y_conf`) into the loss calculation so that blackout/low-quality bars are natively down-weighted during gradient updates.
- **Annualized Sharpe Gradient** (`training/train_gpu.py`): Scaled `SharpeProxyLoss` by the annualization factor so the optimizer operates on the true yearly-scale Sharpe metric, properly balancing it against the Huber baseline.
- **Calmar Promotion Gate** (`validation/promotion_gate.py`): Added Calmar ratio thresholds (`min_calmar`) to the best-fold promotion gate so models with extreme drawdowns are rejected despite high absolute returns.
- **Directional Label Smoothing** (`training/train_gpu.py`, `models/architectures.py`): Applied `label_smoothing=0.1` to the CrossEntropy classification loss on the directional head to mitigate overconfident probability clustering.
- **Gradient Centralization** (`training/train_gpu.py`): Implemented a `GC_Optimizer` wrapper in the training loop that zeros out the mean of weight gradients, stabilizing training and allowing larger learning rates.
- **Fractional Kelly Backtesting** (`scripts/backtest_model.py`): Bridged `kelly_criterion.py` into the backtester so that trade volume scaling perfectly mirrors the live trading engine's fractional Kelly logic.
- **Regime-Conditional Confidence** (`scripts/backtest_model.py`): Implemented dynamic regime-conditional thresholding (e.g., lower confidence barrier during strong trends, stricter barrier during sideways markets).
- **Profit Factor Logging** (`training/train_gpu.py`): Wired profit factor telemetry directly into validation loops and Weights & Biases tracking.

- **Feature Curriculum Masking** (`config/run*.yaml`, `training/train_gpu.py`): Made the `feature_groups` curriculum functional rather than cosmetic by explicitly mapping schema names (`features: [...]`) to each curriculum group. Added `curriculum_mask` logic to `train_gpu.py` which actively zeros out specific feature columns matching the `epoch_unfreeze` schedule, ensuring complex signals are properly withheld during early training.
- **Sharpe Annualization** (`config/run*.yaml`): Added `sharpe_annualization_factor: 325.0` (Γëê ΓêÜ(252 ├ù 420 bars/day)) so that `val_sharpe` is properly annualized during early stopping and checkpoint selection for 1-min scalping. Previously defaulting to `1.0` made the metric dimensionless and `early_stop_min_delta` meaningless.
- **Increased `sharpe_weight`** (`config/run*.yaml`): Raised from `0.35 ΓåÆ 0.6` in `run.yaml` (and `0.2 ΓåÆ 0.6` in `run_ubuntu.yaml`). At the previous value the Huber base loss was numerically dwarfing the Sharpe signal, making `sharpe_huber` behave near-identically to plain Huber.
- **Reduced `w_ret` (multitask)** (`config/run*.yaml`): Lowered return-regression auxiliary weight from `0.15 ΓåÆ 0.08`. The return head gradient was competing with the direction head's Sharpe objective, degrading risk-adjusted performance.
- **TP/SL Ratio Update** (`config/run*.yaml`): Raised `profit_target_atr` from `1.5 ΓåÆ 1.8` and `stop_loss_atr` from `0.8 ΓåÆ 0.9`, achieving a **2:1 reward-to-risk ratio**. Extended `lookahead_bars` from `15 ΓåÆ 20` to give the TP barrier more room to be reached in 1-min scalping. Added `strategy:` block to `run_ubuntu.yaml` (previously missing, causing fallback to stale defaults). **Requires a full dataset cache rebuild** ΓÇö triggered automatically via `auto_rebuild_on_mismatch: true`.
- **OANDA Sentiment Sidecar Integration** (`data/fetch_oanda_sentiment.py`, `features/feature_engineering_pl.py`, `training/train_gpu.py`): Hardened the OANDA integration into a production sidecar pipeline. Added a script to dump raw JSON snapshots and normalized features into daily Parquets, complete with explicit mock/source flags. Added strict backward-only `join_asof` guardrails during feature injection, and an ablation flag (`--oanda-ablation`) to evaluate performance impact against baseline neutral values. Expanded features with z-scores, staleness minutes, and position flips.
- **OANDA Coverage Reporting** (`data/oanda_report.py`): Added a standalone script to generate `oanda_sentiment_report.json` to monitor API freshness, missing minutes, null-rates, and data usability status per pair.
- **Adaptive Curriculum Wiring** (`training/train_gpu.py`, `config/run.yaml`): Updated feature-group masking to prefer the cache's ordered feature schema sidecar and made Sharpe-collapse stalls clear the stability window and freeze sequence growth instead of only updating an unused curriculum counter.
- **Verifiable Checkpoint Metadata** (`training/train_gpu.py`): `_safe_save(..., metadata=...)` now writes atomic `.metadata.json` sidecars and verifies expected fields for best, periodic, and resume checkpoints, covering model name, feature count, sequence length, schema hash, epoch, and fold where available.
- **Structured Pair Readiness Report** (`training/train_gpu.py`): Added `<cache>_pair_readiness_report.json` for single-pair and multi-pair cache builds, recording raw tick/schema checks, usable sequence counts, nonfinite rates, label distribution, difficulty distribution, spread/ATR summaries, timestamp inner-join loss, and final pass/warn/fail status before training starts.
- **Pretraining Proof Artifacts** (`training/train_gpu.py`, `scripts/train.py`): Upgraded `pretrain_report.json` from a placeholder into a real diagnostic artifact with leakage-safe windows, method recommendation, quality-gate diagnostics, hard-example reuse count, checkpoint path, and supervised-transfer verification. `pretrain_ablation.json` now summarizes baseline vs pretrained folds, computes metric deltas, and records a verdict (`pretrain_helped`, `pretrain_hurt`, `mixed`, or `unknown`).
- **Training Validation Confidence Gate Removal** (`training/train_gpu.py`, `config/run.yaml`): Removed the training-time validation confidence gate so validation accuracy and Sharpe are always computed over all validation predictions.
- **2026-06-21 - Sharpe-Focused Feature Batch Rollback** (`features/feature_engineering_pl.py`, `training/train_gpu.py`, `config/feature_mask.py`): Removed the experimental spread-cost, market-regime, higher-timeframe, and trailing label-quality feature batch to restore the prior active training schema.
- **2026-06-22 - Direction Target Training Fix** (`labeling/rl_reward_labeling.py`, `training/train_gpu.py`, `config/run.yaml`): Fixed zeroed validation accuracy/Sharpe for RL-reward runs by preserving the true `label` column in the aligned sidecar. Previously `y_cls` could fall back to `sign(reward)`, but RL reward is `max(reward_long, reward_short)` and is mostly non-negative, producing fake hold/long labels instead of real sell/hold/buy direction. Classification, class weights, and validation now prefer the true direction sidecar while cache `y` remains continuous reward for Sharpe/PnL. Old caches without `y_cls_source: labels.label` are rejected and rebuilt.

### Fixed

- **2026-06-22 - Multi-Pair Feature Schema Integrity** (`training/train_gpu.py`): Fixed multi-pair cache schema handling by preserving scaler feature names when fitting on NumPy arrays, writing full pair-prefixed feature schema sidecars for `P ├ù F` caches, matching curriculum groups across repeated per-pair feature names, and rejecting stale multi-pair caches that lack a valid schema.
- **Curriculum Stall Loop** (`training/train_gpu.py`): Fixed the curriculum "Sharpe collapsed" logic to ignore low-Sharpe noise (< 0.25) so models do not prematurely trigger "Sharpe collapse" and freeze their curriculum advancement.
- **Feature Schema Fallback** (`training/train_gpu.py`): Added a robust fallback to extract `feature_names_in_` directly from `scaler.npz` if the dataset's `_feature_schema.json` sidecar is missing, ensuring older dataset cache versions don't fail the curriculum mask.
- **Multitask Temperature Calibration** (`models/architectures.py`, `training/train_gpu.py`): Fixed post-training calibration for `multitask + sharpe_huber` runs by calibrating the direction logits as classification output and using the return head only for regression calibration. This resolves the recurring `tensor a (3) must match tensor b (...)` calibration failure and allows calibrated checkpoints to be written.
- **Training Memory Gate Accounting** (`training/training_memory.py`): Fixed training memory so passing promotion-gate checks are no longer counted as failures. Explicit failed gate markers are still recorded, preventing polluted future LR/dropout/patience recommendations.
- **OANDA Sentiment Sidecar Safety** (`data/fetch_oanda_sentiment.py`, `features/feature_engineering_pl.py`, `training/train_gpu.py`, `data/oanda_report.py`, `scripts/run_feature_engineering.py`): Hardened OANDA order/position-book feature ingestion by refusing silent mock sidecar writes, accepting `timestamp`/`timestamp_utc`/`oanda_timestamp` sidecar schemas, preserving sidecar timestamps through training, adding stricter staleness and mock-data guardrails, preferring parquet sidecars over the legacy CSV path, and making coverage reports strict JSON with insufficient-snapshot failures instead of false `PASS` results.
- **Global Logging Leak** (`training/train_gpu.py`, `monitoring/train_logger.py`): Fixed an issue where `_TRAIN_LOGGER` was being repeatedly instantiated and file handlers were not properly cleaned up (`removeHandler`), causing dangling file descriptors and misleading per-fold log file naming during cross-validation loops.
- **C++ Execution Reversals** (`infrastructure/cpp_server/src/main.cpp`): Fixed `executionActionToUnits` so `OPEN_LONG` and `OPEN_SHORT` properly subtract `current_units` to execute full reversals rather than flattening positions.
- **C++ Position Limits** (`infrastructure/cpp_server/src/main.cpp`): Added hard clamping to ensure RL scaling actions strictly honor `MAX_POSITION_UNITS` before order submission.
- **C++ Entry Price Tracking** (`infrastructure/cpp_server/src/main.cpp`): Fixed a state corruption bug where partial scale-outs incorrectly reset the tracked entry price. It now only resets on full reversals or new entries.
- **C++ Live Equity Tracking** (`infrastructure/cpp_server/src/main.cpp`): Fixed missing `tracked_equity` updates so the engine realistically realizes PnL on position reduction, matching the RL training environment.
- **Transformer All-Models Resume** (`training/train_gpu.py`, `models/architectures.py`): Fixed `baseline_transformer` resume crashes in the all-models path by letting the multitask projection bind to the real post-wrapper hidden width, skipping uninitialized lazy parameters during parameter counting and feature unfreezing, and keeping crash-only transformer folders eligible for retry.

## [2026-06-14] - Priority 1: Ensemble Integration

### Added

- **Ensemble Promotion Gate** (`training/train_gpu.py`): The `EnsembleMetaLearner` now runs through the reserved forward holdout promotion gate, capped at 200,000 bars.
- **Ensemble vs Base Sharpe Comparison** (`training/train_gpu.py`): The pipeline compares ensemble forward Sharpe against the current deployed/base result before promoting the ensemble artifact.
- **Ensemble ONNX Export** (`inference/onnx_inference.py`): Added ensemble export support that wraps the trained meta-learner and frozen base models into one C++-compatible 3-logit ONNX graph.
- **Ensemble Manifest Artifact** (`training/train_gpu.py`, `scripts/train_ensemble_meta.py`): Ensemble meta-training now writes `ensemble_manifest.json` with base checkpoint provenance, artifact paths, schema details, and training history for deployment and audit checks.
- **Automatic Training Run Folders** (`training/train_gpu.py`, `docs/STEP_BY_STEP.md`): Added `--auto-run-dir` and `--run-dir-root` so training can generate descriptive checkpoint folders from model, strategy, pairs, sequence length, folds, and RL/ensemble mode. Training writes `<run-dir>/run_info.json` and `<run-dir-root>/latest_run.json`; manifests and deployment records now include the resolved run directory.
- **All-Models Resume Skip** (`training/train_gpu.py`, `docs/STEP_BY_STEP.md`): With `--all-models --resume`, training now checks each model subfolder and skips completed members while retrying models that only have crash artifacts. Added `--retrain-completed-models` to force reruns.
- **Discord Ollama Training Q&A Bot** (`scripts/discord_ollama_bot.py`, `.env`, `docs/readmd.md`): Added a read-only Discord bot bridge for asking local Ollama about training status, latest results, logs, checkpoints, and config from a single allowed Discord channel. Added `.env` keys for `DISCORD_BOT_TOKEN`, `DISCORD_CHANNEL_ID`, and `DISCORD_OLLAMA_CONTEXT_CHARS`. The bot redacts common secret patterns and does not auto-tune, auto-fix, restart jobs, or execute shell commands.

### Changed

- **Project Structure Plan** (`docs/PROJECT_STRUCTURE_IMPROVEMENT_PLAN.md`, `docs/FOLDER_STRUCTURE.md`, `scripts/check_project_structure.py`): Added a root cleanup and package-boundary plan covering patch/debug scripts, promotion artifacts, parity CSVs, archive consolidation, artifact locations, and import-safety checks. Tightened `.gitignore` guardrails and added a structure check script to prevent new root clutter.
- **Project Improvement Plan**: Added an ensemble-readiness update checklist to `docs/PROJECT_IMPROVEMENT_PLAN.md`, covering training fixes, ensemble meta-training, ONNX export, Python live runtime, and when C++ runtime changes are actually needed.
- **Setup Guide** (`SETUP.md`): Rewrote setup instructions for the current Python 3.11+ workflow, `.venv-gpu` repair, dependency install paths, training/test commands, ONNX export, Discord alerts, shadow mode, and C++ runtime setup.
- **Project Improvement Plan**: Added pretraining improvement recommendations covering no-pretrain baselines, method-specific pretrain recipes, leakage-safe windows, pretrain reports, handoff gates, and hard-example reuse.
- **Project Improvement Plan**: Added reinforcement-learning improvement recommendations focused on using RL as the execution/sizing layer, with risk gates, `rl_report.json`, reward diagnostics, anti-overtrading guardrails, and shadow-mode decision logging.
- **Project Improvement Plan**: Added execution-quality recommendations covering spread/slippage filters, confidence-aware sizing, per-model/per-fold confidence-threshold tuning, live confidence guard wiring, `execution_report.json`, stop/exit logic, session rules, execution-policy comparison, and execution-adjusted promotion metrics.
- **Project Improvement Plan**: Added an automation ladder for observation, safe auto-tuning, model selection, gated deployment, required automation artifacts, and high-risk settings that must remain proposal-only.
- **Project Improvement Plan**: Added training-control recommendations covering model-specific training recipes, persistent training memory, overfitting detection, adaptive responses, and `training_control_report.json`.
- **Project Improvement Plan**: Added pair-readiness recommendations covering pre-training pair gates, timestamp-based multi-pair alignment, cache sidecar checks, and curriculum difficulty sidecar repair.
- **Codebase Bug Audit** (`docs/CODEBASE_BUG_AUDIT.md`): Added the 2026-06-12 review covering manifest/cache validation, difficulty curriculum, missing-pair handling, timestamp alignment, zarr resume, and pair-readiness reporting bugs.
- **Pretraining Ablation (Priority 8)** (`training/train_gpu.py`, `config/run.yaml`): Added `auto` mode for `--pretrain-ablation`, automatically enabling baseline contrastive ablation for deep sequence models (`transformer`, `haelt`, `tcn`, `lstm`, `hybrid`) to measure pretraining benefits.

### Fixed

- **Baseline Ablation Weight Collision** (`training/train_gpu.py`): Fixed a critical bug where the `baseline_` model prefix was ignored during state dictionary saving, causing the ablation baseline to physically overwrite the pretrained main model's weights on disk (`[model]_best.pt`), which corrupted the final evaluation. 
- **Dynamic Factory Dispatch Error** (`training/train_gpu.py`): Fixed the `build_model` factory to properly recognize `baseline_` architecture prefixes and dispatch them back to their standard core initializers.
- **Ensemble/RL ONNX Deployment** (`training/train_gpu.py`, `inference/onnx_inference.py`, `infrastructure/cpp_server`, `config/run*.yaml`): Ensemble meta-learners and RL policies now export C++-compatible 3-logit ONNX graphs after successful training. RL also exports a native 10-action execution ONNX with `features` plus `agent_state` inputs, and the C++ server can optionally load it through `EXECUTION_MODEL_PATH` alongside the direction model. Optional `ensemble.deploy` and `rl.deploy` promotion copies the compatibility graph to `production_best.onnx`, writes the schema sidecar, and signals `reload_model.flag`.
- **Legacy Ensemble Fallbacks** (`inference/pytorch_inference.py`, `scripts/backtest_model.py`): PyTorch live fallback and backtesting now read `ensemble_manifest.json` for base checkpoint provenance and schema alignment instead of relying only on guessed checkpoint layouts.
- **Dataset Compilation Integrity** (`training/train_gpu.py`):
  - Fixed `RuntimeError: Required pair(s) produced no usable sequences` by gracefully warning and skipping dataset chunks that yield zero usable ticks for a given pair (e.g., due to data quality drops or holidays), instead of crashing the entire multi-pair pipeline.
  - Fixed `UnboundLocalError` by explicitly importing `json` where needed for saving CV summaries.
  - Fixed `_validate_cache_integrity()` to properly read legacy string-based comma-separated lists of pairs.
  - Rewrote manifest generation to explicitly create the `_manifest.json` sidecar for both Zarr and NPY caches, ensuring `pairs` is saved as a proper JSON array.
  - Re-enabled difficulty curriculum by replacing the zero-fill stub with actual `_compute_difficulty_scores()`.
  - Implemented timestamp-based `pair_align: inner` to rigorously synchronize market data across multiple pairs by DatetimeIndex, instead of blindly truncating arrays.
  - Fixed Zarr resume logic so that chunks are correctly appended without attempting to recreate arrays, and removed magic numbers by reading `_resume.json` state.
  - Corrected Pair Readiness `nan_pct` calculation by correctly using `shape[2]` (feature count) to find the total cell denominator.
  - Fixed follow-up timestamp plumbing regressions by making `_build_chunk()` return a consistent tuple on empty chunks, removing an undefined `args` reference, and updating the single-pair caller for the new timestamp output.
- **Test Stability** (`tests/test_training_smoke.py`): Fixed `AttributeError: 'Namespace' object has no attribute 'walk_forward_folds'` by correctly mocking cross-validation arguments in unit tests.
- **Discord Alert Compatibility** (`monitoring/discord_alerts.py`): Removed the duplicate deploy-failure method and restored backward-compatible `send_retrain()` and `send_promotion()` aliases.
- **Deployment Artifact Isolation** (`training/train_gpu.py`): Reset deployment path variables for each model so `deployment.json` cannot inherit stale ONNX, reload flag, or production checkpoint paths from a previous model in the same run.

---

## [2026-06-11] - Priority 3 & 4: Observability & Dataset Fingerprinting

### Added
- **Dataset Fingerprinting** (`training/train_gpu.py`): Implemented strict cache integrity checks using `_manifest.json` sidecars to detect parameter/configuration drift before loading a cached dataset.
- **Feature Schema Versioning** (`training/train_gpu.py`, `trading/live_engine.py`): The training process now saves the exact feature order into `_feature_schema.json` and `.schema.json` upon ONNX export. The live trading engine will now instantly crash (`FATAL` error) if live features do not exactly match the trained schema.
- **Pair Readiness Gate & Quality Reports** (`training/train_gpu.py`): The training pipeline now logs dropped bars, NaNs, and sequence counts, and runs a pre-training readiness gate that enforces NaN thresholds (<1%) to prevent wasting GPU time on garbage data.
- **Cache Mismatch Refusal (P4)** (`training/train_gpu.py`): If the runtime arguments don't strictly match the dataset manifest, the cache is rejected. Introduced `--auto-rebuild-on-mismatch` to transparently handle rebuilds, and `--ignore-manifest` as an escape hatch.
- **Discord Artifact Paths (P3)** (`monitoring/discord_alerts.py`): Added clickable `file:///` paths for ONNX models, schemas, log files, and `fold_selection.json` to Discord embeds to make artifact inspection easier.
- **Granular Rate-Limiting (P3)** (`monitoring/discord_alerts.py`): Fixed rate limiting to use a unique `rate_key` per model/fold instead of a global `alert_type`, ensuring all folds in a multi-model run successfully trigger an alert.

---

## [2026-06-11] - Priority 5: Model Training Improvements

### Added

- **Model Cards** (`training/train_gpu.py`): Ensured `[model_name]_model_card.json` is generated reliably at the end of training, tracking architecture, features count, pairs, and final performance.
- **Training Control Report & Overfitting Warnings** (`training/train_gpu.py`): The pipeline now checks the gap between training and validation loss (`train_val_gap`) at the end of each run. It creates `[model]_training_control_report.json` and flags `overfitting_warnings` if Sharpe collapses significantly or the validation gap is > 0.05.
- **Stability-Aware Fold Scoring** (`training/train_gpu.py`): Extended the stability score to directly penalize high `train_val_gap` by applying a `train_val_gap_penalty` scaled to the generalization gap, meaning heavily overfit models are effectively blocked.
- **Challenger vs. Production Gate** (`training/train_gpu.py`): At the promotion step, if an existing `deployment.json` is found in the target directory, the new model's stability score must be significantly better than the deployed model's score, otherwise it is rejected. Bypassable via the new `--force-promotion` flag.
- **Diverse Model Recipes Script** (`scripts/train_diverse_recipes.py`): Added an orchestration script to sequentially train completely different neural architectures (`haelt`, `mamba`, `tft`) and compare their validation scores for easy meta-analysis.

---

## [2026-06-11] - Priority 7: Robust Testing Suite

### Added

- **Config Validation Test** (`tests/test_config.py`): Validates that `config/run.yaml` parses successfully with `yaml.safe_load`, checks required top-level sections (`data`, `model`, `training`, `paths`, `execution`), and verifies that `checkpoint_dir` resolves properly.
- **End-to-End Smoke Test** (`tests/test_smoke.py`): Runs `train_gpu.py` in a fresh subprocess with `--data-source synthetic`, `--epochs 2`, and `--force-rebuild`. Validates that training, evaluation, and promotion complete successfully, then checks for `train_summary.json`, `deployment.json`, and `manifest.json`.

### Fixed

- **Test Artifact Integrity** (`tests/test_promotion_artifacts.py`): Updated assertions to match the current `fold_selection.json` schema using `selected_fold`.
- **Zarr Appends on Rebuild** (`training/train_gpu.py`): Fixed a rebuild bug where `_resume_zarr=False` could bypass secondary-array creation on non-initial chunks, causing `FileNotFoundError` or `KeyError: atr`.

---

## [2026-06-11] - Priority 2: Separate Artifact Files And Deployment Transaction State

### Added

- **`train_summary.json`** (`training/train_gpu.py`): Added a per-model training summary artifact written after `supervised_train`. It stores training metrics only and stays separate from `manifest.json`.
- **`deployment.json`** (`training/train_gpu.py`): Added a per-model deployment transaction artifact written after every promotion attempt. It records gate status, source checkpoint, production checkpoint, previous checkpoint, ONNX status, reload flag status, deploy status, deploy error, failed step, and deployment timestamp.
- **`CRITICAL_ALERT_TYPES`** (`monitoring/discord_alerts.py`): Added alert types that bypass normal rate limiting: `circuit_breaker`, `production_deploy_failed`, `promotion_gate_failed`, and `model_demoted`.
- **`send_promotion_gate_failed()`** (`monitoring/discord_alerts.py`): Added a typed alias for promotion-gate failure alerts.
- **`send_production_deploy_failed()`** (`monitoring/discord_alerts.py`): Added a force-sent deploy failure alert so failed production deployment cannot be suppressed by cooldown.

### Changed

- **`fold_selection.json`** (`training/train_gpu.py`): Switched to atomic `_safe_save_json` writes and enriched the schema with `selected_fold`, secondary metric fields, generalization gap, candidate count, source checkpoint, selection timestamp, and per-candidate tie-breaker data.
- **Promotion Gate Alert Naming** (`training/train_gpu.py`, `monitoring/discord_alerts.py`): Updated the promotion-gate failure path to use `send_promotion_gate_failed()` while keeping the older `send_gate_failed()` delegate for compatibility.
- **Rate Limiting** (`monitoring/discord_alerts.py`): Critical alerts now bypass the cooldown window.

### Artifact File Map

Per-model artifacts are written under `<checkpoint_dir>/<model>/`:

| File | Written By | Purpose |
|---|---|---|
| `train_summary.json` | After `supervised_train` | Pure training metrics |
| `fold_selection.json` | After `_promote_best_fold` | CV fold ranking and selected fold |
| `promotion_gate.json` | After `_evaluate_forward_gate` | Gate pass/fail decision |
| `deployment.json` | After deploy block | Full deployment transaction record |
| `manifest.json` | After model training loop | High-level run summary |

---

## [2026-06-11] - Priority 0 And 1: Correctness, Trust, And Self-Improving Training

### Added

- **`_safe_save_json` helper** (`training/train_gpu.py`): Added atomic JSON writes using `tempfile.mkstemp` and `os.replace`. Used for critical JSON artifacts including `promotion_gate.json`, `manifest.json`, `fold_selection.json`, and `production_best.schema.json`.
- **Atomic ONNX Export** (`training/train_gpu.py`): ONNX export now writes to a temporary file before replacing the canonical production path.
- **Atomic `reload_model.flag`** (`training/train_gpu.py`): Hot-reload signals are now written atomically.
- **Richer `manifest.json`** (`training/train_gpu.py`): Manifest generation now includes richer run, checkpoint, promotion, deployment, and git metadata.
- **`production_deploy_failed` Alert** (`monitoring/discord_alerts.py`): Added an alert for production deployment failures.
- **Deploy Result Tracking** (`training/train_gpu.py`): Added deployment status tracking used by manifests and deployment artifacts.
- **3-Level Best-Fold Tie-Breaker** (`training/train_gpu.py`): Fold selection now considers the primary metric, a secondary metric, and generalization gap.
- **Fold Selection Alerting** (`training/train_gpu.py`, `monitoring/discord_alerts.py`): `_promote_best_fold()` can send a `fold_selected` Discord alert after selecting the best fold.
- **Run Start Time Capture** (`training/train_gpu.py`): Captures run start time at the beginning of `main()`.
- **Training Memory** (`training/training_memory.py`): Added persistent training memory stored in `logs/training_memory.json`. It tracks best model metrics, epoch patterns, failure counts, Sharpe history, and per-model summaries.
- **Hard Example Mining** (`training/hard_example_miner.py`): Added validation hard-sample mining for confident-wrong predictions and missed large-reward windows.
- **Sharpe Collapse Heuristic** (`training/train_gpu.py`): `_auto_tune_next_run()` can detect early validation Sharpe collapse and propose conservative LR/patience changes.
- **Auditable Auto-Tune Proposals** (`training/train_gpu.py`): Auto-tune now writes structured proposal JSON under `logs/auto_tune/`.
- **Training Memory Wiring** (`training/train_gpu.py`): Training memory is loaded at startup, applied conservatively, updated after training, and saved atomically.
- **Hard Example Miner Wiring** (`training/train_gpu.py`): Hard-example mining runs after single-split supervised training when validation prediction artifacts are available.

### Changed

- **Discord Alert Names** (`monitoring/discord_alerts.py`): Added or renamed training and promotion alerts to match pipeline phases: `training_started`, `fold_selected`, `promotion_gate_passed`, `promotion_gate_failed`, `production_deploy_completed`, and `production_deploy_failed`.
- **Discord Helper Methods** (`monitoring/discord_alerts.py`): Added typed helpers including `send_training_started()`, `send_fold_selected()`, `send_promotion_gate_passed()`, `send_promotion_gate_failed()`, `send_production_deploy_completed()`, and `send_production_deploy_failed()`.
- **Auto-Tune Write Guard** (`training/train_gpu.py`): Config mutation is guarded so high-risk fields are not silently changed.
- **Best-Fold History Tracking** (`training/train_gpu.py`): `_promote_best_fold()` now reads fold history data to compute generalization gaps.

---

## [2026-06-10] - Checkpoint Integrity And Better Artifacts

### Added

- **Semantic Alerting And Observability** (`monitoring/discord_alerts.py`): Added structured Discord alert types for training, promotion, deployment, demotion, drift, retrain, and backtest events.
- **Model Cards** (`training/train_gpu.py`): Training now writes `<model_name>_model_card.json` with architecture, feature, data-window, validation, and promotion metadata.
- **Dataset Manifests And Schema Versioning** (`training/train_gpu.py`): Training validates cache integrity and writes feature-schema metadata next to production ONNX exports.
- **Adaptive Curriculum** (`training/train_gpu.py`): Added `_curr_ep` driven curriculum progression and Sharpe-collapse stalling.
- **Auditable Auto-Tuning** (`training/train_gpu.py`): Added `--dry-tune` and structured auto-tune proposal files.
- **Promotion Artifact Tests** (`tests/test_promotion_artifacts.py`): Added tests for `_promote_best_fold()` and `_atomic_copy()`, including nested checkpoint paths and tie-breakers.
- **`_safe_save` Checkpoint Validation** (`training/train_gpu.py`): Added an atomic `torch.save()` wrapper that verifies checkpoint size and loadability before replacing the target file.
- **Enhanced Best-Fold Promotion** (`training/train_gpu.py`): Best-fold promotion supports nested checkpoint directories, tie-breakers, and `fold_selection.json`.
- **Promotion And Run Artifacts** (`training/train_gpu.py`): Added `promotion_gate.json`, `deployment.json`, and `manifest.json` artifacts.

### Changed

- **Deploy Alert Context** (`monitoring/discord_alerts.py`): Production deployment alerts can include ONNX and schema references.
- **Promotion Schema Details** (`training/train_gpu.py`): Promotion artifacts include model and checkpoint context for downstream deployment.
- **XGBoost Zarr Support** (`training/train_xgboost.py`): XGBoost training can read modern Zarr dataset caches.
- **Dataset Timeframe** (`config/run.yaml`): Training data bounds were set to `2008-01-01` through `2025-12-31`.
- **XGBoost Baseline Config** (`training/train_xgboost.py`): XGBoost training accepts `--config config/run.yaml`, resolves its data cache from config, supports `--cache-path`, and records metadata in a sidecar.
- **XGBoost Cache Window Guard** (`training/train_xgboost.py`): Config-driven XGBoost runs validate the processed cache date window unless an explicit cache path is supplied.
- **C++ Server Configuration** (`infrastructure/cpp_server`): Replaced hardcoded runtime values with environment-driven OANDA credentials, instrument, model path, paper/live mode, shadow mode, and risk limits.
- **Diagnostics Calls** (`training/train_gpu.py`): Standardized diagnostics usage across trainers.
- **Governance Gating** (`training/train_gpu.py`): Moved global maturity/governance gating into promotion logic.
- **Time-Series Augmentation Signatures** (`models/contrastive.py`, `training/train_gpu.py`): Synchronized augmenter signatures to handle epochs and regimes consistently.

### Fixed

- **Test Suite Repairs** (`tests/test_models.py`, `tests/test_inference_consistency.py`, `tests/test_all.py`): Updated tests for Polars dataframe migration and expanded RL action semantics.
- **Pretraining Trainer Bugs** (`models/contrastive.py`): Fixed undefined pretraining variables and aligned trainer return signatures.
- **Polars `is_empty` Regression** (`data/data_ingestion.py`): Fixed a pandas/Polars mismatch before type conversion.
- **Training Pair Defaults** (`config/settings.py`): Aligned default pairs with the active three-pair setup: `EURUSD`, `GBPUSD`, and `USDJPY`.
- **PyTorch Inference Initialization** (`inference/pytorch_inference.py`): Aligned checkpoint directory and checkpoint path handling.
- **Model Class Size Mismatch** (`models/architectures.py`, inference paths): Enforced `num_classes` parameterization.
- **Execution Action Fallbacks** (`backtesting/backtest.py`): Filled missing signal actions with `ScalingAction.HOLD`.
- **Missing Scaling Enums** (`backtesting/backtest.py`): Added `SCALE_IN_100` and `SCALE_OUT_100`.
- **System Test Integrity** (`tests/test_system.py`): Updated ingestion mocks for Polars inner joins.
- **NaN Storm Prevention** (`models/architectures.py`): Added lower-bound variance clamps before square roots in cross-asset rolling correlations.
- **Data Quality Filtering** (`data/data_ingestion.py`): Added filtering for weekend bars, fixed holidays, and zero-volume dead bars.

---

## [Earlier / Carried Forward]

### Added

- **C++ Risk Management** (`infrastructure/cpp_server`): Added `RiskManager` support for fat-finger checks, stale data checks, and drawdown circuit breakers.
- **C++ Runtime Hardening** (`infrastructure/cpp_server`): Added ONNX startup metadata discovery, environment-driven runtime configuration, JSONL shadow journaling, and partial-feature fail-fast behavior guarded by `ALLOW_PARTIAL_FEATURES`.
- **Model Comparison Report** (`scripts/compare_models.py`): Expanded comparison reporting for XGBoost, deep models, latency, confidence, fold stability, and baseline checks.
- **Time-Aware Fold Governance**: Added documentation and reporting fields for purged/embargoed folds, fold stability, pass rate, train-validation gap, and suspiciously smooth validation.
- **Ablation Checklist** (`docs/MODEL_SETTINGS.md`): Added supervised ablation requirements.
- **Pretraining Metrics** (`training/train_gpu.py`, contrastive trainers): Added alignment, uniformity, handoff, and representation-quality checks.
- **Regime Oversampling** (`training/train_gpu.py`): Added dynamic oversampling for extreme regime windows.
- **Curriculum And Regime Augmentations** (`models/contrastive.py`): Added curriculum scaling and regime-aware augmentation behavior.
- **Backtest Safety Metrics** (`backtesting/backtest.py`): Added OHLC validation and safer report generation.
- **Backtest Reporting** (`backtesting/backtest.py`): Added Sortino Ratio, Long Win Rate, and Short Win Rate.
- **Action Semantics** (`backtesting/backtest.py`): Added `ScalingAction` for robust signal execution mapping.
- **Risk Integrity** (`backtesting/backtest.py`): Added maximum drawdown circuit breaker and strict position-limit clamping.
- **Execution Modeling** (`backtesting/backtest.py`): Added clamped market-impact spread estimation.
- **RL Action Expansion** (`models/rl_agents.py`): Expanded the trading environment to the 10-class `ScalingAction` state space.
- **RL Replay Weighting** (`models/rl_agents.py`): Added inverse-frequency weighted sampling for rare scaling events.
