# Audit: Trained Models & Dataset (2026-09-25)

**Scope:** trained checkpoints (TFT, Mamba, HAELT, GNN × 7 walk-forward folds, ensemble meta, RL ensemble) and the training dataset
`data/processed/dataset_scalping_5m_EURUSD-GBPUSD-USDCAD-USDJPY_..._2008-01-01_2025-12-30.zarr` (185,201 samples × 120 × 584).

**Mode:** sections A–M are findings; fixes applied so far are listed in N (fix log).

**Bottom line:** the models show no evidence of learned signal. The "certified for deployment" result rests on an RL Sharpe calculation that leaves out trading costs.

**Also:**
- The saved ensemble has an out-of-sample R² of about 0 (E1).
- The failures come from the shared dataset (features, labels, scaling), not the architectures (F).
- Live ONNX inference crashes on models with a single-value output (M1).

**Contents**
| Section | Topic |
|---|---|
| A | Trained models |
| B | Dataset |
| C | Test suite state |
| D | First recommendations |
| E | Follow-up findings: ensemble, `close_ffd`, features, scaler, CPAR, config |
| F | Per-model inputs, settings and results |
| G | Target dataset layout (one base, several views) |
| H | Folds, hard example mining and training techniques |
| I | Evaluation, deployment, risk and operations |
| J | Techniques expected to help |
| K | Roadmap |
| L | Replacing the CPAR label |
| M | Code scan by phase: bugs and dead code |
| N | Fix log |

---

## A. Trained models

### A1. Every "best" checkpoint is a warmup epoch (P0)
- In all 28 fold runs, `best_epoch` is 0 or 1.
- Epochs 0–1 run direction-only warmup on a class-balanced subset (`direction_warmup_epochs=2`). Their validation loss (~1.36) is on a different, lower scale than the full multitask loss (~1.47).
- The selection check `improved = _sacs_score < _best_sacs_score` (`training/supervised_loop.py:2420`) compares the two directly, so a warmup epoch always wins.
- As a result, the remaining 10–38 epochs per fold were wasted.

### A2. Models collapsed to the class prior (P0)
- All four architectures reach the same validation loss in a given fold (fold 2 ≈ 0.984, fold 4 ≈ 1.277 for every model).
- `dir_acc` is 0.42–0.49 and flat across epochs.
- Validation loss never drops below its epoch-0 value.

### A3. Validation Sharpe is noise (P1)
- Per-epoch `val_sharpe` swings between −34 and +39. The annualization factor is 325 on 5-minute bars.
- Model selection, early stopping and the "Sharpe collapse" controller signals all depend on this number.

### A4. Ensemble meta-learner did not learn (P1)
- Loss went from 7.639 to 7.626 over 15 epochs (`checkpoints/ensemble/ensemble_meta_final.pt.json`).
- The latest `ensemble_meta_best.pt` was trained on only 15,000 samples.

### A5. RL Sharpe excludes trading costs (P0)
- In `models/rl_agents.py`, `_exec_cost` subtracts costs from `equity`, but `episode_pnl.append(realised_pnl)` (line 488) records gross PnL.
- Sharpe is computed from `episode_pnl`. Examples from `rl_report.json`:

| Agent | Return | Max DD | Sharpe (reported) | Costs |
|---|---|---|---|---|
| ppo seed 1337 | −122.5% | 262% | **+13.5** | 125,446 |
| ppo seed 1437 | −1.4% | 3.3% | +11.7 | 1,719 |
| ppo seed 1537 | −110.6% | 147% | +6.6 | 77,369 |
| soft-vote ensemble | +7.7% | 6.8% | +6.8 | 1,632 |

### A6. Deployment certification is invalid (P0)
- `checkpoints/ensemble/promotion_gate.json` says `CERTIFIED_READY_FOR_DEPLOYMENT`, based on 364 trades, Sharpe 6.79 and +7.7%.
- It was issued under older rules. The current `scripts/auto_optimal_roadmap.py` gate would reject it:
  - conflict rate 57% vs the 50% maximum
  - agent agreement 12% vs the 50% minimum
- The 3 individual agents all lost money.
- `trading/live_engine.py:3645` accepts this certification status.

### A7. Wasted compute (P2)
- SACS runs 5 extra full validation passes on every epoch.
- Supervised folds averaged 5–25% GPU use. The RL stage ran for about 6 hours at 2.5% GPU.

### A8. Metadata gaps (P2)
- `schema_hash: "unknown"` in every checkpoint's metadata.

---

## B. Dataset

### B1. Features are not normalized (P0)
- About 5% of last-step values have |x| > 50, and the maximum is 1e6. Examples:
  - raw price levels (`USDJPY::open/high/low/close` ≈ 109, `bb_upper/lower`)
  - `volume` ~8–11k, `noise_to_signal_60` ~8k
  - COT positions (`cot_net_comm` 176,722; `cot_net_hf` 160,137)
  - a 1,000,000 placeholder in `eco_surprise` and `eco_revision`
  - RSI, ADX and chop index on a 0–100 scale
- Raw price levels drift over time, so a model can learn the period instead of patterns.

### B2. USDJPY uses EURUSD's pip size (P0)
- `USDJPY::ret_5` reaches 171 and `ret_20` reaches 684, roughly 100× larger than they should be.
- The median JPY spread is stored as 3.8e-5 in price units, when about 0.01 is realistic. JPY costs are understated by roughly 100×.
- Validation cost uses a hard-coded `pip_size=0.0001` (from `LABELING`).

### B3. 79 of 584 features are constant (P1)
- Constant for every pair: `fb_0`–`fb_7` (FinBERT), `sentiment_raw`, `sentiment_decayed*`, `cat_geopolitical`, `eco_surprise_missing/staleness`, `session_label`, `trend_quality`, `vol_clock_pace`, `vol_clock_hot`.
- The news and sentiment pipeline is not reaching the dataset.

### B4. Label distribution shift and pair imbalance (P1)
- Overall: SELL 43.2%, HOLD 12.5%, BUY 44.4%.
- HOLD is 8–13% in most of the history but **28% in the final tenth** (the evaluation period).
- USDJPY is only about 10% of rows (18,251 of 185,201).
- Labels are clipped at ±50.

### B5. Samples overlap heavily (P1)
- `X[i+1][:-1] == X[i][1:]`: each sample is the previous one shifted by one bar.
- Labels look 30 bars ahead, so neighbouring labels overlap almost completely.
- There are far fewer independent samples than rows, which inflates any per-sample Sharpe.

### B6. Coverage and report inconsistencies (P2)
- The readiness report counts about 970k sequences across pairs. The cache holds 185k (about 19%).
- The readiness report shows HOLD at about 0% (EURUSD: 127 of 418k). The cache has 12.5%. The report and the cache don't describe the same build.
- The report flags `missing_hours` of about 108–122k per pair.
- The pairs share only 124 common timestamps under inner alignment.

### B7. Most label-correlated features measure volatility or liquidity (info)
- The top |corr| with the target is 0.14–0.16, from `amihud_illiq`, `time_cos`, `vol_ratio_6_20`, `atr_ratio_6_20`, `tar` and `atr_15m/60`.
- These track the size of the target, not its direction.

---

## C. Test suite state (before any change)
- `tests/test_all.py` fails to import `_NUMBA_IMPORT_OK` from `labeling.triple_barrier_labeling`.
- These tests already fail on the unmodified code:
  - `test_rl_market_arrays.py`: 3 tests
  - `test_ppo_greedy_inference.py`: 2 tests
  - `test_audit_remaining_fixes.py`: 4 tests
  - `test_monte_carlo.py::test_pipeline_legacy_run_schemas`
- 56 collection or setup errors in the wider run.

---

## D. Recommendations (priority order)

1. **Revoke the certificate:** delete or rename `checkpoints/ensemble/promotion_gate.json`. Make the RL Sharpe subtract costs, then re-run the gate under the current rules.
2. **Checkpoint selection:** leave the direction-warmup epochs out of best-checkpoint selection, or reset the best score when warmup ends. Then retrain.
3. **Per-pair pip size:** use 0.01 for JPY in features, spread and cost calculations.
4. **Normalization:** z-score each feature per fold using statistics from the training part only (or use rank transforms). Remove raw price levels. Replace placeholder values with NaN plus a missing indicator. Log-scale volume and COT positions.
5. **Dead features:** drop the 79 constant features, or fix the news and sentiment feed.
6. **Baseline first:** before retraining the deep models, get a simple baseline (logistic regression or XGBoost, with purge and embargo gaps between folds) to beat the class-frequency guess. If it can't, revisit the labels (30-bar lookahead, TP 1.2 / SL 0.8 ATR).
7. **Honest metrics:** compute Sharpe on non-overlapping trades after costs, with a bootstrap confidence interval, and use it for model selection and the gate.
8. **Handle distribution shift:** balance or reweight by pair and by period, and report results per pair and per period.
9. **Dataset build:** rebuild the dataset, regenerate a readiness report that matches the cache, and record a real `schema_hash`.
10. **Compute:** turn SACS off during training, or run it only on the final few candidate checkpoints. Fix the RL stage's low GPU use.
11. **Tests:** fix the failing tests so they can guard these fixes.

---

## E. Additional findings (follow-up session, 2026-09-25)

### E1. Ensemble meta-learner has zero out-of-sample skill (P0)
The saved `checkpoints/ensemble/ensemble_meta_best.pt` was loaded with its four bases (`haelt/mamba/gnn/tft_best.pt`) and evaluated on 6,000 contiguous samples at each of two points in the dataset:

| Slice | var(y) | Ensemble MSE | R² | corr(pred, y) | Sign accuracy (BUY/SELL only) |
|---|---|---|---|---|---|
| Holdout tail (last 6k) | 9.344 | 9.333 | **0.001** | 0.064 | 0.524 |
| Middle of dataset | 7.072 | 7.074 | **−0.000** | 0.041 | 0.512 |

- **Base models output almost nothing:** prediction std is 0.005–0.095 against a label std of about 2.7. Correlation with y ranges from −0.009 to 0.058.
- **Weights barely move:** mean weights are about 0.22–0.30 per model, with a per-sample std of 0.03–0.09. The ensemble is effectively an equal average of near-zero predictions.
- **Base models are highly correlated (0.7–0.84 on the holdout),** so they add little diversity.
- **Training loss looked better than it was:** it was ~5.62 against a label variance of ~7.15 in the training window. That gap came from in-sample data and the `−0.1 × entropy` term, and doesn't carry over to new data.

### E2. Ensemble training design problems (P1)
Files: `models/ensemble.py`, `scripts/train_ensemble_meta.py`, `training/post_train.py`.
- **The output can't be rescaled.** It is `softmax(weights) · base_preds`, a convex combination with no bias or scale term. The ensemble can never be larger than its largest base prediction, so it can't correct under-scaled bases (see E1).
- **The ensemble mixes units.** It averages multitask models' `return_hat` with CE models' `logit_buy − logit_sell` (`_base_pred_to_batch_vector`).
- **The "best" checkpoint is picked on training loss.** `train_meta_learner` has no validation loader. The `--meta-validation-frac` slice is reserved but never evaluated.
- **The objective is fixed MSE on a heavy-tailed CPAR reward (±17).** The loss is dominated by tail samples. It also ignores direction and costs.
- **The entropy bonus `−diversity_weight × H(w)` pushes weights toward uniform,** which works against the meta-learner's purpose.
- **Predictions are in-sample.** The script refuses to run without `--allow-in-sample-meta` because no out-of-fold (OOF) predictions exist, so the deployed meta was trained on in-sample base predictions.
- **Only 15,000 samples were used** in the latest run.
- **Artifacts don't match:**
  - `ensemble_meta_best.onnx` (09-21) is older than `ensemble_meta_best.pt` (09-23), so ONNX and PyTorch serve different models.
  - `ensemble_manifest.json` (09-20) points to `*_fold0_config.json`, while `fold_selection.json` chose fold 2.
- **Loading doesn't fully match the checkpoint.** `tft_best.pt` is missing `vsn.norm` weights (2 keys), and `load_base_model` falls back to `strict=False` without any warning.
- **Fold selection is biased.** `fold_selection.json` picks the single best of 7 folds by val_loss, and its `secondary_value` just duplicates `metric_value`.

### E3. Root cause of several dead or degenerate features: `close_ffd` misuse (P0)
- `close_ffd` (FFD, d=0.4, `data/data_ingestion.py:1015`) is a scaled **price level** of about 0.04 × price, not a return series.
- It is used as if it were a return series:
  - `ret_{5,20,60} = close_ffd.rolling_sum(w)` (`features/engineering/microstructure.py:239`) → returns ≈ w × price level. In the cache they correlate |r| > 0.995 with `open`.
  - `rsi_14` treats `close_ffd` as the per-bar change (`microstructure.py:222`) → about 100 constantly on a synthetic random walk.
- Fix: compute returns and RSI from `log(close).diff()`.

### E4. Other feature issues (P1)
- **279 feature pairs have |corr| > 0.995:** OHLC, bid/ask close, `bb_upper/lower` and `ret_*` are near-duplicate price levels. `bb_mid` was dropped as collinear, but `bb_upper/lower` were kept (`config/feature_mask.py`).
- **Lookahead:** `features/multipair.py:125` (`asia_london_gap`) gives every row of a day that day's *last* Asia close, including rows earlier in the Asia session.
- **Lookahead:** `features/multipair.py:250` uses `ffill().bfill()`, which back-fills from the future.
- **Synthetic yields replace real ones without failing:** `features/macro_features.py:235` swaps in seeded synthetic yields when FRED fails. Carry and spread features are also the least stationary in the cache (~1.5σ mean shift between halves).
- **`noise_to_signal_60` blows up** because it divides by a mean return close to zero.
- **`hurst_exponent` is misnamed:** it is `0.5 + 0.25 × lag-1 autocorrelation`, not a Hurst estimate.

### E5. Scaler and train/serve issues (P1)
- **Scaler fit leaks validation data:** `_fit_scaler_from_cache` fits on the whole cache, including validation and holdout rows (`training/dataset_builder.py:558`).
- **`_merge_scalers` ignores its robust branch.** sklearn's `RobustScaler` has no `n_samples_seen_`, so the function falls back to returning the first worker's scaler.
- **Wrong scaler type on the serial path:** it hard-codes `StandardScaler` (`dataset_builder.py:2904`) while `data.scaler_type: robust`.
- **Inference tiles scalers across pairs:** a 146-column scaler can be applied to 584 columns, putting EURUSD statistics on USDJPY (`inference/pytorch_inference.py:164`).
- **Training and live inputs differ:** the feature-stability mask and the curriculum group mask zero features during training and validation, but live inference doesn't apply them.

### E6. CPAR label issues (P1)
- **Labels ignore trading costs:** in `(cpar_l − cpar_s)/2` the bid/ask spread cancels, so the direction label is the sign of the mid-price move (`labeling/cpar_labeling.py:98`).
- **Heavy tails:** targets are divided by `atr_6` (30-minute ATR), which gives ±17 tails.
- **Settings are ignored:** `penalty=0.5` is hard-coded, and `profit_atr_mult/stop_atr_mult` are unused even though they appear in the cache tag.
- **Label exits don't match the backtest:** labels exit only at the 30-bar barrier, while the backtest uses 12/18-pip or 1.2/1.8-ATR stops.
- **Misleading function name:** `compute_triple_barrier_labels` returns CPAR. Verify that `label_method: triple_barrier` (tabular) really gets triple-barrier labels.

### E7. Config / training-control issues (P1)
- **Duplicate key:** `direction_training` appears twice in `config/run.yaml` (lines 327 and 476). PyYAML keeps the last one, which disables the probe and sets `min_true_class_share: 0.02`.
- **Settings that do nothing:**
  - `training.lr: 3e-5` is overridden by the per-model profile (runs used 2.7e-4). Model `dropout` is overridden the same way.
  - `sharpe_weight` and `direction_weight` apply only to other loss types, and `loss: huber` is set.
  - `early_stop_metric: cost_sharpe` isn't used by DynStop.
- **Early stopping never fires** (`training/supervised_loop.py:2435`):
  - it doesn't count until after warmup
  - it is blocked until the SWA start (epoch 30 of 40)
  - the first plateau halves the LR and resets patience
  - the SI/EWC correction makes the composite improve over time on its own
- **Too many overlapping training controls:**
  - focal loss (upweights hard samples) runs alongside self-paced learning (downweights them)
  - class imbalance is handled three separate ways
  - label smoothing is set to both 0.1 and 0.05
  - SI is forced on for a fresh run (line 583)
  - mixup is applied to time series
- **Too little data for the model size:** `n_ticks: 20M` yields 185k samples over 2008–2025, about 14% of 5-minute bars. That is roughly 6k independent labels for 584 × 120 inputs. The checkpoint folder is named `2015_2025` while the data starts in 2008.

### E8. Additional recommendations
1. **Retire the current ensemble:** don't use `ensemble_meta_best.*` or the RL ensemble built on it.
2. **Rebuild stacking on real OOF predictions:**
   - take each base model's prediction from the fold where that row was in validation
   - fit a **linear or logistic stacker** (with intercept and scale) on the OOF predictions
   - select it on a held-out validation slice
   - report out-of-sample R², IC and after-cost Sharpe across all folds (averaged, not best-of)
3. **Standardize the base outputs** to the same units, e.g. calibrated P(buy) − P(sell).
4. **Keep artifacts consistent:** export ONNX from the same `.pt` in the same run, and record fold/config provenance in the manifest.
5. **Only train an ensemble once bases have signal:** a base model should show IC > 0 consistently across folds first. Averaging models that each have zero skill still gives zero skill.

---

## F. Per-model inputs, settings and results

### F1. What all the deep models share (HAELT, Mamba, GNN, TFT)
All four were trained on the same data and inputs, with the same target and the same loss. Only the network in the middle differs. That is why they all reach the same validation loss on each fold.

| Item | Value | Problem |
|---|---|---|
| Dataset | One cache: 4 pairs, 5-minute bars, 2008–2025, 185,201 samples | Only about 14% of the available bars (capped by `n_ticks: 20M`) |
| Input | 120 bars × 584 features (146 per pair × 4 pairs) | 79 are constant. 279 pairs have \|corr\| > 0.995, mostly raw price levels (B1, B3, E3, E4) |
| Input wrapper | `MultiPair` (4 pairs × 146 features + a 16-dim pair embedding) + `MultiTask` head | — |
| Scaling | One `RobustScaler` over the whole cache | Includes validation rows; USDJPY uses EURUSD's pip size (B2, E5) |
| Label | CPAR reward (30 bars ahead) + sell/hold/buy class | Ignores costs; ±17 outliers; hold is 12.5% overall but 28% in the final tenth (B4, E6) |
| Loss | Multitask: direction CE + return Huber + confidence BCE + quantile | Focal loss, class weights, label smoothing and mixup all stacked together (E7) |
| Validation | 7 folds, each trained on all earlier data, with gaps between train and validation | Best epoch is 0–1 every time; one best fold is picked instead of averaging all 7 (A1, E2) |

### F2. Where the models differ (from the fold-2 checkpoint configs)

| Model | Architecture | Params | Bars actually used | Intended bars | Dropout | Validation loss (folds 0–6) |
|---|---|---|---|---|---|---|
| HAELT | LSTM + Transformer hybrid, d_model 256, 3 layers | 1.32M | 120 | 120 | 0.175 | 0.98–1.54 |
| Mamba | State-space model, d_model 128, 4 layers | 0.75M | **120** | 60 | 0.156 | 0.98–1.54 |
| GNN | Graph across pairs, hidden 64, 3 layers | 0.52M | **120** | 80 | 0.163 | 0.99–1.54 |
| TFT | Temporal Fusion Transformer, hidden 128, 2 layers | 0.88M | 120 | 120 | 0.153 | 0.98–1.54 |

- **Per-model settings aren't applied (P1).** Mamba and GNN were meant to use 60 and 80 bars (`config/models.py`, `config/models/*.yaml`), but all four trained on 120. Dropout is 0.15–0.18 everywhere, neither the 0.35 in `run.yaml` nor the per-model values. Settings are overridden somewhere between the YAML files, the model profiles and the command line.
- **The models are large for the data (P1).** They have 0.5–1.3M parameters, but because samples overlap heavily there are only about 6k independent labels. Each model can memorise the training data in one epoch.
- **Changing the architecture doesn't help.** With identical inputs and labels, all four converge to the same prediction: the class frequencies. Fixing the shared dataset matters more than any model setting.

### F3. Downstream models

| Model | Uses | Notes |
|---|---|---|
| Ensemble meta-learner | The 4 base models' outputs plus a small attention summary of the 584 features; 15k samples | Out-of-sample R² ≈ 0; trained on in-sample predictions (E1, E2) |
| RL (3 PPO agents) | The frozen supervised network's internal features as observations (`encoder_obs: true`) + CPAR reward; `run_rl.yaml` sets 60 bars and 10M ticks | Builds on a network with no signal. Its Sharpe ignores costs. Two of three agents lost more than 100% (A5, A6) |
| CatBoost / XGBoost | `run_tabular.yaml`: 30 bars, triple-barrier labels, 5M ticks | Never trained. The label function they call may return CPAR, not triple-barrier (E6) |
| BYOL pretraining | Same cache and features, 18 epochs | Learns from the same broken features, so it can't add signal |

### F4. Recommendation
Fix the shared dataset first (features, labels, scaling). Then apply the per-model bar lengths properly, shrink the models to d_model 64–128, and use a gradient-boosted tree baseline as the first check that there is any signal.

---

## G. Target dataset layout: one base, several views

Don't build a separate dataset per model:
- **Stacking needs aligned rows.** The ensemble learns from each model's predictions on the same timestamps.
- **Fair comparison needs one shared dataset.** Sharing one dataset is what exposed the prior collapse in A2.
- **Every extra build copies every bug** and multiplies the build time.

| Layer | Shared or per-model | What goes in it |
|---|---|---|
| Base feature table | Shared | Clean features per bar and pair, timestamp, fold ID, per-fold scaler statistics |
| Labels | Shared | Cost-aware triple-barrier / CPAR, class labels, bars ahead; keyed by timestamp |
| Folds and holdout | Shared | The same folds, gaps and final holdout for every model |
| Model view | Per family | Cut at load time: bar length, feature subset, sampling rate |

| Family | Bars | Features | Notes |
|---|---|---|---|
| HAELT / TFT | 120 | Full clean set | — |
| Mamba | 60 | Full clean set | Cut the last 60 bars at load time; no separate cache |
| GNN | 80 | Per-pair features as graph nodes | Pairs aligned on the same timestamps |
| XGBoost / CatBoost | 1 bar + summary statistics | Subset | Small table built from the base |
| RL | 60 | The supervised network's internal features + execution data | Same timeline and holdout |

**Two changes are needed:**
1. **Cut views at load time** instead of putting the bar length in the cache name. The current naming led to separate 80/90/120-bar caches and to the lost Mamba/GNN bar lengths (F2).
2. **Store a fold ID on every row.**

---

## H. Folds, hard example mining and other training techniques

### H1. Folds
| Current | Recommendation |
|---|---|
| 7 folds, each trained on all earlier data | Keep |
| Dynamic gap ≥ bars + lookahead + delay | Keep; recalculate if the label horizon changes |
| Separate purge of 120 bars | Optional (features only look backward) |
| Pick the single best fold | **Average all folds; require every fold to pass** |
| Deploy one fold's model (fold 2) | Retrain one final model on all data up to the holdout, with the epoch count taken from the folds |
| Splits worked out per script | Store a shared fold ID on every row |
| 10% promotion holdout | Keep; evaluate on it once, at the end |

### H2. Hard example mining: turn off
`training/hard_example_miner.py` oversamples the 15% of training samples with the highest loss (up to 2×).
- At about 50/50 accuracy, those samples are mostly mislabelled or random.
- It conflicts with focal loss (which also upweights hard samples) and self-paced learning (which pushes them down).
- **Revisit only after a model shows stable skill.** Then cap the extra weight at about 1.3× and ignore samples with small or ambiguous labels.

### H3. Other techniques
| Technique | Verdict | Reason |
|---|---|---|
| Difficulty curriculum | Off | Built on noisy difficulty scores; changes the training data between epochs |
| Feature-group curriculum | Off | Training/live mismatch (E5) |
| Feature-stability mask | Off | Training/live mismatch; different random mask each epoch |
| Self-paced learning | Off | Conflicts with focal loss and hard mining |
| Focal loss-weighting scheme + multitask focal | Off | Double-counted; amplifies noise |
| Class weights (0.85–1.15) | Keep | Mild |
| Label smoothing | Keep 0.05 only | Remove the conflicting 0.1 |
| Entropy / class-balance terms | Off | Class imbalance already handled by class weights |
| Mixup | Off | Blending time windows makes unrealistic sequences |
| Volatility sampler | Off for now | Changes the class mix between training and validation |
| EWC / SI | Off | Meant for learning several tasks in sequence |
| SWA | Off for now | Blocks early stopping until epoch 30 |
| SACS | Off | 5 extra validation passes every epoch; drives the warmup-epoch selection (A1) |
| Direction-only warmup | Off, or exclude from best-checkpoint selection | Loss on a different scale (A1) |
| Dynamic early stop | Replace | Plain patience 5 on after-cost Sharpe, counted from epoch 0 |
| Warmup + cosine LR, grad clip | Keep | — |
| Weight decay | Keep; raise to 0.05 | More regularisation for a small dataset |
| BYOL pretraining | Off until features are fixed; then compare with and without | Learns from broken features |
| Temperature calibration | Keep, fitted per fold | — |
| Overconfidence penalty | Off | Calibration covers it |
| Adversarial / distillation / diversity fine-tuning | Leave off | — |
| MC dropout | Keep for sizing later | Costly |

**Minimal starting setup:** cross-entropy + Huber, light class weights, label smoothing 0.05, dropout, weight decay 0.05, warmup + cosine LR, simple early stopping, calibration. Add techniques back one at a time, keeping each only if it improves the average after-cost result across all folds.

---

## I. Evaluation, deployment, risk and operations

### I1. Evaluation and backtesting (P0)
- **One after-cost measure everywhere:** trades that don't overlap, after spread, commission and slippage, with a bootstrap confidence interval. Use it for early stopping, model selection, Optuna and the gate.
- **Make the backtest exits match the labels:** labels exit at 30 bars, the backtest uses 12/18 pips or 1.2/1.8 ATR. Pick one rule.
- **Realistic costs:** spread and slippage per pair and per session from bid/ask data; also test at double the cost.
- **Final test period:** 6–12 months, untouched, evaluated once.
- **Baseline strategies:** always flat, buy-and-hold, momentum, moving-average crossover.
- **Confidence threshold:** `min_confidence: 0.45` is near random for three classes. Choose it on validation data after calibration (about 0.55–0.65).

### I2. Deployment gate and live trading (P0)
- **Strict gate, in code:**
  - at least 1,000 trades
  - every fold profitable after costs
  - max drawdown under 15%
  - no ensemble member losing money
  - lower bound of the Sharpe confidence interval above 0
- **Certifications expire:** store the rules version and data hash with each certification; `trading/live_engine.py` must reject certifications from older rules.
- **Staged rollout:** shadow → at least 4 weeks of paper trading → small real positions → scale up while live results stay within the backtest's range.
- **Same feature code in training and live:** load the scaler, feature list and mask from the checkpoint; fail on mismatch; remove the scaler-tiling fallback; add a daily offline recompute check.

### I3. Risk and position sizing (unverified in code)
- Fixed small positions or volatility targeting, not Kelly, until win-rate estimates are trustworthy.
- Limit total USD exposure (all four pairs involve USD).
- Test that the hard limits (daily loss, drawdown halt, VaR) actually trigger in paper trading.
- Turn off regime scaling and MC-dropout sizing until validated.

### I4. Hyperparameter tuning (Optuna)
- **Pause** until the XGBoost signal check (K step 4) passes.
- **When resuming:** score on the average after-cost measure across all folds, search about 5 settings, and use a tuning window separate from the holdout.

### I5. Monitoring and retraining
- Set `retraining.promote_on_complete: false`; every retrain goes through the gate and paper trading.
- **Monitor model health, not only features:** live prediction correlation with outcomes, hit rate and after-cost P&L against the backtest's range.
- Log every prediction, its features and the outcome, so live results can be re-scored offline.

### I6. Data sources
- Fail the build on missing news, FRED or cross-asset data; no synthetic or zero fallbacks.
- Fix the news pipeline or remove its 79 dead columns.
- Coverage report per pair and year, checked against the cache.

### I7. Config and code health
- **One resolved config per run,** saved with the checkpoint. Fail on unknown or duplicate YAML keys.
- **Archive unused paths:** extra run profiles, the second and third training and RL frameworks, the duplicate C++ servers, one-off scripts.
- **Fix failing tests (C) and add tests for:**
  - returns computed from price, not `close_ffd`
  - no constant features and no values above 50 after scaling
  - JPY pip size
  - leak checks
  - Sharpe after costs
  - the ensemble's ONNX and PyTorch files matching
- **Reproducibility:** record the data hash, git commit, resolved config and seed with every checkpoint.

### I8. Research direction
- Try longer horizons (15-minute to 1-day), where moves are larger relative to costs.
- Test on 1 pair with fewer than 60 features first.
- Add complexity only if it improves the average after-cost result across folds.

---

## J. Techniques expected to help

### J1. Tier 1: do first
| Technique | Why | How |
|---|---|---|
| Meta-labeling (López de Prado) | A second model only decides whether to take a simple rule's trade; that yes/no filter is much easier to learn than direction | `labeling/triple_barrier_meta.py`; primary = momentum or breakout rule; XGBoost as the filter |
| Triple-barrier labels with costs and a no-trade zone | Labels = "profitable after costs", matching how you trade | Barriers = backtest stop-loss/take-profit incl. spread; hold when the move is under the cost |
| Gradient-boosted trees first | Handle noisy, mixed-scale inputs; train in minutes | `train_xgboost.py`; 1 bar plus lagged summary statistics |
| Stationary, scale-free features | Models can't learn from price levels | Log returns, return ÷ ATR, rolling z-scores, distance to VWAP ÷ ATR, spread ÷ ATR, session features |
| Train-only scaling per fold | Removes the scaler leak | Fit the scaler on each fold's training rows only |
| Weighting by label uniqueness | Overlapping 30-bar labels make neighbouring rows near-copies | Weight = 1 ÷ number of overlapping label windows, passed as `sample_weight` |

### J2. Tier 2: once Tier 1 works
| Technique | Why | How |
|---|---|---|
| Combinatorial purged CV (CPCV) | Many test paths instead of 7, so you see a distribution of Sharpe ratios | Extend `cv_splits.py` |
| Deflated Sharpe / probability of backtest overfitting | Corrects for having tried many models and settings | Compute over all trials; use in the gate |
| Volatility-targeted sizing | Keeps risk steady across calm and wild periods | Position ∝ target volatility ÷ realised volatility, capped |
| Calibration + threshold chosen on validation data | Trading only high-confidence signals raises profit per trade above costs | Temperature or isotonic scaling per fold; threshold that maximises after-cost profit |
| Session and news filters | Much of the cost comes in the Asian session and around news | Wire `features/no_trade_zones.py` into both labels and backtest |
| Longer horizons | Moves are larger relative to costs | Build 15-minute or 1-hour views from the same base |
| Feature selection by stability across folds | Fewer features, less overfitting | Permutation importance or SHAP per fold; keep features ranked high in, say, at least 5 of 7 folds |

### J3. Tier 3: after a working baseline
| Technique | How |
|---|---|
| Small deep model | TFT or LSTM, d_model 64, 60 bars, cross-entropy, dropout 0.3, weight decay 0.05 |
| Stacking | Logistic regression on out-of-fold probabilities |
| Seed averaging | Average 3–5 seeds of the same model |
| Regime features | Volatility percentile, ADX, range width as inputs (not regime-switching models) |
| Cross-pair features | USD index from the 4 pairs, each pair's return minus the basket, relative strength |
| Self-supervised pretraining | Only on fixed features; keep only if the with-vs-without comparison shows a gain |

### J4. Skip for now
- **RL for entry decisions.** It needs proven signal and a realistic simulator first; later, RL is more useful for execution (order timing).
- **Hard mining, mixup, EWC, SI, SACS, adversarial training, curriculum.** These amplify noise on data like this.
- **Large transformers, GNNs and Mamba.** Not enough independent data yet.
- **More news sources** until the current feed actually produces varying values.

**Realistic target:** after-cost Sharpe of about 0.5–1.5 on the final test period, with every fold positive. Double-digit Sharpe ratios signal a bug.

---

## K. Roadmap

0. **Fix the runtime crashes (M1),** starting with the missing `import os` in `inference/onnx_inference.py`, and wire in `trading/preflight_check.py` (M4).
1. **Stop the damage:** revoke `promotion_gate.json`, keep RL in shadow mode, fix the RL Sharpe to include costs, tighten the gate (I2).
2. **Fix the dataset:**
   - returns and RSI from `log(close)`
   - remove raw price levels
   - drop constant features
   - clean the 1e6 placeholders and extreme values
   - JPY pip size
   - remove the leaks
   - per-fold scaler
   - cost-aware labels
   - remove the `n_ticks` cap
   - build-time checks
   - (B, E3–E6)
3. **One after-cost evaluation;** backtest exits match the labels (I1).
4. **Signal check:** XGBoost meta-labeler with uniqueness weights on shared folds (J1). **Stop and rethink the labels or horizon if it fails.**
5. **Minimal deep training profile** (H3); fix the per-model settings merge (F2); average across folds.
6. **CPCV, deflated Sharpe, volatility-targeted sizing, paper trading** (J2, I2).
7. **Only then:** stacking on out-of-fold predictions, RL for execution, Optuna.

---

## L. Replacing the CPAR label

### L1. What CPAR does now
For each bar, `labeling/cpar_labeling.py` looks 30 bars ahead and scores both a long and a short trade:
- **Long score** = (exit price − entry price) ÷ ATR − 0.5 × (worst drawdown during the trade ÷ ATR)
- **Short score** = the same, mirrored
- **Target** = (long score − short score) ÷ 2
- **Class** = the sign of the target

It is a volatility-scaled forward return with a penalty for drawdown during the trade. Its weaknesses (E6):
- the spread cancels out, so trading costs don't count
- it always exits at bar 30, with no stop-loss or take-profit
- dividing by a 30-minute ATR gives ±17 outliers
- the hold class comes from small targets, not from whether a trade would be worth taking

### L2. Replacement options
| Replacement | What the label says | Best for | Verdict |
|---|---|---|---|
| **Cost-aware triple-barrier** | +1 if take-profit is hit first, −1 if stop-loss is hit first, 0 if neither by the time limit; entry at ask/bid, spread and commission included | Classification; matches how you trade | **Main label** |
| **Meta-label** | 1 if a simple rule's trade would have been profitable after costs, else 0 | Filtering trades; an easier yes/no problem | **Add second** (J1) |
| **Cost-adjusted forward return ÷ volatility** | (return over h bars − cost) ÷ 60-bar volatility, capped at ±3 | Regression target, stacking, IC measurement | Return target for the multitask head |
| **Trend-scanning** (López de Prado) | Sign of the strongest trend over several horizons (t-statistic of a slope fit) | Longer, trend-following horizons | Try later, with 1-hour bars |
| **Fixed-horizon sign** | Sign of the 30-bar return | — | Avoid: noisy, ignores costs |

### L3. Recommended setup
- **Classification label:** triple-barrier with stop-loss 1.0 × ATR60 and take-profit 1.5 × ATR60 (ATR60 with a minimum floor), a 30-bar time limit, and entry at ask for longs / bid for shorts. Label 0 when the gain after costs is under 1× the spread.
- **Return target:** cost-adjusted forward return ÷ 60-bar volatility, capped at ±3.
- **Row weights:** 1 ÷ number of overlapping label windows.
- **Consistency:** the backtest uses exactly the same stop-loss, take-profit and time limit.

### L4. What already exists vs. what's missing
**Already in the repo:**
- `labeling/triple_barrier_labeling.py`
- `labeling/triple_barrier_meta.py`
- `label_method: triple_barrier` in `config/run_tabular.yaml`

**Missing:**
1. Costs in the entry and exit prices.
2. ATR60 with a minimum floor.
3. Making sure `label_method: triple_barrier` really gets triple-barrier labels. The CPAR function is also called `compute_triple_barrier_labels`, so it may be returning CPAR.
4. Changing the default `training.label_method` from `cpar`. This changes the cache name, so the dataset must be rebuilt.

---

## M. Code scan by phase: bugs and dead code

**Method:**
- `ruff` bug rules (undefined names, redefinitions, loop closures, mutable defaults)
- `vulture` for unused code
- an import-reachability graph from every entry point (training, live engine, API, scripts, tests)
- a count of swallowed exceptions
- manual checks of every undefined name

### M1. Confirmed runtime crashes (undefined names)
| Phase | Location | Bug | Impact |
|---|---|---|---|
| **Inference (live)** | `inference/onnx_inference.py:1124` | `os` never imported | **Any ONNX model with a single-value output (CPAR/regression head) crashes with a NameError on every prediction** |
| Inference | `inference/pytorch_inference.py:50` | `os` never imported, but the error is caught by `except Exception` | `PREDICTION_THRESHOLD` is silently ignored; the threshold is always 0.35 |
| Data | `data/feature_cache.py:99` | `fe` is never defined | The feature-cache build always crashes (`data.use_feature_cache` is off, so the feature is effectively dead) |
| Data | `data/sidecar_registry.py:110` | `datetime` and `UTC` not imported | `validate_sidecar_index()` always crashes |
| Monitoring | `monitoring/sidecar.py:406` | `datetime` not imported | The sidecar crashes when no `run_name` is passed |
| Features | `features/finbert_sentiment.py:460` | `_log` is never defined | When the FinBERT warmup fails, the error handler itself crashes; this may relate to the dead sentiment features (B3) |
| Training | `training/scale_model.py:280-281` | `_match_target_shape` was removed | Distillation with a regression head always crashes |
| Training | `training/train_xgboost.py:686` | `val_diраcc` contains **Cyrillic** letters | XGBoost crashes at the end when W&B is enabled |
| Training | `training/loop_losses.py:50` | `Callable` not imported | Type annotation only; harmless because of `from __future__ import annotations` |

### M2. Logic bugs by phase (from sections B and E, collected here)
| Phase | Bug |
|---|---|
| Data ingestion | `close_ffd` is a price level used as a return series (E3); synthetic-yield fallback (E4) |
| Features | `ret_*` and `rsi_14` degenerate (E3); `asia_london_gap` lookahead; `bfill` lookahead; `noise_to_signal_60` blows up; `hurst_exponent` misnamed; 79 constant features; USDJPY uses EURUSD's pip size (B1–B3) |
| Labeling | CPAR spread cancels; hard-coded penalty; take-profit/stop-loss ignored; function misnamed (E6) |
| Dataset build | Scaler fitted on the whole cache; `_merge_scalers` ignores its robust branch; serial path hard-codes `StandardScaler`; `is_classification` parameter unused (`dataset_builder.py:129`); 4 names defined twice (`_x_path`, `_y_path`, `_zarr_open_group`, `_TRAIN_LOGGER`, lines 106–120) |
| Training | Warmup epochs picked as best (A1); DynStop never fires; `early_stop_metric` ignored; per-model bar lengths and dropout lost (F2); `build_model` imported twice in `train_gpu.py:383`; duplicate `direction_training` key (E7) |
| Models / ensemble | Output can't be rescaled; units mixed; best checkpoint picked on training loss; stale ONNX; checkpoints loaded with `strict=False` without a warning (E2) |
| RL | Sharpe excludes costs (A5); `y_labels` parameter unused (`rl_runner.py:68`) |
| Evaluation / gate | Old certification accepted by the live engine (A6); fold selection copies `metric_value` into `secondary_value` (E2) |
| Inference / live | ONNX crash (M1); scaler tiling across pairs; training/live mask mismatch (E5) |
| Features / guards | `lookahead_guard.py:146` ignores its `max_nan_tail_ratio` argument, so that check is never enforced; `no_trade_zones.py:330` ignores `trend_quantile` |
| Drift | `model_drift.py:453` ignores `champion_equity`; `data_drift.py:201` ignores `max_bucket_rows` |

### M3. Silently swallowed exceptions
`except ...:` followed only by `pass`, `continue` or `return`:

| Phase | Count |
|---|---|
| training | **81** |
| trading | **39** |
| monitoring | 26 |
| data | 24 |
| features | 9 |
| inference | 5 |
| backtesting / execution | 4 each |
| models | 3 |
| risk | 1 |

Many of the silent failures above (synthetic fallbacks, the ignored threshold, dead features) come from this pattern. In `training/`, `trading/` and `inference/`, replace these with logged warnings, or fail hard on anything that affects features, labels, scaling or orders.

### M4. Dead code
**Library modules never imported from any entry point, and with no `__main__`** (safe to delete or archive):
- `contracts/validation/` (drift, gates, reporter; about 1,100 lines)
- `execution/order_manager.py`
- `features/audio_sentiment.py`
- `features/incremental.py`
- `models/ensemble_regime.py`
- `monitoring/attention_logger.py`
- `training/model_cards.py`
- `training/train_utils.py`
- **`trading/preflight_check.py`**: the live engine never runs its preflight checks. Wire it in rather than deleting it.

**Standalone tools** (they have `__main__` but nothing references them):
- `training/health_check.py` (1,051 lines)
- `training/smoke_test.py`
- `training/train_min.py`
- `training/continuous_learner.py`
- `training/feature_selection.py`
- `training/scale_model.py` (which also crashes, M1)
- `monitoring/visualize_performance.py`
- `trading/questdb_logger.py`
- `infrastructure/{deployment,news_pipeline,timescale_kafka}.py`

Keep only the ones you actually run.

**Imported only by tests** (31 modules), including:
- `feature_store/*`
- `lineage/*`
- `drift/*`
- `evaluation/metrics.py`
- `monitoring/{alerting,drift_detection}.py`
- `retraining/{orchestrator,pipeline}.py`
- `risk/{fx_greeks,portfolio_monitor}.py`
- `execution/realism.py`
- `backtesting/gpu_backtester.py`
- `validation/{promotion_audit,model_diagnostics,rolling_retraining}.py`

The config enables some of these (`feature_store.enabled`, `drift_detection.enabled`, `retraining.enabled`), but production code never calls them. Either wire them in or remove them and their config keys.

**Duplicate implementations** (a source of results that don't match):
- `compute_asia_london_gap` exists in both `features/multipair.py` and `features/advanced_features.py`.
- Sharpe is computed 5 ways: `training/sharpe_annualization.py`, `train_gpu._sharpe_ann_factor`, `train_xgboost.compute_sharpe`, `train_catboost.compute_sharpe`, `evaluation/metrics.sharpe_ratio`, plus the RL `summary()`.
- Monte Carlo: `evaluation/monte_carlo.py` and `monitoring/monte_carlo.py`.
- Drift: `drift/`, `monitoring/drift_detection.py`, `monitoring/drift_gate.py`, `contracts/validation/drift.py`.
- Execution: `execution/` and `risk/execution.py`.

**Clutter:**
- `_scratch/` (36 files) and `scratch/` (7 files) are local only (not in git); delete them.
- Tracked by accident: root-level `test_gpu.py`, `test_polars.py`, `_overlapped.pyd.bak`, `out_train.txt`, `ruff_f821.json` and `catboost_info/`.
- Small cleanups: 24 `raise` statements inside `except` blocks without `from` (B904), 6 unused variables, one bare `except` (`scripts/auto_optimal_roadmap.py:286`) and one mutable default argument (`scripts/run_paper_trading_demo.py:63`).

### M5. Fix order
1. **M1 crashes:** add the missing imports, fix the Cyrillic identifier, define `fe`, and restore `_match_target_shape`. The ONNX one first, because it breaks live inference.
2. **Wire `trading/preflight_check.py` into the live engine.**
3. **Enforce the ignored arguments:** `max_nan_tail_ratio` in the lookahead guard, and `trend_quantile`.
4. **Replace the silent `except` blocks** in data, features, dataset build and inference with hard failures.
5. **Merge the duplicate implementations:** one Sharpe function (the after-cost measure from I1), one Monte Carlo, one drift module, one `asia_london_gap`.
6. **Delete dead modules and scratch directories;** decide whether to wire in or remove the test-only feature-store, lineage and retraining packages.
7. **Enable `ruff` checks F821/F811/B023/E722 in CI** so new undefined names fail the build.

---

## N. Fix log

### Day 1: crashes and safety (done 2026-09-25)
| Commit | Fix | Audit refs |
|---|---|---|
| `090700e` | `import os` in `inference/onnx_inference.py` and `pytorch_inference.py`. Scalar-head ONNX predictions no longer crash, and `PREDICTION_THRESHOLD` is honoured again. | M1 |
| `6dc9ff2` | Remaining undefined names: Cyrillic `val_diраcc`, `_log`, `datetime`/`UTC`, `_match_target_shape`, `fe`, `Callable` | M1 |
| `b55fcec` | **Gate v2.** New `validation/gate_policy.py` holds the gate rules and a `gate_version`.<br>• The roadmap certifier now requires at least 1,000 trades, drawdown ≤ 15%, and every agent profitable; any recorded reason blocks certification (fold-variance reasons used to be ignored).<br>• The live engine accepts only current-version certifications.<br>• **Both existing 2026-09-21 certifications are now refused.** | A6, I2 |
| `6d54d64` | RL `episode_pnl` records each bar's P&L after costs, so the Sharpe includes costs (previously gross). | A5 |
| `e0b705e` | Merged the duplicate `direction_training` block in `run.yaml` (no behaviour change). New test fails on duplicate keys in any config file. | E7 |
| `ddcf4f3` | Test-suite guard: `ruff` F821/F823/B023/E722 over the library packages; `ruff` added to `requirements-dev.txt` | M5 |

**Notes:**
- The RL and ensemble checkpoints under `checkpoints/` are git-ignored and weren't changed. They can't be deployed because their certifications fail gate v2. Re-run `scripts/auto_optimal_roadmap.py` to re-certify under the new rules; expect a rejection until the dataset is fixed.
- `tests/test_ppo_greedy_inference.py::test_rl_inference_agent_passes_greedy_true` still fails. It was already failing before these changes (C).

### Day 2: training, dataset, ensemble (branch `claude/audit-fixes`)
| Area | Fix | Audit refs |
|---|---|---|
| Checkpoint selection | Direction-warmup epochs are only a placeholder best. The first full epoch resets the best score, so a warmup epoch can no longer win. | A1 |
| Early stopping | Removed the SWA guard that held runs open until epoch 30/40, and the SI/EWC "correction" that made the composite improve by itself. | E7 |
| SACS | Off by default (`training.sacs_enabled: false`). It cost 5 extra validation passes per epoch. | A7 |
| Features | `ret_{5,20,60}`, `rsi_14`, `macd*` and `vwma_ret` now use log returns (bps) or price-relative values instead of `close_ffd`. They are scale-free, so JPY is correct. `noise_to_signal_60` is bounded. | E3, B2, E4 |
| Features | `eco_surprise` / `eco_revision` are unit-free, in [−1, 1]. The ±1e6 placeholder values are gone. | B1 |
| Lookahead | `asia_london_gap` Asia rows no longer see the same day's Asia close. Multi-pair features no longer `bfill`. | E4 |
| Macro | FRED failure gives missing values (→ constant 0, which gets dropped), not seeded synthetic yields. `FOREX_ALLOW_SYNTHETIC_YIELDS=1` restores the old behaviour. | E4 |
| Spread / costs | Cached `spread` uses each pair's pip size (JPY was 100× too small). | B2 |
| Labels | CPAR direction label is BUY/SELL only when that side's CPAR (net of spread) is positive, otherwise HOLD. The MAE penalty reads `cpar_mae_penalty`. | E6 |
| Cache | New `DATASET_BUILD_VERSION` (`_ba0925`) in the cache tag, so these code changes force a rebuild. | B6 |
| Scaling | The train-only scaler now fails loudly instead of falling back to the leaky cache-wide scaler, and is fit on last timesteps (memory). `_merge_scalers` merges RobustScalers. The serial path honours `scaler_type`. Scaled inputs are clipped to ±10 in training and inference. | E5, B1 |
| Price levels | Raw price columns (OHLC, bid/ask, BB bands, VWAP) are neutralised in the train scaler (they transform to ≈0). The 146-column schema is unchanged, and inference gets the same transform through the checkpoint scaler. | B1, E4 |
| Inference | A single-pair scaler is no longer tiled across 4 pairs; a mismatch raises. | E5 |
| Ensemble | Each base now scales its own input with its `*_scaler.npz` (bases used to get raw features). Added a learnable output scale and bias. The best checkpoint is chosen on held-out MSE. The entropy push toward uniform weights is removed. The scaler sidecar is copied with the promoted `*_best.pt`, and `fold_selection.json` `secondary_value` is fixed. | E1, E2 |
| Tests | `tests/test_audit_2026_09_25_fixes.py` (8 tests). | — |

**Still needs you (can't be done from code alone):**
1. **Rebuild the dataset.** The cache tag changed, so the next training run rebuilds it from ticks. Expect several hours.
2. **Retrain all bases**, which produces scaler sidecars. Then retrain the ensemble and RL, and re-run the gate.
3. **Not fixed here:** the news/FinBERT feed (B3, constant features); per-pair/period reweighting (B4); honest non-overlapping Sharpe with CIs (A3/D7); the tabular baseline (D6); RL GPU use; the `cudnn` DLLs missing from `.venv` (PyTorch won't import there; `.venv311` works).

### Next
- Wire `trading/preflight_check.py` into the live engine (M4).
- Rebuild dataset → baseline model (D6) → retrain.

---

## Dataset build check (2026-09-25, 3-week 4-pair build, 2024-03-04 → 03-22)

A real `--build-only` run of the current code, inspected array by array. Problems found and fixed:

| Problem | Effect | Fix |
|---|---|---|
| `resample_to_bars`: `pl.col("ask") - pl.col("bid").alias("spread_check")` is named `ask` and **overwrote every JPY ask price with the spread** (~0.005). Present since `dfe4103`. | USDJPY `ask_close` ≈ 0: cost-aware labels lost their spread (0% HOLD), and spread features and cached spread were wrong. | Stray expression removed; JPY spread now 0.50 pips, HOLD share 16.5%. |
| Freshly downloaded ticks carry the timestamp in the **index**; `_build_chunk` silently returned nothing without a `timestamp_utc` column. | **USDCAD dropped entirely** (2.19M raw ticks → 0 bars); the pair readiness gate failed. | `_ensure_timestamp_column` moves a datetime index into the column; a missing timestamp now raises instead of dropping the pair. |
| HMM regime decode: singular covariance. | Every window fell back to uniform "regime" probabilities. | Symmetrise + 1e-4 ridge, `allow_singular=True`. |
| Cross-asset factor model: "SVD did not converge". | No PCA/Granger/lead-lag factor features in any window. | The panel is cleaned (inf/NaN → 0) and rarely moving (forward-filled daily) columns are dropped first. |
| Cross-asset loader tried stooq first for every asset (4 symbols × 15 s timeouts) even with a cached Yahoo copy, and mislabelled the provider. | ~60 s per series, ~20 min per build. | Checks every provider's cache first, skips a provider after it fails outright, logs the provider that answered. Build time went from 1,643 s to 582 s. |
| Volatility clock: 7-day lookback longer than a build window. | `vol_clock_pace` / `vol_clock_hot` constant. | Lookback adapts to the available days (min 1 day of samples). |

**Verified on the rebuilt cache:**
- all arrays are row-aligned, with timestamps monotonic, none missing and no duplicates
- the scalar label equals the market pair's label 100% of the time
- per-pair label mix is 38–43% SELL / 14–26% HOLD / 36–43% BUY
- BUY/SELL labels agree with the forward price move for all pairs (BUY +7 to +14 pips, SELL −9 to −18)
- spreads are 0.2–1.1 pips
- `ret_*` are on a comparable bps scale across EUR and JPY, `rsi_14` averages 51, `eco_surprise` stays in [−1, 1], `sentiment_raw` varies and `noise_to_signal_60` is bounded

**Expected or acceptable:**
- constant `fb_0..7` (live placeholders)
- constant COT columns within a 3-week window (weekly data)
- raw price levels correlating with the forward return in a short trending sample; they're neutralised in the scaler

`DATASET_BUILD_VERSION` is now `a0925c`.
