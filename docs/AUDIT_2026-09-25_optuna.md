# Audit: Optuna Hyperparameter Search (2026-09-25)

**Scope:** `scripts/optuna_tune.py`, `training/hpo.py::build_optuna_search`, the trial-summary code in `training/train_gpu.py`, and the stored study `logs/optuna/optuna_tft_cheap_cost_sharpe_0f686a1209.db`.
**Mode:** findings only, no code changed. Checked against HEAD `cd4ef41`, which already fixed several Optuna issues.

**Bottom line:** Optuna has produced no usable result. The only study has 20 FAIL trials, 2 COMPLETE trials that both scored `-inf`, and 1 stuck in RUNNING. Even when trials succeed, the default setup evaluates only the 2 direction-warmup epochs, so it can't measure the settings it's tuning.

---

## Study state (evidence)

| State | Count | Notes |
|---|---|---|
| FAIL | 20 (trials 0–19) | Each died about 2 minutes after starting, with no error saved in `stdout_tail`. The logs end on config-drift warnings with no traceback. |
| COMPLETE | 2 (trials 20, 21) | Both scored `-inf`: `missing_metric: cost_sharpe`. |
| RUNNING | 1 (trial 22) | Orphaned. The process is gone, but the trial was never marked failed. |

Trials 20–21 ran before commit `cd4ef41`. Finding O2 shows the problem still exists after it.

---

## P0: the search can't find good settings

### O1. The 2-epoch proxy only ever runs warmup
- In `cheap` mode (the default), each trial trains for `epochs: 2`.
- `direction_warmup_epochs` defaults to 2, and those epochs train a direction-only loss on a class-balanced subset (see the model audit, A1).
- So every proxy trial is scored entirely on warmup epochs. The full multitask loss, the Sharpe/return heads and the curriculum never run.
- The rankings therefore say nothing about how a setting behaves in full training. Deep mode (8 epochs) still leaves only 6 non-warmup epochs.

### O2. The `cost_sharpe` objective is always `-inf` in walk-forward mode
- Trials run with `--walk-forward-folds N`, so the summary records `train_mode = walk_forward_cv` even when N is 1.
- In `train_gpu.py:1165-1173`, `best_cost_sharpe` is only written when `early_stop_metric == "cost_sharpe"`. The default is `val_loss`, which the trial summaries confirm.
- The fallback in `_metric_score` looks for `cost_aware_sharpe` in the checkpoint history. It isn't found there, so the score becomes `-inf`.
- When every value is `-inf`, the TPE sampler has nothing to learn from and behaves like random search. `study.best_trial` is then an arbitrary `-inf` trial, and it still gets exported.
- There is a second problem in the same code. When `early_stop_metric = val_loss`, `fold_best_metrics` holds validation losses, so any Sharpe computed from it mixes up the two metrics.

### O3. Failed trials record nothing
- 20 of 22 trials failed with an empty `stdout_tail` and no `diagnostics`.
- `study.optimize(catch=(RuntimeError, CalledProcessError))` doesn't catch other exceptions. A `FileNotFoundError` from a missing checkpoint, or an error from `torch.load`, stops the whole study.
- An interrupted run leaves trials in RUNNING (trial 22). `load_if_exists=True` then resumes the study with those zombie trials still in place.

### O4. The best config is never applied
- The comment says the tuner will back up and overwrite `run.yaml`, but `_export_best_config` only writes to `config/optuna/…` and prints "To apply: copy …".
- `--auto` / `--launch-training` then trains from the unchanged `config/run.yaml` (`ACTIVE_RUN_CONFIG`). The full run ignores the Optuna result.

### O5. The confirmation step doesn't affect what gets exported
- `_confirm_top_trials` retrains the top K trials with 7 folds and 18 epochs and writes a report.
- `_export_best_config` still exports `study.best_trial`, the proxy winner. The confirmation results are never used to choose.

---

## P1: noisy or wrong measurement

### O6. The Sharpe used for scoring is noise
- Per-epoch validation Sharpe swings between −34 and +39 (model audit, A3).
- The penalties in `_metric_score` are ±0.03–0.10 in size, so they can't change the ranking.
- The "−0.10 if the curriculum never advanced" penalty applies to every 2-epoch trial, because the curriculum is pinned. It's a constant offset.
- `best_val_sharpe` is the maximum over epochs, which favours lucky epochs.

### O7. Pruning acts on a single noisy epoch
- `MedianPruner(n_startup_trials=5, n_warmup_steps=1)` with 2-epoch proxies means a trial is pruned based on one warmup-epoch reading of a noisy Sharpe.
- A Sharpe swing of about ±30 makes the median comparison close to a coin flip.

### O8. Curriculum settings aren't reaching the trainer
- The trial logs show `[CurriculumDrift] curriculum.seq_schedule: settings != run_optuna_tft_proxy_N.yaml`. The trainer used `seq_len=120` from its settings, not the trial's schedule.
- Even in deep mode, the sampled curriculum parameters may be ignored.
- The generated seq schedule can also go backwards in time, for example `{epoch 10: 60}, {epoch 1: 90}`, because `_build_seq_schedule` produces a second ramp before the first when epochs are few.

### O9. Trials don't match production training
- Trials use `disable_pretrain=True`, but the exported and launched run pretrains.
- Trials also use a different sample count from the production cache (188,311 vs 185,201) and a single fold, with one seed and no repeats.
- These settings interact with pretraining, so the proxy optimum may not carry over.

### O10. The search space includes settings with no effect
- `batch_size` choices come from `_hardware_safe_batch_choices`, but the learning rate is sampled independently of batch size, so the two are coupled but searched as if independent.
- `mt_focal_gamma` and `mt_class_balance_weight` are searched while the dataset has almost no HOLD class in some builds (model audit, B6), so they may have no effect.
- The ranges were never checked against the flat-loss regime the models are stuck in (model audit, A2). No setting can escape a dataset with no learnable signal.

---

## P2: housekeeping

- **O11.** The study name includes a hash of `run.yaml`, so any config edit starts a new study and throws away earlier trials. The single `.db` holds no transferable knowledge.
- **O12.** Trial checkpoints `checkpoints/optuna_tft_proxy_1..22` are deleted per trial but otherwise never cleaned up.
- **O13.** Only `tft`, `haelt` and `transformer` can be tuned. `mamba` and `gnn`, which are in the deployed ensemble, can't.
- **O14.** Nothing checks trial-to-trial variance. There's no rerun of the best trial with other seeds before export.

---

## Recommendations (priority order)

1. **Fix the underlying problems first.** Fix the dataset (normalization, JPY pip size) and warmup checkpoint selection before tuning anything. Tuning a model that learns nothing wastes compute.
2. **Proxy length.** Set the proxy to at least `direction_warmup_epochs + 4` epochs, or set `direction_warmup_epochs=0` for proxy trials. Score only epochs after warmup.
3. **Objective plumbing.** Always write `best_cost_sharpe`, meaning the honest net Sharpe, into the trial summary, whatever `early_stop_metric` is. Fail loudly instead of returning `-inf`, and abort the study if more than N trials are missing the metric.
4. **Robust objective.** Score each trial by the median across 2–3 folds of the Sharpe after costs on non-overlapping trades, measured at the selected epoch rather than the maximum. Include a penalty for variance across folds.
5. **Failure handling.** Catch `Exception` in `optimize`, save the tail of stderr, and mark stale RUNNING trials as FAIL on resume.
6. **Choose from confirmation.** Export the best confirmed trial, not the proxy best. Before export, require that it beats the current `run.yaml` baseline by more than the variance measured across seeds.
7. **Apply the result.** Either make `--launch-training` use the exported config path or remove the "applied automatically" comment. Don't overwrite `run.yaml` without a backup.
8. **Curriculum.** Make trial curriculum settings override the defaults (resolve the drift warning), keep schedules in epoch order, and search curriculum only in deep mode.
9. **Pruning.** Use a patient pruner (`n_warmup_steps` of at least 3 epochs after warmup) or `SuccessiveHalving` with a minimum resource above the warmup length.
10. **Search space.** Sample the learning rate relative to batch size, drop parameters that have no effect, and add `mamba` and `gnn`.

---

## Fix log (branch `claude/audit-fixes`)

| Finding | Fix |
|---|---|
| O1 | Proxy defaults: cheap is 6 epochs × 2 folds, deep is 12 epochs. The tuner refuses `--epochs` below direction-warmup + 2. |
| O2 | `train_gpu` always writes `best_cost_sharpe`: the median over folds of the post-warmup (best + final) / 2 cost Sharpe, whatever `early_stop_metric` is. `fold_best_metric_name` records what `fold_best_metrics` holds. |
| O3 | `optimize(catch=(Exception,))`. Stale RUNNING trials are marked FAIL on resume. The study aborts export when no trial has a finite score. |
| O4 | `--launch-training` trains from the exported config, not the unchanged `run.yaml`. The misleading "auto-apply" comment is removed. |
| O5 | Export picks the best *confirmed* trial when confirmation ran (`selected_by` in the summary). |
| O6 | Curriculum bonuses and penalties are skipped for proxies shorter than 12 epochs. |
| O7 | `MedianPruner(n_warmup_steps = warmup + 2)`. |
| O8 | Seq schedules stay inside the run, in epoch order, and only increase. **Still open:** `CurriculumDrift` (trial schedule not reaching the trainer). |
| Not fixed | O9 (pretraining off in trials), O10 (search space), O11–O14. |
