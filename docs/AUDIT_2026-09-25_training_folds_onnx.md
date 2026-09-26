# Audit: Training Procedure, Best-Fold Selection & ONNX Export (2026-09-25)

**Scope:**
- walk-forward training (`training/cv_splits.py`, `train_gpu.py` fold loop)
- best-fold promotion and the challenger check (`post_train._promote_best_fold`, `fold_selection.json`, `deployment.json`)
- ONNX export and serving (`inference/onnx_inference.py`), including the RL schema verification (`training/rl_runner.py`)

**Method:** static review at `main` @ `6791044`, plus the saved fold configs and selections from the 2026-09 runs.

**Bottom line:**
- The deployed model for every architecture is the **fold-2** model. It was trained on the *oldest ~37%* of the history, and picked because fold 2's validation period happened to be the easiest to score.
- It has never seen the most recent ~60% of the data.
- The ONNX artifacts normalise inputs differently from training, and nothing checks ONNX outputs against PyTorch.

---

## Training & fold selection

### T1. The deployed model is one fold, trained on a fraction of the data (P0)
- `walk_forward_splits` uses expanding windows with `edges = linspace(0, n, n_folds + 2)`. Fold *k* trains on `[0, edges[k+1] − embargo − purge)` and validates on `[edges[k+1], edges[k+2])`.
- With 7 folds, fold 2 trains on the first **3/8** of the CV range. Fold 6, the latest, trains on 7/8.
- `_promote_best_fold` copies the winning fold's checkpoint to `<model>_best.pt`. That file is what the ensemble, the gate and live all load.
- `fold_selection.json` shows **fold 2 selected for HAELT, GNN and Mamba** (val_loss 1.062, 1.064 and 1.064), and the same for TFT.
- **Fix:** use CV only to *estimate* performance and choose hyperparameters. Then refit one model on all data up to the holdout, with the epoch count taken from the folds (for example the median best epoch), and gate that refit model on the holdout.

### T2. Folds are compared on non-comparable scores (P0)
- The "best" fold is the one with the lowest `best_val_loss` (or highest `best_val_sharpe_proxy` when `early_stop_metric` is a Sharpe).
- Each fold's loss is measured on *its own* validation period. Fold 2's validation set scores 0.98–1.06 for every architecture, while fold 1's scores about 1.54. The ranking reflects how hard each period was, not model quality.
- That's why all four architectures independently "choose" fold 2: a data artefact.
- The Sharpe variant is the label proxy (±30), and it takes the max over epochs.
- **Fix:** no fold picking (see T1). If a fold must be chosen, compare them on the same data, which in practice means the holdout gate.

### T3. The challenger check compares numbers that aren't comparable (P1)
- `deployment.json` stores the deployed fold's `metric_value` (a validation loss).
- A new run is accepted only if its best-fold loss is lower by `min_delta = 0.001`.
- The two numbers come from different folds, and after a dataset rebuild or loss change (today's fixes) from different data and objectives. 0.001 is far below fold-to-fold noise.
- **Fix:** challenger vs production on the same holdout, with the honest net Sharpe and its CI.

### T4. Fold provenance isn't recorded (P1)
- `*_fold*_config.json` holds hyperparameters and best metrics, but no train/validation index ranges, dates, cache hash or seed.
- You can't tell from the artefacts what a fold was trained on.
- **Fix:** record `train_range`, `val_range`, their timestamps (from `t_ns`), the cache tag and `DATASET_BUILD_VERSION`, the seed, and the selected epoch in each fold config.

### T5. Selection statistic within a fold (P2)
- Within each fold the best epoch is now the honest CI lower bound (2026-09-25, S7). But the fold-level `best_val_sharpe_proxy` in the config is still the old label-Sharpe maximum, and `_promote_best_fold` reads it.
- It's also used by `auto_optimal_roadmap`, which now reads the honest `fold_cost_sharpes` instead.
- **Fix:** write the honest fold statistic into the fold config, and drop `best_val_sharpe_proxy` from all selection paths.

### T6. Folds are near-duplicates of each other's data (P2)
- The folds are expanding, so fold 6 contains folds 0–5's training data. The 7 fold results are strongly dependent, and averaging them overstates the evidence.
- Report them as a time series of validation periods, not as 7 independent trials. The Deflated Sharpe trial count counts fold checkpoints; that's acceptable as a conservative *upper* count, but not as evidence of independence.

---

## ONNX

### O1. The fused scaler isn't the training scaler (P0)
- `export_to_onnx` fuses `load_inference_scaler(cache_path)`, the **cache-wide** `scaler.npz`, into the graph as `(x − center) / scale`.
- Training uses the **per-fold train-only** scaler saved beside the checkpoint (`*_scaler.npz`). It has different statistics, clips to ±10 and neutralises price-level and volume columns.
- The ONNX graph has no clip and no neutralisation, so the ONNX model sees inputs the PyTorch model never saw.
- If no cache scaler is found the export continues **without** normalisation ("exporting without normalization"), and the ONNX live engine (`predict_proba`) feeds raw features.
- **Fix:** fuse the checkpoint's own sidecar scaler plus the ±10 clip, and refuse to export when the sidecar is missing.

### O2. No output-parity check (P0)
- `_export_onnx` only checks that onnxruntime can *load* the file. It never compares ONNX outputs with PyTorch on the same inputs.
- The dynamo-exporter fallback (`dynamo=False`) and the MHA fast-path toggles are exactly where silent numerical differences come from.
- `tests/test_inference_consistency.py` currently fails, on a checkpoint/builder head-shape mismatch (3 vs 1). The existing consistency check isn't guarding anything.
- **Fix:** after export, run N random plus N real cache windows through both. Require `max|Δ| < 1e-4` on logits (or on action probabilities), else delete the file and fail.

### O3. Weights load with `strict=False` everywhere (P1)
- Every export path loads with `load_state_dict(..., strict=False)`: the model, the ensemble bases, and the RL policy and value nets.
- A renamed or reshaped layer silently keeps its random initialisation and exports.
- **Fix:** `strict=True`, or the `_strict_load_report` threshold (≥ 99% of keys loaded, head included), and log what's missing.

### O4. ONNX artefacts drift from the PyTorch checkpoints (P1)
- `ensemble_meta_best.onnx` (09-21) is older than `ensemble_meta_best.pt` (09-23), so the two serve different models (model audit E2).
- Nothing ties an `.onnx` to the `.pt` it came from.
- **Fix:** write `<name>.onnx.json` with the SHA-256 of the source `.pt`, the scaler, the features sidecar and the export config. Live, preflight and the certificate (gate v3 already hashes artefacts) should check it. Re-export whenever the `.pt` changes.

### O5. Output semantics changed today (P2)
- Since the S8 fix the export wrapper emits **log action-probabilities** [SELL, HOLD, BUY] for multitask models, where it used to emit direction logits.
- Consumers that apply softmax (Python ONNX engine, C++ `onnx_runner`) get the intended probabilities. Anything that treated output[0] as a raw logit or a scalar score would be wrong.
- **Fix:** bump an `onnx_output_semantics` field in the sidecar JSON and have consumers check it. Re-export all models after retraining.

### O6. Fixed sequence length and single-pair view (P2)
- The dynamic axes cover batch only; `seq_len` is baked in, which is fine if enforced.
- Live multi-pair assembly (L10) must match the export's `n_features`. `_verify_onnx_schema_deployment` checks the schema for RL. Apply the same verification to supervised and ensemble exports, using the `_features.json` sidecar.

---

## Recommended order

1. **T1 + T2:** stop deploying a single fold; refit on all pre-holdout data and gate the refit (the gate already works on the holdout).
2. **O1 + O2:** export with the checkpoint's own scaler and clip, then verify ONNX against PyTorch before keeping the file.
3. **T3, T4, O3, O4:** challenger on the same holdout, fold provenance, strict loading, and hash-linked ONNX sidecars.
4. **T5, T6, O5, O6.**
