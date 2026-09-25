# Audit: Self-Supervised Pretraining (2026-09-25)

**Scope:**
- `training/pretrain_runner.py` and `training/pretrain_adapter.py`
- `pretrain/` (contrastive, masked, forecast, cluster trainers; loss scaling; guardrails; handoff)
- the encoder transfer in `training/supervised_loop._load_pretrained_encoder`
- the saved `pretrain_report.json` for TFT, HAELT, Mamba and GNN (runs of 2026-09-15 to 09-20)

**Method:** static review plus the saved reports. Nothing was run.

**Bottom line:**
- Pretraining learned nothing usable. The losses are about 1.5 billion and flat or rising.
- Its weights were still loaded into every model, including one the quality gate had discarded.
- The only with-versus-without comparison on record says it **hurt**.

---

## What the reports show

| Model | Method | Epochs | Loss first → last | Embed std | Gate | Loaded? |
|---|---|---|---|---|---|---|
| TFT | masked | 18 | 1.49e9 → 1.75e9 | 0.054 | "passed" | yes (100%) |
| HAELT | masked | 12 | 1.49e9 → 1.65e9 | 0.43 | "passed" | yes (100%); ablation verdict **pretrain_hurt** |
| Mamba | forecast | 14 | 1.37e9 → 1.51e9 | 0.38 | "passed" | yes (100%) |
| GNN | cluster | 5 | 0.196 | **0.0** | **discarded** (failed_low_uniformity) | **yes (100%)** |

Alignment and uniformity are reported as 0.0 for the masked and forecast runs, where they aren't computed. A 0 reads like a real value.

---

## P0

### P1. Discarded encoders are loaded anyway
- When the quality gate fails, `run_pretrain` writes `status: discarded` and returns a fresh model.
- But `contrastive_encoder.pt` was already written during training and is **not deleted**.
- `supervised_loop._load_pretrained_encoder` loads whatever `contrastive_encoder.pt` is in the checkpoint directory, without reading the report.
- GNN's collapsed encoder (embedding std 0.0) went into all 7 folds.
- **Fix:** on discard, delete or rename the checkpoint. The loader should also require `status == "completed"` and an accepted gate result in `pretrain_report.json`.

### P2. Pretraining sees unscaled inputs; supervised training sees scaled ones
- `_read_pretrain_spans` feeds raw cache windows to the encoder, with only `nan_to_num`: no scaler, no clip.
- Supervised training feeds the same encoder robust-scaled, ±10-clipped inputs, with price levels and volume neutralized.
- The transferred first layers, meaning input projections and normalization statistics, were fitted to a different input distribution, so the "pretrained" weights start supervised training miscalibrated.
- The inputs also include the ±1e6 placeholders and raw COT, volume and price-level columns flagged in the dataset audit (B1).
- **Fix:** pretrain on exactly the supervised transform: a scaler fit on the pretrain window, the same clip, and the same neutralized columns.

### P3. The reconstruction loss is dominated by a few near-constant features
- `normalized_mse_loss` divides each feature's error by that feature's std *within the batch*, and treats anything with std ≥ `min_scale = 1e-5` as active.
- Near-constant columns (placeholders, slow macro and COT series, sticky flags) get std around 1e-5 to 1e-3, so their errors are amplified by 1e3 to 1e5.
- Together with P2's raw inputs, that's how the loss reaches about 1.5e9. The objective is reconstructing noise in a few near-constant columns, so the loss never falls and the embedding learns little.
- **Fix:** scaled inputs (P2) plus a sensible scale floor, for example `min_scale` equal to a fraction of the global median std. Drop columns that are constant within the window from the loss.

### P4. The quality gate passes non-learning runs
- For masked and forecast methods, the gate only checks embedding collapse (std < 0.015).
- A loss that rises over training (TFT 1.49e9 → 1.75e9) still counts as "passed".
- There is no check that the loss beats a trivial baseline, such as predicting the per-feature mean or the last value, and no held-out loss at all.
- **Fix:** gate on a held-out pretext loss that beats a naive baseline by a margin and falls over training. Record "not computed" rather than 0.0 for alignment and uniformity.

### P5. The only evidence says pretraining hurts, and it's ignored
- HAELT's recorded ablation verdict is `pretrain_hurt`, but the weights were loaded anyway. For the other models no ablation ran (`ablation: null`).
- `--pretrain-ablation` defaults to off in the run and in the smoke test.
- **Fix:** make pretraining opt-in per model, and load an encoder only when an ablation on the same folds shows a gain (honest net Sharpe or validation loss beyond noise). Otherwise skip it; it cost 5–6 minutes per model per run for nothing.

---

## P1

### P6. One encoder spans every walk-forward fold (look-ahead into early folds)
- Pretraining runs once, on windows 0–166,524 (the whole CV region), before fold 0.
- Fold 0 then fine-tunes an encoder that has already modelled the input distribution of folds 1–6's validation periods, which lie in its future.
- The pretext task is unsupervised, so labels don't leak, but the representation does, and early-fold validation looks better than it would live.
- **Fix:** pretrain per fold on that fold's training span only, or on a span that ends before the first validation window.

### P7. The handoff threshold is on the wrong scale
- `pretrain_handoff_loss = 0.18` is compared with losses around 1.5e9, so the early-handoff rule can never fire.
- `min_delta = 0.002` is likewise meaningless at that scale. Every run went to its full epoch budget.
- These thresholds only make sense once P2 and P3 give an O(1) normalized loss.

### P8. One checkpoint file per directory, shared across methods
- Every method saves `contrastive_encoder.pt` (plus `_ep*.pt`) into the model's checkpoint directory. `_load_pretrained_encoder` picks the first candidate that exists.
- A stale encoder from an earlier run or method (for example the root `checkpoints/contrastive_encoder.pt`, or `_ep1` files) can be loaded against a new config.
- The 100% `frac_loaded` only shows that the shapes matched, not that the file came from this run.
- **Fix:** name the checkpoint by run id and method, record its hash in the report, and have the loader verify the report and hash.

### P9. The transfer check can't detect a useless transfer
- `_strict_load_report(min_frac_loaded=0.6)` only checks key and shape overlap.
- A collapsed or raw-scale encoder loads at 100%.
- Combined with P1 and P5, the system reports "loaded_into_supervised_training: true" as success regardless of quality.

---

## P2

- **P10.** The GNN cluster run reports `final_embedding_std: 0.0` together with `uniformity: -0.08`, which contradicts itself. The diagnostics are computed on `_last_w` (the last sampled block) and not on held-out data.
- **P11.** `pretrain_regime` ("regime-aware") is on in the run, but it only applies to `tscl`; the masked, forecast and cluster runs report `regime_aware: false`. The config suggests a feature that isn't used.
- **P12.** Hard-example injection reads `logs/hard_examples.json`, which the *supervised* miner produced from label-based losses. It mixes supervised difficulty into an unsupervised task and depends on a file from an earlier run. The indices are range-checked but not tied to the current cache.
- **P13.** When the ZMQ or feed isn't involved, pretraining still uses about 5–6 minutes per model per run (`stage_timings`), about 25 minutes per 4-model run, for no measured benefit.

---

## Recommended order

1. **Now (cheap):**
   - P1: honor the discard (delete the file, check the report).
   - P5: turn pretraining off by default (`pretrain: false`), or load only on a positive ablation.
   - This removes the known harm immediately.
2. **Before re-enabling:**
   - P2 + P3: same inputs as supervised training, and a sane loss floor.
   - P4 + P7: a held-out, baseline-relative gate with thresholds on the new scale.
   - P6: per-fold spans.
   - P8: run-scoped checkpoints with a hash.
3. **Re-evaluate:** run the with/without ablation on the rebuilt dataset. Keep pretraining only for models where it wins across folds.

---

## Fix log

| # | Fix |
|---|---|
| P1 | Every discard path renames `contrastive_encoder.pt` and its `_ep*.pt` copies to `*.discarded`. The loader requires the report to say `completed` with the gate `passed`. |
| P2 | Pretraining inputs go through a RobustScaler fit on the pretrain window only, with the same ±10 clip and price-level/volume neutralisation as supervised training (`_fit_pretrain_scaler`). |
| P3 | `normalized_mse_loss` scale floor raised from 1e-5 to 1e-2, so near-constant channels are no longer amplified. |
| P4 / P10 | The last 10% of the pretrain window is held out. The gate requires the held-out pretext MSE to be < 0.95 × a predict-the-mean baseline and the training loss to fall; otherwise it discards. Diagnostics are computed on the held-out block. |
| P5 / P13 | `pretrain.enabled: false` by default. The loader refuses an encoder whose ablation verdict is `pretrain_hurt`, and one trained on unscaled inputs (older runs). |
| P6 | `train_gpu` ends the pretrain window before the first CV validation window, minus the embargo. |
| P7 | With scaled inputs and the loss floor the pretext loss is O(1), so the 0.18 / 0.002 handoff thresholds are on the right scale. |
| P8 / P9 | The report records `checkpoint_sha256`. The loader verifies it along with status, gate result, ablation verdict and `scaled_inputs`. |
| P11 | A note is logged when `regime_aware` is set for a method other than tscl. |
| P12 | Hard-example injection is opt-in (`pretrain.hard_examples`, default false). |

**Tests:** `tests/test_pretrain_fixes_2026_09_25.py`. The existing pretrain adapter, loss-scaling and upgrade tests pass: 44 in total.

**Consequence:** every existing encoder is refused by the loader (no `scaled_inputs` or hash).
