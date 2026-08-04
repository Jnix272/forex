# Config Consistency — Settings, YAML, Curriculum, Dataset Schema

**Updated:** 2026-08-04  
**Audience:** anyone changing `config/settings.py`, `config/run.yaml`, curriculum groups, or the dataset builder.  
**Index:** [`README.md`](README.md) · tracker: [`CONTINUE.md`](CONTINUE.md)

When YAML is loaded (`--config config/run.yaml`), it is authoritative. `settings.py` is the no-YAML fallback and documentation. Drift between them used to be silent; the gates below make it loud.

---

## Where checks run

| When | Entry | What runs |
|------|--------|-----------|
| Preflight | `validate_run_config` / `--validate-config` | Curriculum drift, FEATURE_MASK orphans, section mismatches, args↔YAML |
| Dataset build | `training/dataset_builder.py` | Multi-part feature-schema gate (first chunk + final write) |
| Training start | `training/supervised_loop.py` | Curriculum / market / settings↔YAML warnings |

**CLI**

```bash
# Preflight only
uv run python -m training.train_gpu --validate-config --config config/run.yaml

# Dataset build (gate on by default with --integrity-gate)
uv run python -m training.train_gpu --force-rebuild ...
# Disable schema/mismatch gate only:
uv run python -m training.train_gpu --no-feature-schema-gate ...
```

Artifacts after a build:

- `{cache}_feature_schema.json` — ordered column names in `X`
- `{cache}_feature_schema_audit.json` — multi-part report (`parts.*`)

---

## Part 1 — Built schema (`built_schema`)

Compares the **locked feature columns** in the dataset cache to:

1. **Curriculum groups** (`run.yaml` → `curriculum.feature_groups`)  
   Missing names → **error** (freeze/unfreeze would silently no-op).
2. **Market roles** — `close`/`mid_close`, ATR (`atr_6`…), spread  
   Missing close/ATR → **error**; missing spread → **warning** (may be synthesized).
3. **FEATURE_MASK** — enabled-but-absent, extras not in mask → **warning**.

Code: `config/curriculum_audit.py` → `audit_built_dataset_schema()`.

Features themselves are built by `FeatureEngineer.build()` inside the dataset builder; curriculum only masks columns at train time.

---

## Part 2 — Settings ↔ YAML by section (`settings_yaml`)

Compares shared keys between `settings.py` dicts and matching YAML sections:

| YAML section | settings attr |
|--------------|---------------|
| `training` | `TRAINING` |
| `curriculum` | `CURRICULUM` |
| `backtest` | `BACKTEST` |
| `rl` | `RL` |
| `execution` | `EXECUTION` |
| `risk` | `RISK` |
| `data` | `DATA` |
| `distillation` | `DISTILLATION` |
| `monitoring` | `MONITORING` |

**Critical keys (fail closed)** — must stay synced as fallbacks:

- `training.seq_len`
- `training.loss`
- `training.sharpe_annualization_factor`
- `backtest.atr_stop_mult`
- `rl.reward.overtrade`

**Warnings** — other shared drift (`epochs`, `batch_size`, `grad_clip`, …) is expected when YAML overrides for a quick run; treat as documentation smell, not a load bug.

Curriculum **schedules** (`epoch_unfreeze`, `seq_schedule`, `difficulty_schedule`, early-stop scalars) are also checked by `audit_settings_yaml_curriculum_drift()` (errors on mismatch). Feature *name lists* live only in YAML.

Code: `config/config_mismatch_audit.py` → `audit_settings_yaml_section_mismatches()`.

---

## Part 3 — Args ↔ YAML (`args_yaml`)

Compares resolved CLI/args (after `_apply_yaml_config`) to the YAML file. Catches silent load failures (bad indent, missing `strategy:` mapping).

**Errors** if these diverge: `seq_len`, `loss`, `bar_freq`, `strategy_mode`, `profit_target_atr`, `stop_loss_atr`, `lookahead_bars`.  
**Warnings** for softer training knobs (`batch_size`, `epochs`, …).

Code: `config/config_mismatch_audit.py` → `audit_args_vs_yaml_mismatches()`.

---

## Curriculum groups (active `run.yaml`)

| Group | `epoch_unfreeze` | Role |
|-------|-----------------:|------|
| microstructure | 0 (always on) | OFI, Kyle, VPIN, … |
| momentum | 0 (always on) | yield / COT momentum |
| session | 0 (always on) | clock features |
| execution_cost | 2 | spreads, latency |
| volatility | 4 | vol-of-vol, Hurst, … |
| cross_asset | 8 | yield spreads, carry |
| news | 10 | eco / news cats |
| macro | 12 | COT, sentiment, FinBERT |
| market_regime | 16 | regime quality |
| higher_timeframe | 20 | 5m / 15m / 1h context |

`label_quality` is **commented out** in `run.yaml` (features not implemented). It may still appear in `run_ubuntu.yaml` — prefer `run.yaml` as the active source of truth.

---

## What lives where

| Concern | Source of truth |
|---------|-----------------|
| Feature column names in `X` | Built by `FeatureEngineer`; documented in `config/feature_mask.py` |
| Curriculum freeze schedule + feature lists | `config/run.yaml` |
| Curriculum schedule stubs (no feature lists) | `config/settings.py` → `CURRICULUM` |
| Training hyperparams for a run | `config/run.yaml` (overrides settings) |
| Strategy ATR / bar_freq / mode | `config/run.yaml` → `strategy:` |

---

## Tests

```bash
uv run pytest tests/test_curriculum_audit.py -q
```

Covers: YAML parse / strategy indent, curriculum orphans/overlaps, settings↔YAML curriculum drift, built-schema gate, section critical-key mismatches, multi-part enforce raise + audit JSON.

---

## How to change config safely

1. Edit **`config/run.yaml`** first (active).
2. If you change a **critical** key or curriculum `epoch_unfreeze` / schedules, update **`settings.py`** stubs to match.
3. Run `tests/test_curriculum_audit.py` and `--validate-config`.
4. Rebuild the dataset if feature columns or `seq_len` / strategy ATR / label params changed (`--force-rebuild`).
