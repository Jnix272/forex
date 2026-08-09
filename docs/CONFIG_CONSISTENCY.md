# Config Consistency — Settings, YAML, Curriculum, Dataset Schema

**Updated:** 2026-08-06  
**Audience:** anyone changing `config/settings.py`, `config/run.yaml`, curriculum groups, or the dataset builder.  
**Index:** [`README.md`](README.md) · status: [`IMPROVEMENTS.md`](IMPROVEMENTS.md) · next: [`CONTINUE.md`](CONTINUE.md)

When YAML is loaded (`--config config/run.yaml`), it is authoritative. `settings.py` is the no-YAML fallback and documentation. Drift between them used to be silent; the gates below make it loud.

---

## Where checks run

| When | Entry | What runs |
|------|--------|-----------|
| Preflight | `validate_run_config` / `--validate-config` | Curriculum drift, FEATURE_MASK orphans, section mismatches, args↔YAML |
| Dataset build | `training/dataset_builder.py` | Multi-part feature-schema gate (first chunk + final write) |
| Training start | `training/supervised_loop.py` | Curriculum / market / settings↔YAML warnings |
| Config apply | `training/gpu_cli._sync_runtime_config` | `execution:` → `EXECUTION`; **`risk:` → `LIVE_RISK`** (session/regime deep-merge) |

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

## Nested YAML blocks attached to args

| YAML section | Args / runtime | Notes |
|--------------|----------------|-------|
| `curriculum` | `args.curriculum` | Feature groups + schedules |
| `execution` | `args.execution` → `settings.EXECUTION` | Whole-dict sync |
| `risk` | `args.risk` → **`settings.LIVE_RISK`** | Deep-merge session_limits + regime_scale; aliases `volatile→crisis`, `ranging→mean_rev`, `unknown→normal` |
| `sidecar` | `args.sidecar` | Monitoring Sidecar; default `enabled: false` |
| `training.use_mixup` / `use_volatility_sampler` | `args.use_mixup` / `use_volatility_sampler` | Applied in supervised loop |
| `direction_training.use_mixup` / `use_volatility_sampler` | same dests | Direction head twins |

### Quick mode vs ensemble / RL

`quick.enabled: true` still caps epochs and **forces** `train_ensemble=False` / `rl_train=False` (smoke intent). Default `run.yaml` keeps **`quick.enabled: false`** so `ensemble.enabled` / `rl.enabled` actually run. When quick overrides YAML that had them on, gpu_cli prints a WARN.

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

**Critical keys (fail closed)** — must stay synced as fallbacks
(`config_mismatch_audit.CRITICAL_SHARED_KEYS`):

- `training.seq_len`
- `training.loss`
- `training.sharpe_annualization_factor`
- `training.lr_warmup_epochs`
- `backtest.atr_stop_mult`
- `rl.reward.overtrade`
- `validation.embargo_bars`
- `validation.purge_bars`
- `pretrain.epochs` (canonical `run.yaml` only; profile YAMLs treat this as scale)
- `distillation.temperature`

Also fail-closed: **`strategy.*` ↔ `LABELING`** for `lookahead_bars`,
`profit_target_atr`, `stop_loss_atr` (`audit_strategy_vs_labeling`).
Canonical defaults: **LH=30 / TP=1.2 / SL=0.8** (scalping profile + `run.yaml`).

**PRETRAIN naming:** YAML / settings use `pretrain.epochs` (and `min_epochs`).
CLI / argparse still expose `--pretrain-epochs` → `pretrain_epochs`; settings
keeps both keys mirrored. Prefer editing YAML `pretrain.epochs`.

**`run_ubuntu.yaml`:** hardware / scale profile only (`PROFILE_SCALE_KEYS` —
epochs, batch_size, patience, chunk_size, paths, hardware.profile). Strategy,
labeling, and critical shared keys must match `run.yaml` — **Ubuntu is not a
second strategy/labeling source of truth**.

**LABEL_REGIME** (settings-only policy for RL barriers / session cost+horizon /
spread_z horizon gate): edits bust the training cache via `lr{digest6}` in
`cache_integrity` (with FEATURE_MASK `fm*`). Session overlap keys:
`asia_london`, `london_ny`. Status: [`IMPROVEMENTS.md`](IMPROVEMENTS.md); detail: [`SESSION_AUDIT.md`](SESSION_AUDIT.md).

**Warnings** — other shared drift (`grad_clip`, …) is expected when YAML overrides for a quick run; treat as documentation smell, not a load bug.

Curriculum **schedules** (`epoch_unfreeze`, `seq_schedule`, `difficulty_schedule`, early-stop scalars) are also checked by `audit_settings_yaml_curriculum_drift()` (errors on mismatch). Feature *name lists* live only in YAML.

Code: `config/config_mismatch_audit.py` → `audit_settings_yaml_section_mismatches()`.

---

## Part 3 — Args ↔ YAML (`args_yaml`)

Compares resolved CLI/args (after `_apply_yaml_config`) to the YAML file. Catches silent load failures (bad indent, missing `strategy:` mapping).

**Warnings** when resolved args diverge from YAML (`seq_len`, `loss`, strategy ATR, `batch_size`, …).  
CLI flags and strategy profiles intentionally override YAML — hard fail closed lives in settings↔YAML critical keys and the strategy-block presence check, not every args drift.  
Skipped entirely when `--config` was not passed (no false “YAML may not have applied” on argparse defaults).

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

`label_quality` is **commented out** in `run.yaml` and `run_ubuntu.yaml` (features not implemented). Prefer keeping both files aligned; **Ubuntu = HW/scale only** (paths, hardware.profile, PROFILE_SCALE_KEYS) — not a second strategy/labeling source of truth.

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
