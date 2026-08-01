# Per-model production configs

Each YAML here contains **only** architecture-specific overrides sourced from `config/models.py`.
Shared training settings (epochs, batch size, loss, walk-forward, etc.) still come from `config/run.yaml`.

## Commands

**Fair architecture sweep** — identical hyperparameters from `run.yaml` for every model:

```powershell
python scripts/train.py -- --all-models --fair-sweep
```

**Production single model** — merge this file with your base run config:

```powershell
python scripts/train.py --config config/models/mamba.yaml
```

For a full run, point `--config` at a copy that includes data paths, or pass extra flags after `--`.

**Production all models** — profiles applied automatically per architecture (default):

```powershell
python scripts/train.py -- --all-models
```

Disable profiles explicitly:

```powershell
python scripts/train.py -- --all-models --no-model-profile
```

## How profiles are applied

When `--model-profile` is on (default), `training/train_gpu.py` merges `architecture_config(model)` onto training args inside `_member_training_args()` before each model trains. Explicit CLI flags (`--lr`, `--dropout`, etc.) are never overwritten.

Use `--fair-sweep` or `--no-model-profile` when comparing architectures on equal footing.
