# Setup Guide

This guide sets up the Forex Scaling Model project for Python training, ONNX export, live/shadow inference, and the optional C++ execution server.

## 1. Requirements

- Python 3.11 or newer. The project metadata requires `>=3.11`.
- Windows PowerShell is the main local workflow.
- NVIDIA GPU is recommended for full training.
- Visual Studio 2022 with the "Desktop development with C++" workload is required only for the C++ execution server.

## 2. Python Environment

Use a project virtual environment named `.venv-gpu`.

If `.venv-gpu` already exists but commands fail with an error like:

```text
Unable to create process using ... pythoncore-3.12-64\python.exe
```

the venv points to a missing Python install. Recreate it:

```powershell
Remove-Item -Recurse -Force .venv-gpu
py -3.11 -m venv .venv-gpu
.\.venv-gpu\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

If `py -3.11` is not available, install Python 3.11+ and rerun the command.

For normal development and tests:

```powershell
python -m pip install -r requirements.txt
```

For the full GPU/training stack:

```powershell
python -m pip install -r requirements_gpu.txt
```

Quick environment check:

```powershell
python --version
python -c "import torch, numpy, pandas, polars, yaml; print('python env ok')"
python -m pytest tests\test_config.py tests\test_promotion_artifacts.py -q
```

## 3. Configuration

Primary runtime config lives in:

```text
config/run.yaml
```

Important paths and outputs:

```text
checkpoints/
logs/
logs/auto_tune/
logs/training_memory.json
```

The active project flow writes per-model artifacts under:

```text
<checkpoint_dir>/<model>/
```

Expected artifacts include:

```text
train_summary.json
fold_selection.json
promotion_gate.json
deployment.json
manifest.json
<model>_model_card.json
```

## 4. Training

Run a small synthetic smoke training pass first:

```powershell
python training\train_gpu.py --data-source synthetic --model haelt --epochs 2 --force-rebuild --no-wandb --checkpoint-dir checkpoints\smoke
```

Run the configured training pipeline:

```powershell
python training\train_gpu.py --config config\run.yaml --no-wandb
```

Train all registered deep models:

```powershell
python training\train_gpu.py --config config\run.yaml --all-models --no-wandb
```

Train the ensemble meta learner after base checkpoints exist:

```powershell
python scripts\train_ensemble_meta.py --help
```

Recommended ensemble progression:

1. Train individual model recipes first.
2. Backtest each model by itself.
3. Train the ensemble meta learner.
4. Export one combined ONNX when possible.
5. Run ensemble in Python shadow mode before changing C++.

## 5. Validation And Tests

Run focused tests:

```powershell
python -m pytest tests\test_config.py tests\test_promotion_artifacts.py -q
```

Run the synthetic smoke test:

```powershell
python -m pytest tests\test_smoke.py -q
```

If `pytest` is missing, install dev dependencies:

```powershell
python -m pip install -r requirements-dev.txt
```

If `numpy` or `torch` imports fail, the wrong Python interpreter is active. Re-activate `.venv-gpu`:

```powershell
.\.venv-gpu\Scripts\Activate.ps1
where python
python -c "import sys; print(sys.executable)"
```

## 6. ONNX Export

Training promotion can automatically export:

```text
production_best.onnx
production_best.schema.json
reload_model.flag
```

Manual export can be done through the inference module:

```powershell
python inference\onnx_inference.py export --checkpoint checkpoints\haelt_best.pt --model haelt --seq-len 60 --n-feat 128 --output checkpoints\production_best.onnx
```

Use the real feature count from the trained model/cache. A mismatched `--n-feat` will create an ONNX model that does not match live inference.

## 7. Python Live And Shadow Runtime

The Python live engine watches for:

```text
reload_model.flag
production_best.pt
production_best.onnx
production_best.schema.json
```

Keep new models in shadow mode first. Production deployment should only happen after:

- checkpoint loads successfully
- ONNX export succeeds
- feature schema matches
- promotion gate passes
- backtest/forward holdout is acceptable

## 8. Discord Alerts

Set the webhook only if you want Discord notifications:

```powershell
$env:DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
$env:DISCORD_USER_ID="optional_user_id"
```

Current alert phases include:

```text
training_started
training_epoch
training_completed
fold_selected
promotion_gate_passed
promotion_gate_failed
production_deploy_completed
production_deploy_failed
model_demoted
```

Critical alerts bypass normal rate limiting.

## 9. C++ Execution Server

The C++ server is optional and should stay single-ONNX first. Do not add multi-model C++ ensemble runtime until the Python ensemble has passed shadow and forward-holdout validation.

### Install Visual Studio

Install Visual Studio 2022 with:

```text
Desktop development with C++
```

### Install vcpkg

```powershell
git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
```

Install C++ dependencies:

```powershell
C:\vcpkg\vcpkg.exe install curl nlohmann-json onnxruntime
```

### Build

Open this folder in Visual Studio:

```text
infrastructure/cpp_server/
```

Select the CMake target:

```text
ForexExecutionServer
```

Optional parity target:

```text
ParityCheck
```

### Runtime Environment Variables

Required:

```powershell
$env:OANDA_ACCOUNT_ID="..."
$env:OANDA_BEARER_TOKEN="..."
```

Optional:

```powershell
$env:MODEL_PATH="checkpoints\production_best.onnx"
$env:OANDA_INSTRUMENT="EUR_USD"
$env:OANDA_PAPER="1"
$env:SHADOW_MODE="1"
$env:SHADOW_JOURNAL="logs\cpp_shadow_journal.jsonl"
$env:MAX_POSITION_UNITS="100000"
$env:MAX_STALE_DATA_MS="5000"
$env:MAX_DRAWDOWN="500"
```

Only for shadow experiments when the C++ feature builder is not fully feature-complete:

```powershell
$env:ALLOW_PARTIAL_FEATURES="1"
```

Do not use partial features for live trading.

## 10. Troubleshooting

### `.venv-gpu` launcher is broken

Recreate the venv:

```powershell
Remove-Item -Recurse -Force .venv-gpu
py -3.11 -m venv .venv-gpu
.\.venv-gpu\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_gpu.txt
```

### `pytest` is missing

```powershell
python -m pip install -r requirements-dev.txt
```

### `numpy` or `torch` is missing

You are likely using the wrong interpreter:

```powershell
where python
python -c "import sys; print(sys.executable)"
```

Activate `.venv-gpu` again and reinstall requirements.

### ONNX model expects more features than C++ can build

Use shadow mode only:

```powershell
$env:SHADOW_MODE="1"
$env:ALLOW_PARTIAL_FEATURES="1"
```

Then complete feature parity before live trading.

