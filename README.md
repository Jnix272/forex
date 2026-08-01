# Forex Scaling Model

A production-grade algorithmic trading pipeline designed for high-frequency forex scaling and directional trading. This project implements a comprehensive full-stack machine learning solution, encompassing feature engineering, self-supervised contrastive pretraining, reinforcement learning (RL) based execution, and robust backtesting.

## Overview

The core architecture is built around regime-aware pretraining and continuous model iteration. It solves the critical problem of domain shift in financial time-series by utilizing regime-specific augmentations and extreme regime oversampling. The learned embeddings are passed through a maturity ladder to a reinforcement learning handoff for optimal execution and scaling behavior.

### Key Components

1. **Pretraining & Contrastive Learning (`pretrain/`)**
   - Implements `TSCLTrainer` (Time-Series Contrastive Learning) and `BYOLTrainer` for representation learning.
   - **Regime-Aware Dynamics**: The `RegimeAwareTSCLTrainer` oversamples extreme market regimes to prevent minority class collapse.
   - **Curriculum Augmentation**: `TimeSeriesAugmenter` progressively scales the intensity of jitter, scale, crop, and channel shuffle augmentations. It alters the augmentation strategy conditionally based on whether the market is trending or mean-reverting.
   - **Supervised Handoff**: Monitors alignment, uniformity, and embedding standard deviation. Automatically hands off embeddings when quality gates are met during training plateaus.
   - **Automated Pretraining Ablation**: Dynamically runs a controlled baseline vs. pretrained comparison run in a single execution to strictly measure and report the benefit of representation learning.

2. **Feature Engineering & Labeling (`features/`, `labeling/`)**
   - Supports generation of advanced indicators (e.g., fast trend scores, regime labels, MACD, RSI, ATR).
   - Generates continuous RL reward labels using `compute_rl_reward_labels_regime`, which embeds slippage, spread, latency, and bad-win penalties directly into the reward signal instead of relying on discrete classification labels.
   - **Robust Dataset Compilation**: Automatically parses and joins multi-pair ticks, gracefully skipping holidays and missing-data chunks to build stable multi-dimensional Zarr arrays.

3. **Robust Backtesting (`backtesting/`)**
   - `ForexScalingBacktest` executes bar-by-bar simulations accounting for the "Golden Rule" (trading on bid/ask, not mid).
   - **Execution Modeling**: Precisely simulates commission, dynamic slippage, and Square Root Law market impact based on trade volume.
   - **Risk Integrity**: Incorporates max drawdown circuit breakers and position limits.
   - Outputs comprehensive tear sheets via `pyfolio` alongside rich metrics (Sharpe, Sortino, hit rates, win-loss ratio).

4. **Inference & Governance (`inference/`, `deployment/`)**
   - Supports ONNX and PyTorch runtimes (`PyTorchInferenceEngine`).
   - `ShadowModeManager`: Manages the promotion of models via a strict governance protocol and MATURITY_LADDER logic before pushing them to live trading.
   - **`export_onnx.py`**: Utility to reliably export PyTorch checkpoints into statically compiled `.onnx` binaries for C++ execution, resolving dynamic sequence lengths and feature sizes.
   - Ensemble meta-learners and RL policies are exported to C++-compatible 3-logit ONNX graphs after successful training; RL also emits a native 10-action execution ONNX for optional two-model deployment.

5. **Live Execution (`infrastructure/cpp_server/`)**
   - A zero-allocation C++ execution server designed for ultra-low latency High-Frequency Trading (HFT).
   - **`OandaClient`**: Connects via REST/FIX to OANDA for live streaming and instant `MARKET` order execution. Pulls native UNIX timestamps to detect data staleness.
   - **`RingBuffer`**: An $O(1)$ circular array that maintains the last $N$ ticks and dynamically computes TA-Lib features (SMA, EMA, ATR) without memory reallocation.
   - **`RiskManager`**: A Tier-1 hardware-level circuit breaker enforcing "Fat Finger" position limits, daily drawdown maximums, and network latency "Stale Data" kill switches.
   - **`ModelRunner`**: Wraps the ONNX Runtime C++ API to execute exported deep learning models (HAELT, TFT) natively under 1 millisecond.
   - **Two-model execution**: Set `MODEL_PATH` to the ensemble/supervised direction ONNX and `EXECUTION_MODEL_PATH` to `rl_<algo>_execution.onnx` to let the 10-action RL policy manage open, scale, reduce, close, and hold decisions.
   - **`ParityChecker`**: A mathematical validation module that reads raw exported Python ticks and guarantees that C++ feature calculations match Python feature calculations to the 5th decimal place.
   - **Shadow Mode**: Compiler flag to run inference and calculate latency on live data without sending actual orders.


6. **Model Governance & Auditability**
   - **Model Cards**: Automatically generates a standardized <model_name>_model_card.json artifact for every trained model, tracking its architecture, training window, feature parity, label strategy, and cross-validation performance. This ensures strict tracking of model lineage and performance history prior to promotion.

7. **Semantic Observability**
   - **Discord Control Panel**: The Discord alerter enforces strict semantic alerting phases (e.g. 	raining_started, promotion_gate_passed, production_deploy_completed), directly embedding execution artifacts and ONNX schema hashes into the channel for a clean, auditable timeline.
\n## Prerequisites

- **Environment**: Python 3.10+
- **Hardware**: CUDA-capable GPU highly recommended for PyTorch pretraining and RL backpropagation. (Tested natively on Windows/WSL).
- **Core Dependencies**:
  - `torch`
  - `numpy`, `pandas`, `polars`
  - `pyfolio` (for tear sheet generation)

## Getting Started

1. **Generate/Ingest Data**
   Prepare tick-level or minute-level OHLCV bars via the data ingestion pipeline:
   ```python
   from data.data_ingestion import ForexDataPipeline
   ```

2. **Feature Engineering**
   Engineer your features to include regime labels:
   ```python
   from features.feature_engineering import FeatureEngineer
   ```

3. **Training & Automated Ablation**
   Kick off the training loops via `train_gpu.py`. The engine supports `haelt`, `transformer`, `tcn`, `mamba`, and other temporal architectures. The pipeline now features an **Automated Pretraining Ablation** sequence (`--pretrain-ablation auto`):
   - **Baseline Training**: First, it trains a complete supervised control model *without* pretraining.
   - **Contrastive Pretraining**: Next, it runs unsupervised representation learning to map deep market structures before looking at labels.
   - **Main Model Training**: It then trains the final model using the pretrained encoder weights.
   - **Ablation Reporting**: Finally, it automatically compares the Baseline and Main models, outputting a `pretrain_ablation.json` artifact to mathematically prove the performance gain from pretraining.

   The engine also includes an **Auto-Tuner** that evaluates the current run's metrics (overfitting, early stopping, gate rejections) and automatically adapts hyperparameters in `config/run.yaml` for the next training cycle.

4. **Backtesting**
   Test signals against the `ForexScalingBacktest` engine. Ensure you pass your `bars` (with bid/ask spread data) and generated `signals` containing `ScalingAction` states (e.g., `OPEN_LONG`, `SCALE_IN_50`).

## Changelog
See [CHANGELOG.md](CHANGELOG.md) for the latest project updates and version history.

## C++ Execution Server Setup (Visual Studio)

If you are deploying the C++ execution server natively on Windows using Visual Studio, follow these steps:

1. **Install Visual Studio 2022**: Ensure you have installed the "Desktop development with C++" workload.
2. **Install `vcpkg`**:
   ```cmd
   git clone https://github.com/Microsoft/vcpkg.git
   cd vcpkg
   bootstrap-vcpkg.bat
   vcpkg integrate install
   ```
3. **Install Dependencies**:
   ```cmd
   vcpkg install curl nlohmann-json onnxruntime
   ```
4. **Compile the Server**:
   Navigate to the `infrastructure/cpp_server/` directory, open the folder in Visual Studio (using the built-in CMake integration), and click "Build". Ensure you set your environment variables for `OANDA_ACCOUNT_ID` and `OANDA_BEARER_TOKEN`.
