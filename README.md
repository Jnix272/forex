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

4. **Inference & Governance (`inference/`, `retraining/`)**
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
   - **Discord Control Panel**: The Discord alerter enforces strict semantic alerting phases (e.g. `training_started`, `promotion_gate_passed`, `production_deploy_completed`), directly embedding execution artifacts and ONNX schema hashes into the channel for a clean, auditable timeline.

## Prerequisites

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

## Data Pipeline & CLI (automatic)

The end-to-end data lifecycle is one command — downloads, consolidated-DuckDB
migration, feature engineering, training, and backtesting are automated and
idempotent. Use `./run.sh` (Linux/macOS) or `.\run.ps1` (Windows), which wrap
`scripts/run_pipeline.py`:

```bash
./run.sh download --start 2008-01-01   # prices -> news -> COT -> calendar,
                                       # then auto-runs the DuckDB migration
./run.sh migrate                       # refresh consolidated DuckDB only
./run.sh validate                      # runs data quality checks on consolidated DB
./run.sh data                          # download -> migrate -> validate
./run.sh train                         # training (auto-migrates first if stale)
./run.sh backtest                      # backtest (auto-migrates first if stale)
./run.sh all                           # download -> migrate -> validate -> train -> backtest
```

- **Automatic DuckDB migration** (`scripts/migrate_to_duckdb.py`): consolidates
  the compacted tick parquet into `data/store/forex_ticks.duckdb`. It appends new
  pairs, refreshes re-downloaded pairs (DELETE + INSERT), and skips unchanged
  pairs in a ~0.5s no-op, tracked in `data/store/.migrate_manifest.json`.
- The migration is auto-triggered after `download`, as part of `data`/`all`, and as
  a pre-flight check before `train`/`backtest` when the DB is stale.
  Disable with `--skip-migrate` or `--no-auto-migrate`.
- **Secondary indexes**: DuckDB's native row-group min/max zonemaps automatically accelerate `WHERE pair AND timestamp` filters on the 1.74B-row table. Secondary ART indexes (`idx_ticks_pair`, `idx_ticks_ts`, `idx_ticks_pair_ts`) were evaluated but failed commit on this 14 GB RAM configuration regardless of `memory_limit`. The zonemap approach is lower‑overhead and the recommended path for this dataset.

## Latest Updates (2026-08-13)

### Data Building Pipeline Improvements (2026-08-13)
- **P1**: Unified Data Contracts with Pydantic validation for tick/bar/feature/label/dataset stages - 5 stage-specific contracts with schema hashing, column constraints, and SQL-expression invariants
- **P2**: Data Lineage tracking with LineageTracker, FileLineageStore/SQLiteLineageStore, EventType enum, and graph reconstruction
- **P3**: Incremental/Streaming Feature Computation with IncrementalFeatureEngine featuring EMA/rolling state, StreamingFeatureProcessor with warmup, and FeatureStateStore with Redis fallback
- **P4**: Automated Data Quality Gates with 12 quality checks (nulls, infinities, duplicates, monotonicity, weekends, bid/ask, OHLC, spread, variance, correlations) and 11 remediation actions with auto-remediation
- **P5**: Feature Store Integration with ParquetFeatureStore (partitioned storage), FeatureVersion metadata, FeatureRegistry with categorization/deprecation, and FeatureMaterializer orchestrating full pipeline
- **P6**: Configuration-Driven Pipeline with PipelineConfig hierarchical dataclasses from YAML, PipelineOrchestrator sequential stages with validation/quality/gates/drift/lineage, SchemaDriftDetector PSI-based detection, and ValidationReporter JSON+HTML

### Test Suite Stabilization & Bug Fixes (2026-08-09)
- Critical bugs fixed in iTransformer, ClusterContrastiveTrainer, ZarrStreamDataset, promotion gates, visualizer, and supervised loop
- ✅ **The `supervised_loop.py` epoch loop is repaired** (2026-08-09): stripped all legacy adaptive-curriculum references, routed through `CurriculumManager`, wired `graph_pgd` auto-select for GNN models, applied per-sample curriculum weights to the loss, and switched `OneCycleLR` to `total_steps` mode. Training smoke test passes.
- Fixed `iTransformerScalper` LayerNorm shape mismatch when used with `MultiTaskWrapper`
- Fixed `ClusterContrastiveTrainer.nt_xent` undefined variables
- Fixed `_fit_fold_scaler` and `_decompress_block` StandardScaler 3D input handling
- Fixed `ZarrStreamDataset` worker block splitting and scalar index handling
- Fixed `_close_position` override_price parameter for stop/TP execution in backtest
- Fixed `supervised_loop.py` indentation error in epoch loop

### Data Quality & Observability (2026-08-09)
- Structured logging infrastructure across entire data pipeline (replaces 20k+ bare print statements)
- Cross-asset per-asset logging with full fallback chain visibility
- FRED dual-path visibility - explicit logging of real vs synthetic data
- COT load unification - single helper with logging for main + worker paths
- Regime detection structured logging - success vs fallback tracking
- Pipeline standardization fix - resolves `ColumnNotFoundError: unable to find column "mid"`

### Phase 3 Architectural Replacements (2026-08-08)
- **P3-1**: Replace AdversarialGenerator with PGD/FGSM/FreeLB gradient-based attacks — classes + factory done; `supervised_loop` wiring (`graph_pgd`) **pending**
- **P3-2**: Migrate Curriculum to Composer/Lightning callbacks
- **P3-3**: Migrate Pretraining to lightly-ssl / Solo-learn adapters
- **P3-4**: Migrate RL to CleanRL / Stable-Baselines3 adapters
- **P3-5**: Export scaler in ONNX graph (single artifact with fused normalization)

### Seven Critical Audit Bugs Fixed (2026-08-08)
- **R-1/R-2**: Parametric VaR covariance scaling (was wrong by ~10,000×)
- **P1**: Promotion gate silent-default (net_pnl as gross_pnl, costs=0)
- **A8/A9**: Causal conv padding (symmetric → asymmetric left-only)
- **I3**: PPO greedy inference flag for deterministic evaluation
- **A4**: Positional encoding for Transformer branches
- **EWC**: Fisher diagonal normalization (divide by samples_processed)
- **RA2**: HER self-match (strict future sampling)

### EWC / Synaptic Intelligence Setup Fixes (2026-08-11)
- **YAML wiring**: `training.enable_ewc/ewc_lambda/enable_si/si_lambda` now map through `_YAML_MAP` (previously silently dropped)
- **SI raw-gradient path integral**: importance Ω computed from unclipped/uncentralized backward gradients, not the clipped/centralized surrogate
- **`ewc_lambda` fallback aligned to 1000.0** (was 100.0 in two spots)
- **Regime-drift SI λ**: `si_lambda` scaled per epoch by `1/(1 + max_shift²)` from the FeatureStabilityMonitor (λ↘ on regime shocks, re-locks when the regime stabilizes)
- **Dead EWC import removed** from `train_gpu.py`; `--enable-ewc` on a fresh run now prints a deferred notice instead of silently no-oping

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
