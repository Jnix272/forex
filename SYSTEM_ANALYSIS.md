# System Analysis: Forex Scaling Model v6.5+

## 1. System Overview and Component Mechanics

The Forex Scaling Model is a robust, modular "tick-to-trade" machine learning and reinforcement learning (RL) pipeline. It aims to predict forex movements and execute trades while applying rigid quant-style governance and risk management protocols.

The system is highly decoupled and operates in six sequential phases:

### Phase 1: Data Pipeline (`data/`)
- **Ingestion & Compaction**: Fetches raw tick data (from Databento, Dukascopy, etc.) or generates synthetic ticks. It resamples the ticks into frequency-based bars (e.g., 1-minute), applies session filters (e.g., Asian/London/NY overlap), and handles fractional differencing for stationarity.
- **Economic Calendar & Macro Yields**: Integrates macroeconomic factors (like US10Y vs. G10 spreads and NFP/CPI events) using `EcoCalendarFeatureBuilder` and `MacroYieldFeatureBuilder`.

### Phase 2: Feature Engineering (`features/`)
- **Polars Core Engine**: Employs Polars for high-performance memory-safe data manipulation. Features remain in Polars DataFrames until they hit the machine learning model boundaries.
- **Advanced Features**: Builds micro-structure and momentum indicators such as Level 2 Order Book proxies, Tick Volume Imbalance (TVI), Hurst exponent, Fractal Dimensions, and cross-asset correlations. 
- **Sentiment Analysis**: Evaluates financial headlines using a 3-tier NLP pipeline (Ollama Mistral, FinBERT, and VADER).

### Phase 3: Labeling (`labeling/`)
- **Supervised Labels**: Implements the classic Triple-Barrier method (stop-loss, profit-target, and time horizons).
- **RL Reward Shaping**: For Reinforcement Learning agents, it uses a Differential Sharpe ratio optimizer (Moody & Saffell) to compute continuous or discrete rewards based on PnL distributions rather than static price targets.

### Phase 4: Models & Training (`models/`, `training/`)
- **Supervised Base Models**: Uses diverse architectures including Gradient Boosted Trees (CatBoost/XGBoost), Deep Learning (Temporal Convolution, Mamba, Multi-Timeframe Transformers), and Graph Neural Networks (Granger Causality GNN for cross-asset correlations).
- **Reinforcement Learning**: Employs PPO and DQN with advanced features like Curriculum Learning (progressing from low-volatility to full-data news events) and Hindsight Experience Replay (HER) to relabel failed trades.
- **Meta-Learner Ensemble**: A stacked MLP acts as a meta-learner dynamically weighting predictions from the sub-models based on market regimes.

### Phase 5: Backtesting & Validation (`backtesting/`, `validation/`, `monitoring/`)
- **Walk-Forward Testing**: Evaluates models on sequential out-of-sample data. It uses Monte Carlo and Slippage calibrators to provide robust confidence intervals.
- **Promotion Gate**: A deployed model must pass hard thresholds for out-of-sample Sharpe Ratio (>1.82), Maximum Drawdown, Profit Factor, and transaction cost margins before promotion.
- **Shadow Mode**: Evaluates candidate models against live deployed models on unseen bars. 

### Phase 6: Trading & Risk Execution (`trading/`, `risk/`, `sizing/`)
- **Execution Strategy**: Employs Almgren-Chriss policies to minimize market impact on large orders.
- **Risk Management**: Utilizes Regime-Conditional Kelly fraction sizing, Portfolio Value-at-Risk (VaR) enforcement, and Drawdown Aware Exit Policies to halt or size down trading based on equity curves.
- **Real-Time Monitoring**: Exports metrics via a Prometheus Exporter to Grafana, detects alpha decay using Page-Hinkley drift detection (`DemotionMonitor`), and alerts via Discord webhooks.

---

## 2. Discovered Bugs, Errors, and Inefficiencies

During the codebase analysis, several bugs, inefficiencies, and code smells were identified. These issues do not necessarily crash the system immediately but pose logical flaws or maintenance burdens.

### 2.1 Logical Bugs & Dead Code
- **`dashboard.py` (Dead Code / Unused Variable)**: 
  In the `run_interactive_backtest` function, at around line 181, there is a dead code statement where the ATR value is evaluated but not assigned to a variable or utilized.
  ```python
  cur_close = close[i]
  data["atr"][i]  # <-- BUG: Evaluated but completely ignored. Should be assigned to a variable if needed.
  cur_spread = data["spread"][i]
  ```
- **`main.py` (Global Warning Suppression)**:
  At the top of the file (line 16), all warnings are universally suppressed:
  ```python
  import warnings
  warnings.filterwarnings("ignore")
  ```
  This is a critical anti-pattern in a production machine learning system. It hides important `DeprecationWarnings`, `RuntimeWarnings` (e.g., divide by zero in pandas/numpy), and `UserWarnings` from PyTorch/Polars, making future debugging extremely difficult.

### 2.2 System Inefficiencies & Fragilities
- **`run_e2e_tests.py` (Fragile XML Cleanup)**:
  The script blindly attempts to remove `e2e_report.xml` with a generic `OSError` catch block. In concurrent CI/CD environments, this could introduce race conditions or mask permission issues. 
- **Data Boundary Handling**:
  While the architecture claims strict separation, there are instances where `Pandas` and `NumPy` are instantiated directly inside simulation loops (e.g., `visualize_backtest.py` creating DataFrames inside `run_simulated_backtest`). Relying on explicit `pd.Series()` loops inside Python `for` loops (e.g., in `dashboard.py`) degrades performance significantly compared to fully vectorized Polars or NumPy operations.
- **`api/main.py` Pydantic Validation Inefficiency**:
  The API manually validates floats for `NaN` and `Inf` using explicit `for` loops inside the endpoint.
  ```python
  for i, r in enumerate(payload.returns):
      if r is None or not math.isfinite(r):
          ...
  ```
  This O(N) loop blocks the FastAPI event loop for large arrays of returns. It would be significantly more efficient to cast to a NumPy array first and use `np.isfinite().all()` or leverage a custom Pydantic validator at the model level to handle this during deserialization.
- **Hardcoded Plotly Thresholds (`visualize_backtest.py`)**:
  The visualization script hardcodes a `min_confidence` line of `0.45` in `build_plotly_chart` which might not dynamically align with the model's actual calibrated probability thresholds, potentially leading to misleading UI interpretations of trade entries.
