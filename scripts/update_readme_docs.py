import re

with open("README.md", encoding="utf-8") as f:
    content = f.read()

# The huge changelog paragraph looks like:
# "**Version 6.7** fixes two critical training bottlenecks... **Version 6.6** improves..."
# Let's find it. It's in the second paragraph of the file.

def format_changelog_paragraph(text):
    # Find the big paragraph starting with "**Version"
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line.startswith("**Version 6.7**"):
            # This is the paragraph!
            # Replace "**Version X.Y** " with "\n### Version X.Y\n- "
            # and split sentences. But it's easier to just use regex on the whole paragraph.
            p = line

            # Split by "**Version "
            parts = re.split(r'\*\*(Version \d+\.\d+)\*\*', p)
            new_p = ""
            for j in range(1, len(parts), 2):
                version = parts[j]
                desc = parts[j+1].strip()
                # break desc into bullet points if it uses semicolons to list features
                desc_bullets = desc.replace('; ', '.\n- ').replace('; and ', '.\n- ').replace('; plus ', '.\n- ')
                if desc_bullets.startswith("-"):
                    desc_bullets = desc_bullets[1:].strip()
                new_p += f"\n### {version}\n- {desc_bullets}\n"

            lines[i] = new_p.strip()
            break

    return '\n'.join(lines)

content = format_changelog_paragraph(content)

new_sections = """
## Risk & execution

### Drift Gate Tuning
The `drift_gate` monitors real-time feature drift against the training distribution.
- **PSI Threshold (`psi_threshold`)**: Set to `0.2` by default. For high-volatility forex features (e.g. news impact or fast momentum), a threshold of `0.25 - 0.3` is more appropriate to avoid false positives during normal market session transitions.
- **KS p-value (`ks_pvalue_threshold`)**: Set to `0.05`. If the two-sample Kolmogorov-Smirnov test drops below this, the distributions differ significantly.
- **Operational Impact (`fail_open: false`)**: When catastrophic drift is detected, the pipeline will **halt live trading** to protect capital. You must investigate the shifted features and manually resume or retrain.

### Session Limits Definition
The configuration `risk.session_limits: london: {max_lots: 2.0, max_trades: 6}` governs trade entry caps.
- **Note on Scaling:** The `max_trades: 6` limit applies strictly to **complete trade entries** (opening a new base position).
- The RL agent's scaling actions (Scale In +25%, Scale Out -25%, etc.) do *not* consume the `max_trades` limit. 6 trades could encompass 1 entry + 5 scale-in/scale-out actions, but it will never exceed 6 distinct underlying positions.

## Model governance & promotion

### Continuous Training & Auto-Retrain
When significant regime shift or drift is detected, the pipeline may auto-trigger retraining.
- **Trigger**: Handled by the drift watcher or scheduled cron.
- **Lockfile**: It creates `checkpoints/retrain_in_progress.lock`.
- **Live Trading**: While the lockfile exists, the live trading engine continues operating seamlessly using the previously promoted checkpoint.
- **Promotion**: Once retraining finishes (usually 2-4 hours), the new model undergoes the promotion gates. If it passes, the symlink is updated, the lockfile is removed, and the live engine hot-reloads the new weights.

### Model Promotion Criteria
The promotion script does not just "pick the winner by Sharpe." It enforces strict minimum thresholds to ensure a model is actually viable before replacing the production baseline.
Promotion gates (all must pass):
- `val_sharpe > 0.5` (below = not promotable)
- `val_dir_acc > 0.38` (above random baseline of 0.33)
- `max_drawdown < 0.15` (on OOS backtest)
- `n_trades > 100` (enough to be statistically meaningful)

## Live trading

### OANDA Live Broker Setup
The live engine uses `--broker paper` by default. For live execution on OANDA:
1. **API Keys**: Add your credentials to `.env`:
   ```
   OANDA_API_KEY=your_v20_token
   OANDA_ACCOUNT_ID=xxx-xxx-xxxxxxx-xxx
   OANDA_ENV=practice  # Change to 'live' for real money
   ```
2. **Launch**: Start the engine with `python trading/live_engine.py --broker oanda`.
3. **Network**: Ensure your server's IP is allowlisted if required by your OANDA account settings.

## Feature engineering

### Fractional Differentiation
Forex raw price series are inherently non-stationary. Passing non-stationary data into deep learning architectures (especially long-sequence Mamba models) causes the State Space Model's hidden state to diverge over long sequences.
- **Preprocessing**: We apply Fractional Differentiation ($d \\approx 0.4$ for EUR/USD) to achieve stationarity while preserving maximum memory/momentum, applied *before* the feature pipeline.

### Options Implied Volatility (IV) Signals
The cross-asset and macro feature groups explicitly include Options market data.
- **Derived Volatility**: 1W ATM Implied Volatility and 25-delta risk reversals are pulled via the OANDA REST API.
- **Usage**: These are among the strongest short-term directional signals for scalping. We augment the ATR-based `slippage_vol_alpha` execution proxy with actual IV to drastically improve dynamic slippage prediction.

## Data sources

### COT Data Setup
The `cot_net` feature is populated in the `slow_cols` cache.
- **Source**: CFTC "Traders in Financial Futures" (TFF) reports.
- **CME Code**: Uses code `099741` for EUR/USD.
- **Alignment**: The weekly Friday prints are forward-filled daily across the tick data.
- **Warning**: If you do not run the COT download script to fetch the historical data, the pipeline will silently produce all-zero `cot_net` features.

## Installation & Hardware

### Data Loading Performance (`num_workers: 0`)
In the hardware constraints section, `num_workers: 0` is recommended for Windows stability to prevent multiprocessing deadlocks.
- **Impact**: This forces a single-threaded DataLoader on the massive 250K-chunk Zarr file. Data loading will be the bottleneck for the first 5-8 epochs until the OS page cache fully warms up.
- **Mitigation**: Setting `thread_prefetch_batches: 4` helps hide this latency, but users should expect early epochs to be significantly slower.

### WSL2 Data Pipeline Setup
For massive I/O improvements on Windows, moving the `data/` directory to the WSL2 ext4 filesystem is highly recommended.
1. Install WSL2 (Ubuntu).
2. Move the `data/` directory into your Linux home folder (e.g., `~/forex_scaling_model/data`).
3. Set your `.env` paths using the UNC format: `\\\\wsl.localhost\\Ubuntu\\home\\user\\forex_scaling_model\\data`.

## Experiment tracking & observability

### Knowledge Distillation Workflow
The distillation framework (Teacher $\\rightarrow$ Student) allows deploying lighter models.
- **When to distill**: After the large Mamba model converges, distill its logits into the HAELT architecture for faster inference during high-frequency deployment.
- **Direction**: Mamba (Teacher) $\\rightarrow$ HAELT (Student).
- **Temperature**: The default is 2.0, but for forex, a temperature of **3.0 - 4.0** often yields better student generalization when the teacher has high confidence on sparse news signals.

## Backtesting & validation

### Backtest Confidence Intervals
The `--mc_sims 500` flag runs 500 Monte Carlo simulations of the trade sequence to generate a Confidence Interval (CI) for the Sharpe ratio.
- **Interpretation**: A strategy with "Sharpe 0.8 $\\pm$ 0.3" (95% CI) is NOT the same as "Sharpe 0.8 $\\pm$ 0.05".
- **Rule**: Only promote models where the **lower bound** of the CI remains above `0.5`.

## Cloud deployment (RunPod)

### RunPod Setup Commands
Deployment automation is handled in the `deploy/` folder.
- Execute `python deploy/runpod_setup.py --template pytorch_2.1 --gpu RTX_4090` to automatically provision a pod, sync the `.venv-gpu` dependencies, and mount the WSL2 data volumes securely.
"""

content += "\n\n" + new_sections

with open("README.md", "w", encoding="utf-8") as f:
    f.write(content)
print("README updated and changelog formatted.")
