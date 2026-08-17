"""
scripts/continuous_finetune.py
==============================
Automates the 'Rolling Window' Continuous Fine-Tuning pipeline.
Wakes up at a specified UTC hour (e.g., NY Close), downloads the last N days
of price and news data, and fine-tunes the Foundation Model for 1 epoch.

Usage:
  python scripts/continuous_finetune.py --daemon
  python scripts/continuous_finetune.py --lookback-days 1 --run-time 17:00
"""

import argparse
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import active_checkpoint_dir

DEFAULT_CONFIG = _ROOT / "config" / "run.yaml"


def get_python_exe():
    from scripts._python_env import python_exe as _resolve

    return _resolve()


def _label_horizon_bars() -> int:
    """Base LH x max LABEL_REGIME horizon mults + delay (matches cv_splits floor)."""
    try:
        from config.settings import LABELING
        from labeling.rl_reward_labeling import max_label_horizon_mult

        lookahead = int(LABELING.get("lookahead_bars", 30))
        delay = int(LABELING.get("execution_delay_bars", 1))
        # Embargo must cover longest regime/session-scaled label horizon.
        return max(1, int(lookahead * max_label_horizon_mult())) + delay
    except Exception:
        return 16


def _embargo_bars_from_config(cfg: dict) -> int:
    training = cfg.get("training") or {}
    seq_len = int(training.get("seq_len", 60))
    return seq_len + _label_horizon_bars()


def _assert_price_data_fresh(pair: str, window_end: datetime, max_stale_days: int = 5) -> None:
    """Fail fast if downloaded ticks do not reach the label-safe window end."""
    from data.sources import ForexDataManager

    probe_start = (window_end - timedelta(days=3)).strftime("%Y-%m-%d")
    probe_end = window_end.strftime("%Y-%m-%d")
    mgr = ForexDataManager(verbose=False)
    df = mgr.load(
        pair=pair,
        source="dukascopy",
        start=probe_start,
        end=probe_end,
        session_only=False,
    )
    if df is None or len(df) == 0:
        raise RuntimeError(f"No price data for {pair} in {probe_start}..{probe_end} after download")
    ts_col = "timestamp_utc" if "timestamp_utc" in df.columns else None
    if ts_col:
        last_ts = pd.Timestamp(df[ts_col].max()).tz_convert("UTC")
    else:
        last_ts = pd.Timestamp(df.index.max()).tz_convert("UTC")
    min_ok = window_end - timedelta(days=max_stale_days)
    if last_ts < min_ok:
        raise RuntimeError(
            f"Stale price data for {pair}: last tick {last_ts.isoformat()} "
            f"< required {min_ok.isoformat()} (window end {window_end.isoformat()})"
        )
    print(f"[Freshness] {pair} last tick {last_ts.isoformat()} (ok for window end {window_end.date()})")


def write_temp_finetune_config(
    base_config: Path,
    start_date: str,
    end_date: str,
    epochs: int = 1,
) -> Path:
    """Write a temporary YAML for fine-tuning without mutating config/run.yaml."""
    with open(base_config, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    config.setdefault("data", {})["start"] = start_date
    config["data"]["end"] = end_date
    config.setdefault("training", {})["epochs"] = epochs
    config["training"]["resume"] = False
    config["training"]["val_split"] = 0.05
    config.setdefault("walk_forward", {})["enabled"] = False
    tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w",
        suffix="_finetune.yaml",
        delete=False,
        dir=str(_ROOT / "config"),
    )
    yaml.dump(config, tmp, sort_keys=False)
    tmp.close()
    return Path(tmp.name)


def run_pipeline(pair: str, lookback_days: int):
    """Executes download + warm-start fine-tune on a label-safe window."""
    now_utc = datetime.now(UTC)
    horizon = _label_horizon_bars()
    embargo = _embargo_bars_from_config(yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")) or {})
    end_dt = now_utc - timedelta(minutes=horizon + embargo)
    start_dt = end_dt - timedelta(days=lookback_days)

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    python_exe = get_python_exe()

    print(f"\n[{now_utc.isoformat()}] Starting Continuous Fine-Tuning Pipeline")
    print(
        f"Target Window: {start_str} -> {end_str} for pair {pair} "
        f"(end trimmed by label_horizon={horizon} + embargo={embargo} bars)\n"
    )

    print(">>> 1. Fetching Price Data (OHLCV) ...")
    cmd_price = [
        python_exe,
        "scripts/download_data.py",
        "--pairs",
        pair,
        "--start",
        start_str,
        "--end",
        end_str,
        "--no-cross-asset",
    ]
    subprocess.run(cmd_price, cwd=str(_ROOT), check=True)
    _assert_price_data_fresh(pair, end_dt)

    print("\n>>> 2. Fetching Macroeconomic News ...")
    cmd_news = [
        python_exe,
        "scripts/download_historical_news.py",
        "--pairs",
        pair,
        "--start",
        start_str,
        "--end",
        end_str,
        "--append",
    ]
    subprocess.run(cmd_news, cwd=str(_ROOT), check=True)

    print("\n>>> 3. Initiating Transfer Learning (PyTorch) ...")
    tmp_cfg = write_temp_finetune_config(DEFAULT_CONFIG, start_str, end_str, epochs=1)
    try:
        cmd_train = [
            python_exe,
            "training/train_gpu.py",
            "--config",
            str(tmp_cfg),
            "--finetune-warm-start",
        ]
        flag = active_checkpoint_dir(DEFAULT_CONFIG) / "needs_retrain.flag"
        if flag.exists():
            print(f"[Finetune] Consuming {flag.name}")
            try:
                flag.unlink()
            except OSError:
                pass
        subprocess.run(cmd_train, cwd=str(_ROOT), check=True)
    finally:
        try:
            tmp_cfg.unlink(missing_ok=True)
        except OSError:
            pass

    print(f"\n[{datetime.now(UTC).isoformat()}] Fine-Tuning Pipeline Complete!")


def main():
    parser = argparse.ArgumentParser(description="Rolling Window Fine-Tuner")
    parser.add_argument("--pair", type=str, default="EURUSD", help="Forex pair to tune")
    parser.add_argument("--lookback-days", type=int, default=1, help="Days of history to tune on")
    parser.add_argument("--daemon", action="store_true", help="Run continuously and wake up daily")
    parser.add_argument("--run-time", type=str, default="17:00", help="UTC time to execute (HH:MM)")

    args = parser.parse_args()

    if not args.daemon:
        run_pipeline(args.pair, args.lookback_days)
        return

    target_hour, target_minute = map(int, args.run_time.split(":"))
    print(f"Daemon started. Waking up daily at {args.run_time} UTC to fine-tune {args.pair}.")

    while True:
        now_utc = datetime.now(UTC)
        target_time = now_utc.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)

        if now_utc >= target_time:
            target_time += timedelta(days=1)

        sleep_seconds = (target_time - now_utc).total_seconds()
        print(
            f"[{now_utc.isoformat()}] Sleeping for {sleep_seconds / 3600:.2f} hours until {target_time.isoformat()} ..."
        )
        time.sleep(sleep_seconds)

        try:
            run_pipeline(args.pair, args.lookback_days)
        except Exception as e:
            print(f"Pipeline failed: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
