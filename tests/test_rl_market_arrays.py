"""RL market cache + env smoke tests (no full training run)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_rl_env_nonzero_pnl_on_price_moves():
    from models.rl_agents import ForexTradingEnv, ScalingAction

    n = 80
    prices = np.linspace(1.0800, 1.0900, n, dtype=np.float32)
    atr = np.full(n, 0.0010, dtype=np.float32)
    spreads = np.full(n, 0.00008, dtype=np.float32)
    feats = np.zeros((n, 4), dtype=np.float32)

    env = ForexTradingEnv(
        features=feats,
        prices=prices,
        atr=atr,
        spreads=spreads,
        random_reset=False,
        episode_len=n - 2,
    )
    env.reset()
    env.start_idx = 0
    env.end_idx = n - 2
    env.idx = 0

    total_pnl = 0.0
    for _ in range(n - 3):
        action = ScalingAction.OPEN_LONG.value if env.idx < (n - 3) // 2 else ScalingAction.OPEN_SHORT.value
        _, _, done, info = env.step(action)
        total_pnl += float(info.get("pnl", 0.0))
        if done:
            break

    assert total_pnl != 0.0, "expected non-zero realised PnL when prices trend"


def test_market_bar_arrays_align_with_sequences():
    import pandas as pd
    from features.feature_engineering import FeatureEngineer
    from training.train_gpu import _market_bar_arrays_from_feats

    n = 120
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    feats = pd.DataFrame(
        {
            "close": np.linspace(1.08, 1.09, n),
            "atr_6": np.full(n, 0.0008),
            "spread_pips": np.full(n, 0.8),
        },
        index=ts,
    )
    feats.index.name = "timestamp_utc"
    X = feats.iloc[10:].copy()
    fe = FeatureEngineer(atr_window=6)
    seq_len = 10
    close_seq, atr_seq, spread_seq = _market_bar_arrays_from_feats(feats, X.index, fe, seq_len)
    assert len(close_seq) == len(X) - seq_len + 1
    assert close_seq[-1] > close_seq[0]
    assert atr_seq.std() >= 0
    assert spread_seq.min() > 0


def test_require_rl_market_cache_raises_without_arrays(tmp_path):
    from training.train_gpu import _require_rl_market_cache
    import pytest

    fake = str(tmp_path / "missing.zarr")
    with pytest.raises(RuntimeError, match="rebuild-cache"):
        _require_rl_market_cache(fake)


if __name__ == "__main__":
    test_rl_env_nonzero_pnl_on_price_moves()
    test_market_bar_arrays_align_with_sequences()
    print("OK: rl market tests passed")
