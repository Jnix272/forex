import numpy as np
import pandas as pd
import pytest

from data.point_in_time import align_asof_available, assert_point_in_time
from execution.realism import EmpiricalFillModel, realistic_utility_labels
from validation.rolling_retraining import rolling_retraining_splits, untouched_lockbox_indices


def _market(n=12):
    idx = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    close = np.full(n, 1.1000)
    close[2:] = 1.1010
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.0001,
            "low": close - 0.0001,
            "close": close,
            "bid_close": close - 0.00005,
            "ask_close": close + 0.00005,
        },
        index=idx,
    )
    feats = pd.DataFrame({"atr_6": 0.0002, "spread_pips": 1.0}, index=idx)
    return bars, feats


def test_labels_enter_after_execution_delay_and_emit_action_utilities():
    bars, feats = _market()
    out = realistic_utility_labels(
        bars,
        feats,
        lookahead_bars=3,
        execution_delay_bars=2,
        fill_model=EmpiricalFillModel(randomize=False, base_slippage_pips=0),
    )
    assert {"utility_long", "utility_hold", "utility_short", "optimal_side"} <= set(out)
    # The price jump occurs before the delayed fill, so it cannot become free profit.
    assert out.iloc[0].utility_long <= 0


def test_ambiguous_bar_is_resolved_conservatively_stop_first():
    bars, feats = _market()
    bars.iloc[1, bars.columns.get_loc("high")] = 1.102
    bars.iloc[1, bars.columns.get_loc("low")] = 1.098
    out = realistic_utility_labels(
        bars,
        feats,
        lookahead_bars=2,
        execution_delay_bars=1,
        fill_model=EmpiricalFillModel(randomize=False, base_slippage_pips=0),
    )
    assert out.iloc[0].utility_long < 0


def test_point_in_time_join_waits_until_available_time():
    idx = pd.date_range("2025-01-01 12:00", periods=3, freq="h", tz="UTC")
    obs = pd.DataFrame({"event_time": [idx[0]], "available_time": [idx[1]], "cpi": [3.1]})
    aligned = align_asof_available(idx, obs, value_columns=["cpi"])
    assert np.isnan(aligned.iloc[0, 0])
    assert aligned.iloc[1, 0] == 3.1
    assert_point_in_time(obs)
    with pytest.raises(ValueError):
        assert_point_in_time(obs.drop(columns="available_time"))


def test_rolling_splits_keep_purge_and_lockbox_untouched():
    splits = list(
        rolling_retraining_splits(
            100, retrain_every=10, validation_size=10, min_train_size=40, purge_bars=5, lockbox_size=20
        )
    )
    assert splits
    for train, val in splits:
        assert val[0] - train[-1] - 1 == 5
        assert val[-1] < 80
    assert untouched_lockbox_indices(100, 20)[0] == 80
