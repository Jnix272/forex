import pandas as pd
import numpy as np
from training.dataset_builder import _compute_difficulty_scores

def test_compute_difficulty_scores_session_signal():
    # Create a DataFrame with datetime index covering a range of hours
    idx = pd.date_range('2023-01-01 00:00', periods=24, freq='h')
    df = pd.DataFrame(index=idx)
    # seq_len = 1, each bar's difficulty is based on its own signal
    diff = _compute_difficulty_scores(df, seq_len=1)
    expected = []
    for hour in idx.hour:
        if hour >= 21 or hour < 1:
            expected.append(2)  # rollover/hard
        elif 1 <= hour < 7:
            expected.append(1)  # asia/medium
        elif 18 <= hour < 21:
            expected.append(1)  # late NY/medium
        else:
            expected.append(0)  # easy
    assert diff.tolist() == expected
