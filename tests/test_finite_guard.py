from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infrastructure.numerics import ensure_finite_tensor, sanitize_array, sanitize_frame


def test_sanitize_frame_replaces_nan_and_inf():
    df = pd.DataFrame({"a": [1.0, np.inf], "b": [np.nan, -np.inf]})
    out = sanitize_frame(df, fill_value=0.0, context="test")
    assert np.isfinite(out.values).all()
    assert out.iloc[1, 0] == 0.0


def test_sanitize_array_replaces_nan_and_inf():
    arr = np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    out = sanitize_array(arr, fill_value=0.0, context="test")
    assert np.isfinite(out).all()


def test_ensure_finite_tensor_raises_on_non_finite():
    torch = pytest.importorskip("torch")
    with pytest.raises(RuntimeError, match="Non-finite"):
        ensure_finite_tensor(torch.tensor(float("nan")), context="loss")
