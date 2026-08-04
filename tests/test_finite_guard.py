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


def test_sanitize_array_coerces_empty_strings():
    """Empty-string missing values must not raise (Dukascopy mixed columns)."""
    arr = np.array([["1.0", ""], ["2.5", "nan"]], dtype=object)
    out = sanitize_array(arr, fill_value=0.0, context="test")
    assert out.shape == (2, 2)
    assert np.isfinite(out).all()
    assert out[0, 0] == pytest.approx(1.0)
    assert out[0, 1] == 0.0


def test_sanitize_frame_skips_price_columns():
    df = pd.DataFrame({
        "close": [150.0, 151.0],
        "feat": [100.0, np.nan],
    })
    out = sanitize_frame(df, fill_value=0.0, context="test")
    assert out["close"].tolist() == [150.0, 151.0]
    assert out["feat"].iloc[0] == 20.0  # clipped
    assert out["feat"].iloc[1] == 0.0


def test_ensure_finite_tensor_raises_on_non_finite():
    torch = pytest.importorskip("torch")
    with pytest.raises(RuntimeError, match="Non-finite"):
        ensure_finite_tensor(torch.tensor(float("nan")), context="loss")
