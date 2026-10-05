"""Regression tests for the feature-audit fixes: regime Hurst input, rolling HMM
regime_class, sentiment decay, COT staleness and CFTC contract mapping."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _price_path(n: int, seed: int = 0, vol_blocks: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sig = np.full(n, 1e-4)
    if vol_blocks:
        block = 3000
        for b in range(n // block + 1):
            sig[b * block : (b + 1) * block] = (0.5e-4, 1e-4, 3e-4)[b % 3]
    return 1.1 * np.exp(np.cumsum(rng.normal(0.0, 1.0, n) * sig))


def test_regime_label_not_constant_on_price_levels():
    from features.regime_detection import detect_regimes_polars

    out = detect_regimes_polars(pl.DataFrame({"close": _price_path(6000)}), step=5)
    h = out["hurst_dfa"].to_numpy()[200:]
    assert 0.35 < float(np.median(h)) < 0.7
    assert len(set(out["regime_label"].to_list())) == 3


def test_regime_class_ordered_by_volatility_and_all_states_used():
    pytest.importorskip("hmmlearn")
    from features.regime_detection import detect_regimes_polars

    close = _price_path(18000, seed=1, vol_blocks=True)
    out = detect_regimes_polars(pl.DataFrame({"close": close}))
    rc = out["regime_class"].to_numpy()
    absret = np.abs(np.diff(np.log(close), prepend=np.log(close[0])))
    tail = slice(3000, None)
    counts = np.bincount(rc[tail], minlength=3)
    assert (counts > 0).all(), counts
    means = [absret[tail][rc[tail] == k].mean() for k in range(3)]
    assert means[0] < means[1] < means[2], means


def test_causal_hmm_decode_is_causal():
    pytest.importorskip("hmmlearn")
    from features.regime_detection import _causal_hmm_decode

    rng = np.random.default_rng(3)
    feat = np.column_stack([rng.normal(size=5000), np.abs(rng.normal(size=5000))])
    p_full, s_full = _causal_hmm_decode(feat, 3, skip_head=60)
    p_cut, s_cut = _causal_hmm_decode(feat[:3600], 3, skip_head=60)
    np.testing.assert_allclose(p_full[:3600], p_cut, atol=1e-9)
    np.testing.assert_array_equal(s_full[:3600], s_cut)


def _bars(start: datetime, n: int, minutes: int = 5) -> pl.DataFrame:
    ts = [start + timedelta(minutes=minutes * i) for i in range(n)]
    return pl.DataFrame({"timestamp_utc": ts}).with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))


def test_sentiment_decays_from_event_time():
    from features.engineering.core import _join_asof_available
    from features.engineering.sentiment import sentiment_tiers

    t0 = datetime(2024, 1, 2, 0, 0, tzinfo=UTC)
    bars = _bars(t0, 12 * 24)
    events = pl.DataFrame(
        {"timestamp_utc": [t0, t0 + timedelta(hours=12)], "sentiment": [0.8, -0.4]}
    ).with_columns(pl.col("timestamp_utc").cast(pl.Datetime("ns", "UTC")))
    F = _join_asof_available(bars, events, keep_available_as="_sentiment_event_time")
    out = sentiment_tiers(F, decay_lam=0.1, fb_dim=0)
    raw = out["sentiment_raw"].to_numpy()
    dec = out["sentiment_decayed"].to_numpy()
    assert not np.allclose(raw, dec)
    ts = out["timestamp_utc"].to_list()
    ev_time = out["_sentiment_event_time"].to_list()
    i = next(k for k, t in enumerate(ts) if t >= t0 + timedelta(hours=6))
    hours = (ts[i] - ev_time[i]).total_seconds() / 3600.0
    assert dec[i] == pytest.approx(raw[i] * np.exp(-0.1 * hours))
    j = next(k for k, t in enumerate(ts) if ev_time[k] is not None and ev_time[k] > t0 + timedelta(hours=11))
    assert raw[j] == pytest.approx(-0.4)
    assert dec[j] == pytest.approx(raw[j], rel=0.02)


def test_sentiment_decay_without_event_time_uses_value_changes():
    from features.engineering.sentiment import sentiment_decay

    t0 = datetime(2024, 1, 2, tzinfo=UTC)
    df = _bars(t0, 4, minutes=60).with_columns(pl.Series("sentiment_raw", [0.5, 0.5, 0.5, -0.2]))
    dec = sentiment_decay(df, lam=0.1)
    np.testing.assert_allclose(dec, [0.5, 0.5 * np.exp(-0.1), 0.5 * np.exp(-0.2), -0.2])


def _cot(pair: str, releases: list[datetime], values: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp_utc": releases,
            "pair": [pair] * len(releases),
            "net_hedge_fund": values,
            "net_commercial": [-v for v in values],
        }
    ).with_columns(pl.col("timestamp_utc").cast(pl.Datetime("us", "UTC")))


def test_cot_zeroed_before_first_report_and_after_series_goes_stale():
    from features.cot_features import add_cot_features

    first = datetime(2024, 1, 5, 20, 30, tzinfo=UTC)
    releases = [first + timedelta(weeks=k) for k in range(6)]
    cot = _cot("GBPUSD", releases, [1000.0 + k for k in range(6)])
    bars = _bars(first - timedelta(days=2), 24 * 60, minutes=60)
    out = add_cot_features(bars, cot, "GBPUSD")
    ts = out["timestamp_utc"].to_list()
    hf = out["cot_net_hf"].to_numpy()
    last = releases[-1]
    before = np.array([t < first for t in ts])
    fresh = np.array([first <= t <= last + timedelta(days=21) for t in ts])
    stale = np.array([t > last + timedelta(days=21) for t in ts])
    assert (hf[before] == 0).all()
    assert (hf[fresh] > 0).all()
    assert stale.any() and (hf[stale] == 0).all()
    assert "_cot_release" not in out.columns


def test_download_cot_maps_renamed_contracts_by_code():
    import importlib.util

    spec = importlib.util.spec_from_file_location("download_cot", ROOT / "scripts" / "download_cot.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    df = pd.DataFrame(
        {
            "Market_and_Exchange_Names": [
                "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE",
                "NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE",
                "SOMETHING ELSE",
                "EURO FX - CHICAGO MERCANTILE EXCHANGE",
            ],
            "CFTC_Contract_Market_Code": ["096742", 112741, "999999", " 99741"],
        }
    )
    assert mod._map_pairs(df).tolist()[:2] == ["GBPUSD", "NZDUSD"]
    assert pd.isna(mod._map_pairs(df).iloc[2])
    assert mod._map_pairs(df).iloc[3] == "EURUSD"
