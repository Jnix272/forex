"""
Tests for multi-modal sentiment (Improvement #3):
financial NER, lexicon scoring, topic modeling, bar fusion.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from features.sentiment_fusion import (
    NER_CATEGORIES,
    build_sentiment_features,
    financial_ner_counts,
    fit_topic_model,
    lexicon_score,
)

_HEADLINES = [
    "ECB hikes rates EUR/USD surges to 18-month high",
    "US Dollar falls as NFP misses expectations, recession fears grow",
    "Fed holds rates steady, markets calm",
    "BOJ signals dovish stance, USD/JPY weakens",
    "CPI inflation accelerates, hawkish comments from FOMC members",
    "GBP/USD rallies after strong UK GDP growth data",
]


def _bars(n=24, start=datetime(2024, 1, 1, tzinfo=UTC)):
    return [start + timedelta(hours=i) for i in range(n)]


def _events(n_ev=3):
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "timestamp_utc": [
                start + timedelta(hours=1),
                start + timedelta(hours=2),
                start + timedelta(hours=1, minutes=30),
            ],
            "source": ["news", "news", "social"],
            "text": _HEADLINES[:3],
        }
    )


# ---------------------------------------------------------------------------
# Financial NER
# ---------------------------------------------------------------------------


def test_ner_detects_known_categories():
    out = financial_ner_counts(_HEADLINES)
    assert out.shape == (len(_HEADLINES), len(NER_CATEGORIES))
    # ECB hikes rates EUR/USD
    assert out[0, NER_CATEGORIES.index("rate_hike")] >= 1
    assert out[0, NER_CATEGORIES.index("pair_mentions")] >= 1
    assert out[0, NER_CATEGORIES.index("cb_mentions")] >= 1
    # NFP miss
    assert out[1, NER_CATEGORIES.index("nfp")] >= 1
    # Fed holds
    assert out[2, NER_CATEGORIES.index("rate_hold")] >= 1
    assert out[2, NER_CATEGORIES.index("cb_mentions")] >= 1
    # BOJ dovish USD/JPY
    assert out[3, NER_CATEGORIES.index("dovish")] >= 1
    assert out[3, NER_CATEGORIES.index("pair_mentions")] >= 1
    # CPI + hawkish
    assert out[4, NER_CATEGORIES.index("cpi")] >= 2
    assert out[4, NER_CATEGORIES.index("hawkish")] >= 1


def test_ner_empty_and_neutral():
    out = financial_ner_counts(["", "just some neutral text"])
    assert (out == 0).all()


def test_lexicon_score_ranges_and_direction():
    sc = lexicon_score(_HEADLINES)
    assert np.all((sc >= -1.0) & (sc <= 1.0))
    assert sc[0] > 0.0  # EUR/USD surges
    assert sc[1] < 0.0  # US Dollar falls, NFP miss
    assert abs(sc[2]) < 0.5  # Fed holds = neutral-ish


# ---------------------------------------------------------------------------
# Topic model
# ---------------------------------------------------------------------------


def test_topic_model_weights_sum_to_one():
    W, top = fit_topic_model(_HEADLINES, n_topics=3)
    assert W.shape[0] == len(_HEADLINES)
    assert W.shape[1] == 3
    assert np.allclose(W.sum(axis=1), 1.0)
    assert all(len(t) <= 8 for t in top)


def test_topic_model_deterministic():
    a, _ = fit_topic_model(_HEADLINES, n_topics=3, seed=0)
    b, _ = fit_topic_model(_HEADLINES, n_topics=3, seed=0)
    assert np.allclose(a, b, atol=1e-9)


def test_topic_model_empty_corpus():
    W, top = fit_topic_model([], n_topics=3)
    assert W.shape == (0, 3)
    assert len(top) == 3


# ---------------------------------------------------------------------------
# Bar fusion
# ---------------------------------------------------------------------------


def test_build_sentiment_features_counts_by_source():
    F = build_sentiment_features(_events(), _bars(), lam=0.2, dt_sec=3600, n_topics=3)
    assert len(F) == 24
    # event at hour 1 (news) lands at bar 1
    assert F["sent_news_count"].to_list()[1] > 0.5
    # social event at hour 1.5 lands at bar 1
    assert F["sent_social_count"].to_list()[1] > 0.5
    assert F["sent_news_count"].to_list()[2] > 0.5  # hour-2 news
    assert F["sent_modalities"].to_list()[1] >= 2.0


def test_build_sentiment_features_ner_columns():
    F = build_sentiment_features(_events(), _bars(), lam=0.2, dt_sec=3600, n_topics=3)
    for cat in NER_CATEGORIES:
        assert f"ner_{cat}" in F.columns
    assert F["ner_rate_hike"].to_list()[1] >= 1.0
    assert F["ner_pair_mentions"].to_list()[1] >= 1.0


def test_build_sentiment_features_fusion_with_cot():
    cot = pl.Series(np.random.default_rng(0).normal(size=24))
    F = build_sentiment_features(_events(), _bars(), lam=0.2, dt_sec=3600, n_topics=3, cot_z=cot)
    assert "sent_cot" in F.columns
    assert F["sent_modalities"].to_list()[1] >= 3.0
    assert np.all(np.isfinite(F["sent_fused"].to_numpy()))
    assert np.all((F["sent_agreement"].to_numpy() >= 0.0) & (F["sent_agreement"].to_numpy() <= 1.0))


def test_build_sentiment_features_no_events():
    bars = _bars()
    empty = pl.DataFrame({"timestamp_utc": [], "text": []})
    F = build_sentiment_features(empty, bars, n_topics=3)
    assert len(F) == len(bars)
    assert F["sent_news_count"].sum() == 0.0
    assert F["sent_fused"].sum() == 0.0


def test_build_sentiment_features_all_zero_events():
    bars = _bars()
    zero = pl.DataFrame(
        {
            "timestamp_utc": [bars[0]],
            "source": ["news"],
            "text": ["nothing relevant here"],
        }
    )
    F = build_sentiment_features(zero, bars, lam=0.2, dt_sec=3600, n_topics=3)
    assert F["sent_news_count"].to_list()[0] == 1.0
    assert np.isfinite(F["sent_news"].to_numpy()).all()


def test_build_sentiment_features_lexicon_fallback_when_no_sent_col():
    ev = _events().drop("sentiment") if "sentiment" in _events().columns else _events()
    F = build_sentiment_features(ev, _bars(), lam=0.2, dt_sec=3600, n_topics=3)
    assert "sent_news" in F.columns


def test_add_sentiment_features_appends_to_bars():
    from features.sentiment_fusion import add_sentiment_features

    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = pl.DataFrame(
        {
            "timestamp_utc": [start + timedelta(hours=i) for i in range(6)],
            "close": [1.0 + 0.01 * i for i in range(6)],
        }
    )
    ev = pl.DataFrame(
        {
            "timestamp_utc": [start + timedelta(hours=1)],
            "source": ["news"],
            "text": ["ECB hikes rates EUR/USD surges"],
        }
    )
    out = add_sentiment_features(bars, ev, lam=0.2, dt_sec=3600, n_topics=2)
    assert "sent_news_count" in out.columns
    assert "ner_rate_hike" in out.columns
    assert out["close"].to_list() == bars["close"].to_list()
