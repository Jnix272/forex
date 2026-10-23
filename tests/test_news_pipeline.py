from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from infrastructure.news_pipeline import StreamingNewsEnricher


@pytest.fixture
def enricher():
    with (
        patch("infrastructure.news_pipeline.SentimentPipeline"),
        patch("infrastructure.news_pipeline.KafkaConsumer"),
        patch("infrastructure.news_pipeline.KafkaProducer"),
    ):
        enricher_instance = StreamingNewsEnricher.__new__(StreamingNewsEnricher)
        enricher_instance.event_classifier = None
        yield enricher_instance


@pytest.mark.parametrize(
    "keyword",
    [
        "nfp",
        "NONFARM PAYROLLS",
        "CPI",
        "Inflation",
        "FOMC",
        "Rate Decision",
        "FED",
        "ECB",
    ],
)
def test_is_high_impact_fast_path_keywords(enricher, keyword):
    headline = f"Breaking news regarding {keyword} expected today."
    assert enricher.is_high_impact(headline) is True


def test_is_high_impact_no_keyword_no_classifier(enricher):
    headline = "Stock market opens flat on quiet trading day"
    assert enricher.is_high_impact(headline) is False


def test_is_high_impact_classifier_high_confidence(enricher):
    mock_classifier = MagicMock()
    mock_classifier.return_value = {
        "labels": ["macroeconomic policy decision", "routine market news"],
        "scores": [0.85, 0.15],
    }
    enricher.event_classifier = mock_classifier

    headline = "Central bank hints at unexpected monetary policy shift"
    assert enricher.is_high_impact(headline) is True
    mock_classifier.assert_called_once_with(
        headline, candidate_labels=["macroeconomic policy decision", "routine market news"]
    )


def test_is_high_impact_classifier_low_confidence(enricher):
    mock_classifier = MagicMock()
    mock_classifier.return_value = {
        "labels": ["macroeconomic policy decision", "routine market news"],
        "scores": [0.55, 0.45],
    }
    enricher.event_classifier = mock_classifier

    headline = "Some ambiguous financial update"
    assert enricher.is_high_impact(headline) is False


def test_is_high_impact_classifier_other_top_label(enricher):
    mock_classifier = MagicMock()
    mock_classifier.return_value = {
        "labels": ["routine market news", "macroeconomic policy decision"],
        "scores": [0.90, 0.10],
    }
    enricher.event_classifier = mock_classifier

    headline = "Company X announces Q3 earnings"
    assert enricher.is_high_impact(headline) is False


def test_is_high_impact_classifier_exception_handled(enricher, caplog):
    mock_classifier = MagicMock()
    mock_classifier.side_effect = RuntimeError("Model inference failed")
    enricher.event_classifier = mock_classifier

    headline = "Headline causing classifier error"
    with caplog.at_level("ERROR"):
        assert enricher.is_high_impact(headline) is False
    assert "Classification error: Model inference failed" in caplog.text
