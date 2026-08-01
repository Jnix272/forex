from __future__ import annotations

import json

import pytest

from scripts.download_historical_news import (
    _parse_json_response,
    _read_existing_days,
    _retry_sleep_seconds,
    append_failures,
    write_rows,
)


def test_parse_json_response_accepts_unescaped_control_characters():
    payload = '{"articles":[{"title":"Euro edges higher \x01", "seendate":"20180124T084500Z"}]}'

    parsed = _parse_json_response(payload, "https://api.gdeltproject.org/api/v2/doc/doc")

    assert parsed["articles"][0]["title"].startswith("Euro edges higher")


def test_parse_json_response_rejects_trailing_data():
    payload = json.dumps({"articles": []}) + " not-json"

    with pytest.raises(ValueError, match="trailing data"):
        _parse_json_response(payload, "https://api.gdeltproject.org/api/v2/doc/doc")


def test_retry_sleep_uses_exponential_floor_with_jitter(monkeypatch):
    monkeypatch.setattr("scripts.download_historical_news.random.uniform", lambda _lo, _hi: 0.5)

    assert _retry_sleep_seconds(Exception("timeout"), sleep_s=5, attempt=3) == pytest.approx(20.5)


def test_read_existing_days_filters_gdelt_rows(tmp_path):
    out = tmp_path / "news.csv"
    write_rows(
        out,
        [
            {
                "timestamp_utc": "2018-01-02T08:45:00Z",
                "event_type": "headline",
                "currency": "EUR",
                "impact": "medium",
                "headline": "Euro edges higher",
                "actual": "",
                "forecast": "",
                "source": "gdelt",
                "url": "https://example.com/a",
            },
            {
                "timestamp_utc": "2018-01-03T08:45:00Z",
                "event_type": "headline",
                "currency": "EUR",
                "impact": "medium",
                "headline": "Other source",
                "actual": "",
                "forecast": "",
                "source": "eodhd_news",
                "url": "https://example.com/b",
            },
        ],
        append=False,
    )

    assert _read_existing_days(out, source="gdelt") == {("EUR", "2018-01-02")}


def test_append_failures_dedupes_chunks(tmp_path):
    out = tmp_path / "failures.csv"
    failure = {
        "source": "gdelt",
        "pair": "EURUSD",
        "start_utc": "2018-01-01T00:00:00Z",
        "end_utc": "2018-01-01T23:59:59Z",
        "reason": "HTTP Error 429",
    }

    assert append_failures(out, [failure]) == 1
    assert append_failures(out, [failure]) == 0
