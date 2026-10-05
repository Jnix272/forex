import io
import zipfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scripts.download_gdelt2_bulk import COUNTRY_TO_CURRENCY, download_and_parse_gdelt


def test_country_to_currency_vectorized_mapping():
    # Test currency resolution logic directly
    targets = list(COUNTRY_TO_CURRENCY.keys())
    df_subset = pd.DataFrame(
        {
            "actor1": ["USA", "ABC", "EUR", "XYZ"],
            "actor2": ["GBR", "JPN", "CAN", "AUS"],
        }
    )

    mask = df_subset["actor1"].isin(targets) | df_subset["actor2"].isin(targets)
    df_filtered = df_subset[mask].copy()

    actor1_currency = df_filtered["actor1"].map(COUNTRY_TO_CURRENCY)
    actor2_currency = df_filtered["actor2"].map(COUNTRY_TO_CURRENCY)
    df_filtered["currency"] = actor1_currency.fillna(actor2_currency)

    expected = ["USD", "JPY", "EUR", "AUD"]
    assert df_filtered["currency"].tolist() == expected


def test_download_and_parse_gdelt_success():
    dt = datetime(2024, 1, 1, 12, 0, 0)
    url = "http://data.gdeltproject.org/gdeltv2/20240101120000.export.CSV.zip"

    # Build mock CSV lines (61 tab-separated columns)
    # Col 1 (sqldate), Col 7 (actor1), Col 17 (actor2), Col 34 (avgtone), Col 60 (url)
    row1 = [""] * 61
    row1[1] = "20240101"
    row1[7] = "USA"
    row1[17] = "GBR"
    row1[34] = "2.5"
    row1[60] = "https://example.com/1"

    row2 = [""] * 61
    row2[1] = "20240101"
    row2[7] = "UNKNOWN"
    row2[17] = "JPN"
    row2[34] = "-1.2"
    row2[60] = "https://example.com/2"

    csv_content = "\t".join(row1) + "\n" + "\t".join(row2) + "\n"

    # Zip the CSV
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("20240101120000.export.CSV", csv_content)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = zip_buffer.getvalue()

    with patch("requests.get", return_value=mock_resp):
        res_dt, df_final = download_and_parse_gdelt((dt, url))

    assert res_dt == dt
    assert df_final is not None
    assert len(df_final) == 2
    assert df_final["currency"].tolist() == ["USD", "JPY"]
    assert df_final["sentiment_score"].tolist() == [2.5, -1.2]
    assert df_final["source"].tolist() == ["gdelt2_bulk", "gdelt2_bulk"]
