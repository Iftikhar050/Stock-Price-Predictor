import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.psx_predictor.scraper.client import PSXScraper

MOCK_HTML = """
<table>
    <thead>
        <tr><th>DATE</th><th>OPEN</th><th>HIGH</th><th>LOW</th><th>CLOSE</th><th>VOLUME</th></tr>
    </thead>
    <tbody>
        <tr><td>Jan 3, 2025</td><td>481.99</td><td>496.00</td><td>474.01</td><td>485.38</td><td>4,496,408</td></tr>
    </tbody>
</table>
"""

@patch('requests.Session.post')
def test_fetch_raw_data_success(mock_post):
    mock_response = MagicMock()
    mock_response.text = MOCK_HTML
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    scraper = PSXScraper()
    data = scraper.fetch_raw_data("PSO")
    
    assert data is not None
    assert "<table>" in data

def test_clean_and_format():
    scraper = PSXScraper()
    df = scraper.clean_and_format(MOCK_HTML, "PSO")
    
    assert not df.empty
    assert df.iloc[0]['ticker'] == "PSO"
    assert df.iloc[0]['open'] == 481.99
    assert df.iloc[0]['high'] == 496.00
    assert df.iloc[0]['low'] == 474.01
    assert df.iloc[0]['close'] == 485.38
    assert df.iloc[0]['volume'] == 4496408
    assert str(df.iloc[0]['date']) == "2025-01-03"

@patch('src.psx_predictor.scraper.client.upsert_stock_data')
@patch('src.psx_predictor.scraper.client.PSXScraper.fetch_raw_data')
def test_sync_ticker(mock_fetch, mock_upsert):
    mock_fetch.return_value = MOCK_HTML
    mock_upsert.return_value = True
    
    scraper = PSXScraper()
    success = scraper.sync_ticker("PSO")
    
    assert success is True
    mock_upsert.assert_called_once()
