import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.psx_predictor.scraper.client import PSXScraper

def create_mock_yf_df():
    data = {
        'Date': [pd.Timestamp('2025-01-03')],
        'Open': [481.99],
        'High': [496.00],
        'Low': [474.01],
        'Close': [485.38],
        'Volume': [4496408]
    }
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df

@patch('yfinance.download')
def test_fetch_raw_data_success(mock_download):
    mock_df = create_mock_yf_df()
    mock_download.return_value = mock_df

    scraper = PSXScraper()
    data = scraper.fetch_raw_data("PSO")
    
    assert data is not None
    assert not data.empty

def test_clean_and_format():
    scraper = PSXScraper()
    raw_df = create_mock_yf_df()
    df = scraper.clean_and_format(raw_df, "PSO")
    
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
    mock_fetch.return_value = create_mock_yf_df()
    mock_upsert.return_value = True
    
    scraper = PSXScraper()
    success = scraper.sync_ticker("PSO")
    
    assert success is True
    mock_upsert.assert_called_once()

