import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.psx_predictor.db.repository import upsert_stock_data

@patch('src.psx_predictor.db.repository.engine.begin')
def test_upsert_stock_data_success(mock_engine_begin):
    # Setup mock connection and execution result
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_conn.execute.return_value = mock_result
    
    # Setup context manager for engine.begin()
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_conn
    mock_engine_begin.return_value = mock_ctx

    df = pd.DataFrame([
        {"ticker": "PSO", "date": pd.to_datetime("2023-01-01").date(), "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000}
    ])
    
    success = upsert_stock_data(df)
    
    assert success is True
    mock_conn.execute.assert_called_once()

def test_upsert_stock_data_empty():
    df = pd.DataFrame()
    success = upsert_stock_data(df)
    assert success is False

def test_upsert_stock_data_missing_columns():
    # Missing 'date', 'close', etc.
    df = pd.DataFrame([{"ticker": "PSO", "open": 100.0}]) 
    success = upsert_stock_data(df)
    assert success is False
