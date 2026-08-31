"""Thin wrapper that re‑uses the data‑preparation logic from the training scripts.
The function is deliberately simple – it loads the CSV produced by `build_features.py`
and builds the target variable exactly as the training scripts do.
"""
import os
import pandas as pd

# Base directory where processed feature CSVs are stored (same as training scripts)
BASE_FEATURES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "processed")
)

import os
import pandas as pd
import numpy as np
from sqlalchemy import text
from src.psx_predictor.db.connection import engine

# Base directory where processed feature CSVs are stored (same as training scripts)
BASE_FEATURES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "processed")
)

def get_ticker_sectors():
    """Fetches the sector for each ticker from the database."""
    query = text("SELECT ticker, sector FROM stock_metadata")
    with engine.connect() as conn:
        res = conn.execute(query).fetchall()
    return {row[0]: row[1] for row in res}

def prepare_data(ticker: str, ticker_sectors: dict = None):
    """Loads engineered features and creates the target variable for a given ticker.
    
    This is the single source of truth for model input data preparation, shared 
    by both production training scripts and walk-forward evaluation.
    """
    csv_path = os.path.join(BASE_FEATURES_DIR, f"{ticker.lower()}_features.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}. Run build_features.py first.")
        
    df = pd.read_csv(csv_path).copy()
    if df.empty:
        raise ValueError(f"Empty feature data for {ticker}")
    
    # 1. Create Target Variable (Next Day's Return)
    df['target_return_t1'] = (df['close'].shift(-1) - df['close']) / df['close']
    df.dropna(subset=['target_return_t1'], inplace=True)
    df = df.copy()
    
    # 2. Select Features (X) and Target (y)
    exclude_cols = ['date', 'created_at', 'target_return_t1', 'close']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    
    # Only add sector if we have mapping (e.g. for xgboost)
    if ticker_sectors is not None:
        X['sector'] = X['ticker'].map(ticker_sectors)

    # Clean numeric columns to remove inf / nan / float32 overflow
    num_cols = X.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X[num_cols] = X[num_cols].clip(lower=-1e9, upper=1e9)
        
    y = df['target_return_t1']
    dates = pd.to_datetime(df['date'])
    current_close = df['close']
    
    return X, y, dates, current_close

def feature_fn(ticker: str, start_date: str, end_date: str, ticker_sectors: dict = None):
    """Return X, y, dates, close for a ticker within the given date range.
    The returned DataFrames are already filtered to the inclusive interval.
    """
    X, y, dates, close = prepare_data(ticker, ticker_sectors)
    
    # Filter by date range
    mask = (dates >= start_date) & (dates <= end_date)
    
    # We drop 'ticker' from X for the models? Wait, in original train_xgboost it kept ticker. 
    # train_baseline excluded ticker. Let's keep the return as is, but in run_walk_forward, it doesn't drop anything.
    # We should let the models handle whether they use it or not, or drop it if it's not xgboost.
    
    # However, to be perfectly safe, since walk-forward runs on `feature_fn`, 
    # let's return the filtered rows
    X_filtered = X.loc[mask].copy()
    y_filtered = y.loc[mask].copy()
    dates_filtered = dates.loc[mask].copy()
    close_filtered = close.loc[mask].copy()
    
    return X_filtered, y_filtered, dates_filtered, close_filtered
