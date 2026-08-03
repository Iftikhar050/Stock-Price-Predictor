import os
import sys
import logging
import pandas as pd
from sqlalchemy import text

# Add the project root to the sys.path to allow imports when running as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.psx_predictor.db.connection import engine

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

# Define output path based on current script location
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

def load_data(ticker: str) -> pd.DataFrame:
    """
    Queries PostgreSQL for ticker data, sorts chronologically, 
    and saves the baseline cleaned DataFrame as a CSV.
    """
    logger.info(f"Querying database for {ticker}...")
    query = text("SELECT * FROM stock_eod_data WHERE ticker = :ticker ORDER BY date ASC")
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker.upper()})
        
    if df.empty:
        logger.warning(f"No data found for ticker {ticker}. Have you run the scraper yet?")
        return df
        
    # Ensure strict datetime sorting
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    cleaned_path = os.path.join(PROCESSED_DIR, f"{ticker.lower()}_cleaned.csv")
    df.to_csv(cleaned_path, index=False)
    
    logger.info(f"Loaded {len(df)} records and saved baseline data to {cleaned_path}")
    return df

def calculate_sma(df: pd.DataFrame, col: str = 'close', windows: list = [7, 21, 50]) -> pd.DataFrame:
    """Calculates Simple Moving Averages for specified windows."""
    for window in windows:
        df[f'sma_{window}'] = df[col].rolling(window=window).mean()
    return df

def calculate_rsi(df: pd.DataFrame, col: str = 'close', window: int = 14) -> pd.DataFrame:
    """Calculates Relative Strength Index (RSI) using Wilder's Smoothing."""
    delta = df[col].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    # Calculate exponential moving average using Wilder's alpha
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
    return df

def calculate_macd(df: pd.DataFrame, col: str = 'close', fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculates MACD Line, Signal Line, and MACD Histogram."""
    ema_fast = df[col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[col].ewm(span=slow, adjust=False).mean()
    
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df

def calculate_lag_features(df: pd.DataFrame, col: str = 'close', lags: list = [1, 2, 3]) -> pd.DataFrame:
    """Calculates percentage returns and standard lag features."""
    # Daily percentage return (Target Variable base)
    df['daily_return'] = df[col].pct_change()
    
    # Lagged features (t-1, t-2, t-3)
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        df[f'return_lag_{lag}'] = df['daily_return'].shift(lag)
        
    return df

def build_features(ticker: str) -> pd.DataFrame:
    """Orchestrates the entire feature engineering pipeline."""
    df = load_data(ticker)
    if df.empty:
        return df
        
    logger.info(f"Generating technical features for {ticker}...")
    
    # 1. Technical Indicators
    df = calculate_sma(df, col='close', windows=[7, 21, 50])
    df = calculate_rsi(df, col='close', window=14)
    df = calculate_macd(df, col='close')
    
    # 2. Daily Returns and Lag Features (t-1, t-2, t-3)
    df = calculate_lag_features(df, col='close', lags=[1, 2, 3])
    
    # 3. Handle Missing Values
    # Since rolling windows (e.g., SMA-50) require at least 50 days of data to compute,
    # the first ~49 rows will contain NaNs. We drop these to prevent model poisoning.
    initial_len = len(df)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
    logger.info(f"Dropped {initial_len - len(df)} rows containing NaNs introduced by rolling windows.")
    
    # 4. Save finalized dataset
    final_path = os.path.join(PROCESSED_DIR, f"{ticker.lower()}_features.csv")
    df.to_csv(final_path, index=False)
    logger.info(f"Saved finalized feature dataset ready for ML to {final_path}")
    
    return df

TICKERS = ['PSO', 'FFC', 'NBP', 'MEBL', 'OGDC', 'LUCK']

if __name__ == '__main__':
    # When run directly, run the pipeline for all tickers
    for ticker in TICKERS:
        build_features(ticker)
