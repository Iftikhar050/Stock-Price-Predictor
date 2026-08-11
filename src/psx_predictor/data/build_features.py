import os
import sys
import logging
import pandas as pd
import numpy as np
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
        sma = df[col].rolling(window=window).mean()
        df[f'sma_{window}_dist'] = (df[col] / sma) - 1.0
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
        df[f'return_lag_{lag}'] = df['daily_return'].shift(lag)
        
    return df

def calculate_bollinger_bands(df: pd.DataFrame, col: str = 'close', window: int = 20) -> pd.DataFrame:
    """Calculates Bollinger Bands (Upper, Lower, and Middle)."""
    bb_middle = df[col].rolling(window=window).mean()
    rolling_std = df[col].rolling(window=window).std()
    bb_upper = bb_middle + (rolling_std * 2)
    bb_lower = bb_middle - (rolling_std * 2)
    
    df['bb_middle_dist'] = (df[col] / bb_middle) - 1.0
    df['bb_upper_dist'] = (df[col] / bb_upper) - 1.0
    df['bb_lower_dist'] = (df[col] / bb_lower) - 1.0
    return df

def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Volume Weighted Average Price (VWAP)."""
    # Typical Price = (High + Low + Close) / 3
    # VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
    # Since this is daily data over a long period, a rolling VWAP is better to avoid infinite accumulation.
    # Let's use a 14-day rolling VWAP.
    window = 14
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    tp_v = typical_price * df['volume']
    
    vwap = tp_v.rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
    df['vwap_14_dist'] = (df['close'] / vwap) - 1.0
    return df

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts calendar and time-based features."""
    df['day_of_week'] = df['date'].dt.dayofweek
    return df

def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates On-Balance Volume (OBV)."""
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv'] = obv
    return df

def merge_sentiment(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Queries stock_news_sentiment, merges it, and applies a 3-day decay factor
    to handle missing days realistically instead of infinite forward-fill.
    """
    logger.info(f"Merging sentiment data for {ticker}...")
    query = text("SELECT date, sentiment_score FROM stock_news_sentiment WHERE ticker = :ticker ORDER BY date ASC")
    
    with engine.connect() as conn:
        sentiment_df = pd.read_sql(query, conn, params={"ticker": ticker.upper()})
        
    if sentiment_df.empty:
        logger.warning(f"No sentiment data found for {ticker}. Filling with 0.0")
        df['sentiment_score'] = 0.0
        return df
        
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    
    # Left join onto the main EOD dataframe
    df = pd.merge(df, sentiment_df, on='date', how='left')
    
    # Implement 3-day decay logic
    # We create shifted columns to see the sentiment of previous days
    df['sent_lag_1'] = df['sentiment_score'].shift(1)
    df['sent_lag_2'] = df['sentiment_score'].shift(2)
    df['sent_lag_3'] = df['sentiment_score'].shift(3)
    
    # Apply decay: if today is NaN, check lag 1 (x0.5). If lag 1 is NaN, check lag 2 (x0.25). 
    # If lag 2 is NaN, check lag 3 (x0.125). Else 0.0.
    
    def apply_decay(row):
        if pd.notna(row['sentiment_score']):
            return row['sentiment_score']
        if pd.notna(row['sent_lag_1']):
            return row['sent_lag_1'] * 0.5
        if pd.notna(row['sent_lag_2']):
            return row['sent_lag_2'] * 0.25
        if pd.notna(row['sent_lag_3']):
            return row['sent_lag_3'] * 0.125
        return 0.0
        
    df['sentiment_score'] = df.apply(apply_decay, axis=1)
    
    # Drop intermediate lag columns
    df.drop(columns=['sent_lag_1', 'sent_lag_2', 'sent_lag_3'], inplace=True)
    return df

def merge_dividends(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Queries stock_dividends, merges it, and engineers dividend features:
    - days_since_dividend
    - dividend_yield
    - is_ex_dividend_week
    """
    logger.info(f"Merging dividend data for {ticker}...")
    query = text("SELECT ex_dividend_date as date, dividend_amount FROM stock_dividends WHERE ticker = :ticker ORDER BY date ASC")
    
    with engine.connect() as conn:
        div_df = pd.read_sql(query, conn, params={"ticker": ticker.upper()})
        
    if div_df.empty:
        logger.warning(f"No dividend data found for {ticker}.")
        df['days_since_dividend'] = 9999
        df['dividend_yield'] = 0.0
        df['is_ex_dividend_week'] = 0
        return df
        
    div_df['date'] = pd.to_datetime(div_df['date'])
    
    # Left join onto the main EOD dataframe
    df = pd.merge(df, div_df, on='date', how='left')
    
    # days_since_dividend
    # Identify indices where a dividend occurred
    df['dividend_amount'] = df['dividend_amount'].fillna(0)
    
    # We want a forward fill for the date of the last dividend
    div_dates = df.loc[df['dividend_amount'] > 0, 'date']
    df['last_div_date'] = pd.Series(index=df.index, dtype='datetime64[ns]')
    df.loc[div_dates.index, 'last_div_date'] = div_dates
    df['last_div_date'] = df['last_div_date'].ffill()
    
    # Calculate days since last dividend
    df['days_since_dividend'] = (df['date'] - df['last_div_date']).dt.days
    df['days_since_dividend'] = df['days_since_dividend'].fillna(9999) # For days before any dividend
    
    # dividend_yield: trailing 12 months dividend sum / price
    # Since we don't have exactly 12 months rolling easily without dates, we'll just use the last dividend amount * 4 (assuming quarterly) for a rough annualized yield, or just the last dividend amount / close. Let's just use last dividend amount / close price
    
    df['last_div_amount'] = df['dividend_amount'].replace(0, pd.NA).ffill().fillna(0)
    df['dividend_yield'] = df['last_div_amount'] / df['close']
    
    # is_ex_dividend_week: is the current date within 7 days of ANY dividend date (past or future)?
    # Actually, predicting requires looking forward, so 'is_ex_dividend_week' (upcoming) would be cheating if not known. 
    # But usually, announcement date is known before ex-div date. So it's fair if within [-7, 0] days of ex_dividend_date.
    # Let's say: is the date within 7 days PRIOR to an ex-dividend date?
    df['next_div_date'] = pd.Series(index=df.index, dtype='datetime64[ns]')
    df.loc[div_dates.index, 'next_div_date'] = div_dates
    df['next_div_date'] = df['next_div_date'].bfill()
    df['days_to_next_dividend'] = (df['next_div_date'] - df['date']).dt.days
    
    df['is_ex_dividend_week'] = ((df['days_to_next_dividend'] >= 0) & (df['days_to_next_dividend'] <= 7)).astype(int)
    
    # Drop intermediate columns
    df.drop(columns=['dividend_amount', 'last_div_date', 'last_div_amount', 'next_div_date', 'days_to_next_dividend'], inplace=True)
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
    
    # 2.5 New Distinct Features (Volatility, Volume Context, Time)
    df = calculate_bollinger_bands(df, col='close', window=20)
    df = calculate_vwap(df)
    df = extract_time_features(df)
    df = calculate_obv(df)
    
    # 3. Merge Sentiment Data with Decay
    df = merge_sentiment(df, ticker)
    
    # 3.5 Merge Dividend Data
    df = merge_dividends(df, ticker)
    
    # 3.8 Spread Features
    df['daily_spread'] = (df['high'] - df['low']) / df['close']
    high_low_diff = df['high'] - df['low']
    df['close_pos'] = np.where(high_low_diff == 0, 0.5, (df['close'] - df['low']) / high_low_diff)
    
    # 4. Handle Missing Values
    # Since rolling windows (e.g., SMA-50) require at least 50 days of data to compute,
    # the first ~49 rows will contain NaNs. We drop these to prevent model poisoning.
    initial_len = len(df)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
    logger.info(f"Dropped {initial_len - len(df)} rows containing NaNs introduced by rolling windows.")
    
    cols_to_drop = ['open', 'high', 'low', 'volume']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    
    # 5. Save finalized dataset
    final_path = os.path.join(PROCESSED_DIR, f"{ticker.lower()}_features.csv")
    df.to_csv(final_path, index=False)
    logger.info(f"Saved finalized feature dataset ready for ML to {final_path}")
    
    return df

TICKERS = ['PSO', 'FFC', 'NBP', 'MEBL', 'OGDC', 'LUCK']

if __name__ == '__main__':
    # When run directly, run the pipeline for all tickers
    for ticker in TICKERS:
        build_features(ticker)
