import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy import text

# Add the project root to the sys.path to allow imports when running as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.psx_predictor.db.connection import engine
from src.psx_predictor.data.feature_corporate_events import generate_event_features
from src.psx_predictor.data.feature_calendar_events import generate_calendar_features
from src.psx_predictor.data.feature_political_events import build_political_features
from src.psx_predictor.data.fetch_sbp_additional import fetch_sbp_additional
from src.psx_predictor.data.fetch_pakistan_activity import fetch_pakistan_activity


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

# Feature set version – bump when features change
FEATURE_SET_VERSION = "v2"

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
    """Calculates Simple Moving Averages (raw values) and distance-to-SMA for specified windows.
    Note: sma_N_dist is the distance-to-SMA, NOT the raw SMA itself. Both are added.
    """
    for window in windows:
        sma = df[col].rolling(window=window).mean()
        df[f'sma_{window}'] = sma  # raw SMA (PDF Group #27 requirement)
        df[f'sma_{window}_dist'] = (df[col] / sma) - 1.0  # existing distance feature preserved
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

def calculate_lag_features(df: pd.DataFrame, col: str = 'close', lags: list = [1, 2, 3, 5, 10]) -> pd.DataFrame:
    """Calculates percentage returns and standard lag features."""
    # Daily percentage return (Target Variable base)
    df['daily_return'] = df[col].pct_change()
    
    # Lagged features
    for lag in lags:
        df[f'return_lag_{lag}'] = df['daily_return'].shift(lag)
        
    return df

def calculate_bollinger_bands(df: pd.DataFrame, col: str = 'close', window: int = 20) -> pd.DataFrame:
    """Calculates Bollinger Bands (Upper, Lower, and Middle)."""
    bb_middle = df[col].rolling(window=window).mean()
    rolling_std = df[col].rolling(window=window).std()
    bb_upper = bb_middle + (rolling_std * 2)
    bb_lower = bb_middle - (rolling_std * 2)
    
    df['bollinger_mavg'] = bb_middle
    df['bollinger_hband'] = bb_upper
    df['bollinger_lband'] = bb_lower
    df['bollinger_width'] = ((bb_upper - bb_lower) / bb_middle.replace(0, np.nan)).fillna(0.0)
    df['bb_middle_dist'] = (df[col] / bb_middle) - 1.0
    df['bb_upper_dist'] = (df[col] / bb_upper) - 1.0
    df['bb_lower_dist'] = (df[col] / bb_lower) - 1.0
    return df

def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates raw VWAP and distance-to-VWAP.
    Note: vwap_14_dist is the distance-to-VWAP, NOT raw VWAP. Both are added.
    Source: Intraday approximation using daily OHLCV (14-day rolling window).
    """
    window = 14
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    tp_v = typical_price * df['volume']
    vwap_rolling_sum = df['volume'].rolling(window=window).sum().replace(0, np.nan)
    vwap = tp_v.rolling(window=window).sum() / vwap_rolling_sum
    df['vwap'] = vwap  # raw VWAP (PDF Group #27 requirement)
    df['vwap_14_dist'] = (df['close'] / vwap.replace(0, np.nan)) - 1.0  # existing distance preserved
    return df

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts calendar and time-based features."""
    day_of_week = df['date'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * day_of_week / 5)
    df['dow_cos'] = np.cos(2 * np.pi * day_of_week / 5)
    return df

def calculate_obv(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """Calculates normalized On-Balance Volume (OBV) via rolling z-score."""
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    obv_mean = obv.rolling(window=window).mean()
    obv_std = obv.rolling(window=window).std()
    
    # Avoid division by zero
    obv_std = obv_std.replace(0, 1)
    
    df['obv'] = (obv - obv_mean) / obv_std
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
        
    # Also query corporate_announcements_pucars raw text announcements
    pucars_query = text("SELECT announcement_date as date, sentiment_score as pucars_sentiment FROM corporate_announcements_pucars WHERE ticker = :ticker ORDER BY announcement_date ASC")
    with engine.connect() as conn:
        pucars_df = pd.read_sql(pucars_query, conn, params={"ticker": ticker.upper()})
        
    if not pucars_df.empty:
        pucars_df['date'] = pd.to_datetime(pucars_df['date'])
        pucars_df = pucars_df.groupby('date')['pucars_sentiment'].mean().reset_index()
        df = pd.merge(df, pucars_df, on='date', how='left')
        df['pucars_sentiment'] = df['pucars_sentiment'].fillna(0.0)
        df['sentiment_score'] = np.where(df['pucars_sentiment'] != 0, df['pucars_sentiment'], df['sentiment_score'])
        df.drop(columns=['pucars_sentiment'], inplace=True, errors='ignore')
        
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
    df['days_since_dividend'] = df['days_since_dividend'].fillna(365).clip(upper=365) # Cap at 365
    
    # dividend_yield: trailing 12 months dividend sum / price
    # Since we don't have exactly 12 months rolling easily without dates, we'll just use the last dividend amount * 4 (assuming quarterly) for a rough annualized yield, or just the last dividend amount / close. Let's just use last dividend amount / close price
    
    df['last_div_amount'] = df['dividend_amount'].replace(0, np.nan).ffill().fillna(0).astype(float)
    df['dividend_yield'] = (df['last_div_amount'] / df['close']).astype(float)
    
    # NOTE: Dropped is_ex_dividend_week and forward-looking next_div_date to avoid lookahead leak,
    # as we do not currently have the actual announcement date available.
    
    # Drop intermediate columns
    df.drop(columns=['dividend_amount', 'last_div_date', 'last_div_amount'], inplace=True)
    return df

def merge_fundamentals(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Queries stock_fundamentals, merges it using as-of/backward-fill semantics.
    Computes dynamic pe_ratio (close / eps_trailing).
    """
    logger.info(f"Merging fundamentals data for {ticker}...")
    query = text("SELECT report_date as date, * FROM stock_fundamentals WHERE ticker = :ticker ORDER BY report_date ASC")
    
    with engine.connect() as conn:
        fund_df = pd.read_sql(query, conn, params={"ticker": ticker.upper()})
        if 'report_date' in fund_df.columns:
            fund_df.drop(columns=['report_date'], inplace=True, errors='ignore')
        if 'ticker' in fund_df.columns:
            fund_df.drop(columns=['ticker'], inplace=True, errors='ignore')
        if 'created_at' in fund_df.columns:
            fund_df.drop(columns=['created_at'], inplace=True, errors='ignore')
        
    # Propagate ownership, insider, and ratio metrics across all report dates
    non_date_cols = [c for c in fund_df.columns if c != 'date']
    fund_df[non_date_cols] = fund_df[non_date_cols].bfill().ffill()
    
    if fund_df.empty:
        logger.warning(f"No fundamentals data found for {ticker}.")
        df['eps_trailing'] = 0.0
        df['pe_ratio'] = np.nan
        df['roe'] = 0.0
        df['debt_to_equity'] = 0.0
        df['book_value_per_share'] = 0.0
        df['eps_growth_yoy'] = 0.0
        df['revenue'] = 0.0
        df['net_income'] = 0.0
        df['free_cash_flow'] = 0.0
        df['operating_cash_flow'] = 0.0
        df['total_assets'] = 0.0
        df['total_debt'] = 0.0
        df['ebitda'] = 0.0
        df['total_cash'] = 0.0
        df['shares_outstanding'] = 0.0
        df['pb_ratio'] = np.nan
        df['profit_margin'] = 0.0
        df['roa'] = 0.0
        df['peg_ratio'] = np.nan
        df['ev'] = np.nan
        df['ev_ebitda'] = np.nan
        df['ev_sales'] = np.nan
        df['pe_percentile_1y'] = np.nan
        df['pe_percentile_3y'] = np.nan
        return df
        
    if len(fund_df) < 4:
        logger.warning(f"Fundamentals coverage is thin for {ticker} (only {len(fund_df)} rows).")
        
    fund_df['date'] = pd.to_datetime(fund_df['date'])
    fund_df = fund_df.sort_values('date').reset_index(drop=True)
    
    if 'eps_growth_yoy' not in fund_df.columns or fund_df['eps_growth_yoy'].isna().all():
        fund_df['eps_growth_yoy_raw'] = fund_df['eps'].pct_change(periods=4)
        fund_df['gap_days'] = (fund_df['date'] - fund_df['date'].shift(4)).dt.days
        fund_df['eps_growth_yoy'] = np.where(fund_df['gap_days'].between(340, 390), fund_df['eps_growth_yoy_raw'], np.nan)
        if 'eps_growth_yoy_raw' in fund_df.columns:
            fund_df.drop(columns=['eps_growth_yoy_raw', 'gap_days'], inplace=True, errors='ignore')
    
    fund_df.rename(columns={'eps': 'eps_trailing'}, inplace=True)
    
    df = pd.merge(df, fund_df, on='date', how='left')
    
    cols_to_fill = [c for c in fund_df.columns if c not in ['date', 'ticker', 'created_at'] and c in df.columns]
    df[cols_to_fill] = df[cols_to_fill].ffill().bfill().fillna(0.0).astype(float)
    
    # Calculate Ratios
    df['pe_ratio'] = np.nan
    mask_eps = (df['eps_trailing'] != 0) & (df['eps_trailing'].notna())
    if mask_eps.any():
        df.loc[mask_eps, 'pe_ratio'] = df.loc[mask_eps, 'close'] / df.loc[mask_eps, 'eps_trailing']
        
    if 'peg_ratio' not in df.columns or (df['peg_ratio'] == 0).all():
        df['peg_ratio'] = np.nan
        mask_peg = mask_eps & (df['eps_growth_yoy'] != 0) & (df['eps_growth_yoy'].notna())
        if mask_peg.any():
            df.loc[mask_peg, 'peg_ratio'] = df.loc[mask_peg, 'pe_ratio'] / (df.loc[mask_peg, 'eps_growth_yoy'] * 100)
        
    df['pb_ratio'] = np.nan
    mask_bv = (df['book_value_per_share'] != 0) & (df['book_value_per_share'].notna())
    if mask_bv.any():
        df.loc[mask_bv, 'pb_ratio'] = df.loc[mask_bv, 'close'] / df.loc[mask_bv, 'book_value_per_share']
        
    df['profit_margin'] = 0.0
    mask_rev = (df['revenue'] != 0) & (df['revenue'].notna())
    if mask_rev.any():
        df.loc[mask_rev, 'profit_margin'] = df.loc[mask_rev, 'net_income'] / df.loc[mask_rev, 'revenue']
        
    df['roa'] = 0.0
    mask_assets = (df['total_assets'] != 0) & (df['total_assets'].notna())
    if mask_assets.any():
        df.loc[mask_assets, 'roa'] = df.loc[mask_assets, 'net_income'] / df.loc[mask_assets, 'total_assets']
        
    df['ev'] = np.nan
    mask_shares = (df['shares_outstanding'] != 0) & (df['shares_outstanding'].notna())
    if mask_shares.any():
        df.loc[mask_shares, 'ev'] = (df.loc[mask_shares, 'close'] * df.loc[mask_shares, 'shares_outstanding']) + df.loc[mask_shares, 'total_debt'] - df.loc[mask_shares, 'total_cash']
        
    df['ev_ebitda'] = np.nan
    mask_ebitda = (df['ebitda'] != 0) & (df['ebitda'].notna()) & df['ev'].notna()
    if mask_ebitda.any():
        df.loc[mask_ebitda, 'ev_ebitda'] = df.loc[mask_ebitda, 'ev'] / df.loc[mask_ebitda, 'ebitda']
        
    df['ev_sales'] = np.nan
    mask_ev_rev = (df['revenue'] != 0) & (df['revenue'].notna()) & df['ev'].notna()
    if mask_ev_rev.any():
        df.loc[mask_ev_rev, 'ev_sales'] = df.loc[mask_ev_rev, 'ev'] / df.loc[mask_ev_rev, 'revenue']
        
    # Percentiles (vectorized rolling rank)
    df['pe_percentile_1y'] = df['pe_ratio'].rolling(252, min_periods=60).rank(pct=True).fillna(0.5)
    df['pe_percentile_3y'] = df['pe_ratio'].rolling(756, min_periods=252).rank(pct=True).fillna(0.5)

    # Priority 1: Missing Fundamentals & Sector-Relative Valuation
    df['interest_bearing_debt'] = df['total_debt'].fillna(0.0) if 'total_debt' in df.columns else 0.0

    growth_est = df['eps_growth_yoy'].clip(-0.3, 0.5) if 'eps_growth_yoy' in df.columns else 0.10
    fwd_eps = df['eps_trailing'] * (1.0 + growth_est)
    mask_fwd = (fwd_eps > 0) & fwd_eps.notna() & (df['close'] > 0)
    df['forward_pe'] = 0.0
    if mask_fwd.any():
        df.loc[mask_fwd, 'forward_pe'] = df.loc[mask_fwd, 'close'] / fwd_eps.loc[mask_fwd]

    if 'operating_cash_flow' in df.columns and 'shares_outstanding' in df.columns:
        cf_per_share = (df['operating_cash_flow'] / df['shares_outstanding'].replace(0, np.nan)).fillna(0.0)
        mask_cf = (cf_per_share > 0) & (df['close'] > 0)
        df['price_to_cash_flow'] = 0.0
        if mask_cf.any():
            df.loc[mask_cf, 'price_to_cash_flow'] = df.loc[mask_cf, 'close'] / cf_per_share.loc[mask_cf]
    else:
        df['price_to_cash_flow'] = 0.0

    sector_pe_baseline = 7.5 if ticker.upper() in ['MEBL', 'MCB', 'UBL', 'HBL', 'NBP', 'BAFL', 'BAHL'] else 6.0
    sector_pb_baseline = 1.2 if ticker.upper() in ['MEBL', 'MCB', 'UBL', 'HBL', 'NBP', 'BAFL', 'BAHL'] else 0.9
    df['sector_pe_avg'] = sector_pe_baseline
    df['sector_pb_avg'] = sector_pb_baseline

    val_cols = [
        'pe_ratio', 'peg_ratio', 'pb_ratio', 'profit_margin', 'roa', 'ev', 'ev_ebitda', 'ev_sales',
        'pe_percentile_1y', 'pe_percentile_3y', 'interest_bearing_debt', 'forward_pe',
        'price_to_cash_flow', 'sector_pe_avg', 'sector_pb_avg'
    ]
    df[val_cols] = df[val_cols].fillna(0.0)

    bank_tickers = ['MEBL', 'MCB', 'UBL', 'HBL', 'NBP', 'BAFL', 'BAHL']
    bank_cols = ['net_interest_margin', 'casa_ratio', 'casa_deposits', 'total_advances', 'total_deposits',
                 'npl_ratio', 'provisioning_coverage', 'capital_adequacy_ratio', 'adr_ratio', 'idr_ratio']
    if ticker.upper() not in bank_tickers:
        for bc in bank_cols:
            df[bc] = np.nan
    
    return df


def merge_macro_indicators(df: pd.DataFrame, ticker: str, sector: str) -> pd.DataFrame:
    """
    Queries macro_indicators, merges it, computes returns, conditionally includes oil.
    """
    logger.info(f"Merging macro indicators for {ticker} (Sector: {sector})...")
    query = text("SELECT * FROM macro_indicators ORDER BY date ASC")
    
    with engine.connect() as conn:
        macro_df = pd.read_sql(query, conn)
        
    if macro_df.empty:
        logger.warning(f"No macro data found.")
        df['sbp_policy_rate'] = 0.0
        df['days_since_rate_change'] = 0.0
        df['pkr_usd_change_pct'] = 0.0
        df['oil_return_pct'] = 0.0
        return df
        
    macro_df['date'] = pd.to_datetime(macro_df['date'])
    
    pct_cols = ['sp500', 'nasdaq', 'dxy', 'gold', 'copper', 'coal', 'cotton', 'gas', 
                'nikkei', 'hang_seng', 'shanghai', 'ftse', 'dax', 'vix']
                
    raw_level_cols = ['pkr_usd_rate', 'brent_oil_price', 'us10y_yield'] + \
                     [f"{col}_close" if col not in ['gold', 'copper', 'coal', 'cotton', 'gas'] else f"{col}_price" for col in pct_cols]
                     
    for c in raw_level_cols:
        if c in macro_df.columns:
            macro_df[c] = macro_df[c].ffill().bfill()
            
    macro_df['pkr_usd_change_pct'] = macro_df['pkr_usd_rate'].pct_change()
    macro_df['oil_return_pct'] = macro_df['brent_oil_price'].pct_change()
    
    # Derived PDF Metrics
    if 'sbp_policy_rate' in macro_df.columns and 'cpi_headline' in macro_df.columns:
        macro_df['real_interest_rate'] = macro_df['sbp_policy_rate'] - macro_df['cpi_headline']
    if 'pib_10y' in macro_df.columns and 'tbill_3m' in macro_df.columns:
        macro_df['yield_curve_slope'] = macro_df['pib_10y'] - macro_df['tbill_3m']
    if 'monthly_remittances' in macro_df.columns:
        macro_df['remittances_yoy'] = macro_df['monthly_remittances'].pct_change(periods=365)
    
    # Calculate returns for all continuous indicators
    for col in pct_cols:
        col_name = f"{col}_close" if col not in ['gold', 'copper', 'coal', 'cotton', 'gas'] else f"{col}_price"
        if col_name in macro_df.columns:
            macro_df[f"{col}_return_pct"] = macro_df[col_name].pct_change()
    
    macro_cols = [c for c in macro_df.columns if c not in ['date', 'created_at']]
    cols_to_merge = ['date'] + macro_cols
    
    df = pd.merge(df, macro_df[cols_to_merge], on='date', how='left')
    
    for col in macro_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill().fillna(0.0)
    
    if sector and any(s in sector.lower() for s in ['oil & gas', 'refinery', 'power generation']):
        df['oil_return_pct'] = df['oil_return_pct'].fillna(0.0)
    else:
        df['oil_return_pct'] = 0.0
        
    return df


def merge_market_index(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Joins daily return of the KSE-100 index and calculates sector average return.
    """
    logger.info(f"Merging market index & sector data for {ticker}...")
    
    # Fetch Market Index (Synthetic proxy excluding the current ticker)
    query_idx = text("""
        SELECT date, AVG(close) as market_close
        FROM stock_eod_data
        WHERE ticker != :ticker
        GROUP BY date
        ORDER BY date ASC
    """)
    with engine.connect() as conn:
        idx_df = pd.read_sql(query_idx, conn, params={"ticker": ticker})
    
    if idx_df.empty:
        logger.warning("No market index data found (or no other tickers available). Skipping market features.")
        df['market_return'] = 0.0
        df['market_return_lag_1'] = 0.0
        df['relative_strength_20'] = 0.0
        df['sector_return'] = 0.0
        return df
        
    idx_df['date'] = pd.to_datetime(idx_df['date'])
    idx_df['market_return'] = idx_df['market_close'].pct_change()
    idx_df['market_return_lag_1'] = idx_df['market_return'].shift(1)
    
    df = pd.merge(df, idx_df[['date', 'market_return', 'market_return_lag_1']], on='date', how='left')
    df['market_return'] = df['market_return'].fillna(0.0)
    df['market_return_lag_1'] = df['market_return_lag_1'].fillna(0.0)
    
    # Merge NCCPL Institutional Flows from stock_market_index table
    # Fix P0-E: use explicit column list to prevent _x/_y collision
    query_flows = text("""SELECT date, fipi_net_usd_m, lipi_mutual_funds_net, lipi_banks_net,
        lipi_insurance_net, lipi_companies_net, lipi_individuals_net
        FROM stock_market_index ORDER BY date ASC""")
    with engine.connect() as conn:
        flows_df = pd.read_sql(query_flows, conn)
        
    if not flows_df.empty:
        flows_df['date'] = pd.to_datetime(flows_df['date'])
        flow_cols = ['fipi_net_usd_m', 'lipi_mutual_funds_net', 'lipi_banks_net',
                     'lipi_insurance_net', 'lipi_companies_net', 'lipi_individuals_net']
        existing_flow_cols = [c for c in flow_cols if c in df.columns]
        if existing_flow_cols:
            df.drop(columns=existing_flow_cols, inplace=True)
        df = pd.merge(df, flows_df, on='date', how='left')
        for fc in flow_cols:
            if fc in df.columns:
                df[fc] = df[fc].ffill().bfill().fillna(0.0)

    # Dynamic Market Breadth across all 103 stocks in DB
    try:
        query_breadth = text("""
            SELECT date,
                   SUM(CASE WHEN close > open THEN 1 ELSE 0 END)::float / COUNT(*) as advancing_stocks_pct,
                   SUM(CASE WHEN close < open THEN 1 ELSE 0 END)::float / COUNT(*) as declining_stocks_pct,
                   SUM(CASE WHEN close > open THEN 1 ELSE 0 END)::float / NULLIF(SUM(CASE WHEN close < open THEN 1 ELSE 0 END), 0) as market_breadth_ratio,
                   SUM(volume) as market_total_volume,
                   SUM(close * volume) as market_total_traded_value
            FROM stock_eod_data
            GROUP BY date
            ORDER BY date ASC
        """)
        with engine.connect() as conn:
            breadth_df = pd.read_sql(query_breadth, conn)
        if not breadth_df.empty:
            breadth_df['date'] = pd.to_datetime(breadth_df['date'])
            breadth_df['market_breadth_ratio'] = breadth_df['market_breadth_ratio'].fillna(1.0)
            b_cols = ['advancing_stocks_pct', 'declining_stocks_pct', 'market_breadth_ratio', 'market_total_volume', 'market_total_traded_value']
            existing_b_cols = [c for c in b_cols if c in df.columns]
            if existing_b_cols:
                df.drop(columns=existing_b_cols, inplace=True)
            df = pd.merge(df, breadth_df, on='date', how='left')
            for bc in b_cols:
                if bc in df.columns:
                    df[bc] = df[bc].ffill().bfill().fillna(0.0)
    except Exception as e_mb:
        logger.warning(f"Could not compute daily market breadth: {e_mb}")
    
    # Calculate Relative Strength
    if 'daily_return' not in df.columns:
        df['daily_return'] = df['close'].pct_change()
    
    diff = df['daily_return'] - df['market_return']
    df['relative_strength_20'] = diff.rolling(window=20).sum().fillna(0.0)
    
    # Sector Return
    query_sec = text("""
        SELECT e.date, e.close
        FROM stock_eod_data e
        JOIN stock_metadata m ON e.ticker = m.ticker
        WHERE m.sector = (SELECT sector FROM stock_metadata WHERE ticker = :ticker)
          AND e.ticker != :ticker
        ORDER BY e.date ASC
    """)
    with engine.connect() as conn:
        sec_raw_df = pd.read_sql(query_sec, conn, params={"ticker": ticker})
        
    if sec_raw_df.empty:
        df['sector_return'] = df['market_return'] # fallback to market return if no peers
    else:
        sec_df = sec_raw_df.groupby('date')['close'].mean().reset_index()
        sec_df['date'] = pd.to_datetime(sec_df['date'])
        sec_df['sector_return'] = sec_df['close'].pct_change()
        df = pd.merge(df, sec_df[['date', 'sector_return']], on='date', how='left')
        df['sector_return'] = df['sector_return'].fillna(df['market_return'])
        
    sec_diff = df['daily_return'] - df['sector_return']
    df['sector_relative_strength_20'] = sec_diff.rolling(window=20).sum().fillna(0.0)
        
    return df

def calculate_relative_volume(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculates relative volume over a rolling window."""
    vol_ma = df['volume'].rolling(window=window).mean()
    # Avoid division by zero
    vol_ma = vol_ma.replace(0, 1)
    df['relative_volume'] = df['volume'] / vol_ma
    return df

def calculate_realized_volatility(df: pd.DataFrame, windows: list = [10, 20]) -> pd.DataFrame:
    """Calculates rolling realized volatility (standard deviation of daily return)."""
    if 'daily_return' not in df.columns:
        df['daily_return'] = df['close'].pct_change()
        
    for window in windows:
        df[f'return_vol_{window}'] = df['daily_return'].rolling(window=window).std()
    return df

def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculates Average True Range (ATR)."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=window).mean()
    return df

def calculate_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculates Average Directional Index (ADX-14).
    Definition: ADX measures trend strength, 0-100 scale.
    Source: Computed from daily OHLCV (High, Low, Close).
    Frequency: Daily.
    Leakage: None — uses only past values.
    """
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    mask = plus_dm > minus_dm
    plus_dm[~mask] = 0
    minus_dm[mask] = 0

    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    atr_w = tr.rolling(window=window, min_periods=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window, min_periods=window).mean() / atr_w.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(window=window, min_periods=window).mean() / atr_w.replace(0, np.nan))

    dx_denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / dx_denom
    df['adx'] = dx.rolling(window=window, min_periods=window).mean().fillna(0.0)
    df['plus_di'] = plus_di.fillna(0.0)
    df['minus_di'] = minus_di.fillna(0.0)
    return df


def calculate_computed_pdf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes advanced PDF framework features:
    - Multi-Horizon Momentum (3d, 10d, 50d, 100d, 200d)
    - Risk & Volatility (Rolling 60d/252d Beta, Downside Beta, Max Drawdown 252d, Market Volatility 20d)
    - Cross-Asset Correlations (60d rolling corr with KSE100, Brent, Policy Rate, Gold, PKR/USD)
    - Volume & Liquidity (Turnover ratio, PV Trend momentum, CMF 20d)
    - Calendar Effects (Seasonality flags, Fiscal Year-End)
    - Advanced Oscillators (Stochastic %K/%D, Williams %R, CCI-14, Bollinger Width)
    - Microstructure Volatility (Parkinson Volatility 20d, Garman-Klass Volatility 20d)
    - Higher Order Moments (Return Skewness 20d, Return Kurtosis 20d)
    - Market & Sector Relative Performance (Advance-Decline Line, Stock Relative Strength vs Sector)
    """
    # 1. Multi-Horizon Momentum
    for w in [3, 10, 50, 100, 200]:
        df[f'return_{w}d'] = df['close'].pct_change(periods=w).fillna(0.0)
        
    # 2. Maximum Drawdown (252d)
    rolling_max = df['close'].rolling(window=252, min_periods=20).max()
    df['max_drawdown_252d'] = ((df['close'] - rolling_max) / rolling_max).fillna(0.0)
    
    # 3. Market Volatility (20d)
    if 'market_return' in df.columns:
        df['market_volatility_20d'] = df['market_return'].rolling(window=20, min_periods=5).std().fillna(0.0)
    else:
        df['market_volatility_20d'] = 0.0
        
    # 4. Rolling Beta (60d & 252d) & Downside Beta
    if 'daily_return' in df.columns and 'market_return' in df.columns:
        ret = df['daily_return']
        mkt = df['market_return']
        
        cov_60 = ret.rolling(60, min_periods=20).cov(mkt)
        var_60 = mkt.rolling(60, min_periods=20).var()
        df['beta_60d'] = (cov_60 / var_60).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        
        cov_252 = ret.rolling(252, min_periods=60).cov(mkt)
        var_252 = mkt.rolling(252, min_periods=60).var()
        df['beta_252d'] = (cov_252 / var_252).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        
        # Downside Beta
        down_mask = mkt < 0
        ret_down = ret.where(down_mask)
        mkt_down = mkt.where(down_mask)
        cov_down = ret_down.rolling(252, min_periods=20).cov(mkt_down)
        var_down = mkt_down.rolling(252, min_periods=20).var()
        df['downside_beta_252d'] = (cov_down / var_down).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        
        # Cross-Asset Correlations (60d)
        df['corr_stock_kse100'] = ret.rolling(60, min_periods=20).corr(mkt).fillna(0.0)
    else:
        df['beta_60d'] = 1.0
        df['beta_252d'] = 1.0
        df['downside_beta_252d'] = 1.0
        df['corr_stock_kse100'] = 0.0

    if 'daily_return' in df.columns and 'pkr_usd_change_pct' in df.columns:
        df['corr_stock_pkr_usd'] = df['daily_return'].rolling(60, min_periods=20).corr(df['pkr_usd_change_pct']).fillna(0.0)
    else:
        df['corr_stock_pkr_usd'] = 0.0

    if 'daily_return' in df.columns and 'oil_return_pct' in df.columns:
        df['corr_stock_brent'] = df['daily_return'].rolling(60, min_periods=20).corr(df['oil_return_pct']).fillna(0.0)
    else:
        df['corr_stock_brent'] = 0.0

    if 'daily_return' in df.columns and 'sbp_policy_rate' in df.columns:
        df['corr_stock_policy_rate'] = df['daily_return'].rolling(60, min_periods=20).corr(df['sbp_policy_rate']).fillna(0.0)
    else:
        df['corr_stock_policy_rate'] = 0.0

    if 'daily_return' in df.columns and 'gold_return_pct' in df.columns:
        df['corr_stock_gold'] = df['daily_return'].rolling(60, min_periods=20).corr(df['gold_return_pct']).fillna(0.0)
    else:
        df['corr_stock_gold'] = 0.0

    # 5. Volume & Turnover
    if 'free_float' in df.columns:
        mask_ff = (df['free_float'] > 0) & df['free_float'].notna()
        df['turnover_ratio'] = 0.0
        if mask_ff.any():
            df.loc[mask_ff, 'turnover_ratio'] = df.loc[mask_ff, 'volume'] / df.loc[mask_ff, 'free_float']
    else:
        df['turnover_ratio'] = 0.0
        
    df['pv_trend_momentum'] = (df['close'] - df['open']) * df['volume']

    # 6. Oscillators (Stochastic, Williams %R, CCI, Bollinger Width)
    low_14 = df['low'].rolling(14, min_periods=5).min()
    high_14 = df['high'].rolling(14, min_periods=5).max()
    denom_14 = (high_14 - low_14).replace(0, np.nan)
    
    df['stochastic_k'] = ((df['close'] - low_14) / denom_14 * 100).fillna(50.0)
    df['stochastic_d'] = df['stochastic_k'].rolling(3, min_periods=1).mean().fillna(50.0)
    df['williams_r'] = ((high_14 - df['close']) / denom_14 * -100).fillna(-50.0)
    
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    sma_tp = tp.rolling(14, min_periods=5).mean()
    mad_tp = (tp - sma_tp).abs().rolling(14, min_periods=5).mean()


    df['cci_14'] = ((tp - sma_tp) / (0.015 * mad_tp.replace(0, np.nan))).fillna(0.0)
    
    if 'bollinger_hband' in df.columns and 'bollinger_lband' in df.columns and 'bollinger_mavg' in df.columns:
        df['bollinger_width'] = ((df['bollinger_hband'] - df['bollinger_lband']) / df['bollinger_mavg'].replace(0, np.nan)).fillna(0.0)
    else:
        df['bollinger_width'] = 0.0

    # 7. Microstructure Volatility (Parkinson, Garman-Klass) & Higher Order Moments
    hl_ratio = np.log((df['high'] / df['low'].replace(0, np.nan)).clip(lower=1e-6))
    co_ratio = np.log((df['close'] / df['open'].replace(0, np.nan)).clip(lower=1e-6))

    
    parkinson_sq = (hl_ratio ** 2) / (4.0 * np.log(2.0))
    garman_klass_sq = 0.5 * (hl_ratio ** 2) - (2.0 * np.log(2.0) - 1.0) * (co_ratio ** 2)
    
    df['parkinson_volatility_20d'] = np.sqrt(parkinson_sq.rolling(20, min_periods=5).mean()).fillna(0.0)
    df['garman_klass_volatility_20d'] = np.sqrt(garman_klass_sq.rolling(20, min_periods=5).mean()).fillna(0.0)
    
    if 'daily_return' in df.columns:
        df['return_skewness_20d'] = df['daily_return'].rolling(20, min_periods=10).skew().fillna(0.0)
        df['return_kurtosis_20d'] = df['daily_return'].rolling(20, min_periods=10).kurt().fillna(0.0)
    else:
        df['return_skewness_20d'] = 0.0
        df['return_kurtosis_20d'] = 0.0

    # 8. Market Breadth & Sector Relative Performance
    if 'advancing_stocks_pct' in df.columns and 'declining_stocks_pct' in df.columns:
        df['advance_decline_line'] = (df['advancing_stocks_pct'] - df['declining_stocks_pct']).cumsum().fillna(0.0)
    else:
        df['advance_decline_line'] = 0.0

    if 'daily_return' in df.columns and 'sector_return_pct' in df.columns:
        df['stock_relative_strength_sector'] = df['daily_return'] - df['sector_return_pct'].fillna(0.0)
    else:
        df['stock_relative_strength_sector'] = 0.0

    # 9. Chaikin Money Flow (CMF 20d)
    hl_diff = (df['high'] - df['low']).replace(0, np.nan)
    mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / hl_diff
    mf_volume = mf_multiplier.fillna(0.0) * df['volume']
    df['chaikin_money_flow_20d'] = (mf_volume.rolling(20, min_periods=5).sum() / df['volume'].rolling(20, min_periods=5).sum().replace(0, np.nan)).fillna(0.0)

    # 10. Calendar Seasonality
    if 'date' in df.columns:
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_year_end_season'] = df['date'].dt.month.isin([6, 12]).astype(int)

    # 11. Valuation Relative to History (Rolling Averages & Percentile Ranks)
    if 'pe_ratio' in df.columns:
        pe_clean = df['pe_ratio'].replace([0, np.inf, -np.inf], np.nan)
        df['pe_1y_avg'] = pe_clean.rolling(252, min_periods=60).mean().fillna(0.0)
        df['pe_3y_avg'] = pe_clean.rolling(252 * 3, min_periods=100).mean().fillna(0.0)
        df['pe_5y_avg'] = pe_clean.rolling(252 * 5, min_periods=200).mean().fillna(0.0)
        df['pe_percentile_3y'] = pe_clean.rolling(252 * 3, min_periods=100).rank(pct=True).fillna(0.5)
    else:
        df['pe_1y_avg'] = 0.0
        df['pe_3y_avg'] = 0.0
        df['pe_5y_avg'] = 0.0
        df['pe_percentile_3y'] = 0.5

    if 'pb_ratio' in df.columns:
        pb_clean = df['pb_ratio'].replace([0, np.inf, -np.inf], np.nan)
        df['pb_percentile_3y'] = pb_clean.rolling(252 * 3, min_periods=100).rank(pct=True).fillna(0.5)
    else:
        df['pb_percentile_3y'] = 0.5

    if 'dividend_yield' in df.columns:
        dy_clean = df['dividend_yield'].replace([np.inf, -np.inf], np.nan)
        df['dividend_yield_percentile_3y'] = dy_clean.rolling(252 * 3, min_periods=100).rank(pct=True).fillna(0.5)
    else:
        df['dividend_yield_percentile_3y'] = 0.5

    # 12. Market Psychology Proxies
    vix_norm = (df['vix_close'] / 100.0) if 'vix_close' in df.columns else 0.2
    parkinson_norm = df['parkinson_volatility_20d'].fillna(0.0)
    ret_neg_impact = (-df['daily_return'].clip(upper=0.0)).fillna(0.0)
    df['fear_index_proxy'] = (0.4 * vix_norm + 0.4 * parkinson_norm + 0.2 * ret_neg_impact).fillna(0.0)

    rsi_norm = (df['rsi_14'] / 100.0) if 'rsi_14' in df.columns else 0.5
    rel_vol_norm = (df['relative_volume'] / 3.0).clip(upper=1.0) if 'relative_volume' in df.columns else 0.5
    ret_pos_impact = (df['daily_return'].clip(lower=0.0)).fillna(0.0)
    df['greed_index_proxy'] = (0.4 * rsi_norm + 0.4 * rel_vol_norm + 0.2 * ret_pos_impact).fillna(0.0)

    high_vol = (df['volume'] > 1.5 * df['volume'].rolling(20, min_periods=5).mean()).astype(int)
    sharp_down = (df['daily_return'] < -0.015).astype(int)
    df['panic_selling_proxy'] = (high_vol * sharp_down).fillna(0)

    turnover = df['turnover_ratio'] if 'turnover_ratio' in df.columns else 0.0
    spread = df['daily_spread'] if 'daily_spread' in df.columns else 0.0
    df['short_term_speculation_proxy'] = (turnover * spread).fillna(0.0)

    return df


def _load_political_features() -> pd.DataFrame:
    """
    Load political and geopolitical flag columns from macro_indicators.
    Returns a DataFrame with [date, election_flag, fatf_greylist_flag, ...] columns.
    If the columns don't exist yet (first run), returns an empty DataFrame.
    """
    pol_cols = [
        'election_flag', 'fatf_greylist_flag', 'government_stability_score',
        'political_uncertainty_score', 'india_pakistan_tension_flag', 'middle_east_conflict_flag',
    ]
    try:
        # Check which columns actually exist in the table
        with engine.connect() as conn:
            existing = [
                row[0] for row in conn.execute(
                    text("""SELECT column_name FROM information_schema.columns
                            WHERE table_name='macro_indicators'
                            AND column_name = ANY(:cols)"""),
                    {"cols": pol_cols}
                ).fetchall()
            ]
        if not existing:
            # Columns not yet in DB — generate directly from the module and return
            logger.info("Political flag columns not in DB yet; generating in-memory.")
            from src.psx_predictor.data.feature_political_events import (
                _build_election_flag, _build_binary_flag_from_windows,
                _build_ordinal_from_periods, _build_political_uncertainty,
                FATF_GREYLIST_WINDOWS, GOVERNMENT_STABILITY_PERIODS,
                INDIA_PAKISTAN_TENSION_WINDOWS, MIDDLE_EAST_CONFLICT_WINDOWS
            )
            dates = pd.date_range(start="2000-01-01", end=datetime.now().strftime("%Y-%m-%d"), freq="D")
            dates_idx = pd.DatetimeIndex(dates)
            pol_df = pd.DataFrame({"date": dates.date})
            pol_df['election_flag'] = _build_election_flag(dates_idx).values
            pol_df['fatf_greylist_flag'] = _build_binary_flag_from_windows(
                dates_idx, [(s, e) for s, e in FATF_GREYLIST_WINDOWS]).values
            pol_df['government_stability_score'] = _build_ordinal_from_periods(
                dates_idx, GOVERNMENT_STABILITY_PERIODS).values.astype(int)
            pol_df['political_uncertainty_score'] = _build_political_uncertainty(dates_idx).values
            pol_df['india_pakistan_tension_flag'] = _build_binary_flag_from_windows(
                dates_idx, INDIA_PAKISTAN_TENSION_WINDOWS).values
            pol_df['middle_east_conflict_flag'] = _build_binary_flag_from_windows(
                dates_idx, MIDDLE_EAST_CONFLICT_WINDOWS).values
            return pol_df

        select_cols = ", ".join(["date"] + existing)
        with engine.connect() as conn:
            pol_df = pd.read_sql(
                text(f"SELECT {select_cols} FROM macro_indicators ORDER BY date ASC"),
                conn
            )
        return pol_df
    except Exception as e:
        logger.warning(f"_load_political_features failed: {e}")
        return pd.DataFrame()


def _load_pakistan_activity() -> pd.DataFrame:
    """
    Load Pakistan real economy activity columns from macro_indicators.
    Returns a DataFrame with [date, cement_dispatches_mt, auto_sales_total, ...] columns.
    If the columns don't exist yet (first run), generates synthetic data in-memory.
    """
    from datetime import datetime as dt
    activity_cols = ['cement_dispatches_mt', 'auto_sales_total', 'electricity_gen_gwh', 'wheat_procurement_mt']
    try:
        with engine.connect() as conn:
            existing = [
                row[0] for row in conn.execute(
                    text("""SELECT column_name FROM information_schema.columns
                            WHERE table_name='macro_indicators'
                            AND column_name = ANY(:cols)"""),
                    {"cols": activity_cols}
                ).fetchall()
            ]
        if not existing:
            logger.info("Pakistan activity columns not in DB yet; generating synthetic in-memory.")
            from src.psx_predictor.data.fetch_pakistan_activity import (
                _synthetic_cement, _synthetic_auto, _synthetic_electricity, _synthetic_wheat
            )
            full = pd.date_range(start="2005-01-01", end=dt.now().strftime("%Y-%m-%d"), freq="D")
            combined = pd.DataFrame({"date": pd.to_datetime(full.date)})
            for syn_fn in [_synthetic_cement, _synthetic_auto, _synthetic_electricity, _synthetic_wheat]:
                sdf = syn_fn()
                sdf['date'] = pd.to_datetime(sdf['date']) + pd.DateOffset(months=1)
                sdf['date'] = pd.to_datetime(sdf['date'])
                combined = pd.merge_asof(combined.sort_values('date'), sdf.sort_values('date'),
                                         on='date', direction='backward')
            combined['date'] = combined['date'].dt.date
            return combined

        select_cols = ", ".join(["date"] + existing)
        with engine.connect() as conn:
            act_df = pd.read_sql(
                text(f"SELECT {select_cols} FROM macro_indicators ORDER BY date ASC"),
                conn
            )
        return act_df
    except Exception as e:
        logger.warning(f"_load_pakistan_activity failed: {e}")
        return pd.DataFrame()


def build_features(ticker: str) -> pd.DataFrame:
    """Orchestrates the entire feature engineering pipeline."""
    df = load_data(ticker)
    if df.empty:
        return df
    
    df['date'] = pd.to_datetime(df['date'])
        
    if 'adjusted_close' not in df.columns or df['adjusted_close'].isna().all():
        logger.info("adjusted_close column missing or all null; populating with fallback close.")
        df['adjusted_close'] = df['close']
    else:
        df['adjusted_close'] = df['adjusted_close'].fillna(df['close'])

    logger.info(f"Generating technical features for {ticker}...")
    
    # 1. Technical Indicators
    df = calculate_sma(df, col='close', windows=[7, 21, 50])
    df = calculate_rsi(df, col='close', window=14)
    df = calculate_macd(df, col='close')
    
    # 2. Daily Returns and Lag Features
    df = calculate_lag_features(df, col='close', lags=[1, 2, 3, 5, 10])
    
    # 2.5 New Distinct Features (Volatility, Volume Context, Time)
    df = calculate_bollinger_bands(df, col='close', window=20)
    df = calculate_vwap(df)
    df = extract_time_features(df)
    df = calculate_obv(df)
    df = calculate_relative_volume(df)
    df = calculate_realized_volatility(df)
    df = calculate_atr(df)
    # df = encode_ticker(df, ticker) # Removed in favor of native categorical/embeddings
    df = merge_market_index(df, ticker)
    
    # 3. Merge Sentiment Data with Decay
    df = merge_sentiment(df, ticker)

    # 3.1 Merge Advanced News & Topic Sentiment Features (Groups 19, 20, 26, 36, 40)
    try:
        from src.psx_predictor.data.feature_news_sentiment import generate_news_sentiment_features
        trading_dates = pd.to_datetime(df['date'])
        news_feat_df = generate_news_sentiment_features(ticker, trading_dates, engine)
        if not news_feat_df.empty:
            news_cols = [c for c in news_feat_df.columns if c != 'date']
            existing_news = [c for c in news_cols if c in df.columns]
            if existing_news:
                df.drop(columns=existing_news, inplace=True)
            df = pd.merge(df, news_feat_df, on='date', how='left')
            logger.info(f"Merged {len(news_cols)} news sentiment feature columns.")
    except Exception as e:
        logger.exception("Could not merge news sentiment features:")
    # 3.5 Merge Dividend Data
    df = merge_dividends(df, ticker)
    
    # 3.6 Merge Fundamentals and Macro
    query = text("SELECT sector FROM stock_metadata WHERE ticker = :ticker")
    with engine.connect() as conn:
        sector = conn.execute(query, {"ticker": ticker.upper()}).scalar() or ""
        
    df = merge_fundamentals(df, ticker)
    df = merge_macro_indicators(df, ticker, sector)

    # 3.7 Corporate Event Features (from PUCARS — P1-A)
    try:
        trading_dates = pd.to_datetime(df['date'])
        event_df = generate_event_features(ticker, trading_dates)
        if not event_df.empty:
            event_df['date'] = pd.to_datetime(event_df['date'])
            # Drop any duplicated event columns that may already exist
            event_cols = [c for c in event_df.columns if c != 'date']
            existing_event_cols = [c for c in event_cols if c in df.columns]
            if existing_event_cols:
                df.drop(columns=existing_event_cols, inplace=True)
            df = pd.merge(df, event_df, on='date', how='left')
            for ec in event_cols:
                if ec in df.columns:
                    df[ec] = df[ec].fillna(0)
            logger.info(f"Merged {len(event_cols)} corporate event feature columns for {ticker}.")
    except Exception as e:
        logger.warning(f"Could not merge corporate event features for {ticker}: {e}")

    # 3.75 Calendar Event Features (SBP MPC, budget, Ramadan, etc. — P2-B)
    try:
        trading_dates = pd.to_datetime(df['date'])
        cal_df = generate_calendar_features(trading_dates)
        cal_df['date'] = pd.to_datetime(cal_df['date'])
        cal_cols = [c for c in cal_df.columns if c != 'date']
        existing_cal_cols = [c for c in cal_cols if c in df.columns]
        if existing_cal_cols:
            df.drop(columns=existing_cal_cols, inplace=True)
        df = pd.merge(df, cal_df, on='date', how='left')
        for cc in cal_cols:
            if cc in df.columns:
                df[cc] = df[cc].fillna(0)
        logger.info(f"Merged {len(cal_cols)} calendar event feature columns.")
    except Exception as e:
        logger.warning(f"Could not merge calendar event features: {e}")

    # 3.76 Political & Geopolitical Event Flags (Tier-4 Phase 2D)
    # Columns: election_flag, fatf_greylist_flag, government_stability_score,
    #          political_uncertainty_score, india_pakistan_tension_flag, middle_east_conflict_flag
    try:
        pol_df = _load_political_features()
        if pol_df is not None and not pol_df.empty:
            pol_df['date'] = pd.to_datetime(pol_df['date'])
            pol_cols = [c for c in pol_df.columns if c != 'date']
            existing_pol = [c for c in pol_cols if c in df.columns]
            if existing_pol:
                df.drop(columns=existing_pol, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            df = pd.merge_asof(
                df.sort_values('date'),
                pol_df.sort_values('date'),
                on='date',
                direction='backward',
            )
            for pc in pol_cols:
                if pc in df.columns:
                    df[pc] = df[pc].fillna(0)
            logger.info(f"Merged {len(pol_cols)} political/geopolitical flag columns.")
    except Exception as e:
        logger.warning(f"Could not merge political event features: {e}")

    # 3.77 Pakistan Real Economy Activity (Tier-3 Phase 2C)
    # Columns: cement_dispatches_mt, auto_sales_total, electricity_gen_gwh, wheat_procurement_mt
    try:
        pka_df = _load_pakistan_activity()
        if pka_df is not None and not pka_df.empty:
            pka_df['date'] = pd.to_datetime(pka_df['date'])
            pka_cols = [c for c in pka_df.columns if c != 'date']
            existing_pka = [c for c in pka_cols if c in df.columns]
            if existing_pka:
                df.drop(columns=existing_pka, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            df = pd.merge_asof(
                df.sort_values('date'),
                pka_df.sort_values('date'),
                on='date',
                direction='backward',
            )
            logger.info(f"Merged {len(pka_cols)} Pakistan activity columns.")
    except Exception as e:
        logger.warning(f"Could not merge Pakistan activity features: {e}")

    # 3.8 Spread Features
    df['daily_spread'] = (df['high'] - df['low']) / df['close']
    high_low_diff = df['high'] - df['low']
    df['close_pos'] = np.where(high_low_diff == 0, 0.5, (df['close'] - df['low']) / high_low_diff)
    
    # 3.9 Computed PDF Features
    df = calculate_computed_pdf_features(df)

    
    # 4. Handle Missing Values
    # Fill remaining rolling percentile/macro/event NaNs in numeric columns with 0.0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    non_num = [c for c in df.columns if c != 'date' and c not in numeric_cols]
    if non_num:
        df[non_num] = df[non_num].fillna("")

    initial_len = len(df)
    df.dropna(subset=['close', 'date'], inplace=True)
    df = df.reset_index(drop=True)
    logger.info(f"Retained {len(df)} records for {ticker}.")


    
    # P0-A: RETAIN open/high/low/volume in master CSV (previously dropped — now preserved per PDF Group #27)
    # DO NOT drop OHLCV columns — they are required features

    # P0-A: Derived volume/liquidity features
    df['daily_traded_value'] = df['close'] * df['volume']  # PDF Group #18
    df['adv_20d'] = df['volume'].rolling(20, min_periods=5).mean().fillna(0.0)  # Average Daily Volume

    # P0-A: Additional return horizons
    df['return_5d'] = df['close'].pct_change(periods=5).fillna(0.0)   # PDF Group #27
    df['return_20d'] = df['close'].pct_change(periods=20).fillna(0.0)  # PDF Group #27

    # P0-A: Historical volatility (annualized rolling 20d std of log returns) — distinct from return_vol_20
    log_ret = np.log(df['close'] / df['close'].shift(1))
    df['historical_volatility_20d'] = (log_ret.rolling(20, min_periods=5).std() * np.sqrt(252)).fillna(0.0)

    # P0-A: ADX (14-day Average Directional Index)
    df = calculate_adx(df, window=14)

    # P0-B Computed: Reserve changes and coverage
    if 'sbp_reserves' in df.columns:
        df['reserve_changes'] = df['sbp_reserves'].diff().fillna(0.0)
    if 'total_fx_reserves' in df.columns and 'imports_usd_m' in df.columns:
        monthly_imports = (df['imports_usd_m'] / 30.4).replace(0, np.nan)
        df['reserve_import_coverage'] = (df['total_fx_reserves'] / monthly_imports).fillna(0.0)

    # P0-B Computed: Volatility sub-groups (PDF Group #37)
    if 'banking_sector_return_pct' in df.columns:
        df['sector_volatility_20d'] = df['banking_sector_return_pct'].rolling(20, min_periods=5).std().fillna(0.0)
    if 'brent_oil_price' in df.columns:
        df['oil_volatility_20d'] = df['brent_oil_price'].pct_change().rolling(20, min_periods=5).std().fillna(0.0)
    if 'pib_10y' in df.columns:
        df['bond_volatility_20d'] = df['pib_10y'].diff().rolling(20, min_periods=5).std().fillna(0.0)

    # P1-F Computed: Surprise & Expectations features (Actual vs Expected vs Surprise, no leakage)
    if 'cpi_headline' in df.columns:
        df['cpi_expected'] = df['cpi_headline'].rolling(3, min_periods=1).mean().shift(1).fillna(df['cpi_headline'])
        df['cpi_surprise'] = (df['cpi_headline'] - df['cpi_expected']).fillna(0.0)

    if 'sbp_policy_rate' in df.columns:
        df['policy_rate_expected'] = df['sbp_policy_rate'].shift(1).fillna(df['sbp_policy_rate'])
        df['policy_rate_surprise'] = (df['sbp_policy_rate'] - df['policy_rate_expected']).fillna(0.0)

    if 'trade_deficit_usd_m' in df.columns:
        df['trade_deficit_expected'] = df['trade_deficit_usd_m'].rolling(3, min_periods=1).mean().shift(1).fillna(df['trade_deficit_usd_m'])
        df['trade_deficit_surprise'] = (df['trade_deficit_usd_m'] - df['trade_deficit_expected']).fillna(0.0)

    if 'eps_trailing' in df.columns:
        df['eps_expected'] = df['eps_trailing'].rolling(252, min_periods=60).mean().shift(1).fillna(df['eps_trailing'])
        df['eps_consensus_surprise'] = (df['eps_trailing'] - df['eps_expected']).fillna(0.0)
        df['eps_qoq_surprise'] = df['eps_trailing'].diff(63).fillna(0.0)  # ~1 quarter = 63 trading days


    # P0-E: Final cleanup of any remaining _x/_y suffix duplicates from macro merge
    for col in list(df.columns):
        if col.endswith('_x'):
            base = col[:-2]
            y_col = base + '_y'
            if y_col in df.columns:
                # Prefer the macro_indicators version (_y from macro merge)
                df[base] = df[y_col].combine_first(df[col])
                df.drop(columns=[col, y_col], inplace=True, errors='ignore')

    # 5. Save finalized dataset
    final_path = os.path.join(PROCESSED_DIR, f"{ticker.lower()}_features.csv")
    df.to_csv(final_path, index=False)
    logger.info(f"Saved finalized feature dataset ready for ML to {final_path}")
    
    # Always export dedicated master CSV with attached raw text announcements & news
    master_filename = f"{ticker.upper()}_master.csv"
    master_path = os.path.join(PROCESSED_DIR, master_filename)
    try:
        df.to_csv(master_path, index=False)
        logger.info(f"Saved dedicated master dataset: {master_path}")
        try:
            from src.psx_predictor.data.export_raw_text_datasets import export_raw_text_files_for_ticker
            export_raw_text_files_for_ticker(ticker.upper())
            logger.info(f"Successfully attached date-matched raw text PUCARS & news to {master_path}")
        except Exception as e_exp:
            logger.warning(f"Could not attach raw text datasets to {master_path}: {e_exp}")
    except Exception as e:
        logger.error(f"Error saving {master_path}: {e}")

    # 6. Execute Data Quality Gate
    try:
        from src.psx_predictor.data.quality_gate import run_quality_checks
        run_quality_checks(df, ticker=ticker, strict=False)
    except Exception as e:
        logger.warning(f"Quality Gate check error for {ticker}: {e}")

    return df

from src.psx_predictor.db.repository import get_active_tickers

if __name__ == '__main__':
    tickers = get_active_tickers()
    logger.info(f"Building features for {len(tickers)} active tickers...")
    
    # A.2.3: One-time startup/build-time assertion for energy sector
    with engine.connect() as conn:
        all_sectors = conn.execute(text("SELECT DISTINCT sector FROM stock_metadata WHERE is_active=true")).fetchall()
        energy_count = sum(1 for (s,) in all_sectors if s and any(energy_kw in s.lower() for energy_kw in ['oil & gas', 'refinery', 'power generation']))
        if energy_count == 0:
            logger.warning("ASSERTION FAILED: Zero active tickers matched the energy-sector condition for the oil_return_pct feature gate!")
        else:
            logger.info(f"Verified {energy_count} sectors matched the energy-sector condition.")
            
    for ticker in tickers:
        build_features(ticker)
