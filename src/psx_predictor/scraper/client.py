import os
import logging
import pandas as pd
import yfinance as yf
from typing import Optional
from datetime import datetime, timedelta

from src.psx_predictor.db.repository import upsert_stock_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

class PSXScraper:
    """
    Scraper for the PSX Data Portal End-of-Day (EOD) data.
    Now utilizes yfinance to bypass PSX anti-bot measures.
    """
    def __init__(self, user_agent: Optional[str] = None):
        pass # No longer needed for yfinance

    def fetch_raw_data(self, ticker: str) -> pd.DataFrame:
        """
        Downloads historical data from Yahoo Finance for a specific ticker.
        Appends .KA suffix as required by Yahoo Finance for Karachi Stock Exchange.
        """
        yf_ticker = f"{ticker.upper()}.KA"
        logger.info(f"Fetching historical OHLCV data for {yf_ticker} from Yahoo Finance")
        
        try:
            # We fetch max available data to ensure history is complete
            # In a true production environment, we might just fetch the last year to save bandwidth if DB is mostly populated,
            # but yfinance is fast enough that fetching "max" is usually fine.
            data = yf.download(yf_ticker, period="max", progress=False)
            return data
        except Exception as e:
            logger.error(f"Network error while fetching data for {ticker}: {e}")
            return pd.DataFrame()

    def clean_and_format(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Cleans and formats the yfinance DataFrame into a consistent Pandas DataFrame
        aligned with our OHLCV SQL schema.
        """
        if df.empty:
            logger.warning(f"No valid data returned from yfinance for {ticker}.")
            return pd.DataFrame()

        # Handle yfinance multi-index columns if present (yfinance >= 0.2.x sometimes returns them)
        if isinstance(df.columns, pd.MultiIndex):
            # Drop the ticker level (level 1 usually)
            df.columns = df.columns.droplevel(1)
            
        df = df.reset_index()
        
        # Expected columns from yfinance: Date, Open, High, Low, Close, Adj Close, Volume
        df.columns = [str(col).lower() for col in df.columns]
        
        # We don't strictly need adj close right now since PSX historical is unadjusted usually
        required_cols = {"date", "open", "high", "low", "close", "volume"}
        if not required_cols.issubset(set(df.columns)):
            logger.error(f"Missing expected columns in the yfinance data for {ticker}. Found: {df.columns.tolist()}")
            return pd.DataFrame()

        df['date'] = pd.to_datetime(df['date']).dt.date
        df['ticker'] = ticker.upper()
        
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df.dropna(subset=['date', 'close'], inplace=True)
        
        df = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]
        
        return df

    def sync_ticker(self, ticker: str) -> bool:
        """
        End-to-End method: Fetches, cleans, and upserts a ticker's data into PostgreSQL.
        """
        raw_df = self.fetch_raw_data(ticker)
        if raw_df.empty:
            return False
            
        df = self.clean_and_format(raw_df, ticker)
        
        if df.empty:
            logger.warning(f"Cleaned DataFrame for {ticker} is empty. Skipping DB insert.")
            return False
            
        logger.info(f"Successfully cleaned data for {ticker}. Proceeding to upsert {len(df)} records.")
        
        success = upsert_stock_data(df)
        return success
