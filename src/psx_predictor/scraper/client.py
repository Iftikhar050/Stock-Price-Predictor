import os
import logging
import requests
import pandas as pd
from io import StringIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional

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
    Uses the historical POST endpoint to retrieve the full OHLCV table.
    """
    BASE_URL = "https://dps.psx.com.pk/historical"
    
    def __init__(self, user_agent: Optional[str] = None):
        self.session = requests.Session()
        
        # Configure robust retry logic (urllib3 retries under the hood)
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        # Load user agent from env or default
        ua = user_agent or os.getenv(
            "USER_AGENT", 
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        
        # Custom headers vital for bypassing anti-bot measures on PSX DPS
        self.session.headers.update({
            "User-Agent": ua,
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
        })

    def fetch_raw_data(self, ticker: str) -> Optional[str]:
        """
        Sends a POST request to the PSX historical endpoint for a specific ticker
        and retrieves the HTML string containing the full data table.
        """
        data = {'symbol': ticker.upper()}
        try:
            logger.info(f"Fetching historical OHLCV data for {ticker} from {self.BASE_URL}")
            response = self.session.post(self.BASE_URL, data=data, timeout=15)
            response.raise_for_status()
            
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error while fetching data for {ticker}: {e}")
            return None

    def clean_and_format(self, raw_html: str, ticker: str) -> pd.DataFrame:
        """
        Cleans and formats the raw HTML table into a consistent Pandas DataFrame
        aligned with our OHLCV SQL schema.
        """
        if not raw_html or '<table' not in raw_html.lower():
            logger.warning(f"No valid HTML table found in the response for {ticker}.")
            return pd.DataFrame()

        try:
            # Parse the HTML tables in the response. pd.read_html requires lxml or html5lib
            dfs = pd.read_html(StringIO(raw_html))
            if not dfs:
                logger.warning(f"No table extracted by pandas for {ticker}.")
                return pd.DataFrame()
            
            df = dfs[0]
        except ValueError as e:
            logger.error(f"Failed to parse HTML tables for {ticker}: {e}")
            return pd.DataFrame()
        
        # The expected columns from the PSX Historical Table are:
        # ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
        # We need to map them to lowercase to match our schema
        df.columns = [col.lower() for col in df.columns]
        
        # Verify required columns are present
        required_cols = {"date", "open", "high", "low", "close", "volume"}
        if not required_cols.issubset(set(df.columns)):
            logger.error(f"Missing expected columns in the historical table for {ticker}. Found: {df.columns.tolist()}")
            return pd.DataFrame()

        # Clean and parse the string date (e.g., 'Jan 3, 2025') into a proper ISO YYYY-MM-DD date.
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
             
        df['ticker'] = ticker.upper()
        
        # Coerce numeric columns to Float/Int, discarding invalid strings (like '-') to NaN
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            # We remove commas in numbers like "4,496,408" before coercion
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Drop any records that failed date parsing
        df.dropna(subset=['date'], inplace=True)
        
        # Select and reorder to match the DB schema precisely
        df = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]
        
        return df

    def sync_ticker(self, ticker: str) -> bool:
        """
        End-to-End method: Fetches, cleans, and upserts a ticker's data into PostgreSQL.
        """
        raw_html = self.fetch_raw_data(ticker)
        if not raw_html:
            return False
            
        df = self.clean_and_format(raw_html, ticker)
        
        if df.empty:
            logger.warning(f"Cleaned DataFrame for {ticker} is empty. Skipping DB insert.")
            return False
            
        logger.info(f"Successfully cleaned data for {ticker}. Proceeding to upsert {len(df)} records.")
        
        # Call the repository function created in Step 1
        success = upsert_stock_data(df)
        return success
