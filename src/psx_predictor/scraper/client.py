import os
import logging
import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
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

    def fetch_psx_dps_fallback(self, ticker: str) -> pd.DataFrame:
        """
        Fallback EOD source for tickers Yahoo Finance doesn't carry under the
        `.KA` suffix (confirmed to happen for some smaller-cap PSX names).
        Scrapes dps.psx.com.pk's own historical endpoint directly. Note: this
        endpoint only serves the trailing ~1 year for individual equities (no
        pagination parameters are honored) so coverage will be shorter than
        the yfinance path - real recent data is still far better than none.
        """
        url = "https://dps.psx.com.pk/historical"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
        }
        logger.info(f"Falling back to PSX DPS historical data for {ticker} (yfinance had no data)")
        try:
            r = requests.post(url, data={"symbol": ticker.upper()}, headers=headers, timeout=15)
            if r.status_code != 200:
                return pd.DataFrame()

            soup = BeautifulSoup(r.text, "html.parser")
            tables = soup.find_all("table")
            if not tables:
                return pd.DataFrame()

            rows = []
            for row in tables[0].find_all("tr")[1:]:
                cols = [c.text.strip() for c in row.find_all(["td", "th"])]
                if len(cols) < 6:
                    continue
                try:
                    rows.append({
                        "date": datetime.strptime(cols[0], "%b %d, %Y").date(),
                        "open": float(cols[1].replace(",", "")),
                        "high": float(cols[2].replace(",", "")),
                        "low": float(cols[3].replace(",", "")),
                        "close": float(cols[4].replace(",", "")),
                        "volume": float(cols[5].replace(",", "")),
                    })
                except (ValueError, IndexError):
                    continue

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows)
            df["ticker"] = ticker.upper()
            return df[["ticker", "date", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.error(f"Error fetching PSX DPS fallback data for {ticker}: {e}")
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
            df = self.fetch_psx_dps_fallback(ticker)
        else:
            df = self.clean_and_format(raw_df, ticker)
            # yfinance can go permanently stale for a symbol after a corporate
            # action (confirmed 2026-09 for ENGRO: its .KA feed stopped updating
            # entirely after the Dec 2024 Dawood Hercules/Engro/DH Partners merger,
            # while PSX's own site kept publishing fresh daily bars) without ever
            # returning empty - the old check above only caught a fully-dead feed,
            # not a stale-but-non-empty one. Detect staleness and top up with
            # whatever newer bars the PSX DPS fallback has.
            if not df.empty:
                latest = pd.to_datetime(df['date']).max()
                if (pd.Timestamp.now().normalize() - latest).days > 7:
                    fallback_df = self.fetch_psx_dps_fallback(ticker)
                    if not fallback_df.empty:
                        fallback_df = fallback_df[pd.to_datetime(fallback_df['date']) > latest]
                        if not fallback_df.empty:
                            logger.info(
                                f"{ticker}: yfinance feed stale since {latest.date()} - "
                                f"topping up with {len(fallback_df)} newer PSX DPS row(s)."
                            )
                            df = pd.concat([df, fallback_df], ignore_index=True)

        if df.empty:
            logger.warning(f"No EOD data available for {ticker} from either Yahoo Finance or PSX DPS. Skipping DB insert.")
            return False

        df = self._normalize_ohlc_bounds(df, ticker)

        logger.info(f"Successfully cleaned data for {ticker}. Proceeding to upsert {len(df)} records.")

        success = upsert_stock_data(df)
        return success

    def _normalize_ohlc_bounds(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Yahoo Finance's split/dividend adjustment math occasionally produces
        bars where open/close fall slightly outside [low, high] - confirmed
        via a direct DB audit (~3,500 rows across 102 tickers, mostly
        pre-2011 history, each violation tiny relative to the price level:
        an adjustment-rounding artifact, not corrupted data). Rather than
        ship an internally-inconsistent bar, widen high/low to the true
        max/min across all four fields so every bar is self-consistent by
        construction. This never invents a price - it only ensures the
        reported range actually contains the reported open/close.
        """
        ohlc_cols = ["open", "high", "low", "close"]
        if not set(ohlc_cols).issubset(df.columns):
            return df
        violations = ((df["open"] < df["low"]) | (df["open"] > df["high"]) |
                      (df["close"] < df["low"]) | (df["close"] > df["high"])).sum()
        if violations:
            logger.info(f"Normalizing {violations} internally-inconsistent OHLC bar(s) for {ticker}.")
        df["high"] = df[ohlc_cols].max(axis=1)
        df["low"] = df[ohlc_cols].min(axis=1)
        return df
