import os
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, date
from typing import Optional
from sqlalchemy.orm import Session

from src.psx_predictor.db.connection import get_db, engine
from src.psx_predictor.db.models import StockDividend

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

class DividendScraper:
    """
    Scraper for fetching historical dividend data using Yahoo Finance API endpoints
    by appending '.KA' to the ticker for the Karachi Stock Exchange.
    """
    def __init__(self):
        pass # No longer needed since yfinance manages the session

    def fetch_dividends(self, ticker: str) -> pd.DataFrame:
        """
        Fetches historical dividends for a given ticker from Yahoo Finance.
        Returns a DataFrame with columns: ['ticker', 'ex_dividend_date', 'dividend_amount']
        """
        yf_ticker = f"{ticker.upper()}.KA"
        
        try:
            stock = yf.Ticker(yf_ticker)
            divs = stock.dividends
            
            if divs.empty:
                logger.info(f"No dividend events found for {ticker}.")
                return pd.DataFrame()
                
            records = []
            for date_idx, amount in divs.items():
                # yfinance returns pandas Timestamp in index
                ex_date = date_idx.date() if hasattr(date_idx, 'date') else pd.to_datetime(date_idx).date()
                records.append({
                    'ticker': ticker.upper(),
                    'ex_dividend_date': ex_date,
                    'dividend_amount': float(amount),
                    'dividend_type': 'Cash'
                })
                
            df = pd.DataFrame(records)
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch dividends for {ticker}: {e}")
            return pd.DataFrame()

    def sync_dividends(self, ticker: str) -> bool:
        """
        Fetches and upserts dividend data into the database.
        """
        df = self.fetch_dividends(ticker)
        if df.empty:
            return False
            
        logger.info(f"Fetched {len(df)} dividend records for {ticker}. Upserting to database...")
        
        try:
            with Session(engine) as session:
                for _, row in df.iterrows():
                    # Check if exists
                    existing = session.query(StockDividend).filter(
                        StockDividend.ticker == row['ticker'],
                        StockDividend.ex_dividend_date == row['ex_dividend_date']
                    ).first()
                    
                    if not existing:
                        new_div = StockDividend(
                            ticker=row['ticker'],
                            ex_dividend_date=row['ex_dividend_date'],
                            dividend_amount=row['dividend_amount'],
                            dividend_type=row['dividend_type']
                        )
                        session.add(new_div)
                    else:
                        existing.dividend_amount = row['dividend_amount']
                
                session.commit()
                logger.info(f"Successfully synced dividends for {ticker}.")
                return True
        except Exception as e:
            logger.error(f"Error upserting dividends for {ticker}: {e}")
            return False

if __name__ == "__main__":
    scraper = DividendScraper()
    for t in ['PSO', 'FFC', 'NBP', 'MEBL', 'OGDC', 'LUCK']:
        scraper.sync_dividends(t)
