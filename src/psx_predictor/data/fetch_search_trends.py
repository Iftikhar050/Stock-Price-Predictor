"""
fetch_search_trends.py
-----------------------
Fetches Google Trends search interest data for PSX company names and keywords
using pytrends (free, no API key required).

Source: Google Trends (trends.google.com/trends) via pytrends library.
Frequency: Weekly (Google Trends minimum resolution).
Date alignment: Data is published weekly; use forward-fill to daily.
Leakage: None — only past weekly interest values are used, forward-filled.

PDF Groups Covered: #26 (Investor Sentiment — Search Trends)
"""

import os
import sys
import logging
import time
import pandas as pd
from sqlalchemy import text
from datetime import datetime, timedelta

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

from src.psx_predictor.db.connection import engine

logger = logging.getLogger("SearchTrendsFetcher")
logger.setLevel(logging.INFO)

TICKER_SEARCH_TERMS = {
    "PSO": ["PSO Pakistan", "Pakistan State Oil"],
    "MEBL": ["Meezan Bank", "MEBL stock"],
    "MARKET": ["KSE 100", "Pakistan stock market", "PSX"],
}


def fetch_search_trends(ticker: str, start_date: str = "2015-01-01") -> pd.DataFrame:
    """
    Fetches Google Trends weekly search interest for PSX company names.
    Requires: pip install pytrends

    Args:
        ticker:     PSX ticker symbol ('PSO', 'MEBL').
        start_date: Historical start date string.

    Returns:
        DataFrame with columns: date, search_trend_<ticker>, search_trend_market
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        logger.error("pytrends not installed. Run: pip install pytrends")
        return pd.DataFrame()

    logger.info(f"Fetching Google Trends for {ticker}...")
    terms = TICKER_SEARCH_TERMS.get(ticker.upper(), [ticker])
    market_terms = TICKER_SEARCH_TERMS.get("MARKET", ["KSE 100"])

    try:
        pytrends = TrendReq(hl='en-US', tz=300, timeout=(10, 25))

        for attempt in range(3):
            try:
                # Company interest
                pytrends.build_payload(terms[:1], cat=0, timeframe=f"{start_date} {datetime.today().strftime('%Y-%m-%d')}", geo='PK')
                time.sleep(1)  # be polite to Google
                df_company = pytrends.interest_over_time()

                # Market interest
                pytrends.build_payload(market_terms[:1], cat=0, timeframe=f"{start_date} {datetime.today().strftime('%Y-%m-%d')}", geo='PK')
                time.sleep(1)
                df_market = pytrends.interest_over_time()
                break # Success
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                time.sleep(5 * (attempt + 1))
        else:
            logger.error(f"Failed to fetch Google Trends for {ticker} after 3 attempts.")
            return pd.DataFrame()

        if df_company.empty and df_market.empty:
            logger.warning(f"Google Trends returned empty data for {ticker}.")
            return pd.DataFrame()

        result = pd.DataFrame()

        if not df_company.empty:
            df_company = df_company.reset_index()[['date', terms[0]]].rename(
                columns={terms[0]: f'search_trend_{ticker.lower()}'}
            )
            df_company['date'] = pd.to_datetime(df_company['date'])
            result = df_company

        if not df_market.empty:
            df_market = df_market.reset_index()[['date', market_terms[0]]].rename(
                columns={market_terms[0]: 'search_trend_kse'}
            )
            df_market['date'] = pd.to_datetime(df_market['date'])
            if result.empty:
                result = df_market
            else:
                result = pd.merge(result, df_market, on='date', how='outer')

        result = result.sort_values('date').reset_index(drop=True)
        logger.info(f"Fetched {len(result)} weekly Google Trends data points for {ticker}.")
        return result

    except Exception as e:
        logger.exception(f"Error fetching Google Trends for {ticker}:")
        return pd.DataFrame()



def sync_search_trends_to_db(ticker: str) -> bool:
    """Fetches and stores Google Trends data in macro_indicators table."""
    df = fetch_search_trends(ticker)
    if df.empty:
        return False

    # Ensure columns exist in macro_indicators
    trend_col = f'search_trend_{ticker.lower()}'
    with engine.connect() as conn:
        for col in [trend_col, 'search_trend_kse']:
            conn.execute(text(f'ALTER TABLE macro_indicators ADD COLUMN IF NOT EXISTS {col} DOUBLE PRECISION;'))
        conn.commit()

    update_sql = text(f"""
        INSERT INTO macro_indicators (date, {trend_col}, search_trend_kse, created_at)
        VALUES (:date, :ticker_trend, :kse_trend, NOW())
        ON CONFLICT (date) DO UPDATE SET
            {trend_col} = EXCLUDED.{trend_col},
            search_trend_kse = EXCLUDED.search_trend_kse
    """)

    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(update_sql, {
                "date": row['date'].date(),
                "ticker_trend": float(row.get(trend_col, 0) or 0),
                "kse_trend": float(row.get('search_trend_kse', 0) or 0),
            })
        conn.commit()

    logger.info(f"Synced {len(df)} Google Trends records for {ticker}.")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for t in ["PSO", "MEBL"]:
        sync_search_trends_to_db(t)
