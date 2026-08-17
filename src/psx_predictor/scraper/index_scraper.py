import logging
import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import text
from datetime import datetime
from src.psx_predictor.db.connection import engine
from src.psx_predictor.db.models import StockMarketIndex

logger = logging.getLogger(__name__)

def fetch_market_index(start_date: str = '2010-01-01', end_date: str = None) -> pd.DataFrame:
    """
    Fetches the daily KSE-100 index data.
    
    NOTE: Due to current anti-scraping measures on dps.psx.com.pk, this function 
    computes a synthetic equally-weighted proxy index from the existing EOD data in 
    the database to ensure the pipeline runs reliably. In a production environment with
    an official PSX data feed or a robust headless scraper, replace this block with
    the actual API call to fetch real KSE-100 OHLCV data.
    """
    logger.info("Fetching market index data (using synthetic proxy for Phase 1)...")
    
    query = """
        SELECT date, 
               AVG(open) as open,
               AVG(high) as high,
               AVG(low) as low,
               AVG(close) as close,
               SUM(volume) as volume
        FROM stock_eod_data
        GROUP BY date
        ORDER BY date ASC
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
        
    if df.empty:
        return pd.DataFrame()
        
    df['index_name'] = 'KSE100'
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

def sync_market_index():
    logger.warning("===============================================================")
    logger.warning("WARNING: The market index being synced is a SYNTHETIC PROXY.")
    logger.warning("This is NOT the real KSE-100 index. It is computed internally")
    logger.warning("from available ticker data due to PSX anti-scraping measures.")
    logger.warning("===============================================================")
    
    df = fetch_market_index()
    if df.empty:
        logger.warning("No market index data fetched.")
        return False
        
    records = df.to_dict(orient='records')
    stmt = insert(StockMarketIndex).values(records)
    
    update_dict = {
        'open': stmt.excluded.open,
        'high': stmt.excluded.high,
        'low': stmt.excluded.low,
        'close': stmt.excluded.close,
        'volume': stmt.excluded.volume
    }
    
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=['index_name', 'date'],
        set_=update_dict
    )

    try:
        with engine.begin() as conn:
            result = conn.execute(upsert_stmt)
            logger.info(f"Successfully upserted data into stock_market_index. Rows affected: {result.rowcount}")
        return True
    except Exception as e:
        logger.error(f"Error upserting market index: {e}")
        return False

if __name__ == "__main__":
    sync_market_index()
