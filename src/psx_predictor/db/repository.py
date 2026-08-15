import logging
import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from src.psx_predictor.db.connection import engine
from src.psx_predictor.db.models import StockEODData, StockNews, StockNewsSentiment

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

def get_active_tickers() -> list[str]:
    """
    Returns a list of active tickers from the stock_metadata table.
    Acts as the single source of truth for the ticker universe.
    """
    from sqlalchemy import text
    query = text("SELECT ticker FROM stock_metadata WHERE is_active = true ORDER BY ticker ASC")
    try:
        with engine.connect() as conn:
            result = conn.execute(query).fetchall()
            return [row[0] for row in result]
    except Exception as e:
        logger.error(f"Error fetching active tickers: {e}")
        # Fallback for safety if table doesn't exist during early init
        return ['PSO', 'FFC', 'NBP', 'MEBL', 'OGDC', 'LUCK']

def upsert_stock_data(df: pd.DataFrame) -> bool:
    """
    Efficiently bulk upserts a Pandas DataFrame into the stock_eod_data PostgreSQL table.
    
    Uses INSERT ... ON CONFLICT DO UPDATE to ensure idempotency. If a row with
    the same (ticker, date) exists, it updates the OHLCV columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data. 
                           Expected columns: 'ticker', 'date', 'open', 'high', 'low', 'close', 'volume'
                           
    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    if df is None or df.empty:
        logger.warning("Provided DataFrame is empty. Nothing to insert.")
        return False

    required_columns = {'ticker', 'date', 'open', 'high', 'low', 'close', 'volume'}
    if not required_columns.issubset(set(df.columns)):
        missing = required_columns - set(df.columns)
        logger.error(f"DataFrame is missing required columns: {missing}")
        return False

    # Convert DataFrame to list of dictionaries for bulk insert
    # Ensure NaN/NaT are handled gracefully, though pandas to_dict usually handles it
    records = df.to_dict(orient='records')

    # Construct the PostgreSQL-specific insert statement
    stmt = insert(StockEODData).values(records)
    
    # Define the ON CONFLICT action
    # We conflict on the primary key: (ticker, date)
    update_dict = {
        'open': stmt.excluded.open,
        'high': stmt.excluded.high,
        'low': stmt.excluded.low,
        'close': stmt.excluded.close,
        'volume': stmt.excluded.volume
    }
    
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=['ticker', 'date'],
        set_=update_dict
    )

    try:
        # engine.begin() acts as a context manager that automatically starts
        # a transaction and commits at the end, rolling back on exceptions.
        with engine.begin() as conn:
            result = conn.execute(upsert_stmt)
            logger.info(f"Successfully upserted data into stock_eod_data. Rows affected: {result.rowcount}")
        return True
    except SQLAlchemyError as e:
        logger.error(f"Database error during upsert operation: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during upsert operation: {e}")
        return False

def upsert_stock_news(df: pd.DataFrame) -> bool:
    """
    Inserts raw news articles. Deduplication happens before this step, so we simply insert.
    """
    if df is None or df.empty:
        return False

    records = df.to_dict(orient='records')
    stmt = insert(StockNews).values(records)
    
    # URL and Ticker are not a composite PK, so if we wanted to avoid duplicate inserts on the DB level,
    # we would need a unique constraint on (url, ticker). For now, we trust the deduplicator.
    # However, to be safe, we can do a DO NOTHING on conflict if we add a unique constraint later.
    try:
        with engine.begin() as conn:
            conn.execute(stmt)
        return True
    except Exception as e:
        logger.error(f"Database error inserting news: {e}")
        return False

def upsert_news_sentiment(df: pd.DataFrame) -> bool:
    """
    Upserts daily aggregated sentiment into stock_news_sentiment.
    """
    if df is None or df.empty:
        return False

    records = df.to_dict(orient='records')
    stmt = insert(StockNewsSentiment).values(records)
    
    update_dict = {
        'sentiment_score': stmt.excluded.sentiment_score,
        'article_count': stmt.excluded.article_count
    }
    
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=['ticker', 'date'],
        set_=update_dict
    )

    try:
        with engine.begin() as conn:
            conn.execute(upsert_stmt)
        return True
    except Exception as e:
        logger.error(f"Database error upserting sentiment: {e}")
        return False

