import logging
import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from src.psx_predictor.db.connection import engine
from src.psx_predictor.db.models import (
    StockEODData, StockNews, StockNewsSentiment, StockFundamentals,
    MacroIndicators, CorporateEvent, TopicSentimentDaily
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

def get_active_tickers(allow_fallback: bool = False) -> list[str]:
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
        if not allow_fallback:
            raise RuntimeError(f"Database error while fetching tickers, and allow_fallback=False. {e}")
        # Fallback for safety if table doesn't exist during early init
        logger.warning("Falling back to hardcoded list of 6 tickers.")
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

    records = df.to_dict(orient='records')
    stmt = insert(StockEODData).values(records)
    
    update_dict = {
        'open': stmt.excluded.open,
        'high': stmt.excluded.high,
        'low': stmt.excluded.low,
        'close': stmt.excluded.close,
        'adjusted_close': stmt.excluded.adjusted_close,
        'volume': stmt.excluded.volume
    }
    
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=['ticker', 'date'],
        set_=update_dict
    )

    try:
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

def upsert_stock_fundamentals(df: pd.DataFrame) -> bool:
    """
    Upserts fundamental data into stock_fundamentals.
    Only updates columns that are explicitly provided in df.columns.
    """
    if df is None or df.empty:
        return False

    records = df.to_dict(orient='records')
    stmt = insert(StockFundamentals).values(records)
    
    update_dict = {}
    for col in df.columns:
        if col not in ['ticker', 'report_date'] and hasattr(stmt.excluded, col):
            update_dict[col] = getattr(stmt.excluded, col)
            
    if not update_dict:
        return False

    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=['ticker', 'report_date'],
        set_=update_dict
    )

    try:
        with engine.begin() as conn:
            conn.execute(upsert_stmt)
        return True
    except Exception as e:
        logger.error(f"Database error upserting fundamentals: {e}")
        return False

def upsert_macro_indicators(df: pd.DataFrame) -> bool:
    """
    Upserts macro data into macro_indicators.
    Only updates columns that are explicitly provided in df.columns to prevent overwriting other sources.
    Filters out any extra DataFrame columns not defined on the MacroIndicators model.
    """
    if df is None or df.empty:
        return False

    valid_table_cols = set(MacroIndicators.__table__.columns.keys())
    valid_cols = [c for c in df.columns if c in valid_table_cols]
    if not valid_cols or 'date' not in valid_cols:
        logger.warning("No valid MacroIndicators table columns found in DataFrame.")
        return False

    clean_df = df[valid_cols]
    records = clean_df.to_dict(orient='records')
    stmt = insert(MacroIndicators).values(records)
    
    update_dict = {}
    for col in valid_cols:
        if col != 'date' and hasattr(stmt.excluded, col):
            update_dict[col] = getattr(stmt.excluded, col)
            
    if not update_dict:
        logger.warning("No updateable columns found in DataFrame.")
        return False
    
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=['date'],
        set_=update_dict
    )

    try:
        with engine.begin() as conn:
            conn.execute(upsert_stmt)
        return True
    except Exception as e:
        logger.error(f"Database error upserting macro indicators: {e}")
        return False


def upsert_corporate_events(df: pd.DataFrame) -> bool:
    """
    Upserts structured corporate event records into the corporate_events table.
    Deduplicates on (symbol, published_at, title) via ON CONFLICT DO UPDATE.
    """
    if df is None or df.empty:
        logger.warning("No corporate events to upsert.")
        return False

    required = {'symbol', 'published_at', 'trading_date', 'title'}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        logger.error(f"Corporate events DataFrame missing columns: {missing}")
        return False

    records = df.to_dict(orient='records')
    stmt = insert(CorporateEvent).values(records)

    try:
        with engine.begin() as conn:
            conn.execute(stmt)
        logger.info(f"Inserted {len(records)} corporate event records.")
        return True
    except Exception as e:
        logger.error(f"Database error upserting corporate events: {e}")
        return False


def upsert_topic_sentiment(df: pd.DataFrame) -> bool:
    """
    Upserts daily aggregated topic sentiment into topic_sentiment_daily.
    PK: (date, topic).
    """
    if df is None or df.empty:
        return False

    required = {'date', 'topic', 'sentiment_score', 'article_count'}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        logger.error(f"Topic sentiment DataFrame missing columns: {missing}")
        return False

    records = df.to_dict(orient='records')
    stmt = insert(TopicSentimentDaily).values(records)

    update_dict = {
        'sentiment_score': stmt.excluded.sentiment_score,
        'article_count': stmt.excluded.article_count,
        'sentiment_std': stmt.excluded.sentiment_std,
    }

    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=['date', 'topic'],
        set_=update_dict
    )

    try:
        with engine.begin() as conn:
            conn.execute(upsert_stmt)
        logger.info(f"Upserted {len(records)} topic sentiment records.")
        return True
    except Exception as e:
        logger.error(f"Database error upserting topic sentiment: {e}")
        return False
