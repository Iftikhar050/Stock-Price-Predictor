import os
import sys
import time
import logging
import requests
import pandas as pd
from datetime import datetime
from sqlalchemy import text
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

# Load environment variables from .env
load_dotenv(os.path.join(ROOT_DIR, ".env"))

from src.psx_predictor.db.connection import engine
from src.psx_predictor.db.repository import upsert_macro_indicators

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AlphaVantageFetcher")

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "LZ5KLTE63PU0FRT3")

class AlphaVantageFetcher:
    """
    Ingests financial news sentiment, global commodities, US interest rates,
    and macroeconomic indicators from Alpha Vantage API into PostgreSQL.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or ALPHA_VANTAGE_API_KEY
        self.base_url = "https://www.alphavantage.co/query"

    def fetch_news_sentiment(self, topics: str = "financial_markets,energy_transportation,economy_monetary", limit: int = 50) -> pd.DataFrame:
        """
        Fetches market news and AI-derived sentiment scores from Alpha Vantage NEWS_SENTIMENT endpoint.
        """
        logger.info(f"Fetching Alpha Vantage News & Sentiment for topics: {topics}...")
        params = {
            "function": "NEWS_SENTIMENT",
            "topics": topics,
            "limit": limit,
            "apikey": self.api_key
        }
        
        try:
            r = requests.get(self.base_url, params=params, timeout=15)
            if r.status_code == 200:
                data = r.json()
                feed = data.get("feed", [])
                if not feed:
                    logger.warning(f"Alpha Vantage returned empty news feed: {data.get('Information', data.get('Note', 'No data'))}")
                    return pd.DataFrame()
                    
                articles = []
                for item in feed:
                    pub_time = item.get("time_published", "")
                    try:
                        pub_dt = datetime.strptime(pub_time, "%Y%m%dT%H%M%S")
                    except ValueError:
                        pub_dt = datetime.now()
                        
                    articles.append({
                        "ticker": "MACRO",
                        "published_at": pub_dt,
                        "headline": item.get("title", ""),
                        "summary": item.get("summary", ""),
                        "content": item.get("summary", ""),
                        "source": item.get("source", "AlphaVantage"),
                        "url": item.get("url", ""),
                        "sentiment_score": float(item.get("overall_sentiment_score", 0.0))
                    })
                    
                df_news = pd.DataFrame(articles)
                logger.info(f"Fetched {len(df_news)} news articles with NLP sentiment scores from Alpha Vantage!")
                return df_news
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")
        return pd.DataFrame()

    def sync_news_to_db(self, df_news: pd.DataFrame) -> bool:
        """Inserts fetched Alpha Vantage news articles into PostgreSQL stock_news table."""
        if df_news.empty:
            return False
            
        insert_sql = text("""
            INSERT INTO stock_news (
                ticker, published_at, headline, summary, content, source, url, sentiment_score, created_at
            ) VALUES (
                :ticker, :published_at, :headline, :summary, :content, :source, :url, :sentiment_score, NOW()
            )
            ON CONFLICT DO NOTHING
        """)
        
        with engine.connect() as conn:
            for idx, row in df_news.iterrows():
                conn.execute(insert_sql, {
                    "ticker": row["ticker"],
                    "published_at": row["published_at"],
                    "headline": row["headline"],
                    "summary": row["summary"],
                    "content": row["content"],
                    "source": row["source"],
                    "url": row["url"],
                    "sentiment_score": row["sentiment_score"]
                })
            conn.commit()
            
        logger.info(f"Successfully synced {len(df_news)} Alpha Vantage news items into stock_news table!")
        return True

    def fetch_economic_indicator(self, function_name: str, col_name: str) -> pd.DataFrame:
        """
        Fetches economic indicators (TREASURY_YIELD, FEDERAL_FUNDS_RATE, CPI, INFLATION, WTI, BRENT, NATURAL_GAS, COPPER).
        """
        logger.info(f"Fetching Alpha Vantage economic indicator: {function_name}...")
        params = {
            "function": function_name,
            "apikey": self.api_key
        }
        
        try:
            r = requests.get(self.base_url, params=params, timeout=15)
            if r.status_code == 200:
                data = r.json()
                time_series = data.get("data", [])
                if time_series:
                    df = pd.DataFrame(time_series)
                    df['date'] = pd.to_datetime(df['date']).dt.date
                    df[col_name] = pd.to_numeric(df['value'], errors='coerce')
                    df = df[['date', col_name]].dropna()
                    logger.info(f"Fetched {len(df)} records for {col_name} via Alpha Vantage {function_name}!")
                    return df
                else:
                    logger.warning(f"Alpha Vantage message for {function_name}: {data.get('Information', data.get('Note', 'No data'))}")
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage indicator {function_name}: {e}")
        return pd.DataFrame()

    def sync_all(self):
        """Syncs news sentiment and macro indicators from Alpha Vantage into PostgreSQL."""
        logger.info(f"Starting Alpha Vantage sync using API Key: {self.api_key[:5]}***")
        
        # 1. News Sentiment
        df_news = self.fetch_news_sentiment(limit=50)
        if not df_news.empty:
            self.sync_news_to_db(df_news)
            
        # 2. Key Macro Indicators (respecting 5 calls/min limit with 12s sleep)
        indicators = [
            ("TREASURY_YIELD", "us10y_yield"),
            ("FEDERAL_FUNDS_RATE", "fed_funds_rate"),
            ("BRENT", "brent_oil_price"),
            ("WTI", "wti_oil_price")
        ]
        
        for func_name, col_name in indicators:
            time.sleep(12) # Rate limiting
            df_ind = self.fetch_economic_indicator(func_name, col_name)
            if not df_ind.empty:
                upsert_macro_indicators(df_ind)
                
        logger.info("Alpha Vantage full sync complete!")

if __name__ == "__main__":
    av = AlphaVantageFetcher()
    av.sync_all()

