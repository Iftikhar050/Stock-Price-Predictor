import logging
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Tuple

from .config import COMPANIES
from .models import Article
from .collectors.google_news import GoogleNewsCollector
from .company_matcher import CompanyMatcher
from .deduplicator import Deduplicator
from .sentiment import VaderSentimentEngine
from src.psx_predictor.db.repository import upsert_stock_news, upsert_news_sentiment

logger = logging.getLogger(__name__)

class NewsAggregator:
    """
    Orchestrates the entire news intelligence pipeline.
    """
    def __init__(self):
        self.collectors = [
            GoogleNewsCollector(period='7d') # Use 7d initially to populate history
        ]
        self.sentiment_engine = VaderSentimentEngine()
        
    def _apply_trading_day_cutoff(self, published_at: datetime) -> datetime.date:
        """
        Applies Trading Day Cutoff to prevent data leakage.
        News between Previous Day 16:00 PKT (11:00 UTC) and Current Day 15:59 PKT (10:59 UTC)
        is assigned to Current Day's trading session.
        """
        # published_at is in UTC. PSX Close is 15:30 PKT (10:30 UTC) or 16:00 PKT (11:00 UTC).
        # If published AFTER 11:00 UTC, it affects the NEXT trading day.
        cutoff_hour_utc = 11
        
        if published_at.hour >= cutoff_hour_utc:
            # Shift to next day
            trading_date = (published_at + timedelta(days=1)).date()
        else:
            trading_date = published_at.date()
            
        return trading_date
        
    def run_pipeline(self):
        logger.info("Starting Financial News Intelligence Pipeline...")
        
        all_raw_articles = []
        
        # 1. Collect from all sources
        for collector in self.collectors:
            for ticker, company in COMPANIES.items():
                logger.info(f"Collecting {company.name} from {collector.source_name}...")
                try:
                    articles = collector.fetch_news(company)
                    all_raw_articles.extend(articles)
                except Exception as e:
                    logger.error(f"Collector {collector.source_name} failed for {ticker}: {e}")
                    
        logger.info(f"Total raw articles collected: {len(all_raw_articles)}")
        if not all_raw_articles:
            logger.warning("No articles collected. Pipeline finishing early.")
            return
            
        # 2. Company Matching
        matched_articles = []
        for article in all_raw_articles:
            matched_articles.extend(CompanyMatcher.match_article(article))
            
        logger.info(f"Articles after intelligent matching (including multi-ticker splits): {len(matched_articles)}")
        
        # 3. Deduplication
        existing_urls = set() # Ideally load from DB, but for now we deduplicate the current batch
        existing_hashes = set()
        unique_articles = Deduplicator.filter_duplicates(matched_articles, existing_urls, existing_hashes)
        logger.info(f"Articles after deduplication: {len(unique_articles)}")
        
        # 4. Sentiment Analysis
        analyzed_articles = self.sentiment_engine.analyze(unique_articles)
        
        # 5. Insert Raw Articles to DB
        articles_dicts = [a.model_dump() for a in analyzed_articles]
        df_raw = pd.DataFrame(articles_dicts)
        
        # Ensure timezone-aware datetime for pandas before inserting to PostgreSQL
        df_raw['published_at'] = pd.to_datetime(df_raw['published_at'], utc=True)
        
        # Reorder to match DB columns (drop None/Pydantic defaults if needed, though dict is fine)
        success_raw = upsert_stock_news(df_raw)
        logger.info(f"Raw news insertion success: {success_raw}")
        
        # 6. Daily Aggregation
        # Apply Trading Day Cutoff
        df_raw['trading_date'] = df_raw['published_at'].apply(self._apply_trading_day_cutoff)
        
        # Group by Ticker and Trading Date
        df_agg = df_raw.groupby(['ticker', 'trading_date']).agg(
            sentiment_score=('sentiment_score', 'mean'),
            article_count=('headline', 'count')
        ).reset_index()
        
        df_agg.rename(columns={'trading_date': 'date'}, inplace=True)
        
        # Upsert Aggregated Sentiment
        success_agg = upsert_news_sentiment(df_agg)
        logger.info(f"Aggregated sentiment upsert success: {success_agg}")
        logger.info("Pipeline Complete!")

if __name__ == "__main__":
    # Setup logging for direct execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    aggregator = NewsAggregator()
    aggregator.run_pipeline()
