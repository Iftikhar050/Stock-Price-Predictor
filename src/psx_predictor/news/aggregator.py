"""
aggregator.py
--------------
Orchestrates the entire News Intelligence Pipeline:

1. Collect from all sources (Google News, Local Pakistani RSS, Alpha Vantage)
2. Company matching (multi-ticker split)
3. Deduplication
4. Topic classification (macro_event_classifier)
5. Hybrid sentiment analysis (FinBERT + VADER)
6. Trading-day cutoff (16:00 PKT / 11:00 UTC)
7. Insert raw articles to DB (stock_news)
8. Daily aggregation per ticker (stock_news_sentiment)
9. Daily aggregation per topic (topic_sentiment_daily)

All news published after 16:00 PKT (11:00 UTC) on day T is assigned
to day T+1's trading session to prevent look-ahead leakage.
"""
import logging
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from typing import List

from .config import COMPANIES
from .models import Article
from .collectors.google_news import GoogleNewsCollector
from .company_matcher import CompanyMatcher
from .deduplicator import Deduplicator
from .sentiment import HybridSentimentEngine, VaderSentimentEngine
from .classifiers.macro_event_classifier import classify
from src.psx_predictor.db.repository import (
    upsert_stock_news,
    upsert_news_sentiment,
    upsert_topic_sentiment,
)

logger = logging.getLogger(__name__)


class NewsAggregator:
    """
    Orchestrates the entire news intelligence pipeline.
    """

    def __init__(self, use_finbert: bool = True):
        self.collectors = [
            GoogleNewsCollector(period="7d"),  # Use 7d initially to populate history
        ]

        # Add historical backfill collector (Dawn, Business Recorder, Profit)
        try:
            from .collectors.archive_backfill_collector import ArchiveBackfillCollector
            self.archive_collector = ArchiveBackfillCollector()
            self.collectors.append(self.archive_collector)
        except Exception as e:
            logger.warning(f"ArchiveBackfillCollector init warning: {e}")
            self.archive_collector = None

        # Try to add local news collector (requires feedparser)
        try:
            from .collectors.local_news_collector import LocalNewsCollector
            self.local_collector = LocalNewsCollector()
            self.collectors.append(self.local_collector)
        except ImportError:
            logger.warning("LocalNewsCollector unavailable (missing feedparser). Skipping.")
            self.local_collector = None

        # Initialize sentiment engine
        try:
            self.sentiment_engine = HybridSentimentEngine(prefer_finbert=use_finbert)
        except Exception:
            logger.warning("HybridSentimentEngine init failed; falling back to VADER.")
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

        if hasattr(published_at, "hour"):
            if published_at.hour >= cutoff_hour_utc:
                # Shift to next day
                trading_date = (published_at + timedelta(days=1)).date()
            else:
                trading_date = published_at.date()
        else:
            trading_date = published_at

        return trading_date

    def run_pipeline(self):
        logger.info("Starting Financial News Intelligence Pipeline...")

        all_raw_articles: List[Article] = []

        # ── 1. Collect from all company-specific sources ──────────────────
        for collector in self.collectors:
            for ticker, company in COMPANIES.items():
                logger.info(f"Collecting {company.name} from {collector.source_name}...")
                try:
                    articles = collector.fetch_news(company)
                    all_raw_articles.extend(articles)
                except Exception as e:
                    logger.error(f"Collector {collector.source_name} failed for {ticker}: {e}")

        # ── 1b. Collect unfiltered articles for macro/political classification ──
        macro_articles: List[Article] = []
        if self.archive_collector:
            try:
                unfiltered_backfill = self.archive_collector.fetch_macro_news()
                macro_articles.extend(unfiltered_backfill)
                logger.info(f"Collected {len(unfiltered_backfill)} macro backfill articles.")
            except Exception as e:
                logger.error(f"Unfiltered archive backfill news collection failed: {e}")

        if self.local_collector:
            try:
                unfiltered = self.local_collector.fetch_all_unfiltered()
                macro_articles.extend(unfiltered)
                logger.info(f"Collected {len(unfiltered)} unfiltered macro articles from local RSS.")
            except Exception as e:
                logger.error(f"Unfiltered local news collection failed: {e}")

        logger.info(f"Total raw company articles collected: {len(all_raw_articles)}")
        logger.info(f"Total raw macro articles collected: {len(macro_articles)}")

        if not all_raw_articles and not macro_articles:
            logger.warning("No articles collected. Pipeline finishing early.")
            return

        # ── 2. Company Matching ───────────────────────────────────────────
        matched_articles = []
        for article in all_raw_articles:
            matched_articles.extend(CompanyMatcher.match_article(article))

        logger.info(f"Articles after intelligent matching (including multi-ticker splits): {len(matched_articles)}")

        # ── 3. Deduplication ──────────────────────────────────────────────
        existing_urls = set()
        existing_hashes = set()
        unique_articles = Deduplicator.filter_duplicates(matched_articles, existing_urls, existing_hashes)
        logger.info(f"Articles after deduplication: {len(unique_articles)}")

        # ── 4. Topic Classification ───────────────────────────────────────
        for article in unique_articles:
            tags = classify(
                article.headline,
                article.summary or article.content or "",
            )
            article.topic_category = tags.topic_category

        # Classify macro articles too (these are ticker-agnostic)
        for article in macro_articles:
            tags = classify(
                article.headline,
                article.summary or article.content or "",
            )
            article.topic_category = tags.topic_category
            article.ticker = "MACRO"  # Tag as macro-level

        # ── 5. Sentiment Analysis ─────────────────────────────────────────
        analyzed_articles = self.sentiment_engine.analyze(unique_articles)
        analyzed_macro = self.sentiment_engine.analyze(macro_articles)

        all_analyzed = analyzed_articles + analyzed_macro

        # ── 6. Insert Raw Articles to DB ──────────────────────────────────
        if all_analyzed:
            articles_dicts = [a.model_dump() for a in all_analyzed]
            df_raw = pd.DataFrame(articles_dicts)

            # Ensure timezone-aware datetime for pandas before inserting to PostgreSQL
            df_raw["published_at"] = pd.to_datetime(df_raw["published_at"], utc=True)

            success_raw = upsert_stock_news(df_raw)
            logger.info(f"Raw news insertion success: {success_raw}")
        else:
            df_raw = pd.DataFrame()

        # ── 7. Daily Aggregation per Ticker ───────────────────────────────
        if not df_raw.empty:
            # Apply Trading Day Cutoff
            df_raw["trading_date"] = df_raw["published_at"].apply(
                self._apply_trading_day_cutoff
            )

            # Per-ticker aggregation
            ticker_articles = df_raw[df_raw["ticker"] != "MACRO"]
            if not ticker_articles.empty:
                df_agg = (
                    ticker_articles.groupby(["ticker", "trading_date"])
                    .agg(
                        sentiment_score=("sentiment_score", "mean"),
                        article_count=("headline", "count"),
                    )
                    .reset_index()
                )
                df_agg.rename(columns={"trading_date": "date"}, inplace=True)

                success_agg = upsert_news_sentiment(df_agg)
                logger.info(f"Aggregated ticker sentiment upsert success: {success_agg}")

            # ── 8. Daily Aggregation per Topic ────────────────────────────
            topic_articles = df_raw[df_raw["topic_category"].notna()]
            if not topic_articles.empty:
                df_topic = (
                    topic_articles.groupby(["topic_category", "trading_date"])
                    .agg(
                        sentiment_score=("sentiment_score", "mean"),
                        article_count=("headline", "count"),
                        sentiment_std=("sentiment_score", "std"),
                    )
                    .reset_index()
                )
                df_topic.rename(
                    columns={"topic_category": "topic", "trading_date": "date"},
                    inplace=True,
                )
                df_topic["sentiment_std"] = df_topic["sentiment_std"].fillna(0.0)

                success_topic = upsert_topic_sentiment(df_topic)
                logger.info(f"Topic sentiment upsert success: {success_topic}")

        logger.info("Pipeline Complete!")


if __name__ == "__main__":
    # Setup logging for direct execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    aggregator = NewsAggregator(use_finbert=False)  # Use VADER for quick test
    aggregator.run_pipeline()
