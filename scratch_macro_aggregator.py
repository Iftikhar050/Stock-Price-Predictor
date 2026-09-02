import logging
import pandas as pd
from src.psx_predictor.news.collectors.archive_backfill_collector import ArchiveBackfillCollector
from src.psx_predictor.news.classifiers.macro_event_classifier import classify
from src.psx_predictor.news.sentiment import VaderSentimentEngine
from src.psx_predictor.db.repository import upsert_topic_sentiment

logging.basicConfig(level=logging.INFO)

collector = ArchiveBackfillCollector()
macro_articles = collector.fetch_macro_news()

# Classify
for article in macro_articles:
    tags = classify(article.headline, article.summary or article.content or "")
    article.topic_category = tags.topic_category

# Sentiment
engine = VaderSentimentEngine()
analyzed_macro = engine.analyze(macro_articles)

# Insert Topic Sentiment
df_raw = pd.DataFrame([a.model_dump() for a in analyzed_macro])
if not df_raw.empty:
    df_raw["trading_date"] = pd.to_datetime(df_raw["published_at"], utc=True).dt.date
    topic_articles = df_raw[df_raw["topic_category"].notna()]
    if not topic_articles.empty:
        df_topic = topic_articles.groupby(["topic_category", "trading_date"]).agg(
            sentiment_score=("sentiment_score", "mean"),
            article_count=("headline", "count"),
            sentiment_std=("sentiment_score", "std"),
        ).reset_index()
        df_topic.rename(columns={"topic_category": "topic", "trading_date": "date"}, inplace=True)
        df_topic["sentiment_std"] = df_topic["sentiment_std"].fillna(0.0)
        upsert_topic_sentiment(df_topic)
        print("Success! Topics inserted:")
        print(df_topic["topic"].unique())
