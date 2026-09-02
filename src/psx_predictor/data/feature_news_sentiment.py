"""
feature_news_sentiment.py
--------------------------
Engineers all PDF news/sentiment feature columns from the stock_news,
stock_news_sentiment, corporate_events, and topic_sentiment_daily tables.

Covers:
  - Group 26: Investor Sentiment (daily sentiment, decay, momentum, dispersion)
  - Group 36: News/Event Shocks (article volume z-scores, shock sentiment magnitude)
  - Group 40: Market Psychology (enriched fear/greed proxies from sentiment data)
  - Groups 19/20: Political/Geopolitical daily sentiment from topic_sentiment_daily

All features are computed with strict no-look-ahead semantics:
  - Trading day cutoff (16:00 PKT / 11:00 UTC) already applied in the aggregator
  - All rolling/decay computations use only past values
  - No future data leaks into any feature
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

logger = logging.getLogger("NewsFeatureBuilder")
logger.setLevel(logging.INFO)


def generate_news_sentiment_features(
    ticker: str,
    trading_dates: pd.DatetimeIndex,
    engine,
) -> pd.DataFrame:
    """
    Generate news & sentiment feature columns for a ticker, aligned to trading dates.

    Args:
        ticker: Stock ticker (e.g. 'PSO').
        trading_dates: DatetimeIndex of trading dates from stock_eod_data.
        engine: SQLAlchemy engine for DB queries.

    Returns:
        DataFrame indexed by date with all sentiment/news features.
    """
    logger.info(f"Generating news sentiment features for {ticker}...")

    base_df = pd.DataFrame({"date": trading_dates})
    base_df["date"] = pd.to_datetime(base_df["date"])
    base_df = base_df.sort_values("date").reset_index(drop=True)

    # ── 1. Per-ticker daily sentiment from stock_news_sentiment ────────────
    try:
        query = text(
            "SELECT date, sentiment_score, article_count "
            "FROM stock_news_sentiment WHERE ticker = :ticker ORDER BY date ASC"
        )
        with engine.connect() as conn:
            sent_df = pd.read_sql(query, conn, params={"ticker": ticker.upper()})
    except Exception as e:
        logger.warning(f"Could not load ticker sentiment for {ticker}: {e}")
        sent_df = pd.DataFrame()

    if not sent_df.empty:
        sent_df["date"] = pd.to_datetime(sent_df["date"])
        base_df = pd.merge(base_df, sent_df, on="date", how="left")
    else:
        base_df["sentiment_score"] = np.nan
        base_df["article_count"] = 0

    base_df["article_count"] = base_df["article_count"].fillna(0).astype(int)

    # ── 2. Sentiment decay features ────────────────────────────────────────
    # 3-day exponential decay: if today has no sentiment, inherit from yesterday
    # with half-life = 1 day (decay factor 0.5 per day)
    raw_sent = base_df["sentiment_score"].copy()

    # Forward-fill with exponential decay (3-day window)
    decayed = raw_sent.copy()
    for i in range(1, len(decayed)):
        if pd.isna(decayed.iloc[i]):
            if pd.notna(decayed.iloc[i - 1]):
                decayed.iloc[i] = decayed.iloc[i - 1] * 0.5
    base_df["sentiment_score_3d_decay"] = decayed.fillna(0.0)
    base_df["sentiment_score"] = base_df["sentiment_score"].fillna(0.0)

    # 7-day SMA of sentiment
    base_df["sentiment_score_7d_sma"] = (
        base_df["sentiment_score"]
        .rolling(window=7, min_periods=1)
        .mean()
        .fillna(0.0)
    )

    # Sentiment momentum: 3d decay minus 7d SMA (short-term vs. medium-term)
    base_df["sentiment_momentum_3d_vs_7d"] = (
        base_df["sentiment_score_3d_decay"] - base_df["sentiment_score_7d_sma"]
    )

    # Sentiment dispersion: 7-day rolling std (measures consensus vs. disagreement)
    base_df["sentiment_dispersion_7d"] = (
        base_df["sentiment_score"]
        .rolling(window=7, min_periods=2)
        .std()
        .fillna(0.0)
    )

    # ── 3. News volume shock features (Group 36) ──────────────────────────
    base_df["news_volume_daily"] = base_df["article_count"]

    # 20-day rolling mean and std of article count
    vol_mean = base_df["news_volume_daily"].rolling(window=20, min_periods=5).mean()
    vol_std = base_df["news_volume_daily"].rolling(window=20, min_periods=5).std().replace(0, 1)

    base_df["news_volume_zscore_20d"] = (
        (base_df["news_volume_daily"] - vol_mean) / vol_std
    ).fillna(0.0)

    # Binary shock flag: z-score > 2.0 indicates an event shock
    base_df["news_shock_flag"] = (base_df["news_volume_zscore_20d"] > 2.0).astype(int)

    # Shock sentiment magnitude: sentiment on shock days, 0 otherwise
    base_df["news_shock_sentiment"] = np.where(
        base_df["news_shock_flag"] == 1,
        base_df["sentiment_score"],
        0.0,
    )

    # ── 4. Topic-level sentiment features (Groups 19, 20, 26) ─────────────
    try:
        query_topics = text(
            "SELECT date, topic, sentiment_score, article_count, sentiment_std "
            "FROM topic_sentiment_daily ORDER BY date ASC"
        )
        with engine.connect() as conn:
            topic_df = pd.read_sql(query_topics, conn)
    except Exception as e:
        logger.warning(f"Could not load topic sentiment: {e}")
        topic_df = pd.DataFrame()

    topic_features = {
        "POLITICAL": "political_news_sentiment",
        "GEOPOLITICAL": "geopolitical_news_sentiment",
        "MACRO_ECONOMIC": "macro_news_sentiment",
        "CORPORATE": "corporate_news_sentiment",
        "SECTOR_SPECIFIC": "sector_news_sentiment",
    }

    if not topic_df.empty:
        topic_df["date"] = pd.to_datetime(topic_df["date"])

        for topic_name, col_prefix in topic_features.items():
            topic_slice = topic_df[topic_df["topic"] == topic_name][
                ["date", "sentiment_score", "article_count"]
            ].rename(columns={
                "sentiment_score": f"{col_prefix}_3d",
                "article_count": f"{col_prefix}_count",
            })

            if not topic_slice.empty:
                base_df = pd.merge(base_df, topic_slice, on="date", how="left")
                # Apply 3-day rolling mean for smoothing
                raw_col = f"{col_prefix}_3d"
                base_df[raw_col] = (
                    base_df[raw_col]
                    .ffill(limit=3)
                    .fillna(0.0)
                )
            else:
                base_df[f"{col_prefix}_3d"] = 0.0
                base_df[f"{col_prefix}_count"] = 0
    else:
        for topic_name, col_prefix in topic_features.items():
            base_df[f"{col_prefix}_3d"] = 0.0
            base_df[f"{col_prefix}_count"] = 0

    # ── 5. Corporate event features from corporate_events table ───────────
    try:
        query_events = text(
            "SELECT trading_date as date, event_type, sentiment_score "
            "FROM corporate_events WHERE symbol = :ticker ORDER BY trading_date ASC"
        )
        with engine.connect() as conn:
            events_df = pd.read_sql(query_events, conn, params={"ticker": ticker.upper()})
    except Exception as e:
        logger.warning(f"Could not load corporate events for {ticker}: {e}")
        events_df = pd.DataFrame()

    if not events_df.empty:
        events_df["date"] = pd.to_datetime(events_df["date"])

        # Pivot event types into binary columns
        event_types = events_df["event_type"].dropna().unique()
        for etype in event_types:
            col_name = etype if etype.endswith("_event") else f"{etype}_event"
            if col_name not in base_df.columns:
                mask = events_df["event_type"] == etype
                event_dates = set(events_df.loc[mask, "date"].dt.date)
                base_df[col_name] = base_df["date"].dt.date.isin(event_dates).astype(int)

        # Days since last corporate event (any type)
        event_dates_set = set(events_df["date"].dt.date)
        base_df["days_since_corp_event"] = 9999
        last_event_date = None
        for i, row in base_df.iterrows():
            d = row["date"].date() if hasattr(row["date"], "date") else row["date"]
            if d in event_dates_set:
                last_event_date = d
            if last_event_date is not None:
                base_df.at[i, "days_since_corp_event"] = (d - last_event_date).days

        base_df["days_since_corp_event"] = base_df["days_since_corp_event"].clip(upper=365)

    # ── 6. Search interest features (from macro_indicators) ───────────────
    search_col = f"search_trend_{ticker.lower()}"
    try:
        query_search = text(
            f"SELECT date, {search_col} FROM macro_indicators "
            f"WHERE {search_col} IS NOT NULL ORDER BY date ASC"
        )
        with engine.connect() as conn:
            search_df = pd.read_sql(query_search, conn)

        if not search_df.empty:
            search_df["date"] = pd.to_datetime(search_df["date"])
            base_df = pd.merge(base_df, search_df, on="date", how="left")
            base_df[search_col] = base_df[search_col].ffill().fillna(0.0)

            # Spike detection: > 1.5 std above 20-day mean
            s_mean = base_df[search_col].rolling(20, min_periods=5).mean()
            s_std = base_df[search_col].rolling(20, min_periods=5).std().replace(0, 1)
            base_df["search_volume_spike_flag"] = (
                (base_df[search_col] - s_mean) / s_std > 1.5
            ).astype(int)
        else:
            base_df[search_col] = 0.0
            base_df["search_volume_spike_flag"] = 0
    except Exception as e:
        logger.info(f"Search trend column {search_col} not available: {e}")
        base_df[search_col] = 0.0
        base_df["search_volume_spike_flag"] = 0

    # ── 7. Sentiment coverage era column ("backfilled" / "live" / "unavailable") ──
    try:
        query_sources = text(
            "SELECT DISTINCT DATE(published_at) as date, source "
            "FROM stock_news WHERE ticker = :ticker OR ticker = 'MACRO'"
        )
        with engine.connect() as conn:
            source_df = pd.read_sql(query_sources, conn, params={"ticker": ticker.upper()})
    except Exception as e:
        logger.warning(f"Could not load article sources for era tagging: {e}")
        source_df = pd.DataFrame()

    era_map = {}
    if not source_df.empty:
        source_df["date"] = pd.to_datetime(source_df["date"])
        for d, grp in source_df.groupby("date"):
            sources = grp["source"].astype(str).str.lower().tolist()
            if any(s.endswith("_archive_backfill") for s in sources):
                era_map[d] = "backfilled"
            elif any(s and not s.endswith("_archive_backfill") for s in sources):
                era_map[d] = "live"
            else:
                era_map[d] = "unavailable"

    base_df["sentiment_coverage_era"] = base_df["date"].map(era_map).fillna("unavailable")

    # ── 8. Fill NaNs ──────────────────────────────────────────────────────
    numeric_cols = base_df.select_dtypes(include=[np.number]).columns
    base_df[numeric_cols] = base_df[numeric_cols].fillna(0.0)

    logger.info(
        f"Generated {len(base_df.columns) - 1} news/sentiment feature columns "
        f"for {ticker} ({len(base_df)} rows)."
    )
    return base_df
