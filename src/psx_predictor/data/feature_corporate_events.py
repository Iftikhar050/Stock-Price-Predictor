"""
feature_corporate_events.py
----------------------------
Converts PSX PUCARS raw corporate announcements from PostgreSQL into
time-aligned binary, decay, and categorical event features for use in the
PSO and MEBL master feature datasets.

Rules:
- Uses announcement_date as effective date (event is public on this date).
- Forward-fill exponential decay for N days after event. Never backward-fill.
- No look-ahead bias: each event becomes available only on its announcement_date.
- Level: Daily (aligned to stock_eod_data trading dates via merge_asof).
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

from src.psx_predictor.db.connection import engine

logger = logging.getLogger("CorporateEventsFeatures")
logger.setLevel(logging.INFO)

# Mapping from PUCARS category keywords → canonical event column names
CATEGORY_MAP = {
    "earnings":          "earnings_event",
    "financial result":  "earnings_event",
    "quarterly":         "earnings_event",
    "annual":            "earnings_event",
    "dividend":          "dividend_event",
    "bonus":             "bonus_event",
    "right":             "rights_event",
    "rights":            "rights_event",
    "acquisition":       "acquisition_event",
    "merger":            "merger_event",
    "contract":          "major_contract_event",
    "agreement":         "major_contract_event",
    "shutdown":          "plant_shutdown_event",
    "suspend":           "plant_shutdown_event",
    "expansion":         "capacity_expansion_event",
    "capacity":          "capacity_expansion_event",
    "management":        "management_change_event",
    "ceo":               "management_change_event",
    "cfo":               "management_change_event",
    "director":          "management_change_event",
    "regulatory":        "regulatory_approval_event",
    "approval":          "regulatory_approval_event",
    "secp":              "regulatory_approval_event",
    "nepra":             "regulatory_approval_event",
    "litigation":        "litigation_event",
    "court":             "litigation_event",
    "buyback":           "share_buyback_event",
    "repurchase":        "share_buyback_event",
    "insider":           "insider_transaction_event",
    "sponsor":           "sponsor_transaction_event",
    "shareholding":      "sponsor_transaction_event",
}

EVENT_COLS = list(set(CATEGORY_MAP.values()))
DECAY_HALF_LIFE = 5  # half-life in trading days for exponential decay


def _map_category(text_val: str) -> str | None:
    """Maps a PUCARS category string to a canonical event column name."""
    if not text_val:
        return None
    text_lower = str(text_val).lower()
    for keyword, col in CATEGORY_MAP.items():
        if keyword in text_lower:
            return col
    return None


def generate_event_features(ticker: str, trading_dates: pd.DatetimeIndex = None) -> pd.DataFrame:
    """
    Generates time-aligned corporate event features for a ticker.

    Args:
        ticker:         PSX ticker symbol (e.g. 'MEBL', 'PSO').
        trading_dates:  DatetimeIndex of trading dates from stock_eod_data.
                        If None, loads from DB.

    Returns:
        DataFrame indexed by date with binary event flags, sentiment scores,
        days-since-event, and exponential decay features.
    """
    logger.info(f"Generating corporate event features for {ticker}...")

    # Load PUCARS announcements from DB
    query = text("""
        SELECT announcement_date, category, headline_raw_text, body_raw_text, sentiment_score
        FROM corporate_announcements_pucars
        WHERE ticker = :ticker
        ORDER BY announcement_date ASC
    """)
    with engine.connect() as conn:
        pucars_df = pd.read_sql(query, conn, params={"ticker": ticker.upper()})

    if pucars_df.empty:
        logger.warning(f"No PUCARS data for {ticker}. Returning empty event features.")
        if trading_dates is None:
            return pd.DataFrame()
        empty = pd.DataFrame(index=trading_dates)
        empty.index.name = 'date'
        for col in EVENT_COLS + ['days_since_last_event', 'event_sentiment_score']:
            empty[col] = 0
        return empty.reset_index()

    pucars_df['announcement_date'] = pd.to_datetime(pucars_df['announcement_date'])

    # Load trading dates from DB if not provided
    if trading_dates is None:
        q_dates = text("SELECT DISTINCT date FROM stock_eod_data WHERE ticker = :ticker ORDER BY date ASC")
        with engine.connect() as conn:
            dates_df = pd.read_sql(q_dates, conn, params={"ticker": ticker.upper()})
        trading_dates = pd.to_datetime(dates_df['date'])

    base_df = pd.DataFrame({'date': trading_dates}).set_index('date')

    # Initialize event columns
    for col in EVENT_COLS:
        base_df[col] = 0
    base_df['event_sentiment_score'] = 0.0

    # Map category to event type and set binary flags (no leakage: only on announcement_date)
    for _, row in pucars_df.iterrows():
        event_date = row['announcement_date']
        event_col = _map_category(row.get('category', '')) or _map_category(row.get('headline_raw_text', ''))
        if event_date in base_df.index and event_col and event_col in base_df.columns:
            base_df.loc[event_date, event_col] = 1
            base_df.loc[event_date, 'event_sentiment_score'] = float(row.get('sentiment_score', 0.0) or 0.0)

    # Compute days_since_last_event (any event type)
    any_event = base_df[EVENT_COLS].max(axis=1)
    event_dates_idx = base_df.index[any_event > 0]
    base_df['days_since_last_event'] = 9999
    last_event = None
    for date in base_df.index:
        if any_event.loc[date] > 0:
            last_event = date
        if last_event is not None:
            base_df.loc[date, 'days_since_last_event'] = (date - last_event).days
    base_df['days_since_last_event'] = base_df['days_since_last_event'].clip(upper=365)

    # Exponential decay of event sentiment (half_life = 5 trading days)
    # Decay: score × exp(-ln(2)/half_life × days_since_event)
    # Only forward from event date, never backward
    decay_factor = np.log(2) / DECAY_HALF_LIFE
    base_df['event_sentiment_decay'] = 0.0
    for date in event_dates_idx:
        raw_score = base_df.loc[date, 'event_sentiment_score']
        if raw_score == 0.0:
            continue
        future = base_df.loc[date:].index
        for d in future:
            days = (d - date).days
            if days > 30:  # cap decay at 30 calendar days
                break
            base_df.loc[d, 'event_sentiment_decay'] += raw_score * np.exp(-decay_factor * days)

    base_df.reset_index(inplace=True)
    logger.info(f"Generated {len(EVENT_COLS)} event feature columns for {ticker} ({len(pucars_df)} PUCARS records).")
    return base_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for t in ["PSO", "MEBL"]:
        df = generate_event_features(t)
        print(f"{t}: {df.shape}")
        print(df[['date'] + EVENT_COLS].tail())
