"""
market_data.py — Shared DB-backed price/change snapshot for all active tickers.

One window-function query replaces what used to be a live-scrape fan-out across
every ticker (main.py's old /api/market_performers). Used by market_performers,
the screener, and compare.
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from sqlalchemy import text
from src.psx_predictor.db.connection import engine
from src.psx_predictor.api.json_safe import safe_float


def get_price_snapshot() -> dict:
    """Returns {ticker: {price, change, change_percent, volume, week_52_high, week_52_low}}
    for every active ticker, computed from stock_eod_data. week_52_high/low use a trailing
    252-trading-day window (rows, not calendar days, since EOD data only has trading days).

    Uses a LATERAL join to pull only the latest 252 rows per ticker via the
    (ticker, date) index, then aggregates with plain MAX/MIN. A prior version
    computed MAX/MIN OVER (ROWS BETWEEN 251 PRECEDING ...) across the *entire*
    history for every row before filtering to the latest one — since MIN/MAX
    aren't invertible, Postgres can't use a sliding-window algorithm for that
    frame and instead rescans up to 252 rows per row of history, which took
    ~60s. This version takes <100ms for the same result."""
    query = text("""
        SELECT m.ticker,
               MAX(CASE WHEN sub.rn = 1 THEN sub.close END) AS close,
               MAX(CASE WHEN sub.rn = 1 THEN sub.volume END) AS volume,
               MAX(CASE WHEN sub.rn = 2 THEN sub.close END) AS prev_close,
               MAX(sub.high) AS week_52_high,
               MIN(sub.low) AS week_52_low
        FROM stock_metadata m
        CROSS JOIN LATERAL (
            SELECT close, volume, high, low,
                   ROW_NUMBER() OVER (ORDER BY date DESC) AS rn
            FROM stock_eod_data e
            WHERE e.ticker = m.ticker
            ORDER BY e.date DESC
            LIMIT 252
        ) sub
        WHERE m.is_active = true
        GROUP BY m.ticker
    """)
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().fetchall()

    snapshot = {}
    for r in rows:
        close = safe_float(r["close"]) or 0.0
        prev_close = safe_float(r["prev_close"])
        prev_close = prev_close if prev_close is not None else close
        change = close - prev_close
        change_percent = (change / prev_close * 100) if prev_close else 0.0
        week_52_high = safe_float(r["week_52_high"])
        week_52_low = safe_float(r["week_52_low"])
        snapshot[r["ticker"]] = {
            "price": round(close, 2),
            "change": round(change, 2),
            "change_percent": round(change_percent, 2),
            "volume": int(r["volume"]) if r["volume"] is not None else 0,
            "week_52_high": round(week_52_high, 2) if week_52_high is not None else None,
            "week_52_low": round(week_52_low, 2) if week_52_low is not None else None,
        }
    return snapshot
