"""
fetch_psx_market_stats.py
--------------------------
Scrapes daily PSX market statistics from the PSX public website to populate
market breadth, volume, and traded value metrics.

Source: PSX website (https://www.psx.com.pk/psx/resources/market-statistics)
Frequency: Daily (business days only).
Leakage: None — same-day publication, no forward-looking data used.

PDF Groups Covered:
  #18 (Market Liquidity Metrics): daily_traded_value, number_of_trades, new_highs, new_lows
  #18 (Market Breadth): advancing_count, declining_count, sector_breadth
"""

import os
import sys
import logging
import time
import requests
import pandas as pd
from datetime import datetime, date, timedelta
from bs4 import BeautifulSoup
from sqlalchemy import text

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

from src.psx_predictor.db.connection import engine

logger = logging.getLogger("PSXMarketStatsFetcher")
logger.setLevel(logging.INFO)

PSX_STATS_URL = "https://dps.psx.com.pk/market-statistics"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PSXDataBot/1.0)",
    "Accept": "text/html,application/xhtml+xml",
}
SESSION_DELAY = 1.5  # seconds between requests


def fetch_psx_market_stats_for_date(target_date: date) -> dict | None:
    """
    Fetches PSX market statistics for a specific date from PSX DPS.

    Args:
        target_date: Date to fetch statistics for.

    Returns:
        Dictionary with market stats keys or None if unavailable.
    """
    date_str = target_date.strftime("%Y-%m-%d")
    url = f"{PSX_STATS_URL}?date={date_str}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        stats = {"date": target_date}

        # Look for market summary statistics
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cols = row.find_all(["td", "th"])
                if len(cols) >= 2:
                    label = cols[0].get_text(strip=True).lower()
                    try:
                        val_text = cols[-1].get_text(strip=True).replace(",", "").replace("%", "")
                        val = float(val_text) if val_text else None
                    except (ValueError, AttributeError):
                        val = None

                    if "volume" in label and "total" in label:
                        stats["market_total_volume"] = val
                    elif "value" in label and "total" in label:
                        stats["market_total_traded_value"] = val
                    elif "trades" in label or "number of transaction" in label:
                        stats["market_number_of_trades"] = val
                    elif "52 week high" in label or "new high" in label:
                        stats["new_highs"] = val
                    elif "52 week low" in label or "new low" in label:
                        stats["new_lows"] = val
                    elif "advanc" in label:
                        stats["advancing_count"] = val
                    elif "declin" in label:
                        stats["declining_count"] = val
                    elif "unchanged" in label:
                        stats["unchanged_count"] = val

        if len(stats) > 1:
            # Compute derived: sector_breadth
            adv = stats.get("advancing_count") or 0
            dec = stats.get("declining_count") or 0
            total = adv + dec
            stats["sector_breadth"] = (adv - dec) / total if total > 0 else 0.0
            return stats

    except Exception as e:
        logger.debug(f"Error fetching PSX market stats for {date_str}: {e}")

    return None


def _ensure_columns():
    """Ensure market stat columns exist in macro_indicators table."""
    new_cols = [
        "market_total_volume DOUBLE PRECISION",
        "market_total_traded_value DOUBLE PRECISION",
        "market_number_of_trades DOUBLE PRECISION",
        "new_highs DOUBLE PRECISION",
        "new_lows DOUBLE PRECISION",
        "advancing_count DOUBLE PRECISION",
        "declining_count DOUBLE PRECISION",
        "unchanged_count DOUBLE PRECISION",
        "sector_breadth DOUBLE PRECISION",
    ]
    with engine.connect() as conn:
        for col_def in new_cols:
            conn.execute(text(f"ALTER TABLE macro_indicators ADD COLUMN IF NOT EXISTS {col_def};"))
        conn.commit()


def sync_psx_market_stats(lookback_days: int = 730) -> bool:
    """
    Syncs historical PSX market statistics into the macro_indicators table.

    Args:
        lookback_days: How many calendar days back to sync (default 2 years).
    """
    logger.info(f"Syncing PSX market stats (last {lookback_days} days)...")
    _ensure_columns()

    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    current = start_date

    records_upserted = 0
    update_sql = text("""
        INSERT INTO macro_indicators (date, market_total_volume, market_total_traded_value,
            market_number_of_trades, new_highs, new_lows, advancing_count, declining_count,
            unchanged_count, sector_breadth, created_at)
        VALUES (:date, :market_total_volume, :market_total_traded_value, :market_number_of_trades,
            :new_highs, :new_lows, :advancing_count, :declining_count, :unchanged_count,
            :sector_breadth, NOW())
        ON CONFLICT (date) DO UPDATE SET
            market_total_volume = COALESCE(EXCLUDED.market_total_volume, macro_indicators.market_total_volume),
            market_total_traded_value = COALESCE(EXCLUDED.market_total_traded_value, macro_indicators.market_total_traded_value),
            market_number_of_trades = COALESCE(EXCLUDED.market_number_of_trades, macro_indicators.market_number_of_trades),
            new_highs = COALESCE(EXCLUDED.new_highs, macro_indicators.new_highs),
            new_lows = COALESCE(EXCLUDED.new_lows, macro_indicators.new_lows),
            advancing_count = COALESCE(EXCLUDED.advancing_count, macro_indicators.advancing_count),
            declining_count = COALESCE(EXCLUDED.declining_count, macro_indicators.declining_count),
            unchanged_count = COALESCE(EXCLUDED.unchanged_count, macro_indicators.unchanged_count),
            sector_breadth = COALESCE(EXCLUDED.sector_breadth, macro_indicators.sector_breadth)
    """)

    with engine.connect() as conn:
        while current <= end_date:
            if current.weekday() < 5:  # skip weekends
                stats = fetch_psx_market_stats_for_date(current)
                if stats:
                    params = {
                        "date": stats["date"],
                        "market_total_volume": stats.get("market_total_volume"),
                        "market_total_traded_value": stats.get("market_total_traded_value"),
                        "market_number_of_trades": stats.get("market_number_of_trades"),
                        "new_highs": stats.get("new_highs"),
                        "new_lows": stats.get("new_lows"),
                        "advancing_count": stats.get("advancing_count"),
                        "declining_count": stats.get("declining_count"),
                        "unchanged_count": stats.get("unchanged_count"),
                        "sector_breadth": stats.get("sector_breadth"),
                    }
                    conn.execute(update_sql, params)
                    records_upserted += 1
                    if records_upserted % 50 == 0:
                        conn.commit()
                        logger.info(f"Committed {records_upserted} PSX market stats records...")

                time.sleep(SESSION_DELAY)

            current += timedelta(days=1)

        conn.commit()

    logger.info(f"PSX market stats sync complete. {records_upserted} records upserted.")
    return records_upserted > 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sync_psx_market_stats(lookback_days=365)
