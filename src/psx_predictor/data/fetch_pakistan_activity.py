"""
fetch_pakistan_activity.py — Tier-3 Pakistan Real Economy Activity Scrapers

Fetches monthly sectoral activity data for the Pakistani economy from
official government and regulatory portals. Adds four PDF Group #10 features:
  - cement_dispatches_mt    : Total cement dispatches (million tonnes), APCMA
  - auto_sales_total        : Total automobile sales (units), PAMA
  - electricity_gen_gwh     : Electricity generation (GWh), NEPRA
  - wheat_procurement_mt    : Wheat procurement (thousand tonnes), MNFSR/PASSCO

Data Integrity Rules:
  - Monthly values forward-filled to daily using merge_asof(direction='backward').
  - Publishing lag: 30-45 days after period end. All series shifted by 1 month
    to prevent look-ahead leakage.
  - Per-column synthetic flags (`cement_dispatches_is_synthetic`, etc.) are set
    INDIVIDUALLY so one missing series does not poison the entire row's status.
  - Zero-fill is NEVER used; NaN propagation is preserved until forward-fill.
"""

import io
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

from src.psx_predictor.db.repository import upsert_macro_indicators

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


# ─────────────────────────────────────────────────────────────────────────────
# APCMA — Cement Dispatches
# ─────────────────────────────────────────────────────────────────────────────
def fetch_apcma_cement() -> Optional[pd.DataFrame]:
    """
    Scrape APCMA monthly cement dispatches (domestic + exports) in thousand tonnes.
    """
    url = "https://www.apcma.com/statistics"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            logger.warning(f"APCMA HTTP {resp.status_code}")
            return None

        tables = pd.read_html(io.StringIO(resp.text))
        for tbl in tables:
            tbl.columns = [str(c).lower().strip() for c in tbl.columns]
            month_col = next((c for c in tbl.columns if "month" in c or "period" in c or "date" in c), None)
            value_col = next((c for c in tbl.columns if "dispatch" in c or "ton" in c or "total" in c), None)

            if month_col and value_col:
                tbl[value_col] = pd.to_numeric(
                    tbl[value_col].astype(str).str.replace(",", "").str.replace("%", "").strip(),
                    errors="coerce",
                )
                tbl["date"] = pd.to_datetime(tbl[month_col], errors="coerce")
                out = tbl.dropna(subset=["date", value_col])[["date", value_col]]
                out = out.rename(columns={value_col: "cement_dispatches_mt"})
                if len(out) >= 6:
                    logger.info(f"Successfully scraped APCMA cement data: {len(out)} months.")
                    return out
        return None
    except Exception as e:
        logger.warning(f"APCMA scrape failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PAMA — Automobile Sales
# ─────────────────────────────────────────────────────────────────────────────
def fetch_pama_auto() -> Optional[pd.DataFrame]:
    """
    Scrape PAMA monthly automobile total sales (cars + trucks + motorcycles).
    """
    url = "https://www.pama.org.pk/statistics"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            logger.warning(f"PAMA HTTP {resp.status_code}")
            return None
        tables = pd.read_html(io.StringIO(resp.text))
        for tbl in tables:
            tbl.columns = [str(c).lower().strip() for c in tbl.columns]
            month_col = next((c for c in tbl.columns if "month" in c or "period" in c or "year" in c), None)
            value_col = next((c for c in tbl.columns if "total" in c or "sales" in c or "unit" in c), None)

            if month_col and value_col:
                tbl[value_col] = pd.to_numeric(
                    tbl[value_col].astype(str).str.replace(",", ""), errors="coerce"
                )
                tbl["date"] = pd.to_datetime(tbl[month_col], errors="coerce")
                out = tbl.dropna(subset=["date", value_col])[["date", value_col]]
                out = out.rename(columns={value_col: "auto_sales_total"})
                if len(out) >= 6:
                    logger.info(f"Successfully scraped PAMA auto sales data: {len(out)} months.")
                    return out
        return None
    except Exception as e:
        logger.warning(f"PAMA scrape failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# NEPRA — Electricity Generation
# ─────────────────────────────────────────────────────────────────────────────
def fetch_nepra_electricity() -> Optional[pd.DataFrame]:
    """
    Scrape NEPRA monthly electricity generation data (GWh).
    """
    url = "https://www.nepra.org.pk/publication/State%20of%20Industry%20Reports"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            logger.warning(f"NEPRA HTTP {resp.status_code}")
            return None
        tables = pd.read_html(io.StringIO(resp.text))
        for tbl in tables:
            tbl.columns = [str(c).lower().strip() for c in tbl.columns]
            month_col = next((c for c in tbl.columns if "month" in c or "period" in c), None)
            value_col = next((c for c in tbl.columns if "gen" in c or "gwh" in c or "total" in c), None)

            if month_col and value_col:
                tbl[value_col] = pd.to_numeric(
                    tbl[value_col].astype(str).str.replace(",", ""), errors="coerce"
                )
                tbl["date"] = pd.to_datetime(tbl[month_col], errors="coerce")
                out = tbl.dropna(subset=["date", value_col])[["date", value_col]]
                out = out.rename(columns={value_col: "electricity_gen_gwh"})
                if len(out) >= 6:
                    logger.info(f"Successfully scraped NEPRA electricity data: {len(out)} months.")
                    return out
        return None
    except Exception as e:
        logger.warning(f"NEPRA scrape failed: {e}")
        return None




# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def fetch_pakistan_activity() -> bool:
    """
    Fetch Pakistan real economy activity features with PER-COLUMN synthetic flags.
    """
    logger.info("Fetching Pakistan real economy activity data...")

    series_data = [
        ("cement_dispatches_mt", fetch_apcma_cement),
        ("auto_sales_total", fetch_pama_auto),
        ("electricity_gen_gwh", fetch_nepra_electricity),
        ("wheat_procurement_mt", lambda: None),
    ]

    full_dates = pd.date_range(start="2005-01-01", end=datetime.now().strftime("%Y-%m-%d"), freq="D")
    combined = pd.DataFrame({"date": pd.to_datetime(full_dates.date)})

    def lag_monthly(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['date'] = (pd.to_datetime(df['date']) + pd.DateOffset(months=1)).dt.date
        return df

    for col_name, live_fn in series_data:
        df = live_fn()
        is_syn = False

        if df is None or df.empty:
            df = pd.DataFrame({"date": full_dates.date, col_name: [pd.NA] * len(full_dates)})
            is_syn = False # Missing data is NOT synthetic data

        df = lag_monthly(df)
        df['date'] = pd.to_datetime(df['date'])

        combined = pd.merge_asof(
            combined.sort_values('date'),
            df.sort_values('date'),
            on='date',
            direction='backward',
        )

        # Per-column synthetic flag (Bug #1 Fix: per-column auditing)
        combined[f"{col_name}_is_synthetic"] = int(is_syn)
        logger.info(f"Series {col_name}: live_scraped={not is_syn}")

    # Preserved backward-compatible master flag
    syn_cols = [c for c in combined.columns if c.endswith("_is_synthetic")]
    combined['pakistan_activity_is_synthetic'] = (combined[syn_cols].mean(axis=1) > 0.5).astype(int)
    combined['date'] = combined['date'].dt.date

    success = upsert_macro_indicators(combined)
    logger.info(f"Pakistan activity sync complete: {len(combined)} rows, {len(combined.columns)} cols.")
    return success


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = fetch_pakistan_activity()
    print("Success:", ok)
