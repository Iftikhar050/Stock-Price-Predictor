"""
fetch_sbp_additional.py — Tier-2 SBP EasyData API Extensions

Fetches additional monetary and balance-of-payments series from the
State Bank of Pakistan EasyData REST API that go beyond what
sbp_easydata_scraper.py already covers.

New columns added (all upserted into macro_indicators table):
  - private_sector_credit_growth  (monthly, % YoY change in credit outstanding)
  - banking_deposits_growth        (monthly, % YoY change in total deposits)
  - sbp_omo_net_outstanding        (weekly, PKR billions, OMO repo + outright net)
  - t_bill_cutoff_3m               (weekly, %)
  - t_bill_cutoff_6m               (weekly, %)
  - forward_usd_pkr_3m             (monthly, PKR per USD 3-month forward)
  - reer_index                     (monthly, index, SBP Real Effective Exchange Rate)
  - external_debt_total_usd_bn     (quarterly, USD billions)

Frequency handling:
  - Monthly/quarterly series are forward-filled to daily using merge_asof.
  - All series are published with a reporting lag: we shift by the stated lag
    to avoid look-ahead leakage.
  - Per-column synthetic flags prevent one unlisted series from poisoning all columns.
"""

import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date
from src.psx_predictor.db.repository import upsert_macro_indicators

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

SBP_API_KEY = "9FD9ADC4862DECD60AE3691139A265883C1CA2AD"
SBP_BASE_URL = "https://easydata.sbp.org.pk/api"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}

# ---------------------------------------------------------------------------
# SBP EasyData series IDs
# Discover at: https://easydata.sbp.org.pk/  (Browse Data → Monetary/External)
# ---------------------------------------------------------------------------
SERIES_MAP = {
    # Series ID                              : (column_name, lag_months, fill_method)
    "MA_MON_CREDIT_PRIVATE_SECTOR_YOY":       ("private_sector_credit_growth", 1, "ffill"),
    "MA_MON_DEPOSIT_GROWTH_YOY":              ("banking_deposits_growth", 1, "ffill"),
    "MA_OMO_NET_OUTSTANDING":                 ("sbp_omo_net_outstanding", 0, "ffill"),
    "MA_TBILL_3M_CUTOFF":                     ("t_bill_cutoff_3m", 0, "ffill"),
    "MA_TBILL_6M_CUTOFF":                     ("t_bill_cutoff_6m", 0, "ffill"),
    "MA_FWD_PKR_USD_3M":                      ("forward_usd_pkr_3m", 0, "ffill"),
    "MA_REER_INDEX":                          ("reer_index", 1, "ffill"),
    "MA_EXTERNAL_DEBT_TOTAL":                 ("external_debt_total_usd_bn", 1, "ffill"),
}


def _fetch_series(series_id: str) -> pd.DataFrame:
    """
    Call SBP EasyData REST endpoint for a single time-series.
    Returns DataFrame with columns ['date', 'value'] on success, or empty on failure.
    """
    url = f"{SBP_BASE_URL}/GetSeriesData"
    params = {
        "SeriesKey": series_id,
        "ApiKey": SBP_API_KEY,
        "format": "json",
    }
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"SBP EasyData non-200 for {series_id}: {resp.status_code}")
            return pd.DataFrame(columns=['date', 'value'])
        data = resp.json()
        records = data.get("data", data.get("Data", []))
        if not records:
            logger.warning(f"Empty data returned for SBP series {series_id}")
            return pd.DataFrame(columns=['date', 'value'])
        df = pd.DataFrame(records)
        df.columns = [c.lower() for c in df.columns]
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['value']).sort_values('date').reset_index(drop=True)
        return df[['date', 'value']]
    except Exception as e:
        logger.warning(f"Failed to fetch SBP series {series_id}: {e}")
        return pd.DataFrame(columns=['date', 'value'])


def _build_fallback_series(column_name: str, start: str = "2005-01-01") -> pd.DataFrame:
    """
    If SBP API is unavailable, build a realistic fallback.
    Flagged via companion per-column synthetic flag.
    """
    logger.warning(f"Building synthetic fallback for {column_name}")
    dates = pd.date_range(start=start, end=datetime.now().strftime("%Y-%m-%d"), freq="ME")
    np.random.seed(abs(hash(column_name)) % (2**32))

    defaults = {
        "private_sector_credit_growth": (10.0, 5.0),
        "banking_deposits_growth": (12.0, 4.0),
        "sbp_omo_net_outstanding": (500.0, 200.0),
        "t_bill_cutoff_3m": (10.0, 3.0),
        "t_bill_cutoff_6m": (10.5, 3.0),
        "forward_usd_pkr_3m": (280.0, 50.0),
        "reer_index": (100.0, 10.0),
        "external_debt_total_usd_bn": (100.0, 20.0),
    }
    mean, std = defaults.get(column_name, (50.0, 10.0))
    vals = np.random.normal(mean, std, len(dates)).clip(0)

    df = pd.DataFrame({"date": dates.date, "value": vals})
    return df


def fetch_sbp_additional() -> bool:
    """
    Fetch all additional SBP series, forward-fill to daily, and upsert into macro_indicators.
    Uses PER-COLUMN synthetic flags so one bad series ID does not poison all series.
    """
    logger.info("Starting SBP EasyData additional series fetch...")

    full_dates = pd.date_range(start="2005-01-01", end=datetime.now().strftime("%Y-%m-%d"), freq="D")
    combined = pd.DataFrame({"date": full_dates.date})

    for series_id, (col_name, lag_months, fill_method) in SERIES_MAP.items():
        raw = _fetch_series(series_id)
        synthetic = False

        if raw.empty or len(raw) < 10:
            raw = _build_fallback_series(col_name)
            synthetic = True

        if lag_months > 0:
            raw['date'] = pd.to_datetime(raw['date']) + pd.DateOffset(months=lag_months)
            raw['date'] = raw['date'].dt.date

        raw = raw.rename(columns={"value": col_name})
        raw['date'] = pd.to_datetime(raw['date'])
        combined['date'] = pd.to_datetime(combined['date'])

        combined = pd.merge_asof(
            combined.sort_values('date'),
            raw.sort_values('date'),
            on='date',
            direction='backward',
        )
        # Bug #2 Fix: Per-column synthetic flag
        combined[f"{col_name}_is_synthetic"] = int(synthetic)
        combined['date'] = combined['date'].dt.date

    # Preserved backward-compatible master flag (1 only if >50% of series are synthetic)
    syn_cols = [c for c in combined.columns if c.endswith("_is_synthetic")]
    combined['sbp_additional_is_synthetic'] = (combined[syn_cols].mean(axis=1) > 0.5).astype(int)

    success = upsert_macro_indicators(combined)
    logger.info(f"SBP additional series upsert complete: {len(combined)} rows, {len(combined.columns)} columns.")
    return success


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = fetch_sbp_additional()
    print("Success:", ok)
