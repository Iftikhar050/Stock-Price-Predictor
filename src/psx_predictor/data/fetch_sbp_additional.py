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
import os
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

# No hardcoded fallback: the key must come from the environment. A leaked key
# baked into source ships forever in git history even after being "removed".
SBP_API_KEY = os.environ.get("SBP_API_KEY", "")
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
    "TS_GP_BAM_M2_W.M000480":                 ("private_sector_credit_growth", 1, "ffill"),
    "TS_GP_BAM_M2_W.M000490":                 ("banking_deposits_growth", 1, "ffill"),
    "MA_OMO_NET_OUTSTANDING":                 ("sbp_omo_net_outstanding", 0, "ffill"),
    "TS_GP_BAM_SIRTBIL_AH.TB0010":            ("t_bill_cutoff_3m", 0, "ffill"),
    "TS_GP_BAM_SIRTBIL_AH.TB0020":            ("t_bill_cutoff_6m", 0, "ffill"),
    "MA_FWD_PKR_USD_3M":                      ("forward_usd_pkr_3m", 0, "ffill"),
    "TS_GP_ER_REERNEER_M.R00010":              ("reer_index", 1, "ffill"),
    "TS_GP_ED_PKEDLOUT_Q.STO00570":            ("external_debt_total_usd_bn", 1, "ffill"),
}


def _fetch_series(series_id: str) -> pd.DataFrame:
    """
    Call SBP EasyData REST endpoint for a single time-series.
    Returns DataFrame with columns ['date', 'value'] on success, or empty on failure.
    """
    url = f"https://easydata.sbp.org.pk/api/v1/series/{series_id}/data"
    params = {
        "api_key": SBP_API_KEY,
        "format": "json",
        # Without an explicit date range the API only returns the single
        # most recent observation, which then fails the >=10-row sanity
        # check below and gets discarded entirely.
        "start_date": "2005-01-01",
        "end_date": datetime.now().strftime("%Y-%m-%d"),
    }
    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper()
        resp = scraper.get(url, params=params, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"SBP EasyData non-200 for {series_id}: {resp.status_code}")
            return pd.DataFrame(columns=['date', 'value'])
        
        payload = resp.json()
        if "columns" in payload and "rows" in payload:
            cols = [c.lower() for c in payload["columns"]]
            rows = payload["rows"]
            if not rows:
                return pd.DataFrame(columns=['date', 'value'])
            df = pd.DataFrame(rows, columns=cols)
            date_col = "observation date"
            val_col = "observation value"
            if date_col not in df.columns or val_col not in df.columns:
                return pd.DataFrame(columns=['date', 'value'])
            df = df.rename(columns={date_col: "date", val_col: "value"})
        else:
            data = payload
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



def fetch_sbp_additional() -> bool:
    """
    Fetches tier-2 SBP EasyData macro series.
    """
    logger.info("Starting SBP EasyData tier-2 series fetch...")

    if not SBP_API_KEY:
        logger.warning(
            "SBP_API_KEY is not set - skipping the live EasyData series fetch. "
            "The manual-PDF-derived columns below will still be attempted."
        )

    import pdfplumber
    import re
    from src.psx_predictor.utils.manual_data_loader import inspect_and_log_all
    from dateutil.relativedelta import relativedelta

    manual_records = []
    
    try:
        files = inspect_and_log_all("sbp_bulletin")
        for file in files:
            ext = file.suffix.lower()
            if ext != '.pdf': continue
            
            logger.info(f"Extracting M2 and Remittances from text of {file.name}")
            
            # Extract date from filename (e.g. Monthly_Statistical_Bulletin_Feb-26.pdf)
            match = re.search(r'([A-Za-z]+)-(\d+)', file.name)
            if not match: continue
            
            month_str, year_str = match.groups()
            if len(year_str) == 2: year_str = "20" + year_str
            try:
                file_dt = datetime.strptime(f"{month_str[:3].title()} {year_str}", "%b %Y")
            except ValueError:
                continue
                
            text = ""
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages[:20]:
                    p_text = page.extract_text()
                    if p_text and "Selected Economic Indicators" in p_text and "Broad Money" in p_text:
                        text = p_text
                        break
                        
            if not text: continue
            text = text.replace('\n', ' ')
            
            # Find M2: Broad Money (M2) @ " 38,977.9 39,385.6 ...
            m2_line = re.search(r'Broad Money \(M2\).*?((?:[\d,]+\.\d\s*)+)', text)
            m2_vals = []
            if m2_line:
                m2_vals = [float(v.replace(',', '')) for v in m2_line.group(1).split()]
                
            # Find Remittances: Workers Remittances " 3,214.5 3,138.2 ...
            remit_line = re.search(r'Workers.*?Remittances.*?((?:[\d,]+\.\d\s*)+)', text)
            remit_vals = []
            if remit_line:
                remit_vals = [float(v.replace(',', '')) for v in remit_line.group(1).split()]
                
            # The latest value in the Selected Economic Indicators table corresponds to file_dt - 1 month.
            latest_dt = file_dt - relativedelta(months=1)
            
            for i, val in enumerate(reversed(m2_vals)):
                dt = latest_dt - relativedelta(months=i)
                manual_records.append({'date': dt.date(), 'm2_money_supply': val})
                
            for i, val in enumerate(reversed(remit_vals)):
                dt = latest_dt - relativedelta(months=i)
                manual_records.append({'date': dt.date(), 'monthly_remittances': val})
                
        if manual_records:
            manual_df = pd.DataFrame(manual_records)
            manual_df = manual_df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
            logger.info(f"Loaded {len(manual_df)} combined records for M2/Remittances from manual PDFs.")
        else:
            manual_df = pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error parsing manual SBP bulletin via text: {e}")
        manual_df = pd.DataFrame()
        
    # 2. Live API Fetch

    full_dates = pd.date_range(start="2005-01-01", end=datetime.now().strftime("%Y-%m-%d"), freq="D")
    combined = pd.DataFrame({"date": full_dates.date})

    for series_id, (col_name, lag_months, fill_method) in SERIES_MAP.items():
        raw = _fetch_series(series_id)
        synthetic = False

        if raw.empty or len(raw) < 10:
            synthetic = False
            raw = pd.DataFrame({'date': full_dates.date, 'value': [pd.NA] * len(full_dates)})
            
        if lag_months > 0 and not raw['value'].isna().all():
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
    
    # Forward fill manual data to daily frequency
    if not manual_df.empty:
        manual_df['date'] = pd.to_datetime(manual_df['date'])
        combined['date'] = pd.to_datetime(combined['date'])
        
        # Merge M2 Money Supply if exists
        if 'm2_money_supply' in manual_df.columns:
            m2_df = manual_df[['date', 'm2_money_supply']].dropna().sort_values('date')
            if not m2_df.empty:
                combined = pd.merge_asof(
                    combined.sort_values('date'),
                    m2_df,
                    on='date',
                    direction='backward'
                )
                
        # Merge Remittances if exists
        if 'monthly_remittances' in manual_df.columns:
            remit_df = manual_df[['date', 'monthly_remittances']].dropna().sort_values('date')
            if not remit_df.empty:
                combined = pd.merge_asof(
                    combined.sort_values('date'),
                    remit_df,
                    on='date',
                    direction='backward'
                )
                
        combined['date'] = combined['date'].dt.date

    success = upsert_macro_indicators(combined)
    logger.info(f"SBP additional series upsert complete: {len(combined)} rows, {len(combined.columns)} columns.")
    return success


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = fetch_sbp_additional()
    print("Success:", ok)
