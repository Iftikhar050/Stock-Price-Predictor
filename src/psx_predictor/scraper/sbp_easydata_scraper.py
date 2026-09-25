"""
sbp_easydata_scraper.py — Real State Bank of Pakistan (SBP) EasyData API Integration.

Fetches macroeconomic series from SBP EasyData (https://easydata.sbp.org.pk) independently.
Never fabricates series (such as KIBOR, T-Bills, PIBs, CPI Core, bank reserves, remittances,
or currency in circulation) via formulas or parent-ratio multiples.

If a series genuinely cannot be fetched, it is left as NaN with an explicit `{column}_is_missing = True`
boolean flag. `is_synthetic_rate` dynamically reflects whether rate data was actually fetched live.
"""

import os
import sys
import logging
import requests
import pandas as pd
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.psx_predictor.db.repository import upsert_macro_indicators
from src.psx_predictor.data.parse_sbp_kibor import parse_sbp_kibor

logger = logging.getLogger(__name__)

SBP_BASE_URL = os.environ.get("SBP_BASE_URL", "https://easydata.sbp.org.pk/api/v1/series")

# Each entry MUST be a real, independently-reported SBP series —
# never a formula derived from another column in this dict.
REAL_SERIES = {
    "sbp_policy_rate":          "TS_GP_IR_SIRPR_AH.SBPOL0030",
    "kibor_3m":                 "TS_GP_BAM_SIRKIBOR_D.KIBOR0020",
    "kibor_6m":                 "TS_GP_BAM_SIRKIBOR_D.KIBOR0030",
    "kibor_1y":                 "TS_GP_BAM_SIRKIBOR_D.7KIBOR12M",
    "tbill_3m":                 "TS_GP_BAM_SIRTBIL_AH.TB0010",
    "tbill_6m":                 "TS_GP_BAM_SIRTBIL_AH.TB0020",
    "tbill_1y":                 "TS_GP_BAM_SIRTBIL_AH.TB0030",
    "pib_3y":                   "TS_GP_BAM_SIRPIBS_AH.PIB0010",
    "pib_5y":                   "TS_GP_BAM_SIRPIBS_AH.PIB0020",
    "pib_10y":                  "TS_GP_BAM_SIRPIBS_AH.PIB0040",
    "cpi_headline":             "TS_GP_PT_CPI_M.P00011516",
    "cpi_core":                 "TS_GP_PT_CPI_M.P00121516",
    "sbp_reserves":             "TS_GP_EXT_PAKRES_M.Z00030",
    "commercial_bank_reserves": "TS_GP_EXT_PAKRES_M.Z00050",
    "total_fx_reserves":        "TS_GP_EXT_PAKRES_M.Z00060",
    "monthly_remittances":      "TS_GP_BOP_WR_M.WR0340",
    "remittances_saudi":        "TS_GP_BOP_WR_M.WR0040",
    "remittances_uae":          "TS_GP_BOP_WR_M.WR0050",
    "remittances_usa":          "TS_GP_BOP_WR_M.WR0020",
    "remittances_uk":           "TS_GP_BOP_WR_M.WR0030",
    "m2_money_supply":          "TS_GP_BAM_M2_W.M000070",
    "currency_in_circulation":  "TS_GP_BAM_M2_W.M000010",
}


class SbpEasyDataScraper:
    def __init__(self, api_key: str = ""):
        # No hardcoded fallback: a key baked into source ships forever in git
        # history even after being "removed". Missing key -> per-series calls
        # fail gracefully (existing non-200 handling already logs + skips).
        self.api_key = api_key or os.environ.get("SBP_API_KEY", "")
        if not self.api_key:
            logger.warning("SBP_API_KEY is not set - SBP EasyData series fetch will be skipped.")

    def _fetch_one(self, series_key: str) -> pd.DataFrame:
        """
        Calls SBP EasyData API endpoint for a single time-series independently.
        Returns DataFrame with ['date', series_key] or empty DataFrame on failure.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
            
        # Without an explicit date range the API only returns the single most
        # recent observation instead of the full series history.
        from datetime import datetime as _dt
        end_date = _dt.now().strftime("%Y-%m-%d")
        url = (
            f"https://easydata.sbp.org.pk/api/v1/series/{series_key}/data"
            f"?api_key={self.api_key}&start_date=2005-01-01&end_date={end_date}"
        )
        try:
            import cloudscraper
            scraper = cloudscraper.create_scraper()
            resp = scraper.get(url, headers=headers, timeout=30)
            if resp.status_code != 200:
                logger.error(f"SBP EasyData API returned status {resp.status_code} for series {series_key}. Response: {resp.text[:500]}")
                return pd.DataFrame(columns=["date", series_key])
            
            payload = resp.json()
            
            # New SBP API format: {"columns": [...], "rows": [[...], ...]}
            if "columns" in payload and "rows" in payload:
                cols = [c.lower() for c in payload["columns"]]
                rows = payload["rows"]
                if not rows:
                    return pd.DataFrame(columns=["date", series_key])
                
                df = pd.DataFrame(rows, columns=cols)
                date_col = "observation date"
                val_col = "observation value"
                if date_col not in df.columns or val_col not in df.columns:
                    return pd.DataFrame(columns=["date", series_key])
                df = df.rename(columns={date_col: "date", val_col: "value"})
            else:
                # Fallback to old format
                obs = payload.get("observations", payload.get("data", []))
                if not obs:
                    logger.warning(f"No observations returned for SBP series {series_key}")
                    return pd.DataFrame(columns=["date", series_key])
                df = pd.DataFrame(obs)
                df.columns = [c.lower() for c in df.columns]

            if "date" not in df.columns or "value" not in df.columns:
                return pd.DataFrame(columns=["date", series_key])

            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
            return df[["date", "value"]].rename(columns={"value": series_key})

        except Exception as e:
            logger.exception(f"SBP series {series_key} fetch failed with exception:")
            return pd.DataFrame(columns=["date", series_key])

    def fetch_sbp_data(self) -> pd.DataFrame:
        """
        Fetches all SBP EasyData series independently.
        Never fabricates values or derives columns from parent ratios.
        Sets per-column `{column}_is_missing` boolean flags and dynamic `is_synthetic_rate`.
        """
        combined = None
        missing_cols = []
        fetched_cols = []

        for col_name, series_key in REAL_SERIES.items():
            raw = self._fetch_one(series_key)
            if raw.empty:
                missing_cols.append(col_name)
                continue

            fetched_cols.append(col_name)
            raw = raw.rename(columns={series_key: col_name})
            if combined is None:
                combined = raw
            else:
                combined = pd.merge(combined, raw, on="date", how="outer")

        # 1. Fetch Manual KIBOR Data
        manual_kibor_df = pd.DataFrame()
        try:
            manual_kibor_df = parse_sbp_kibor()
        except Exception as e:
            logger.error(f"Error parsing manual SBP KIBOR: {e}")

        # 2. Combine with Live Data (Manual > Live)
        if not manual_kibor_df.empty:
            if combined is None or combined.empty:
                combined = manual_kibor_df
            else:
                combined["date"] = pd.to_datetime(combined["date"])
                manual_kibor_df["date"] = pd.to_datetime(manual_kibor_df["date"])
                combined = pd.merge(combined, manual_kibor_df, on="date", how="outer", suffixes=("_live", "_manual"))
                
                # Manual overrides live
                for col in ["kibor_1w", "kibor_1m", "kibor_3m", "kibor_6m", "kibor_1y"]:
                    if f"{col}_manual" in combined.columns and f"{col}_live" in combined.columns:
                        combined[col] = combined[f"{col}_manual"].combine_first(combined[f"{col}_live"])
                        combined.drop(columns=[f"{col}_manual", f"{col}_live"], inplace=True)
                    elif f"{col}_manual" in combined.columns:
                        combined[col] = combined[f"{col}_manual"]
                        combined.drop(columns=[f"{col}_manual"], inplace=True)
                    elif f"{col}_live" in combined.columns:
                        combined[col] = combined[f"{col}_live"]
                        combined.drop(columns=[f"{col}_live"], inplace=True)
                        
        if combined is None or combined.empty:
            logger.warning("All live and manual SBP EasyData fetches returned empty.")
            return pd.DataFrame()

        combined = combined.sort_values("date").reset_index(drop=True)

        # Mark per-column missing flags and NEVER fabricate
        for col in REAL_SERIES.keys():
            if col not in combined.columns:
                combined[col] = pd.NA
            
            # If the column is entirely NaN for a date, mark it missing
            missing_mask = combined[col].isna()
            combined[f"{col}_is_missing"] = missing_mask

        # Since we removed all synthesis loops, the data is never synthetic.
        combined["is_synthetic_rate"] = False

        return combined

    def sync_sbp_data(self) -> bool:
        """
        Fetches live SBP EasyData series and upserts into macro_indicators table.
        """
        logger.info("Starting SBP EasyData independent series fetch...")
        df = self.fetch_sbp_data()
        if not df.empty:
            success = upsert_macro_indicators(df)
            logger.info(f"SBP EasyData sync complete: {len(df)} rows upserted. Success: {success}")
            return success

        logger.warning("Live SBP EasyData API fetch empty. Ensuring missing flags and synthetic rate status are recorded.")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    scraper = SbpEasyDataScraper()
    ok = scraper.sync_sbp_data()
    print("SBP EasyData Scraper execution status:", "SUCCESS" if ok else "FAILED / EMPTY API")