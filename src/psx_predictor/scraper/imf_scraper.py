import sys
sys.path.append('.')
import requests
import logging
import pandas as pd
import numpy as np
from datetime import date, datetime
from src.psx_predictor.db.repository import upsert_macro_indicators

logger = logging.getLogger(__name__)

class ImfScraper:
    """
    Scraper for the official IMF DataMapper REST API for Pakistan.
    Syncs 13 real IMF DataMapper macroeconomic/fiscal indicators into PostgreSQL.

    SDR allocation/holdings, quota, loan-program disbursements, and net
    financial position are NOT included here: they come from the IMF
    Financial Data Query Tool (imf.org/external/np/fin/tad/query.aspx), a
    form-based page with no confirmed JSON/REST endpoint, not the DataMapper
    API this class actually calls - a previous version hardcoded 6 static
    constants for them (values that were already off by an order of
    magnitude vs. IMF's own published figures) and labeled the sync "20
    indicators". They were removed rather than left wrong; wiring up a real
    FDQT source is open follow-up work, not something to fake in the
    meantime.
    """
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
        }
        self.indicator_map = {
            "NGDP_RPCH": "imf_real_gdp_growth",
            "PCPIPCH": "imf_cpi_inflation",
            "GGXWDG_NGDP": "imf_govt_gross_debt_pct_gdp",
            "BCA_NGDPD": "imf_current_account_balance_pct_gdp",
            "GGR_NGDP": "imf_govt_revenue_pct_gdp",
            "GGX_NGDP": "imf_govt_expenditure_pct_gdp",
            "GGXCNL_NGDP": "imf_govt_fiscal_balance_pct_gdp",
            "NID_NGDP": "imf_investment_pct_gdp",
            "NGSD_NGDP": "imf_national_savings_pct_gdp",
            "LUR": "imf_unemployment_rate",
            "TX_RPCH": "imf_export_volume_growth",
            "TM_RPCH": "imf_import_volume_growth",
            "NGDPD": "imf_gdp_usd_billions"
        }

    def fetch_imf_datamapper(self):
        """Fetches annual macroeconomic series for Pakistan from IMF DataMapper."""
        logger.info("Fetching Pakistan indicators from IMF DataMapper REST API...")
        codes = "/".join(self.indicator_map.keys())
        url = f"https://www.imf.org/external/datamapper/api/v1/{codes}/PAK"
        
        try:
            r = requests.get(url, headers=self.headers, timeout=20)
            if r.status_code == 200:
                data = r.json().get("values", {})
                records = {}
                
                # Build 2005 - 2026 daily date range
                dates = pd.date_range(start="2005-01-01", end=date.today().strftime("%Y-%m-%d"), freq="B")
                df_daily = pd.DataFrame({"date": dates.date})
                
                for imf_code, col_name in self.indicator_map.items():
                    pak_series = data.get(imf_code, {}).get("PAK", {})
                    if pak_series:
                        # Convert year dict to DataFrame
                        df_yr = pd.DataFrame(list(pak_series.items()), columns=['year_str', col_name])
                        df_yr['year'] = pd.to_numeric(df_yr['year_str'], errors='coerce')
                        df_yr = df_yr.dropna(subset=['year'])
                        
                        # Merge onto daily dates using year match. Forward-fill only:
                        # a year's real figure must never be projected backward onto
                        # earlier years that don't have it yet (look-ahead bias), and
                        # an indicator with no data at all stays NaN rather than 0.0 -
                        # a silent zero reads as "no fiscal deficit", not "unknown".
                        df_daily['year'] = pd.to_datetime(df_daily['date']).dt.year
                        df_daily = pd.merge(df_daily, df_yr[['year', col_name]], on='year', how='left')
                        df_daily[col_name] = df_daily[col_name].ffill()
                        df_daily.drop(columns=['year'], inplace=True, errors='ignore')
                        
                return df_daily
        except Exception as e:
            logger.error(f"Error fetching IMF DataMapper: {e}")
        return pd.DataFrame()

    def sync_imf_indicators(self):
        """Syncs the 13 real IMF DataMapper indicators for Pakistan into macro_indicators.

        Does not write imf_sdr_allocation_bal, imf_sdr_holdings_bal,
        imf_total_loans_outstanding, imf_quota_sdrs, imf_tranche_disbursements,
        imf_net_financial_position, or imf_primary_balance_pct_gdp - see the
        class docstring for why. Those columns are left untouched (NULL if
        never populated) rather than filled with a hardcoded guess.
        """
        logger.info("Syncing 13 real IMF DataMapper macroeconomic indicators...")
        df_imf = self.fetch_imf_datamapper()

        if df_imf.empty or len(df_imf) == 0:
            logger.warning("IMF DataMapper returned no data - nothing to sync.")
            return False

        upsert_macro_indicators(df_imf)
        logger.info(f"Successfully synced {len(df_imf)} records of real IMF Pakistan indicators into PostgreSQL!")
        return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = ImfScraper()
    scraper.sync_imf_indicators()
