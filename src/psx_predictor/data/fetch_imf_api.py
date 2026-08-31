import os
import sys
import logging
import requests
import pandas as pd
from datetime import datetime
from sqlalchemy import text

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

from src.psx_predictor.db.connection import engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IMFDataCollector")

IMF_API_BASE = "http://dataservices.imf.org/REST/SDMX_JSON.svc"

def fetch_imf_pakistan_series():
    """
    Fetches official macro-economic series for Pakistan (PAK) from the IMF SDMX/JSON REST API
    and populates macro_indicators in PostgreSQL.
    """
    logger.info("Fetching official Pakistan (PAK) data series from IMF SDMX REST API...")
    
    # Official IMF WEO Indicator codes
    indicators = {
        "NGDP_RPCH": "imf_real_gdp_growth",
        "PCPIPCH": "imf_cpi_inflation",
        "GGXWDG_NGDP": "imf_govt_gross_debt_pct_gdp",
        "BCA_NGDPD": "imf_current_account_balance_pct_gdp",
        "GGR_NGDP": "imf_govt_revenue_pct_gdp",
        "GGX_NGDP": "imf_govt_expenditure_pct_gdp",
        "GGXCNL_NGDP": "imf_govt_fiscal_balance_pct_gdp",
        "GGXONLB_NGDP": "imf_primary_balance_pct_gdp",
        "NID_NGDP": "imf_investment_pct_gdp",
        "NGSD_NGDP": "imf_national_savings_pct_gdp",
        "LUR": "imf_unemployment_rate",
        "TX_RPCH": "imf_export_volume_growth",
        "TM_RPCH": "imf_import_volume_growth",
        "NGDPD": "imf_gdp_usd_billions"
    }
    
    records_by_year = {}
    
    for code, col_name in indicators.items():
        url = f"{IMF_API_BASE}/CompactData/WEO/A.PAK.{code}"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                series = data.get("CompactData", {}).get("DataSet", {}).get("Series", {})
                obs_list = series.get("Obs", [])
                if isinstance(obs_list, dict):
                    obs_list = [obs_list]
                
                for obs in obs_list:
                    time_period = obs.get("@TIME_PERIOD")
                    val = obs.get("@OBS_VALUE")
                    if time_period and val is not None:
                        year = int(time_period)
                        if year not in records_by_year:
                            records_by_year[year] = {}
                        records_by_year[year][col_name] = float(val)
                logger.info(f"Successfully retrieved IMF indicator: {code} -> {col_name}")
            else:
                logger.warning(f"IMF API returned status code {r.status_code} for indicator {code}")
        except Exception as e:
            logger.error(f"Error fetching IMF indicator {code}: {e}")

    if not records_by_year:
        logger.warning("No data retrieved from IMF API; using local fallback dataset.")
        return False
        
    logger.info(f"Parsed {len(records_by_year)} annual IMF indicator records for Pakistan.")
    
    # Expand annual IMF figures across daily dates in macro_indicators
    daily_records = []
    for year, metrics in sorted(records_by_year.items()):
        # Map to annual date (Dec 31st of each year)
        dt_str = f"{year}-12-31"
        row = {"date": dt_str}
        row.update(metrics)
        daily_records.append(row)
        
    df_imf = pd.DataFrame(daily_records)
    df_imf['date'] = pd.to_datetime(df_imf['date']).astype('datetime64[ns]')
    
    # Forward-fill to annual grid
    with engine.connect() as conn:
        all_dates = conn.execute(text("SELECT DISTINCT date FROM stock_eod_data ORDER BY date ASC")).fetchall()
        
    df_grid = pd.DataFrame(all_dates, columns=['date'])
    df_grid['date'] = pd.to_datetime(df_grid['date']).astype('datetime64[ns]')
    
    df_merged = pd.merge_asof(df_grid.sort_values('date'), df_imf.sort_values('date'), on='date', direction='backward')
    df_merged = df_merged.bfill().fillna(0.0)
    
    # Update PostgreSQL macro_indicators table
    update_sql = text("""
        INSERT INTO macro_indicators (
            date, imf_real_gdp_growth, imf_cpi_inflation, imf_govt_gross_debt_pct_gdp,
            imf_current_account_balance_pct_gdp, imf_govt_revenue_pct_gdp, imf_govt_expenditure_pct_gdp,
            imf_govt_fiscal_balance_pct_gdp, imf_primary_balance_pct_gdp, imf_investment_pct_gdp,
            imf_national_savings_pct_gdp, imf_unemployment_rate, imf_export_volume_growth,
            imf_import_volume_growth, imf_gdp_usd_billions
        ) VALUES (
            :date, :imf_real_gdp_growth, :imf_cpi_inflation, :imf_govt_gross_debt_pct_gdp,
            :imf_current_account_balance_pct_gdp, :imf_govt_revenue_pct_gdp, :imf_govt_expenditure_pct_gdp,
            :imf_govt_fiscal_balance_pct_gdp, :imf_primary_balance_pct_gdp, :imf_investment_pct_gdp,
            :imf_national_savings_pct_gdp, :imf_unemployment_rate, :imf_export_volume_growth,
            :imf_import_volume_growth, :imf_gdp_usd_billions
        )
        ON CONFLICT (date) DO UPDATE SET
            imf_real_gdp_growth = EXCLUDED.imf_real_gdp_growth,
            imf_cpi_inflation = EXCLUDED.imf_cpi_inflation,
            imf_govt_gross_debt_pct_gdp = EXCLUDED.imf_govt_gross_debt_pct_gdp,
            imf_current_account_balance_pct_gdp = EXCLUDED.imf_current_account_balance_pct_gdp,
            imf_govt_revenue_pct_gdp = EXCLUDED.imf_govt_revenue_pct_gdp,
            imf_govt_expenditure_pct_gdp = EXCLUDED.imf_govt_expenditure_pct_gdp,
            imf_govt_fiscal_balance_pct_gdp = EXCLUDED.imf_govt_fiscal_balance_pct_gdp,
            imf_primary_balance_pct_gdp = EXCLUDED.imf_primary_balance_pct_gdp,
            imf_investment_pct_gdp = EXCLUDED.imf_investment_pct_gdp,
            imf_national_savings_pct_gdp = EXCLUDED.imf_national_savings_pct_gdp,
            imf_unemployment_rate = EXCLUDED.imf_unemployment_rate,
            imf_export_volume_growth = EXCLUDED.imf_export_volume_growth,
            imf_import_volume_growth = EXCLUDED.imf_import_volume_growth,
            imf_gdp_usd_billions = EXCLUDED.imf_gdp_usd_billions;
    """)
    
    with engine.connect() as conn:
        for idx, row in df_merged.iterrows():
            conn.execute(update_sql, {
                "date": row['date'].date(),
                "imf_real_gdp_growth": float(row.get('imf_real_gdp_growth', 0.0)),
                "imf_cpi_inflation": float(row.get('imf_cpi_inflation', 0.0)),
                "imf_govt_gross_debt_pct_gdp": float(row.get('imf_govt_gross_debt_pct_gdp', 0.0)),
                "imf_current_account_balance_pct_gdp": float(row.get('imf_current_account_balance_pct_gdp', 0.0)),
                "imf_govt_revenue_pct_gdp": float(row.get('imf_govt_revenue_pct_gdp', 0.0)),
                "imf_govt_expenditure_pct_gdp": float(row.get('imf_govt_expenditure_pct_gdp', 0.0)),
                "imf_govt_fiscal_balance_pct_gdp": float(row.get('imf_govt_fiscal_balance_pct_gdp', 0.0)),
                "imf_primary_balance_pct_gdp": float(row.get('imf_primary_balance_pct_gdp', 0.0)),
                "imf_investment_pct_gdp": float(row.get('imf_investment_pct_gdp', 0.0)),
                "imf_national_savings_pct_gdp": float(row.get('imf_national_savings_pct_gdp', 0.0)),
                "imf_unemployment_rate": float(row.get('imf_unemployment_rate', 0.0)),
                "imf_export_volume_growth": float(row.get('imf_export_volume_growth', 0.0)),
                "imf_import_volume_growth": float(row.get('imf_import_volume_growth', 0.0)),
                "imf_gdp_usd_billions": float(row.get('imf_gdp_usd_billions', 0.0))
            })
        conn.commit()
        
    logger.info("Successfully updated PostgreSQL macro_indicators with official IMF SDMX REST API series!")
    return True

if __name__ == "__main__":
    fetch_imf_pakistan_series()
