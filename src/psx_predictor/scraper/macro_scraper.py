import logging
import pandas as pd
import yfinance as yf
from datetime import datetime
from src.psx_predictor.db.repository import upsert_macro_indicators

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

class MacroScraper:
    def sync_macro(self) -> bool:
        logger.info("Fetching macro indicators from Yahoo Finance")
        try:
            # PKR=X is USD/PKR
            # BZ=F is Brent Crude Oil Last Day Financ
            pkr_data = yf.download("PKR=X", period="max", progress=False)
            oil_data = yf.download("BZ=F", period="max", progress=False)
            
            if pkr_data.empty and oil_data.empty:
                logger.warning("Failed to fetch both PKR and Oil data.")
                return False
                
            # Clean and merge
            def clean_yf(df, col_name):
                if df.empty:
                    return pd.DataFrame(columns=['date', col_name])
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                df = df.reset_index()
                df['date'] = pd.to_datetime(df['Date']).dt.date
                df = df[['date', 'Close']].rename(columns={'Close': col_name})
                df = df.dropna()
                return df
                
            pkr_df = clean_yf(pkr_data, 'pkr_usd_rate')
            oil_df = clean_yf(oil_data, 'brent_oil_price')
            
            if pkr_df.empty and oil_df.empty:
                return False
                
            if not pkr_df.empty and not oil_df.empty:
                macro_df = pd.merge(pkr_df, oil_df, on='date', how='outer')
            elif not pkr_df.empty:
                macro_df = pkr_df
                macro_df['brent_oil_price'] = None
            else:
                macro_df = oil_df
                macro_df['pkr_usd_rate'] = None
                
            # For SBP policy rate, since it changes infrequently and requires custom scraping from SBP,
            # we will set it to the current 22.0% (or historical approx) for Phase 1. 
            # In a full production env, we'd add an SBP-specific scraper here.
            logger.warning("Using hardcoded flat placeholder (22.0) for sbp_policy_rate. This is NOT real historical data.")
            macro_df['sbp_policy_rate'] = 22.0 
            macro_df['is_synthetic_rate'] = True
            
            # Sort and fill
            macro_df = macro_df.sort_values('date').reset_index(drop=True)
            
            success = upsert_macro_indicators(macro_df)
            return success
        except Exception as e:
            logger.error(f"Error fetching macro indicators: {e}")
            return False
