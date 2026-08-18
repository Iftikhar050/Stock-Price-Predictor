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
            
            # Global Indices
            sp500_data = yf.download("^GSPC", period="max", progress=False)
            nasdaq_data = yf.download("^IXIC", period="max", progress=False)
            dxy_data = yf.download("DX-Y.NYB", period="max", progress=False)
            us10y_data = yf.download("^TNX", period="max", progress=False)
            
            if pkr_data.empty and oil_data.empty and sp500_data.empty:
                logger.warning("Failed to fetch macro data.")
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
            sp500_df = clean_yf(sp500_data, 'sp500_close')
            nasdaq_df = clean_yf(nasdaq_data, 'nasdaq_close')
            dxy_df = clean_yf(dxy_data, 'dxy_close')
            us10y_df = clean_yf(us10y_data, 'us10y_yield')
            
            dataframes = [df for df in [pkr_df, oil_df, sp500_df, nasdaq_df, dxy_df, us10y_df] if not df.empty]
            
            if not dataframes:
                return False
                
            macro_df = dataframes[0]
            for df in dataframes[1:]:
                macro_df = pd.merge(macro_df, df, on='date', how='outer')
                
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
