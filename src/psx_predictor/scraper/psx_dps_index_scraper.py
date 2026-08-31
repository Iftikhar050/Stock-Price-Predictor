import sys
sys.path.append('.')
import requests
import logging
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
from src.psx_predictor.db.repository import upsert_macro_indicators

logger = logging.getLogger(__name__)

class PsxDpsIndexScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest"
        }
        self.indices = {
            "KMI30": ("kmi30_index_level", "kmi30_return_pct"),
            "KSE30": ("kse30_index_level", "kse30_return_pct"),
            "ALLSHR": ("all_share_index_level", "all_share_return_pct"),
            "BKTI": ("banking_sector_index_level", "banking_sector_return_pct"),
            "OGTI": ("oil_gas_sector_index_level", "oil_gas_sector_return_pct")
        }

    def fetch_index_series(self, symbol):
        url = "https://dps.psx.com.pk/historical"
        try:
            r = requests.post(url, data={"symbol": symbol}, headers=self.headers, timeout=15)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                tables = soup.find_all('table')
                if tables:
                    rows = tables[0].find_all('tr')
                    data = []
                    for row in rows[1:]:
                        cols = [ele.text.strip() for ele in row.find_all(['td', 'th'])]
                        if len(cols) >= 5:
                            # cols[0] is date e.g. "Aug 24, 2026", cols[4] is close e.g. "251,515.82"
                            d_str = cols[0]
                            c_str = cols[4].replace(',', '')
                            try:
                                dt = datetime.strptime(d_str, "%b %d, %Y").date()
                                close_val = float(c_str)
                                data.append({"date": dt, "close": close_val})
                            except Exception:
                                continue
                    if data:
                        df = pd.DataFrame(data).sort_values('date').reset_index(drop=True)
                        return df
        except Exception as e:
            logger.error(f"Error fetching PSX index {symbol}: {e}")
        return pd.DataFrame()

    def sync_all_indices(self):
        logger.info("Syncing official PSX indices (KMI-30, KSE-30, All-Share, BKTI, OGTI)...")
        master_df = None
        
        for sym, (level_col, return_col) in self.indices.items():
            df = self.fetch_index_series(sym)
            if not df.empty:
                df[level_col] = df['close']
                df[return_col] = df['close'].pct_change().fillna(0.0)
                df = df[['date', level_col, return_col]]
                
                if master_df is None:
                    master_df = df
                else:
                    master_df = pd.merge(master_df, df, on='date', how='outer')
                    
        if master_df is not None and not master_df.empty:
            master_df = master_df.sort_values('date').reset_index(drop=True)
            upsert_macro_indicators(master_df)
            logger.info(f"Successfully synced {len(master_df)} rows of official PSX sector indices!")
            return True
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = PsxDpsIndexScraper()
    scraper.sync_all_indices()
