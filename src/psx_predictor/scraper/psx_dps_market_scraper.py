import sys
sys.path.append('.')
import requests
import logging
import pandas as pd
import numpy as np
from datetime import date, datetime
from src.psx_predictor.db.repository import upsert_macro_indicators

logger = logging.getLogger(__name__)

class PsxDpsMarketScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest"
        }
        
    def fetch_market_breadth(self):
        """
        Fetches live PSX Market Breadth (Advancing %, Declining %, Breadth Ratio)
        from https://dps.psx.com.pk/data/symbol-position
        """
        url = "https://dps.psx.com.pk/data/symbol-position"
        try:
            r = requests.get(url, headers=self.headers, timeout=10)
            if r.status_code == 200:
                data = r.json()
                data_dict = {item['name']: item['value'] for item in data}
                
                adv_pct = data_dict.get("ADV", 0.0)
                dec_pct = data_dict.get("DEC", 0.0)
                breadth_ratio = adv_pct / dec_pct if dec_pct > 0 else 1.0
                
                today_date = date.today()
                df = pd.DataFrame([{
                    "date": today_date,
                    "advancing_stocks_pct": adv_pct,
                    "declining_stocks_pct": dec_pct,
                    "market_breadth_ratio": breadth_ratio
                }])
                
                upsert_macro_indicators(df)
                logger.info(f"Successfully synced PSX Market Breadth for {today_date}: ADV={adv_pct:.2%}, DEC={dec_pct:.2%}, Ratio={breadth_ratio:.2f}")
                return True
        except Exception as e:
            logger.error(f"Failed to fetch PSX Market Breadth: {e}")
            return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = PsxDpsMarketScraper()
    scraper.fetch_market_breadth()
