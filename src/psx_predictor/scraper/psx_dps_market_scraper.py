import sys
sys.path.append('.')
import requests
import logging
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from datetime import date, datetime
from src.psx_predictor.db.repository import upsert_macro_indicators, upsert_circuit_breaker_events

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

    def fetch_circuit_breakers(self):
        """
        Scrapes today's list of symbols that hit their upper/lower circuit
        limit from https://dps.psx.com.pk/circuit-breakers. This is a live
        snapshot with no historical archive, so it can only be collected
        going forward (run daily, e.g. as part of the EOD pipeline).
        """
        url = "https://dps.psx.com.pk/circuit-breakers"
        try:
            r = requests.get(url, headers={"User-Agent": self.headers["User-Agent"]}, timeout=10)
            if r.status_code != 200:
                return False

            soup = BeautifulSoup(r.text, "html.parser")
            tables = soup.find_all("table")
            if len(tables) < 2:
                logger.warning("Circuit breakers page did not return the expected two tables.")
                return False

            today_date = date.today()
            rows = []
            for table, direction in [(tables[0], "upper"), (tables[1], "lower")]:
                for row in table.find_all("tr")[1:]:
                    cols = [c.text.strip() for c in row.find_all(["td", "th"])]
                    if not cols:
                        continue
                    rows.append({"ticker": cols[0].upper(), "date": today_date, "direction": direction})

            if not rows:
                logger.info("No symbols hit a circuit limit today.")
                return True

            df = pd.DataFrame(rows)
            upsert_circuit_breaker_events(df)
            logger.info(f"Synced {len(df)} circuit-breaker hits for {today_date}.")
            return True
        except Exception as e:
            logger.error(f"Failed to fetch PSX circuit breakers: {e}")
            return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = PsxDpsMarketScraper()
    scraper.fetch_market_breadth()
