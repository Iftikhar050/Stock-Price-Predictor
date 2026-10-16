import sys
sys.path.append('.')
import requests
import logging
import pandas as pd
from datetime import date
from bs4 import BeautifulSoup
from src.psx_predictor.db.repository import upsert_macro_indicators

logger = logging.getLogger(__name__)

class PsxDpsIndexScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        # Confirmed 2026-10-13: the old POST /historical (symbol=KSE30 etc.) path
        # this scraper used has been silently 403'ing since ~2026-09-16 - it
        # logged "PSX Index sync: Failed" on every single pipeline run since
        # then (visible only as an INFO line, not an ERROR, so it went
        # unnoticed for about a month while the frontend kept serving a
        # Sept-15 snapshot). GET https://dps.psx.com.pk/indices still works and
        # is in fact PSX's own live indices page - it only gives a live
        # snapshot (no historical series), so this now upserts one row for
        # today's date per run rather than backfilling a whole series like the
        # old endpoint did. Also lets us carry the REAL KSE100 for the first
        # time (previously only the synthetic MarketIndex proxy existed).
        self.indices = {
            "KSE100": ("kse100_real_index_level", "kse100_real_return_pct"),
            "KMI30": ("kmi30_index_level", "kmi30_return_pct"),
            "KSE30": ("kse30_index_level", "kse30_return_pct"),
            "ALLSHR": ("all_share_index_level", "all_share_return_pct"),
            "BKTI": ("banking_sector_index_level", "banking_sector_return_pct"),
            "OGTI": ("oil_gas_sector_index_level", "oil_gas_sector_return_pct"),
        }

    def fetch_live_snapshot(self) -> dict:
        """Returns {symbol: (current_level, pct_change)} parsed from PSX's own
        live indices table. pct_change is the fraction (e.g. -0.0028 for
        -0.28%), matching the *_return_pct convention used elsewhere."""
        url = "https://dps.psx.com.pk/indices"
        out = {}
        try:
            r = requests.get(url, headers=self.headers, timeout=15)
            if r.status_code != 200:
                logger.error(f"PSX indices page returned HTTP {r.status_code}")
                return out
            soup = BeautifulSoup(r.text, 'html.parser')
            tables = soup.find_all('table')
            if not tables:
                return out
            for row in tables[0].find_all('tr')[1:]:
                cols = [c.text.strip() for c in row.find_all(['td', 'th'])]
                # ['Index', 'High', 'Low', 'Current', 'Change', '% Change']
                if len(cols) < 6:
                    continue
                symbol = cols[0]
                if symbol not in self.indices:
                    continue
                try:
                    current = float(cols[3].replace(',', ''))
                    pct_change = float(cols[5].replace('%', '').replace(',', '')) / 100.0
                    out[symbol] = (current, pct_change)
                except ValueError:
                    continue
        except Exception as e:
            logger.error(f"Error fetching PSX live indices snapshot: {e}")
        return out

    def sync_all_indices(self):
        logger.info("Syncing official PSX indices (KSE-100, KMI-30, KSE-30, All-Share, BKTI, OGTI)...")
        snapshot = self.fetch_live_snapshot()
        if not snapshot:
            return False

        row = {"date": date.today()}
        for sym, (level_col, return_col) in self.indices.items():
            if sym in snapshot:
                row[level_col] = snapshot[sym][0]
                row[return_col] = snapshot[sym][1]

        if len(row) <= 1:
            return False

        upsert_macro_indicators(pd.DataFrame([row]))
        logger.info(f"Successfully synced today's snapshot for {len(snapshot)} official PSX indices!")
        return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = PsxDpsIndexScraper()
    scraper.sync_all_indices()
