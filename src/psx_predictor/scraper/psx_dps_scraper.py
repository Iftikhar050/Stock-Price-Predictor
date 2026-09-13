import logging
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
from src.psx_predictor.db.repository import upsert_stock_fundamentals

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

class PsxDpsScraper:
    """
    Direct scraper for official PSX Data Portal (dps.psx.com.pk) company financials.
    Pulls historical quarterly & annual statements, YoY EPS growth, Net Margin, and PEG ratio.
    """
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }

    def _parse_val(self, val_str: str) -> float:
        if not val_str or val_str == '-' or val_str == 'N/A':
            return np.nan
        val_clean = val_str.replace(',', '').replace('%', '').strip()
        if val_clean.startswith('(') and val_clean.endswith(')'):
            return -float(val_clean[1:-1])
        try:
            return float(val_clean)
        except ValueError:
            return np.nan

    def _reconcile_annual_scale(self, rev_val: float, net_val: float, eps_val: float, shares_val: float) -> tuple:
        """
        PSX's DPS annual-financials table doesn't render Sales/Profit-after-Tax at a
        fixed, predictable order of magnitude - confirmed by live-scraping PSO/MEBL/FCCL
        and cross-checking against their own displayed EPS: every fresh scrape currently
        implies the raw Sales/Profit digits are short by ~1000x relative to what
        (EPS * shares outstanding) says net income should be, and stored historical rows
        show yet other scales for the same ticker/year. Rather than hardcode a multiplier
        that will silently drift wrong again the next time PSX changes its rendering, use
        EPS (a stable per-share ratio) and shares outstanding (scraped from this same
        page's stats block) as a live anchor: derive the scale factor needed to match
        them, snapped to the nearest power of 10, and apply it to both net_income and
        revenue (reported in the same table, same row-set, so presumed to share whatever
        scale convention that row-set used for this scrape).
        """
        if np.isnan(eps_val) or eps_val == 0 or np.isnan(shares_val) or shares_val <= 0:
            return rev_val, net_val  # no stable anchor available - leave as scraped

        expected_net = eps_val * shares_val

        if np.isnan(net_val) or net_val == 0:
            # Profit-after-Tax missing/zero on the page - derive it from the EPS anchor.
            # No anchor exists for revenue's scale in this branch, so leave it untouched.
            return rev_val, expected_net

        ratio = expected_net / net_val
        if 1 / 15.0 < abs(ratio) < 15.0:
            return rev_val, net_val  # within a plausible band (share count drifts some year to year via bonus issues) - trust the scraped figure

        scale = 10 ** round(np.log10(abs(ratio)))
        net_val = net_val * scale
        if not np.isnan(rev_val):
            rev_val = rev_val * scale
        return rev_val, net_val

    @staticmethod
    def _plausibility_gate(rev_val: float, net_val: float) -> tuple:
        """
        Final safety net independent of the EPS anchor above (covers rows where EPS
        itself failed to parse): a real PSX-listed company's net profit margin is never
        remotely close to +-100x of revenue. A handful of historical scrapes produced
        single/double/triple-digit "revenue" values (e.g. 17, 25, 30, 100) alongside a
        real-looking net_income in the tens of millions - obvious mis-parses, not a
        scale issue. Null rather than keep a value nothing can anchor as real.
        """
        if np.isnan(rev_val) or np.isnan(net_val) or rev_val == 0:
            return rev_val, net_val
        margin = net_val / rev_val
        if abs(margin) > 3.0:
            return np.nan, net_val
        return rev_val, net_val

    def scrape_company_financials(self, ticker: str) -> bool:
        url = f"https://dps.psx.com.pk/company/{ticker}"
        logger.info(f"Scraping official PSX DPS financial statements for {ticker} from {url}...")
        try:
            r = requests.get(url, headers=self.headers, timeout=10)
            if r.status_code != 200:
                logger.error(f"Failed to fetch {url}, status code: {r.status_code}")
                return False
                
            soup = BeautifulSoup(r.text, 'html.parser')
            tables = soup.find_all('table')
            
            if len(tables) < 7:
                logger.warning(f"Unexpected table structure on PSX DPS page for {ticker}. Found {len(tables)} tables.")
                return False

            # Table 5: Annual Financials
            # Table 6: Quarterly Financials
            # Table 7: Financial Ratios
            
            # 0. Extract Stats items (Market Cap, Shares, Free Float)
            stats = {}
            for item in soup.find_all('div', class_='stats_item'):
                label = item.find('div', class_='stats_label')
                value = item.find('div', class_='stats_value')
                if label and value:
                    stats[label.text.strip()] = value.text.strip()
                    
            mcap_val = self._parse_val(stats.get("Market Cap (000's)"))
            if not np.isnan(mcap_val):
                mcap_val = mcap_val * 1000.0  # Convert from thousands
            shares_val = self._parse_val(stats.get("Shares"))
            
            ff_shares_val = np.nan
            ff_pct_val = np.nan
            
            # Note: There are two "Free Float" labels in stats, one for shares and one for %
            for item in soup.find_all('div', class_='stats_item'):
                label = item.find('div', class_='stats_label')
                value = item.find('div', class_='stats_value')
                if label and value and 'free float' in label.text.lower():
                    v_text = value.text.strip()
                    if '%' in v_text:
                        ff_pct_val = self._parse_val(v_text) / 100.0
                    else:
                        ff_shares_val = self._parse_val(v_text)

            records = []
            
            # 1. Process Table 7 (Ratios per year)
            ratio_table = tables[6]
            ratio_rows = ratio_table.find_all('tr')
            years = [ele.text.strip() for ele in ratio_rows[0].find_all(['td', 'th']) if ele.text.strip()]
            
            ratio_data = {}
            for row in ratio_rows[1:]:
                cols = [ele.text.strip() for ele in row.find_all(['td', 'th'])]
                if not cols:
                    continue
                metric_name = cols[0]
                values = cols[1:]
                ratio_data[metric_name] = dict(zip(years, values))
                
            # 2. Process Table 5 (Annual Financial Statements)
            annual_table = tables[4]
            annual_rows = annual_table.find_all('tr')
            ann_years = [ele.text.strip() for ele in annual_rows[0].find_all(['td', 'th']) if ele.text.strip()]
            
            ann_data = {}
            for row in annual_rows[1:]:
                cols = [ele.text.strip() for ele in row.find_all(['td', 'th'])]
                if not cols:
                    continue
                metric_name = cols[0]
                values = cols[1:]
                ann_data[metric_name] = dict(zip(ann_years, values))
                
            for yr in ann_years:
                if not yr.isdigit():
                    continue
                rep_date = datetime.strptime(f"{yr}-12-31", "%Y-%m-%d").date()
                
                rev_str = ann_data.get('Sales', {}).get(yr) or ann_data.get('Mark-up Earned', {}).get(yr)
                net_str = ann_data.get('Profit after Taxation', {}).get(yr)
                eps_str = ann_data.get('EPS', {}).get(yr)
                
                eps_growth_str = ratio_data.get('EPS Growth (%)', {}).get(yr)
                peg_str = ratio_data.get('PEG', {}).get(yr)
                gross_margin_str = ratio_data.get('Gross Profit Margin (%)', {}).get(yr)
                net_margin_str = ratio_data.get('Net Profit Margin (%)', {}).get(yr)

                rev_val = self._parse_val(rev_str)
                net_val = self._parse_val(net_str)
                eps_val = self._parse_val(eps_str)
                rev_val, net_val = self._reconcile_annual_scale(rev_val, net_val, eps_val, shares_val)
                rev_val, net_val = self._plausibility_gate(rev_val, net_val)

                rec = {
                    'ticker': ticker,
                    'report_date': rep_date,
                    'revenue': rev_val,
                    'net_income': net_val,
                    'eps': eps_val,
                    'eps_growth_yoy': self._parse_val(eps_growth_str) / 100.0 if eps_growth_str and not np.isnan(self._parse_val(eps_growth_str)) else np.nan,
                    'peg_ratio': self._parse_val(peg_str),
                    'gross_profit_margin': self._parse_val(gross_margin_str) / 100.0 if gross_margin_str and not np.isnan(self._parse_val(gross_margin_str)) else np.nan,
                    'net_profit_margin': self._parse_val(net_margin_str) / 100.0 if net_margin_str and not np.isnan(self._parse_val(net_margin_str)) else np.nan,
                    'shares_outstanding': shares_val,
                    'free_float': ff_shares_val,
                    'free_float_pct': ff_pct_val,
                    'market_cap': mcap_val
                }
                records.append(rec)
                
            if not records:
                logger.warning(f"No financial records extracted for {ticker}.")
                return False
                
            df = pd.DataFrame(records)
            df = df.dropna(how='all', subset=['revenue', 'net_income', 'eps', 'eps_growth_yoy'])
            
            success = upsert_stock_fundamentals(df)
            if success:
                logger.info(f"Successfully upserted official PSX DPS fundamentals for {ticker} ({len(df)} reports).")
            return success
            
        except Exception as e:
            logger.error(f"Error scraping PSX DPS for {ticker}: {e}")
            return False

if __name__ == "__main__":
    scraper = PsxDpsScraper()
    for t in ['MEBL', 'PSO']:
        scraper.scrape_company_financials(t)
