import os
import sys
import logging
import pandas as pd
import requests
import bs4

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OCACEnergyFetcher")

def fetch_ocac_petroleum_sales() -> bool:
    """
    Attempts to fetch monthly OCAC petroleum sales dispatches (MS, HSD, FO
    volumes), circular debt, and refinery margin from the OCAC portal.

    There is no synthetic fallback here. If the live page is unreachable, has
    moved (confirmed 404 as of this writing - OCAC appears to have
    restructured their statistics pages), or its table structure can't be
    matched to the target columns, this returns False and writes nothing -
    petroleum_sales_volume/circular_debt_level/refinery_margin stay NULL
    rather than being filled with np.random noise dressed up as data.
    Finding OCAC's current statistics URL and real table layout is tracked as
    open follow-up work, not something to guess at here.
    """
    logger.info("Fetching OCAC Pakistan petroleum sales dispatches...")
    url = "https://www.ocac.org.pk/sales-of-petroleum-products/"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    records = []
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            soup = bs4.BeautifulSoup(r.text, 'html.parser')
            tables = soup.find_all('table')
            for table in tables:
                try:
                    df = pd.read_html(str(table))[0]
                    if any('ms' in str(c).lower() or 'hsd' in str(c).lower() or 'sales' in str(c).lower() for c in df.columns):
                        logger.info(f"Parsed OCAC sales table: {df.shape}")
                        for idx, row in df.iterrows():
                            records.append(row)
                except Exception:
                    continue
        else:
            logger.warning(f"OCAC portal HTTP {r.status_code} - no data available, leaving columns unchanged.")
    except Exception as e:
        logger.warning(f"Error scraping OCAC portal: {e} - leaving columns unchanged.")

    if not records:
        logger.info(
            "No OCAC petroleum sales data recovered from a live scrape. "
            "petroleum_sales_volume / circular_debt_level / refinery_margin "
            "will remain NULL for this run (no synthetic fallback)."
        )
        return False

    # TODO: map `records` (raw scraped rows) to petroleum_sales_volume /
    # circular_debt_level / refinery_margin and upsert via
    # src.psx_predictor.db.repository.upsert_macro_indicators once a real,
    # current OCAC table layout is confirmed. Intentionally not implemented
    # against unverified table structure.
    logger.warning(f"Scraped {len(records)} raw OCAC rows but column mapping to target fields is not yet implemented.")
    return False

if __name__ == "__main__":
    fetch_ocac_petroleum_sales()
