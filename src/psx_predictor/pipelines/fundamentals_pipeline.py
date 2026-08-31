import os
import sys
import logging
from typing import List, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

from src.psx_predictor.scraper.client import PSXScraper
from src.psx_predictor.scraper.fundamentals_scraper import FundamentalsScraper
from src.psx_predictor.scraper.dividend_scraper import DividendScraper
from src.psx_predictor.data.fetch_pso_metrics import collect_pso_metrics
from src.psx_predictor.data.fetch_mebl_metrics import collect_mebl_metrics

logger = logging.getLogger("FundamentalsPipeline")
logger.setLevel(logging.INFO)

def run_fundamentals_pipeline(tickers: Optional[List[str]] = None) -> bool:
    """
    Executes the Sector & Company Fundamentals Pipeline:
    1. Ingests PSX EOD Price & Volume time series.
    2. Ingests Quarterly/Annual Financial Statements.
    3. Ingests Dividend Payout & Corporate Action History.
    4. Ingests Targeted Sector Metrics (PSO Energy & MEBL Banking).
    """
    if tickers is None:
        tickers = ["PSO", "MEBL"]
        
    logger.info("=========================================")
    logger.info("STARTING SECTOR & COMPANY FUNDAMENTALS PIPELINE")
    logger.info("=========================================")
    
    success = True
    psx_scraper = PSXScraper()
    fund_scraper = FundamentalsScraper()
    div_scraper = DividendScraper()
    
    # 1. PSX EOD Data
    logger.info("\n[Step 1/4] Syncing PSX EOD Stock Price & Volume Series...")
    for ticker in tickers:
        try:
            res1 = psx_scraper.sync_ticker(ticker)
            logger.info(f" EOD data sync for {ticker}: {'Success' if res1 else 'Failed'}")
        except Exception as e:
            logger.error(f" Error syncing EOD data for {ticker}: {e}")
            success = False

    # 2. Quarterly Financial Statements
    logger.info("\n[Step 2/4] Syncing Financial Statements & Fundamentals...")
    for ticker in tickers:
        try:
            res2 = fund_scraper.sync_fundamentals(ticker)
            logger.info(f" Financial statements sync for {ticker}: {'Success' if res2 else 'Failed'}")
        except Exception as e:
            logger.error(f" Error syncing financial statements for {ticker}: {e}")
            success = False

    # 3. Dividends & Corporate Payouts
    logger.info("\n[Step 3/4] Syncing Dividend History...")
    for ticker in tickers:
        try:
            res3 = div_scraper.sync_dividends(ticker)
            logger.info(f" Dividend sync for {ticker}: {'Success' if res3 else 'Failed'}")
        except Exception as e:
            logger.error(f" Error syncing dividends for {ticker}: {e}")
            success = False

    # 4. Sector-Specific Deep Metrics
    logger.info("\n[Step 4/4] Syncing Dedicated Sector Metrics (Energy & Banking)...")
    try:
        if "PSO" in [t.upper() for t in tickers]:
            logger.info(" Syncing PSO Energy Sector Metrics...")
            collect_pso_metrics()
            
        if "MEBL" in [t.upper() for t in tickers]:
            logger.info(" Syncing MEBL Banking Sector Metrics...")
            collect_mebl_metrics()
    except Exception as e:
        logger.error(f" Error syncing sector metrics: {e}")
        success = False

    logger.info("=========================================")
    logger.info("SECTOR & COMPANY FUNDAMENTALS PIPELINE COMPLETE")
    logger.info("=========================================")
    return success

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    run_fundamentals_pipeline()
