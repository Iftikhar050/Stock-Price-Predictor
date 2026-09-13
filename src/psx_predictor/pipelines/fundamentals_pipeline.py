import os
import sys
import logging
from typing import List, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

import pandas as pd

from src.psx_predictor.scraper.client import PSXScraper
from src.psx_predictor.scraper.fundamentals_scraper import FundamentalsScraper
from src.psx_predictor.scraper.dividend_scraper import DividendScraper
from src.psx_predictor.scraper.psx_dps_scraper import PsxDpsScraper
from src.psx_predictor.data.parse_fundamentals_manual import parse_fundamentals_manual, upsert_fundamentals
from src.psx_predictor.pipelines.rate_limit import throttle

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
    dps_scraper = PsxDpsScraper()
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
        throttle()

    # 2. Quarterly Financial Statements
    logger.info("\n[Step 2/4] Syncing Financial Statements & Fundamentals...")
    for ticker in tickers:
        try:
            # First attempt: Manual files
            manual_df = pd.DataFrame()
            if ticker == "PSO":
                manual_df = parse_fundamentals_manual("fundamentals_pso")
            elif ticker == "MEBL":
                manual_df = parse_fundamentals_manual("fundamentals_mebl")
                
            res_yfinance = fund_scraper.sync_fundamentals(ticker)
            logger.info(f" Yahoo Finance statements sync for {ticker}: {'Success' if res_yfinance else 'Failed'}")

            res_dps = dps_scraper.scrape_company_financials(ticker)
            logger.info(f" PSX DPS statements sync for {ticker}: {'Success' if res_dps else 'Failed'}")

            # Manual filing PDFs win over both live sources ("Manual File > Live Fetch > NaN"),
            # so upsert last: same (ticker, report_date) rows from yfinance/DPS above get
            # overwritten by the deeper, verified manual data where a manual record exists.
            if not manual_df.empty:
                logger.info(f" Loaded {len(manual_df)} manual fundamental records for {ticker}")
                upsert_fundamentals(manual_df)
                logger.info(f" Upserted manual fundamentals for {ticker} (takes priority over live fetch)")
        except Exception as e:
            logger.error(f" Error syncing financial statements for {ticker}: {e}")
            success = False
        throttle()

    # 3. Dividends & Corporate Payouts
    logger.info("\n[Step 3/4] Syncing Dividend History...")
    for ticker in tickers:
        try:
            res3 = div_scraper.sync_dividends(ticker)
            logger.info(f" Dividend sync for {ticker}: {'Success' if res3 else 'Failed'}")
        except Exception as e:
            logger.error(f" Error syncing dividends for {ticker}: {e}")
            success = False
        throttle()

    # NOTE: Bank-specific (NIM, CASA, NPL, capital adequacy, ADR/IDR) and OMC-specific
    # (refinery margin, circular debt, government receivables) metrics are not collected
    # yet - there is no live, freely-scrapable source for these regulatory disclosures
    # confirmed working. Do not hardcode placeholder values here; leave these columns
    # null until a real source (e.g. SBP prudential returns, PSX DPS financial notes) is wired up.

    logger.info("=========================================")
    logger.info("SECTOR & COMPANY FUNDAMENTALS PIPELINE COMPLETE")
    logger.info("=========================================")
    return success

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    run_fundamentals_pipeline()
