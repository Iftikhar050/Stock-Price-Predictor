import os
import sys
import logging
from typing import List, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

from src.psx_predictor.data.fetch_nccpl_flows import fetch_nccpl_flows
from src.psx_predictor.scraper.psx_insider_scraper import PsxInsiderScraper
from src.psx_predictor.scraper.index_scraper import sync_market_index

logger = logging.getLogger("InstitutionalFlowsPipeline")
logger.setLevel(logging.INFO)

def run_institutional_flows_pipeline(tickers: Optional[List[str]] = None) -> bool:
    """
    Executes the Institutional Flows & Market Structure Pipeline:
    1. Ingests NCCPL FIPI / LIPI Institutional Investor Flows.
    2. Ingests SECP Insider Trading Activity & Sponsor/Institutional Shareholding.
    3. Syncs PSX KSE-100 & Sector Market Index Series.
    """
    if tickers is None:
        tickers = ["PSO", "MEBL"]
        
    logger.info("=========================================")
    logger.info("STARTING INSTITUTIONAL FLOWS & MARKET STRUCTURE PIPELINE")
    logger.info("=========================================")
    
    success = True
    
    # 1. NCCPL FIPI / LIPI Investor Flows
    logger.info("\n[Step 1/3] Syncing NCCPL FIPI / LIPI Institutional Investor Flows...")
    try:
        res1 = fetch_nccpl_flows()
        logger.info(f" NCCPL Flows sync: {'Success' if res1 else 'Failed'}")
    except Exception as e:
        logger.error(f" Error syncing NCCPL flows: {e}")
        success = False

    # 2. SECP Insider Trading & Shareholding Structure
    logger.info("\n[Step 2/3] Syncing SECP Insider Trading & Shareholding Ratios...")
    insider_scraper = PsxInsiderScraper()
    for ticker in tickers:
        try:
            res2 = insider_scraper.sync_insider_and_shareholding(ticker)
            logger.info(f" Insider & Shareholding sync for {ticker}: {'Success' if res2 else 'Failed'}")
        except Exception as e:
            logger.error(f" Error syncing insider trades for {ticker}: {e}")
            success = False

    # 3. PSX Market Index Series
    logger.info("\n[Step 3/3] Syncing PSX Market & Sector Indices...")
    try:
        res3 = sync_market_index()
        logger.info(f" PSX Index sync: {'Success' if res3 else 'Failed'}")
    except Exception as e:
        logger.error(f" Error syncing PSX market index: {e}")
        success = False

    logger.info("=========================================")
    logger.info("INSTITUTIONAL FLOWS & MARKET STRUCTURE PIPELINE COMPLETE")
    logger.info("=========================================")
    return success

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    run_institutional_flows_pipeline()
