import os
import sys
import logging

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

from src.psx_predictor.scraper.sbp_easydata_scraper import SbpEasyDataScraper
from src.psx_predictor.scraper.imf_scraper import ImfScraper
from src.psx_predictor.data.fetch_pbs_stats import fetch_pbs_stats
from src.psx_predictor.scraper.macro_scraper import MacroScraper

logger = logging.getLogger("MacroPipeline")
logger.setLevel(logging.INFO)

def run_macro_pipeline() -> bool:
    """
    Executes the Macroeconomic & Central Bank Pipeline:
    1. Ingests State Bank of Pakistan (SBP) Monetary & Interest Rate Series.
    2. Ingests IMF DataMapper REST API Macro Projections.
    3. Ingests Pakistan Bureau of Statistics (PBS) CPI, LSM & Trade Data.
    4. Ingests Global Commodity Futures & International Equity/Rate Benchmarks.
    """
    logger.info("=========================================")
    logger.info("STARTING MACROECONOMIC & CENTRAL BANK PIPELINE")
    logger.info("=========================================")
    
    success = True
    
    # 1. State Bank of Pakistan (SBP) Rates & Reserves
    logger.info("\n[Step 1/4] Syncing State Bank of Pakistan (SBP) Rates & Fixed Income...")
    try:
        scraper = SbpEasyDataScraper()
        res1 = scraper.sync_sbp_data()
        logger.info(f" SBP Indicators sync: {'Success' if res1 else 'Failed'}")
    except Exception as e:
        logger.error(f" Error syncing SBP indicators: {e}")
        success = False

    # 2. IMF DataMapper API
    logger.info("\n[Step 2/4] Syncing IMF DataMapper Projections...")
    try:
        imf = ImfScraper()
        res2 = imf.sync_imf_indicators()
        logger.info(f" IMF Indicators sync: {'Success' if res2 else 'Failed'}")
    except Exception as e:
        logger.error(f" Error syncing IMF indicators: {e}")
        success = False

    # 3. Pakistan Bureau of Statistics (PBS) Stats
    logger.info("\n[Step 3/4] Syncing PBS Inflation, LSM & Trade Balance...")
    try:
        res3 = fetch_pbs_stats()
        logger.info(f" PBS Statistics sync: {'Success' if res3 else 'Failed'}")
    except Exception as e:
        logger.error(f" Error syncing PBS statistics: {e}")
        success = False

    # 4. Global Commodities & International Benchmarks
    logger.info("\n[Step 4/4] Syncing Global Commodities & Market Indices...")
    try:
        ms = MacroScraper()
        res4 = ms.sync_macro()
        logger.info(f" Global Macro & Commodities sync: {'Success' if res4 else 'Failed'}")
    except Exception as e:
        logger.error(f" Error syncing Global macro & commodities: {e}")
        success = False

    logger.info("=========================================")
    logger.info("MACROECONOMIC & CENTRAL BANK PIPELINE COMPLETE")
    logger.info("=========================================")
    return success

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    run_macro_pipeline()
