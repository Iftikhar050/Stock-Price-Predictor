import os
import sys
import logging

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

from src.psx_predictor.scraper.sbp_easydata_scraper import SbpEasyDataScraper
from src.psx_predictor.scraper.imf_scraper import ImfScraper
from src.psx_predictor.scraper.macro_scraper import MacroScraper
from src.psx_predictor.scraper.psx_dps_index_scraper import PsxDpsIndexScraper

logger = logging.getLogger("MacroPipeline")
logger.setLevel(logging.INFO)

def run_macro_pipeline() -> bool:
    """
    Executes the Macroeconomic & Central Bank Pipeline:
    1. Ingests State Bank of Pakistan (SBP) Monetary & Interest Rate Series
       (this is also the real source for cpi_headline/cpi_core - SBP EasyData
       republishes PBS's CPI release).
    2. Ingests IMF DataMapper REST API Macro Projections.
    3. Ingests Global Commodity Futures & International Equity/Rate Benchmarks.
    4. Ingests PSX's own KMI-30 / KSE-30 / All-Share / sector indices.

    NOTE ON STEP 4: kmi30_index_level / kse30_index_level / all_share_index_level
    were confirmed genuinely live in the database (populated through the
    current date) despite this scraper never having been called from any
    pipeline or orchestrator prior to this fix - it was apparently being run
    by hand out-of-band. Wiring it in here removes that dependency on
    undocumented tribal knowledge.

    NOTE: A direct Pakistan Bureau of Statistics (PBS) step used to run here
    (fetch_pbs_stats). It had no real, free, scrapable source behind it - its
    CPI/WPI/trade figures were hardcoded literals through a future 2026 date
    - and because it ran after the real SBP step above with an ON CONFLICT DO
    UPDATE on the same (date) key, it was silently overwriting the real
    SBP-sourced cpi_headline/cpi_core on every pipeline run. It has been
    removed rather than fixed in place; see fetch_pbs_stats.py's module
    docstring before reintroducing anything that writes cpi_headline/
    cpi_core/cpi_food.
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

    # 3. Global Commodities & International Benchmarks
    logger.info("\n[Step 3/4] Syncing Global Commodities & Market Indices...")
    try:
        ms = MacroScraper()
        res3 = ms.sync_macro()
        logger.info(f" Global Macro & Commodities sync: {'Success' if res3 else 'Failed'}")
    except Exception as e:
        logger.error(f" Error syncing Global macro & commodities: {e}")
        success = False

    # 4. PSX Official Indices (KMI-30, KSE-30, All-Share, sector indices)
    logger.info("\n[Step 4/4] Syncing PSX Official Indices...")
    try:
        idx_scraper = PsxDpsIndexScraper()
        res4 = idx_scraper.sync_all_indices()
        logger.info(f" PSX Index sync: {'Success' if res4 else 'Failed'}")
    except Exception as e:
        logger.error(f" Error syncing PSX indices: {e}")
        success = False

    logger.info("=========================================")
    logger.info("MACROECONOMIC & CENTRAL BANK PIPELINE COMPLETE")
    logger.info("=========================================")
    return success

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    run_macro_pipeline()
