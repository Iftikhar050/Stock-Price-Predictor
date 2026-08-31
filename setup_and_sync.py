import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.psx_predictor.db.connection import engine
from src.psx_predictor.db.models import Base
from src.psx_predictor.db.repository import get_active_tickers
from src.psx_predictor.pipelines.orchestrator import run_full_data_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SetupAndSync")

def main():
    logger.info("1. Verifying & Creating Database Tables...")
    try:
        Base.metadata.create_all(engine)
        logger.info(" Success: Database tables verified/created.")
    except Exception as e:
        logger.error(f" ERROR connecting to database. Check credentials in .env! Error: {e}")
        sys.exit(1)

    active_tickers = get_active_tickers()
    if not active_tickers:
        active_tickers = ["PSO", "MEBL"]
        
    logger.info(f"2. Loaded {len(active_tickers)} active target tickers: {active_tickers}")
    logger.info("3. Executing Master Domain Data Ingestion Pipelines...")
    
    success = run_full_data_pipeline(active_tickers)
    
    if success:
        logger.info("=======================================================")
        logger.info("   DATA SYNC & PIPELINE EXECUTION COMPLETE             ")
        logger.info("=======================================================")
    else:
        logger.warning("Pipeline executed with warnings. Check logs above.")

if __name__ == '__main__':
    main()
