import logging
import sys
import os
from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.psx_predictor.db.models import StockFundamentals, Base
from src.psx_predictor.db.connection import engine
from src.psx_predictor.scraper.fundamentals_scraper import FundamentalsScraper
from src.psx_predictor.db.repository import get_active_tickers
from src.psx_predictor.data.build_features import build_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_update():
    logger.info("Dropping stock_fundamentals table to apply new schema...")
    StockFundamentals.__table__.drop(engine, checkfirst=True)
    
    logger.info("Recreating table...")
    Base.metadata.create_all(engine)
    
    tickers = get_active_tickers()
    scraper = FundamentalsScraper()
    
    logger.info(f"Re-scraping fundamentals for {len(tickers)} tickers. This may take a moment...")
    success_count = 0
    for t in tickers:
        success = scraper.sync_fundamentals(t)
        if success:
            success_count += 1
            
    logger.info(f"Successfully scraped fundamentals for {success_count} out of {len(tickers)} tickers.")
    
    logger.info("Rebuilding feature CSVs...")
    for t in tickers:
        build_features(t)
        
    logger.info("Update complete!")

if __name__ == "__main__":
    run_update()
