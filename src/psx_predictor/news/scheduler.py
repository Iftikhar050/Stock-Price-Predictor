import time
import schedule
import logging
from .aggregator import NewsAggregator

logger = logging.getLogger(__name__)

def job():
    logger.info("Running scheduled News Intelligence Pipeline...")
    try:
        aggregator = NewsAggregator()
        aggregator.run_pipeline()
    except Exception as e:
        logger.error(f"Scheduled pipeline failed: {e}")

def start_scheduler():
    # Run every day at 16:15 PKT (11:15 UTC), just after market closes 
    # to capture all news for the current trading day.
    schedule.every().day.at("11:15").do(job)
    
    logger.info("Scheduler started. Waiting for next run...")
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    start_scheduler()
