import os
import sys
import logging
from typing import List, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

from src.psx_predictor.news.alpha_vantage_fetcher import AlphaVantageFetcher
from src.psx_predictor.news.aggregator import NewsAggregator
from src.psx_predictor.data.fetch_psx_pucars import fetch_pucars_announcements
from src.psx_predictor.data.export_raw_text_datasets import export_raw_text_files_for_ticker

logger = logging.getLogger("NewsSentimentPipeline")
logger.setLevel(logging.INFO)

def run_news_sentiment_pipeline(tickers: Optional[List[str]] = None, limit: int = 50, use_finbert: bool = False) -> bool:
    """
    Executes the Master News & Sentiment Intelligence Pipeline:
    1. Ingests Alpha Vantage news sentiment feed with AI-derived sentiment scores.
    2. Runs NewsAggregator (Google News, Local Pakistani RSS, Topic Classifier, FinBERT/VADER sentiment).
    3. Ingests PSX PUCARS corporate notices and structured corporate events for target tickers.
    4. Exports standalone raw text announcement & news CSV datasets.
    """
    if tickers is None:
        tickers = ["PSO", "MEBL"]
        
    logger.info("=========================================")
    logger.info("STARTING MASTER NEWS & SENTIMENT PIPELINE")
    logger.info("=========================================")
    
    success = True
    
    # 1. Alpha Vantage Market & Macro News Sentiment
    logger.info("\n[Step 1/4] Syncing Alpha Vantage News & Sentiment Feed...")
    try:
        av = AlphaVantageFetcher()
        av.sync_all()
        logger.info(" Alpha Vantage news & sentiment sync completed.")
    except Exception as e:
        logger.error(f" Error in Alpha Vantage news fetcher: {e}")
        success = False

    # 2. Multi-Source News Aggregator & Topic Sentiment Engine
    logger.info("\n[Step 2/4] Running Multi-Source News Aggregator & Topic Sentiment Engine...")
    try:
        aggregator = NewsAggregator(use_finbert=use_finbert)
        aggregator.run_pipeline()
        logger.info(" Multi-source news aggregator completed.")
    except Exception as e:
        logger.error(f" Error in NewsAggregator pipeline: {e}")
        success = False
        
    # 3. PSX PUCARS Raw Corporate Announcements & Structured Events
    logger.info("\n[Step 3/4] Syncing PSX PUCARS Corporate Announcements & Events...")
    for ticker in tickers:
        try:
            logger.info(f" Ingesting PUCARS notices for {ticker}...")
            res = fetch_pucars_announcements(ticker)
            logger.info(f" PUCARS sync for {ticker}: {'Success' if res else 'Failed'}")
        except Exception as e:
            logger.error(f" Error ingesting PUCARS for {ticker}: {e}")
            success = False
            
    # 4. Export Raw Text Datasets
    logger.info("\n[Step 4/4] Exporting Date-Matched Raw Text Datasets...")
    for ticker in tickers:
        try:
            export_raw_text_files_for_ticker(ticker)
            logger.info(f" Exported raw text datasets for {ticker}.")
        except Exception as e:
            logger.error(f" Error exporting raw text datasets for {ticker}: {e}")
            success = False

    logger.info("=========================================")
    logger.info("NEWS & SENTIMENT PIPELINE COMPLETE")
    logger.info("=========================================")
    return success

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    run_news_sentiment_pipeline()
