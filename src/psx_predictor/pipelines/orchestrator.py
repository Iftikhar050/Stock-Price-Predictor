import os
import sys
import logging
from typing import List, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

from src.psx_predictor.pipelines.macro_pipeline import run_macro_pipeline
from src.psx_predictor.pipelines.institutional_flows_pipeline import run_institutional_flows_pipeline
from src.psx_predictor.pipelines.fundamentals_pipeline import run_fundamentals_pipeline
from src.psx_predictor.pipelines.news_sentiment_pipeline import run_news_sentiment_pipeline
from src.psx_predictor.data.build_features import build_features

logger = logging.getLogger("PipelineOrchestrator")
logger.setLevel(logging.INFO)

def run_full_data_pipeline(tickers: Optional[List[str]] = None) -> bool:
    """
    Executes the Master Domain Data Ingestion & Feature Engineering Pipeline:
    1. Macroeconomic & Central Bank Pipeline (SBP, IMF, PBS, Commodities)
    2. Institutional Flows & Market Structure Pipeline (NCCPL, SECP, Indices)
    3. Sector & Company Fundamentals Pipeline (EOD, Financial Statements, Sector Metrics)
    4. News & Sentiment Pipeline (Alpha Vantage, PUCARS, NLP Sentiment, Raw Text Datasets)
    5. Re-builds finalized ML feature matrices & master CSV datasets (PSO_master.csv, MEBL_master.csv).
    """
    if tickers is None:
        tickers = ["PSO", "MEBL"]
        
    logger.info("=========================================")
    logger.info("COMMENCING MASTER DOMAIN DATA INGESTION PIPELINE")
    logger.info("=========================================")
    
    # 1. Macro & Central Bank Pipeline
    logger.info("\n--- Phase 1/5: Macroeconomic & Central Bank Pipeline ---")
    run_macro_pipeline()
    
    # 2. Institutional Flows & Market Structure Pipeline
    logger.info("\n--- Phase 2/5: Institutional Flows & Market Structure Pipeline ---")
    run_institutional_flows_pipeline(tickers)
    
    # 3. Sector & Fundamentals Pipeline
    logger.info("\n--- Phase 3/5: Sector & Company Fundamentals Pipeline ---")
    run_fundamentals_pipeline(tickers)
    
    # 4. News & Sentiment Pipeline
    logger.info("\n--- Phase 4/5: News & Sentiment Pipeline ---")
    run_news_sentiment_pipeline(tickers)

    # 4.5 Supplementary Data Sources (PSX Market Stats, Google Trends)
    logger.info("\n--- Phase 4.5/5: PSX Market Stats & Search Trends ---")
    try:
        from src.psx_predictor.data.fetch_psx_market_stats import sync_psx_market_stats
        sync_psx_market_stats(lookback_days=7)
        logger.info(" PSX Market Stats sync completed.")
    except Exception as e:
        logger.warning(f" PSX Market Stats sync skipped: {e}")

    try:
        from src.psx_predictor.data.fetch_search_trends import sync_search_trends_to_db
        for ticker in tickers:
            sync_search_trends_to_db(ticker)
        logger.info(" Google Trends search sentiment sync completed.")
    except Exception as e:
        logger.warning(f" Google Trends sync skipped (pytrends may not be installed): {e}")

    # 4.6 Phase 2 — Additional PDF Framework Features
    logger.info("\n--- Phase 4.6/5: Phase 2 — SBP Additional, Pakistan Activity, Political Flags ---")

    # Tier-2: SBP EasyData additional monetary/BOP series
    try:
        from src.psx_predictor.data.fetch_sbp_additional import fetch_sbp_additional
        fetch_sbp_additional()
        logger.info(" SBP additional series (REER, OMO, credit growth, T-bill cutoffs) completed.")
    except Exception as e:
        logger.warning(f" SBP additional series skipped: {e}")

    # Tier-3: Pakistan real economy activity
    try:
        from src.psx_predictor.data.fetch_pakistan_activity import fetch_pakistan_activity
        fetch_pakistan_activity()
        logger.info(" Pakistan activity data (cement, auto, electricity, wheat) completed.")
    except Exception as e:
        logger.warning(f" Pakistan activity data skipped: {e}")

    # Tier-4: Political & geopolitical event flags (computed — no external API)
    try:
        from src.psx_predictor.data.feature_political_events import build_political_features
        build_political_features()
        logger.info(" Political & geopolitical event flags generated.")
    except Exception as e:
        logger.warning(f" Political event flags skipped: {e}")

    # 5. Master Feature & Dataset Generation

    logger.info("\n--- Phase 5/5: Feature Matrix & Master CSV Dataset Generation ---")
    for ticker in tickers:
        try:
            df = build_features(ticker)
            logger.info(f" Successfully generated master feature set for {ticker} (Shape: {df.shape})")
        except Exception as e:
            logger.error(f" Error building features for {ticker}: {e}")

    logger.info("=========================================")
    logger.info("MASTER DOMAIN DATA INGESTION PIPELINE COMPLETE")
    logger.info("=========================================")
    return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    run_full_data_pipeline()
