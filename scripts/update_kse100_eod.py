"""
Syncs the latest EOD data (including Monday, Oct 5) for all 97 KSE-100 tickers,
re-generates their master feature datasets, promotes them to data/processed/,
and reloads API models.
"""
import os
import sys
import json
import shutil
import glob
import requests
import logging

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.psx_predictor.scraper.client import PSXScraper
from src.psx_predictor.scraper.dividend_scraper import DividendScraper
from src.psx_predictor.scraper.psx_dps_index_scraper import PsxDpsIndexScraper
from src.psx_predictor.data.build_features import build_features
from src.psx_predictor.data.export_raw_text_datasets import export_raw_text_files_for_ticker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UpdateKSE100EOD")

MANIFEST_PATH = os.path.join(ROOT_DIR, "data", "phase_manifest.json")

def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    tickers = manifest.get("phase_1", [])
    # Extra tickers on the command line (e.g. PSO MEBL, which aren't in phase_1).
    tickers += [t.upper() for t in sys.argv[1:] if t.upper() not in tickers]
    logger.info(f"Syncing EOD data & rebuilding features for {len(tickers)} KSE-100 tickers...")
    
    scraper = PSXScraper()
    div_scraper = DividendScraper()
    staging_dir = os.path.join(ROOT_DIR, "data", "staging", "phase_1")
    processed_dir = os.path.join(ROOT_DIR, "data", "processed")

    from src.psx_predictor.data import build_features as build_features_module
    from src.psx_predictor.data import export_raw_text_datasets as export_raw_text_module
    build_features_module.PROCESSED_DIR = staging_dir
    export_raw_text_module.PROCESSED_DIR = staging_dir

    # PSX only publishes a live index snapshot (no history), so today's
    # KSE-100/KMI-30/sector index levels are stored only if this runs today -
    # and before the feature rebuild, which reads them. Run after the 3:30 PM
    # close, or the stored levels are intraday.
    try:
        if not PsxDpsIndexScraper().sync_all_indices():
            logger.error("PSX index sync failed - index cards and index features stay on the last synced day.")
    except Exception as e:
        logger.error(f"Error syncing PSX indices: {e}")

    for i, ticker in enumerate(tickers, 1):
        try:
            logger.info(f"[{i}/{len(tickers)}] Syncing EOD for {ticker}...")
            scraper.sync_ticker(ticker)
            div_scraper.sync_dividends(ticker)
            build_features(ticker)
            export_raw_text_files_for_ticker(ticker)
        except Exception as e:
            logger.error(f"Error updating {ticker}: {e}")

    copied = 0
    for f in glob.glob(os.path.join(staging_dir, "*.*")):
        shutil.copy2(f, processed_dir)
        copied += 1
    logger.info(f"Promoted {copied} files to data/processed!")

    try:
        r = requests.post(f"http://localhost:{os.getenv('API_PORT', '8010')}/api/reload_models", headers={'X-API-Key': 'local-dev-key'})
        logger.info(f"Reload API status: {r.status_code} {r.json()}")
    except Exception as e:
        logger.error(f"Error reloading API: {e}")

if __name__ == "__main__":
    main()
