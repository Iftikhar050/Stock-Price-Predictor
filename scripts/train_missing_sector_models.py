"""
Train sector-specific "specialized" (SPX) models for the 22 actively-maintained
KSE-100 tickers that fell outside every bucket in the existing SECTOR_MAPPING
(predict.py) / SECTOR_DICTIONARY (train_all_sector_models.py) - these tickers
never had any specialized/SPX prediction available, regardless of data
freshness. Their real PSX sectors (from data/kse100_constituents.csv) are
mostly singleton or 2-3-ticker sectors, so this reuses the exact same
pooled-training function as train_all_sector_models.py (one-hot ticker
encoding degenerates harmlessly to a constant column for singleton sectors).
"""
import os
import sys
import json
import logging

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from scripts.train_all_sector_models import train_sector_model
from src.psx_predictor.models.utils import choose_global_cutoff
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MissingSectorModelTrainer")

NEW_SECTOR_DICTIONARY = {
    "cable_electrical": ["PAEL"],
    "chemical": ["LOTCHEM", "LCI"],
    "closed_end_funds": ["HGFA"],
    "engineering": ["INIL", "ISL"],
    "glass_ceramics": ["GHGL", "TGL"],
    "insurance": ["AICL"],
    "leasing": ["PGLC"],
    "leather_tanneries": ["SRVI"],
    "misc_other": ["PABC", "PSEL", "SHFA"],
    "modarabas": ["FHAM"],
    "paper_packaging": ["PKGS"],
    "property": ["JVDC"],
    "reit": ["DCR"],
    "sugar": ["JDWS"],
    "tobacco": ["PAKT"],
    "transport": ["PIBTL"],
    "woollen": ["BNWM"],
}

REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def main():
    logger.info("===============================================================")
    logger.info("   MISSING-COVERAGE PSX SECTOR MODEL TRAINING (22 tickers)    ")
    logger.info("===============================================================")

    cutoff_str, _ = choose_global_cutoff(test_trading_days=250, min_train_trading_days=500)
    cutoff_date = pd.to_datetime(cutoff_str)
    logger.info(f"Global Cutoff Date: {cutoff_str}")

    results = {}
    for sector_key, tickers in NEW_SECTOR_DICTIONARY.items():
        res = train_sector_model(sector_key, tickers, cutoff_date)
        if res:
            results[sector_key] = res

    report_path = os.path.join(REPORTS_DIR, "missing_sectors_model_evaluation.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Successfully trained {len(results)} new sector models! Summary saved to {report_path}")


if __name__ == "__main__":
    main()
