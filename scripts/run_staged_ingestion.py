"""Runs the full data pipeline for one phase of the PSX universe expansion, writing master
CSVs to data/staging/phase_N/ instead of data/processed/ - so unreviewed/unpromoted tickers
never mix into the verified dataset (currently just PSO, MEBL).

Only build_features.py's PROCESSED_DIR write path needs redirecting (see plan: PSX Universe
Expansion - Phased Staged Ingestion) - the macro/institutional-flows/fundamentals/news
sub-pipelines write to shared DB tables scoped by ticker, which is safe for new tickers.
Deliberately does NOT touch stock_metadata.is_active - new tickers stay out of the DB-driven
active set until a human reviews a phase's quality report and explicitly promotes it.

Usage:
  python scripts/run_staged_ingestion.py phase_1
  python scripts/run_staged_ingestion.py phase_1 --tickers HBL,LUCK   (dry run on a subset)
"""
import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

MANIFEST_PATH = os.path.join(ROOT_DIR, "data", "phase_manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", help="Phase key from data/phase_manifest.json, e.g. phase_1")
    parser.add_argument("--tickers", help="Comma-separated subset override, for a dry run")
    args = parser.parse_args()

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    if args.phase not in manifest:
        print(f"Unknown phase '{args.phase}'. Available: {list(manifest.keys())}")
        sys.exit(1)

    tickers = args.tickers.split(",") if args.tickers else manifest[args.phase]

    staging_dir = os.path.join(ROOT_DIR, "data", "staging", args.phase)
    os.makedirs(staging_dir, exist_ok=True)

    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # build_features.py and export_raw_text_datasets.py each define their own module-level
    # PROCESSED_DIR global (no shared config) - both write ticker output and must be
    # redirected, or raw-text exports leak into data/processed/ even when the master CSV
    # itself correctly lands in staging (confirmed by a dry run before this fix).
    from src.psx_predictor.data import build_features as build_features_module
    from src.psx_predictor.data import export_raw_text_datasets as export_raw_text_module
    build_features_module.PROCESSED_DIR = staging_dir
    export_raw_text_module.PROCESSED_DIR = staging_dir

    from src.psx_predictor.pipelines.orchestrator import run_full_data_pipeline

    print(f"=== Staged ingestion: {args.phase} ({len(tickers)} tickers) -> {staging_dir} ===")
    print(f"Tickers: {tickers}")

    run_full_data_pipeline(tickers)

    print(f"=== {args.phase} complete. Output in {staging_dir} ===")


if __name__ == "__main__":
    main()
