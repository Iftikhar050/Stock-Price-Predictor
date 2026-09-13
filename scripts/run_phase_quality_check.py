"""Runs the existing aggregate quality gate (quality_gate.run_quality_checks_all_tickers)
against a staged phase's output instead of data/processed/, and writes the summary as a
review artifact into that same staging folder - the gate before anyone promotes a phase.

Usage: python scripts/run_phase_quality_check.py phase_1
Output: data/staging/phase_1/quality_summary.json
"""
import argparse
import json
import logging
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", help="Phase key, e.g. phase_1 (must match a data/staging/<phase> folder)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    staging_dir = os.path.join(ROOT_DIR, "data", "staging", args.phase)
    if not os.path.isdir(staging_dir):
        print(f"No such staging folder: {staging_dir}")
        sys.exit(1)

    from src.psx_predictor.data import quality_gate
    quality_gate.PROCESSED_DIR = staging_dir

    summary = quality_gate.run_quality_checks_all_tickers()

    out_path = os.path.join(staging_dir, "quality_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved quality summary to {out_path}")


if __name__ == "__main__":
    main()
