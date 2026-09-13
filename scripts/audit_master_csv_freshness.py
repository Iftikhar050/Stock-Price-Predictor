"""Audits the 105 pre-existing data/processed/{TICKER}_master.csv files (everything except
PSO/MEBL, which are the only two known to be post-remediation) for staleness.

None of these 105 files are tracked in git (confirmed via `git log -- data/processed/<f>` for
a sample), meaning they predate the fabrication-fix commits (5a8327a, 4aff73e) that only ever
touched PSO_master.csv/MEBL_master.csv. This script confirms that cheaply: a current-schema
master CSV (per build_features.py's DEAD_COLUMNS drop) must contain none of the columns that
build_features.py already knows are dead/fabricated/no-signal. Any ticker whose CSV still has
one or more of those columns is stale and needs regenerating.

Usage: python scripts/audit_master_csv_freshness.py
Output: data/staging/audit_report.csv (ticker, status, stale_columns, missing_from_reference, n_cols)
"""
import os
import sys

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.psx_predictor.data.build_features import DEAD_COLUMNS

PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
STAGING_DIR = os.path.join(ROOT_DIR, "data", "staging")
REFERENCE_TICKER = "PSO"  # known-good, post-remediation schema
OUT_PATH = os.path.join(STAGING_DIR, "audit_report.csv")

DEAD_COLUMNS_SET = set(DEAD_COLUMNS)


def reference_columns() -> set:
    ref_path = os.path.join(PROCESSED_DIR, f"{REFERENCE_TICKER}_master.csv")
    cols = set(pd.read_csv(ref_path, nrows=0).columns.tolist())
    non_ticker_specific = cols - {"ticker"}
    return non_ticker_specific


def audit_ticker(ticker: str, ref_cols: set) -> dict:
    path = os.path.join(PROCESSED_DIR, f"{ticker}_master.csv")
    try:
        cols = set(pd.read_csv(path, nrows=0).columns.tolist())
    except Exception as e:
        return {"ticker": ticker, "status": "ERROR", "stale_columns": str(e),
                "missing_from_reference": "", "n_cols": 0}

    stale = sorted(cols & DEAD_COLUMNS_SET)
    missing = sorted(ref_cols - cols - {"ticker"})

    if stale:
        status = "STALE"
    elif missing:
        status = "SCHEMA_DRIFT"
    else:
        status = "OK"

    return {
        "ticker": ticker,
        "status": status,
        "stale_columns": ";".join(stale),
        "missing_from_reference": ";".join(missing),
        "n_cols": len(cols),
    }


def main():
    os.makedirs(STAGING_DIR, exist_ok=True)
    ref_cols = reference_columns()

    tickers = sorted(
        f[:-len("_master.csv")]
        for f in os.listdir(PROCESSED_DIR)
        if f.endswith("_master.csv") and f[:-len("_master.csv")] not in ("PSO", "MEBL")
    )

    print(f"Auditing {len(tickers)} existing master CSVs against reference schema "
          f"({REFERENCE_TICKER}, {len(ref_cols)} cols, {len(DEAD_COLUMNS_SET)} known-dead columns)...")

    rows = [audit_ticker(t, ref_cols) for t in tickers]
    report = pd.DataFrame(rows)
    report.to_csv(OUT_PATH, index=False)

    counts = report["status"].value_counts()
    print("\nResult:")
    for status, n in counts.items():
        print(f"  {status}: {n}")
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
