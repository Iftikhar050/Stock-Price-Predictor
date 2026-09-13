"""Builds data/phase_manifest.json: the ticker-to-phase assignment for the staged PSX
universe expansion (see plan: PSX Universe Expansion - Phased Staged Ingestion).

Phase 1 = KSE-100 constituents (data/kse100_constituents.csv) - supersedes the 103 stale
on-disk data/processed/*_master.csv files (all confirmed stale by
scripts/audit_master_csv_freshness.py), plus PSO/MEBL which are already done and excluded
here since they don't need re-ingestion.

Phase 2+ = the remaining PSX equity universe (data/psx_listed_companies.csv, from
scripts/fetch_psx_listed_companies.py), in fixed batches of 100, in the order PSX's own
symbol endpoint returns them (alphabetical), excluding anything already in Phase 1.

Usage: python scripts/build_phase_manifest.py
Output: data/phase_manifest.json
"""
import json
import os

import pandas as pd

BATCH_SIZE = 100
DONE_TICKERS = {"PSO", "MEBL"}
OUT_PATH = os.path.join("data", "phase_manifest.json")


def build_phase_manifest():
    kse100 = pd.read_csv("data/kse100_constituents.csv")["ticker"].tolist()
    full_universe = pd.read_csv("data/psx_listed_companies.csv")["ticker"].tolist()

    phase_1 = sorted(set(kse100) - DONE_TICKERS)

    already_assigned = set(phase_1) | DONE_TICKERS
    remaining = [t for t in full_universe if t not in already_assigned]

    manifest = {"phase_1": phase_1}
    for i in range(0, len(remaining), BATCH_SIZE):
        phase_num = 2 + (i // BATCH_SIZE)
        manifest[f"phase_{phase_num}"] = remaining[i:i + BATCH_SIZE]

    with open(OUT_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Built {len(manifest)} phases covering {sum(len(v) for v in manifest.values())} tickers:")
    for phase, tickers in manifest.items():
        print(f"  {phase}: {len(tickers)} tickers")
    print(f"\nAlready done (excluded): {sorted(DONE_TICKERS)}")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    build_phase_manifest()
