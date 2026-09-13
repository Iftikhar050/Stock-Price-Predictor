"""Scrapes PSX's own official symbol list (dps.psx.com.pk/symbols) to build the full PSX
equity ticker universe, needed to define Phase 2+ of the ticker expansion (Phase 1 is just
the existing KSE-100 constituents list). Mirrors scripts/fetch_kse100.py's output shape.

The endpoint returns every listed instrument (equities, bonds/TFCs, ETFs). We keep ordinary
equities only (isETF=False, isDebt=False) - bonds/TFCs aren't stocks and ETFs aren't single
companies, so neither fits this project's per-ticker feature-engineering pipeline.

Usage: python scripts/fetch_psx_listed_companies.py
Output: data/psx_listed_companies.csv (ticker, company_name, sector, fetched_at)
"""
import os

import pandas as pd
import requests

URL = "https://dps.psx.com.pk/symbols"
OUT_PATH = os.path.join("data", "psx_listed_companies.csv")


def fetch_psx_listed_companies():
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(URL, headers=headers, timeout=20)
    res.raise_for_status()
    symbols = res.json()

    equities = [s for s in symbols if not s.get("isETF") and not s.get("isDebt")]

    data = [
        {
            "ticker": s["symbol"],
            "company_name": s.get("name", ""),
            "sector": s.get("sectorName", ""),
            "fetched_at": pd.Timestamp.now().strftime("%Y-%m-%d"),
        }
        for s in equities
    ]

    df = pd.DataFrame(data).sort_values("ticker").reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Fetched {len(symbols)} total symbols ({len(symbols) - len(equities)} bonds/ETFs excluded).")
    print(f"Saved {len(df)} equity tickers to {OUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    fetch_psx_listed_companies()
