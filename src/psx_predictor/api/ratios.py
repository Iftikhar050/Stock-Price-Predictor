"""
ratios.py — Reads the already-computed valuation ratios out of each ticker's
data/processed/{TICKER}_master.csv rather than recomputing them from raw DB
tables. The feature pipeline (build_features.py) is the single source of
truth for these derived columns (pe_ratio, pb_ratio, market_cap, etc.) —
duplicating that math in SQL would risk drifting from it.
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import pandas as pd

from src.psx_predictor.api.state import psx_cache
from src.psx_predictor.api.cache import get_cached, set_cached

PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

RATIO_COLUMNS = [
    "date", "close", "market_cap", "pe_ratio", "pb_ratio",
    "dividend_yield", "roe", "eps_trailing", "sector_pe_avg", "sector_pb_avg",
    "shares_outstanding", "free_float", "free_float_pct",
    "insider_buy_shares_30d", "insider_sell_shares_30d", "insider_net_flow_30d",
    # Derived valuation ratios/percentiles - genuinely computed from real
    # price + fundamentals data by build_features.py, just never allowlisted
    # here before (confirmed via the dead-column audit that none of these
    # are on the universally-dead/zero-variance list).
    "pe_percentile_1y", "pe_percentile_3y", "pb_percentile_3y", "dividend_yield_percentile_3y",
    "pe_1y_avg", "pe_3y_avg", "pe_5y_avg", "forward_pe",
    "price_to_cash_flow", "ev", "ev_ebitda", "ev_sales",
    "profit_margin", "roa", "peg_ratio", "book_value_per_share", "debt_to_equity",
]

# Reading the last row of a ~400-column master.csv with pandas costs ~0.3s
# per ticker - fine for one company page, but the screener/sectors endpoints
# call this for every active ticker (~107x per computation), and multiple
# pages hit those endpoints concurrently on a cold cache (Home + CuratedLists
# alone fire 4 independent /api/screener requests). Without this cache that
# was measured at ~85s wall time for a cold homepage load. Ratios only change
# when the feature pipeline reruns (at most a few times a day), so a 5-minute
# TTL is generous, not stale.
RATIOS_CACHE_TTL = 300

# build_features.py's end-of-pipeline "Handle Missing Values" step zero-fills
# every remaining numeric NaN across the whole master CSV (needed so ML
# training never sees NaN) - including these columns, which erases the
# earlier NaN this file's own ratio-calculation logic deliberately left for
# tickers with no shares-outstanding/fundamentals coverage (5 tickers as of
# 2026-09-08: MEHT, PGLC, POWER, TGL, YOUW). A literal 0 is definitionally
# impossible for any of these on a real, actively-traded stock (price is
# never $0, so market cap/P-E/P-B/shares outstanding can't be exactly zero
# either) - unlike dividend_yield or roe, where a genuine 0 is valid data and
# must NOT be touched here. Treat 0 as the "missing" sentinel it actually is.
ZERO_MEANS_MISSING = {"market_cap", "pe_ratio", "pb_ratio", "shares_outstanding"}


def get_latest_ratios(ticker: str) -> dict | None:
    """Last row of {ticker}_master.csv, limited to the ratio columns above."""
    ticker = ticker.upper()
    cache_key = f"ratios_{ticker}"
    cached = get_cached(psx_cache, cache_key, RATIOS_CACHE_TTL)
    if cached is not None:
        return cached

    path = os.path.join(PROCESSED_DIR, f"{ticker}_master.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, usecols=lambda c: c in RATIO_COLUMNS)
    except Exception:
        return None
    if df.empty:
        return None
    row = df.iloc[-1]
    result = {}
    for col in RATIO_COLUMNS:
        if col not in row:
            result[col] = None
            continue
        val = row[col]
        if pd.isna(val) or (col in ZERO_MEANS_MISSING and float(val) == 0.0):
            result[col] = None
        elif col == "date":
            result[col] = str(val)
        else:
            result[col] = float(val)

    set_cached(psx_cache, cache_key, result)
    return result
