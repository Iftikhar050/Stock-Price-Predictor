"""
technicals.py — Reads the already-computed technical indicators out of each
ticker's data/processed/{TICKER}_master.csv, mirroring ratios.py's pattern:
build_features.py is the single source of truth for these derived columns
(sma_*, rsi_14, macd, bollinger_*, adx, etc.) - this module just reads the
last row rather than recomputing anything. These were computed for every
one of the 107 tickers already but never exposed through any endpoint before
this - confirmed via the dead-column audit (data/dead_columns_report.md)
that none of them are on the universally-dead/zero-variance list.
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

TECHNICAL_COLUMNS = [
    "date",
    "sma_7", "sma_21", "sma_50", "sma_7_dist", "sma_21_dist", "sma_50_dist",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bollinger_mavg", "bollinger_hband", "bollinger_lband", "bollinger_width",
    "vwap", "vwap_14_dist", "obv", "atr", "adx", "plus_di", "minus_di",
    "stochastic_k", "stochastic_d", "williams_r", "cci_14",
    "beta_60d", "beta_252d", "historical_volatility_20d",
]

# Matches ratios.py's TTL - both only change when the feature pipeline reruns
# (at most a few times a day), so 5 minutes is generous, not stale.
TECHNICALS_CACHE_TTL = 300


def get_latest_technicals(ticker: str) -> dict | None:
    """Last row of {ticker}_master.csv, limited to the technical-indicator columns above."""
    ticker = ticker.upper()
    cache_key = f"technicals_{ticker}"
    cached = get_cached(psx_cache, cache_key, TECHNICALS_CACHE_TTL)
    if cached is not None:
        return cached

    path = os.path.join(PROCESSED_DIR, f"{ticker}_master.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, usecols=lambda c: c in TECHNICAL_COLUMNS)
    except Exception:
        return None
    if df.empty:
        return None
    row = df.iloc[-1]
    result = {}
    for col in TECHNICAL_COLUMNS:
        if col not in row:
            result[col] = None
            continue
        val = row[col]
        if pd.isna(val):
            result[col] = None
        elif col == "date":
            result[col] = str(val)
        else:
            result[col] = float(val)

    set_cached(psx_cache, cache_key, result)
    return result
