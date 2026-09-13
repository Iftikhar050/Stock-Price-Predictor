"""
macro.py — Forex, commodities, and economic-indicator endpoints, backed by
macro_indicators. That table was already populated for ML feature engineering
but never exposed to the frontend until now. Also a market-wide announcements
feed from corporate_announcements_pucars (previously only exposed per-ticker
via /api/company/{ticker}/events).

Several macro_indicators columns are stale or effectively dead (e.g.
total_fx_reserves' last real value is from 2024, coal_price/palm_oil_price
have no recent real data at all, and several float columns store NaN instead
of SQL NULL - see json_safe.py). Rather than hand-maintain an exclusion list
that will silently drift out of date, each field is checked for a real
(non-NaN) value inside a freshness window and dropped from the response
entirely if none exists, instead of showing a stale/fabricated "current" value.
"""
import os
import sys
from datetime import timedelta

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from fastapi import APIRouter, Query
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import text

from src.psx_predictor.db.connection import engine
from src.psx_predictor.api.state import psx_cache
from src.psx_predictor.api.cache import get_cached, set_cached
from src.psx_predictor.api.json_safe import safe_float

router = APIRouter()

MACRO_CACHE_TTL = 3600  # forex/commodities/economy update at most once a day
ANNOUNCEMENTS_CACHE_TTL = 120

FOREX_FIELDS = [
    {"key": "pkr_usd_rate", "label": "US Dollar", "unit": "PKR"},
    {"key": "eur_pkr_rate", "label": "Euro", "unit": "PKR"},
    {"key": "gbp_pkr_rate", "label": "British Pound", "unit": "PKR"},
    {"key": "cny_pkr_rate", "label": "Chinese Yuan", "unit": "PKR"},
]

COMMODITY_FIELDS = [
    {"key": "gold_price", "label": "Gold", "unit": "USD/oz"},
    {"key": "brent_oil_price", "label": "Brent Crude Oil", "unit": "USD/bbl"},
    {"key": "wti_oil_price", "label": "WTI Crude Oil", "unit": "USD/bbl"},
    {"key": "cotton_price", "label": "Cotton", "unit": "USD/lb"},
    {"key": "copper_price", "label": "Copper", "unit": "USD/lb"},
    {"key": "gas_price", "label": "Natural Gas", "unit": "USD/MMBtu"},
    {"key": "aluminum_price", "label": "Aluminum", "unit": "USD/ton"},
    {"key": "wheat_price", "label": "Wheat", "unit": "USD/bushel"},
    {"key": "steel_price", "label": "Steel", "unit": "USD/ton"},
    {"key": "urea_price", "label": "Urea", "unit": "USD/ton"},
    {"key": "lng_price", "label": "LNG", "unit": "USD/MMBtu"},
    {"key": "coal_price", "label": "Coal", "unit": "USD/ton"},
    {"key": "palm_oil_price", "label": "Palm Oil", "unit": "USD/ton"},
]

ECONOMY_FIELDS = [
    {"key": "cpi_headline", "label": "CPI Inflation (Headline)", "unit": "%"},
    {"key": "cpi_core", "label": "CPI Inflation (Core)", "unit": "%"},
    # cpi_food intentionally omitted: its only writer (fetch_pbs_stats.py) was
    # a fully fabricated source and has been removed. cpi_food also was never
    # declared on the MacroIndicators ORM model (schema drift) - don't
    # reintroduce it here without both a real source and a model column.
    {"key": "sbp_policy_rate", "label": "SBP Policy Rate", "unit": "%"},
    {"key": "kibor_3m", "label": "KIBOR 3-Month", "unit": "%"},
    {"key": "kibor_6m", "label": "KIBOR 6-Month", "unit": "%"},
    {"key": "kibor_1y", "label": "KIBOR 1-Year", "unit": "%"},
    {"key": "sbp_reserves", "label": "SBP FX Reserves", "unit": "USD M"},
    {"key": "total_fx_reserves", "label": "Total FX Reserves", "unit": "USD M"},
    {"key": "monthly_remittances", "label": "Monthly Remittances (Total)", "unit": "USD M"},
    {"key": "remittances_saudi", "label": "Remittances — Saudi Arabia", "unit": "USD M"},
    {"key": "remittances_uae", "label": "Remittances — UAE", "unit": "USD M"},
    {"key": "remittances_usa", "label": "Remittances — USA", "unit": "USD M"},
    {"key": "remittances_uk", "label": "Remittances — UK", "unit": "USD M"},
    {"key": "m2_money_supply", "label": "M2 Money Supply", "unit": "PKR B"},
]

# Global equity indices, the US yield curve, and Pakistan sovereign bond
# yields - all genuinely collected (yfinance / SBP EasyData) for the macro
# feature set but never exposed via any endpoint before. Confirmed live via
# direct DB query before adding here: all 22 fields have real values through
# the current date, just with a later coverage-start than the FX/commodity
# fields above (25-55% of history, not 100% - these series simply weren't
# being collected from day one).
GLOBAL_MARKETS_FIELDS = [
    {"key": "sp500_close", "label": "S&P 500", "unit": "USD"},
    {"key": "nasdaq_close", "label": "Nasdaq Composite", "unit": "USD"},
    {"key": "dow_jones_close", "label": "Dow Jones Industrial Average", "unit": "USD"},
    {"key": "ftse_close", "label": "FTSE 100", "unit": "GBP"},
    {"key": "dax_close", "label": "DAX", "unit": "EUR"},
    {"key": "nikkei_close", "label": "Nikkei 225", "unit": "JPY"},
    {"key": "hang_seng_close", "label": "Hang Seng", "unit": "HKD"},
    {"key": "shanghai_close", "label": "Shanghai Composite", "unit": "CNY"},
    {"key": "msci_em_close", "label": "MSCI Emerging Markets", "unit": "USD"},
    {"key": "msci_fm_close", "label": "MSCI Frontier Markets", "unit": "USD"},
    {"key": "dxy_close", "label": "US Dollar Index (DXY)", "unit": "index"},
    {"key": "vix_close", "label": "VIX Volatility Index", "unit": "index"},
    {"key": "us2y_yield", "label": "US 2-Year Treasury Yield", "unit": "%"},
    {"key": "us5y_yield", "label": "US 5-Year Treasury Yield", "unit": "%"},
    {"key": "us10y_yield", "label": "US 10-Year Treasury Yield", "unit": "%"},
    {"key": "us_yield_curve_2y10y", "label": "US 2s10s Yield Curve Spread", "unit": "%"},
    {"key": "tbill_3m", "label": "Pakistan T-Bill 3-Month Cutoff", "unit": "%"},
    {"key": "tbill_6m", "label": "Pakistan T-Bill 6-Month Cutoff", "unit": "%"},
    {"key": "tbill_1y", "label": "Pakistan T-Bill 1-Year Cutoff", "unit": "%"},
    {"key": "pib_3y", "label": "Pakistan Investment Bond 3-Year", "unit": "%"},
    {"key": "pib_5y", "label": "Pakistan Investment Bond 5-Year", "unit": "%"},
    {"key": "pib_10y", "label": "Pakistan Investment Bond 10-Year", "unit": "%"},
]

# Forex/commodities should update near-daily - a "current" value older than
# this is more likely a broken feed than a genuinely unchanged price.
# Economic indicators (CPI, SBP reserves, remittances) are legitimately
# monthly/quarterly and get a much longer allowance before being dropped.
FRESH_WINDOW_DAILY = 21
FRESH_WINDOW_MACRO = 120
HISTORY_LOOKBACK_DAYS = 730


def _build_group(fields: list[dict], fresh_window_days: int) -> list[dict]:
    keys = [f["key"] for f in fields]
    query = text(f"""
        SELECT date, {", ".join(keys)} FROM macro_indicators
        WHERE date >= (CURRENT_DATE - CAST(:lookback AS INTEGER))
        ORDER BY date ASC
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"lookback": HISTORY_LOOKBACK_DAYS}).mappings().fetchall()

    if not rows:
        return []

    latest_date = rows[-1]["date"]
    cutoff = latest_date - timedelta(days=fresh_window_days)

    out = []
    for f in fields:
        key = f["key"]
        history = []
        last_value = None
        last_date = None
        for r in rows:
            v = safe_float(r[key])
            if v is None:
                continue
            history.append({"date": r["date"].isoformat(), "value": v})
            last_value = v
            last_date = r["date"]
        if last_value is None or last_date < cutoff:
            continue  # no real value ever, or too stale to present as current
        out.append({
            "key": key,
            "label": f["label"],
            "unit": f["unit"],
            "value": last_value,
            "as_of": last_date.isoformat(),
            "history": history[-365:],  # cap payload size
        })
    return out


def _compute_forex():
    return {"items": _build_group(FOREX_FIELDS, FRESH_WINDOW_DAILY)}


def _compute_commodities():
    return {"items": _build_group(COMMODITY_FIELDS, FRESH_WINDOW_DAILY)}


def _compute_economy():
    return {"items": _build_group(ECONOMY_FIELDS, FRESH_WINDOW_MACRO)}


def _compute_global_markets():
    return {"items": _build_group(GLOBAL_MARKETS_FIELDS, FRESH_WINDOW_DAILY)}


def _get_macro_cached(cache_key: str, compute_fn):
    cached = get_cached(psx_cache, cache_key, MACRO_CACHE_TTL)
    if cached is not None:
        return cached
    result = compute_fn()
    set_cached(psx_cache, cache_key, result)
    return result


@router.get("/api/forex")
async def get_forex():
    return await run_in_threadpool(_get_macro_cached, "macro_forex", _compute_forex)


@router.get("/api/commodities")
async def get_commodities():
    return await run_in_threadpool(_get_macro_cached, "macro_commodities", _compute_commodities)


@router.get("/api/economy")
async def get_economy():
    return await run_in_threadpool(_get_macro_cached, "macro_economy", _compute_economy)


@router.get("/api/global_markets")
async def get_global_markets():
    return await run_in_threadpool(_get_macro_cached, "macro_global_markets", _compute_global_markets)


def _compute_announcements(limit: int, offset: int):
    query = text("""
        SELECT ticker, announcement_date, category, headline_raw_text, sentiment_score
        FROM corporate_announcements_pucars
        ORDER BY announcement_date DESC
        LIMIT :limit OFFSET :offset
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"limit": limit, "offset": offset}).mappings().fetchall()
    return {
        "announcements": [
            {
                "ticker": r["ticker"],
                "date": r["announcement_date"].isoformat() if r["announcement_date"] else None,
                "type": r["category"],
                "title": r["headline_raw_text"],
                "sentiment_score": safe_float(r["sentiment_score"]),
            }
            for r in rows
        ]
    }


@router.get("/api/announcements")
async def get_announcements(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    cache_key = f"announcements_{limit}_{offset}"
    return await run_in_threadpool(
        _get_macro_cached, cache_key, lambda: _compute_announcements(limit, offset)
    )
