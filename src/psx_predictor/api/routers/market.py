# market.py
# ---------------------------------------------------------
# Market-wide endpoints: ticker list, market performers, sector overview.
#
# /api/market_performers used to fan out a live scrape across every active
# ticker via ThreadPoolExecutor on every 15s cache miss — the single biggest
# reliability risk in the old API (hammered dps.psx.com.pk under any real
# traffic). It's now a single DB query (see api/market_data.py); the response
# shape (symbol/price/change/change_percent/volume) is unchanged so the
# existing frontend MarketPerformers component needs no changes.
# ---------------------------------------------------------
import os
import sys
import logging
import threading
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from sqlalchemy import text
from src.psx_predictor.db.connection import engine
from src.psx_predictor.api.state import psx_cache
from src.psx_predictor.api.cache import get_cached, set_cached
from src.psx_predictor.api.market_data import get_price_snapshot
from src.psx_predictor.api.ratios import get_latest_ratios

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
if not logger.handlers:
    ch = logging.StreamHandler()
    logger.addHandler(ch)

router = APIRouter()

CACHE_TTL = 15
SECTOR_DETAIL_CACHE_TTL = 120  # loops get_latest_ratios() per ticker in the sector - see _get_sector_detail_cached

# PSX's own published sector indices (BKTI/OGTI, scraped in psx_dps_index_scraper.py)
# cover these stock_metadata.sector values. Oil & Gas is a single combined PSX
# index covering both Exploration and Marketing companies.
SECTOR_INDEX_MAP = {
    "Commercial Banks": "banking_sector_index_level",
    "Oil & Gas Marketing": "oil_gas_sector_index_level",
    "Oil & Gas Exploration": "oil_gas_sector_index_level",
}


class TickerInfo(BaseModel):
    ticker: str
    name: str
    sector: str


def _compute_all_tickers():
    query = text("SELECT ticker, company_name, sector FROM stock_metadata WHERE is_active = true ORDER BY ticker ASC")
    with engine.connect() as conn:
        result = conn.execute(query).fetchall()
    return [TickerInfo(ticker=row.ticker, name=row.company_name, sector=row.sector) for row in result]


@router.get("/api/tickers", response_model=list[TickerInfo])
async def get_all_tickers():
    cache_key = "all_tickers"
    cached = get_cached(psx_cache, cache_key, CACHE_TTL * 10)  # cache for longer since it rarely changes
    if cached is not None:
        return cached

    try:
        # Every other endpoint in this file offloads its DB call to the
        # threadpool - this one used to run the query directly on the async
        # event loop, blocking every other in-flight request for its duration.
        tickers_list = await run_in_threadpool(_compute_all_tickers)
        set_cached(psx_cache, cache_key, tickers_list)
        return tickers_list
    except Exception as e:
        logger.error(f"Failed to fetch tickers: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while fetching tickers.")


@router.get("/api/market_performers")
async def get_market_performers():
    cache_key = "market_performers"
    cached = get_cached(psx_cache, cache_key, CACHE_TTL)
    if cached is not None:
        return cached

    snapshot = await run_in_threadpool(get_price_snapshot)
    results = [{"symbol": ticker, **data} for ticker, data in snapshot.items()]

    top_active = sorted(results, key=lambda x: x['volume'], reverse=True)
    top_advancers = sorted(results, key=lambda x: x['change_percent'], reverse=True)
    top_decliners = sorted(results, key=lambda x: x['change_percent'])

    result = {
        "top_active": top_active,
        "top_advancers": top_advancers,
        "top_decliners": top_decliners
    }
    set_cached(psx_cache, cache_key, result)
    return result


def _compute_sectors_overview():
    query = text("SELECT ticker, sector FROM stock_metadata WHERE is_active = true")
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    by_sector: dict[str, list[str]] = {}
    for row in rows:
        by_sector.setdefault(row.sector or "Unknown", []).append(row.ticker)

    latest_macro_query = text("SELECT * FROM macro_indicators ORDER BY date DESC LIMIT 1")
    with engine.connect() as conn:
        latest_macro = conn.execute(latest_macro_query).mappings().fetchone()

    sectors_out = []
    for sector, tickers in by_sector.items():
        ratios = [get_latest_ratios(t) for t in tickers]
        ratios = [r for r in ratios if r is not None]
        pe_vals = [r["pe_ratio"] for r in ratios if r.get("pe_ratio") is not None]
        pb_vals = [r["pb_ratio"] for r in ratios if r.get("pb_ratio") is not None]

        index_col = SECTOR_INDEX_MAP.get(sector)
        index_level = None
        if index_col and latest_macro is not None:
            val = latest_macro.get(index_col)
            index_level = float(val) if val is not None else None

        sectors_out.append({
            "sector": sector,
            "ticker_count": len(tickers),
            "tickers": sorted(tickers),
            "avg_pe": round(sum(pe_vals) / len(pe_vals), 2) if pe_vals else None,
            "avg_pb": round(sum(pb_vals) / len(pb_vals), 2) if pb_vals else None,
            "index_level": index_level,
        })

    sectors_out.sort(key=lambda s: s["sector"])
    return {"sectors": sectors_out}


_sectors_compute_lock = threading.Lock()


def _get_sectors_overview_cached():
    """Same thundering-herd guard as screener.py's cache - this also loops
    get_latest_ratios() over every active ticker, so concurrent cold requests
    would otherwise each redundantly repeat that work."""
    cache_key = "sectors_overview"
    cached = get_cached(psx_cache, cache_key, CACHE_TTL * 10)
    if cached is not None:
        return cached
    with _sectors_compute_lock:
        cached = get_cached(psx_cache, cache_key, CACHE_TTL * 10)
        if cached is not None:
            return cached
        result = _compute_sectors_overview()
        set_cached(psx_cache, cache_key, result)
        return result


@router.get("/api/sectors")
async def get_sectors():
    return await run_in_threadpool(_get_sectors_overview_cached)


def _compute_sector_detail(sector: str):
    query = text("SELECT ticker, company_name, sector FROM stock_metadata WHERE is_active = true AND sector = :sector ORDER BY ticker ASC")
    with engine.connect() as conn:
        rows = conn.execute(query, {"sector": sector}).fetchall()

    if not rows:
        return None

    snapshot = get_price_snapshot()

    tickers_out = []
    for row in rows:
        price_data = snapshot.get(row.ticker, {"price": None, "change": None, "change_percent": None, "volume": None})
        ratios = get_latest_ratios(row.ticker) or {}
        tickers_out.append({
            "ticker": row.ticker,
            "name": row.company_name,
            "sector": row.sector,
            **price_data,
            "market_cap": ratios.get("market_cap"),
            "pe_ratio": ratios.get("pe_ratio"),
            "pb_ratio": ratios.get("pb_ratio"),
            "dividend_yield": ratios.get("dividend_yield"),
            "roe": ratios.get("roe"),
        })
    return {"sector": sector, "tickers": tickers_out}


_sector_detail_locks: dict[str, threading.Lock] = {}
_sector_detail_locks_guard = threading.Lock()


def _get_sector_detail_cached(sector: str):
    """Same pattern as the screener/sectors-overview caches: this loops
    get_latest_ratios() over every ticker in the sector, and is hit both by
    the Sectors page and every CompanyDetail page's Peers tab, so it's worth
    caching (previously uncached - every visit redid the full per-ticker
    loop from scratch)."""
    cache_key = f"sector_detail_{sector}"
    cached = get_cached(psx_cache, cache_key, SECTOR_DETAIL_CACHE_TTL)
    if cached is not None:
        return cached
    with _sector_detail_locks_guard:
        lock = _sector_detail_locks.setdefault(sector, threading.Lock())
    with lock:
        cached = get_cached(psx_cache, cache_key, SECTOR_DETAIL_CACHE_TTL)
        if cached is not None:
            return cached
        result = _compute_sector_detail(sector)
        set_cached(psx_cache, cache_key, result)
        return result


@router.get("/api/sectors/{sector}")
async def get_sector_detail(sector: str):
    result = await run_in_threadpool(_get_sector_detail_cached, sector)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No active tickers found for sector '{sector}'.")
    return result


# Real broad-market indices scraped in psx_dps_index_scraper.py. Deliberately
# excludes "KSE100" - stock_market_index's KSE100 row is a documented
# synthetic equal-weighted proxy (is_synthetic_index=True), not the real
# PSX-published index, so it's never surfaced here as market data.
INDEX_DEFS = [
    {"key": "kmi30", "name": "KMI-30", "column": "kmi30_index_level"},
    {"key": "kse30", "name": "KSE-30", "column": "kse30_index_level"},
    {"key": "all_share", "name": "All Share Index", "column": "all_share_index_level"},
]


def _compute_indices():
    query = text("""
        SELECT date, kmi30_index_level, kse30_index_level, all_share_index_level
        FROM macro_indicators ORDER BY date ASC
    """)
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().fetchall()

    indices_out = []
    for idx in INDEX_DEFS:
        col = idx["column"]
        history = [
            {"date": r["date"].isoformat(), "level": float(r[col])}
            for r in rows if r[col] is not None
        ]
        if not history:
            indices_out.append({**idx, "level": None, "change_percent": None, "history": []})
            continue
        latest = history[-1]["level"]
        prev = history[-2]["level"] if len(history) > 1 else latest
        change_percent = round((latest - prev) / prev * 100, 2) if prev else 0.0
        indices_out.append({
            "key": idx["key"],
            "name": idx["name"],
            "level": round(latest, 2),
            "change_percent": change_percent,
            "history": history[-400:],  # ~1.5 trading years, enough for a YTD/1Y comparison
        })
    return {"indices": indices_out}


@router.get("/api/indices")
async def get_indices():
    cache_key = "indices_overview"
    cached = get_cached(psx_cache, cache_key, CACHE_TTL * 10)
    if cached is not None:
        return cached
    result = await run_in_threadpool(_compute_indices)
    set_cached(psx_cache, cache_key, result)
    return result
