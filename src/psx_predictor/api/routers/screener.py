# screener.py
# ---------------------------------------------------------
# Cross-ticker screener and comparison endpoints.
# ---------------------------------------------------------
import os
import sys
import threading
from fastapi import APIRouter, HTTPException, Query

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from sqlalchemy import text
from src.psx_predictor.db.connection import engine
from src.psx_predictor.config import VALID_TICKERS
from src.psx_predictor.api.state import psx_cache
from src.psx_predictor.api.cache import get_cached, set_cached
from src.psx_predictor.api.market_data import get_price_snapshot
from src.psx_predictor.api.ratios import get_latest_ratios
from fastapi.concurrency import run_in_threadpool

router = APIRouter()

# The expensive part of rebuilding this (get_latest_ratios() per ticker) is
# itself cached for 300s (see ratios.py), so keeping this well under that
# means most 60s-TTL expiries used to still redo the full ~107-ticker CSV
# sweep. ratios are derived from EOD data and only change when the feature
# pipeline reruns (at most a few times a day) - 120s keeps price/volume data
# reasonably fresh while usually landing inside the still-warm ratios cache.
SCREENER_CACHE_TTL = 120
SORTABLE_FIELDS = {
    "price", "change_percent", "volume", "market_cap", "pe_ratio", "pb_ratio", "dividend_yield", "roe",
    "pct_from_52w_high", "pct_from_52w_low",
}

_EMPTY_PRICE_DATA = {
    "price": None, "change": None, "change_percent": None, "volume": None,
    "week_52_high": None, "week_52_low": None,
}


def _compute_screener_rows():
    query = text("SELECT ticker, company_name, sector FROM stock_metadata WHERE is_active = true ORDER BY ticker ASC")
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    snapshot = get_price_snapshot()

    out = []
    for row in rows:
        price_data = snapshot.get(row.ticker, _EMPTY_PRICE_DATA)
        ratios = get_latest_ratios(row.ticker) or {}

        price = price_data.get("price")
        week_52_high = price_data.get("week_52_high")
        week_52_low = price_data.get("week_52_low")
        # How close the current price sits to its 52-week high/low, as a %.
        # At the high: pct_from_52w_high == 0 (it can't exceed its own high).
        # At the low: pct_from_52w_low == 0. Screener presets sort on these directly.
        pct_from_52w_high = (
            round((price - week_52_high) / week_52_high * 100, 2)
            if price is not None and week_52_high else None
        )
        pct_from_52w_low = (
            round((price - week_52_low) / week_52_low * 100, 2)
            if price is not None and week_52_low else None
        )

        out.append({
            "ticker": row.ticker,
            "name": row.company_name,
            "sector": row.sector,
            **price_data,
            "pct_from_52w_high": pct_from_52w_high,
            "pct_from_52w_low": pct_from_52w_low,
            "market_cap": ratios.get("market_cap"),
            "pe_ratio": ratios.get("pe_ratio"),
            "pb_ratio": ratios.get("pb_ratio"),
            "dividend_yield": ratios.get("dividend_yield"),
            "roe": ratios.get("roe"),
        })
    return out


_screener_compute_lock = threading.Lock()


def _get_all_screener_rows_cached():
    """Reads every active ticker's master.csv (~0.3s each) when the cache is
    cold. Home page + CuratedLists alone fire 4 concurrent /api/screener
    requests - without this lock, all 4 would see a simultaneous cache miss
    and redundantly repeat the full 107-ticker computation at once (measured
    ~85s wall time). The lock makes the first request compute while the rest
    block, then reuse its result instead of duplicating the work."""
    cache_key = "screener_all_rows"
    cached = get_cached(psx_cache, cache_key, SCREENER_CACHE_TTL)
    if cached is not None:
        return cached
    with _screener_compute_lock:
        # Re-check: another thread may have finished computing while we waited.
        cached = get_cached(psx_cache, cache_key, SCREENER_CACHE_TTL)
        if cached is not None:
            return cached
        rows = _compute_screener_rows()
        set_cached(psx_cache, cache_key, rows)
        return rows


@router.get("/api/screener")
async def get_screener(
    sector: str | None = Query(None),
    min_pe: float | None = Query(None),
    max_pe: float | None = Query(None),
    sort_by: str = Query("market_cap"),
    order: str = Query("desc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
):
    if sort_by not in SORTABLE_FIELDS:
        raise HTTPException(status_code=400, detail=f"sort_by must be one of {sorted(SORTABLE_FIELDS)}")
    if order not in ("asc", "desc"):
        raise HTTPException(status_code=400, detail="order must be 'asc' or 'desc'")

    rows = await run_in_threadpool(_get_all_screener_rows_cached)

    filtered = rows
    if sector:
        filtered = [r for r in filtered if r["sector"] == sector]
    if min_pe is not None:
        filtered = [r for r in filtered if r["pe_ratio"] is not None and r["pe_ratio"] >= min_pe]
    if max_pe is not None:
        filtered = [r for r in filtered if r["pe_ratio"] is not None and r["pe_ratio"] <= max_pe]

    filtered.sort(key=lambda r: (r[sort_by] is None, r[sort_by]), reverse=(order == "desc"))

    total = len(filtered)
    start = (page - 1) * page_size
    page_rows = filtered[start:start + page_size]

    # Universe-wide counts (from the full unfiltered `rows`, not `filtered`)
    # for a "502 companies - 155 up / 310 down" style summary line, unaffected
    # by whatever sector/P-E filter or sort preset is currently active.
    total_tracked = len(rows)
    total_up = sum(1 for r in rows if r["change_percent"] is not None and r["change_percent"] > 0)
    total_down = sum(1 for r in rows if r["change_percent"] is not None and r["change_percent"] < 0)

    return {
        "total": total, "page": page, "page_size": page_size, "results": page_rows,
        "total_tracked": total_tracked, "total_up": total_up, "total_down": total_down,
    }


@router.get("/api/compare")
async def compare_tickers(tickers: str = Query(..., description="Comma-separated tickers, e.g. PSO,MEBL,OGDC")):
    requested = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not (2 <= len(requested) <= 5):
        raise HTTPException(status_code=400, detail="Provide between 2 and 5 comma-separated tickers.")

    invalid = [t for t in requested if t not in VALID_TICKERS]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid ticker(s): {invalid}")

    rows = await run_in_threadpool(_get_all_screener_rows_cached)
    by_ticker = {r["ticker"]: r for r in rows}
    result = [by_ticker[t] for t in requested if t in by_ticker]
    return {"results": result}
