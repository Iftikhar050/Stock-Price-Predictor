# company.py
# ---------------------------------------------------------
# Per-company endpoints: live profile/realtime price (still scraped, now with
# retry/backoff + DB fallback), plus new DB-backed fundamentals/history/events
# endpoints for the company-detail page.
# ---------------------------------------------------------
import os
import sys
import time
import logging
from datetime import datetime
from bs4 import BeautifulSoup
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from sqlalchemy import text
from src.psx_predictor.db.connection import engine
from src.psx_predictor.config import VALID_TICKERS
from src.psx_predictor.api.state import psx_cache
from src.psx_predictor.api.cache import get_cached, set_cached
from src.psx_predictor.api.http_client import get_with_retry, post_with_retry
from src.psx_predictor.api.json_safe import safe_float
from src.psx_predictor.api.market_data import get_price_snapshot
from src.psx_predictor.api.ratios import get_latest_ratios
from src.psx_predictor.api.technicals import get_latest_technicals

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
if not logger.handlers:
    ch = logging.StreamHandler()
    logger.addHandler(ch)

router = APIRouter()

PROFILE_CACHE_TTL = 24 * 3600  # company profile/description/directors rarely change
REALTIME_CACHE_TTL = 30

RANGE_TO_DAYS = {"7D": 7, "30D": 30, "90D": 90, "1Y": 365, "5Y": 365 * 5}


def _fallback_profile_from_db(ticker: str) -> dict:
    """DB-only profile used when the live PSX scrape fails, so the page degrades
    gracefully instead of 404ing outright. Never fabricates description/people/details
    that only exist on the live site — those stay empty."""
    query = text("SELECT company_name, sector FROM stock_metadata WHERE ticker = :ticker")
    with engine.connect() as conn:
        row = conn.execute(query, {"ticker": ticker.upper()}).fetchone()
    return {
        "name": row.company_name if row else ticker.upper(),
        "sector": row.sector if row else "Unknown Sector",
        "description": "",
        "people": [],
        "details": {},
    }


def fetch_psx_company_profile(ticker: str):
    cache_key = f"profile_{ticker.upper()}"
    cached = get_cached(psx_cache, cache_key, PROFILE_CACHE_TTL)
    if cached is not None:
        return cached

    url = f"https://dps.psx.com.pk/company/{ticker.upper()}"
    try:
        r = get_with_retry(url)
    except Exception as e:
        logger.error(f"Company profile scrape failed for {ticker}, falling back to DB: {e}")
        return _fallback_profile_from_db(ticker)

    if r.status_code != 200:
        logger.error(f"Company profile scrape non-200 for {ticker} ({r.status_code}), falling back to DB.")
        return _fallback_profile_from_db(ticker)

    soup = BeautifulSoup(r.text, 'html.parser')

    # 1. Name & Sector
    name_el = soup.select_one('.quote__name')
    sector_el = soup.select_one('.quote__sector')
    if name_el:
        # PSX nests a status badge (e.g. "XD" for ex-dividend, "XB" for ex-bonus)
        # directly inside .quote__name with no separating whitespace in the raw
        # HTML, so .text alone yields "AGP LimitedXD". Strip the badge first.
        tag_el = name_el.select_one('.tag')
        if tag_el:
            tag_el.extract()
        name = name_el.text.strip()
    else:
        name = ticker.upper()
    # PSX's live profile page renders sector labels in its own casing/wording
    # (e.g. "OIL & GAS MARKETING COMPANIES") that doesn't exact-match
    # stock_metadata.sector ("Oil & Gas Marketing") - the value used by the
    # Peers tab's /api/sectors/{sector} lookup and by the Screener/Sectors
    # pages. Prefer the DB's canonical sector so those exact-match lookups
    # actually resolve; fall back to the scraped text only if this ticker
    # has no DB row (shouldn't happen for VALID_TICKERS, but stay safe).
    scraped_sector = sector_el.text.strip() if sector_el else None
    with engine.connect() as conn:
        db_sector_row = conn.execute(
            text("SELECT sector FROM stock_metadata WHERE ticker = :ticker"),
            {"ticker": ticker.upper()},
        ).fetchone()
    sector = (db_sector_row.sector if db_sector_row and db_sector_row.sector else None) or scraped_sector or "Unknown Sector"

    # 2. Description
    desc_el = soup.select_one('.profile__item--decription p')
    desc = desc_el.text.strip() if desc_el else "No description available."

    # 3. People
    people = []
    trs = soup.select('.profile__item--people tr')
    for tr in trs:
        tds = tr.find_all('td')
        if len(tds) == 2:
            people.append({
                "name": tds[0].text.strip(),
                "role": tds[1].text.strip()
            })

    # 4. Other Details (Address, Web, Auditor, etc)
    details = {}
    items = soup.select('.profile__item')
    for item in items:
        heads = item.select('.item__head')
        for head in heads:
            key = head.text.strip()
            # The value is usually in the next <p> tag
            val_node = head.find_next_sibling('p')
            if val_node:
                details[key] = val_node.text.strip()

    result = {
        "name": name,
        "sector": sector,
        "description": desc,
        "people": people,
        "details": details
    }
    set_cached(psx_cache, cache_key, result)
    return result


@router.get("/api/company/{ticker}")
async def get_company_profile(ticker: str):
    ticker = ticker.upper()
    if ticker not in VALID_TICKERS:
        raise HTTPException(status_code=400, detail=f"Invalid ticker. Must be one of {VALID_TICKERS}")

    profile = await run_in_threadpool(fetch_psx_company_profile, ticker)
    if not profile:
        raise HTTPException(status_code=404, detail="Company profile not found on PSX.")
    return profile


class RealtimePriceResponse(BaseModel):
    ticker: str
    price: float
    change: float
    change_percent: float


def _fallback_realtime_from_db(ticker: str) -> RealtimePriceResponse | None:
    """Latest two closes from stock_eod_data, used when the live scrape fails."""
    query = text("""
        SELECT date, close FROM stock_eod_data
        WHERE ticker = :ticker
        ORDER BY date DESC
        LIMIT 2
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"ticker": ticker.upper()}).fetchall()
    if not rows:
        return None
    latest_close = float(rows[0].close)
    prev_close = float(rows[1].close) if len(rows) > 1 else latest_close
    change = latest_close - prev_close
    change_percent = (change / prev_close * 100) if prev_close else 0.0
    return RealtimePriceResponse(
        ticker=ticker.upper(),
        price=round(latest_close, 2),
        change=round(change, 2),
        change_percent=round(change_percent, 2),
    )


@router.get("/api/realtime/{ticker}", response_model=RealtimePriceResponse)
async def get_realtime_price(ticker: str):
    ticker = ticker.upper()
    if ticker not in VALID_TICKERS:
        raise HTTPException(status_code=400, detail=f"Invalid ticker. Must be one of {VALID_TICKERS}")

    cache_key = f"realtime_{ticker}"
    cached = get_cached(psx_cache, cache_key, REALTIME_CACHE_TTL)
    if cached is not None:
        return cached

    url = f"https://dps.psx.com.pk/company/{ticker}"

    def fetch():
        return get_with_retry(url, timeout=5)

    try:
        r = await run_in_threadpool(fetch)
        if r.status_code != 200:
            raise ValueError(f"non-200 status {r.status_code}")

        soup = BeautifulSoup(r.text, 'html.parser')

        price_el = soup.select_one('.quote__close')
        price_str = price_el.text.strip().replace('Rs.', '').replace(',', '')
        price = float(price_str)

        change_el = soup.select_one('.change__value')
        change_str = change_el.text.strip().replace(',', '')
        change = float(change_str)

        percent_el = soup.select_one('.change__percent')
        percent_str = percent_el.text.strip().replace('(', '').replace(')', '').replace('%', '')
        percent = float(percent_str)

        result = RealtimePriceResponse(
            ticker=ticker,
            price=price,
            change=change,
            change_percent=percent
        )
        set_cached(psx_cache, cache_key, result)
        return result
    except Exception as e:
        logger.error(f"Live realtime scrape failed for {ticker}, falling back to DB: {str(e)}")
        fallback = _fallback_realtime_from_db(ticker)
        if fallback is None:
            raise HTTPException(status_code=404, detail="No price data available for this ticker.")
        return fallback


def _serialize_fundamentals_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        elif isinstance(v, float):
            out[k] = safe_float(v)  # some fundamentals rows store NaN instead of NULL; JSON has no NaN literal
        else:
            out[k] = v
    return out


def _compute_company_fundamentals(ticker: str):
    query = text("SELECT * FROM stock_fundamentals WHERE ticker = :ticker ORDER BY report_date ASC")
    with engine.connect() as conn:
        rows = conn.execute(query, {"ticker": ticker}).mappings().fetchall()
    return {"ticker": ticker, "fundamentals": [_serialize_fundamentals_row(dict(r)) for r in rows]}


@router.get("/api/company/{ticker}/fundamentals")
async def get_company_fundamentals(ticker: str):
    """Full fundamentals/ratio history for trend charts (revenue, margins, ROE, etc.)."""
    ticker = ticker.upper()
    if ticker not in VALID_TICKERS:
        raise HTTPException(status_code=400, detail=f"Invalid ticker. Must be one of {VALID_TICKERS}")

    return await run_in_threadpool(_compute_company_fundamentals, ticker)


@router.get("/api/company/{ticker}/snapshot")
async def get_company_snapshot(ticker: str):
    """Compact stat-grid data (52W range, market cap, shares outstanding, free
    float, insider 30d flow) for the company-detail page. Combines the same
    price-snapshot and ratios helpers the screener already uses, keyed to one
    ticker instead of all of them."""
    ticker = ticker.upper()
    if ticker not in VALID_TICKERS:
        raise HTTPException(status_code=400, detail=f"Invalid ticker. Must be one of {VALID_TICKERS}")

    def _compute():
        price_data = get_price_snapshot().get(ticker, {})
        ratios = get_latest_ratios(ticker) or {}
        return {
            "ticker": ticker,
            "week_52_high": price_data.get("week_52_high"),
            "week_52_low": price_data.get("week_52_low"),
            "market_cap": ratios.get("market_cap"),
            "pe_ratio": ratios.get("pe_ratio"),
            "pb_ratio": ratios.get("pb_ratio"),
            "sector_pe_avg": ratios.get("sector_pe_avg"),
            "sector_pb_avg": ratios.get("sector_pb_avg"),
            "dividend_yield": ratios.get("dividend_yield"),
            "roe": ratios.get("roe"),
            "eps_trailing": ratios.get("eps_trailing"),
            "shares_outstanding": ratios.get("shares_outstanding"),
            "free_float": ratios.get("free_float"),
            "free_float_pct": ratios.get("free_float_pct"),
            "insider_buy_shares_30d": ratios.get("insider_buy_shares_30d"),
            "insider_sell_shares_30d": ratios.get("insider_sell_shares_30d"),
            "insider_net_flow_30d": ratios.get("insider_net_flow_30d"),
            # Derived valuation ratios/percentiles (see ratios.py RATIO_COLUMNS)
            "pe_percentile_1y": ratios.get("pe_percentile_1y"),
            "pe_percentile_3y": ratios.get("pe_percentile_3y"),
            "pb_percentile_3y": ratios.get("pb_percentile_3y"),
            "dividend_yield_percentile_3y": ratios.get("dividend_yield_percentile_3y"),
            "pe_1y_avg": ratios.get("pe_1y_avg"),
            "pe_3y_avg": ratios.get("pe_3y_avg"),
            "pe_5y_avg": ratios.get("pe_5y_avg"),
            "forward_pe": ratios.get("forward_pe"),
            "price_to_cash_flow": ratios.get("price_to_cash_flow"),
            "ev": ratios.get("ev"),
            "ev_ebitda": ratios.get("ev_ebitda"),
            "ev_sales": ratios.get("ev_sales"),
            "profit_margin": ratios.get("profit_margin"),
            "roa": ratios.get("roa"),
            "peg_ratio": ratios.get("peg_ratio"),
            "book_value_per_share": ratios.get("book_value_per_share"),
            "debt_to_equity": ratios.get("debt_to_equity"),
        }

    return await run_in_threadpool(_compute)


@router.get("/api/company/{ticker}/technicals")
async def get_company_technicals(ticker: str):
    """Latest technical-indicator readings (moving averages, RSI, MACD,
    Bollinger Bands, ADX/DI, stochastic, Williams %R, CCI, beta, historical
    volatility) computed by build_features.py. These were already computed
    for every ticker but had never been exposed via any endpoint before."""
    ticker = ticker.upper()
    if ticker not in VALID_TICKERS:
        raise HTTPException(status_code=400, detail=f"Invalid ticker. Must be one of {VALID_TICKERS}")

    technicals = await run_in_threadpool(get_latest_technicals, ticker)
    if technicals is None:
        raise HTTPException(status_code=404, detail=f"No technical data available for {ticker}")
    return {"ticker": ticker, **technicals}


def _compute_company_dividends(ticker: str):
    query = text("""
        SELECT ex_dividend_date, announcement_date, dividend_amount, dividend_type
        FROM stock_dividends
        WHERE ticker = :ticker
        ORDER BY ex_dividend_date DESC
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"ticker": ticker}).mappings().fetchall()

    dividends = [
        {
            "ex_date": r["ex_dividend_date"].isoformat() if r["ex_dividend_date"] else None,
            "announcement_date": r["announcement_date"].isoformat() if r["announcement_date"] else None,
            "amount": safe_float(r["dividend_amount"]),
            "type": r["dividend_type"],
        }
        for r in rows
    ]
    return {"ticker": ticker, "dividends": dividends}


@router.get("/api/company/{ticker}/dividends")
async def get_company_dividends(ticker: str):
    """Full dividend payout history (the /events feed only ever shows the latest one)."""
    ticker = ticker.upper()
    if ticker not in VALID_TICKERS:
        raise HTTPException(status_code=400, detail=f"Invalid ticker. Must be one of {VALID_TICKERS}")

    return await run_in_threadpool(_compute_company_dividends, ticker)


FINANCIALS_PORTAL_BASE = "https://financials.psx.com.pk"
REPORTS_CACHE_TTL = 24 * 3600  # report filings rarely change once posted


def _fetch_reports_for_year(ticker: str, year: int) -> list[dict]:
    """Calls PSX's own Financial Portal AJAX endpoint (reverse-engineered from
    financials.psx.com.pk - it's what powers that site's own "Download File"
    button). We don't host or store the PDFs, just link out to PSX's originals."""
    resp = post_with_retry(
        f"{FINANCIALS_PORTAL_BASE}/annQtrStmts.php",
        data={"name": "get_comp_y_data", "smbCode": ticker, "year": year},
        timeout=10,
    )
    if resp.status_code != 200:
        return []
    try:
        rows = resp.json()
    except ValueError:
        return []

    out = []
    for row in rows:
        soup = BeautifulSoup(row.get("Reports", ""), "html.parser")
        link = soup.find("a")
        if not link or not link.get("href"):
            continue
        href = link["href"]
        download_url = href if href.startswith("http") else f"{FINANCIALS_PORTAL_BASE}/{href.lstrip('/')}"
        out.append({
            "type": link.text.strip() or "Report",
            "period_ended": row.get("period_ended"),
            "posting_date": row.get("posting_date"),
            "download_url": download_url,
            "year": year,
        })
    return out


@router.get("/api/company/{ticker}/reports")
async def get_company_reports(ticker: str, years: int = Query(3, ge=1, le=10)):
    """Real Annual/Quarterly report filing links for the last N years, sourced
    live from PSX's official Financial Portal."""
    ticker = ticker.upper()
    if ticker not in VALID_TICKERS:
        raise HTTPException(status_code=400, detail=f"Invalid ticker. Must be one of {VALID_TICKERS}")

    cache_key = f"reports_{ticker}_{years}"
    cached = get_cached(psx_cache, cache_key, REPORTS_CACHE_TTL)
    if cached is not None:
        return cached

    def _compute():
        current_year = datetime.now().year
        all_reports = []
        for y in range(current_year, current_year - years, -1):
            try:
                all_reports.extend(_fetch_reports_for_year(ticker, y))
            except Exception as e:
                logger.error(f"Failed to fetch {y} financial reports for {ticker}: {e}")
        all_reports.sort(key=lambda r: r.get("posting_date") or "", reverse=True)
        return {"ticker": ticker, "reports": all_reports}

    result = await run_in_threadpool(_compute)
    set_cached(psx_cache, cache_key, result)
    return result


def _compute_company_history(ticker: str, range_key: str):
    if range_key == "MAX":
        query = text("""
            SELECT date, open, high, low, close, volume FROM stock_eod_data
            WHERE ticker = :ticker ORDER BY date ASC
        """)
        params = {"ticker": ticker}
    else:
        days = RANGE_TO_DAYS[range_key]
        query = text("""
            SELECT date, open, high, low, close, volume FROM stock_eod_data
            WHERE ticker = :ticker AND date >= (CURRENT_DATE - CAST(:days AS INTEGER))
            ORDER BY date ASC
        """)
        params = {"ticker": ticker, "days": days}

    with engine.connect() as conn:
        rows = conn.execute(query, params).mappings().fetchall()

    history = [
        {
            "date": r["date"].isoformat(),
            "open": safe_float(r["open"]),
            "high": safe_float(r["high"]),
            "low": safe_float(r["low"]),
            "close": safe_float(r["close"]),
            "volume": int(r["volume"]) if r["volume"] is not None else None,
        }
        for r in rows
    ]
    return {"ticker": ticker, "range": range_key, "history": history}


@router.get("/api/company/{ticker}/history")
async def get_company_history(ticker: str, range: str = Query("1Y", description="One of 7D, 30D, 90D, 1Y, 5Y, MAX")):
    """Plain OHLCV price history, decoupled from the ML-heavy /api/predict endpoint."""
    ticker = ticker.upper()
    if ticker not in VALID_TICKERS:
        raise HTTPException(status_code=400, detail=f"Invalid ticker. Must be one of {VALID_TICKERS}")

    range_key = range.upper()
    if range_key != "MAX" and range_key not in RANGE_TO_DAYS:
        raise HTTPException(status_code=400, detail=f"Invalid range. Must be one of {list(RANGE_TO_DAYS.keys())} or MAX")

    return await run_in_threadpool(_compute_company_history, ticker, range_key)


def _compute_company_events(ticker: str, limit: int, offset: int):
    query = text("""
        SELECT ex_dividend_date AS event_date, 'dividend' AS event_type,
               (dividend_type || ' Dividend: Rs. ' || dividend_amount) AS title,
               NULL::float AS sentiment_score
        FROM stock_dividends WHERE ticker = :ticker
        UNION ALL
        SELECT announcement_date AS event_date, category AS event_type,
               headline_raw_text AS title, sentiment_score
        FROM corporate_announcements_pucars WHERE ticker = :ticker
        ORDER BY event_date DESC
        LIMIT :limit OFFSET :offset
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"ticker": ticker, "limit": limit, "offset": offset}).mappings().fetchall()

    events = [
        {
            "date": r["event_date"].isoformat() if r["event_date"] is not None else None,
            "type": r["event_type"],
            "title": r["title"],
            "sentiment_score": safe_float(r["sentiment_score"]),
        }
        for r in rows
    ]
    return {"ticker": ticker, "events": events}


@router.get("/api/company/{ticker}/events")
async def get_company_events(ticker: str, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    """Merged timeline of dividends and corporate announcements."""
    ticker = ticker.upper()
    if ticker not in VALID_TICKERS:
        raise HTTPException(status_code=400, detail=f"Invalid ticker. Must be one of {VALID_TICKERS}")

    return await run_in_threadpool(_compute_company_events, ticker, limit, offset)
