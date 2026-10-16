# dividends.py
# ---------------------------------------------------------
# Market-wide dividend table: trailing-12-month payout, yield, growth,
# latest ex-date, payout ratio and payout frequency per active ticker.
# ---------------------------------------------------------
import os
import sys
import math
import threading
from collections import defaultdict
from datetime import date, timedelta

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from sqlalchemy import text
from src.psx_predictor.db.connection import engine
from src.psx_predictor.api.state import psx_cache
from src.psx_predictor.api.cache import get_cached, set_cached
from src.psx_predictor.api.market_data import get_price_snapshot

router = APIRouter()

CACHE_TTL = 600
DUPLICATE_WINDOW_DAYS = 21
# A quarterly EPS row this many times the median of its neighbours is a
# full-year total mislabelled into the quarterly series, not a quarter.
ANNUAL_ROW_MULTIPLE = 2.2
MAX_PLAUSIBLE_PAYOUT = 3.0


def _dedupe_payouts(rows):
    """stock_dividends carries the same payout twice from two sources - one
    with the real ex-date, one snapped to the 1st of the month. Collapse
    equal amounts within a few weeks of each other, keeping the real date."""
    kept = []
    for ex_date, amount in sorted(rows):
        prev = kept[-1] if kept else None
        if prev and abs(prev[1] - amount) < 1e-6 and (ex_date - prev[0]).days <= DUPLICATE_WINDOW_DAYS:
            if prev[0].day == 1 and ex_date.day != 1:
                kept[-1] = (ex_date, amount)
            continue
        kept.append((ex_date, amount))
    return kept


def _ttm_eps(quarters, today):
    """Sum of the four most recent genuinely-quarterly EPS figures."""
    eps = [(d, v) for d, v in quarters if d <= today and v is not None and not math.isnan(v)]
    eps.sort(reverse=True)
    eps = eps[:8]
    if len(eps) < 4:
        return None
    values = sorted(v for _, v in eps)
    median = values[len(values) // 2]
    if median > 0:
        eps = [(d, v) for d, v in eps if v <= median * ANNUAL_ROW_MULTIPLE]
    if len(eps) < 4:
        return None
    return sum(v for _, v in eps[:4])


FREQUENCY_LABELS = {4: "Quarterly", 3: "3x a Year", 2: "Semi-Annual", 1: "Annual"}
# An annual payer's next dividend lands roughly 12 months after the last one,
# often a few weeks later - so "currently paying" needs slack past a year.
CURRENT_PAYER_DAYS = 455
PRIOR_YEAR_MAX_DAYS = 820
IRREGULAR_GAP_DAYS = 450
FREQUENCY_SAMPLES = (4, 5, 6)


def _payouts_per_year(paid):
    """Payouts per year from the average gap between recent payouts, or None
    when irregular. Counting payouts inside a calendar window instead made
    annual payers flip to 'Irregular' whenever an ex-date drifted a few days.
    Average, not median: PSX quarterly payers space their final dividend
    unevenly, which pushes the median gap past a quarter. Several sample
    sizes, keeping the highest rate, so a skipped year further back doesn't
    hide a company's current cadence."""
    best = None
    for k in FREQUENCY_SAMPLES:
        dates = [d for d, _ in paid[-k:]]
        if len(dates) < 2:
            continue
        mean_gap = max((dates[-1] - dates[0]).days / (len(dates) - 1), 1)
        if mean_gap > IRREGULAR_GAP_DAYS:
            continue
        per_year = min(4, max(1, round(365 / mean_gap)))
        best = per_year if best is None else max(best, per_year)
    return best


def _annual_dividend(paid, n):
    """(this year's total, last year's total, growth_note). Counting the last
    n payouts instead of a fixed 365-day window keeps a year-end dividend
    that slips a few weeks from landing twice in one window - which
    otherwise doubles yield and fakes huge growth."""
    if n is None:
        anchor = paid[-1][0]
        current = sum(a for d, a in paid if d > anchor - timedelta(days=365))
        prior = sum(a for d, a in paid if anchor - timedelta(days=730) < d <= anchor - timedelta(days=365))
        if prior:
            return current, prior, None
        older = any(d <= anchor - timedelta(days=365) for d, _ in paid)
        return current, 0.0, "resumed" if older else "limited_history"
    current = sum(a for _, a in paid[-n:])
    if len(paid) < 2 * n:
        return current, 0.0, "limited_history"
    prior_block = paid[-2 * n:-n]
    # A "prior year" that ended long before this one is a skipped year, not
    # a comparable one - comparing against it fakes growth.
    if prior_block[0][0] < paid[-1][0] - timedelta(days=PRIOR_YEAR_MAX_DAYS):
        return current, 0.0, "resumed"
    return current, sum(a for _, a in prior_block), None


def _compute_dividends():
    today = date.today()
    with engine.connect() as conn:
        meta = conn.execute(text(
            "SELECT ticker, company_name, sector FROM stock_metadata WHERE is_active = true"
        )).fetchall()
        div_rows = conn.execute(text(
            "SELECT ticker, ex_dividend_date, dividend_amount FROM stock_dividends "
            "WHERE dividend_type = 'Cash' AND ex_dividend_date IS NOT NULL AND dividend_amount > 0"
        )).fetchall()
        eps_rows = conn.execute(text(
            "SELECT ticker, report_date, eps FROM stock_fundamentals WHERE report_date >= :since"
        ), {"since": today - timedelta(days=800)}).fetchall()

    payouts_by_ticker = defaultdict(list)
    for r in div_rows:
        payouts_by_ticker[r.ticker].append((r.ex_dividend_date, float(r.dividend_amount)))

    eps_by_ticker = defaultdict(list)
    for r in eps_rows:
        eps_by_ticker[r.ticker].append((r.report_date, float(r.eps) if r.eps is not None else None))

    prices = get_price_snapshot()
    current_cutoff = today - timedelta(days=CURRENT_PAYER_DAYS)

    payers, non_payers = [], []
    for m in meta:
        payouts = _dedupe_payouts(payouts_by_ticker.get(m.ticker, []))
        paid = [(d, a) for d, a in payouts if d <= today]
        last_date, last_amount = paid[-1] if paid else (None, None)

        if last_date is None or last_date < current_cutoff:
            non_payers.append({
                "symbol": m.ticker,
                "name": m.company_name,
                "sector": m.sector,
                "last_ex_dividend_date": last_date.isoformat() if last_date else None,
                "last_amount": round(last_amount, 2) if last_amount is not None else None,
            })
            continue

        per_year = _payouts_per_year(paid)
        frequency = FREQUENCY_LABELS.get(per_year, "Irregular")
        annual, prior, growth_note = _annual_dividend(paid, per_year)
        price = (prices.get(m.ticker) or {}).get("price")
        div_yield = annual / price if price else None
        growth = annual / prior - 1 if prior > 0 else None

        payout, payout_note = None, None
        eps = _ttm_eps(eps_by_ticker.get(m.ticker, []), today)
        if eps is None:
            payout_note = "no_eps"
        elif eps <= 0:
            payout_note = "loss"
        elif annual / eps > MAX_PLAUSIBLE_PAYOUT:
            payout_note = "not_meaningful"
        else:
            payout = annual / eps

        payers.append({
            "symbol": m.ticker,
            "name": m.company_name,
            "sector": m.sector,
            "price": price,
            "dividend": round(annual, 2),
            "dividend_yield": round(div_yield, 4) if div_yield is not None else None,
            "dividend_growth": round(growth, 4) if growth is not None else None,
            "growth_note": growth_note,
            "ex_dividend_date": last_date.isoformat(),
            "last_amount": round(last_amount, 2),
            "payout_ratio": round(payout, 4) if payout is not None else None,
            "payout_note": payout_note,
            "frequency": frequency,
        })

    payers.sort(key=lambda x: (x["dividend_yield"] is None, -(x["dividend_yield"] or 0)))
    non_payers.sort(key=lambda x: x["last_ex_dividend_date"] or "", reverse=True)
    return {"as_of": today.isoformat(), "dividends": payers, "non_payers": non_payers}


_compute_lock = threading.Lock()


def _get_dividends_cached():
    cached = get_cached(psx_cache, "dividends_overview", CACHE_TTL)
    if cached is not None:
        return cached
    with _compute_lock:
        cached = get_cached(psx_cache, "dividends_overview", CACHE_TTL)
        if cached is not None:
            return cached
        result = _compute_dividends()
        set_cached(psx_cache, "dividends_overview", result)
        return result


@router.get("/api/dividends")
async def get_dividends():
    return await run_in_threadpool(_get_dividends_cached)
