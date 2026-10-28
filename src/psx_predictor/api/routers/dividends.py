# dividends.py
# ---------------------------------------------------------
# Market-wide dividend table: trailing-12-month payout, yield, growth,
# latest ex-date, payout ratio and payout frequency per active ticker, plus
# the next expected dividend and recent dividend news.
# ---------------------------------------------------------
import os
import sys
import math
import re
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


# Matching a payout to "the same one last year" allows this much drift.
SEASON_SLACK_DAYS = 45
UPCOMING_DAYS = 30
NEWS_DAYS = 90
NEWS_LIMIT = 150
# Board meeting notices go out about a week ahead; an undated one older than
# this has most likely already been held.
UNDATED_MEETING_DAYS = 10
# A date in a notice's title further out than this isn't the meeting date.
MEETING_LEAD_DAYS = 60


def _estimate_next_payout(paid, per_year, today):
    """Next ex-date projected from the company's own calendar: the payout
    that followed the one a year before the latest, moved forward a year.
    Uses the real seasonal pattern rather than an even 365/n spacing - PSX
    quarterly payers space their final dividend unevenly. None for irregular
    payers or a single payout on record."""
    if per_year is None or len(paid) < 2:
        return None
    last = paid[-1][0]
    threshold = last - timedelta(days=365 - SEASON_SLACK_DAYS)
    for d, _ in paid:
        if d <= threshold:
            continue
        estimate = d + timedelta(days=365)
        while estimate < today - timedelta(days=SEASON_SLACK_DAYS):
            estimate += timedelta(days=365)
        if estimate > last:
            return estimate
    return None


MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}
MONTH_RE = "(" + "|".join(MONTHS) + r"|sept|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
DATE_PATTERNS = [
    # 27 October 2026, 27th October, 2026
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + MONTH_RE + r"\.?,?\s+(\d{4})", re.I), ("d", "m", "y")),
    # October 27, 2026
    (re.compile(MONTH_RE + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", re.I), ("m", "d", "y")),
    # 27-10-2026, 27/10/2026, 27.10.2026
    (re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b"), ("d", "m", "y")),
]


def _month_number(token):
    token = token.lower()
    return MONTHS.get(token) or next((n for name, n in MONTHS.items() if name.startswith(token[:3])), None)


def _dates_in_title(title):
    found = []
    for pattern, order in DATE_PATTERNS:
        for match in pattern.finditer(title):
            parts = dict(zip(order, match.groups()))
            month = int(parts["m"]) if parts["m"].isdigit() else _month_number(parts["m"])
            try:
                found.append(date(int(parts["y"]), month, int(parts["d"])))
            except (TypeError, ValueError):
                continue
    return found


def _meeting_date(title, published):
    """The meeting date written in a board meeting notice's title, if any.
    Titles also carry period-end dates ("quarter ended September 30, 2026")
    and those always precede the notice, so only a date after publication
    counts."""
    dates = [d for d in _dates_in_title(title)
             if published <= d <= published + timedelta(days=MEETING_LEAD_DAYS)]
    return min(dates) if dates else None


MEETING_RE = re.compile(r"board\s*meeting", re.I)
# Meetings that won't consider a dividend, or have already happened.
MEETING_SKIP_RE = re.compile(
    r"other than financial|revoked|results of|decisions? of|in progress|outcome|unclaimed", re.I)


def _upcoming_meetings(notices, today):
    """{symbol: (meeting_date or None, notice_date, url)} for board meetings
    still ahead. Only each company's latest board meeting notice counts: a
    later revocation or "results of the meeting" notice supersedes it."""
    latest = {}
    for n in notices:
        if MEETING_RE.search(n.title):
            prev = latest.get(n.symbol)
            if prev is None or n.published > prev.published:
                latest[n.symbol] = n
    meetings = {}
    for symbol, n in latest.items():
        if MEETING_SKIP_RE.search(n.title):
            continue
        meeting = _meeting_date(n.title, n.published)
        if meeting is not None:
            if meeting >= today:
                meetings[symbol] = (meeting, n.published, n.url)
        elif n.published >= today - timedelta(days=UNDATED_MEETING_DAYS):
            meetings[symbol] = (None, n.published, n.url)
    return meetings


NEWS_KINDS = [
    ("book_closure", re.compile(r"book\s*clos", re.I)),
    ("paid", re.compile(r"\bcredit|disburse|payment of|dispatch|\bpaid\b|\bpays\b", re.I)),
    ("declared", re.compile(r"declar|announce|approve|recommend|propos", re.I)),
]
DIVIDEND_RE = re.compile(r"dividend|book\s*clos|payout", re.I)
NEWS_SKIP_RE = re.compile(r"unclaimed|federal consolidated fund", re.I)
# "Lucky Cement credits ... - Investify": the source repeated as a suffix.
SOURCE_SUFFIX_RE = re.compile(r"\s+[-|]\s+[^-|]{2,40}$")


def _news_kind(title):
    for kind, pattern in NEWS_KINDS:
        if pattern.search(title):
            return kind
    return "news"


def _dividend_news(notices, articles, names):
    """Dividend items from PSX notices and media headlines, newest first.
    News sites re-run the same headline days apart, so a repeat headline for
    a company is dropped whatever its date."""
    items, seen = [], set()
    rows = sorted(notices + articles, key=lambda r: r[1], reverse=True)
    for symbol, published, title, url, source in rows:
        if symbol not in names or not DIVIDEND_RE.search(title) or NEWS_SKIP_RE.search(title):
            continue
        if source and title.endswith(source):
            title = SOURCE_SUFFIX_RE.sub("", title)
        key = (symbol, " ".join(title.lower().split()))
        if key in seen:
            continue
        seen.add(key)
        items.append({
            "symbol": symbol,
            "name": names[symbol],
            "date": published.isoformat(),
            "title": title.strip(),
            "kind": _news_kind(title),
            "source": source or "PSX",
            "url": url,
        })
    items.sort(key=lambda x: x["date"], reverse=True)
    return items[:NEWS_LIMIT]


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
        # PSX notices are stamped at midnight Pakistan time.
        notices = conn.execute(text(
            "SELECT symbol, (published_at AT TIME ZONE 'Asia/Karachi')::date AS published, title, url "
            "FROM corporate_events WHERE published_at >= :since AND title IS NOT NULL"
        ), {"since": today - timedelta(days=NEWS_DAYS)}).fetchall()
        articles = conn.execute(text(
            "SELECT ticker, (published_at AT TIME ZONE 'Asia/Karachi')::date AS published, headline, url, source "
            "FROM stock_news WHERE published_at >= :since AND headline ~* 'dividend|book clos|payout'"
        ), {"since": today - timedelta(days=NEWS_DAYS)}).fetchall()

    payouts_by_ticker = defaultdict(list)
    for r in div_rows:
        payouts_by_ticker[r.ticker].append((r.ex_dividend_date, float(r.dividend_amount)))

    eps_by_ticker = defaultdict(list)
    for r in eps_rows:
        eps_by_ticker[r.ticker].append((r.report_date, float(r.eps) if r.eps is not None else None))

    prices = get_price_snapshot()
    current_cutoff = today - timedelta(days=CURRENT_PAYER_DAYS)
    meetings = _upcoming_meetings(notices, today)
    names = {m.ticker: m.company_name for m in meta}

    payers, non_payers, upcoming = [], [], []
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

        # A dated board meeting beats the estimate: it's where PSX companies
        # declare dividends. Undated notices only go in the upcoming list.
        estimate = _estimate_next_payout(paid, per_year, today)
        meeting_date, notice_date, notice_url = meetings.get(m.ticker, (None, None, None))
        if meeting_date is not None:
            next_date, next_kind = meeting_date, "board_meeting"
        elif estimate is not None:
            next_date, next_kind = estimate, "estimated"
        else:
            next_date, next_kind = None, None

        if notice_date is not None and (meeting_date is None or meeting_date <= today + timedelta(days=UPCOMING_DAYS)):
            upcoming.append({"symbol": m.ticker, "name": m.company_name, "kind": "board_meeting",
                             "date": meeting_date.isoformat() if meeting_date else None,
                             "notice_date": notice_date.isoformat(), "url": notice_url,
                             "last_amount": round(last_amount, 2)})
        if estimate is not None and today <= estimate <= today + timedelta(days=UPCOMING_DAYS) and meeting_date is None:
            upcoming.append({"symbol": m.ticker, "name": m.company_name, "kind": "estimated",
                             "date": estimate.isoformat(), "notice_date": None, "url": None,
                             "last_amount": round(last_amount, 2)})

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
            "next_date": next_date.isoformat() if next_date else None,
            "next_kind": next_kind,
            "meeting_announced": notice_date is not None and meeting_date is None,
        })

    payers.sort(key=lambda x: (x["dividend_yield"] is None, -(x["dividend_yield"] or 0)))
    non_payers.sort(key=lambda x: x["last_ex_dividend_date"] or "", reverse=True)
    # Dated items first by date; undated board meetings after, newest notice first.
    upcoming.sort(key=lambda x: (x["date"] is None, x["date"] or "", x["notice_date"] or ""))
    news = _dividend_news(
        [(n.symbol, n.published, n.title, n.url, None) for n in notices],
        [(a.ticker, a.published, a.headline, a.url, a.source) for a in articles],
        names,
    )
    return {"as_of": today.isoformat(), "dividends": payers, "non_payers": non_payers,
            "upcoming": upcoming, "news": news}


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
