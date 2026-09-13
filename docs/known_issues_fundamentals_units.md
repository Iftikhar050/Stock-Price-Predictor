# Known issue: `stock_fundamentals` mixes incompatible units/sources

**Status**: the dominant cause (PSX DPS annual scraper) fixed 2026-09-08. See "Fix applied" below. A smaller, effectively unfixable residual remains from `fundamentals_scraper.py`'s (yfinance) rolling quarterly window - see "Residual issue" below.

## Fix applied (2026-09-08)

Root cause, confirmed by live-scraping `dps.psx.com.pk/company/{PSO,MEBL,FCCL}` and cross-checking their own displayed EPS: PSX's DPS annual-financials table (`tables[4]`, the "Sales"/"Profit after Taxation" rows) does not render at a fixed, predictable order of magnitude. Every ticker checked showed the raw scraped Profit-after-Taxation figure short by ~1000x relative to what `EPS * shares_outstanding` (also scraped from the same page's stats block) implies it should be - and separately, the *same* historical year for the same ticker had scaled differently across different past scrapes (e.g. MEBL 2022 net income was stored as `45,007` from an older scrape vs. `45,006,610` live today - the same real profit, PSX having rendered it at a different precision at different times). `psx_dps_scraper.py`'s `_parse_val` did zero unit conversion, just parsing whatever digit string happened to be on the page.

Fix (`psx_dps_scraper.py`): added `_reconcile_annual_scale()`, which uses `EPS * shares_outstanding` as a live, stable anchor - it derives whatever power-of-10 scale factor is needed to bring the scraped net_income in line with that anchor (tolerating up to 15x drift for genuine share-count growth from bonus issues over the years) and applies the same factor to revenue (same table, same row-set, presumed same rendering convention). Added `_plausibility_gate()` as a second, EPS-independent safety net: nulls revenue if the implied net margin against net_income exceeds +-300%, which catches the handful of historical rows (e.g. PSO revenue values of `17`/`25`/`30`/`100`) that were plain mis-parses, not a scale issue, and have no anchor to recover from.

Verified against PSO/MEBL/FCCL: post-fix annual figures produce plausible, sector-consistent net margins (PSO ~0.2-3.5%, an OMC's typically thin margin; MEBL ~19-21%, a bank's typical markup margin; FCCL ~11-17%, a cement maker's typical margin) - re-scraped live and re-upserted successfully.

## Residual issue (not fixed, likely unfixable via re-scraping)

A handful of quarterly rows sourced from `fundamentals_scraper.py` (yfinance) for 2022-2024 report_dates (e.g. MEBL `2022-09-30`: net_income=`28,596`) are implausibly small and were never corrected by this fix - `_reconcile_annual_scale` only runs inside `psx_dps_scraper.py`. Checked current yfinance output directly (`yf.Ticker('MEBL.KA').quarterly_financials`): it now returns correctly-scaled raw PKR (e.g. ~2.1-2.6e10 net income per quarter, matching what's already correctly stored for the most recent ~5 quarters) - yfinance itself is not currently buggy. But `quarterly_financials` only exposes a rolling ~5-quarter window, so 2022-2024 dates are no longer returned at all; whatever was captured for those dates at some earlier point (evidently at a different, smaller scale) can no longer be re-fetched to correct. This is a data-depth limitation, not an active code bug - consistent with the separately-tracked "frozen fundamentals" finding (real quarterly coverage for most tickers only goes back to when the pipeline first ran, not full ticker history).

## What's wrong

`revenue`/`net_income`/`eps` (and likely other line items) in `stock_fundamentals` are populated by two independent scrapers that don't agree on units, and at least one of them occasionally parses garbage values:

- `src/psx_predictor/scraper/fundamentals_scraper.py` — Yahoo Finance (`yf.Ticker(...).quarterly_financials` etc.), quarterly cadence.
- `src/psx_predictor/scraper/psx_dps_scraper.py` — PSX's own DPS company page (`dps.psx.com.pk/company/{ticker}`), annual cadence, parsed from an HTML table assumed to be in "Rs. millions" per PSX convention (unconfirmed/unverified in code — no unit normalization is applied).

Both write into the same `revenue`/`net_income` columns via `upsert_stock_fundamentals`, keyed by `(ticker, report_date)`. Neither scraper scales its output to a common unit before writing.

## Evidence

Queried directly from the live DB (`SELECT ticker, report_date, revenue, net_income FROM stock_fundamentals WHERE ticker IN ('MEBL','PSO') ORDER BY ticker, report_date`):

- **PSO**: `revenue` values across consecutive report dates include `25`, `3.31e8`, `7.53e8`, `1.11e12`, `30`, `1.20e12`, `17`, `2.45e12`, `100`, `3.39e12`, `100`, `3.57e12`, `8.92e11`. The values of `17`/`25`/`30`/`100` are not real revenue figures for a company PSO's size (its real quarterly revenue is in the hundreds of billions of PKR) - these look like a parser picking up the wrong table cell (possibly a ratio, a row count, or a stray "-" being coerced to a small number) rather than an actual revenue figure.
- **MEBL**: `revenue` sits around `3.1e4`-`4.3e5` for 2016-2024 (consistent with a "Rs. millions" convention: ~31,000-431,000 million = ~31-431 billion PKR, plausible for a bank of MEBL's size), then jumps to `8.1e10` for 2024-12-31 onward (consistent with a raw-PKR convention, ~1000x larger) - both ranges are individually plausible, but mixed in the same column they produce a fake ~1000x "revenue" discontinuity that isn't real.

Reproduced live and confirmed the same category of issue appears on a freshly-added ticker (FCCL) the moment `PsxDpsScraper().scrape_company_financials('FCCL')` was run for the first time on 2026-09-08 - not specific to old data, it's a live, current bug in both scrapers' value handling.

## Why this matters

- `quality_gate.py`'s "frozen data" warnings on `revenue`/`net_income`/`eps_trailing` are partly a symptom of this - the metric counts *value changes*, so a column oscillating between two incompatible unit conventions can look like it has "enough" changes while being meaningless.
- These columns feed `build_features.py`'s master CSVs directly (no unit reconciliation happens downstream), so this propagates into ML training features and into `/api/company/{ticker}/fundamentals`.
- Backfilling more tickers via `psx_dps_scraper.py` (a real, working source - see `docs/pipelines.md`) without fixing this first would spread the unit-mismatch pattern from 2 tickers to potentially all 91 tickers that have any fundamentals coverage.

## What a real fix needs to establish, before writing code

1. **Confirm PSX DPS's actual unit convention** for the annual-statements table (`tables[4]` in `psx_dps_scraper.py`) - likely "Rs. in millions" per the existing assumption in `parse_fundamentals.py`, but not verified against a page footer/header label.
2. **Confirm yfinance's actual unit convention** for `quarterly_financials`/`quarterly_balance_sheet`/`quarterly_cashflow` on PSX (`.KA`) tickers - Yahoo typically reports raw currency units for non-US tickers, but this should be verified against a known-correct figure (e.g. cross-check PSO's real published quarterly revenue for one quarter against what yfinance returns).
3. **Find and fix the source of the `17`/`25`/`30`/`100`-style nonsense values** in whichever scraper produces them - almost certainly a wrong-cell parse in `psx_dps_scraper.py`'s ratio/annual table row-matching (`ann_data.get('Sales', {}).get(yr) or ann_data.get('Mark-up Earned', {}).get(yr)`) picking up a stray value when the expected row label isn't found exactly.
4. Decide a **merge policy** once units are normalized: prefer one source over the other per period, or clearly separate them (e.g. don't merge annual and quarterly into the same `report_date`-keyed row at all).

## Suggested next step

Before writing a fix, spend a short research pass confirming (1) and (2) above against 2-3 tickers with hand-verifiable real figures (PSO and MEBL both publish audited financials publicly), then decide the merge policy in (4) before touching the scrapers.
