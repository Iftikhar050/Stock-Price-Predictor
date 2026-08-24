# PSX Data Framework: Status Report

This report evaluates the current state of our `Stock-Price-Predictor` codebase against the requirements outlined in the "PSX Data Framework" document. The PDF organizes features into 10 hierarchical categories and prioritizes 25 key variables.

## Summary of Progress

Our current architecture heavily leans toward **Stock/Technical (Groups 1-2)** and **Valuation/Fundamentals (Groups 3-4)**. We have successfully laid the groundwork for hierarchical features (e.g., mapping oil strictly to energy sectors), but we are missing a significant portion of the **Macroeconomic (Group 6-7)** and **Global (Group 8-9)** datasets.

Out of the 25 top-priority variables identified by the framework, we have **fully implemented 9**, **partially implemented 4**, and are **missing 12**.

---

## Group-by-Group Detailed Audit

### Group 1: Stock (OHLC, returns, volume, volatility)
- **Status:** **Excellent**
- **Done:** OHLC and volume are ingested robustly via `yfinance`. Daily returns and lags (1, 2, 3, 5, 10) are calculated. Rolling realized volatility (10/20-day), Average True Range (ATR), and relative volume (20-day) are actively engineered in `build_features.py`.
- **Missing:** Explicit momentum scalars (e.g., $P_t - P_{t-n}$), though lag returns largely cover this mathematical space.

### Group 2: Technical (RSI, MACD, ATR, momentum)
- **Status:** **Strong**
- **Done:** Core indicators (SMA 7/21/50, RSI-14, MACD, Bollinger Bands, VWAP, and On-Balance Volume) are fully implemented.
- **Missing:** Advanced/niche indicators (Stochastic, Williams %R, CCI, ADX, GARCH). Order-book factors (Bid-ask spread, Market depth) are missing since we rely on EOD data, not live order-book feeds.

### Group 3: Company Fundamentals (EPS, revenue, profit, debt, ROE)
- **Status:** **Excellent**
- **Done:** EPS (trailing & YoY growth), ROE, Debt-to-Equity, and Book Value per Share. Newly added: Revenue, Net Income, Free Cash Flow, Operating Cash Flow, Total Assets, and Total Debt are fully scraped and merged using proper point-in-time logic to avoid lookahead bias.
- **Missing:** EBITDA, Gross/Operating profit (we only grab Net Income currently), and forward guidance. 

### Group 4: Valuation (P/E, P/B, EV/EBITDA, dividend yield)
- **Status:** **Excellent**
- **Done:** Dynamic P/E ratio, Price-to-Book (P/B), Profit Margin, Return on Assets (ROA), EV/EBITDA, EV/Sales, PEG ratio, and Historical valuation percentiles (1-year and 3-year rolling P/E percentiles) are fully implemented. Dividend Yield and "Days Since Dividend" are also fully integrated.
- **Missing:** None of the core valuation metrics from the framework are missing.

### Group 5: Sector (Sector return, sector volume, sector earnings)
- **Status:** **Moderate**
- **Done:** Sector average returns and a 20-day Sector Relative Strength metric are dynamically calculated by grouping peers.
- **Missing:** Sector volume, sector P/E, and sector-wide earnings metrics. Capital rotation tracking is absent.

### Group 6: Pakistan Macro (CPI, GDP, policy rate, KIBOR, M2)
- **Status:** **Weak**
- **Done:** The database schema supports `sbp_policy_rate`.
- **Missing:** The policy rate is currently hardcoded to a 22.0% placeholder in `macro_scraper.py`. We are completely missing inflation metrics (CPI, core inflation), GDP growth, KIBOR, M2 (Money supply), and government bond yields (T-bills/PIB).

### Group 7: FX/External (USD/PKR, reserves, current account, remittances)
- **Status:** **Weak**
- **Done:** USD/PKR exchange rate and daily percent change are successfully scraped via Yahoo Finance (`PKR=X`).
- **Missing:** Foreign exchange reserves, Current account balance, Remittances, and Import/Export data.

### Group 8: Commodities (Brent, coal, gold, cotton, fertilizer)
- **Status:** **Fair (Hierarchical)**
- **Done:** Brent Crude Oil (`BZ=F`) is scraped and mapped exclusively to relevant sectors (Energy, Power, Refinery) as recommended by the framework (Commodity -> Sector -> Earnings).
- **Missing:** We lack data for Coal, Gold, Cotton, Wheat, and Fertilizer prices, which limits predictive power for Cement, Textile, and Fertilizer sectors.

### Group 9: Global Stock Markets & Interest Rates
- **Status:** **Fair**
- **Done:** Global equities (S&P 500, Nasdaq), Global yields (US 10Y), and the Dollar Index (DXY) are now successfully scraped via Yahoo Finance in `macro_scraper.py` and stored in the database.
- **Missing:** Foreign Investor Flows are still absent from the ingestion pipeline.

### Group 10: Events / Sentiment / Psychology
- **Status:** **Moderate**
- **Done:** News sentiment is ingested (`sentiment_score`) and we have implemented a realistic 3-day decay factor to handle missing days.
- **Missing:** The framework emphasizes *surprises* (e.g., Actual EPS vs Expected EPS, CPI surprise) and event shocks (IMF announcements, political events). We currently lack any mechanism to track analyst expectations or discrete event dates.

---

## Audit of the "Top 25 Variables" Priority List

The framework explicitly lists ~25 variables to prioritize. Here is our exact standing:

| # | Variable | Status | Implementation Detail |
|---|---|---|---|
| 1 | KSE-100 return | 🟢 **Done** | `market_return` |
| 2 | Stock return | 🟢 **Done** | `daily_return` |
| 3 | Stock volume | 🟢 **Done** | `volume` |
| 4 | Volume/20-day average | 🟢 **Done** | `relative_volume` |
| 5 | Stock volatility | 🟢 **Done** | `return_vol_20`, `atr` |
| 6 | Sector return | 🟢 **Done** | `sector_return` |
| 7 | Sector volume | 🔴 **Missing** | Not currently grouped |
| 8 | RSI | 🟢 **Done** | `rsi_14` |
| 9 | 5-day momentum | 🟡 **Partial** | Handled via 5-day lag returns |
| 10 | 20-day momentum | 🟡 **Partial** | Handled via `relative_strength_20` |
| 11 | USD/PKR | 🟢 **Done** | `pkr_usd_rate` |
| 12 | USD/PKR change | 🟢 **Done** | `pkr_usd_change_pct` |
| 13 | SBP policy rate | 🟡 **Partial** | Hardcoded placeholder in scraper |
| 14 | KIBOR | 🔴 **Missing** | Not scraped |
| 15 | CPI | 🔴 **Missing** | Not scraped |
| 16 | CPI surprise | 🔴 **Missing** | Not scraped |
| 17 | T-bill yield | 🔴 **Missing** | Not scraped |
| 18 | US 10Y yield | 🟢 **Done** | `us10y_yield` (scraped, pending feature merge) |
| 19 | S&P 500 return | 🟢 **Done** | `sp500_close` (scraped, pending feature merge) |
| 20 | Nasdaq return | 🟢 **Done** | `nasdaq_close` (scraped, pending feature merge) |
| 21 | DXY | 🟢 **Done** | `dxy_close` (scraped, pending feature merge) |
| 22 | Brent crude | 🟢 **Done** | `oil_return_pct` |
| 23 | Foreign investor flow| 🔴 **Missing** | Not scraped (Requires NCCPL/PSX data) |
| 24 | Market breadth | 🔴 **Missing** | Not calculated (Advance/Decline ratio) |
| 25 | EPS/earnings surprise| 🟡 **Partial** | Have YoY growth, missing "surprise" |

## Next Steps Recommended
To align closer with the framework:
1. **Economic Data Integration:** Source KIBOR, CPI, and SBP rates from an official API or State Bank dataset (e.g., DB.nomics or official SBP endpoints) to replace placeholders and missing values.
2. **Sentiment/Event Shocks Expansion:** Track event dates (earnings, budget announcements) or scrape more comprehensive local news sources to build reliable Event metrics.
