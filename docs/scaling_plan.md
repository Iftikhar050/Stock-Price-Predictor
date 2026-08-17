# Full-Market Scaling Plan

This document outlines the architecture and phased approach for scaling the `Stock-Price-Predictor` from a 6-ticker proof-of-concept to full KSE-100 coverage and beyond.

## Phase 1: Data Foundation (Completed)
- **Goal:** Unblock the ingestion pipeline to support dynamic multi-ticker tracking and market context.
- **Accomplishments:**
  - Expanded schema with `StockMetadata` and `StockMarketIndex` tables.
  - Implemented dynamic global cutoff dates (`choose_global_cutoff`) to prevent temporal leakage across differently-aged listings.
  - Migrated hardcoded `TICKERS` lists to dynamic DB-driven `get_active_tickers()` lookups.
  - Re-architected model ingestion (XGBoost & LSTM) to support native categorical/embedding representations of ticker and sector.
  - Proved validity through ablation studies showing Top-4 feature importance for the new market-context variables (Market Return, Sector Return, Dividend Yield).
  - **Decision Log (Market Index):** The current synthetic KSE-100 proxy is a stopgap due to anti-scraping on the PSX DPS. The real fix to be implemented is pulling the official `^KSE` index from Yahoo Finance (or Investing.com if Yahoo lacks full historical depth) to provide truly exogenous market signal without self-referential ticker leakage.

## Phase 2: Live Scraper Refactoring (Completed)
- **Accomplishments:**
  - Integrated `yfinance` to bypass the rigid bot-detection blockers on the official PSX DPS site.
  - Successfully scraped robust historical OHLCV data for all KSE-100 tickers seamlessly.
- **Next Steps:**
  - **Option A (Preferred):** Integrate a headless browser scraping strategy (e.g., Selenium or Playwright) into `scraper.client` to bypass simple bot-detection headers.
  - **Option B (Alternative):** Source a reliable third-party API (e.g., Yahoo Finance, Investing.com) that covers Pakistani equities for historical OHLCV data.
  - Ensure scraper maintains current retry logic, random backoffs, and per-ticker isolation to prevent pipeline halts.

## Phase 3: Hardware Scaling & Serving (Completed)
- **Accomplishments:**
  - **Compute Scaling:** Analyzed standard pandas vectorized feature engineering. It successfully handles 30+ tickers and ~15 years of data in just 25 seconds, eliminating the need for Dask/PySpark overhead.
  - **Inference Pipeline:** Updated FastAPI endpoints to query the dynamic `active_tickers` database table on boot.
  - **Orchestration:** Implemented a master `run_pipeline.py` script. Powered by Python's `schedule`, this script automates the daily sequence: Scrape -> Feature Eng -> Train -> API Hot-Reload without any human intervention.
