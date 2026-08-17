import os
import sys

# Ensure the root directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.psx_predictor.db.connection import engine
from src.psx_predictor.db.models import Base
from src.psx_predictor.scraper.client import PSXScraper
from src.psx_predictor.scraper.dividend_scraper import DividendScraper
from src.psx_predictor.scraper.fundamentals_scraper import FundamentalsScraper
from src.psx_predictor.scraper.macro_scraper import MacroScraper

from src.psx_predictor.db.repository import get_active_tickers
from src.psx_predictor.scraper.index_scraper import sync_market_index
import time
def main():
    print("1. Creating Database Tables...")
    try:
        # This will create the stock_eod_data table if it doesn't exist
        Base.metadata.create_all(engine)
        print("   Success: Database tables verified/created.")
    except Exception as e:
        print(f"   ERROR connecting to database. Is PostgreSQL running and credentials correct in .env? Error: {e}")
        sys.exit(1)

    print("\n2. Scraping and Syncing market index...")
    idx_success = sync_market_index()
    if idx_success:
        print("   Success! Market index data synchronized.")
    else:
        print("   ERROR: Failed to sync market index.")

    print("\n2.5 Scraping and Syncing Macro Indicators...")
    macro_scraper = MacroScraper()
    macro_success = macro_scraper.sync_macro()
    if macro_success:
        print("   Success! Macro indicators synchronized.")
    else:
        print("   WARNING: Failed to sync macro indicators.")

    print("\n3. Scraping and Syncing data to PostgreSQL...")
    scraper = PSXScraper()
    active_tickers = get_active_tickers()
    
    eod_failures = []
    for ticker in active_tickers:
        print(f"   Syncing {ticker}...")
        try:
            success = scraper.sync_ticker(ticker)
            if success:
                print(f"   Success! {ticker} data written.")
            else:
                print(f"   WARNING: Failed to sync {ticker}.")
                eod_failures.append(ticker)
        except Exception as e:
            print(f"   ERROR: Exception syncing {ticker}: {e}")
            eod_failures.append(ticker)
        time.sleep(1) # Rate limiting

    if eod_failures:
        print(f"\n   EOD scraping completed with {len(eod_failures)} failures: {eod_failures}")
    else:
        print(f"\n   All EOD scraping tasks completed successfully.")
    print("\n4. Scraping and Syncing Dividends to PostgreSQL...")
    div_scraper = DividendScraper()
    div_failures = []
    for ticker in active_tickers:
        print(f"   Syncing dividends for {ticker}...")
        try:
            div_success = div_scraper.sync_dividends(ticker)
            if div_success:
                print(f"   Success! {ticker} dividends written.")
            else:
                print(f"   WARNING: Failed to sync dividends for {ticker}.")
                div_failures.append(ticker)
        except Exception as e:
            print(f"   ERROR: Exception syncing dividends for {ticker}: {e}")
            div_failures.append(ticker)
        time.sleep(1) # Rate limiting
        
    if div_failures:
        print(f"\n   Dividend scraping completed with {len(div_failures)} failures: {div_failures}")
    else:
        print(f"\n   All Dividend scraping tasks completed successfully.")

    print("\n5. Scraping and Syncing Fundamentals to PostgreSQL...")
    fund_scraper = FundamentalsScraper()
    fund_failures = []
    for ticker in active_tickers:
        print(f"   Syncing fundamentals for {ticker}...")
        try:
            fund_success = fund_scraper.sync_fundamentals(ticker)
            if fund_success:
                print(f"   Success! {ticker} fundamentals written.")
            else:
                print(f"   WARNING: Failed to sync fundamentals for {ticker} (or missing data).")
                fund_failures.append(ticker)
        except Exception as e:
            print(f"   ERROR: Exception syncing fundamentals for {ticker}: {e}")
            fund_failures.append(ticker)
        time.sleep(1) # Rate limiting
        
    if fund_failures:
        print(f"\n   Fundamentals scraping completed with {len(fund_failures)} failures: {fund_failures}")
    else:
        print(f"\n   All Fundamentals scraping tasks completed successfully.")

if __name__ == '__main__':
    main()
