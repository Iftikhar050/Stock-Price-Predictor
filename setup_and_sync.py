import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.psx_predictor.db.connection import engine
from src.psx_predictor.db.models import Base
from src.psx_predictor.scraper.client import PSXScraper
from src.psx_predictor.scraper.dividend_scraper import DividendScraper
from src.psx_predictor.scraper.fundamentals_scraper import FundamentalsScraper
from src.psx_predictor.scraper.macro_scraper import MacroScraper

from src.psx_predictor.db.repository import get_active_tickers
from src.psx_predictor.scraper.index_scraper import sync_market_index

def execute_with_backoff(func, *args, max_retries=3, initial_delay=2, **kwargs):
    """Executes a function with adaptive exponential backoff."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            if result:
                return True
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"   [!] Final attempt failed: {e}")
            else:
                print(f"   [!] Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
        
        if attempt < max_retries - 1:
            time.sleep(delay)
            delay *= 2
            
    return False

def sync_category(category_name, tickers, sync_function, delay=1):
    print(f"\nScraping and Syncing {category_name} to PostgreSQL...")
    failures = []
    successes = []
    
    for ticker in tickers:
        print(f"   Syncing {category_name} for {ticker}...")
        success = execute_with_backoff(sync_function, ticker)
        if success:
            print(f"   Success! {ticker} {category_name.lower()} written.")
            successes.append(ticker)
        else:
            print(f"   WARNING: Failed to sync {category_name.lower()} for {ticker}.")
            failures.append(ticker)
        time.sleep(delay)
        
    print(f"\n--- {category_name} Summary ---")
    print(f"Successes: {len(successes)}")
    print(f"Failures: {len(failures)}")
    if failures:
        print(f"Failed Tickers: {failures}")
    return successes, failures

def main():
    print("1. Creating Database Tables...")
    try:
        Base.metadata.create_all(engine)
        print("   Success: Database tables verified/created.")
    except Exception as e:
        print(f"   ERROR connecting to database. Is PostgreSQL running and credentials correct in .env? Error: {e}")
        sys.exit(1)

    print("\n2. Scraping and Syncing market index...")
    if sync_market_index():
        print("   Success! Market index data synchronized.")
    else:
        print("   ERROR: Failed to sync market index.")

    print("\n3. Scraping and Syncing Macro Indicators...")
    macro_scraper = MacroScraper()
    if macro_scraper.sync_macro():
        print("   Success! Macro indicators synchronized.")
    else:
        print("   WARNING: Failed to sync macro indicators.")

    active_tickers = get_active_tickers()
    print(f"\nLoaded {len(active_tickers)} active tickers from database.")
    
    # EOD Data
    scraper = PSXScraper()
    eod_succ, eod_fail = sync_category("EOD Data", active_tickers, scraper.sync_ticker)
    
    # Dividends
    div_scraper = DividendScraper()
    div_succ, div_fail = sync_category("Dividends", active_tickers, div_scraper.sync_dividends)
    
    # Fundamentals
    fund_scraper = FundamentalsScraper()
    fund_succ, fund_fail = sync_category("Fundamentals", active_tickers, fund_scraper.sync_fundamentals)
    
    print("\n=======================================================")
    print("               COMPLETENESS REPORT                     ")
    print("=======================================================")
    print(f"Total Tickers Target: {len(active_tickers)}")
    print(f"EOD Completed:        {len(eod_succ)} / {len(active_tickers)}")
    print(f"Dividends Completed:  {len(div_succ)} / {len(active_tickers)}")
    print(f"Fundamentals Comp:    {len(fund_succ)} / {len(active_tickers)}")
    print("=======================================================\n")

if __name__ == '__main__':
    main()
