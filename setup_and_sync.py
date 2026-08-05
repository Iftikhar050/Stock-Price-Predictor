import os
import sys

# Ensure the root directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.psx_predictor.db.connection import engine
from src.psx_predictor.db.models import Base
from src.psx_predictor.scraper.client import PSXScraper
from src.psx_predictor.scraper.dividend_scraper import DividendScraper

TICKERS = ['PSO', 'FFC', 'NBP', 'MEBL', 'OGDC', 'LUCK']

def main():
    print("1. Creating Database Tables...")
    try:
        # This will create the stock_eod_data table if it doesn't exist
        Base.metadata.create_all(engine)
        print("   Success: Database tables verified/created.")
    except Exception as e:
        print(f"   ERROR connecting to database. Is PostgreSQL running and credentials correct in .env? Error: {e}")
        sys.exit(1)

    print("\n2. Scraping and Syncing data to PostgreSQL...")
    scraper = PSXScraper()
    
    for ticker in TICKERS:
        print(f"   Syncing {ticker}...")
        success = scraper.sync_ticker(ticker)
        if success:
            print(f"   Success! {ticker} data written to PostgreSQL.")
        else:
            print(f"   ERROR: Failed to sync {ticker}.")

    print("\n   All EOD scraping tasks completed. Data is in 'stock_eod_data' table.")
    
    print("\n3. Scraping and Syncing Dividends to PostgreSQL...")
    div_scraper = DividendScraper()
    for ticker in TICKERS:
        print(f"   Syncing dividends for {ticker}...")
        div_success = div_scraper.sync_dividends(ticker)
        if div_success:
            print(f"   Success! {ticker} dividends written to PostgreSQL.")
        else:
            print(f"   ERROR/WARNING: Failed to sync dividends for {ticker}.")

if __name__ == '__main__':
    main()
