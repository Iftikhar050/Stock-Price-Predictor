import os
import sys

# Ensure the root directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.psx_predictor.db.connection import engine
from src.psx_predictor.db.models import Base
from src.psx_predictor.scraper.client import PSXScraper

def main():
    print("1. Creating Database Tables...")
    try:
        # This will create the stock_eod_data table if it doesn't exist
        Base.metadata.create_all(engine)
        print("   Success: Database tables verified/created.")
    except Exception as e:
        print(f"   ERROR connecting to database. Is PostgreSQL running and credentials correct in .env? Error: {e}")
        sys.exit(1)

    print("\n2. Scraping and Syncing PSO data to PostgreSQL...")
    scraper = PSXScraper()
    success = scraper.sync_ticker("PSO")
    
    if success:
        print("   Success! Data has been written to the PostgreSQL database.")
        print("   You should now see the data in your 'stock_eod_data' table.")
    else:
        print("   ERROR: Failed to sync data to PostgreSQL.")

if __name__ == '__main__':
    main()
