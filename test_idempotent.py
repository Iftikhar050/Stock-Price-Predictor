import sys
import os
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.psx_predictor.scraper.fundamentals_scraper import FundamentalsScraper
from src.psx_predictor.scraper.macro_scraper import MacroScraper
from src.psx_predictor.db.connection import engine

def get_counts():
    with engine.connect() as conn:
        fund_count = conn.execute(text("SELECT COUNT(*) FROM stock_fundamentals")).scalar()
        macro_count = conn.execute(text("SELECT COUNT(*) FROM macro_indicators")).scalar()
    return fund_count, macro_count

print("Initial counts:", get_counts())

macro = MacroScraper()
macro.sync_macro()

fund = FundamentalsScraper()
fund.sync_fundamentals("OGDC")

print("After first run counts:", get_counts())

macro.sync_macro()
fund.sync_fundamentals("OGDC")

print("After second run counts:", get_counts())
