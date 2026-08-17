import logging
import pandas as pd
import yfinance as yf
from datetime import datetime
from src.psx_predictor.db.repository import upsert_stock_fundamentals

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

class FundamentalsScraper:
    def sync_fundamentals(self, ticker: str) -> bool:
        yf_ticker = f"{ticker.upper()}.KA"
        logger.info(f"Fetching fundamentals for {yf_ticker} from Yahoo Finance")
        
        try:
            stock = yf.Ticker(yf_ticker)
            info = stock.info
            
            if not info:
                logger.warning(f"No fundamentals info found for {ticker}")
                return False
                
            # Attempt to get latest report date from quarterly financials
            qf = stock.quarterly_financials
            if not qf.empty:
                report_date = qf.columns[0].date()
            else:
                report_date = datetime.now().date()

            # Safely extract metrics
            eps = info.get("trailingEps") or info.get("forwardEps")
            pe_ratio = info.get("trailingPE") or info.get("forwardPE")
            roe = info.get("returnOnEquity")
            debt_to_equity = info.get("debtToEquity")
            book_value = info.get("bookValue")
            
            # If all are None, don't insert
            if eps is None and pe_ratio is None and roe is None and debt_to_equity is None and book_value is None:
                logger.warning(f"All fundamental fields empty for {ticker}. Skipping insert.")
                return False
            
            df = pd.DataFrame([{
                "ticker": ticker.upper(),
                "report_date": report_date,
                "eps": float(eps) if eps is not None else None,
                "pe_ratio": float(pe_ratio) if pe_ratio is not None else None,
                "roe": float(roe) if roe is not None else None,
                "debt_to_equity": float(debt_to_equity) if debt_to_equity is not None else None,
                "book_value_per_share": float(book_value) if book_value is not None else None
            }])
            
            return upsert_stock_fundamentals(df)
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {e}")
            return False
