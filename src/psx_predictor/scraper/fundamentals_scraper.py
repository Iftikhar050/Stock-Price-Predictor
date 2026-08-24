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
        logger.info(f"Fetching historical fundamentals for {yf_ticker} from Yahoo Finance")
        
        try:
            stock = yf.Ticker(yf_ticker)
            info = stock.info
            
            qf = stock.quarterly_financials
            bs = stock.quarterly_balance_sheet
            cf = stock.quarterly_cash_flow if hasattr(stock, 'quarterly_cash_flow') else stock.quarterly_cashflow
            
            if qf is None or qf.empty:
                logger.warning(f"No quarterly financials found for {ticker}")
                return False
                
            rows = []
            
            # Align dates by taking all unique dates from both
            qf_dates = set(qf.columns) if not qf.empty else set()
            bs_dates = set(bs.columns) if not bs.empty else set()
            cf_dates = set(cf.columns) if not cf.empty else set()
            all_dates = sorted(list(qf_dates.union(bs_dates).union(cf_dates)))
            
            for d in all_dates:
                # Helper to safely extract from dataframe
                def get_val(df, field, date, default=None):
                    if df is not None and not df.empty and field in df.index and date in df.columns:
                        val = df.loc[field, date]
                        if pd.notna(val):
                            return float(val)
                    return default

                # EPS
                eps = get_val(qf, "Basic EPS", d)
                if eps is None:
                    eps = get_val(qf, "Diluted EPS", d)
                if eps is None:
                    # Fallback to Net Income / Shares
                    net_income = get_val(qf, "Net Income", d)
                    shares = get_val(qf, "Basic Average Shares", d) or get_val(bs, "Ordinary Shares Number", d)
                    if net_income is not None and shares and shares > 0:
                        eps = net_income / shares

                # Book Value Per Share
                book_value = None
                equity = get_val(bs, "Stockholders Equity", d) or get_val(bs, "Common Stock Equity", d)
                shares = get_val(bs, "Ordinary Shares Number", d) or get_val(qf, "Basic Average Shares", d)
                if equity is not None and shares and shares > 0:
                    book_value = equity / shares

                # Debt to Equity
                debt_to_equity = None
                total_debt = get_val(bs, "Total Debt", d)
                if total_debt is not None and equity and equity > 0:
                    debt_to_equity = total_debt / equity
                    
                # ROE
                roe = None
                net_income = get_val(qf, "Net Income Common Stockholders", d) or get_val(qf, "Net Income", d)
                if net_income is not None and equity and equity > 0:
                    roe = net_income / equity
                    
                # New Metrics
                revenue = get_val(qf, "Total Revenue", d)
                total_assets = get_val(bs, "Total Assets", d)
                operating_cash_flow = get_val(cf, "Operating Cash Flow", d)
                free_cash_flow = get_val(cf, "Free Cash Flow", d)
                
                # Valuation Extensions
                ebitda = get_val(qf, "EBITDA", d) or get_val(qf, "Normalized EBITDA", d)
                total_cash = get_val(bs, "Cash And Cash Equivalents", d) or get_val(bs, "Cash Financial", d)
                shares_outstanding = get_val(bs, "Ordinary Shares Number", d) or get_val(qf, "Basic Average Shares", d)
                
                # PE Ratio is usually dynamic based on price, but info has current PE.
                # Since we are getting historical, we'll leave it None and let build_features handle it.
                pe_ratio = None
                
                # If this date is the very latest one and info is available, we can optionally 
                # use info dict as a fallback for the most recent period if fields are missing.
                if d == all_dates[-1] and info:
                    if eps is None: eps = info.get("trailingEps") or info.get("forwardEps")
                    if pe_ratio is None: pe_ratio = info.get("trailingPE") or info.get("forwardPE")
                    if roe is None: roe = info.get("returnOnEquity")
                    if debt_to_equity is None: debt_to_equity = info.get("debtToEquity")
                    if book_value is None: book_value = info.get("bookValue")
                    if revenue is None: revenue = info.get("totalRevenue")
                    if net_income is None: net_income = info.get("netIncomeToCommon")
                    if free_cash_flow is None: free_cash_flow = info.get("freeCashflow")
                    if operating_cash_flow is None: operating_cash_flow = info.get("operatingCashflow")
                    if total_assets is None: total_assets = info.get("totalAssets")
                    if total_debt is None: total_debt = info.get("totalDebt")
                    if ebitda is None: ebitda = info.get("ebitda")
                    if total_cash is None: total_cash = info.get("totalCash")
                    if shares_outstanding is None: shares_outstanding = info.get("sharesOutstanding")
                
                if eps is None and pe_ratio is None and roe is None and debt_to_equity is None and book_value is None and revenue is None and net_income is None and ebitda is None:
                    continue
                    
                rows.append({
                    "ticker": ticker.upper(),
                    "report_date": d.date() if hasattr(d, 'date') else d,
                    "eps": float(eps) if eps is not None else None,
                    "pe_ratio": float(pe_ratio) if pe_ratio is not None else None,
                    "roe": float(roe) if roe is not None else None,
                    "debt_to_equity": float(debt_to_equity) if debt_to_equity is not None else None,
                    "book_value_per_share": float(book_value) if book_value is not None else None,
                    "revenue": float(revenue) if revenue is not None else None,
                    "net_income": float(net_income) if net_income is not None else None,
                    "free_cash_flow": float(free_cash_flow) if free_cash_flow is not None else None,
                    "operating_cash_flow": float(operating_cash_flow) if operating_cash_flow is not None else None,
                    "total_assets": float(total_assets) if total_assets is not None else None,
                    "total_debt": float(total_debt) if total_debt is not None else None,
                    "ebitda": float(ebitda) if ebitda is not None else None,
                    "total_cash": float(total_cash) if total_cash is not None else None,
                    "shares_outstanding": float(shares_outstanding) if shares_outstanding is not None else None
                })
                
            if not rows:
                logger.warning(f"No valid fundamental fields found for {ticker}. Skipping insert.")
                return False
                
            df = pd.DataFrame(rows)
            return upsert_stock_fundamentals(df)
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {e}")
            return False
