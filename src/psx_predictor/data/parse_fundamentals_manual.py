import os
import sys
import logging
import logging
import pandas as pd
import re
from pathlib import Path
import pdfplumber

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.psx_predictor.utils.manual_data_loader import inspect_and_log_all
from src.psx_predictor.db.connection import engine
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ParseFundamentals")

def fuzzy_match_metric(actual_col: str, possible_names: list) -> bool:
    actual_col = str(actual_col).strip().lower()
    for name in possible_names:
        if name.lower() in actual_col:
            return True
    return False

def is_year_like(val) -> bool:
    val = str(val).strip()
    # E.g. "2023", "23", "CY23", "FY23", "2023-12-31"
    if val.isdigit() and len(val) == 4 and val.startswith("20"):
        return True
    if 'fy' in val.lower() or 'cy' in val.lower():
        return True
    if '-' in val and len(val.split('-')) == 3:
        return True # Looks like date
    return False

def parse_fundamentals_manual(source: str):
    """
    source can be "fundamentals_pso" or "fundamentals_mebl"
    """
    files = inspect_and_log_all(source)
    ticker = source.split('_')[-1].upper()
    all_records = []
    
    metric_map = {
        "revenue": ["profit / return earned", "revenue", "gross sales", "net sales", "turnover"],
        "net_income": ["profit after taxation", "net profit", "profit for the period", "profit after tax", "profit for the year"],
        "eps": ["basic earnings per share", "earnings per share", "basic"],
        "total_assets": ["total assets"],
        "total_debt": ["total liabilities", "deposits and other accounts", "borrowings"],
        "gross_profit": ["gross profit", "net spread earned"]
    }
    
    for file in files:
        logger.info(f"Processing Fundamentals file: {file.name}")
        ext = file.suffix.lower()
        
        if ext in ['.csv', '.xlsx', '.xls']:
            try:
                df = pd.read_csv(file, header=None) if ext == '.csv' else pd.read_excel(file, header=None)
                if df.empty:
                    continue
                    
                # Detect orientation
                # Sniff first row (skipping first column which is usually the metric label in wide format)
                first_row_vals = df.iloc[0, 1:].dropna().values
                is_wide_format = any(is_year_like(v) for v in first_row_vals)
                
                # Try sniffing first column for years
                first_col_vals = df.iloc[1:, 0].dropna().values
                is_long_format = any(is_year_like(v) for v in first_col_vals)
                
                logger.info(f"Orientation Detection for {file.name}: is_wide={is_wide_format}, is_long={is_long_format}")
                
                if is_wide_format and not is_long_format:
                    logger.info(f"File {file.name} is WIDE format (years as columns, metrics as rows)")
                    # The first row contains the dates/years. The first column contains the metrics.
                    dates = df.iloc[0, 1:].values
                    
                    for r_idx in range(1, len(df)):
                        metric_label = df.iloc[r_idx, 0]
                        if pd.isna(metric_label):
                            continue
                            
                        # Check if this metric matches anything we want
                        matched_target = None
                        for target, aliases in metric_map.items():
                            if fuzzy_match_metric(metric_label, aliases):
                                matched_target = target
                                logger.info(f"Matched row label '{metric_label}' to target '{target}'")
                                break
                                
                        if matched_target:
                            # Extract values for all dates
                            for c_idx, date_val in enumerate(dates):
                                if pd.isna(date_val):
                                    continue
                                val = df.iloc[r_idx, c_idx + 1]
                                try:
                                    val = float(str(val).replace(',', '').strip())
                                except (ValueError, TypeError):
                                    val = pd.NA
                                    
                                # We need to normalize date_val to an actual date
                                # Assume Dec 31 if it's just a year
                                date_str = str(date_val).strip()
                                if date_str.isdigit() and len(date_str) == 4:
                                    date_obj = pd.to_datetime(f"{date_str}-12-31").date()
                                else:
                                    try:
                                        date_obj = pd.to_datetime(date_str).date()
                                    except:
                                        continue
                                
                                all_records.append({
                                    "ticker": ticker,
                                    "report_date": date_obj,
                                    "metric": matched_target,
                                    "value": val
                                })
                                
                elif is_long_format or (not is_wide_format and not is_long_format):
                    # Assume long format (dates as rows, metrics as columns)
                    # Use row 0 as header
                    logger.info(f"File {file.name} is LONG format (dates as rows, metrics as columns)")
                    df.columns = df.iloc[0]
                    df = df[1:].reset_index(drop=True)
                    
                    date_col = None
                    for col in df.columns:
                        if fuzzy_match_metric(col, ["date", "period", "year"]):
                            date_col = col
                            break
                            
                    if not date_col:
                        logger.warning(f"Could not find date column in {file.name}")
                        continue
                        
                    matched_cols = {}
                    for target, aliases in metric_map.items():
                        for col in df.columns:
                            if col == date_col: continue
                            if fuzzy_match_metric(col, aliases):
                                matched_cols[target] = col
                                logger.info(f"Matched column label '{col}' to target '{target}'")
                                break
                                
                    for idx, row in df.iterrows():
                        date_str = str(row[date_col]).strip()
                        if date_str.isdigit() and len(date_str) == 4:
                            date_obj = pd.to_datetime(f"{date_str}-12-31").date()
                        else:
                            try:
                                date_obj = pd.to_datetime(date_str).date()
                            except:
                                continue
                                
                        for target, orig_col in matched_cols.items():
                            val = row[orig_col]
                            try:
                                val = float(str(val).replace(',', '').strip())
                            except (ValueError, TypeError):
                                val = pd.NA
                            
                            if pd.notna(val):
                                all_records.append({
                                    "ticker": ticker,
                                    "report_date": date_obj,
                                    "metric": target,
                                    "value": val
                                })
                                
            except Exception as e:
                logger.error(f"Error parsing fundamentals file {file.name}: {e}")
                
        elif ext == '.pdf':
            try:
                date_str = file.stem.split('_')[1]
                if date_str.isdigit() and len(date_str) == 4:
                    report_date = pd.to_datetime(f"{date_str}-12-31").date()
                else:
                    try:
                        report_date = pd.to_datetime(date_str).date()
                    except:
                        continue
                        
                found_targets = set()
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if not text:
                            continue
                            
                        # Heuristic: only process pages that are likely financial statements
                        text_lower = text.lower()
                        is_financial_statement = False
                        if any(kw in text_lower for kw in ['statement of financial position', 'balance sheet', 'statement of profit or loss', 'statement of profit and loss', 'income statement', 'profit and loss account']):
                            is_financial_statement = True
                            
                        if not is_financial_statement:
                            continue
                            
                        lines = text.split('\n')
                        for line in lines:
                            line_lower = line.lower().strip()
                            for target, aliases in metric_map.items():
                                if target in found_targets:
                                    continue
                                    
                                for alias in aliases:
                                    if line_lower.startswith(alias):
                                        # Extract numbers
                                        # handle parentheses as negative
                                        cleaned_line = line.replace('(', '-').replace(')', '')
                                        numbers = re.findall(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?', cleaned_line)
                                        if numbers:
                                            vals = []
                                            for n in numbers:
                                                try:
                                                    vals.append(float(n.replace(',', '')))
                                                except:
                                                    pass
                                                    
                                            if not vals:
                                                continue
                                                
                                            # For P&L with quarter/half-year columns, the current period cumulative is often the 3rd column
                                            # If 4 numbers, assume it's [Q_curr, Q_prev, HY_curr, HY_prev] -> take HY_curr (index 2)
                                            # If 2 numbers, assume it's [Curr, Prev] -> take Curr (index 0)
                                            if len(vals) == 4:
                                                final_val = vals[2]
                                            else:
                                                final_val = vals[0]
                                                
                                            all_records.append({
                                                "ticker": ticker,
                                                "report_date": report_date,
                                                "metric": target,
                                                "value": final_val
                                            })
                                            found_targets.add(target)
                                            logger.info(f"PDF {file.name} - Extracted {target} = {final_val}")
                                            break
                                            
                        if len(found_targets) == len(metric_map):
                            break # Found everything we need
            except Exception as e:
                logger.error(f"Error parsing PDF file {file.name}: {e}")
        else:
            logger.warning(f"Unsupported file type for Fundamentals: {file.name}")
            
    if all_records:
        df_out = pd.DataFrame(all_records)
        # Pivot so metrics become columns
        df_pivot = df_out.pivot_table(index=['ticker', 'report_date'], columns='metric', values='value', aggfunc='first').reset_index()
        # Add missing target columns
        for target in metric_map.keys():
            if target not in df_pivot.columns:
                df_pivot[target] = pd.NA
        return df_pivot
    else:
        logger.warning(f"No valid Fundamentals data extracted for {source}.")
        cols = ["ticker", "report_date"] + list(metric_map.keys())
        return pd.DataFrame(columns=cols)

# Currency-scale fields this parser extracts via regex/fuzzy line-matching from
# PDFs/CSVs with no unit label check (unlike psx_dps_scraper.py's
# _reconcile_annual_scale/_plausibility_gate, which anchor against EPS x shares).
# Confirmed 2026-09: this path wrote MEBL total_debt=17.0 identically across 9
# separate quarters and similar single/double-digit garbage for MEBL/PSO
# total_assets/revenue/net_income - almost certainly a ratio or wrong table row
# mis-mapped into a currency column, not a real value at any unit scale. A large,
# PSX-listed bank/OMC's quarterly revenue/net_income/total_assets/total_debt is
# never below this floor in raw PKR, PKR '000s, or PKR millions - reject rather
# than let it silently corrupt history.
_CURRENCY_FIELDS = ["revenue", "net_income", "total_assets", "total_debt", "gross_profit"]
_MIN_PLAUSIBLE_CURRENCY_VALUE = 1000.0


def upsert_fundamentals(df: pd.DataFrame):
    if df.empty: return
    # Rename columns if needed
    if 'eps' not in df.columns and 'earnings per share' in df.columns:
        df = df.rename(columns={'earnings per share': 'eps'})

    for col in _CURRENCY_FIELDS:
        if col not in df.columns:
            continue
        implausible = df[col].notna() & (df[col].abs() < _MIN_PLAUSIBLE_CURRENCY_VALUE)
        if implausible.any():
            logger.warning(
                f"Rejecting {implausible.sum()} implausible {col} value(s) "
                f"(abs < {_MIN_PLAUSIBLE_CURRENCY_VALUE:.0f}): "
                f"{df.loc[implausible, ['ticker', 'report_date', col]].to_dict('records')}"
            )
            df.loc[implausible, col] = pd.NA

    with engine.connect() as conn:
        for idx, row in df.iterrows():
            if pd.isna(row['report_date']): continue
            # Dynamically build upsert query for the columns we have
            cols = [c for c in df.columns if pd.notna(row[c])]
            vals = {c: row[c] for c in cols}
            
            insert_cols = ", ".join(cols) + ", created_at"
            placeholders = ", ".join([f":{c}" for c in cols]) + ", CURRENT_TIMESTAMP"
            update_clause = ", ".join([f"{c} = EXCLUDED.{c}" for c in cols if c not in ['ticker', 'report_date']])
            
            if not update_clause:
                continue
                
            query = f"""
                INSERT INTO stock_fundamentals ({insert_cols})
                VALUES ({placeholders})
                ON CONFLICT (ticker, report_date)
                DO UPDATE SET {update_clause};
            """
            conn.execute(text(query), vals)
        conn.commit()
    logger.info(f"Upserted {len(df)} fundamental records to database.")

if __name__ == "__main__":
    df_pso = parse_fundamentals_manual("fundamentals_pso")
    print(f"\n--- PSO FINAL DATAFRAME HEAD ---")
    if not df_pso.empty:
        print(df_pso.head())
        upsert_fundamentals(df_pso)
        
    df_mebl = parse_fundamentals_manual("fundamentals_mebl")
    print(f"\n--- MEBL FINAL DATAFRAME HEAD ---")
    if not df_mebl.empty:
        print(df_mebl.head())
        upsert_fundamentals(df_mebl)
