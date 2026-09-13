import os
import sys
import logging
import pdfplumber
import pandas as pd
from pathlib import Path
import re
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.psx_predictor.utils.manual_data_loader import inspect_and_log_all
from src.psx_predictor.db.repository import upsert_macro_indicators

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ParseSbpKibor")

def fuzzy_match_kibor_column(actual_col: str, possible_names: list) -> bool:
    actual_col = str(actual_col).strip().lower()
    for name in possible_names:
        if name.lower() in actual_col:
            return True
    return False

def parse_sbp_kibor():
    files = inspect_and_log_all("sbp_kibor")
    records = []
    
    for file in files:
        logger.info(f"Processing {file.name}")
        ext = file.suffix.lower()
        
        if ext in ['.csv', '.xlsx', '.xls']:
            try:
                df = pd.read_csv(file) if ext == '.csv' else pd.read_excel(file)
                
                # Match date column
                date_col = None
                for col in df.columns:
                    if fuzzy_match_kibor_column(col, ["date", "period", "month"]):
                        date_col = col
                        break
                        
                # Match rate columns
                tenors = {
                    "kibor_1w": ["1 week", "1w", "1-week"],
                    "kibor_1m": ["1 month", "1m", "1-month"],
                    "kibor_3m": ["3 month", "3m", "3-month", "3 - month"],
                    "kibor_6m": ["6 month", "6m", "6-month", "6 - month"],
                    "kibor_1y": ["1 year", "1y", "1-year", "1 - year"]
                }
                
                matched_cols = {}
                for standard_name, aliases in tenors.items():
                    for col in df.columns:
                        if fuzzy_match_kibor_column(col, aliases):
                            matched_cols[standard_name] = col
                            logger.info(f"Matched file column '{col}' to standard column '{standard_name}'")
                            break
                            
                if not date_col:
                    logger.warning(f"Could not find a date column in {file.name}. Actual columns: {list(df.columns)}")
                    continue
                    
                for idx, row in df.iterrows():
                    rec = {"date": pd.to_datetime(row[date_col]).date()}
                    for std_name, orig_col in matched_cols.items():
                        try:
                            rec[std_name] = float(row[orig_col])
                        except (ValueError, TypeError):
                            rec[std_name] = pd.NA
                            
                    records.append(rec)
                    
            except Exception as e:
                logger.error(f"Error parsing tabular file {file.name}: {e}")
                
        elif ext == '.pdf':
            try:
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text and "KIBOR" in text:
                            # Our previous script parsed this exact layout reliably
                            lines = text.split('\n')
                            date_obj = None
                            for line in lines[:5]:
                                m = re.search(r'(\d{1,2}-[A-Za-z]{3}-\d{2,4})', line)
                                if m:
                                    date_str = m.group(1)
                                    if len(date_str.split('-')[2]) == 2:
                                        date_obj = datetime.strptime(date_str, "%d-%b-%y").date()
                                    else:
                                        date_obj = datetime.strptime(date_str, "%d-%b-%Y").date()
                                    break
                                    
                            if not date_obj:
                                # Fallback to filename parsing
                                m = re.search(r'(\d{1,2}-[a-zA-Z]{3}-\d{2,4})', file.name, re.IGNORECASE)
                                if m:
                                    date_str = m.group(1)
                                    if len(date_str.split('-')[2]) == 2:
                                        date_obj = datetime.strptime(date_str, "%d-%b-%y").date()
                                    else:
                                        date_obj = datetime.strptime(date_str, "%d-%b-%Y").date()
                            
                            if not date_obj:
                                logger.warning(f"Could not extract date from PDF {file.name}")
                                continue
                                
                            rec = {"date": date_obj, "kibor_1w": pd.NA, "kibor_1m": pd.NA, 
                                   "kibor_3m": pd.NA, "kibor_6m": pd.NA, "kibor_1y": pd.NA}
                            
                            found_any = False
                            for line in lines:
                                parts = line.split()
                                if not parts:
                                    continue
                                line_lower = line.lower()
                                
                                try:
                                    if "1 - week" in line_lower or "1-week" in line_lower:
                                        rec["kibor_1w"] = float(parts[-1])
                                        found_any = True
                                    elif "1 - month" in line_lower or "1-month" in line_lower:
                                        rec["kibor_1m"] = float(parts[-1])
                                        found_any = True
                                    elif "3 - month" in line_lower or "3-month" in line_lower:
                                        rec["kibor_3m"] = float(parts[-1])
                                        found_any = True
                                    elif "6 - month" in line_lower or "6-month" in line_lower:
                                        rec["kibor_6m"] = float(parts[-1])
                                        found_any = True
                                    elif "1 - year" in line_lower or "1-year" in line_lower:
                                        rec["kibor_1y"] = float(parts[-1])
                                        found_any = True
                                except ValueError:
                                    pass
                                    
                            if found_any:
                                records.append(rec)
                                
            except Exception as e:
                logger.error(f"Error parsing PDF {file.name}: {e}")
                
    if records:
        df_out = pd.DataFrame(records)
        # Standardize columns (add missing tenors as NaN)
        standard_cols = ["date", "kibor_1w", "kibor_1m", "kibor_3m", "kibor_6m", "kibor_1y"]
        for col in standard_cols:
            if col not in df_out.columns:
                df_out[col] = pd.NA
        
        df_out = df_out[standard_cols].sort_values('date').drop_duplicates(subset=['date'], keep='last')
        
        success = upsert_macro_indicators(df_out)
        if success:
            logger.info(f"Successfully upserted {len(df_out)} manual KIBOR records.")
        else:
            logger.error("Failed to upsert manual KIBOR records.")
            
        return df_out
    else:
        logger.warning("No valid KIBOR data extracted from manual files.")
        standard_cols = ["date", "kibor_1w", "kibor_1m", "kibor_3m", "kibor_6m", "kibor_1y"]
        return pd.DataFrame(columns=standard_cols)

if __name__ == "__main__":
    df = parse_sbp_kibor()
    print("\n--- FINAL DATAFRAME HEAD ---")
    print(df.head())
    print("\n--- NULL PERCENTAGE ---")
    print(df.isna().mean())
