import os
import sys
import re
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.psx_predictor.utils.manual_data_loader import inspect_and_log_all

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ParseNccpl")

def fuzzy_match_column(actual_col: str, possible_names: list) -> bool:
    actual_col = str(actual_col).strip().lower()
    for name in possible_names:
        if name.lower() in actual_col:
            return True
    return False

def parse_financial_number(val):
    if pd.isna(val): return 0.0
    val = str(val).strip().replace(',', '')
    if val.startswith('(') and val.endswith(')'):
        try:
            return -float(val[1:-1])
        except ValueError:
            return 0.0
    try:
        return float(val)
    except ValueError:
        return 0.0

def parse_nccpl_fipi_lipi():
    files = inspect_and_log_all("nccpl_fipi_lipi")
    
    category_map = {
        "lipi_mutual_funds_net": ["mutual fund", "mutual funds"],
        "lipi_insurance_net": ["insurance"],
        "fipi_foreign_corporate_net": ["foreign corporate", "fipi corporate", "foreign company"],
        "fipi_foreign_individual_net": ["foreign individual", "fipi individual"],
        "fipi_overseas_pakistani_net": ["overseas pakistani"],
        "lipi_broker_net": ["broker proprietary", "broker prop", "broker"],
        "lipi_companies_net": ["company", "companies", "corporate", "corporates"],
        "lipi_banks_net": ["bank", "banks", "dfi"],
        "lipi_nbfc_net": ["nbfc"],
        "lipi_individuals_net": ["individual", "individuals"]
    }
    
    # We will aggregate records by date
    records_by_date = {}

    for file in files:
        logger.info(f"Processing NCCPL file: {file.name}")
        ext = file.suffix.lower()
        
        if ext in ['.csv', '.xlsx', '.xls']:
            try:
                df = pd.read_csv(file) if ext == '.csv' else pd.read_excel(file)
                
                # Extract date from filename, else mtime
                match = re.search(r'(\d{4}-\d{2}-\d{2})', file.name)
                if match:
                    file_date = pd.to_datetime(match.group(1)).date()
                else:
                    file_date = datetime.fromtimestamp(file.stat().st_mtime).date()
                    
                if file_date not in records_by_date:
                    records_by_date[file_date] = {"date": file_date}
                    
                rec = records_by_date[file_date]

                # If the df has CLIENT TYPE and MARKET TYPE, it's the standard raw NCCPL format
                if 'CLIENT TYPE' in df.columns and 'MARKET TYPE' in df.columns and 'USD' in df.columns:
                    df['CLIENT TYPE'] = df['CLIENT TYPE'].ffill()
                    totals = df[df['MARKET TYPE'].astype(str).str.upper().str.strip() == 'TOTAL']
                    
                    for _, row in totals.iterrows():
                        client_type = str(row['CLIENT TYPE'])
                        val_usd = parse_financial_number(row['USD'])
                        
                        # Match to target
                        for target, aliases in category_map.items():
                            if fuzzy_match_column(client_type, aliases):
                                rec[target] = val_usd
                                break
                else:
                    # Previous horizontal format parsing logic
                    date_col = None
                    for col in df.columns:
                        if fuzzy_match_column(col, ["date", "period"]):
                            date_col = col
                            break
                            
                    if date_col:
                        # Match categories
                        matched_cols = {}
                        for target, aliases in category_map.items():
                            for col in df.columns:
                                if col == date_col: continue
                                if fuzzy_match_column(col, aliases):
                                    matched_cols[target] = col
                                    break
                                    
                        for _, row in df.iterrows():
                            r_date = pd.to_datetime(row[date_col]).date()
                            if r_date not in records_by_date:
                                records_by_date[r_date] = {"date": r_date}
                            
                            r_rec = records_by_date[r_date]
                            for target, orig_col in matched_cols.items():
                                r_rec[target] = parse_financial_number(row[orig_col])

            except Exception as e:
                logger.error(f"Error parsing tabular file {file.name}: {e}")
        else:
            logger.warning(f"Unsupported file type for NCCPL: {file.name}")
            
    if records_by_date:
        combined = pd.DataFrame(list(records_by_date.values()))
        # Add any missing standard columns as NA
        for target in category_map.keys():
            if target not in combined.columns:
                combined[target] = pd.NA
                
        # Sort and deduplicate
        combined = combined.sort_values('date').drop_duplicates(subset=['date'], keep='last')
        return combined
    
    return pd.DataFrame()

if __name__ == "__main__":
    df = parse_nccpl_fipi_lipi()
    print(df.isna().mean())
