import re
import pandas as pd
import pdfplumber
from datetime import date
from pathlib import Path

import logging
from src.psx_predictor.utils.manual_data_loader import inspect_and_log_all
from src.psx_predictor.db.repository import upsert_stock_fundamentals

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_fundamentals_pso() -> pd.DataFrame:
    files = inspect_and_log_all("fundamentals_pso")
    records = []
    for file in files:
        if file.suffix.lower() != '.pdf':
            continue
            
        try:
            with pdfplumber.open(file) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        except Exception as e:
            logger.error(f"Failed to read PSO PDF {file.name}: {e}")
            continue
            
        if "RUPEES IN MILLIONS" not in text:
            logger.warning(f"Unexpected format in PSO PDF {file.name}")
            continue
            
        # Extract years header
        years_match = re.search(r'(\d{4})\s+(\d{4})\s+(\d{4})\s+(\d{4})\s+(\d{4})\s+(\d{4})', text)
        if not years_match:
            logger.warning(f"Could not find years header in {file.name}")
            continue
        years = years_match.groups()
        
        # We need: revenue, net_income, eps_trailing (not in this table, maybe calculate?), total_assets, total_liabilities
        # From text:
        # Net sales 3,149,389 3,571,750 3,391,112 2,451,581 1,204,247 1,108,358
        # Profit / (Loss) after tax 20,911 15,863 5,662 86,223 29,139 (6,466)
        # Non-current assets 81,430 ...
        # Current assets 937,648 ...
        # Non-current liabilities 23,740 ...
        # Current liabilities 745,047 ...
        
        def extract_row(label, text_content):
            match = re.search(rf'{label}\s+([\d,\(\)]+)\s+([\d,\(\)]+)\s+([\d,\(\)]+)\s+([\d,\(\)]+)\s+([\d,\(\)]+)\s+([\d,\(\)]+)', text_content)
            if match:
                vals = []
                for v in match.groups():
                    v_clean = v.replace(',', '')
                    if v_clean.startswith('(') and v_clean.endswith(')'):
                        vals.append(-float(v_clean[1:-1]))
                    else:
                        vals.append(float(v_clean))
                return vals
            return [None] * len(years)

        net_sales = extract_row("Net sales", text)
        pat = extract_row(r"Profit / \(Loss\) after tax", text)
        nc_assets = extract_row("Non-current assets", text)
        c_assets = extract_row("Current assets", text)
        nc_liab = extract_row("Non-current liabilities", text)
        c_liab = extract_row("Current liabilities", text)
        
        for i, year in enumerate(years):
            try:
                dt = date(int(year), 6, 30) # FY ends June 30
            except ValueError:
                continue
                
            rev = net_sales[i] * 1_000_000 if net_sales[i] is not None else None
            ni = pat[i] * 1_000_000 if pat[i] is not None else None
            ta = (nc_assets[i] + c_assets[i]) * 1_000_000 if (nc_assets[i] is not None and c_assets[i] is not None) else None
            tl = (nc_liab[i] + c_liab[i]) * 1_000_000 if (nc_liab[i] is not None and c_liab[i] is not None) else None
            
            if any(v is not None for v in [rev, ni, ta]):
                records.append({
                    'ticker': 'PSO',
                    'report_date': dt,
                    'revenue': rev,
                    'net_income': ni,
                    'total_assets': ta
                })
                
    return pd.DataFrame(records)

def parse_fundamentals_mebl() -> pd.DataFrame:
    files = inspect_and_log_all("fundamentals_mebl")
    records = []
    for file in files:
        if file.suffix.lower() != '.pdf':
            continue
            
        try:
            with pdfplumber.open(file) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        except Exception as e:
            logger.error(f"Failed to read MEBL PDF {file.name}: {e}")
            continue
            
        # The MEBL PDF seems to be a printed webpage without the actual financial tables.
        # We will attempt to parse it, but if it lacks data, log it.
        years_match = re.search(r'(\d{4})\s+(\d{4})\s+(\d{4})', text)
        if not years_match:
            logger.warning(f"Could not find financial data or years in MEBL PDF: {file.name}. It appears to be a webpage printout with links rather than data.")
            continue
            
    return pd.DataFrame(records)

def parse_fundamentals():
    logger.info("Starting manual fundamentals parsing for PSO and MEBL...")
    
    df_pso = parse_fundamentals_pso()
    df_mebl = parse_fundamentals_mebl()
    
    combined = pd.concat([df_pso, df_mebl], ignore_index=True) if not df_pso.empty or not df_mebl.empty else pd.DataFrame()
    
    if not combined.empty:
        success = upsert_stock_fundamentals(combined)
        if success:
            logger.info(f"Successfully upserted {len(combined)} fundamental records.")
        else:
            logger.error("Failed to upsert fundamental records.")
    else:
        logger.warning("No fundamental records extracted.")
        
    return combined

if __name__ == "__main__":
    df = parse_fundamentals()
    print("\n--- FUNDAMENTALS DATAFRAME ---")
    print(df)
