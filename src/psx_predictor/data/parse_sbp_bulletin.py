import os
import sys
import logging
import pdfplumber
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.psx_predictor.utils.manual_data_loader import inspect_and_log_all

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ParseSbpBulletin")

def parse_sbp_bulletin():
    files = inspect_and_log_all("sbp_bulletin")
    
    sections = {
        "Treasury Bills": [],
        "Pakistan Investment Bonds": [],
        "Foreign Exchange Reserves": [],
        "Money Supply": [],
        "Remittances": [],
        "Current Account": []
    }
    
    for file in files:
        logger.info(f"Processing bulletin file: {file.name}")
        ext = file.suffix.lower()
        if ext == '.pdf':
            try:
                with pdfplumber.open(file) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if not text:
                            continue
                            
                        # Check which section this page might belong to
                        matched_section = None
                        for section in sections.keys():
                            if section.lower() in text.lower():
                                matched_section = section
                                break
                                
                        if matched_section:
                            tables = page.extract_tables()
                            if tables:
                                for t_idx, table in enumerate(tables):
                                    if len(table) > 1:
                                        headers = table[0]
                                        logger.info(f"File {file.name} | Page {i+1} | Section '{matched_section}' | Table {t_idx+1} Headers Found: {headers}")
                                        
                                        # Deduplicate headers to avoid pd.concat InvalidIndexError
                                        seen = {}
                                        dedup_headers = []
                                        for h in headers:
                                            h_str = str(h) if h is not None else "Unnamed"
                                            if h_str in seen:
                                                seen[h_str] += 1
                                                dedup_headers.append(f"{h_str}_{seen[h_str]}")
                                            else:
                                                seen[h_str] = 0
                                                dedup_headers.append(h_str)
                                                
                                        df = pd.DataFrame(table[1:], columns=dedup_headers)
                                        df['source_file'] = file.name
                                        sections[matched_section].append(df)
            except Exception as e:
                logger.error(f"Error parsing PDF bulletin {file.name}: {e}")
        else:
            logger.warning(f"Unsupported bulletin file type: {file.name}. Expected .pdf")
            
    # Combine dataframes per section
    dfs_out = {}
    for section, df_list in sections.items():
        if df_list:
            combined = pd.concat(df_list, ignore_index=True)
            dfs_out[section] = combined
        else:
            dfs_out[section] = pd.DataFrame()
            
    return dfs_out

if __name__ == "__main__":
    results = parse_sbp_bulletin()
    for section, df in results.items():
        print(f"\n--- SECTION: {section} ---")
        if not df.empty:
            print(df.head())
            print(f"Total Rows: {len(df)}, Nulls: \n{df.isna().mean()}")
        else:
            print("No data extracted.")
