import os
import sys
import glob
import re
import logging
import pdfplumber
import pandas as pd
from datetime import datetime
from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.psx_predictor.db.connection import engine
from src.psx_predictor.db.repository import upsert_macro_indicators

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LoadSbpKiborPdfs")

def load_sbp_kibor_pdfs(directory: str):
    logger.info(f"Loading SBP KIBOR PDFs from {directory}")
    
    files = glob.glob(os.path.join(directory, "*.pdf"))
    if not files:
        logger.warning(f"No PDFs found in {directory}")
        return
        
    records = []
    
    for file in files:
        try:
            with pdfplumber.open(file) as pdf:
                page = pdf.pages[0]
                text_content = page.extract_text()
                
            lines = text_content.split('\n')
            
            # Parse Date
            # Looking for a line like "30-Jun-2026" or "30-Jun-26"
            date_obj = None
            for line in lines[:5]:
                # Try common formats
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
                fname = os.path.basename(file)
                m = re.search(r'(\d{1,2}-[a-zA-Z]{3}-\d{2,4})', fname, re.IGNORECASE)
                if m:
                    date_str = m.group(1)
                    if len(date_str.split('-')[2]) == 2:
                        date_obj = datetime.strptime(date_str, "%d-%b-%y").date()
                    else:
                        date_obj = datetime.strptime(date_str, "%d-%b-%Y").date()
            
            if not date_obj:
                logger.warning(f"Could not extract date for {file}, skipping.")
                continue
                
            # Parse KIBOR (Offer rate is standard benchmark)
            kibor_3m = None
            kibor_6m = None
            kibor_1y = None
            
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                # E.g. "3 - Month 11.54 11.79" -> Parts: ['3', '-', 'Month', '11.54', '11.79']
                # Sometimes it might be "3-Month 11.54 11.79"
                line_lower = line.lower()
                
                if "3 - month" in line_lower or "3-month" in line_lower:
                    try:
                        kibor_3m = float(parts[-1])  # Offer is the last number
                    except ValueError:
                        pass
                elif "6 - month" in line_lower or "6-month" in line_lower:
                    try:
                        kibor_6m = float(parts[-1])
                    except ValueError:
                        pass
                elif "1 - year" in line_lower or "1-year" in line_lower:
                    try:
                        kibor_1y = float(parts[-1])
                    except ValueError:
                        pass
                        
            if kibor_3m or kibor_6m or kibor_1y:
                records.append({
                    "date": date_obj,
                    "kibor_3m": kibor_3m,
                    "kibor_6m": kibor_6m,
                    "kibor_1y": kibor_1y
                })
                
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
            
    if records:
        df = pd.DataFrame(records)
        df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
        upsert_macro_indicators(df)
        logger.info(f"Successfully processed {len(df)} KIBOR records into database.")
    else:
        logger.warning("No valid KIBOR data extracted.")

if __name__ == "__main__":
    load_sbp_kibor_pdfs("data/raw/manual/sbp_kibor")
