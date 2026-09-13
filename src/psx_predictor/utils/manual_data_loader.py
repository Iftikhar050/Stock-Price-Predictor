import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import pdfplumber

logger = logging.getLogger("ManualDataLoader")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
MANUAL_DATA_DIR = ROOT_DIR / "data" / "raw" / "manual"

def load_manual_files(source: str) -> List[Path]:
    """
    Returns all non-.gitkeep files in data/raw/manual/{source}/
    """
    source_dir = MANUAL_DATA_DIR / source
    if not source_dir.exists():
        logger.warning(f"Manual data directory {source_dir} does not exist.")
        return []
    
    files = []
    for entry in source_dir.iterdir():
        if entry.is_file() and entry.name != ".gitkeep":
            files.append(entry)
            
    return files

def inspect_file(path: Path) -> Dict[str, Any]:
    """
    Defensively sniff file contents before full parsing.
    For .csv/.xlsx, returns columns, n_rows, and a 3-row sample.
    For .pdf, returns n_pages and a text sample.
    """
    ext = path.suffix.lower()
    result = {"file": str(path)}
    
    try:
        if ext in ['.csv', '.xlsx', '.xls']:
            if ext == '.csv':
                df = pd.read_csv(path, nrows=3)
                # Count total rows efficiently
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    n_rows = sum(1 for _ in f) - 1 # excluding header
            else:
                df = pd.read_excel(path, nrows=3)
                n_rows = pd.read_excel(path, usecols=[0]).shape[0]
                
            result["type"] = "tabular"
            result["columns"] = list(df.columns)
            result["n_rows"] = n_rows
            result["sample"] = df.head(3).to_dict(orient="records")
            
        elif ext == '.pdf':
            with pdfplumber.open(path) as pdf:
                result["type"] = "pdf"
                result["n_pages"] = len(pdf.pages)
                if len(pdf.pages) > 0:
                    text = pdf.pages[0].extract_text()
                    result["sample_text"] = text[:500] if text else ""
                else:
                    result["sample_text"] = ""
        else:
            result["type"] = "unknown"
            result["error"] = f"Unsupported extension: {ext}"
            
    except Exception as e:
        result["type"] = "error"
        result["error"] = str(e)
        
    return result

def inspect_and_log_all(source: str) -> List[Path]:
    """
    Loads all files for a given source, inspects them, logs the result, and returns the paths.
    """
    files = load_manual_files(source)
    if not files:
        logger.info(f"No manual files found for source: {source}")
        return []
        
    logger.info(f"Inspecting {len(files)} files for source: {source}")
    for file in files:
        inspection = inspect_file(file)
        # Log carefully so it's readable
        logger.info(f"--- INSPECTION RESULT FOR {file.name} ---")
        if inspection.get("type") == "tabular":
            logger.info(f"  Type: Tabular ({file.suffix})")
            logger.info(f"  Rows: {inspection.get('n_rows')}")
            logger.info(f"  Columns: {inspection.get('columns')}")
            logger.info(f"  Sample: {inspection.get('sample')}")
        elif inspection.get("type") == "pdf":
            logger.info(f"  Type: PDF")
            logger.info(f"  Pages: {inspection.get('n_pages')}")
            logger.info(f"  Sample Text: {repr(inspection.get('sample_text', ''))}")
        else:
            logger.info(f"  Error/Unknown: {inspection.get('error')}")
            
    return files
