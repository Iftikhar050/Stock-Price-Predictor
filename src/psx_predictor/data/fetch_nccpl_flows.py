import os
import sys
import logging

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

from src.psx_predictor.data.parse_nccpl_fipi_lipi import parse_nccpl_fipi_lipi
from src.psx_predictor.db.repository import upsert_macro_indicators

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NCCPLFlowCollector")


def fetch_nccpl_institutional_flows() -> bool:
    """
    Loads National Clearing Company of Pakistan (NCCPL) FIPI (Foreign
    Investor) / LIPI (Local Institutional) portfolio flow figures from
    manually-dropped NCCPL export files (data/raw/manual/nccpl_fipi_lipi/)
    and upserts them into macro_indicators.

    NCCPL does not publish a free, scrapable daily historical time series -
    there is no live source to fetch here. The manual files this reads are
    typically a single day's snapshot download, not a backfilled history.
    This only writes real values for the date(s) actually present in those
    files; it never fabricates a series to cover the gaps. build_features.py
    forward-fills (never backward-fills) macro_indicators columns, so a lone
    recent snapshot correctly reads as NaN for dates before it rather than
    being stamped across all of history.
    """
    logger.info("Loading NCCPL FIPI/LIPI institutional portfolio flows from manual files...")

    df = parse_nccpl_fipi_lipi()
    if df.empty:
        logger.warning(
            "No NCCPL FIPI/LIPI manual files found in data/raw/manual/nccpl_fipi_lipi/ "
            "- nothing to ingest. Drop a fresh NCCPL export there to update this series."
        )
        return False

    ok = upsert_macro_indicators(df)
    if ok:
        dates = sorted(str(d) for d in df['date'].tolist())
        logger.info(f"Ingested real NCCPL FIPI/LIPI flows for {len(df)} date(s): {dates}")
    return ok


# Alias for backward compatibility
fetch_nccpl_flows = fetch_nccpl_institutional_flows

if __name__ == "__main__":
    fetch_nccpl_institutional_flows()
