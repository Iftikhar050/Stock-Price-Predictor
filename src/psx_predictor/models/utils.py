import logging
import json
import os
from datetime import datetime
from sqlalchemy import text
from src.psx_predictor.db.connection import engine

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def refit_on_all_data() -> bool:
    """Whether trainers refit on every row (train + test) before saving.

    The train/test split exists to *measure* a model; the model that serves
    predictions should still learn from the most recent year instead of
    stopping at the cutoff. On by default - set REFIT_ON_ALL_DATA=0 to save the
    train-only fit (e.g. to reproduce a reported test score)."""
    return os.getenv("REFIT_ON_ALL_DATA", "1") != "0"


def evaluate_only() -> bool:
    """EVALUATE_ONLY=1: score the train-only fit and record its held-out
    accuracy, but save no model - the live models stay as they are."""
    return os.getenv("EVALUATE_ONLY") == "1"


HOLDOUT_PATH = os.path.join(MODELS_DIR, "holdout_accuracy.json")


def directional_hits(y_true, y_pred):
    """Per-row 1/0 for a correctly called direction - the same >= 0 convention
    predict.py uses (a flat day counts as up)."""
    import numpy as np
    return (np.asarray(y_true) >= 0) == (np.asarray(y_pred) >= 0)


def save_holdout_accuracy(model_key: str, tickers, hits, dates) -> None:
    """Records each ticker's directional accuracy on the held-out test period
    for one model family, so the website can show accuracy on days the model
    never trained on instead of re-scoring it on its own training days.
    `tickers`, `hits` (1/0 per row, see directional_hits) and `dates` are
    parallel sequences over the pooled test rows."""
    import pandas as pd
    df = pd.DataFrame({"ticker": list(tickers), "hit": list(hits), "date": pd.to_datetime(list(dates))})
    per_ticker = {
        t: {"accuracy": round(float(g["hit"].mean()) * 100, 1), "days": int(len(g))}
        for t, g in df.groupby("ticker")
    }
    data = {}
    if os.path.exists(HOLDOUT_PATH):
        with open(HOLDOUT_PATH) as f:
            data = json.load(f)
    data[model_key] = {
        "test_start": df["date"].min().date().isoformat(),
        "test_end": df["date"].max().date().isoformat(),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "overall_accuracy": round(float(df["hit"].mean()) * 100, 1),
        "tickers": per_ticker,
    }
    with open(HOLDOUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Held-out directional accuracy for '{model_key}': {data[model_key]['overall_accuracy']}% "
                f"over {len(df)} rows, {len(per_ticker)} tickers -> {HOLDOUT_PATH}")

def choose_global_cutoff(test_trading_days: int = 250, min_train_trading_days: int = 500) -> tuple[str, list[str]]:
    """
    Dynamically computes a global train/test cutoff date ensuring the test set has 
    `test_trading_days` of data. Evaluates all tickers to ensure they have at least 
    `min_train_trading_days` of history before the cutoff.
    
    Returns:
        tuple: (cutoff_date_str, valid_tickers_list)
    """
    logger.info("Computing dynamic global train/test cutoff date...")
    
    # Get the global dates using the market index (which represents trading days)
    query_dates = text("SELECT date FROM stock_market_index ORDER BY date DESC")
    with engine.connect() as conn:
        all_dates = [row[0] for row in conn.execute(query_dates).fetchall()]
        
    if len(all_dates) < test_trading_days + min_train_trading_days:
        logger.warning("Not enough global trading days to satisfy the split requirements securely. Using default.")
        cutoff_date = all_dates[len(all_dates) // 5] if all_dates else datetime(2023, 1, 1).date()
    else:
        # The cutoff date is the date exactly `test_trading_days` from the latest available date
        cutoff_date = all_dates[test_trading_days]
        
    cutoff_str = cutoff_date.strftime('%Y-%m-%d')
    logger.info(f"Selected Global Cutoff Date: {cutoff_str} (Ensures ~{test_trading_days} test days)")
    
    # Validate each ticker against the min_train_trading_days requirement
    query_counts = text("""
        SELECT ticker, COUNT(*) as train_days
        FROM stock_eod_data
        WHERE date <= :cutoff
        GROUP BY ticker
    """)
    with engine.connect() as conn:
        counts = conn.execute(query_counts, {"cutoff": cutoff_date}).fetchall()
        
    ticker_counts = {row[0]: row[1] for row in counts}
    
    from src.psx_predictor.db.repository import get_active_tickers
    active_tickers = get_active_tickers()
    
    valid_tickers = []
    excluded_tickers = []
    
    for t in active_tickers:
        if ticker_counts.get(t, 0) >= min_train_trading_days:
            valid_tickers.append(t)
        else:
            excluded_tickers.append(t)
            
    if excluded_tickers:
        logger.warning(f"Excluded {len(excluded_tickers)} tickers due to insufficient training history (<{min_train_trading_days} days before cutoff): {excluded_tickers}")
    
    logger.info(f"Proceeding with {len(valid_tickers)} robust tickers for training.")
    
    # Log to a JSON file alongside model versioning
    os.makedirs(MODELS_DIR, exist_ok=True)
    metadata_log = {
        "timestamp": datetime.now().isoformat(),
        "global_cutoff_date": cutoff_str,
        "test_trading_days_requested": test_trading_days,
        "min_train_trading_days_requested": min_train_trading_days,
        "valid_tickers_count": len(valid_tickers),
        "excluded_tickers": excluded_tickers
    }
    
    log_path = os.path.join(MODELS_DIR, "training_split_metadata.json")
    with open(log_path, 'w') as f:
        json.dump(metadata_log, f, indent=4)
        
    return cutoff_str, valid_tickers
