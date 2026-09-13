"""Thin wrapper that re‑uses the data‑preparation logic from the training scripts.
The function is deliberately simple – it loads the CSV produced by `build_features.py`
and builds the target variable exactly as the training scripts do.
"""
import os
import pandas as pd

# Base directory where processed feature CSVs are stored (same as training scripts)
BASE_FEATURES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "processed")
)

import os
import pandas as pd
import numpy as np
from sqlalchemy import text
from src.psx_predictor.db.connection import engine

# Base directory where processed feature CSVs are stored (same as training scripts)
BASE_FEATURES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "processed")
)

def get_ticker_sectors():
    """Fetches the sector for each ticker from the database."""
    query = text("SELECT ticker, sector FROM stock_metadata")
    with engine.connect() as conn:
        res = conn.execute(query).fetchall()
    return {row[0]: row[1] for row in res}


def get_feature_columns(df: pd.DataFrame, exclude_cols: list = None) -> list:
    """Single source of truth for which master-CSV columns are valid model
    inputs. Both prepare_data() (training) and predict.py (serving) call this
    so the two selection lists can't silently drift apart again - they
    previously did: prepare_data() had no dtype filter at all and would feed
    free-text columns (sentiment_coverage_era, raw_pucars_headline_daily,
    raw_pucars_body_daily, raw_pucars_category_daily, raw_news_headline_daily)
    straight into model.fit(), while predict.py already filtered by dtype.

    'ticker' is deliberately kept (unless the caller explicitly excludes it)
    even though it's non-numeric - each training script has its own handling
    for it (train_baseline.py/train_regression.py drop it, train_xgboost.py
    casts it to a pandas category, train_lstm.py pulls it out for a learned
    embedding). Every other non-numeric column is excluded.
    """
    exclude_cols = exclude_cols or []
    return [
        col for col in df.columns
        if col not in exclude_cols and (col == 'ticker' or pd.api.types.is_numeric_dtype(df[col]))
    ]


def make_dead_zone_labels(returns: pd.Series, dead_zone: float = 0.005) -> pd.Series:
    """Map a continuous return series to {-1, 0, 1} (down / flat / up).

    Mirrors the three-class "central dead-zone" scheme from Shen/Jiang/Zhang
    (CS229 2013): near-zero next-day returns are mostly noise, not signal, and
    collapsing them into a "flat/no-trade" class instead of forcing a binary
    up/down call measurably improves precision on the moves that matter. The
    default 0.5% band is a reasonable starting point for PSX daily returns
    (roughly the scale of typical bid-ask/slippage friction) - callers doing a
    real precision/recall sweep should treat `dead_zone` as tunable per the
    paper's own finding that the optimal window is an empirical choice, not a
    fixed constant.
    """
    labels = pd.Series(0, index=returns.index, dtype=int)
    labels[returns > dead_zone] = 1
    labels[returns < -dead_zone] = -1
    return labels


def prepare_data(ticker: str, ticker_sectors: dict = None, horizon: int = 1, dead_zone: float = None):
    """Loads engineered features and creates the target variable for a given ticker.

    This is the single source of truth for model input data preparation, shared
    by both production training scripts and walk-forward evaluation.

    `horizon` controls the forward-return window the target is built from
    (1 = next trading day, 5 = one week ahead, 20 = ~one month ahead, ...).
    Longer horizons are materially easier to predict than next-day (Shen/
    Jiang/Zhang report 74% 1-day vs. 85% 30-day accuracy on the same
    features) - `horizon` lets training/evaluation code compare across
    windows instead of only ever training the hardest, noisiest target.

    `dead_zone`, if given, also attaches a `target_class` column ({-1,0,1})
    built by `make_dead_zone_labels` on top of the same forward return, for
    classification-style training/evaluation alongside the regression target.
    """
    csv_path = os.path.join(BASE_FEATURES_DIR, f"{ticker.upper()}_master.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}. Run build_features.py first.")

    df = pd.read_csv(csv_path, low_memory=False).copy()
    if df.empty:
        raise ValueError(f"Empty feature data for {ticker}")

    # 1. Create Target Variable (forward return over `horizon` trading days)
    target_col = f"target_return_t{horizon}"
    df[target_col] = (df['close'].shift(-horizon) - df['close']) / df['close']
    df.dropna(subset=[target_col], inplace=True)
    df = df.copy()

    target_class = None
    if dead_zone is not None:
        target_class = make_dead_zone_labels(df[target_col], dead_zone)

    # 2. Select Features (X) and Target (y)
    exclude_cols = ['date', 'created_at', target_col, 'close']
    feature_cols = get_feature_columns(df, exclude_cols)

    X = df[feature_cols].copy()

    # Only add sector if we have mapping (e.g. for xgboost)
    if ticker_sectors is not None:
        X['sector'] = X['ticker'].map(ticker_sectors)

    # Clean numeric columns to remove inf / nan / float32 overflow
    num_cols = X.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X[num_cols] = X[num_cols].clip(lower=-1e9, upper=1e9)

    y = df[target_col]
    dates = pd.to_datetime(df['date'])
    current_close = df['close']

    if target_class is not None:
        return X, y, dates, current_close, target_class
    return X, y, dates, current_close

def feature_fn(ticker: str, start_date: str, end_date: str, ticker_sectors: dict = None, horizon: int = 1, dead_zone: float = None):
    """Return X, y, dates, close (and optionally target_class) for a ticker
    within the given date range. The returned DataFrames are already filtered
    to the inclusive interval.
    """
    result = prepare_data(ticker, ticker_sectors, horizon=horizon, dead_zone=dead_zone)
    if dead_zone is not None:
        X, y, dates, close, target_class = result
    else:
        X, y, dates, close = result
        target_class = None

    # Filter by date range
    mask = (dates >= start_date) & (dates <= end_date)

    # We drop 'ticker' from X for the models? Wait, in original train_xgboost it kept ticker.
    # train_baseline excluded ticker. Let's keep the return as is, but in run_walk_forward, it doesn't drop anything.
    # We should let the models handle whether they use it or not, or drop it if it's not xgboost.

    # However, to be perfectly safe, since walk-forward runs on `feature_fn`,
    # let's return the filtered rows
    X_filtered = X.loc[mask].copy()
    y_filtered = y.loc[mask].copy()
    dates_filtered = dates.loc[mask].copy()
    close_filtered = close.loc[mask].copy()

    if target_class is not None:
        target_class_filtered = target_class.loc[mask].copy()
        return X_filtered, y_filtered, dates_filtered, close_filtered, target_class_filtered
    return X_filtered, y_filtered, dates_filtered, close_filtered
