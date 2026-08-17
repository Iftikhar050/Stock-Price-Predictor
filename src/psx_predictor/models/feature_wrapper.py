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

def feature_fn(ticker: str, start_date: str, end_date: str):
    """Return X, y, dates, close for a ticker within the given date range.
    The returned DataFrames are already filtered to the inclusive interval.
    """
    csv_path = os.path.join(BASE_FEATURES_DIR, f"{ticker.lower()}_features.csv")
    df = pd.read_csv(csv_path)
    # Create target as next‑day return
    df["target_return_t1"] = (df["close"].shift(-1) - df["close"]) / df["close"]
    df = df.dropna(subset=["target_return_t1"])
    # Filter by date range
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    df = df.loc[mask]
    exclude = ["ticker", "date", "created_at", "target_return_t1", "close"]
    X = df.drop(columns=exclude)
    y = df["target_return_t1"]
    dates = df["date"]
    close = df["close"]
    return X, y, dates, close
