"""
master_data.py — One place the API loads a ticker's feature table from.

Locally this is data/processed/{TICKER}_master.csv, the pipeline's output.
A deployment sets SERVING_DATA_DIR to a folder of trimmed Parquet copies
(scripts/export_serving_data.py) so the server doesn't need the full 1.6 GB of
CSVs. It's opt-in rather than "use Parquet if present" so a leftover local
copy can never silently serve stale data after the pipeline rebuilds the CSVs.
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import pandas as pd
import pyarrow.parquet as pq

PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
SERVING_DATA_DIR = os.environ.get("SERVING_DATA_DIR")


def master_path(ticker: str) -> str | None:
    ticker = ticker.upper()
    if SERVING_DATA_DIR:
        path = os.path.join(SERVING_DATA_DIR, f"{ticker}_master.parquet")
    else:
        path = os.path.join(PROCESSED_DIR, f"{ticker}_master.csv")
    return path if os.path.exists(path) else None


def read_master(ticker: str, columns=None, exclude=None) -> pd.DataFrame | None:
    """The ticker's feature table, optionally limited to `columns` (missing
    ones are skipped) or with `exclude` dropped. None if there's no file."""
    path = master_path(ticker)
    if path is None:
        return None

    if path.endswith(".parquet"):
        available = pq.read_schema(path).names
    else:
        available = pd.read_csv(path, nrows=0).columns.tolist()

    wanted = [c for c in available if (columns is None or c in columns) and (exclude is None or c not in exclude)]
    if path.endswith(".parquet"):
        return pd.read_parquet(path, columns=wanted)
    return pd.read_csv(path, usecols=wanted, engine="c")
