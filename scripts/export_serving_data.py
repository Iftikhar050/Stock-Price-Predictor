"""
Builds the trimmed copy of the feature tables that a deployed API serves from
(it reads them when SERVING_DATA_DIR points at the output folder - see
src/psx_predictor/api/master_data.py). data/processed/ is left untouched;
training and the pipeline keep using the full history.

The API reads at most the last 1,855 rows of a ticker (5-year prediction
backtest + 30-day LSTM lookback in predict.py); everything else uses the last
row. Keeping 2,000 rows makes the trimmed copy give identical API output.

Usage: python scripts/export_serving_data.py [output_dir]   (default data/serving)
"""
import glob
import os
import sys

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
SERVING_ROWS = 2000


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT_DIR, "data", "serving")
    os.makedirs(out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_master.csv")))
    total_in = total_out = 0
    for path in paths:
        name = os.path.basename(path).replace(".csv", ".parquet")
        df = pd.read_csv(path, engine="c", low_memory=False).tail(SERVING_ROWS)
        out = os.path.join(out_dir, name)
        df.to_parquet(out, index=False)
        total_in += os.path.getsize(path)
        total_out += os.path.getsize(out)

    print(f"Exported {len(paths)} tickers to {out_dir}: "
          f"{total_in / 1e6:,.0f} MB of CSV -> {total_out / 1e6:,.0f} MB of Parquet")


if __name__ == "__main__":
    main()
