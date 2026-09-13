"""
Scans every {TICKER}_master.csv in data/processed/ and reports, per column,
how many tickers have it constant, all-zero/all-null, or otherwise carrying
no real signal. Aggregated across the whole active universe so systemic gaps
(a column that's dead for every ticker) are distinguished from isolated,
explainable ones (e.g. a bank-only field that's legitimately null for a
non-bank).

Output: a CSV report (one row per column) + a printed summary.
"""
import os
import sys
import glob
import json
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

IGNORE_COLS = {"ticker", "date", "created_at"}

def analyze_file(path: str):
    df = pd.read_csv(path, low_memory=False)
    ticker = os.path.basename(path).replace("_master.csv", "")
    stats = {}
    for col in df.columns:
        if col in IGNORE_COLS:
            continue
        s = df[col]
        n = len(s)
        if n == 0:
            continue
        is_numeric = pd.api.types.is_numeric_dtype(s)
        null_pct = s.isna().mean()
        nonnull = s.dropna()
        nunique = nonnull.nunique()
        is_constant = nunique <= 1
        if is_numeric:
            zero_pct = (nonnull == 0).mean() if len(nonnull) else 1.0
            all_zero = zero_pct >= 0.999
        else:
            empty_pct = (nonnull.astype(str).str.strip() == "").mean() if len(nonnull) else 1.0
            zero_pct = empty_pct
            all_zero = empty_pct >= 0.999
        stats[col] = {
            "is_numeric": is_numeric,
            "null_pct": round(float(null_pct), 4),
            "zero_or_empty_pct": round(float(zero_pct), 4),
            "nunique": int(nunique),
            "is_constant": bool(is_constant),
            "is_all_zero_or_empty": bool(all_zero),
            "constant_value": (nonnull.iloc[0] if is_constant and len(nonnull) else None),
        }
    return ticker, df.shape, stats

def main():
    files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_master.csv")))
    print(f"Found {len(files)} master.csv files to audit.")

    all_columns = set()
    per_ticker_stats = {}
    shapes = {}

    for i, f in enumerate(files, 1):
        try:
            ticker, shape, stats = analyze_file(f)
            per_ticker_stats[ticker] = stats
            shapes[ticker] = shape
            all_columns.update(stats.keys())
            print(f"[{i}/{len(files)}] {ticker}: {shape}")
        except Exception as e:
            print(f"[{i}/{len(files)}] {os.path.basename(f)}: FAILED - {e}")

    n_tickers = len(per_ticker_stats)
    rows = []
    for col in sorted(all_columns):
        present_in = [t for t in per_ticker_stats if col in per_ticker_stats[t]]
        n_present = len(present_in)
        n_constant = sum(1 for t in present_in if per_ticker_stats[t][col]["is_constant"])
        n_all_zero = sum(1 for t in present_in if per_ticker_stats[t][col]["is_all_zero_or_empty"])
        n_fully_null = sum(1 for t in present_in if per_ticker_stats[t][col]["null_pct"] >= 0.999)
        avg_null_pct = np.mean([per_ticker_stats[t][col]["null_pct"] for t in present_in]) if present_in else np.nan
        avg_zero_pct = np.mean([per_ticker_stats[t][col]["zero_or_empty_pct"] for t in present_in]) if present_in else np.nan
        sample_constant_vals = list({per_ticker_stats[t][col]["constant_value"] for t in present_in if per_ticker_stats[t][col]["is_constant"]})[:5]
        rows.append({
            "column": col,
            "n_tickers_present": n_present,
            "n_tickers_missing": n_tickers - n_present,
            "pct_tickers_constant": round(n_constant / n_present, 3) if n_present else None,
            "pct_tickers_all_zero_or_empty": round(n_all_zero / n_present, 3) if n_present else None,
            "pct_tickers_fully_null": round(n_fully_null / n_present, 3) if n_present else None,
            "avg_null_pct": round(float(avg_null_pct), 3) if n_present else None,
            "avg_zero_or_empty_pct": round(float(avg_zero_pct), 3) if n_present else None,
            "sample_constant_values": sample_constant_vals,
        })

    report_df = pd.DataFrame(rows).sort_values("pct_tickers_all_zero_or_empty", ascending=False)
    out_path = os.path.join(PROCESSED_DIR, "_column_quality_report.csv")
    report_df.to_csv(out_path, index=False)
    print(f"\nSaved full report: {out_path}")

    # Also dump the raw shapes to spot schema drift across tickers
    shapes_path = os.path.join(PROCESSED_DIR, "_ticker_shapes.json")
    with open(shapes_path, "w") as fh:
        json.dump({t: list(s) for t, s in shapes.items()}, fh, indent=2)
    print(f"Saved ticker shapes: {shapes_path}")

    print("\n=== Columns that are near-universally dead (>=90% of tickers all-zero/empty) ===")
    dead = report_df[report_df["pct_tickers_all_zero_or_empty"] >= 0.90]
    print(dead[["column", "n_tickers_present", "pct_tickers_all_zero_or_empty", "avg_null_pct"]].to_string(index=False))

if __name__ == "__main__":
    main()
