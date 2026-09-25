"""Builds the pooled, model-ready Phase 1 (KSE-100) training dataset and writes it to a
single Parquet file - the portable input for notebooks/Phase1_XGBoost_Pooled_Training.ipynb,
which is deliberately codebase-independent (no src.psx_predictor imports, no DB connection)
so it can be copied and run on any local Jupyter install. This script is the one piece that
still needs the codebase: it reads the 97 staged per-ticker CSVs in data/staging/phase_1/
and looks up each ticker's sector from the DB.

Mirrors, exactly, the data-prep logic previously inlined in the notebook's Sections 1-4:
- forward return target over HORIZON trading days
- drop rows whose implied move exceeds a PSX-circuit-breaker-scaled threshold (bad ticks)
- drop feature columns identical across all tickers on every date (no cross-sectional
  information for a pooled model - see notebook Section 3 for the full reasoning)
- consolidate the per-ticker search_trend_<ticker> columns into one search_trend_own_ticker
- chronological 80/20 train/test split, recorded as a 'split' column

Plus two feature-engineering fixes added after the EDA in Phase1_XGBoost_Pooled_Training.ipynb
found ~40% of features contributing zero importance to the trained model:

- **Proxy missingness flags** (`WAS_MISSING_PROXY_COLS`): the raw per-ticker master CSVs
  already have every NaN converted to 0.0 upstream in build_features.py, before this script
  ever sees the data - so the true fix (stop zero-filling at the source, in build_features.py,
  and regenerate all 97 master CSVs) is out of scope here and left for later. As a stopgap,
  a `<col>_was_missing` flag is added for columns confirmed (via the EDA's zero-inflation
  audit) to be sector-conditional or sparse-source values where an exact 0.0 is statistically
  implausible as a genuinely computed number (e.g. corr_stock_brent is 0.0 for every single
  non-oil-and-gas ticker - not a real "zero correlation", a "never computed for this sector"
  default). This can't recover the original value, but it lets the model learn to distrust a
  zero when the flag says it's fake, instead of training on it as if it were real.
- **Cross-sectional fundamentals normalization** (`CROSS_SECTIONAL_FUNDAMENTALS_COLS`): raw
  PKR-denominated fundamentals (revenue, total_assets, ...) span orders of magnitude across
  97 tickers of very different sizes, so a tree model can't find a globally useful split
  threshold on the raw value - confirmed these got exactly zero importance in training.
  Replaced in place with each value's percentile rank among all tickers on the same date,
  which is comparable across the whole universe the same way pe_percentile_1y already is.

Plus three more from the same plan, implemented after re-verifying each premise against the
actual rebuilt dataset rather than trusting the original EDA snapshot (which predates
Priority 1/2/3 and is stale):

- **Priority 4 - drop confirmed-dead short-term technicals** (`DEAD_SHORT_TERM_TECHNICAL_COLS`):
  these got exactly zero XGBoost importance at HORIZON=20 - next-day-scale signals with no
  information left at a 20-trading-day target. Only dropped when `--horizon 20` (the horizon
  they were actually shown dead at); kept for shorter horizons where they plausibly still
  matter, per the original single-ticker report.
- **Priority 5 - deduplicate the volatility measures**: re-checked the correlation matrix
  directly on the rebuilt dataset rather than assuming the plan's premise held. Of the 7
  volatility columns, only `garman_klass_volatility_20d` and `parkinson_volatility_20d` are
  actually redundant (r=0.989) - the other 5 (historical/market/sector/oil/bond) are only
  weakly correlated with each other (all |r|<0.17) and each other's importance evidence, so
  they are NOT dropped; the "highly intercorrelated" framing in the original plan overstated
  it from the heatmap's visual impression. `garman_klass_volatility_20d` is dropped (the
  redundant one with the lower of the pair's two near-identical correlations with the
  target); `parkinson_volatility_20d` is kept.
- **Priority 6 - one new sector-relative valuation feature** (`sector_relative_pe_percentile`):
  the existing `pe_percentile_1y`/`pe_percentile_3y` are each ticker's valuation percentile
  against its OWN trailing history, not against its sector peers on the same date - a
  genuinely different signal a tree can't reconstruct from what's already in the dataset.
  This is the one exploratory addition kept from the plan's Priority 6; explicit product/
  interaction terms between top-MI and top-correlation features were considered but not
  added; XGBoost already learns feature interactions natively via sequential splits, so
  hand-built interaction terms mostly add redundant, noisier copies of information the
  model already has direct access to - low expected value for a tree ensemble specifically
  (this reasoning would not hold for a linear model).

Usage:
  python scripts/export_phase1_training_data.py
  python scripts/export_phase1_training_data.py --horizon 5
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.psx_predictor.models.feature_wrapper import get_feature_columns, get_ticker_sectors

PHASE_DIR = os.path.join(ROOT_DIR, "data", "staging", "phase_1")

# Calibrated against this phase's actual return distribution at each horizon (see the
# notebook's Section 3 markdown for how these were chosen) - not a general formula.
OUTLIER_THRESHOLDS = {1: 0.20, 5: 0.40, 20: 0.75}

# Columns where the EDA's zero-inflation audit + a manual check (corr_stock_brent is 0.0 for
# 100% of ABL's rows - a bank, not an oil/gas company) confirmed an exact 0.0 is standing in
# for "never computed for this ticker/sector/quarter", not a real value. Excludes genuine
# binary flags (earnings_season_flag etc.) and technicals that are legitimately often exactly
# 0 by construction (daily_return, macd_hist, ...) - only sector-conditional correlations and
# sparse-source fundamentals/macro-surprise columns, where zero is not a plausible real value.
WAS_MISSING_PROXY_COLS = [
    "corr_stock_brent", "corr_stock_gold", "corr_stock_pkr_usd", "corr_stock_policy_rate",
    "sector_volatility_20d", "bond_volatility_20d",
    "eps_qoq_surprise", "eps_consensus_surprise", "eps_expected",
    "cpi_surprise", "cpi_expected", "policy_rate_expected",
    "pucars_sentiment_daily", "revenue_growth", "profit_growth", "asset_growth",
    "price_to_cash_flow", "fear_index_proxy",
]

# Raw PKR-denominated fundamentals confirmed to get exactly zero XGBoost feature importance
# in their absolute-value form - replaced with each value's cross-sectional percentile rank
# (same idea as pe_percentile_1y/pb_percentile_3y, which already rank well) so a company's
# size relative to its peers on the same date is what the model sees, not an incomparable
# absolute PKR figure.
CROSS_SECTIONAL_FUNDAMENTALS_COLS = [
    "revenue", "total_assets", "total_debt", "shares_outstanding", "gross_profit",
    "operating_cash_flow", "total_cash", "receivables", "inventory", "working_capital",
    "book_value_per_share",
]

# Confirmed exactly-zero XGBoost importance at HORIZON=20 - next-day-scale signals that
# carry no information at a 20-trading-day target (consistent with the EDA's near-zero
# own-history autocorrelation). Only dropped for horizon==20; a future shorter-horizon
# variant should keep these.
DEAD_SHORT_TERM_TECHNICAL_COLS = [
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5", "return_lag_10",
    "daily_return", "close_pos", "daily_spread", "relative_volume", "stochastic_d",
    "short_term_speculation_proxy", "days_since_corp_event", "days_since_last_event",
    "oil_return_pct",
]

# Of the 7 volatility columns, only this pair is actually redundant (r=0.989 on the rebuilt
# dataset - checked directly, not assumed). garman_klass is the one dropped; parkinson is
# kept (marginally higher correlation with |target| of the two: 0.206 vs 0.196).
REDUNDANT_VOLATILITY_COLS = ["garman_klass_volatility_20d"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=20,
                         help="Forward-return horizon in trading days (default: 20)")
    parser.add_argument("--out", default=None,
                         help="Output Parquet path (default: data/staging/phase_1/phase1_model_input_horizon{H}.parquet)")
    args = parser.parse_args()
    horizon = args.horizon

    max_abs_return = OUTLIER_THRESHOLDS.get(horizon, OUTLIER_THRESHOLDS[max(
        (h for h in OUTLIER_THRESHOLDS if h <= horizon), default=1)])

    target_col = f"target_return_t{horizon}"
    staged_tickers = sorted(
        f.replace("_master.csv", "")
        for f in os.listdir(PHASE_DIR)
        if f.endswith("_master.csv")
    )
    print(f"Loading {len(staged_tickers)} staged tickers from {PHASE_DIR} ...")

    frames = []
    for t in staged_tickers:
        df = pd.read_csv(os.path.join(PHASE_DIR, f"{t}_master.csv"), low_memory=False)
        if df.empty:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        df[target_col] = (df["close"].shift(-horizon) - df["close"]) / df["close"]
        df.dropna(subset=[target_col], inplace=True)
        frames.append(df)

    pooled = pd.concat(frames, ignore_index=True)
    pooled["date"] = pd.to_datetime(pooled["date"])

    n_before = len(pooled)
    outlier_mask = pooled[target_col].abs() > max_abs_return
    pooled = pooled[~outlier_mask].copy()
    print(f"Dropped {outlier_mask.sum()} bad-tick rows ({outlier_mask.sum() / n_before * 100:.3f}% of {n_before}) "
          f"with |{horizon}-day return| > {max_abs_return:.0%}")

    ticker_sectors = get_ticker_sectors()

    raw_feature_cols = get_feature_columns(pooled, ["date", "created_at", target_col, "close"])
    search_trend_cols = [c for c in raw_feature_cols if c.startswith("search_trend_") and c != "search_trend_kse"]

    search_trend_own = pd.Series(np.nan, index=pooled.index, dtype="float64")
    for t in pooled["ticker"].unique():
        col = f"search_trend_{t.lower()}"
        if col in pooled.columns:
            mask = pooled["ticker"].values == t
            search_trend_own.loc[mask] = pooled.loc[mask, col]
    pooled["search_trend_own_ticker"] = search_trend_own
    print(f"Consolidated {len(search_trend_cols)} sparse search_trend_<ticker> columns -> search_trend_own_ticker")

    candidate_cols = [c for c in raw_feature_cols if c not in search_trend_cols and c != "ticker"]
    max_nunique_per_day = pooled.groupby("date")[candidate_cols].nunique().max()
    global_cols = max_nunique_per_day[max_nunique_per_day <= 1].index.tolist()
    print(f"Dropping {len(global_cols)} of {len(candidate_cols)} candidate columns "
          f"(identical across all tickers on every date - no cross-sectional information for a pooled model)")

    dead_technical_cols = DEAD_SHORT_TERM_TECHNICAL_COLS if horizon == 20 else []
    if dead_technical_cols:
        present = [c for c in dead_technical_cols if c in candidate_cols]
        print(f"Dropping {len(present)} short-term technicals confirmed dead at horizon=20: {present}")
    else:
        present = []

    redundant_vol_cols = [c for c in REDUNDANT_VOLATILITY_COLS if c in candidate_cols]
    if redundant_vol_cols:
        print(f"Dropping {len(redundant_vol_cols)} redundant volatility column(s) "
              f"(r>0.98 with a kept sibling): {redundant_vol_cols}")

    exclude_cols = (["date", "created_at", target_col, "close"] + search_trend_cols + global_cols
                    + dead_technical_cols + redundant_vol_cols)
    feature_cols = get_feature_columns(pooled, exclude_cols)

    export_df = pooled[feature_cols].copy()
    export_df["sector"] = export_df["ticker"].map(ticker_sectors)
    export_df["date"] = pooled["date"].values

    if "pe_ratio" in export_df.columns:
        export_df["sector_relative_pe_percentile"] = (
            export_df.groupby(["date", "sector"])["pe_ratio"].rank(pct=True, na_option="keep")
        )
        print("Added sector_relative_pe_percentile (valuation rank vs. same-sector peers on the same date)")

    was_missing_cols = [c for c in WAS_MISSING_PROXY_COLS if c in export_df.columns]
    for col in was_missing_cols:
        export_df[f"{col}_was_missing"] = (export_df[col] == 0.0).astype("int8")
    print(f"Added {len(was_missing_cols)} *_was_missing proxy flags: {was_missing_cols}")

    cross_sectional_cols = [c for c in CROSS_SECTIONAL_FUNDAMENTALS_COLS if c in export_df.columns]
    for col in cross_sectional_cols:
        export_df[col] = export_df.groupby("date")[col].rank(pct=True, na_option="keep")
    print(f"Converted {len(cross_sectional_cols)} raw-scale fundamentals to cross-sectional "
          f"percentile rank: {cross_sectional_cols}")

    num_cols = export_df.select_dtypes(include=["number"]).columns
    export_df[num_cols] = export_df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    export_df[num_cols] = export_df[num_cols].clip(lower=-1e9, upper=1e9)

    export_df["close"] = pooled["close"].values
    export_df[target_col] = pooled[target_col].values

    sorted_dates = export_df["date"].sort_values().reset_index(drop=True)
    cutoff_date = sorted_dates.iloc[int(len(sorted_dates) * 0.8)]
    export_df["split"] = np.where(export_df["date"] < cutoff_date, "train", "test")

    lead_cols = ["ticker", "sector", "date", "split", "close", target_col]
    export_df = export_df[lead_cols + [c for c in export_df.columns if c not in lead_cols]]

    out_path = args.out or os.path.join(PHASE_DIR, f"phase1_model_input_horizon{horizon}.parquet")
    export_df.to_parquet(out_path, index=False)
    print(f"\nExported {export_df.shape[0]} rows x {export_df.shape[1]} columns -> {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1e6:.1f} MB")
    print(f"Split cutoff date: {cutoff_date.date()}  "
          f"(train: {(export_df['split'] == 'train').sum()} rows, test: {(export_df['split'] == 'test').sum()} rows)")


if __name__ == "__main__":
    main()
