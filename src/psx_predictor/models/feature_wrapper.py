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


DEAD_COLUMNS_REPORT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "dead_columns_report.md")
)

# Fallback snapshot (2026-09, 107-ticker audit) used only if the report file is
# missing - the report itself, when present, is the live source of truth so this
# list doesn't silently go stale as new tickers are audited.
_DEAD_CONSTANT_COLS_FALLBACK = [
    "adr_ratio", "auto_sales_total", "bonus_event", "capital_adequacy_ratio",
    "casa_deposits", "casa_ratio", "cement_dispatches_mt", "circular_debt_level",
    "cpi_energy", "cpi_food", "cpi_housing", "current_account_balance",
    "electricity_gen_gwh", "exports_usd_m", "forward_usd_pkr_3m", "free_float",
    "free_float_pct", "geopolitical_news_sentiment_3d", "government_receivables",
    "hit_lower_circuit", "hit_upper_circuit", "idr_ratio", "imf_export_volume_growth",
    "imf_govt_expenditure_pct_gdp", "imf_govt_revenue_pct_gdp", "imf_import_volume_growth",
    "imf_investment_pct_gdp", "imf_national_savings_pct_gdp", "imf_net_financial_position",
    "imf_quota_sdrs", "imf_sdr_allocation_bal", "imf_sdr_holdings_bal",
    "imf_total_loans_outstanding", "imf_tranche_disbursements", "imports_usd_m",
    "insider_buy_shares_30d", "insider_net_flow_30d", "insider_sell_shares_30d",
    "institutional_holding_pct", "is_synthetic_rate", "lsm_growth", "m2_money",
    "macro_news_sentiment_3d", "major_contract_event", "market_number_of_trades",
    "net_interest_margin", "new_highs", "new_lows", "npl_ratio",
    "pakistan_activity_is_synthetic", "palm_oil_price", "petroleum_sales_volume",
    "plant_shutdown_event", "political_news_sentiment_3d", "provisioning_coverage",
    "refinery_margin", "reserve_import_coverage", "sbp_additional_is_synthetic",
    "sbp_omo_net_outstanding", "sector_breadth", "sector_news_sentiment_3d",
    "sector_news_sentiment_count", "sector_pb_avg", "sector_pe_avg",
    "sponsor_holding_pct", "stock_relative_strength_sector", "total_advances",
    "total_deposits", "trade_deficit", "trade_deficit_expected", "trade_deficit_surprise",
    "trade_deficit_usd_m", "wheat_procurement_mt", "wpi_index",
]


def get_dead_constant_columns() -> list:
    """Columns confirmed constant (non-NaN) across every ticker in
    data/dead_columns_report.md's 107-ticker audit - pure zero-variance noise for
    any model, pooled or single-ticker. Parses the report so this list tracks
    re-audits automatically instead of drifting out of sync with a hardcoded copy;
    falls back to a fixed snapshot only if the report is missing.

    This is exported for callers to opt into explicitly via get_feature_columns()'s
    `exclude_cols` - it is NOT applied automatically inside get_feature_columns()
    itself, so existing callers (predict.py, train_*.py, the pooled export script)
    keep their current behavior unless they choose to pass this in.
    """
    if not os.path.exists(DEAD_COLUMNS_REPORT_PATH):
        return list(_DEAD_CONSTANT_COLS_FALLBACK)
    cols = []
    in_section = False
    with open(DEAD_COLUMNS_REPORT_PATH, encoding="utf-8") as f:
        for line in f:
            if line.startswith("## Constant (non-NaN) in every ticker"):
                in_section = True
                continue
            if in_section:
                if line.startswith("## "):
                    break
                stripped = line.strip()
                if stripped.startswith("- `") and stripped.endswith("`"):
                    cols.append(stripped[3:-1])
    return cols or list(_DEAD_CONSTANT_COLS_FALLBACK)


# Raw, non-stationary LEVEL features (absolute price/index/commodity levels or
# unscaled fundamentals) that a model can't compare meaningfully across a
# training period that may span a very different price regime than the test
# period - confirmed via get_feature_columns()'s own numeric-dtype scan that each
# of these is present as a raw level, not already a pct-change/distance feature.
# Mapped to the already-engineered normalized counterpart that exists in the same
# CSV where one does (prefer that instead of the level); left unmapped (drop with
# no substitute) where no normalized version of that specific series exists.
RAW_LEVEL_FEATURE_COLS = [
    "adjusted_close",  # 100% identical to `close` on MEBL/PSO - pure redundancy
    "market_cap", "ebitda", "revenue", "ev", "total_assets", "total_debt",
    "brent_oil_price", "wti_oil_price", "sp500_close", "nasdaq_close", "dxy_close",
    "gold_price", "copper_price", "coal_price", "cotton_price", "gas_price",
    "nikkei_close", "hang_seng_close", "shanghai_close", "ftse_close", "dax_close",
    "vix_close", "kmi30_index_level", "kse30_index_level", "all_share_index_level",
    "banking_sector_index_level", "oil_gas_sector_index_level",
    "aluminum_price", "wheat_price", "soybean_price", "msci_em_close", "msci_fm_close",
    "dow_jones_close", "steel_price", "iron_ore_price", "urea_price", "lng_price",
    "tips_etf_price",
]

# level -> its already-engineered normalized counterpart, for callers that want to
# confirm the substitute is actually present before dropping the raw column.
LEVEL_TO_NORMALIZED_COUNTERPART = {
    "brent_oil_price": "oil_return_pct", "wti_oil_price": "oil_return_pct",
    "sp500_close": "sp500_return_pct", "nasdaq_close": "nasdaq_return_pct",
    "dxy_close": "dxy_return_pct", "gold_price": "gold_return_pct",
    "copper_price": "copper_return_pct", "coal_price": "coal_return_pct",
    "cotton_price": "cotton_return_pct", "gas_price": "gas_return_pct",
    "nikkei_close": "nikkei_return_pct", "hang_seng_close": "hang_seng_return_pct",
    "shanghai_close": "shanghai_return_pct", "ftse_close": "ftse_return_pct",
    "dax_close": "dax_return_pct", "vix_close": "vix_return_pct",
    "kmi30_index_level": "kmi30_return_pct", "kse30_index_level": "kse30_return_pct",
    "all_share_index_level": "all_share_return_pct",
    "banking_sector_index_level": "banking_sector_return_pct",
    "oil_gas_sector_index_level": "oil_gas_sector_return_pct",
}


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


# NOTE on the earlier, now-superseded version of this fix: a first pass dropped a
# fixed list of single "bad dates" (volume==0 rows whose forward return looked
# extreme). That was wrong - re-checked directly against the raw volume series and
# found the anomalies aren't isolated single days, they're the EDGES of long
# contiguous volume==0 runs (MEBL alone has one 189-trading-day run, 2009-06-02 to
# 2010-02-19 - this lines up with PSX's real 2008-2009 crisis-era price-floor/thin-
# trading period, plausibly a genuine market event, not fabricated data). Dropping
# only the flagged edge row doesn't remove the discontinuity - it just relocates the
# "fabricated-looking" jump onto whichever row becomes the new neighbor after the
# gap closes, since pandas' shift() re-aligns positionally, not by calendar date.
# The correct, horizon-aware fix: a row's forward return is only valid if NEITHER
# its own day NOR any day within the horizon window ahead of it has volume==0.


def prepare_data(ticker: str, ticker_sectors: dict = None, horizon: int = 1, dead_zone: float = None,
                  exclude_known_bad_dates: bool = False):
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

    `exclude_known_bad_dates` (default False, opt-in): drops rows whose forward
    return spans a volume==0 (non-trading / stale-price) day anywhere in the
    horizon window - see the module-level note above `prepare_data` for why this
    has to be horizon-aware rather than a fixed date list. Left off by default so
    existing callers don't silently change output; single-ticker MEBL/PSO work
    should pass True.
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

    if exclude_known_bad_dates and 'volume' in df.columns:
        # A row's target is only trustworthy if no day in [t, t+horizon] was a
        # volume==0 stale-price day. Computed on the FULL, untrimmed df (same shape
        # the target itself was computed on) so shift() has the true future rows to
        # look at - doing this after dropna would trim the tail first and make the
        # last `horizon` valid rows look like they "run off the end", incorrectly
        # flagging them as stale by a boundary artifact rather than a real one.
        stale = (df['volume'] == 0)
        spans_stale = pd.Series(False, index=df.index)
        for k in range(0, horizon + 1):
            spans_stale = spans_stale | stale.shift(-k).fillna(False)
        df = df[~spans_stale]

    df.dropna(subset=[target_col], inplace=True)
    df = df.reset_index(drop=True).copy()

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


def bin_volatility_regime(train_vol: pd.Series, apply_vol: pd.Series, n_bins: int = 3) -> tuple:
    """Volatility-regime classification labels (0=low ... n_bins-1=high), fit ONLY on
    `train_vol` (bin edges from its tercile/quartile cut points) and applied to
    `apply_vol` (which may be the same series, or a held-out val/test series - always
    fit on train only, per the acceptance criteria, so no test-period volatility level
    leaks into where the bin edges get drawn).

    Returns (labels_for_apply_vol, bin_edges) - bin_edges is returned so a caller can
    re-apply the exact same cut points elsewhere (e.g. scoring a live/production row).
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = train_vol.quantile(quantiles).to_numpy().copy()
    edges[0], edges[-1] = -np.inf, np.inf  # so no in-range value at the extremes falls outside
    labels = pd.cut(apply_vol, bins=edges, labels=False, include_lowest=True)
    return labels, edges


def prepare_volatility_data(ticker: str, ticker_sectors: dict = None, horizon: int = 1,
                             exclude_known_bad_dates: bool = False):
    """Analogous to prepare_data(), but for the MAGNITUDE target instead of signed
    direction: target_vol_t1 = abs(forward return). Signed next-day return is close to
    unpredictable in this data (see feature_wrapper's signal-audit numbers - lag-1
    autocorrelation ~0.00-0.04, feature correlations mostly <0.10); |return| against the
    same features runs meaningfully higher (0.15-0.27 range for the volatility/range
    features specifically) - this is a first-class second target, not a side experiment,
    reusing the exact same X (features) prepare_data() already builds so the two targets
    stay directly comparable on the same rows/columns.

    Regime classification (tercile bins) is NOT returned here - call
    bin_volatility_regime() separately on the y_vol series AFTER you've done your own
    train/test split, so the bin edges are fit on the training slice only, matching
    every other train/test-fit-only rule already used elsewhere in this pipeline.
    """
    result = prepare_data(ticker, ticker_sectors, horizon=horizon, exclude_known_bad_dates=exclude_known_bad_dates)
    X, y, dates, current_close = result
    y_vol = y.abs().rename(f"target_vol_t{horizon}")
    return X, y_vol, dates, current_close


def feature_fn(ticker: str, start_date: str, end_date: str, ticker_sectors: dict = None, horizon: int = 1,
                dead_zone: float = None, exclude_known_bad_dates: bool = False):
    """Return X, y, dates, close (and optionally target_class) for a ticker
    within the given date range. The returned DataFrames are already filtered
    to the inclusive interval.
    """
    result = prepare_data(ticker, ticker_sectors, horizon=horizon, dead_zone=dead_zone,
                           exclude_known_bad_dates=exclude_known_bad_dates)
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
