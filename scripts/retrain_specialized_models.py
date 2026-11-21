"""
Refits the PSO and MEBL specialized 1-day models on all data through today.

  python scripts/retrain_specialized_models.py [PSO MEBL ...]

The bundles (models/<ticker>_1day_best_model.pkl) were first built in the
per-ticker notebooks. Each one stores its model, the exact feature columns and
the walk-forward evaluation, so this refits a fresh copy of the same model -
same class and hyperparameters - on the same cleaned data (prepare_data with
exclude_known_bad_dates=True, which reproduces the original row counts).
The walk-forward summary is kept as-is: it describes the model choice, which
a refit doesn't change.
"""
import os
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.base import clone

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.psx_predictor.models.feature_wrapper import prepare_data

MODELS_DIR = os.path.join(ROOT_DIR, "models")
DEFAULT_TICKERS = ["PSO", "MEBL"]
# A feature that hasn't changed over this many recent rows (~2 trading years)
# can't inform today's prediction.
FROZEN_ROWS = 500
DUPLICATE_CORR = 0.98


def select_features(X, y, base_model):
    """Columns worth keeping for one ticker's model, decided from the rows
    given (the training window only, when evaluating): drops columns that are
    constant, frozen over the last ~2 years (e.g. another sector's index),
    near-duplicates of a column already kept (|rank corr| > 0.98), and those
    the model then never splits on."""
    cols = [c for c in X.columns if X[c].nunique() > 1 and X[c].iloc[-FROZEN_ROWS:].nunique() > 1]
    corr = X[cols].corr(method="spearman").abs()
    keep = []
    for c in cols:
        if not keep or corr.loc[c, keep].max() < DUPLICATE_CORR:
            keep.append(c)
    model = clone(base_model).fit(X[keep], y)
    return [c for c, imp in zip(keep, model.feature_importances_) if imp > 0]


def retrain(ticker):
    path = os.path.join(MODELS_DIR, f"{ticker.lower()}_1day_best_model.pkl")
    bundle = joblib.load(path)

    X, y, dates, _ = prepare_data(ticker, horizon=bundle.get("horizon", 1), exclude_known_bad_dates=True)
    # Columns since retired from the master CSVs (build_features.DEAD_COLUMNS)
    # are dropped rather than zero-filled.
    feature_cols = [c for c in bundle["feature_cols"] if c in X.columns]
    dropped = sorted(set(bundle["feature_cols"]) - set(feature_cols))
    if dropped:
        print(f"{ticker}: dropping retired columns {dropped}")
    if PRUNE:
        X = X[feature_cols].fillna(0.0)
        feature_cols = select_features(X, y, bundle["model"])
        print(f"{ticker}: kept {len(feature_cols)} of {X.shape[1]} features")
    X = X[feature_cols]
    model = clone(bundle["model"])
    model.fit(X, y)

    previous_rows = bundle.get("trained_on_rows")
    bundle.update({
        "model": model,
        "feature_cols": feature_cols,
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "trained_on_rows": len(X),
        "trained_through": pd.to_datetime(dates).max().date().isoformat(),
    })
    joblib.dump(bundle, path)
    print(f"{ticker}: refit {bundle['model_type']} on {len(X)} rows "
          f"(was {previous_rows}), data through {bundle['trained_through']}.")


# --prune: drop constant / frozen / duplicate / unused columns (select_features)
PRUNE = "--prune" in sys.argv

if __name__ == "__main__":
    for t in [a.upper() for a in sys.argv[1:] if not a.startswith("--")] or DEFAULT_TICKERS:
        retrain(t)
