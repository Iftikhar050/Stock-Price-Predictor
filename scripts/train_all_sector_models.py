import os
import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.psx_predictor.models.feature_wrapper import prepare_data
from src.psx_predictor.models.utils import (choose_global_cutoff, refit_on_all_data, evaluate_only,
                                            directional_hits, save_holdout_accuracy)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AllSectorsModelTrainer")

# Define PSX Sector Dictionary
SECTOR_DICTIONARY = {
    "banking": ["ABL", "AKBL", "BAHL", "BAFL", "BOP", "FABL", "HBL", "HMB", "MCB", "MEBL", "NBP", "SCBPL", "UBL"],
    "cement": ["BWCL", "CHCC", "DGKC", "FCCL", "KOHC", "LUCK", "MLCF", "POWER", "PIOC"],
    "oil_exploration": ["MARI", "OGDC", "POL", "PPL"],
    "oil_marketing": ["APL", "PSO", "WAFI", "SNGP", "SSGC"],
    "technology": ["AIRLINK", "AVN", "HUMNL", "PTC", "SYS", "TRG"],
    "automobile": ["ATLH", "GAL", "GHNI", "HCAR", "INDU", "MTL", "SAZEW", "THALL"],
    "fertilizer": ["EFERT", "FATIMA", "FFC"],
    "pharmaceuticals": ["ABOT", "AGP", "CPHL", "HALEON", "GLAXO", "HINOON", "SEARL"],
    "power": ["HUBC", "KEL", "KAPCO"],
    "food": ["COLG", "FFL", "MUREB", "NATF", "NESTLE", "RMPL", "UPFL"],
    "investment_banks": ["AHCL", "ENGROH", "PSX"],
    # Added 2026-10-10: ASTL/MUGHAL (engineering), EPCL (chemical), GATM/NML
    # (textile), NRL/PRL (refinery) - the 7 legacy orphaned tickers that were
    # outside the maintained 99-ticker set, now promoted into phase_1 and
    # folded into their real PSX sector buckets (stock_metadata.sector).
    "refinery": ["ATRL", "CNERGY", "NRL", "PRL"],
    "textile": ["ILP", "KTML", "MEHT", "GADT", "YOUW", "IBFL", "GATM", "NML"],
    # Added 2026-10-09: these 22 KSE-100 tickers previously had no specialized/SPX
    # model at all (no sector bucket covered them) - see scripts/train_missing_sector_models.py
    # for the one-off run that first produced these; folded in here so future full
    # retrains keep covering them too.
    "cable_electrical": ["PAEL"],
    "chemical": ["LOTCHEM", "LCI", "EPCL"],
    "closed_end_funds": ["HGFA"],
    "engineering": ["INIL", "ISL", "ASTL", "MUGHAL"],
    "glass_ceramics": ["GHGL", "TGL"],
    "insurance": ["AICL"],
    "leasing": ["PGLC"],
    "leather_tanneries": ["SRVI"],
    "misc_other": ["PABC", "PSEL", "SHFA"],
    "modarabas": ["FHAM"],
    "paper_packaging": ["PKGS"],
    "property": ["JVDC"],
    "reit": ["DCR"],
    "sugar": ["JDWS"],
    "tobacco": ["PAKT"],
    "transport": ["PIBTL"],
    "woollen": ["BNWM"],
}

# Tickers with custom specialized 1-day models already trained & managed
EXCLUDED_SPECIALIZED_TICKERS = {"MEBL", "PSO"}

MODELS_DIR = os.path.join(ROOT_DIR, "models")
SECTOR_MODELS_DIR = os.path.join(ROOT_DIR, "models", "sector_specific")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
os.makedirs(SECTOR_MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def train_sector_model(sector_key: str, tickers: list, cutoff_date: pd.Timestamp):
    logger.info(f"--- Training Sector Model: {sector_key.upper()} ({len(tickers)} Tickers) ---")
    
    raw_dfs = {}
    for ticker in tickers:
        try:
            # exclude_known_bad_dates=True drops rows whose forward-return window
            # spans a volume==0 (stale/non-trading) day - without it those rows
            # inject fake zero-return labels that bias the model toward "flat"
            # and understate real directional signal. Matches the convention
            # established for the MEBL/PSO specialized models and the pooled
            # KSE-100 retrain - see models/feature_wrapper.py prepare_data().
            X, y, dates, current_close = prepare_data(ticker, exclude_known_bad_dates=True)
            raw_dfs[ticker] = (X, y, dates, current_close)
        except Exception as e:
            logger.warning(f"  [{sector_key}] Could not load feature matrix for {ticker}: {e}")

    if not raw_dfs:
        logger.error(f"  [{sector_key}] No valid ticker datasets found.")
        return None

    # Get common numeric features across sector tickers
    first_ticker = list(raw_dfs.keys())[0]
    numeric_cols = list(raw_dfs[first_ticker][0].select_dtypes(include=['number']).columns)

    X_tr_list, X_te_list = [], []
    y_tr_list, y_te_list = [], []
    close_te_list = []
    
    ticker_scalers = {}
    valid_sector_tickers = []
    ticker_data = {}
    full_rows = {}  # every row per ticker, for the refit-on-all-data step

    for ticker, (X, y, dates, current_close) in raw_dfs.items():
        dates_dt = pd.to_datetime(dates)
        train_mask = dates_dt <= cutoff_date
        test_mask = dates_dt > cutoff_date

        if train_mask.sum() < 150 or test_mask.sum() < 30:
            logger.warning(f"  [{sector_key}] {ticker} has insufficient data split. Skipping.")
            continue

        valid_sector_tickers.append(ticker)

        X_sub = X.reindex(columns=numeric_cols, fill_value=0.0).fillna(0.0).replace([np.inf, -np.inf], 0.0)

        X_tr = X_sub[train_mask].copy()
        X_te = X_sub[test_mask].copy()
        full_rows[ticker] = (X_sub, y)

        # Fit per-ticker scaler
        scaler = StandardScaler()
        X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr), columns=numeric_cols, index=X_tr.index)
        X_te_scaled = pd.DataFrame(scaler.transform(X_te), columns=numeric_cols, index=X_te.index)

        # One-hot encoding for tickers within sector
        for sec_ticker in tickers:
            X_tr_scaled[f"is_ticker_{sec_ticker}"] = 1.0 if sec_ticker == ticker else 0.0
            X_te_scaled[f"is_ticker_{sec_ticker}"] = 1.0 if sec_ticker == ticker else 0.0

        ticker_scalers[ticker] = scaler

        X_tr_list.append(X_tr_scaled)
        X_te_list.append(X_te_scaled)
        y_tr_list.append(y[train_mask])
        y_te_list.append(y[test_mask])
        close_te_list.append(current_close[test_mask])

        ticker_data[ticker] = {
            "dates_test": dates[test_mask].reset_index(drop=True),
            "X_test_scaled": X_te_scaled.reset_index(drop=True),
            "y_test": y[test_mask].reset_index(drop=True),
            "close_test": current_close[test_mask].reset_index(drop=True)
        }

    if not X_tr_list:
        logger.error(f"  [{sector_key}] No tickers passed validation.")
        return None

    X_train = pd.concat(X_tr_list, ignore_index=True)
    X_test = pd.concat(X_te_list, ignore_index=True)
    y_train = pd.concat(y_tr_list, ignore_index=True)
    y_test = pd.concat(y_te_list, ignore_index=True)
    close_test_all = pd.concat(close_te_list, ignore_index=True)

    actual_prices = close_test_all.values * (1.0 + y_test.values)
    baseline_pred_prices = close_test_all.values

    # Metrics Baseline (naive persistence: predict no change, return=0)
    base_mae = mean_absolute_error(actual_prices, baseline_pred_prices)
    base_mape = mean_absolute_percentage_error(actual_prices, baseline_pred_prices) * 100
    # A flat (return=0) prediction has no sign of its own, so "directional accuracy"
    # isn't a meaningful comparison for it - np.mean((y_test>0)==(0>0)) previously
    # computed here was a bug: (0>0) is always False, so it silently measured the
    # fraction of non-positive-return days in the test set, not real baseline skill.
    base_dir = None

    # Train LightGBM
    lgbm_model = LGBMRegressor(
        n_estimators=180,
        learning_rate=0.02,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1
    )
    lgbm_model.fit(X_train, y_train)
    lgbm_preds_return = lgbm_model.predict(X_test)
    lgbm_preds_prices = close_test_all.values * (1.0 + lgbm_preds_return)

    lgbm_mae = mean_absolute_error(actual_prices, lgbm_preds_prices)
    lgbm_mape = mean_absolute_percentage_error(actual_prices, lgbm_preds_prices) * 100
    lgbm_dir = np.mean(np.sign(y_test.values) == np.sign(lgbm_preds_return)) * 100

    # Train XGBoost
    xgb_model = XGBRegressor(
        n_estimators=180,
        learning_rate=0.02,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds_return = xgb_model.predict(X_test)
    xgb_preds_prices = close_test_all.values * (1.0 + xgb_preds_return)

    xgb_mae = mean_absolute_error(actual_prices, xgb_preds_prices)
    xgb_mape = mean_absolute_percentage_error(actual_prices, xgb_preds_prices) * 100
    xgb_dir = np.mean(np.sign(y_test.values) == np.sign(xgb_preds_return)) * 100

    # Select Best Model by directional accuracy, not MAE - MAE rewards a model
    # that hugs "no change" (small error on most days) even if it rarely calls
    # direction correctly, which is the metric this project actually cares about
    # for trading decisions.
    if lgbm_dir >= xgb_dir:
        best_model = lgbm_model
        best_model_name = "LightGBM"
        best_mae = lgbm_mae
        best_mape = lgbm_mape
        best_dir = lgbm_dir
    else:
        best_model = xgb_model
        best_model_name = "XGBoost"
        best_mae = xgb_mae
        best_mape = xgb_mape
        best_dir = xgb_dir

    logger.info(f"  [{sector_key.upper()}] Baseline MAE: Rs. {base_mae:.2f} | LightGBM DirAcc: {lgbm_dir:.2f}% | XGBoost DirAcc: {xgb_dir:.2f}% | Chosen: {best_model_name} (DirAcc {best_dir:.2f}%, MAE Rs.{best_mae:.2f})")

    # Per-ticker held-out hits from the chosen train-only fit, for the website's
    # accuracy-on-unseen-days figure (saved once for all sectors in main()).
    holdout_rows = ([], [], [])
    for t, d in ticker_data.items():
        preds = best_model.predict(d["X_test_scaled"][X_train.columns])
        holdout_rows[0].extend([t] * len(preds))
        holdout_rows[1].extend(directional_hits(d["y_test"], preds))
        holdout_rows[2].extend(d["dates_test"])

    # The metrics above come from the train-only fit. The saved model is a
    # fresh copy of the chosen one refit on every row, with each ticker's
    # scaler re-fit on all of that ticker's rows to match.
    trained_rows = len(X_train)
    if refit_on_all_data() and not evaluate_only():
        X_all_list, y_all_list, ticker_scalers = [], [], {}
        for ticker, (X_full, y_full) in full_rows.items():
            scaler = StandardScaler()
            X_full_scaled = pd.DataFrame(scaler.fit_transform(X_full), columns=numeric_cols, index=X_full.index)
            for sec_ticker in tickers:
                X_full_scaled[f"is_ticker_{sec_ticker}"] = 1.0 if sec_ticker == ticker else 0.0
            ticker_scalers[ticker] = scaler
            X_all_list.append(X_full_scaled)
            y_all_list.append(y_full)
        X_all = pd.concat(X_all_list, ignore_index=True)[X_train.columns]
        y_all = pd.concat(y_all_list, ignore_index=True)
        best_model = clone(best_model)
        best_model.fit(X_all, y_all)
        trained_rows = len(X_all)
        logger.info(f"  [{sector_key.upper()}] Refit {best_model_name} on all {trained_rows} rows (train + test).")

    # Save Sector Model Artifacts
    if not evaluate_only():
        sector_dir = os.path.join(MODELS_DIR, f"{sector_key}_sector")
        os.makedirs(sector_dir, exist_ok=True)

        joblib.dump(best_model, os.path.join(sector_dir, f"{sector_key}_model.pkl"))
        joblib.dump(ticker_scalers, os.path.join(sector_dir, f"{sector_key}_scalers.pkl"))
        joblib.dump(numeric_cols, os.path.join(sector_dir, f"{sector_key}_feature_cols.pkl"))

    return {
        "sector_key": sector_key,
        "holdout_rows": holdout_rows,
        "valid_tickers": valid_sector_tickers,
        "best_model_name": best_model_name,
        "saved_model_trained_on_rows": trained_rows,
        "saved_model_refit_on_all_data": refit_on_all_data(),
        "baseline_mae": round(float(base_mae), 2),
        "baseline_mape": round(float(base_mape), 2),
        "best_model_mae": round(float(best_mae), 2),
        "best_model_mape": round(float(best_mape), 2),
        "directional_accuracy": round(float(best_dir), 2),
        "lightgbm_dir_acc": round(float(lgbm_dir), 2),
        "xgboost_dir_acc": round(float(xgb_dir), 2),
    }

def main():
    logger.info("===============================================================")
    logger.info("   MASTER AUTOMATED PSX SECTOR-SPECIFIC MODEL TRAINING        ")
    logger.info("===============================================================")
    
    cutoff_str, _ = choose_global_cutoff(test_trading_days=250, min_train_trading_days=500)
    cutoff_date = pd.to_datetime(cutoff_str)
    logger.info(f"Global Cutoff Date: {cutoff_str}")

    results = {}
    for sector_key, tickers in SECTOR_DICTIONARY.items():
        res = train_sector_model(sector_key, tickers, cutoff_date)
        if res:
            results[sector_key] = res

    rows = [r.pop("holdout_rows") for r in results.values()]
    save_holdout_accuracy("sector", [t for r in rows for t in r[0]], [h for r in rows for h in r[1]],
                          [d for r in rows for d in r[2]])

    if evaluate_only():
        logger.info("EVALUATE_ONLY=1 - held-out accuracy recorded, no sector models or report saved.")
        return
    report_path = os.path.join(REPORTS_DIR, "all_sectors_model_evaluation.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Successfully trained {len(results)} sector models! Summary saved to {report_path}")

if __name__ == "__main__":
    main()
