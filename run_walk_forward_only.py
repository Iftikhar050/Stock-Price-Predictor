import os
import sys
import uuid
import json
from datetime import datetime, timedelta

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from src.psx_predictor.models.walk_forward import generate_walk_forward_windows, run_walk_forward
from src.psx_predictor.models.model_factories import baseline_factory, ridge_factory, xgboost_factory
from src.psx_predictor.models.feature_wrapper import feature_fn, get_ticker_sectors
from src.psx_predictor.models.registry import register_run
from src.psx_predictor.data.build_features import FEATURE_SET_VERSION
from src.psx_predictor.db.repository import get_active_tickers
from src.psx_predictor.models.train_baseline import MODEL_FILENAME as BASELINE_MODEL_FILENAME
from src.psx_predictor.models.train_regression import MODEL_FILENAME as RIDGE_MODEL_FILENAME
from src.psx_predictor.models.train_xgboost import MODEL_FILENAME as XGBOOST_MODEL_FILENAME

def main():
    tickers = get_active_tickers()
    end_dt = datetime.now().date()
    start_dt = end_dt - timedelta(days=1500)
    windows = generate_walk_forward_windows(start_dt.isoformat(), end_dt.isoformat())
    
    ticker_sectors = get_ticker_sectors()
    
    def _baseline_feature_fn(ticker, start, end):
        X, y, dates, close = feature_fn(ticker, start, end, ticker_sectors=None)
        if 'ticker' in X.columns:
            X = X.drop(columns=['ticker'])
        if 'sector' in X.columns:
            X = X.drop(columns=['sector'])
        return X, y, dates, close

    def _xgb_feature_fn(ticker, start, end):
        X, y, dates, close = feature_fn(ticker, start, end, ticker_sectors=ticker_sectors)
        X['ticker'] = X['ticker'].astype('category')
        X['sector'] = X['sector'].astype('category')
        return X, y, dates, close
        
    print("Running Baseline Walk Forward...")
    baseline_results = run_walk_forward(baseline_factory, _baseline_feature_fn, windows, tickers)
    baseline_id = uuid.uuid4().hex
    register_run(
        run_id=baseline_id,
        model_type="baseline",
        feature_set_version=FEATURE_SET_VERSION,
        ticker_list=tickers,
        results_df=baseline_results,
        model_path=os.path.join(ROOT_DIR, "models", BASELINE_MODEL_FILENAME),
    )
    
    print("Running Ridge Walk Forward...")
    ridge_results = run_walk_forward(ridge_factory, _baseline_feature_fn, windows, tickers)
    ridge_id = uuid.uuid4().hex
    register_run(
        run_id=ridge_id,
        model_type="ridge",
        feature_set_version=FEATURE_SET_VERSION,
        ticker_list=tickers,
        results_df=ridge_results,
        model_path=os.path.join(ROOT_DIR, "models", RIDGE_MODEL_FILENAME),
    )
    
    print("Running XGBoost Walk Forward...")
    xgb_results = run_walk_forward(xgboost_factory, _xgb_feature_fn, windows, tickers)
    xgb_id = uuid.uuid4().hex
    register_run(
        run_id=xgb_id,
        model_type="xgboost",
        feature_set_version=FEATURE_SET_VERSION,
        ticker_list=tickers,
        results_df=xgb_results,
        model_path=os.path.join(ROOT_DIR, "models", XGBOOST_MODEL_FILENAME),
    )
    
    print("All done!")

if __name__ == '__main__':
    main()
