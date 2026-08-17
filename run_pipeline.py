import os
import sys
import time
import subprocess
import schedule
import argparse
import requests
import logging
from datetime import datetime, timedelta
# Additional imports for walk‑forward, registry & promotion
from src.psx_predictor.models.walk_forward import generate_walk_forward_windows, run_walk_forward
from src.psx_predictor.models.model_factories import baseline_factory, ridge_factory, xgboost_factory
from src.psx_predictor.models.feature_wrapper import feature_fn
from src.psx_predictor.models.registry import register_run, load_all_runs
from src.psx_predictor.models.promotion import select_best_model_overall
from src.psx_predictor.data.build_features import FEATURE_SET_VERSION
from src.psx_predictor.db.repository import get_active_tickers
from src.psx_predictor.models.train_baseline import MODEL_FILENAME as BASELINE_MODEL_FILENAME
from src.psx_predictor.models.train_regression import MODEL_FILENAME as RIDGE_MODEL_FILENAME
from src.psx_predictor.models.train_xgboost import MODEL_FILENAME as XGBOOST_MODEL_FILENAME
import uuid
import shutil
import json
import pandas as pd
# Ensure root path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PipelineOrchestrator")

# Define Python executable to use the current environment
PYTHON_EXE = sys.executable

def run_script(script_path: str):
    """Executes a Python script as a subprocess and streams its output."""
    logger.info(f"--- Starting: {script_path} ---")
    try:
        process = subprocess.Popen(
            [PYTHON_EXE, script_path],
            cwd=ROOT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        for line in process.stdout:
            print(line.strip())
            
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Failed executing {script_path} with exit code {process.returncode}")
            return False
            
        logger.info(f"--- Finished: {script_path} ---")
        return True
    except Exception as e:
        logger.error(f"Error running {script_path}: {e}")
        return False

def reload_api_models():
    """Calls the FastAPI backend to reload ML models from disk."""
    logger.info("Triggering API Model Hot-Reload...")
    try:
        api_key = os.getenv("ADMIN_API_KEY")
        if not api_key:
            raise ValueError("ADMIN_API_KEY environment variable is not set. Cannot authenticate for hot-reload.")
        headers = {"X-API-Key": api_key}
        # Assuming the API is running locally on port 8000
        response = requests.post("http://localhost:8000/api/reload_models", headers=headers, timeout=10)
        response.raise_for_status()
        logger.info(f"API Reload Response: {response.json()}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to reload API models (Is the FastAPI server running?): {e}")
        return False

def execute_full_pipeline():
    """Executes the entire daily ETL and ML pipeline sequentially."""
    logger.info("=========================================")
    logger.info("COMMENCING DAILY FULL-MARKET PIPELINE")
    logger.info("=========================================")
    
    start_time = time.time()
    
    # 1. Scrape & Sync Data
    if not run_script(os.path.join("setup_and_sync.py")):
        logger.error("Pipeline aborted at Data Sync phase.")
        return
        
    # 2. Build Features
    if not run_script(os.path.join("src", "psx_predictor", "data", "build_features.py")):
        logger.error("Pipeline aborted at Feature Engineering phase.")
        return
        
    # 3. Train Baseline & Regression (Ensemble Dependencies)
    if not run_script(os.path.join("src", "psx_predictor", "models", "train_baseline.py")):
        logger.error("Pipeline aborted at Baseline Training phase.")
        return
        
    if not run_script(os.path.join("src", "psx_predictor", "models", "train_regression.py")):
        logger.error("Pipeline aborted at Regression Training phase.")
        return
        
    # 4. Train XGBoost
    if not run_script(os.path.join("src", "psx_predictor", "models", "train_xgboost.py")):
        logger.error("Pipeline aborted at XGBoost Training phase.")
        return
        
    # 5. Train PyTorch LSTM
    # We must run this as a module so it finds `src`
    logger.info(f"--- Starting: LSTM Training ---")
    try:
        process = subprocess.Popen(
            [PYTHON_EXE, "-m", "src.psx_predictor.models.train_lstm"],
            cwd=ROOT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in process.stdout:
            print(line.strip())
        process.wait()
        if process.returncode != 0:
            logger.error("Pipeline aborted at LSTM Training phase.")
            return
        logger.info(f"--- Finished: LSTM Training ---")
    except Exception as e:
        logger.error(f"Error running LSTM Training: {e}")
        return
        
    # 6. Walk‑Forward Evaluation & Registration
    logger.info("--- Starting: Walk‑Forward Evaluation ---")
    try:
        # Determine active tickers for evaluation
        tickers = get_active_tickers()
        # Generate windows covering recent history (approx last 1500 days)
        end_dt = datetime.now().date()
        start_dt = end_dt - timedelta(days=1500)
        windows = generate_walk_forward_windows(start_dt.isoformat(), end_dt.isoformat())
        if not windows:
            logger.warning("No walk‑forward windows generated; skipping evaluation.")
        else:
            from src.psx_predictor.models.feature_wrapper import get_ticker_sectors
            ticker_sectors = get_ticker_sectors()
            
            # wrapper functions to pass ticker_sectors and cast/drop features appropriately
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

            # Baseline
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
            # Ridge
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
            # XGBoost
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
            # Promotion – select best qualifying model
            best_run = select_best_model_overall()
            if best_run:
                # Copy the winning model into production location
                prod_dir = os.path.join(ROOT_DIR, "models", "production")
                os.makedirs(prod_dir, exist_ok=True)
                src_path = best_run["model_path"]
                dst_path = os.path.join(prod_dir, os.path.basename(src_path))
                shutil.copyfile(src_path, dst_path)
                # Write pointer file
                pointer_path = os.path.join(prod_dir, "pointer.json")
                with open(pointer_path, "w") as f:
                    json.dump({"run_id": best_run["run_id"], "model_type": best_run["model_type"]}, f)
                logger.info(f"Promoted model {best_run['model_type']} (run {best_run['run_id']}) to production.")
            else:
                logger.warning("No model met promotion criteria; production model unchanged.")
    except Exception as e:
        logger.error(f"Walk‑forward evaluation failed: {e}")
    # 6. Reload Models in Production Server

    if not reload_api_models():
        logger.warning("=========================================")
        logger.warning("WARNING: Pipeline finished but API models failed to reload!")
        logger.warning("The live server is serving stale predictions.")
        logger.warning("=========================================")
    
    duration = time.time() - start_time
    logger.info("=========================================")
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {duration:.2f} SECONDS")
    logger.info("=========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Predictor Pipeline Orchestrator")
    parser.add_argument("--run-now", action="store_true", help="Execute the pipeline immediately instead of scheduling.")
    args = parser.parse_args()
    
    if args.run_now:
        execute_full_pipeline()
    else:
        logger.info("Pipeline Orchestrator Started. Scheduled to run daily at 17:00 (5:00 PM).")
        # Pakistan Stock Exchange closes at 3:30 PM, so 5:00 PM is a safe time to pull EOD data.
        schedule.every().day.at("17:00").do(execute_full_pipeline)
        
        while True:
            schedule.run_pending()
            time.sleep(60)
