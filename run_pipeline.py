import os
import sys
import time
import subprocess
import schedule
import argparse
import requests
import logging

# Ensure root path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PipelineOrchestrator")

# Define Python executable to use the virtual environment
PYTHON_EXE = os.path.join(ROOT_DIR, "venv", "Scripts", "python.exe")

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
        api_key = os.getenv("ADMIN_API_KEY", "super_secret_admin_key")
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
        
    # 6. Reload Models in Production Server
    reload_api_models()
    
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
