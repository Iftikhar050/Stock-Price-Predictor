import os
import sys
import subprocess
import requests

PYTHON_EXEC = sys.executable

print("========================================")
print("   STARTING ML MODEL RETRAINING         ")
print("========================================")

print("\n[1/5] Retraining Random Forest Baseline...")
try:
    subprocess.run([PYTHON_EXEC, "src/psx_predictor/models/train_baseline.py"], check=True)
except subprocess.CalledProcessError:
    print("[ERROR] Error during Random Forest training. Exiting.")
    sys.exit(1)

print("\n[2/5] Retraining Linear Regression Model...")
try:
    subprocess.run([PYTHON_EXEC, "src/psx_predictor/models/train_regression.py"], check=True)
except subprocess.CalledProcessError:
    print("[ERROR] Error during Linear Regression training. Exiting.")
    sys.exit(1)

print("\n[3/5] Retraining XGBoost Model...")
try:
    subprocess.run([PYTHON_EXEC, "src/psx_predictor/models/train_xgboost.py"], check=True)
except subprocess.CalledProcessError:
    print("[ERROR] Error during XGBoost training. Exiting.")
    sys.exit(1)

print("\n[4/5] Retraining Deep Learning LSTM (This may take a minute)...")
try:
    subprocess.run([PYTHON_EXEC, "src/psx_predictor/models/train_lstm.py"], check=True)
except subprocess.CalledProcessError:
    print("[ERROR] Error during LSTM training. Exiting.")
    sys.exit(1)

print("\n[5/5] Hot-reloading new AI models into FastAPI Server...")
try:
    api_key = os.environ.get("ADMIN_API_KEY", "local-dev-key")
    response = requests.post("http://127.0.0.1:8000/api/reload_models", headers={"X-API-Key": api_key})
    if response.status_code == 200:
        print("[SUCCESS] " + response.json().get("message", "Models reloaded!"))
    else:
        print(f"[WARNING] Server returned status {response.status_code}. Is FastAPI running?")
except Exception as e:
    print(f"[WARNING] Could not connect to FastAPI server to reload models. Error: {e}")
    print("If your server isn't running, the new models will load automatically next time you start it.")

print("\n========================================")
print("[SUCCESS] Full Retraining Pipeline Complete!")
print("========================================")
