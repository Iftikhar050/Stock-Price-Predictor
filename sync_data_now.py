import os
import sys
import subprocess

PYTHON_EXEC = sys.executable

print("========================================")
print("   STARTING MANUAL DATA SYNC PIPELINE   ")
print("========================================")

print("\n[1/2] Fetching latest market data...")
try:
    subprocess.run([PYTHON_EXEC, "setup_and_sync.py"], check=True)
except subprocess.CalledProcessError:
    print("❌ Error during data scraping. Exiting.")
    sys.exit(1)

print("\n[2/2] Calculating technical indicators...")
try:
    subprocess.run([PYTHON_EXEC, "src/psx_predictor/data/build_features.py"], check=True)
except subprocess.CalledProcessError:
    print("❌ Error during feature engineering. Exiting.")
    sys.exit(1)

print("\n========================================")
print("✅ Data Sync Complete! Your dashboard now has the latest data.")
print("========================================")
