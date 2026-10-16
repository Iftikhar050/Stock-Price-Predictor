#!/bin/sh
# Container entrypoint: pull the latest trimmed feature tables from Supabase
# Storage, then serve the API on the port Hugging Face Spaces expects.
set -e

export SERVING_DATA_DIR="${SERVING_DATA_DIR:-$HOME/app/data/serving}"
python scripts/serving_data_storage.py download "$SERVING_DATA_DIR"

exec uvicorn src.psx_predictor.api.main:app --host 0.0.0.0 --port "${PORT:-7860}"
