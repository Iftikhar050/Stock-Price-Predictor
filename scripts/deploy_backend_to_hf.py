"""
Pushes the API to a Hugging Face Space (Docker) and sets its secrets.

  python scripts/deploy_backend_to_hf.py <hf-username>/<space-name>

Needs HF_TOKEN (a Hugging Face access token with write permission), from the
environment or deploy/.env.production. Secrets are read from
deploy/.env.production (gitignored - copy deploy/env.production.example) and
stored as Space secrets, never committed; HF_TOKEN itself is never sent to
the Space. Uploads only what the image
needs: code, the models the API loads, and the container files. Feature data
is not uploaded here - the container downloads it from Supabase Storage.
"""
import glob
import os
import sys

from dotenv import dotenv_values
from huggingface_hub import HfApi

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROD_ENV = os.path.join(ROOT_DIR, "deploy", ".env.production")

REQUIRED_SECRETS = [
    "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD",
    "SUPABASE_URL", "SUPABASE_SERVICE_KEY", "ALLOWED_ORIGINS", "ADMIN_API_KEY",
]
OPTIONAL_SECRETS = ["DB_SSLMODE", "DB_POOL_SIZE", "DB_MAX_OVERFLOW", "SERVING_BUCKET"]

MODEL_FILES = [
    "baseline_rf_model.pkl", "lr_model.pkl", "xgboost_model.pkl",
    "feature_scaler_lstm.pkl", "target_scaler_lstm.pkl", "ticker_to_id.pkl", "sector_to_id.pkl",
    "xgb_ticker_categories.pkl", "xgb_sector_categories.pkl", "lstm_model.pth",
    "mebl_1day_best_model.pkl", "pso_1day_best_model.pkl",
]


def upload_patterns():
    models = [f"models/{f}" for f in MODEL_FILES]
    for suffix in ("_model.pkl", "_scalers.pkl", "_feature_cols.pkl"):
        models += [p.replace(os.sep, "/") for p in glob.glob(os.path.join("models", "*_sector", f"*{suffix}"))]
    return [
        "Dockerfile", ".dockerignore", "requirements-api.txt",
        "deploy/start.sh", "scripts/serving_data_storage.py", "src/**/*.py",
        *models,
    ]


def main():
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit(__doc__)
    space_id = sys.argv[1]
    if not os.path.exists(PROD_ENV):
        sys.exit(f"Missing {PROD_ENV} - copy deploy/env.production.example and fill it in.")

    secrets = {k: v for k, v in dotenv_values(PROD_ENV).items() if v}
    token = os.environ.get("HF_TOKEN") or secrets.get("HF_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN (environment or deploy/.env.production) to a Hugging Face write token.")
    missing = [k for k in REQUIRED_SECRETS if k not in secrets]
    if missing:
        sys.exit(f"deploy/.env.production is missing: {', '.join(missing)}")

    os.chdir(ROOT_DIR)
    api = HfApi(token=token)
    api.create_repo(space_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)

    for key in REQUIRED_SECRETS + OPTIONAL_SECRETS:
        if key in secrets:
            api.add_space_secret(space_id, key, secrets[key])

    api.upload_folder(
        repo_id=space_id, repo_type="space", folder_path=ROOT_DIR,
        allow_patterns=upload_patterns(), ignore_patterns=["**/__pycache__/**"],
        commit_message="Deploy API",
    )
    api.upload_file(
        repo_id=space_id, repo_type="space", path_or_fileobj=os.path.join("deploy", "hf_space_README.md"),
        path_in_repo="README.md", commit_message="Space metadata",
    )
    print(f"Deployed. Build logs: https://huggingface.co/spaces/{space_id}")
    print(f"API URL once built:  https://{space_id.replace('/', '-').replace('_', '-').lower()}.hf.space")


if __name__ == "__main__":
    main()
