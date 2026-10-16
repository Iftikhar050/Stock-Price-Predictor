"""
Moves the trimmed feature tables (scripts/export_serving_data.py) between a
local folder and a private Supabase Storage bucket.

  python scripts/serving_data_storage.py upload   [dir]   # after the daily pipeline
  python scripts/serving_data_storage.py download [dir]   # on API container start

Needs SUPABASE_URL and SUPABASE_SERVICE_KEY (Project Settings -> API in the
Supabase dashboard). The bucket is created on first upload. A manifest.json is
uploaded last, so a download never sees a half-finished upload's file list.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(ROOT_DIR, "data", "serving")
BUCKET = os.environ.get("SERVING_BUCKET", "serving-data")
MANIFEST = "manifest.json"
WORKERS = 8


def _config():
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        sys.exit("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set.")
    return f"{url}/storage/v1", {"Authorization": f"Bearer {key}", "apikey": key}


def _ensure_bucket(base, headers):
    r = requests.post(f"{base}/bucket", headers=headers, timeout=30,
                      json={"id": BUCKET, "name": BUCKET, "public": False})
    if r.status_code not in (200, 201) and "already exists" not in r.text.lower():
        r.raise_for_status()


def _put(base, headers, name, data, content_type):
    r = requests.post(f"{base}/object/{BUCKET}/{name}", data=data, timeout=120,
                      headers={**headers, "x-upsert": "true", "Content-Type": content_type})
    r.raise_for_status()


def upload(folder):
    base, headers = _config()
    files = sorted(f for f in os.listdir(folder) if f.endswith(".parquet"))
    if not files:
        sys.exit(f"No .parquet files in {folder} - run scripts/export_serving_data.py first.")
    _ensure_bucket(base, headers)

    def send(name):
        with open(os.path.join(folder, name), "rb") as fh:
            _put(base, headers, name, fh.read(), "application/octet-stream")

    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(send, files))
    manifest = {"files": files}
    _put(base, headers, MANIFEST, json.dumps(manifest).encode(), "application/json")
    total = sum(os.path.getsize(os.path.join(folder, f)) for f in files)
    print(f"Uploaded {len(files)} files ({total / 1e6:,.0f} MB) to bucket '{BUCKET}'.")


def download(folder):
    base, headers = _config()
    os.makedirs(folder, exist_ok=True)
    r = requests.get(f"{base}/object/{BUCKET}/{MANIFEST}", headers=headers, timeout=30)
    r.raise_for_status()
    files = r.json()["files"]

    def fetch(name):
        resp = requests.get(f"{base}/object/{BUCKET}/{name}", headers=headers, timeout=120)
        resp.raise_for_status()
        tmp = os.path.join(folder, name + ".part")
        with open(tmp, "wb") as fh:
            fh.write(resp.content)
        os.replace(tmp, os.path.join(folder, name))

    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(fetch, files))
    print(f"Downloaded {len(files)} files from bucket '{BUCKET}' to {folder}.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("upload", "download"):
        sys.exit(__doc__)
    target = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DIR
    upload(target) if sys.argv[1] == "upload" else download(target)
