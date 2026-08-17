import os
import json
import uuid
import logging
from datetime import datetime
import pandas as pd
import shutil

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REGISTRY_ROOT = os.path.join(ROOT_DIR, "models", "registry")
os.makedirs(REGISTRY_ROOT, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

def _run_dir(run_id: str) -> str:
    return os.path.join(REGISTRY_ROOT, run_id)

def register_run(
    run_id: str,
    model_type: str,
    feature_set_version: str,
    ticker_list: list,
    results_df: pd.DataFrame,
    model_path: str,
) -> None:
    """Persist a full walk‑forward run.

    * ``run_id`` – unique identifier (UUID or ISO timestamp).
    * ``model_type`` – "baseline", "ridge", "xgboost", "lstm".
    * ``feature_set_version`` – version string defined in ``build_features.py``.
    * ``ticker_list`` – list of tickers used in this run.
    * ``results_df`` – DataFrame with per‑ticker, per‑window metrics.
    * ``model_path`` – path to the trained model artifact (pickle or torch state dict).
    """
    run_folder = _run_dir(run_id)
    os.makedirs(run_folder, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Expected model artifact not found: {model_path}")

    # Save raw results
    results_path = os.path.join(run_folder, "results.parquet")
    results_df.to_parquet(results_path, index=False)

    # Compute summary statistics
    summary = {}
    
    numeric_cols = results_df.select_dtypes(include='number').columns
    overall = {}
    for col in numeric_cols:
        overall[f"{col}_mean"] = results_df[col].mean()
        overall[f"{col}_std"] = results_df[col].std()
        
    overall["worst_window_mape"] = results_df.groupby("window_idx")["mape"].mean().max()
    overall["worst_window_directional_accuracy"] = results_df.groupby("window_idx")["directional_accuracy"].mean().min()
    
    summary["overall_mean"] = overall
    
    ticker_means = {}
    for ticker, group in results_df.groupby("ticker"):
        ticker_means[ticker] = {}
        for col in numeric_cols:
            ticker_means[ticker][f"{col}_mean"] = group[col].mean()
            ticker_means[ticker][f"{col}_std"] = group[col].std()
            
    summary["ticker_mean"] = ticker_means
    summary_path = os.path.join(run_folder, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Copy model artifact
    model_dest = os.path.join(run_folder, os.path.basename(model_path))
    shutil.copyfile(model_path, model_dest)

    # Write metadata
    metadata = {
        "run_id": run_id,
        "model_type": model_type,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "feature_set_version": feature_set_version,
        "ticker_list": ticker_list,
        "results_path": results_path,
        "summary_path": summary_path,
        "model_path": model_dest,
    }
    meta_path = os.path.join(run_folder, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Registered run {run_id} for model {model_type}")

def load_all_runs() -> list:
    runs = []
    if not os.path.isdir(REGISTRY_ROOT):
        return runs
    for entry in os.scandir(REGISTRY_ROOT):
        if entry.is_dir():
            meta_path = os.path.join(entry.path, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    runs.append(json.load(f))
    return runs

def get_best_run(model_type: str = None, metric: str = "directional_accuracy_mean") -> dict:
    """Return metadata of the best run according to ``metric``.

    ``metric`` should be a key in the ``overall_mean`` section of the summary.
    Higher is better for directional accuracy; for error metrics callers can invert.
    """
    runs = load_all_runs()
    if model_type:
        runs = [r for r in runs if r.get("model_type") == model_type]
    if not runs:
        return None
    best = None
    best_val = None
    for r in runs:
        summary_path = r.get("summary_path")
        if not summary_path or not os.path.exists(summary_path):
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        overall = summary.get("overall_mean", {})
        val = overall.get(metric)
        if val is None:
            continue
        if best is None or val > best_val:
            best = r
            best_val = val
    return best
