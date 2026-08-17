import json
import os
import logging
from typing import List, Optional

from .registry import load_all_runs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

# Promotion thresholds – adjust as needed
DIRECTIONAL_ACC_MARGIN = 2.0  # percentage points improvement over naive baseline
MAX_MAPE = 15.0               # maximum acceptable MAPE (percentage)
MAX_WORST_WINDOW_MAPE = 30.0  # worst window MAPE must be below this
MAX_TICKER_VARIANCE = 5.0     # max allowed spread between best and worst ticker directional accuracy

def _load_summary(summary_path: str) -> dict:
    with open(summary_path, "r") as f:
        return json.load(f)

def evaluate_promotion_candidate(run_meta: dict) -> bool:
    """Determine if a registered run satisfies the promotion gate.

    ``run_meta`` is a metadata dict as stored by ``registry.register_run``.
    The function loads the corresponding ``summary.json`` and checks:
      * Directional accuracy improves over the naive baseline by at least
        ``DIRECTIONAL_ACC_MARGIN``.
      * Overall MAPE is <= ``MAX_MAPE``.
      * The worst‑case window MAPE (taken from ``overall_mean``) is <=
        ``MAX_WORST_WINDOW_MAPE``.
      * Per‑ticker directional accuracy variance (max - min) is <=
        ``MAX_TICKER_VARIANCE``.
    Returns ``True`` if all criteria are met; otherwise ``False``.
    """
    summary_path = run_meta.get("summary_path")
    if not summary_path or not os.path.exists(summary_path):
        logger.warning(f"Missing summary for run {run_meta.get('run_id')}")
        return False
    summary = _load_summary(summary_path)
    overall = summary.get("overall_mean", {})
    ticker_means = summary.get("ticker_mean", {})

    required_keys = [
        "directional_accuracy",
        "directional_accuracy_mean",
        "directional_accuracy_std",
        "mape",
        "mape_mean",
        "naive_directional_accuracy",
        "naive_mape",
    ]
    if not all(k in overall for k in required_keys):
        logger.warning(f"Summary for run {run_meta.get('run_id')} missing required metrics.")
        return False

    improvement = overall["directional_accuracy_mean"] - overall["naive_directional_accuracy"]
    if improvement < DIRECTIONAL_ACC_MARGIN:
        logger.info(f"Run {run_meta['run_id']} fails DA margin: {improvement:.2f}% < {DIRECTIONAL_ACC_MARGIN}%")
        return False

    if overall["mape_mean"] > MAX_MAPE:
        logger.info(f"Run {run_meta['run_id']} exceeds MAPE limit: {overall['mape_mean']:.2f}% > {MAX_MAPE}%")
        return False

    if overall["mape"] > MAX_WORST_WINDOW_MAPE:
        logger.info(f"Run {run_meta['run_id']} worst‑window MAPE too high: {overall['mape']:.2f}% > {MAX_WORST_WINDOW_MAPE}%")
        return False

    if ticker_means:
        ticker_da = [v.get("directional_accuracy_mean", 0) for v in ticker_means.values()]
        if ticker_da:
            variance = max(ticker_da) - min(ticker_da)
            if variance > MAX_TICKER_VARIANCE:
                logger.info(f"Run {run_meta['run_id']} ticker DA variance {variance:.2f}% exceeds {MAX_TICKER_VARIANCE}%")
                return False

    logger.info(f"Run {run_meta['run_id']} passes promotion criteria.")
    return True

def select_best_model_overall(candidate_run_ids: Optional[List[str]] = None) -> Optional[dict]:
    """Select the best run among candidates that satisfy the promotion gate.

    If ``candidate_run_ids`` is ``None``, all registered runs are considered.
    The function:
      * Filters runs through ``evaluate_promotion_candidate``.
      * Ranks the surviving runs by ``directional_accuracy_mean`` (higher is better).
    Returns the metadata dict of the winning run, or ``None`` if none qualify.
    """
    runs = load_all_runs()
    if candidate_run_ids:
        runs = [r for r in runs if r.get("run_id") in set(candidate_run_ids)]
    qualified = []
    for r in runs:
        try:
            if evaluate_promotion_candidate(r):
                qualified.append(r)
        except Exception as e:
            logger.error(f"Error evaluating run {r.get('run_id')}: {e}")
    if not qualified:
        logger.warning("No runs satisfy promotion criteria.")
        return None
    best = max(qualified, key=lambda x: _load_summary(x["summary_path"]).get("overall_mean", {}).get("directional_accuracy_mean", -1))
    logger.info(f"Best promotional run selected: {best.get('run_id')}")
    return best
