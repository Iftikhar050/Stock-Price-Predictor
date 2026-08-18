import os
import sys
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.psx_predictor.models.registry import load_all_runs, get_best_run

runs = load_all_runs()
if not runs:
    print("No runs found.")
    sys.exit()

# Sort runs by trained_at
runs.sort(key=lambda r: r.get('trained_at', ''))

old_run = runs[0]
new_run = runs[-1]

def print_run_metrics(run, label):
    with open(run['summary_path'], 'r') as f:
        summary = json.load(f)
        acc = summary.get('overall_mean', {}).get('directional_accuracy_mean', 'N/A')
        print(f"{label} Run ID: {run['run_id']}")
        print(f"{label} Trained At: {run['trained_at']}")
        print(f"{label} Directional Accuracy: {acc:.4f}" if isinstance(acc, float) else f"{label} Directional Accuracy: {acc}")

print_run_metrics(old_run, "Old (Stale)")
print("----------------------------")
print_run_metrics(new_run, "New (Post-fix)")
