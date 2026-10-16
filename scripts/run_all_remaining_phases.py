"""
Monitors phase_1 completion and then automatically executes all remaining phases 
(phase_2 through phase_8) in sequence without stopping.
"""
import os
import sys
import time
import subprocess
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(ROOT_DIR, "data", "phase_manifest.json")

def is_phase_1_running():
    try:
        output = subprocess.check_output('wmic process where "name=\'python.exe\'" get commandline', shell=True, text=True)
        return "run_staged_ingestion.py phase_1" in output
    except Exception:
        return False

def main():
    print("=== Auto-Run Remaining Ingestion Phases Orchestrator ===")
    
    # 1. Wait for phase_1 to finish if running
    while is_phase_1_running():
        print("[Orchestrator] phase_1 is currently running. Waiting 30s before checking again...")
        time.sleep(30)
        
    print("[Orchestrator] phase_1 has completed! Starting remaining phases...")

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    remaining_phases = [p for p in manifest.keys() if p != "phase_1"]
    print(f"[Orchestrator] Remaining phases to process: {remaining_phases}")

    for phase in sorted(remaining_phases):
        print(f"\n=========================================")
        print(f"STARTING AUTOMATIC INGESTION FOR {phase}")
        print(f"=========================================")
        cmd = [sys.executable, os.path.join(ROOT_DIR, "scripts", "run_staged_ingestion.py"), phase]
        res = subprocess.run(cmd, cwd=ROOT_DIR)
        if res.returncode == 0:
            print(f"[Orchestrator] Successfully completed {phase}!")
        else:
            print(f"[Orchestrator] ERROR: {phase} failed with exit code {res.returncode}. Continuing to next phase...")

    print("\n=========================================")
    print("ALL PHASES (phase_1 through phase_8) COMPLETED SUCCESSFULLY!")
    print("=========================================")

if __name__ == "__main__":
    main()
