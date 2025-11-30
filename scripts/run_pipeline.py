# scripts/run_pipeline.py
# Robust entry point for GitHub Actions retraining workflow.
# Ensures the correct repository root is added to sys.path by searching
# upward for a directory that contains "models" (or .git).
# Emits helpful debug prints to make CI issues visible.

import logging
import traceback
import sys
from pathlib import Path
import os

def find_repo_root(start_path: Path, markers=("models", ".git"), max_up=6):
    """
    Search upward from start_path for a directory that contains any of the marker names.
    Return the Path if found, otherwise None.
    """
    p = start_path.resolve()
    for i in range(max_up):
        # check current directory for markers
        for m in markers:
            if (p / m).exists():
                return p
        if p.parent == p:
            break
        p = p.parent
    return None

# Determine script location and search upward
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

repo_root = find_repo_root(SCRIPT_DIR)
if repo_root is None:
    # Try a slightly more permissive search (look for 'README.md' which is often in repo root)
    repo_root = find_repo_root(SCRIPT_DIR, markers=("README.md", "models", ".git"))
    
# If still not found, guess two levels up (best-effort fallback)
if repo_root is None:
    repo_root = SCRIPT_PATH.resolve().parents[2]

# Insert into sys.path if not present
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

# Debug prints (these will appear in GitHub Actions logs)
print("DEBUG: __file__ =", __file__)
print("DEBUG: SCRIPT_DIR =", SCRIPT_DIR)
print("DEBUG: Selected REPO_ROOT =", repo_root_str)
print("DEBUG: sys.path[0] =", sys.path[0])

def main():
    try:
        # Import after sys.path is fixed
        from models.price_predictor import BitcoinPricePredictor

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        logger = logging.getLogger(__name__)
        logger.info("Starting retraining pipeline...")

        pp = BitcoinPricePredictor()
        result = pp.run_full_pipeline()

        logger.info(f"Retraining finished. Result: {result}")
    except Exception as e:
        logging.error("Retraining pipeline failed:")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
