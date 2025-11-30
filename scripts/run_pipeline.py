# scripts/run_pipeline.py
# Entry point for GitHub Actions retraining workflow.
# Ensures repo root is on sys.path so imports like `from models.price_predictor import ...` work.

import logging
import traceback
import sys
from pathlib import Path

# ---- Ensure the repository root is on sys.path ----
# When running `python scripts/run_pipeline.py`, sys.path[0] is the scripts/ folder.
# We want to add the repository root (one level up) so `import models...` works.
REPO_ROOT = Path(__file__).resolve().parents[1]  # repo root: ../
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def main():
    try:
        # Import inside main so that the action fails cleanly if imports are broken
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
