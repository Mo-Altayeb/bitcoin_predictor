# scripts/run_pipeline.py
# Entry point for GitHub Actions retraining workflow.
# It runs your full pipeline in a non-interactive way.
# Adjust imports if your module paths differ.

import logging
import traceback

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
