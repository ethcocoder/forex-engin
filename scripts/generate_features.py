import os
import sys
import argparse
import pandas as pd
import structlog
from pathlib import Path

# Ensure the root directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.pipeline import FeaturePipeline
from configs.loader import load_config

logger = structlog.get_logger()

def main():
    parser = argparse.ArgumentParser(description="Generate features from raw historical forex data")
    parser.add_argument("--input", type=str, default="data/EUR_USD_ticks.csv", help="Path to raw CSV data")
    parser.add_argument("--output", type=str, default="data/EUR_USD_features.csv", help="Path to save generated features CSV")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logger.error(f"Input file {args.input} does not exist. Did you run download_data.py?")
        sys.exit(1)
        
    logger.info(f"Loading raw data from {args.input}...")
    df = pd.read_csv(args.input, index_col="timestamp", parse_dates=True)
    
    if df.empty:
        logger.error("Loaded dataframe is empty.")
        sys.exit(1)
        
    logger.info(f"Loaded {len(df)} rows. Instantiating feature pipeline...")
    
    # Load default configuration
    try:
        config = load_config()
    except Exception as e:
        logger.warning(f"Could not load full config, using empty dict: {e}")
        config = {}
        
    # Instantiate the master pipeline
    pipeline = FeaturePipeline(config=config)
    
    logger.info("Executing compute_all() — this may take a moment depending on the data size...")
    features_df = pipeline.compute_all(df)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    logger.info(f"Saving {features_df.shape[1]} features to {args.output}...")
    features_df.to_csv(args.output)
    
    logger.info("Feature generation completed successfully!")

if __name__ == "__main__":
    main()
