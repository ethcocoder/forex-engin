import os
import argparse
import pandas as pd
import structlog
from datetime import datetime
from typing import List, Dict, Any

from configs.loader import load_config
from infrastructure.database.timescaledb.connection import TimescaleDBManager
from infrastructure.database.timescaledb.models import Tick, OHLCV

logger = structlog.get_logger()

def ingest_csv(filepath: str, pair: str, table_type: str, config_path: str, tf_override: str = None):
    """
    Ingests CSV data into TimescaleDB.
    table_type: 'ticks' or 'ohlcv'
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return

    logger.info(f"Loading data from {filepath} for {pair} into {table_type} table")
    df = pd.read_csv(filepath)

    # Standardize column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "time"})

    if "time" not in df.columns:
        logger.error("CSV must contain a 'time' or 'timestamp' column")
        return

    # Convert time to datetime objects
    df["time"] = pd.to_datetime(df["time"])
    df["pair"] = pair

    # Initialize DB Manager
    config = load_config(config_path)
    db_manager = TimescaleDBManager(config)

    data_list = df.to_dict(orient="records")

    model = Tick if table_type == "ticks" else OHLCV

    if table_type == "ohlcv" and "tf" not in df.columns:
        if tf_override:
            tf = tf_override
        else:
            # Default to 1m if not specified, though yfinance often gives 1h or 1d
            # We try to infer from data if possible or just use a default
            time_diff = df["time"].diff().median()
            if time_diff == pd.Timedelta(minutes=1):
                tf = "1m"
            elif time_diff == pd.Timedelta(hours=1):
                tf = "1h"
            elif time_diff == pd.Timedelta(days=1):
                tf = "1d"
            else:
                tf = "1m" # Fallback

            logger.info(f"Inferred timeframe: {tf}")

        for item in data_list:
            item["tf"] = tf

    # In case of Tick but only 'close' is present, mock bid/ask
    if table_type == "ticks" and "bid" not in df.columns and "close" in df.columns:
        logger.warning("Ticks table requested but only OHLCV 'close' found. Mocking bid/ask with 1 pip spread.")
        spread = 0.0001
        for item in data_list:
            item["bid"] = item["close"] - (spread / 2)
            item["ask"] = item["close"] + (spread / 2)
            # Remove OHLC columns not in Tick model
            for k in ["open", "high", "low", "close"]:
                item.pop(k, None)

    logger.info(f"Inserting {len(data_list)} records into {table_type}...")

    with db_manager.session_scope() as session:
        db_manager.bulk_insert(session, model, data_list)

    logger.info("Ingestion completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest CSV data into TimescaleDB")
    parser.add_argument("--file", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--pair", type=str, default="EUR_USD", help="Currency pair")
    parser.add_argument("--type", type=str, choices=["ticks", "ohlcv"], default="ohlcv", help="Table type")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--tf", type=str, default=None, help="Timeframe (e.g., 1m, 1h, 1d) if not in CSV")

    args = parser.parse_args()

    ingest_csv(args.file, args.pair, args.type, args.config, args.tf)
