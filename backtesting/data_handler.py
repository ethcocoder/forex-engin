import os
from abc import ABC, abstractmethod
from typing import Dict, Generator, Optional
import pandas as pd
import structlog

logger = structlog.get_logger()


class BaseDataHandler(ABC):
    """
    Abstract Base Class for historical data ingestion during backtesting.
    """

    @abstractmethod
    def load_data(self) -> None:
        """Loads data from data source."""
        pass

    @abstractmethod
    def get_latest_bar(self, pair: str) -> Optional[Dict]:
        """Returns the most recent bar data as a dictionary."""
        pass

    @abstractmethod
    def stream_bars(self) -> Generator[Dict, None, None]:
        """Generates a stream of chronological bars across all active instruments."""
        pass


class CSVDataHandler(BaseDataHandler):
    """
    Concrete DataHandler that reads CSV files containing historical OHLCV data.
    Assumes standard format: timestamp (index or col), open, high, low, close, volume.
    """

    def __init__(self, csv_dir: str, pairs: list[str]) -> None:
        self.csv_dir = csv_dir
        self.pairs = pairs
        self.data: Dict[str, pd.DataFrame] = {}
        self.latest_bars: Dict[str, Dict] = {}
        
        logger.info("CSVDataHandler initialized", csv_dir=csv_dir, pairs=pairs)

    def load_data(self) -> None:
        """Reads CSV files for each currency pair and sorts them chronologically."""
        for pair in self.pairs:
            file_path = os.path.join(self.csv_dir, f"{pair}.csv")
            if not os.path.exists(file_path):
                # Check for alternative naming e.g., EUR_USD.csv
                alt_path = os.path.join(self.csv_dir, f"{pair[:3]}_{pair[3:]}.csv")
                if os.path.exists(alt_path):
                    file_path = alt_path
                else:
                    raise FileNotFoundError(f"Historical data file not found for pair: {pair} at {file_path}")

            logger.info("Loading historical data", file=file_path)
            df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")
            df.sort_index(inplace=True)
            self.data[pair] = df

    def get_latest_bar(self, pair: str) -> Optional[Dict]:
        return self.latest_bars.get(pair)

    def stream_bars(self) -> Generator[Dict, None, None]:
        """
        Interleaves data from multiple pairs chronologically.
        Yields a dict mapping: event_type -> 'BAR', data -> bar contents.
        """
        # Create generators for each pair
        iterators = {pair: self.data[pair].iterrows() for pair in self.pairs}
        next_rows = {}

        # Initialize the first row for each pair
        for pair, it in list(iterators.items()):
            try:
                ts, row = next(it)
                next_rows[pair] = (ts, row)
            except StopIteration:
                pass

        while next_rows:
            # Find the pair with the earliest timestamp
            earliest_pair = min(next_rows.keys(), key=lambda p: next_rows[p][0])
            ts, row = next_rows[earliest_pair]
            
            # Construct bar packet
            bar = {
                "pair": earliest_pair,
                "timestamp": ts,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"])
            }
            
            # Update latest state cache
            self.latest_bars[earliest_pair] = bar
            
            yield bar

            # Load the next row for the pair that just fired
            try:
                new_ts, new_row = next(iterators[earliest_pair])
                next_rows[earliest_pair] = (new_ts, new_row)
            except StopIteration:
                del next_rows[earliest_pair]
