from typing import List, Tuple, Any
import pandas as pd
import structlog

logger = structlog.get_logger()


class WalkForwardValidator:
    """
    Slices historical dataset into training and test windows.
    Helps prevent overfitting by ensuring models are evaluated out-of-sample.
    """

    def __init__(self, train_days: int = 60, test_days: int = 20) -> None:
        self.train_days = train_days
        self.test_days = test_days
        logger.info("WalkForwardValidator initialized", train=train_days, test=test_days)

    def generate_splits(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Takes a continuous DataFrame indexed by timestamp.
        Yields (train_subset, test_subset) splits.
        """
        if len(df) == 0:
            return []

        # Find min and max timestamps
        min_ts = df.index.min()
        max_ts = df.index.max()

        splits = []
        current_train_start = min_ts

        # Loop and slide windows forward
        while True:
            current_train_end = current_train_start + pd.Timedelta(days=self.train_days)
            current_test_end = current_train_end + pd.Timedelta(days=self.test_days)

            if current_test_end > max_ts:
                # If we don't have enough data for a final full test window, exit loop
                break

            train_slice = df.loc[current_train_start:current_train_end]
            test_slice = df.loc[current_train_end:current_test_end]

            splits.append((train_slice, test_slice))
            
            # Slide training start forward by the test days (non-overlapping testing)
            current_train_start += pd.Timedelta(days=self.test_days)

        logger.info(f"Generated {len(splits)} walk-forward splits")
        return splits
