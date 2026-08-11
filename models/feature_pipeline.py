from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()


@dataclass(frozen=True)
class WalkForwardFold:
    """One chronological evaluation fold with explicit anti-leakage boundaries."""

    fold: int
    train_start: int
    train_end_exclusive: int
    purge_start: int
    test_start: int
    test_end_exclusive: int
    embargo_bars: int


class LeakageSafeFeaturePipeline:
    """Past-only feature engineering and purged chronological validation.

    Features at row ``t`` are based only on observations at or before ``t``.
    Labels are built separately and carry their realised end timestamp. Splitters
    remove training observations whose label horizon can overlap a future test
    block, preventing future price information from crossing the fold boundary.
    """

    def __init__(self, window_sizes: Sequence[int] = (5, 15, 60)) -> None:
        windows = [int(w) for w in window_sizes]
        if not windows or any(w < 2 for w in windows):
            raise ValueError("window_sizes must contain integers of at least two bars")
        self.window_sizes = windows

    @property
    def max_lookback(self) -> int:
        return max(self.window_sizes)

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute strictly past-and-present quote features without imputation.

        Missing warm-up rows are dropped rather than backfilled. Backfilling a
        rolling feature would expose future data to an earlier decision time.
        """
        required = {"timestamp", "bid", "ask", "mid"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required quote columns: {sorted(missing)}")
        if df.empty:
            return df.copy()

        result = df.copy()
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise", format="ISO8601")
        result = result.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last")
        if (result["ask"] < result["bid"]).any():
            raise ValueError("Crossed quotes cannot be used for feature construction")
        if (result[["bid", "ask", "mid"]] <= 0).any().any():
            raise ValueError("Non-positive quotes cannot be used for feature construction")

        result["return_1"] = result["mid"].pct_change()
        result["rel_spread"] = (result["ask"] - result["bid"]) / result["mid"]
        for window in self.window_sizes:
            # The current return and preceding returns are all observable at t.
            result[f"vol_{window}"] = result["return_1"].rolling(
                window=window, min_periods=window
            ).std(ddof=0)
            result[f"momentum_{window}"] = result["mid"].pct_change(window)

        feature_columns = [
            "return_1",
            "rel_spread",
            *[f"vol_{window}" for window in self.window_sizes],
            *[f"momentum_{window}" for window in self.window_sizes],
        ]
        result = result.dropna(subset=feature_columns).reset_index(drop=True)
        logger.info(
            "Leakage-safe features computed",
            records=len(result),
            max_lookback=self.max_lookback,
            features=feature_columns,
        )
        return result

    def attach_executable_labels(
        self,
        df: pd.DataFrame,
        horizon_bars: int = 1,
        min_return: float = 0.0,
    ) -> pd.DataFrame:
        """Attach future *executable* directional labels without contaminating features.

        The long target buys at the current ask and exits at the future bid. The
        short target sells at the current bid and covers at the future ask. The
        method drops incomplete trailing horizons rather than assigning them a
        fabricated negative label.
        """
        if horizon_bars < 1:
            raise ValueError("horizon_bars must be positive")
        required = {"timestamp", "bid", "ask"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required label columns: {sorted(missing)}")
        if df.empty:
            return df.copy()

        result = df.copy().reset_index(drop=True)
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise", format="ISO8601")
        future_bid = result["bid"].shift(-horizon_bars)
        future_ask = result["ask"].shift(-horizon_bars)
        result["long_return_h"] = (future_bid - result["ask"]) / result["ask"]
        result["short_return_h"] = (result["bid"] - future_ask) / result["bid"]
        result["label_end_timestamp"] = result["timestamp"].shift(-horizon_bars)

        complete = result["label_end_timestamp"].notna()
        result = result.loc[complete].copy()
        long_is_best = result["long_return_h"] >= result["short_return_h"]
        long_tradeable = result["long_return_h"] > min_return
        short_tradeable = result["short_return_h"] > min_return
        result["label"] = np.select(
            [long_is_best & long_tradeable, (~long_is_best) & short_tradeable],
            [1, -1],
            default=0,
        ).astype(np.int8)
        result["label_horizon_bars"] = horizon_bars
        return result.reset_index(drop=True)

    def iter_purged_walk_forward_folds(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        test_size: int | None = None,
        purge_bars: int | None = None,
        embargo_bars: int = 0,
    ) -> Iterator[WalkForwardFold]:
        """Yield expanding-window folds with an explicit anti-overlap gap.

        ``purge_bars`` must cover both maximum feature lookback and label horizon.
        In an expanding-only walk-forward design there are no later training rows
        after a test block, so a post-test embargo is recorded but has no further
        rows to remove. It remains part of the fold contract for comparable
        training procedures that add post-test training data.
        """
        if n_splits < 1:
            raise ValueError("n_splits must be positive")
        if embargo_bars < 0:
            raise ValueError("embargo_bars cannot be negative")
        if df.empty:
            return
        if "label_end_timestamp" not in df.columns:
            raise ValueError("Purged splitting requires labels with label_end_timestamp")

        n_rows = len(df)
        if test_size is None:
            test_size = n_rows // (n_splits + 1)
        if test_size < 1:
            raise ValueError("Not enough rows for the requested number of folds")
        if purge_bars is None:
            label_horizon = int(df["label_horizon_bars"].iloc[0]) if "label_horizon_bars" in df else 1
            purge_bars = max(self.max_lookback, label_horizon)
        if purge_bars < 1:
            raise ValueError("purge_bars must be positive")

        first_train_end = n_rows - n_splits * test_size - purge_bars
        if first_train_end <= 0:
            raise ValueError("Not enough rows after purge for the requested folds")

        for fold in range(n_splits):
            train_end = first_train_end + fold * test_size
            test_start = train_end + purge_bars
            test_end = min(test_start + test_size, n_rows)
            if test_end <= test_start:
                break
            yield WalkForwardFold(
                fold=fold,
                train_start=0,
                train_end_exclusive=train_end,
                purge_start=train_end,
                test_start=test_start,
                test_end_exclusive=test_end,
                embargo_bars=embargo_bars,
            )

    def purged_walk_forward_split(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        purge_window: int | None = None,
        embargo_bars: int = 0,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Return materialized folds after verifying label horizons do not cross test starts."""
        splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        for fold in self.iter_purged_walk_forward_folds(
            df,
            n_splits=n_splits,
            purge_bars=purge_window,
            embargo_bars=embargo_bars,
        ):
            train_set = df.iloc[fold.train_start : fold.train_end_exclusive].copy()
            test_set = df.iloc[fold.test_start : fold.test_end_exclusive].copy()
            test_start_time = test_set["timestamp"].iloc[0]
            if not (train_set["label_end_timestamp"] < test_start_time).all():
                raise AssertionError("Purging failed: a training label horizon overlaps the test block")
            splits.append((train_set, test_set))

        logger.info(
            "Purged walk-forward splits generated",
            total_folds=len(splits),
            purge_bars=purge_window or self.max_lookback,
            embargo_bars=embargo_bars,
        )
        return splits
