"""Leakage-resistant chronological cross-validation splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class ExpandingPurgedWalkForwardSplit:
    """Expanding-window validation with label-horizon purge and embargo.

    Every validation block lies strictly after its training observations. The
    horizon purge excludes training labels whose future observation reaches into
    the validation period. The embargo skips observations immediately after each
    validation block before the next fold can begin.
    """

    n_splits: int = 3
    validation_size: int = 0
    label_horizon: int = 1
    embargo_bars: int = 0
    min_train_size: int = 256

    def __post_init__(self) -> None:
        if self.n_splits < 1:
            raise ValueError("n_splits must be positive.")
        if self.label_horizon < 1:
            raise ValueError("label_horizon must be positive.")
        if self.embargo_bars < 0:
            raise ValueError("embargo_bars cannot be negative.")
        if self.min_train_size < 1:
            raise ValueError("min_train_size must be positive.")

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield `(train_indices, validation_indices)` in chronological order."""
        if n_samples <= self.min_train_size + self.label_horizon:
            raise ValueError("Not enough samples for the configured minimum training window.")

        validation_size = self.validation_size
        if validation_size <= 0:
            remaining = n_samples - self.min_train_size - self.label_horizon
            validation_size = remaining // self.n_splits
        if validation_size <= 0:
            raise ValueError("validation_size resolves to zero.")

        for fold in range(self.n_splits):
            validation_start = self.min_train_size + self.label_horizon + fold * (
                validation_size + self.embargo_bars
            )
            validation_end = min(validation_start + validation_size, n_samples)
            if validation_start >= n_samples or validation_end <= validation_start:
                break

            # A training sample at index i observes a label at i + horizon. It
            # must therefore end before the validation start.
            train_end = validation_start - self.label_horizon
            if train_end < self.min_train_size:
                continue

            train_indices = np.arange(0, train_end, dtype=np.int64)
            validation_indices = np.arange(
                validation_start, validation_end, dtype=np.int64
            )
            yield train_indices, validation_indices

    def get_n_splits(self, n_samples: int) -> int:
        return sum(1 for _ in self.split(n_samples))
