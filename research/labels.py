"""Causal and explicit training-label construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForwardReturnLabelSpec:
    """Definition of a close-to-close research target.

    Close prices are permitted only for model research. Execution validation must
    later use executable bid/ask fills and costs; this module never represents a
    close-to-close label as an executable trade return.
    """

    horizon_bars: int
    entry_lag_bars: int = 1
    price_column: str = "close"
    return_kind: Literal["log", "simple"] = "log"
    target_name: str = "forward_return"

    def __post_init__(self) -> None:
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be at least one bar.")
        if self.entry_lag_bars < 1:
            raise ValueError("entry_lag_bars must be at least one bar.")
        if not self.price_column:
            raise ValueError("price_column must be non-empty.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_forward_return_labels(
    frame: pd.DataFrame,
    spec: ForwardReturnLabelSpec,
) -> pd.DataFrame:
    """Build a target and its exact future observation timestamp.

    The final `horizon_bars` rows are deliberately NaN because their future price
    is unavailable. Consumers must drop them rather than impute or backfill them.
    """
    if spec.price_column not in frame:
        raise ValueError(f"Missing price column '{spec.price_column}'.")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Labels require a DatetimeIndex for availability auditing.")

    price = pd.to_numeric(frame[spec.price_column], errors="coerce").astype(float)
    if price.isna().any() or (price <= 0.0).any():
        raise ValueError("Label price must be finite and strictly positive.")

    entry_price = price.shift(-spec.entry_lag_bars)
    exit_price = price.shift(-(spec.entry_lag_bars + spec.horizon_bars))
    if spec.return_kind == "log":
        target = np.log(exit_price / entry_price)
    else:
        target = exit_price / entry_price - 1.0

    future_timestamp = pd.Series(frame.index, index=frame.index).shift(
        -(spec.entry_lag_bars + spec.horizon_bars)
    )
    result = pd.DataFrame(
        {
            spec.target_name: target,
            f"{spec.target_name}_available_at": future_timestamp,
        },
        index=frame.index,
    )
    return result
