"""Strict, auditable input contracts for causal market-data research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Iterable, Optional

import numpy as np
import pandas as pd


class DataContractError(ValueError):
    """Raised when market data is ambiguous, malformed, or unsuitable for research."""


@dataclass(frozen=True)
class DatasetManifest:
    """Immutable description of the exact dataset consumed by a research run."""

    pair: str
    provider: str
    rows: int
    first_timestamp: str
    last_timestamp: str
    columns: tuple[str, ...]
    dtypes: tuple[tuple[str, str], ...]
    schema_sha256: str
    content_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MarketDataContract:
    """Validation policy for ordered, timestamped FX bars or ticks.

    The contract intentionally rejects naive timestamps and duplicate/out-of-order
    observations. A research result is not reproducible if its data availability
    time cannot be established unambiguously.
    """

    pair: str
    provider: str = "unknown"
    required_columns: tuple[str, ...] = (
        "open",
        "high",
        "low",
        "close",
        "volume",
    )
    require_utc_index: bool = True
    require_bid_ask_pair: bool = False

    def validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a validated copy indexed by increasing timezone-aware UTC time."""
        if not isinstance(frame, pd.DataFrame):
            raise DataContractError("Market data must be a pandas DataFrame.")
        if frame.empty:
            raise DataContractError("Market data must contain at least one observation.")

        validated = frame.copy()
        if "timestamp" in validated.columns:
            validated["timestamp"] = pd.to_datetime(validated["timestamp"], utc=True)
            validated = validated.set_index("timestamp")

        if not isinstance(validated.index, pd.DatetimeIndex):
            raise DataContractError(
                "Market data must have a DatetimeIndex or a timestamp column."
            )
        if validated.index.tz is None:
            raise DataContractError(
                "Naive timestamps are forbidden. Parse and localise the source before research."
            )
        if self.require_utc_index and str(validated.index.tz) != "UTC":
            validated.index = validated.index.tz_convert("UTC")

        if not validated.index.is_monotonic_increasing:
            raise DataContractError("Timestamps must be strictly chronological.")
        if validated.index.has_duplicates:
            duplicate_count = int(validated.index.duplicated().sum())
            raise DataContractError(
                f"Duplicate timestamps are forbidden; found {duplicate_count}."
            )

        missing = [column for column in self.required_columns if column not in validated]
        if missing:
            raise DataContractError(f"Missing required market-data columns: {missing}.")

        numeric_columns = list(self.required_columns)
        for column in ("bid", "ask"):
            if column in validated:
                numeric_columns.append(column)
        for column in numeric_columns:
            values = pd.to_numeric(validated[column], errors="coerce")
            if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
                raise DataContractError(
                    f"Column '{column}' contains non-finite or non-numeric observations."
                )
            validated[column] = values.astype(float)

        if (validated[["open", "high", "low", "close"]] <= 0.0).any().any():
            raise DataContractError("OHLC prices must be strictly positive.")
        if (validated["volume"] < 0.0).any():
            raise DataContractError("Volume must be non-negative.")
        if (validated["high"] < validated[["open", "close"]].max(axis=1)).any():
            raise DataContractError("Each high must be at least max(open, close).")
        if (validated["low"] > validated[["open", "close"]].min(axis=1)).any():
            raise DataContractError("Each low must be at most min(open, close).")
        if (validated["high"] < validated["low"]).any():
            raise DataContractError("Each high must be at least its low.")

        has_bid = "bid" in validated
        has_ask = "ask" in validated
        if self.require_bid_ask_pair and not (has_bid and has_ask):
            raise DataContractError("This workflow requires executable bid and ask quotes.")
        if has_bid != has_ask:
            raise DataContractError("Bid and ask must either both be present or both be absent.")
        if has_bid and (validated["ask"] < validated["bid"]).any():
            raise DataContractError("Ask must be greater than or equal to bid.")

        return validated


def _hash_strings(values: Iterable[str]) -> str:
    digest = sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def build_dataset_manifest(
    frame: pd.DataFrame,
    contract: MarketDataContract,
) -> DatasetManifest:
    """Create a content and schema manifest after the data contract has passed."""
    validated = contract.validate(frame)
    schema_values = [
        f"{column}:{validated[column].dtype}" for column in validated.columns
    ]
    row_hashes = pd.util.hash_pandas_object(validated, index=True).to_numpy()
    content_sha256 = sha256(row_hashes.tobytes()).hexdigest()

    return DatasetManifest(
        pair=contract.pair,
        provider=contract.provider,
        rows=len(validated),
        first_timestamp=validated.index[0].isoformat(),
        last_timestamp=validated.index[-1].isoformat(),
        columns=tuple(str(column) for column in validated.columns),
        dtypes=tuple((str(column), str(validated[column].dtype)) for column in validated.columns),
        schema_sha256=_hash_strings(schema_values),
        content_sha256=content_sha256,
    )


@dataclass(frozen=True)
class MarketDataEligibilityPolicy:
    """Minimum data evidence required for an execution-like simulation.

    Contract validation establishes that a dataset is well formed. Eligibility is
    stricter: it records whether the source has the fields needed to represent
    executable trading rather than silently treating reference prices as fills.
    """

    minimum_rows: int = 256
    require_executable_bid_ask: bool = True
    require_positive_volume: bool = True

    def __post_init__(self) -> None:
        if self.minimum_rows < 1:
            raise ValueError("minimum_rows must be positive.")


@dataclass(frozen=True)
class MarketDataEligibilityReport:
    """Auditable determination of whether a source can support a given simulation."""

    eligible: bool
    reasons: tuple[str, ...]
    rows: int
    latest_timestamp: str
    has_executable_bid_ask: bool
    positive_volume_rows: int
    median_bar_interval_seconds: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def assess_market_data_eligibility(
    frame: pd.DataFrame,
    contract: MarketDataContract,
    policy: MarketDataEligibilityPolicy = MarketDataEligibilityPolicy(),
) -> MarketDataEligibilityReport:
    """Assess source adequacy without loosening the underlying data contract."""
    validated = contract.validate(frame)
    reasons: list[str] = []
    if len(validated) < policy.minimum_rows:
        reasons.append("insufficient_history")

    has_executable_bid_ask = {"bid", "ask"}.issubset(validated.columns)
    if policy.require_executable_bid_ask and not has_executable_bid_ask:
        reasons.append("missing_executable_bid_ask")

    positive_volume_rows = int((validated["volume"] > 0.0).sum())
    if policy.require_positive_volume and positive_volume_rows == 0:
        reasons.append("missing_observed_volume")

    if len(validated) > 1:
        deltas = validated.index.to_series().diff().dropna().dt.total_seconds()
        median_bar_interval_seconds: float | None = float(deltas.median())
    else:
        median_bar_interval_seconds = None

    return MarketDataEligibilityReport(
        eligible=not reasons,
        reasons=tuple(reasons),
        rows=len(validated),
        latest_timestamp=validated.index[-1].isoformat(),
        has_executable_bid_ask=has_executable_bid_ask,
        positive_volume_rows=positive_volume_rows,
        median_bar_interval_seconds=median_bar_interval_seconds,
    )
