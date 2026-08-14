"""Causal, reproducible research primitives for the forex trading engine."""

from research.contracts import (
    DataContractError,
    DatasetManifest,
    MarketDataContract,
    MarketDataEligibilityPolicy,
    MarketDataEligibilityReport,
    assess_market_data_eligibility,
    build_dataset_manifest,
)
from research.labels import ForwardReturnLabelSpec, build_forward_return_labels
from research.splits import ExpandingPurgedWalkForwardSplit

__all__ = [
    "DataContractError",
    "DatasetManifest",
    "MarketDataContract",
    "MarketDataEligibilityPolicy",
    "MarketDataEligibilityReport",
    "assess_market_data_eligibility",
    "build_dataset_manifest",
    "ForwardReturnLabelSpec",
    "build_forward_return_labels",
    "ExpandingPurgedWalkForwardSplit",
]
