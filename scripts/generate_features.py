"""Generate causal research features from timestamped FX market data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.pipeline import FeaturePipeline
from research.contracts import MarketDataContract, build_dataset_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate causal features without synthetic market or alternative data."
    )
    parser.add_argument("--input", type=Path, default=Path("data/EUR_USD_ticks.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/EUR_USD_features.csv"))
    parser.add_argument("--pair", default="EUR_USD")
    parser.add_argument("--provider", default="unspecified")
    parser.add_argument(
        "--include-microstructure",
        action="store_true",
        help="Require and use real bid/ask microstructure fields from the input.",
    )
    parser.add_argument(
        "--forward-fill",
        action="store_true",
        help="Apply an explicit causal forward-fill after feature computation.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    raw = pd.read_csv(args.input, parse_dates=["timestamp"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.set_index("timestamp").sort_index()
    contract = MarketDataContract(
        pair=args.pair,
        provider=args.provider,
        require_bid_ask_pair=args.include_microstructure,
    )
    raw = contract.validate(raw)
    manifest = build_dataset_manifest(raw, contract)

    pipeline = FeaturePipeline(
        include_microstructure=args.include_microstructure,
        forward_fill=args.forward_fill,
    )
    features = pipeline.compute_all(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index_label="timestamp")
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        __import__("json").dumps(manifest.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"Generated {features.shape[1]} causal features for {len(features)} rows. "
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
