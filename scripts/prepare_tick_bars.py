#!/usr/bin/env python3
"""Build leakage-safe bars from manifest-verified historical tick data.

The command accepts one CSV produced by ``download_dukascopy_ticks.py`` and its
JSON manifest. It refuses inputs whose recorded hash, source type, schema, or
instrument provenance does not match. No synthetic or unprovenanced data can
pass this gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

REQUIRED_TICK_COLUMNS = {
    "instrument",
    "timestamp",
    "bid",
    "ask",
    "bid_volume",
    "ask_volume",
    "spread",
    "mid",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(dataset_path: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported or missing data manifest schema version")
    if manifest.get("kind") != "real_historical_tick_data":
        raise ValueError("Training input must be verified real_historical_tick_data")
    if manifest.get("dataset_sha256") != sha256_file(dataset_path):
        raise ValueError("Dataset checksum does not match the provenance manifest")
    if Path(manifest.get("dataset_path", "")).name != dataset_path.name:
        raise ValueError("Manifest dataset_path does not identify the supplied dataset")
    if not manifest.get("source") or not manifest.get("source_reference"):
        raise ValueError("Manifest lacks source provenance")
    if manifest.get("research_authorization") not in {"EXPLORATORY_RESEARCH_ONLY", "LICENSED_RESEARCH"}:
        raise ValueError("Manifest lacks an approved, explicit research authorization")
    if manifest.get("live_trading_authorization") != "DENIED":
        raise ValueError("Source manifest must explicitly deny live-trading authorization")
    if int(manifest.get("rows", 0)) <= 0:
        raise ValueError("Manifest reports no usable source rows")
    return manifest


def build_bars(ticks: pd.DataFrame, frequency: str) -> pd.DataFrame:
    ticks = ticks.copy()
    missing = REQUIRED_TICK_COLUMNS - set(ticks.columns)
    if missing:
        raise ValueError(f"Tick dataset missing required columns: {sorted(missing)}")

    ticks["timestamp"] = pd.to_datetime(
        ticks["timestamp"], utc=True, errors="raise", format="ISO8601"
    )
    ticks = ticks.sort_values("timestamp", kind="stable")
    if not ticks["timestamp"].is_monotonic_increasing:
        raise ValueError("Tick timestamps are not monotonic after sorting")
    if ticks["instrument"].nunique() != 1:
        raise ValueError("One instrument per bar-preparation invocation is required")
    if (ticks["ask"] < ticks["bid"]).any() or (ticks[["bid", "ask"]] <= 0).any().any():
        raise ValueError("Tick dataset contains invalid executable quotes")

    ticks = ticks.set_index("timestamp")
    result = pd.DataFrame(
        {
            "open": ticks["mid"].resample(frequency).first(),
            "high": ticks["mid"].resample(frequency).max(),
            "low": ticks["mid"].resample(frequency).min(),
            "close": ticks["mid"].resample(frequency).last(),
            "bid": ticks["bid"].resample(frequency).last(),
            "ask": ticks["ask"].resample(frequency).last(),
            "spread": ticks["spread"].resample(frequency).last(),
            "tick_count": ticks["mid"].resample(frequency).count(),
            "bid_volume": ticks["bid_volume"].resample(frequency).sum(),
            "ask_volume": ticks["ask_volume"].resample(frequency).sum(),
        }
    ).dropna()
    result.insert(0, "instrument", ticks["instrument"].iloc[0])
    result.index.name = "timestamp"
    return result.reset_index()


def main() -> int:
    parser = argparse.ArgumentParser(description="Construct model bars from manifest-verified real tick data")
    parser.add_argument("--ticks", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--frequency", default="1min", help="Pandas resampling interval, e.g. 1s or 1min")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = verify_manifest(args.ticks, args.manifest)
    ticks = pd.read_csv(args.ticks)
    bars = build_bars(ticks, args.frequency)
    if bars.empty:
        raise RuntimeError("No complete bars were built from the verified tick dataset")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(args.output, index=False)
    output_manifest = {
        "schema_version": 1,
        "kind": "derived_real_tick_bars",
        "source_tick_dataset": args.ticks.name,
        "source_tick_sha256": manifest["dataset_sha256"],
        "source_manifest": args.manifest.name,
        "source_class": manifest["source_class"],
        "research_authorization": manifest["research_authorization"],
        "institutional_execution_validation": manifest.get("institutional_execution_validation", "DENIED"),
        "broker_demo_authorization": manifest.get("broker_demo_authorization", "DENIED"),
        "live_trading_authorization": manifest["live_trading_authorization"],
        "instrument": manifest["instrument"],
        "frequency": args.frequency,
        "rows": len(bars),
        "columns": list(bars.columns),
        "output_sha256": sha256_file(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"bars": str(args.output), "rows": len(bars)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
