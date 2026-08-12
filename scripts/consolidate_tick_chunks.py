#!/usr/bin/env python3
"""Consolidate contiguous, manifest-verified free tick chunks for research only."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def load_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "real_historical_tick_data":
        raise ValueError(f"Unsupported manifest kind in {path.name}")
    if manifest.get("research_authorization") != "EXPLORATORY_RESEARCH_ONLY":
        raise ValueError(f"Manifest {path.name} is not explicitly authorized for exploratory research")
    for field in ("institutional_execution_validation", "broker_demo_authorization", "live_trading_authorization"):
        if manifest.get(field) != "DENIED":
            raise ValueError(f"Manifest {path.name} lacks required execution denial: {field}")
    dataset_path = path.parent.parent / manifest["dataset_path"]
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset named by {path.name} is unavailable: {dataset_path}")
    if sha256_file(dataset_path) != manifest.get("dataset_sha256"):
        raise ValueError(f"Dataset checksum mismatch for {dataset_path.name}")
    return dataset_path, manifest


def consolidate(manifest_paths: list[Path], output: Path) -> tuple[Path, Path]:
    if not manifest_paths:
        raise ValueError("At least one input manifest is required")
    loaded = [(path.resolve(), *load_manifest(path.resolve())) for path in manifest_paths]
    loaded.sort(key=lambda item: parse_utc(item[2]["requested_start_utc"]))

    instrument = loaded[0][2]["instrument"]
    source = loaded[0][2]["source"]
    previous_end = None
    frames: list[pd.DataFrame] = []
    chain: list[dict[str, str]] = []
    for source_manifest_path, dataset_path, manifest in loaded:
        if manifest["instrument"] != instrument or manifest["source"] != source:
            raise ValueError("All chunks must use the same instrument and source")
        start, end = parse_utc(manifest["requested_start_utc"]), parse_utc(manifest["requested_end_utc_exclusive"])
        if previous_end is not None and start != previous_end:
            raise ValueError("Input chunks are not contiguous; gap-free ranges are required for consolidation")
        previous_end = end
        frame = pd.read_csv(dataset_path)
        expected_columns = {"instrument", "timestamp", "bid", "ask", "bid_volume", "ask_volume", "spread", "mid"}
        if set(frame.columns) != expected_columns:
            raise ValueError(f"Unexpected tick schema in {dataset_path.name}")
        if frame["instrument"].nunique() != 1 or frame["instrument"].iloc[0] != instrument:
            raise ValueError(f"Instrument mismatch inside {dataset_path.name}")
        frames.append(frame)
        chain.append({
            "manifest": source_manifest_path.name,
            "dataset": dataset_path.name,
            "dataset_sha256": manifest["dataset_sha256"],
        })

    merged = pd.concat(frames, ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="raise", format="ISO8601")
    merged = merged.sort_values("timestamp", kind="stable")
    if merged["timestamp"].duplicated().any() or not merged["timestamp"].is_monotonic_increasing:
        raise ValueError("Consolidated ticks must have unique, strictly ordered timestamps")
    if (merged["ask"] < merged["bid"]).any() or (merged[["bid", "ask"]] <= 0).any().any():
        raise ValueError("Consolidated ticks contain invalid bid/ask quotes")

    first = parse_utc(loaded[0][2]["requested_start_utc"])
    last = parse_utc(loaded[-1][2]["requested_end_utc_exclusive"])
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    consolidated_manifest = {
        "schema_version": 1,
        "kind": "real_historical_tick_data",
        "source": source,
        "source_reference": loaded[0][2]["source_reference"],
        "source_class": "free_public_broker_historical_export",
        "research_authorization": "EXPLORATORY_RESEARCH_ONLY",
        "institutional_execution_validation": "DENIED",
        "broker_demo_authorization": "DENIED",
        "live_trading_authorization": "DENIED",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "instrument": instrument,
        "requested_start_utc": first.isoformat(),
        "requested_end_utc_exclusive": last.isoformat(),
        "dataset_path": str(output.relative_to(output.parents[1])),
        "dataset_sha256": sha256_file(output),
        "rows": int(len(merged)),
        "columns": list(merged.columns),
        "first_timestamp_utc": merged["timestamp"].iloc[0].isoformat(),
        "last_timestamp_utc": merged["timestamp"].iloc[-1].isoformat(),
        "manifest_chain": chain,
        "model_training_authorization": "DENIED until campaign coverage and data-quality gates pass.",
    }
    manifest_path.write_text(json.dumps(consolidated_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Consolidate contiguous manifest-verified free tick chunks")
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset, manifest = consolidate(args.manifests, args.output)
    print(json.dumps({"dataset": str(dataset), "manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
