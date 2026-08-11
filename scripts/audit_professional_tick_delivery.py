#!/usr/bin/env python3
"""Audit a professionally licensed FX tick-data delivery before any research use.

The tool is deliberately provider-agnostic: a vendor-specific field mapping and
an entitlement attestation must accompany every file. It verifies receipt
integrity, canonical field mapping, timestamp semantics, quote validity, and
basic chronological ordering. Its output is evidence only and never authorises
model deployment, broker-demo orders, or live execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

CANONICAL_REQUIRED = ("timestamp", "instrument", "bid", "ask")
ALLOWED_USES = {"research", "model_training", "model_validation"}


@dataclass
class FileAudit:
    path: str
    sha256: str
    rows: int
    invalid_timestamp_rows: int
    invalid_quote_rows: int
    non_monotonic_timestamp_rows: int
    instruments: list[str]
    first_timestamp_utc: str | None
    last_timestamp_utc: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read JSON document {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"JSON document {path} must contain an object")
    return parsed


def validate_contract(contract: dict[str, Any], entitlement: dict[str, Any]) -> None:
    columns = contract.get("columns")
    if not isinstance(columns, dict) or not all(isinstance(columns.get(key), str) for key in CANONICAL_REQUIRED):
        raise ValueError(f"Contract columns must map all canonical fields: {', '.join(CANONICAL_REQUIRED)}")
    if contract.get("timestamp_timezone") != "UTC":
        raise ValueError("Professional delivery contract must declare UTC timestamps")
    if not isinstance(contract.get("provider"), str) or not contract["provider"].strip():
        raise ValueError("Professional delivery contract must identify the provider")
    permitted = set(entitlement.get("permitted_uses", []))
    if not ALLOWED_USES.issubset(permitted):
        raise ValueError("Entitlement must explicitly permit research, model_training, and model_validation")
    if not isinstance(entitlement.get("licence_reference"), str) or not entitlement["licence_reference"].strip():
        raise ValueError("Entitlement must include a licence_reference")


def read_chunks(path: Path, columns: list[str], chunk_rows: int) -> Iterator[pd.DataFrame]:
    if path.suffix.lower() == ".csv":
        yield from pd.read_csv(path, usecols=columns, chunksize=chunk_rows)
        return
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            frame = pd.read_parquet(path, columns=columns)
        except ImportError as exc:
            raise RuntimeError("Parquet support is unavailable; install the approved parquet engine before auditing this delivery") from exc
        yield frame
        return
    raise ValueError(f"Unsupported delivery format: {path.suffix}. Use CSV or Parquet.")


def audit_file(path: Path, contract: dict[str, Any], chunk_rows: int) -> FileAudit:
    mapping = contract["columns"]
    source_columns = list(mapping.values())
    canonical = {source: target for target, source in mapping.items()}
    rows = invalid_timestamp_rows = invalid_quote_rows = non_monotonic_timestamp_rows = 0
    instruments: set[str] = set()
    first_timestamp = last_timestamp = previous_timestamp = None

    for raw in read_chunks(path, source_columns, chunk_rows):
        frame = raw.rename(columns=canonical)
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        bid = pd.to_numeric(frame["bid"], errors="coerce")
        ask = pd.to_numeric(frame["ask"], errors="coerce")
        instruments.update(str(item) for item in frame["instrument"].dropna().unique())

        invalid_timestamp = timestamps.isna()
        invalid_quote = bid.isna() | ask.isna() | (bid <= 0) | (ask <= 0) | (bid > ask)
        valid_times = timestamps.loc[~invalid_timestamp]
        if not valid_times.empty:
            if previous_timestamp is not None and valid_times.iloc[0] < previous_timestamp:
                non_monotonic_timestamp_rows += 1
            non_monotonic_timestamp_rows += int((valid_times.diff().dt.total_seconds() < 0).sum())
            previous_timestamp = valid_times.iloc[-1]
            first_timestamp = first_timestamp or valid_times.iloc[0]
            last_timestamp = valid_times.iloc[-1]

        rows += len(frame)
        invalid_timestamp_rows += int(invalid_timestamp.sum())
        invalid_quote_rows += int(invalid_quote.sum())

    if rows == 0:
        raise ValueError(f"No rows found in delivery file {path}")
    return FileAudit(
        path=str(path),
        sha256=sha256_file(path),
        rows=rows,
        invalid_timestamp_rows=invalid_timestamp_rows,
        invalid_quote_rows=invalid_quote_rows,
        non_monotonic_timestamp_rows=non_monotonic_timestamp_rows,
        instruments=sorted(instruments),
        first_timestamp_utc=first_timestamp.isoformat() if first_timestamp is not None else None,
        last_timestamp_utc=last_timestamp.isoformat() if last_timestamp is not None else None,
    )


def delivery_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        paths = sorted(
            path for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".csv", ".parquet", ".pq"}
        )
        if paths:
            return paths
        raise ValueError(f"No CSV or Parquet delivery files found under {input_path}")
    raise ValueError(f"Input delivery does not exist: {input_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a licensed professional FX tick-data delivery")
    parser.add_argument("--input", type=Path, required=True, help="CSV/Parquet file or a directory containing delivery files")
    parser.add_argument("--contract", type=Path, required=True, help="Provider field-mapping JSON")
    parser.add_argument("--entitlement", type=Path, required=True, help="Licence/usage attestation JSON")
    parser.add_argument("--report", type=Path, required=True, help="Output audit report JSON")
    parser.add_argument("--expected-instruments", default="", help="Optional comma-separated canonical instruments required for coverage")
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    args = parser.parse_args()

    if args.chunk_rows < 1:
        raise SystemExit("--chunk-rows must be positive")
    try:
        paths = delivery_paths(args.input)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    contract, entitlement = load_json(args.contract), load_json(args.entitlement)
    validate_contract(contract, entitlement)
    file_audits = [audit_file(path, contract, args.chunk_rows) for path in paths]

    failures: list[str] = []
    for file_audit in file_audits:
        if file_audit.invalid_timestamp_rows:
            failures.append(f"invalid_timestamps:{Path(file_audit.path).name}")
        if file_audit.invalid_quote_rows:
            failures.append(f"invalid_quotes:{Path(file_audit.path).name}")
        if file_audit.non_monotonic_timestamp_rows:
            failures.append(f"non_monotonic_timestamps:{Path(file_audit.path).name}")
        if not file_audit.instruments:
            failures.append(f"missing_instruments:{Path(file_audit.path).name}")

    observed_instruments = sorted({instrument for file_audit in file_audits for instrument in file_audit.instruments})
    expected_instruments = sorted({item.strip() for item in args.expected_instruments.split(",") if item.strip()})
    missing_instruments = sorted(set(expected_instruments) - set(observed_instruments))
    if missing_instruments:
        failures.append("incomplete_instrument_coverage")
    all_first = [item.first_timestamp_utc for item in file_audits if item.first_timestamp_utc]
    all_last = [item.last_timestamp_utc for item in file_audits if item.last_timestamp_utc]

    report = {
        "schema_version": 2,
        "kind": "professional_tick_delivery_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": contract["provider"],
        "delivery_reference": contract.get("delivery_reference"),
        "licence_reference": entitlement["licence_reference"],
        "permitted_uses": sorted(set(entitlement["permitted_uses"])),
        "files": [item.__dict__ for item in file_audits],
        "coverage": {
            "observed_instruments": observed_instruments,
            "expected_instruments": expected_instruments,
            "missing_instruments": missing_instruments,
            "first_timestamp_utc": min(all_first) if all_first else None,
            "last_timestamp_utc": max(all_last) if all_last else None,
            "file_count": len(file_audits),
        },
        "audit_failures": failures,
        "research_authorization": "DENIED" if failures else "PENDING_TIME_COVERAGE_AND_CROSS_SOURCE_AUDIT",
        "deployment_authorization": "DENIED",
        "next_gate": "Verify complete 2020-2025 coverage by pair, then perform independent-source reconciliation.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.report), "research_authorization": report["research_authorization"], "failures": failures}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
