#!/usr/bin/env python3
"""Download and normalise HistData Generic ASCII tick archives for research.

This is a disclosed secondary source, not a replacement for full-resolution
institutional tick feeds. HistData documents EST timestamps without daylight
saving adjustments, an explicit data-quality disclaimer, and bid/ask fields in
its Generic ASCII tick files. Every output remains non-deployable until a
separate cross-source, gap, and execution-cost audit passes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

SOURCE_HOME = "https://www.histdata.com"
SOURCE_PAGE = SOURCE_HOME + "/download-free-forex-historical-data/?/ascii/tick-data-quotes/{instrument}/{year}/{month}"
SOURCE_FAQ = SOURCE_HOME + "/f-a-q/"


def parse_month(value: str) -> tuple[int, int]:
    if not re.fullmatch(r"\d{4}-\d{2}", value):
        raise ValueError("--month must use YYYY-MM")
    year, month = (int(part) for part in value.split("-"))
    if not 1 <= month <= 12:
        raise ValueError("month must be between 01 and 12")
    return year, month


def fetch_archive(instrument: str, year: int, month: int, timeout: int) -> tuple[bytes, str, dict[str, str]]:
    page_url = SOURCE_PAGE.format(instrument=instrument.lower(), year=year, month=month)
    session = requests.Session()
    session.headers.update({"User-Agent": "ForexEngin/1.0 research provenance downloader"})
    page = session.get(page_url, timeout=timeout)
    page.raise_for_status()

    fields = dict(
        re.findall(
            r'<input\s+type="hidden"\s+name="([^"]+)"\s+id="[^"]+"\s+value="([^"]*)"\s*/?>',
            page.text,
        )
    )
    required = {"tk", "date", "datemonth", "platform", "timeframe", "fxpair"}
    if not required.issubset(fields):
        raise RuntimeError(f"HistData page did not expose expected download form fields: {page_url}")
    if fields["fxpair"].upper() != instrument.upper() or fields["datemonth"] != f"{year}{month:02d}":
        raise RuntimeError("HistData form metadata does not match requested instrument/month")

    response = session.post(SOURCE_HOME + "/get.php", data=fields, timeout=timeout)
    response.raise_for_status()
    payload = response.content
    if not payload.startswith(b"PK\x03\x04"):
        preview = response.text[:300].replace("\n", " ").replace("\r", " ")
        raise RuntimeError(
            "HistData response is not a ZIP archive "
            f"(status={response.status_code}, content_type={response.headers.get('content-type')}, body_preview={preview!r})"
        )
    return payload, page_url, fields


def parse_archive(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise RuntimeError("Expected exactly one CSV file in HistData archive")
        with archive.open(csv_names[0]) as csv_file:
            frame = pd.read_csv(
                csv_file,
                header=None,
                names=["timestamp_est", "bid", "ask", "volume"],
                dtype={"timestamp_est": "string"},
            )

    frame["timestamp_est"] = pd.to_datetime(frame["timestamp_est"], format="%Y%m%d %H%M%S%f", errors="coerce")
    for column in ("bid", "ask", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid_timestamp = frame["timestamp_est"].isna()
    invalid_quote = frame["bid"].isna() | frame["ask"].isna() | (frame["bid"] <= 0) | (frame["ask"] <= 0) | (frame["bid"] > frame["ask"])
    valid = frame.loc[~(invalid_timestamp | invalid_quote)].copy()
    # HistData documents EST without daylight-saving adjustment; UTC is EST + 5 hours.
    valid["timestamp"] = (valid.pop("timestamp_est") + pd.Timedelta(hours=5)).dt.tz_localize("UTC")
    valid = valid[["timestamp", "bid", "ask", "volume"]].sort_values("timestamp", kind="stable").reset_index(drop=True)
    valid.attrs["invalid_rows"] = int((invalid_timestamp | invalid_quote).sum())
    return valid


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a declared secondary HistData tick archive")
    parser.add_argument("--instrument", default="EURUSD")
    parser.add_argument("--month", required=True, help="Calendar month, YYYY-MM")
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--timeout-seconds", type=int, default=60)
    args = parser.parse_args()

    instrument = args.instrument.replace("/", "").upper()
    year, month = parse_month(args.month)
    payload, page_url, fields = fetch_archive(instrument, year, month, args.timeout_seconds)
    ticks = parse_archive(payload)
    if ticks.empty:
        raise RuntimeError("HistData archive contains no valid bid/ask rows")

    base = args.output.resolve()
    raw_dir = base / "raw" / "histdata" / instrument / str(year)
    manifest_dir = base / "manifests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    stem = f"HISTDATA_{instrument}_T_{year}{month:02d}"
    raw_path = raw_dir / f"{stem}.zip"
    csv_path = base / f"{stem}.csv"
    manifest_path = manifest_dir / f"{stem}.manifest.json"
    raw_path.write_bytes(payload)
    ticks.to_csv(csv_path, index=False)

    manifest = {
        "schema_version": 1,
        "kind": "secondary_historical_bid_ask_tick_data",
        "provider": "HistData Generic ASCII",
        "provider_page": page_url,
        "provider_faq": SOURCE_FAQ,
        "instrument": instrument,
        "month": f"{year}-{month:02d}",
        "source_timestamp_convention": "EST without daylight-saving adjustment, converted to UTC by +5 hours",
        "source_quality_notice": "Provider disclaims warranty/certification; file-level gaps must be audited before research use.",
        "source_resolution_notice": "Secondary generic tick source; never substitute for full-resolution institutional tick data in HFT execution validation.",
        "compressed_sha256": hashlib.sha256(payload).hexdigest(),
        "compressed_bytes": len(payload),
        "normalised_csv": str(csv_path),
        "normalised_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "rows": int(len(ticks)),
        "invalid_rows_rejected": int(ticks.attrs.get("invalid_rows", 0)),
        "first_timestamp_utc": ticks["timestamp"].iloc[0].isoformat(),
        "last_timestamp_utc": ticks["timestamp"].iloc[-1].isoformat(),
        "download_form_metadata": {key: fields[key] for key in ("date", "datemonth", "platform", "timeframe", "fxpair")},
        "training_authorization": "RESEARCH_ONLY after source-specific gap audit; NOT AUTHORIZED for HFT/live execution validation.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "manifest": str(manifest_path), "rows": len(ticks), "authorization": manifest["training_authorization"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
