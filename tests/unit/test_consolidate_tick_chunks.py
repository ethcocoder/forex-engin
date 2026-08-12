import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from consolidate_tick_chunks import consolidate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_chunk(root: Path, stem: str, start: str, end: str, timestamp: str) -> Path:
    data_dir = root / "data"
    manifest_dir = data_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    dataset = data_dir / f"{stem}.csv"
    pd.DataFrame(
        {
            "instrument": ["EURUSD"],
            "timestamp": [timestamp],
            "bid": [1.1000],
            "ask": [1.1001],
            "bid_volume": [1.0],
            "ask_volume": [1.0],
            "spread": [0.0001],
            "mid": [1.10005],
        }
    ).to_csv(dataset, index=False)
    manifest = manifest_dir / f"{stem}.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "real_historical_tick_data",
                "source": "Dukascopy Historical Data Export",
                "source_reference": "https://www.dukascopy.com/swiss/english/marketwatch/historical/",
                "source_class": "free_public_broker_historical_export",
                "research_authorization": "EXPLORATORY_RESEARCH_ONLY",
                "institutional_execution_validation": "DENIED",
                "broker_demo_authorization": "DENIED",
                "live_trading_authorization": "DENIED",
                "instrument": "EURUSD",
                "requested_start_utc": start,
                "requested_end_utc_exclusive": end,
                "dataset_path": dataset.name,
                "dataset_sha256": sha256(dataset),
                "rows": 1,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_consolidate_preserves_manifest_chain_and_research_only_scope(tmp_path):
    first = write_chunk(
        tmp_path,
        "EURUSD_ticks_20240101T0000Z_20240101T0100Z",
        "2024-01-01T00:00:00Z",
        "2024-01-01T01:00:00Z",
        "2024-01-01T00:10:00Z",
    )
    second = write_chunk(
        tmp_path,
        "EURUSD_ticks_20240101T0100Z_20240101T0200Z",
        "2024-01-01T01:00:00Z",
        "2024-01-01T02:00:00Z",
        "2024-01-01T01:10:00Z",
    )
    output = tmp_path / "data" / "consolidated" / "EURUSD_2h.csv"

    dataset, manifest = consolidate([second, first], output)
    report = json.loads(manifest.read_text(encoding="utf-8"))

    assert dataset == output
    assert len(pd.read_csv(dataset)) == 2
    assert [item["manifest"] for item in report["manifest_chain"]] == [first.name, second.name]
    assert report["research_authorization"] == "EXPLORATORY_RESEARCH_ONLY"
    assert report["live_trading_authorization"] == "DENIED"


def test_consolidate_rejects_non_contiguous_chunk_ranges(tmp_path):
    first = write_chunk(
        tmp_path,
        "EURUSD_ticks_20240101T0000Z_20240101T0100Z",
        "2024-01-01T00:00:00Z",
        "2024-01-01T01:00:00Z",
        "2024-01-01T00:10:00Z",
    )
    gap = write_chunk(
        tmp_path,
        "EURUSD_ticks_20240101T0200Z_20240101T0300Z",
        "2024-01-01T02:00:00Z",
        "2024-01-01T03:00:00Z",
        "2024-01-01T02:10:00Z",
    )

    with pytest.raises(ValueError, match="not contiguous"):
        consolidate([first, gap], tmp_path / "data" / "consolidated" / "broken.csv")
