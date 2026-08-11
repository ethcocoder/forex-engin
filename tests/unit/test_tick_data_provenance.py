import hashlib
import json
import lzma
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from download_dukascopy_ticks import archive_url, decode_bi5, validate_ticks
from prepare_tick_bars import build_bars, verify_manifest


def test_archive_url_uses_zero_based_months():
    hour = datetime(2024, 1, 2, 3, tzinfo=timezone.utc)
    assert archive_url("EUR/USD", hour).endswith("/EURUSD/2024/00/02/03h_ticks.bi5")


def test_decode_bi5_scales_prices_and_preserves_utc_timestamp():
    source_hour = datetime(2024, 1, 2, 0, tzinfo=timezone.utc)
    payload = lzma.compress(struct.pack(">IIIff", 1_234, 109_876, 109_870, 2.0, 3.0))

    ticks = decode_bi5(payload, "EURUSD", source_hour)

    assert len(ticks) == 1
    assert ticks.loc[0, "timestamp"] == pd.Timestamp("2024-01-02T00:00:01.234Z")
    assert ticks.loc[0, "bid"] == pytest.approx(1.09870)
    assert ticks.loc[0, "ask"] == pytest.approx(1.09876)
    assert ticks.loc[0, "bid_volume"] == pytest.approx(3.0)
    assert ticks.loc[0, "ask_volume"] == pytest.approx(2.0)


def test_quote_validation_removes_crossed_market_rows():
    ticks = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02T00:00:00Z", "2024-01-02T00:00:01Z"]),
            "bid": [1.1, 1.2],
            "ask": [1.1001, 1.1],
            "bid_volume": [1.0, 1.0],
            "ask_volume": [1.0, 1.0],
        }
    )

    valid, invalid_count = validate_ticks(ticks)

    assert invalid_count == 1
    assert len(valid) == 1
    assert valid.iloc[0]["bid"] == pytest.approx(1.1)


def test_manifest_gate_rejects_changed_dataset(tmp_path: Path):
    dataset = tmp_path / "EURUSD_ticks.csv"
    dataset.write_text(
        "instrument,timestamp,bid,ask,bid_volume,ask_volume,spread,mid\n"
        "EURUSD,2024-01-02T00:00:00Z,1.1,1.1001,1,1,0.0001,1.10005\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "ticks.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "real_historical_tick_data",
                "dataset_path": dataset.name,
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "source": "Dukascopy Historical Data Export",
                "source_reference": "https://www.dukascopy.com/swiss/english/marketwatch/historical/",
                "rows": 1,
            }
        ),
        encoding="utf-8",
    )
    verify_manifest(dataset, manifest)

    dataset.write_text(dataset.read_text(encoding="utf-8") + "EURUSD,2024-01-02T00:01:00Z,1.1,1.1001,1,1,0.0001,1.10005\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify_manifest(dataset, manifest)


def test_bar_builder_never_forward_fills_empty_intervals():
    ticks = pd.DataFrame(
        {
            "instrument": ["EURUSD", "EURUSD"],
            "timestamp": ["2024-01-02T00:00:01Z", "2024-01-02T00:02:01Z"],
            "bid": [1.1, 1.2],
            "ask": [1.1001, 1.2001],
            "bid_volume": [1.0, 1.0],
            "ask_volume": [1.0, 1.0],
            "spread": [0.0001, 0.0001],
            "mid": [1.10005, 1.20005],
        }
    )

    bars = build_bars(ticks, "1min")

    assert len(bars) == 2
    assert not (bars["timestamp"] == pd.Timestamp("2024-01-02T00:01:00Z")).any()
