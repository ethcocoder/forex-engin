import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from run_research_validation import validate_manifest_chain


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_chain(tmp_path: Path, research_authorization: str = "EXPLORATORY_RESEARCH_ONLY", live: str = "DENIED"):
    bars = tmp_path / "bars.csv"
    bars.write_text("timestamp,instrument,bid,ask\n2024-01-01T00:00:00Z,EURUSD,1.1,1.1001\n", encoding="utf-8")
    tick_sha = "b" * 64
    tick_manifest = tmp_path / "ticks.manifest.json"
    tick_manifest.write_text(
        json.dumps(
            {
                "kind": "real_historical_tick_data",
                "dataset_sha256": tick_sha,
                "instrument": "EURUSD",
                "research_authorization": research_authorization,
                "institutional_execution_validation": "DENIED",
                "broker_demo_authorization": "DENIED",
                "live_trading_authorization": live,
            }
        ),
        encoding="utf-8",
    )
    bars_manifest = tmp_path / "bars.manifest.json"
    bars_manifest.write_text(
        json.dumps(
            {
                "kind": "derived_real_tick_bars",
                "output_sha256": sha256(bars),
                "source_tick_sha256": tick_sha,
                "instrument": "EURUSD",
                "research_authorization": research_authorization,
                "institutional_execution_validation": "DENIED",
                "broker_demo_authorization": "DENIED",
                "live_trading_authorization": live,
            }
        ),
        encoding="utf-8",
    )
    return bars, bars_manifest, tick_manifest


def test_research_validation_accepts_explicit_exploratory_research_scope(tmp_path):
    bars, bars_manifest, tick_manifest = write_chain(tmp_path)
    chain = validate_manifest_chain(bars, bars_manifest, tick_manifest)
    assert chain["ticks"]["research_authorization"] == "EXPLORATORY_RESEARCH_ONLY"


def test_research_validation_rejects_missing_execution_denial(tmp_path):
    bars, bars_manifest, tick_manifest = write_chain(tmp_path, live="NOT_SET")
    with pytest.raises(ValueError, match="deny execution authorization"):
        validate_manifest_chain(bars, bars_manifest, tick_manifest)
