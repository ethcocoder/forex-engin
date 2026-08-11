import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_professional_tick_delivery import audit_file, validate_contract


@pytest.fixture
def contract():
    return {
        "provider": "LSEG Tick History",
        "delivery_reference": "test-extract-001",
        "timestamp_timezone": "UTC",
        "columns": {
            "timestamp": "EVENT_TIMESTAMP_UTC",
            "instrument": "RIC",
            "bid": "BID",
            "ask": "ASK",
        },
    }


@pytest.fixture
def entitlement():
    return {
        "licence_reference": "licence-test-001",
        "permitted_uses": ["research", "model_training", "model_validation"],
    }


def test_professional_delivery_audit_flags_crossed_quotes_and_non_monotonic_timestamps(tmp_path, contract):
    path = tmp_path / "delivery.csv"
    pd.DataFrame(
        {
            "EVENT_TIMESTAMP_UTC": ["2024-01-01T00:00:00Z", "2024-01-01T00:00:02Z", "2024-01-01T00:00:01Z"],
            "RIC": ["EURUSD=", "EURUSD=", "EURUSD="],
            "BID": [1.10, 1.12, 1.11],
            "ASK": [1.11, 1.11, 1.12],
        }
    ).to_csv(path, index=False)

    report = audit_file(path, contract, chunk_rows=2)

    assert report.rows == 3
    assert report.invalid_quote_rows == 1
    assert report.non_monotonic_timestamp_rows == 1
    assert report.instruments == ["EURUSD="]


def test_professional_contract_requires_utc_and_explicit_training_rights(contract, entitlement):
    validate_contract(contract, entitlement)

    contract["timestamp_timezone"] = "America/New_York"
    with pytest.raises(ValueError, match="UTC"):
        validate_contract(contract, entitlement)

    contract["timestamp_timezone"] = "UTC"
    entitlement["permitted_uses"] = ["research"]
    with pytest.raises(ValueError, match="model_training"):
        validate_contract(contract, entitlement)
