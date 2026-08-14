"""Run one fresh-data, no-order shadow-inference observation.

This command never imports a broker, execution engine, or order-routing module.
It is intentionally limited to feature-schema checks, model inference, readiness
checks, and an auditable blocked/observation-only report.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.contracts import (
    MarketDataContract,
    MarketDataEligibilityPolicy,
    assess_market_data_eligibility,
)


def _load_time_indexed_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp").sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a fresh-data observation only; no broker or order path exists."
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model-experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signal-threshold", type=float, default=0.0)
    args = parser.parse_args()

    metadata_path = args.model_experiment / "run_metadata.json"
    gate_path = args.model_experiment / "readiness_gate_report.json"
    model_path = args.model_experiment / "model.joblib"
    for path in (metadata_path, gate_path, model_path):
        if not path.exists():
            raise FileNotFoundError(f"Required model artifact is missing: {path}")

    raw = _load_time_indexed_csv(args.raw)
    features = _load_time_indexed_csv(args.features)
    eligibility = assess_market_data_eligibility(
        raw,
        MarketDataContract(pair="EUR_USD", provider="shadow_source"),
        MarketDataEligibilityPolicy(minimum_rows=256),
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    expected_columns = metadata.get("feature_columns") or metadata.get(
        "research_metadata", {}
    ).get("feature_columns") or metadata.get("feature_schema", {}).get("columns")
    if not expected_columns:
        raise ValueError("The model artifact does not contain an auditable feature schema.")
    missing = [column for column in expected_columns if column not in features]
    if missing:
        raise ValueError(f"Fresh features do not match the model schema: {missing}")

    candidates = features[expected_columns].apply(pd.to_numeric, errors="coerce")
    finite_rows = np.isfinite(candidates.to_numpy(dtype=float)).all(axis=1)
    if not finite_rows.any():
        raise ValueError("Fresh data contain no finite row for the model feature schema.")
    observation_timestamp = candidates.index[finite_rows][-1]
    feature_vector = candidates.loc[[observation_timestamp]]
    model = joblib.load(model_path)
    interval_lower = None
    interval_upper = None
    base_prediction_std = None
    model_abstain = False
    if hasattr(model, "predict_with_diagnostics"):
        diagnostics = model.predict_with_diagnostics(feature_vector)
        prediction = float(diagnostics["prediction"].iloc[0])
        interval_lower = float(diagnostics["interval_lower"].iloc[0])
        interval_upper = float(diagnostics["interval_upper"].iloc[0])
        base_prediction_std = float(diagnostics["base_prediction_std"].iloc[0])
        model_abstain = bool(diagnostics["abstain"].iloc[0])
    else:
        prediction = float(model.predict(feature_vector)[0])
    hypothetical_direction = (
        "FLAT"
        if model_abstain
        else "LONG"
        if prediction > args.signal_threshold
        else "SHORT"
        if prediction < -args.signal_threshold
        else "FLAT"
    )

    blockers: list[str] = []
    if not gate.get("passed_for_paper_candidate_review", False):
        blockers.append("model_failed_research_promotion_gates")
    if model_abstain:
        blockers.append("model_abstained_due_to_calibrated_uncertainty")
    blockers.extend(f"market_data_{reason}" for reason in eligibility.reasons)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "NO_ORDER_SHADOW_INFERENCE",
        "order_action": "NONE",
        "broker_connection": "NONE",
        "source_rows": int(len(raw)),
        "data_eligibility": eligibility.to_dict(),
        "source_first_timestamp": raw.index[0].isoformat(),
        "source_last_timestamp": raw.index[-1].isoformat(),
        "observation_timestamp": observation_timestamp.isoformat(),
        "observation_close": float(raw.loc[observation_timestamp, "close"]),
        "model_run_id": metadata["run_id"],
        "prediction": prediction,
        "interval_lower": interval_lower,
        "interval_upper": interval_upper,
        "base_prediction_std": base_prediction_std,
        "model_abstain": model_abstain,
        "hypothetical_direction": hypothetical_direction,
        "signal_threshold": args.signal_threshold,
        "promotion_gate_passed": bool(gate.get("passed_for_paper_candidate_review")),
        "risk_blockers": blockers,
        "simulation_status": "BLOCKED" if blockers else "OBSERVATION_ONLY",
        "disclaimer": (
            "This is an observational research output, not an order, recommendation, "
            "broker quote, or live trading authorisation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
