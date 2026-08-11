#!/usr/bin/env python3
"""Provenance gate for model retraining.

The prior helper wrote untrained checkpoint placeholders and was capable of
misrepresenting a successful retraining run. This command deliberately refuses
to create model artifacts. It verifies that a model-training request begins from
bars derived from a real tick-data manifest, then directs the operator to the
model-specific validated training commands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify real-data provenance before model retraining")
    parser.add_argument("--bars-manifest", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.bars_manifest.read_text(encoding="utf-8"))
    if manifest.get("kind") != "derived_real_tick_bars":
        raise SystemExit("Refusing retraining: input is not a manifest-derived real tick-bar dataset.")
    if not manifest.get("source_tick_sha256") or not manifest.get("output_sha256"):
        raise SystemExit("Refusing retraining: the bar manifest has incomplete provenance hashes.")

    raise SystemExit(
        "Input provenance verified. No models were trained by this command. "
        "Run the dedicated leakage-safe validation harness before producing a candidate artifact."
    )


if __name__ == "__main__":
    raise SystemExit(main())
