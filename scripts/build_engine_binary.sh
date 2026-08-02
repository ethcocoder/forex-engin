#!/usr/bin/env bash
# Build a standalone `forexdesk-engine` binary with PyInstaller.
#
# The desktop app NEVER launches this binary — it only *detects* it to show a
# helpful first-run hint. You start the engine yourself (see DEPLOYMENT.md).
#
# Usage:
#   scripts/build_engine_binary.sh            # one-file binary -> dist/forexdesk-engine
#
# Notes:
#   - Build on the SAME OS family / arch as the target (no cross-PyInstaller).
#   - Data files are NOT bundled — they are downloaded via Prepare Data at runtime.
#   - Torch-free: run in the venv from engine-server/deps.sh (torch is mocked).

set -euo pipefail
cd "$(dirname "$0")/.."

VENV_DIR="engine-server/.venv"
PY="${VENV_DIR}/bin/python"

if [[ ! -x "${PY}" ]]; then
  echo "No venv found at ${VENV_DIR} — run engine-server/deps.sh first." >&2
  exit 1
fi

"${PY}" -m pip install --quiet pyinstaller

"${PY}" -m PyInstaller \
  --onefile \
  --name forexdesk-engine \
  --paths engine-server \
  --exclude-module torch \
  --exclude-module sslib \
  --exclude-module stable_baselines3 \
  --distpath dist \
  --workpath build/pyinstaller \
  --specpath build \
  --log-level WARN \
  engine-server/server.py

echo
echo "Built: $(pwd)/dist/forexdesk-engine"
echo "Test it with:  ./dist/forexdesk-engine  (then curl http://127.0.0.1:8737/api/health)"
