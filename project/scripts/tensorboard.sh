#!/usr/bin/env bash
# Run via run.sh; keep this terminal open while viewing TensorBoard.
set -euo pipefail
: "${AGENTIC_ROOT:?Launch through project/scripts/run.sh}"
exec "${PYTHON_BIN:-python3}" -m tensorboard.main \
  --logdir "$AGENTIC_ROOT/tensorboard" --host 127.0.0.1 --port 6006
