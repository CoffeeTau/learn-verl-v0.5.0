#!/usr/bin/env bash
# Usage: bash project/scripts/run.sh bash project/scripts/smoke.sh data|models
set -euo pipefail
: "${AGENTIC_RUNS_DIR:?Use project/scripts/run.sh to launch this script}"
stage="${1:-}"
if [[ "$stage" != data && "$stage" != models ]]; then
    printf 'Usage: bash project/scripts/run.sh bash project/scripts/smoke.sh data|models\n' >&2
    exit 2
fi
shift
report_dir="$AGENTIC_RUNS_DIR/resource_smoke"
mkdir -p "$report_dir"
# Offline mode: tests must use downloaded resources and must not fetch replacements.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
printf 'Report: %s/%s_report.json\nLog: %s/%s.log\n' "$report_dir" "$stage" "$report_dir" "$stage"
"${PYTHON_BIN:-python3}" -u -m project.scripts.smoke "$stage" "$@" 2>&1 | tee "$report_dir/$stage.log"
