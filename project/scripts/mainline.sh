#!/usr/bin/env bash
# Invoke through run.sh: prepare -> index -> v0 (separate processes).
set -euo pipefail
: "${AGENTIC_RUNS_DIR:?Launch through project/scripts/run.sh}"
stage="${1:-}"
case "$stage" in
  prepare) module=project.data_pipeline.prepare ;;
  index) module=project.retrieval.build_index ;;
  v0) module=project.evaluation.v0 ;;
  v1-prepare) module=project.training.prepare ;;
  v1) module=project.training.train ;;
  v2) module=project.training.v2 ;;
  v2-eval) module=project.evaluation.v2 ;;
  v3-retriever) module=project.retrieval.v3 ;;
  v3-eval) module=project.evaluation.v3 ;;
  v1-eval) module=project.evaluation.v1 ;;
  *) printf 'Usage: mainline.sh prepare|index|v0|v1-prepare|v1|v1-eval|v2|v2-eval|v3-retriever|v3-eval [options]\n' >&2; exit 2 ;;
esac
shift
mkdir -p "$AGENTIC_RUNS_DIR/mainline_logs"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
log_path="$AGENTIC_RUNS_DIR/mainline_logs/${stage}_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
"${PYTHON_BIN:-python3}" -u -m "$module" "$@" 2>&1 | tee "$log_path"
