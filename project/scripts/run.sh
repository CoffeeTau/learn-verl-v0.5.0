#!/usr/bin/env bash
# Load machine configuration and run a command from the repository root.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
if [[ -f "$repo_root/.env" ]]; then
    set -a
    source "$repo_root/.env"
    set +a
fi

export AGENTIC_ROOT="${AGENTIC_ROOT:-$repo_root/runtime}"
if [[ "$AGENTIC_ROOT" != /* ]]; then
    printf 'AGENTIC_ROOT must be an absolute path: %s\n' "$AGENTIC_ROOT" >&2
    exit 2
fi
export AGENTIC_MODEL_DIR="$AGENTIC_ROOT/models/Qwen3-4B"
export AGENTIC_RETRIEVER_DIR="$AGENTIC_ROOT/models/e5-base-v2"
export AGENTIC_RAW_DATA_DIR="$AGENTIC_ROOT/data/raw/2wiki"
export AGENTIC_PROCESSED_DATA_DIR="$AGENTIC_ROOT/data/processed/2wiki_v1"
export AGENTIC_CORPUS_DIR="$AGENTIC_ROOT/data/corpus/2wiki_v1"
export AGENTIC_INDEX_DIR="$AGENTIC_ROOT/indexes/2wiki_v1/e5-base-v2"
export AGENTIC_RUNS_DIR="$AGENTIC_ROOT/runs"

cd -- "$repo_root"
if [[ "${1:-}" == --print-paths ]]; then
    for key in AGENTIC_ROOT AGENTIC_MODEL_DIR AGENTIC_RETRIEVER_DIR \
        AGENTIC_RAW_DATA_DIR AGENTIC_PROCESSED_DATA_DIR AGENTIC_CORPUS_DIR \
        AGENTIC_INDEX_DIR AGENTIC_RUNS_DIR; do
        printf '%s=%s\n' "$key" "${!key}"
    done
    exit 0
fi
if [[ $# -eq 0 ]]; then
    printf 'Usage: bash project/scripts/run.sh <command> [args...]\n' >&2
    exit 2
fi
exec "$@"
