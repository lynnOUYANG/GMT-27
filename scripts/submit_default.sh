#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_LOG="${1:-$ROOT/outputs/pipeline.log}"
mkdir -p "$(dirname "$RUN_LOG")"

nohup bash "$ROOT/scripts/run_pipeline.sh" >"$RUN_LOG" 2>&1 &
printf 'pid=%s\nlog=%s\n' "$!" "$RUN_LOG"

