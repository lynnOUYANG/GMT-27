#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash third_party/gspan_cpp/build.sh
PYTHON="${PYTHON:-python}"
PYTHONPATH="$ROOT/third_party/vf2_cpp${PYTHONPATH:+:$PYTHONPATH}" \
  PYTHON="$PYTHON" bash third_party/vf2_cpp/build.sh

