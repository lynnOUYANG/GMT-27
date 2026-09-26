#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
cd "$ROOT"

DATASETS=(molesol molbace zinc molhiv qm9 tox21 aqsol sider)
for dataset in "${DATASETS[@]}"; do
  "$PYTHON" -m src.prepare_motif_artifacts \
    --config "configs/motif_${dataset}.json" \
    --verify-vf2-samples 8
done
