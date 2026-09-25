#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
cd "$ROOT"

DATASETS=(molesol molbace zinc molhiv qm9 tox21 aqsol sider)
for dataset in "${DATASETS[@]}"; do
  "$PYTHON" train_motif_tokenizer.py \
    --config "configs/motif_${dataset}.json" \
    --output-dir "checkpoints/motif_tokenizers/${dataset}" \
    --device "${DEVICE:-cuda}"
done

