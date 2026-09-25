#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
cd "$ROOT"

export NODE_TOKEN_DATA_ROOT="${NODE_TOKEN_DATA_ROOT:-$ROOT/checkpoints/node_tokenizers}"
DATASETS=(molesol molbace zinc molhiv qm9 tox21 aqsol sider)
for dataset in "${DATASETS[@]}"; do
  "$PYTHON" train_node_token.py \
    --dataset "$dataset" \
    --device "${DEVICE:-cuda}"
done

