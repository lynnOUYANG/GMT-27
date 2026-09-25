#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
cd "$ROOT"

DATASETS=(molesol molbace zinc molhiv qm9 tox21 aqsol sider)
declare -A DEFAULT_SEEDS=(
  [molesol]=2 [molbace]=0 [zinc]=0 [molhiv]=0
  [qm9]=0 [tox21]=0 [aqsol]=0 [sider]=0
)
for dataset in "${DATASETS[@]}"; do
  read -r -a seeds <<< "${SEEDS:-${DEFAULT_SEEDS[$dataset]}}"
  "$PYTHON" run_motif_pretrain_finetune.py \
    --config "configs/motif_pretrain_finetune_${dataset}.json" \
    --output-dir "outputs/${dataset}" \
    --device "${DEVICE:-cuda}" \
    --seeds "${seeds[@]}"
done
