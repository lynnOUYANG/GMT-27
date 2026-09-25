#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
cd "$ROOT"

bash scripts/build_extensions.sh
for dataset in molesol molbace; do
  seed=0
  [[ "$dataset" == "molesol" ]] && seed=2
  read -r -a seeds <<< "${SEEDS:-$seed}"
  "$PYTHON" prepare_motif_artifacts.py \
    --config "configs/motif_${dataset}.json" \
    --verify-vf2-samples 8
  "$PYTHON" train_motif_tokenizer.py \
    --config "configs/motif_${dataset}.json" \
    --output-dir "checkpoints/motif_tokenizers/${dataset}" \
    --device "${DEVICE:-cuda}"
  NODE_TOKEN_DATA_ROOT="$ROOT/checkpoints/node_tokenizers" \
    "$PYTHON" train_node_token.py \
    --dataset "$dataset" \
    --device "${DEVICE:-cuda}"
  "$PYTHON" run_motif_pretrain_finetune.py \
    --config "configs/motif_pretrain_finetune_${dataset}.json" \
    --output-dir "outputs/${dataset}" \
    --device "${DEVICE:-cuda}" \
    --seeds "${seeds[@]}"
done
