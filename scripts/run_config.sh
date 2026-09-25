#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
cd "$ROOT"

if [[ "$#" -lt 3 ]]; then
  echo "Usage: $0 pretrain-finetune|finetune CONFIG OUTPUT_DIR [extra arguments]" >&2
  exit 2
fi
stage="$1"
config="$2"
output="$3"
shift 3
case "$stage" in
  pretrain-finetune)
    exec "$PYTHON" run_motif_pretrain_finetune.py --config "$config" \
      --output-dir "$output" --device "${DEVICE:-cuda}" "$@"
    ;;
  finetune)
    exec "$PYTHON" train_frozen_vq.py --config "$config" \
      --output-dir "$output" --device "${DEVICE:-cuda}" "$@"
    ;;
  *)
    echo "Unknown stage: $stage" >&2
    exit 2
    ;;
esac
