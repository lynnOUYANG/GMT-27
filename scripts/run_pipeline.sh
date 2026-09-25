#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash scripts/build_extensions.sh
bash scripts/prepare_all_motif_artifacts.sh
bash scripts/train_motif_tokenizers.sh
bash scripts/train_node_tokenizers.sh
bash scripts/run_downstream.sh

