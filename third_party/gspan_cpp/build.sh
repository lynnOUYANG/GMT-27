#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/Compare"
g++ -std=c++17 -O3 -DNDEBUG -I. \
  graph.cpp misc.cpp dfs.cpp ismin.cpp gspan.cpp main.cpp \
  -o gspan_cli
chmod +x gspan_cli
./gspan_cli -h >/dev/null 2>&1 || test "$?" -eq 255
