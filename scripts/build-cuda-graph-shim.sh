#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT_DIR/src/main/native/cuda/synaptik_cuda_graph_stub.cu"
OUT_DIR="$ROOT_DIR/build/native/cuda"

if [[ "$(uname -s)" == "Darwin" ]]; then
  OUT_LIB="$OUT_DIR/libsynaptik_cuda_graph.dylib"
else
  OUT_LIB="$OUT_DIR/libsynaptik_cuda_graph.so"
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "CUDA nvcc compiler not found on PATH." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

nvcc \
  -shared \
  -Xcompiler -fPIC \
  -O2 \
  -o "$OUT_LIB" \
  "$SRC"

echo "Built CUDA graph shim: $OUT_LIB"
echo "Use with: -Dsynaptik.cuda.graph.lib=$OUT_LIB"
