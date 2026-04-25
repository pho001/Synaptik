#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT_DIR/src/main/native/apple/synaptik_apple_mps_stub.m"
OUT_DIR="$ROOT_DIR/build/native/apple"
OUT_LIB="$OUT_DIR/libsynaptik_apple_mps.dylib"

mkdir -p "$OUT_DIR"

clang \
  -dynamiclib \
  -fobjc-arc \
  -O2 \
  -framework Foundation \
  -framework Metal \
  -framework MetalPerformanceShaders \
  -framework MetalPerformanceShadersGraph \
  -o "$OUT_LIB" \
  "$SRC"

echo "Built Apple MPS shim: $OUT_LIB"
echo "Use with: -Dsynaptik.apple.mps.lib=$OUT_LIB"
