#!/bin/sh
set -eu

EVIDENCE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$EVIDENCE/../../../.." && pwd)
WORK="$EVIDENCE/build/generated-msl-audit"
rm -rf "$WORK"
mkdir -p "$WORK"

export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
[ -d "$DEVELOPER_DIR" ] || { printf '%s\n' "pinned Xcode is absent: $DEVELOPER_DIR" >&2; exit 1; }
python3 "$EVIDENCE/audit-provider.py" inventory \
  "$WORK/provider-inventory.json" "$DEVELOPER_DIR"
SDKROOT=$(DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx --show-sdk-path)
case "$SDKROOT" in
  "$DEVELOPER_DIR"/*) ;;
  *) printf '%s\n' "pinned xcrun resolved outside Xcode: $SDKROOT" >&2; exit 1 ;;
esac

cd "$EVIDENCE"
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx clang \
  -fobjc-arc -Wall -Wextra -Werror -isysroot "$SDKROOT" \
  -I "$ROOT/native/metal-macos-arm64/src" -framework Foundation \
  generated-source-fixture.m -o "$WORK/generated-source-fixture"
"$WORK/generated-source-fixture" "$WORK/generated-fixtures.metal"
./extract-generated-fixtures.py "$WORK/generated-fixtures.metal" "$WORK/fixtures.json"
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx metal \
  -std=metal3.2 -fno-fast-math -Wall -Wextra -Werror -isysroot "$SDKROOT" \
  -c "$WORK/generated-fixtures.metal" -o "$WORK/generated-fixtures.air"
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx metal \
  -std=metal3.2 -fno-fast-math -Wall -Wextra -Werror -isysroot "$SDKROOT" \
  -c "$EVIDENCE/forbidden-floating-fixtures.metal" \
  -o "$WORK/forbidden-floating-fixtures.air"
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx metallib \
  "$WORK/generated-fixtures.air" -o "$WORK/generated-fixtures.metallib"
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx metal-objdump \
  -d "$WORK/generated-fixtures.air" >"$WORK/generated-fixtures.air.ll"
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx metal-objdump \
  -d "$WORK/forbidden-floating-fixtures.air" \
  >"$WORK/forbidden-floating-fixtures.air.ll"
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx metal-nm \
  "$WORK/generated-fixtures.air" >"$WORK/generated-fixtures.air.nm"
./canonicalize-air.py "$WORK" "$SDKROOT" "$ROOT" \
  "$WORK/generated-fixtures.air.ll" "$WORK/generated-fixtures.air.nm" \
  "$WORK/forbidden-floating-fixtures.air.ll"
python3 "$EVIDENCE/audit-provider.py" generated-ledger \
  "$WORK/generated-fixtures.air.ll.canonical" \
  "$WORK/generated-fixtures.air.instructions.json" "$WORK/fixtures.json"
python3 "$EVIDENCE/audit-provider.py" negative-ledger \
  "$WORK/forbidden-floating-fixtures.air.ll.canonical" \
  "$WORK/forbidden-floating-fixtures.rejections.json"
python3 "$EVIDENCE/audit-provider.py" artifacts \
  "$WORK/artifact-observation.json" \
  "$WORK/provider-inventory.json" \
  "$WORK/generated-fixtures.metal" \
  "$WORK/fixtures.json" \
  "$WORK/generated-fixtures.air" \
  "$WORK/generated-fixtures.metallib" \
  "$WORK/generated-fixtures.air.ll.canonical" \
  "$WORK/generated-fixtures.air.nm.canonical" \
  "$WORK/generated-fixtures.air.instructions.json" \
  "$EVIDENCE/forbidden-floating-fixtures.metal" \
  "$WORK/forbidden-floating-fixtures.air" \
  "$WORK/forbidden-floating-fixtures.air.ll.canonical" \
  "$WORK/forbidden-floating-fixtures.rejections.json"
python3 verify-compiled-generated-msl-audit.py "$WORK"
