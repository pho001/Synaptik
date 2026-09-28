#!/bin/sh
set -eu

EVIDENCE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$EVIDENCE/../../../.." && pwd)
WORK="$EVIDENCE/build/anchor-air-audit"
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
SDK="$DEVELOPER_DIR/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk"
export DEVELOPER_DIR

rm -rf "$WORK"
mkdir -p "$WORK"
python3 "$EVIDENCE/../0069/extract-runtime-source.py" "$WORK/anchor-runtime.metal"

cat "$WORK/anchor-runtime.metal" | \
  xcrun --sdk macosx metal \
    -x metal -std=metal3.2 -fno-fast-math -Wall -Wextra -Werror -isysroot "$SDK" \
    -c - -o "$WORK/anchor-runtime.air"
(
  cd "$WORK"
  xcrun --sdk macosx metallib anchor-runtime.air -o anchor-runtime.metallib
  xcrun --sdk macosx metal-objdump -d anchor-runtime.air >anchor-runtime.air.ll
  xcrun --sdk macosx metal-nm anchor-runtime.air >anchor-runtime.air.nm
)
python3 "$EVIDENCE/audit-anchor-air.py" "$WORK" "$WORK/actual.json"
cmp "$EVIDENCE/compiled-anchor-air-audit.json" "$WORK/actual.json"
printf '%s\n' 'Task0071 compiled anchor-epilogue MSL/AIR audit verified'
