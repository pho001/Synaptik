#!/bin/sh
set -eu

EVIDENCE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WORK="$EVIDENCE/.task0069-msl-audit"
[ ! -e "$WORK" ] || { printf '%s\n' "audit workspace already exists: $WORK" >&2; exit 1; }
mkdir "$WORK"
trap 'rm -rf "$WORK"' EXIT HUP INT TERM
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
SDK=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk
export DEVELOPER_DIR

cd "$EVIDENCE"
./extract-runtime-source.py "$WORK/runtime-source.metal"

DEVELOPER_DIR="$DEVELOPER_DIR" xcodebuild -version >"$WORK/xcode-version.txt"
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx metal --version >"$WORK/metal-version.txt"
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx metallib --version >"$WORK/metallib-version.txt"
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx --show-sdk-version >"$WORK/sdk-version.txt"
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx --show-sdk-build-version >"$WORK/sdk-build.txt"
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx --show-sdk-path >"$WORK/sdk-path.txt"
cat "$WORK/runtime-source.metal" | \
  DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx metal \
    -x metal -std=metal3.2 -fno-fast-math -Wall -Wextra -Werror -isysroot "$SDK" \
    -c - -o "$WORK/runtime-source.air"
(
  cd "$WORK"
  DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx metallib \
    runtime-source.air -o runtime-source.metallib
  DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx metal-objdump \
    -d runtime-source.air >runtime-source.air.ll
  DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx metal-nm \
    runtime-source.air >runtime-source.air.nm
)
python3 "$EVIDENCE/verify-compiled-msl-audit.py" "$WORK"
