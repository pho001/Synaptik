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
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx metal \
  -std=metal3.2 -fno-fast-math -Wall -Werror -isysroot "$SDK" \
  -c .task0069-msl-audit/runtime-source.metal \
  -o .task0069-msl-audit/runtime-source.air
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx metallib \
  .task0069-msl-audit/runtime-source.air \
  -o .task0069-msl-audit/runtime-source.metallib
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx metal-objdump \
  -d .task0069-msl-audit/runtime-source.air >"$WORK/runtime-source.air.ll"
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx metal-nm \
  .task0069-msl-audit/runtime-source.air >"$WORK/runtime-source.air.nm"
python3 verify-compiled-msl-audit.py "$WORK"
