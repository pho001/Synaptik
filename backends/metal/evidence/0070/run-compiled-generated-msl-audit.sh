#!/bin/sh
set -eu

EVIDENCE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$EVIDENCE/../../../.." && pwd)
WORK="$EVIDENCE/.task0070-generated-msl-audit"
[ ! -e "$WORK" ] || { printf '%s\n' "audit workspace already exists: $WORK" >&2; exit 1; }
mkdir "$WORK"
trap 'rm -rf "$WORK"' EXIT HUP INT TERM

export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
[ -d "$DEVELOPER_DIR" ] || { printf '%s\n' "pinned Xcode is absent: $DEVELOPER_DIR" >&2; exit 1; }
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
DEVELOPER_DIR="$DEVELOPER_DIR" xcodebuild -version >"$WORK/xcode-version.txt"
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx metal --version >"$WORK/metal-version.txt"
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx metallib --version >"$WORK/metallib-version.txt"
printf '%s\n' "$SDKROOT" >"$WORK/sdk-path.txt"
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx --show-sdk-version >"$WORK/sdk-version.txt"
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx --show-sdk-build-version >"$WORK/sdk-build.txt"
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx metal \
  -std=metal3.2 -fno-fast-math -Wall -Werror -isysroot "$SDKROOT" \
  -c "$WORK/generated-fixtures.metal" -o "$WORK/generated-fixtures.air"
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx metallib \
  "$WORK/generated-fixtures.air" -o "$WORK/generated-fixtures.metallib"
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx metal-objdump \
  -d "$WORK/generated-fixtures.air" >"$WORK/generated-fixtures.air.ll"
DEVELOPER_DIR="$DEVELOPER_DIR" xcrun --sdk macosx metal-nm \
  "$WORK/generated-fixtures.air" >"$WORK/generated-fixtures.air.nm"
./canonicalize-air.py "$WORK" "$SDKROOT" "$ROOT" \
  "$WORK/generated-fixtures.air.ll" "$WORK/generated-fixtures.air.nm"
python3 verify-compiled-generated-msl-audit.py "$WORK"
