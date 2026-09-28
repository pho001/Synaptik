#!/bin/sh
set -eu

EVIDENCE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$EVIDENCE/../../../.." && pwd)
NATIVE="$ROOT/native/metal-macos-arm64"
WORK="$EVIDENCE/.task0070-dispatch-observer"
[ ! -e "$WORK" ] || { printf '%s\n' "observer workspace already exists: $WORK" >&2; exit 1; }
mkdir "$WORK"
trap 'rm -rf "$WORK"' EXIT HUP INT TERM

export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
[ -d "$DEVELOPER_DIR" ] || { printf '%s\n' "pinned Xcode is absent: $DEVELOPER_DIR" >&2; exit 1; }
OBSERVER="$WORK/libsynaptik_metal_test_observer.dylib"
FOUNDATION="$WORK/libsynaptik_metal_foundation.dylib"

xcrun --sdk macosx clang \
  -arch arm64 -mmacosx-version-min=26.0 -dynamiclib -fvisibility=hidden \
  -Wall -Wextra -Werror \
  -Wl,-install_name,@rpath/libsynaptik_metal_test_observer.dylib \
  "$EVIDENCE/dispatch-observer.c" -o "$OBSERVER"
xcrun --sdk macosx clang \
  -arch arm64 -mmacosx-version-min=26.0 -dynamiclib -fobjc-arc -fvisibility=hidden \
  -Wall -Wextra -Werror -DSYNAPTIK_METAL_TEST_DISPATCH_OBSERVER=1 \
  -Wl,-install_name,@rpath/libsynaptik_metal_foundation.dylib \
  -Wl,-rpath,@loader_path -L"$WORK" -lsynaptik_metal_test_observer \
  -framework Foundation -framework Metal -framework MetalPerformanceShadersGraph \
  "$NATIVE/src/synaptik_metal_foundation.m" -o "$FOUNDATION"

cd "$ROOT"
SYNAPTIK_METAL_TEST_LIBRARY="$FOUNDATION" \
SYNAPTIK_METAL_TEST_OBSERVER="$OBSERVER" \
  "$ROOT/gradlew" :backends:metal:test --rerun-tasks --no-build-cache \
    --no-configuration-cache \
    --tests io.github.pho001.synaptik.backend.metal.MetalPointwiseDispatchObserverNativeTest

TEST_CLASS="$ROOT/backends/metal/build/classes/java/test/io/github/pho001/synaptik/backend/metal/MetalPointwiseDispatchObserverNativeTest.class"
TEST_SOURCE="$ROOT/backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalPointwiseDispatchObserverNativeTest.java"
[ -f "$TEST_CLASS" ] || { printf '%s\n' "checkout-local observer class is absent: $TEST_CLASS" >&2; exit 1; }
[ "$TEST_CLASS" -nt "$TEST_SOURCE" ] || { printf '%s\n' "observer class is not current for this checkout: $TEST_CLASS" >&2; exit 1; }
