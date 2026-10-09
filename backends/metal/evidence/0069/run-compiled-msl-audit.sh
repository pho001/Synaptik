#!/bin/sh
set -eu

EVIDENCE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WORK="$EVIDENCE/.task0069-msl-audit"
[ ! -e "$WORK" ] || { printf '%s\n' "audit workspace already exists: $WORK" >&2; exit 1; }
mkdir "$WORK"
trap 'rm -rf "$WORK"' EXIT HUP INT TERM
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.921.5
SDK=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk
export DEVELOPER_DIR TOOLCHAINS
METAL_TOOLCHAIN_BIN=$(dirname "$(/usr/bin/xcrun --sdk macosx --find metal)")
METALLIB="$METAL_TOOLCHAIN_BIN/metallib"
[ -x "$METALLIB" ] || { printf '%s\n' "pinned Metal Toolchain has no metallib: $METALLIB" >&2; exit 1; }

cd "$EVIDENCE"
./extract-runtime-source.py "$WORK/runtime-source.metal"

xcodebuild -version >"$WORK/xcode-version.txt"
/usr/bin/xcrun --sdk macosx metal --version >"$WORK/metal-version.txt"
"$METALLIB" --version >"$WORK/metallib-version.txt"
/usr/bin/xcrun --sdk macosx --show-sdk-version >"$WORK/sdk-version.txt"
/usr/bin/xcrun --sdk macosx --show-sdk-build-version >"$WORK/sdk-build.txt"
/usr/bin/xcrun --sdk macosx --show-sdk-path >"$WORK/sdk-path.txt"
cat "$WORK/runtime-source.metal" | \
  /usr/bin/xcrun --sdk macosx metal \
    -x metal -std=metal3.2 -fno-fast-math -Wall -Wextra -Werror \
    -fmodules-cache-path="$WORK/module-cache" -isysroot "$SDK" \
    -c - -o "$WORK/runtime-source.air"
(
  cd "$WORK"
  "$METALLIB" \
    runtime-source.air -o runtime-source.metallib
  /usr/bin/xcrun --sdk macosx metal-objdump \
    -d runtime-source.air >runtime-source.air.ll
  /usr/bin/xcrun --sdk macosx metal-nm \
    runtime-source.air >runtime-source.air.nm
)
python3 "$EVIDENCE/verify-compiled-msl-audit.py" "$WORK"
