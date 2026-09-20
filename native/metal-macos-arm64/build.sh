#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

mkdir -p "${BUILD_DIR}"
xcrun --sdk macosx clang \
    -arch arm64 \
    -dynamiclib \
    -fobjc-arc \
    -fvisibility=hidden \
    -Wall -Wextra -Werror \
    -framework Foundation \
    -framework Metal \
    "${SCRIPT_DIR}/src/synaptik_metal_foundation.m" \
    -o "${BUILD_DIR}/libsynaptik_metal_foundation.dylib"
