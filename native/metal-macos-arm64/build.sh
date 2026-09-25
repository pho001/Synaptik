#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BUILD_DIR="${SCRIPT_DIR}/build"
OUTPUT="${BUILD_DIR}/libsynaptik_metal_foundation.dylib"
STAGING_DIR=""

cleanup() {
    if [[ -n "${STAGING_DIR}" && -d "${STAGING_DIR}" ]]; then
        rm -rf -- "${STAGING_DIR}"
    fi
}
trap cleanup EXIT

mkdir -p -- "${BUILD_DIR}"
STAGING_DIR="$(mktemp -d "${BUILD_DIR}/.native-build.XXXXXX")"
STAGED_OUTPUT="${STAGING_DIR}/libsynaptik_metal_foundation.dylib"

xcrun --sdk macosx clang \
    -arch arm64 \
    -mmacosx-version-min=26.0 \
    -dynamiclib \
    -fobjc-arc \
    -fvisibility=hidden \
    -Wall -Wextra -Werror \
    -Wl,-install_name,@rpath/libsynaptik_metal_foundation.dylib \
    -framework Foundation \
    -framework Metal \
    -framework MetalPerformanceShadersGraph \
    "${SCRIPT_DIR}/src/synaptik_metal_foundation.m" \
    -o "${STAGED_OUTPUT}"

chmod 0755 "${STAGED_OUTPUT}"
mv -f -- "${STAGED_OUTPUT}" "${OUTPUT}"
rmdir "${STAGING_DIR}"
STAGING_DIR=""
