#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C

fail() {
    printf 'build: %s\n' "$*" >&2
    exit 1
}

ensure_real_directory() {
    local path="$1"
    local label="$2"

    [[ ! -L "${path}" ]] || fail "${label} must not be a symlink"
    if [[ -e "${path}" ]]; then
        [[ -d "${path}" ]] || fail "${label} must be a directory"
    else
        mkdir -- "${path}"
    fi
    [[ ! -L "${path}" && -d "${path}" ]] || fail "${label} must be a real directory"
}

require_replaceable_output() {
    [[ ! -L "${OUTPUT}" ]] || fail "output must not be a symlink"
    if [[ -e "${OUTPUT}" ]]; then
        [[ -f "${OUTPUT}" ]] || fail "output must be a regular file"
    fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BUILD_DIR="${SCRIPT_DIR}/build"
OUTPUT="${BUILD_DIR}/libsynaptik_metal_foundation.dylib"
STAGING_DIR=""

cleanup() {
    if [[ -n "${STAGING_DIR}" \
            && ! -L "${BUILD_DIR}" && -d "${BUILD_DIR}" \
            && ! -L "${STAGING_DIR}" && -d "${STAGING_DIR}" ]]; then
        rm -rf -- "${STAGING_DIR}"
    fi
}
trap cleanup EXIT

ensure_real_directory "${BUILD_DIR}" "build directory"
require_replaceable_output
STAGING_DIR="$(mktemp -d "${BUILD_DIR}/.native-build.XXXXXX")"
[[ ! -L "${STAGING_DIR}" && -d "${STAGING_DIR}" ]] \
    || fail "native staging path must be a real directory"
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

[[ ! -L "${STAGED_OUTPUT}" && -f "${STAGED_OUTPUT}" ]] \
    || fail "compiler output must be a regular non-symlink file"
chmod 0755 "${STAGED_OUTPUT}"
[[ ! -L "${BUILD_DIR}" && -d "${BUILD_DIR}" ]] \
    || fail "build directory must remain a real directory"
require_replaceable_output
mv -f -- "${STAGED_OUTPUT}" "${OUTPUT}"
[[ ! -L "${OUTPUT}" && -f "${OUTPUT}" ]] \
    || fail "published output must be a regular non-symlink file"
rmdir "${STAGING_DIR}"
STAGING_DIR=""
