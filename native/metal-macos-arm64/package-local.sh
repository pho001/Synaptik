#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
umask 022

readonly LIBRARY_NAME="libsynaptik_metal_foundation.dylib"
readonly INSTALL_NAME="@rpath/libsynaptik_metal_foundation.dylib"
readonly SIGNATURE_IDENTIFIER="io.github.pho001.synaptik.metal.foundation"
readonly MANIFEST_NAME="manifest.json"
readonly CHECKSUM_NAME="SHA256SUMS"

fail() {
    printf 'package-local: %s\n' "$*" >&2
    exit 1
}

require_normalized_path_spelling() {
    local path="$1"
    local label="$2"

    [[ -n "${path}" ]] || fail "${label} path must not be empty"
    [[ ! "${path}" =~ [[:cntrl:]] ]] || fail "${label} path must not contain control characters"
    case "${path}" in
        /|.|..|*/|*/.|*/..|./*|../*|*/./*|*/../*|*//*)
            fail "${label} path must not contain redundant separators or components"
            ;;
    esac
}

require_real_directory() {
    local path="$1"
    local label="$2"

    [[ ! -L "${path}" && -d "${path}" ]] || fail "${label} must be a real directory"
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
    require_real_directory "${path}" "${label}"
}

if [[ "$#" -ne 1 ]]; then
    fail "usage: $0 AD_HOC_SIGNED_DYLIB"
fi

INPUT="$1"
require_normalized_path_spelling "${INPUT}" "input dylib"
[[ ! -L "${INPUT}" ]] || fail "input dylib must not be a symlink"
[[ -f "${INPUT}" ]] || fail "input dylib must be a regular file"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
VERIFY_SCRIPT="${SCRIPT_DIR}/verify-package.sh"
BUILD_DIR="${SCRIPT_DIR}/build"
PACKAGE_PARENT="${BUILD_DIR}/package-v1"
FINAL_PACKAGE="${PACKAGE_PARENT}/macos-arm64"
STAGING_DIR=""
BACKUP_ROOT=""
OLD_PACKAGE_MOVED=0
PUBLISHED=0

cleanup() {
    if [[ -n "${STAGING_DIR}" \
            && ! -L "${BUILD_DIR}" && -d "${BUILD_DIR}" \
            && ! -L "${PACKAGE_PARENT}" && -d "${PACKAGE_PARENT}" \
            && ! -L "${STAGING_DIR}" && -d "${STAGING_DIR}" ]]; then
        rm -rf -- "${STAGING_DIR}"
    fi
    if [[ "${OLD_PACKAGE_MOVED}" -eq 1 && "${PUBLISHED}" -eq 0 \
            && -n "${BACKUP_ROOT}" && ! -L "${BACKUP_ROOT}" \
            && ! -L "${BACKUP_ROOT}/macos-arm64" \
            && -d "${BACKUP_ROOT}/macos-arm64" \
            && ! -L "${BUILD_DIR}" && -d "${BUILD_DIR}" \
            && ! -L "${PACKAGE_PARENT}" && -d "${PACKAGE_PARENT}" \
            && ! -e "${FINAL_PACKAGE}" && ! -L "${FINAL_PACKAGE}" ]]; then
        if mv -- "${BACKUP_ROOT}/macos-arm64" "${FINAL_PACKAGE}"; then
            OLD_PACKAGE_MOVED=0
        else
            printf 'package-local: failed to restore previous package from %s\n' \
                "${BACKUP_ROOT}/macos-arm64" >&2
        fi
    fi
    if [[ -n "${BACKUP_ROOT}" \
            && ! -L "${BUILD_DIR}" && -d "${BUILD_DIR}" \
            && ! -L "${PACKAGE_PARENT}" && -d "${PACKAGE_PARENT}" \
            && ! -L "${BACKUP_ROOT}" && -d "${BACKUP_ROOT}" \
            && ! -e "${BACKUP_ROOT}/macos-arm64" && ! -L "${BACKUP_ROOT}/macos-arm64" ]]; then
        rmdir "${BACKUP_ROOT}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

ensure_real_directory "${BUILD_DIR}" "build directory"
ensure_real_directory "${PACKAGE_PARENT}" "package parent"
[[ -f "${VERIFY_SCRIPT}" && -x "${VERIFY_SCRIPT}" && ! -L "${VERIFY_SCRIPT}" ]] \
    || fail "package verifier must be an executable regular repository file"

STAGING_DIR="$(mktemp -d "${PACKAGE_PARENT}/.macos-arm64.stage.XXXXXX")"
[[ ! -L "${STAGING_DIR}" && -d "${STAGING_DIR}" ]] \
    || fail "package staging path must be a real directory"
STAGED_LIBRARY="${STAGING_DIR}/${LIBRARY_NAME}"
STAGED_MANIFEST="${STAGING_DIR}/${MANIFEST_NAME}"
STAGED_CHECKSUMS="${STAGING_DIR}/${CHECKSUM_NAME}"

install -m 0755 "${INPUT}" "${STAGED_LIBRARY}"
LIBRARY_SIZE="$(stat -f '%z' "${STAGED_LIBRARY}")"
LIBRARY_SHA256="$(shasum -a 256 "${STAGED_LIBRARY}" | awk '{ print $1 }')"
[[ "${LIBRARY_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "library SHA-256 is malformed"

printf '%s\n' \
    "{\"schemaVersion\":1,\"artifact\":{\"file\":\"${LIBRARY_NAME}\",\"size\":${LIBRARY_SIZE},\"sha256\":\"${LIBRARY_SHA256}\"},\"platform\":\"macos\",\"architecture\":\"arm64\",\"minimumMacosVersion\":\"26.0\",\"installName\":\"${INSTALL_NAME}\",\"rpaths\":[],\"nativeAbiVersion\":5,\"nodeSchemaVersion\":15,\"linkedFrameworks\":[\"Foundation\",\"Metal\",\"MetalPerformanceShadersGraph\"],\"signature\":{\"kind\":\"adhoc\",\"identifier\":\"${SIGNATURE_IDENTIFIER}\"}}" \
    > "${STAGED_MANIFEST}"
chmod 0644 "${STAGED_MANIFEST}"

MANIFEST_SHA256="$(shasum -a 256 "${STAGED_MANIFEST}" | awk '{ print $1 }')"
[[ "${MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "manifest SHA-256 is malformed"
printf '%s  %s\n%s  %s\n' \
    "${LIBRARY_SHA256}" "${LIBRARY_NAME}" \
    "${MANIFEST_SHA256}" "${MANIFEST_NAME}" \
    > "${STAGED_CHECKSUMS}"
chmod 0644 "${STAGED_CHECKSUMS}"

"${VERIFY_SCRIPT}" "${STAGING_DIR}"
require_real_directory "${BUILD_DIR}" "build directory"
require_real_directory "${PACKAGE_PARENT}" "package parent"

if [[ -L "${FINAL_PACKAGE}" ]]; then
    fail "refusing to replace a symlink package path"
fi
if [[ -e "${FINAL_PACKAGE}" ]]; then
    [[ -d "${FINAL_PACKAGE}" ]] || fail "existing package path is not a directory"
    BACKUP_ROOT="$(mktemp -d "${PACKAGE_PARENT}/.macos-arm64.previous.XXXXXX")"
    [[ ! -L "${BACKUP_ROOT}" && -d "${BACKUP_ROOT}" ]] \
        || fail "package backup path must be a real directory"
    mv -- "${FINAL_PACKAGE}" "${BACKUP_ROOT}/macos-arm64"
    OLD_PACKAGE_MOVED=1
    [[ ! -L "${BACKUP_ROOT}/macos-arm64" && -d "${BACKUP_ROOT}/macos-arm64" ]] \
        || fail "previous package backup must be a real directory"
fi
require_real_directory "${BUILD_DIR}" "build directory"
require_real_directory "${PACKAGE_PARENT}" "package parent"

mv -- "${STAGING_DIR}" "${FINAL_PACKAGE}"
STAGING_DIR=""
[[ ! -L "${FINAL_PACKAGE}" && -d "${FINAL_PACKAGE}" ]] \
    || fail "published package must be a real directory"
PUBLISHED=1

if [[ "${OLD_PACKAGE_MOVED}" -eq 1 ]]; then
    [[ ! -L "${BACKUP_ROOT}" && -d "${BACKUP_ROOT}" ]] \
        || fail "package backup path must remain a real directory"
    rm -rf -- "${BACKUP_ROOT}/macos-arm64"
    OLD_PACKAGE_MOVED=0
fi
if [[ -n "${BACKUP_ROOT}" ]]; then
    rmdir "${BACKUP_ROOT}"
    BACKUP_ROOT=""
fi

printf 'Published verified local Metal package: %s\n' "${FINAL_PACKAGE}"
