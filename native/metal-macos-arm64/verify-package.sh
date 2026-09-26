#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C

readonly LIBRARY_NAME="libsynaptik_metal_foundation.dylib"
readonly INSTALL_NAME="@rpath/libsynaptik_metal_foundation.dylib"
readonly SIGNATURE_IDENTIFIER="io.github.pho001.synaptik.metal.foundation"
readonly MANIFEST_NAME="manifest.json"
readonly CHECKSUM_NAME="SHA256SUMS"
readonly EXPECTED_FILE_DESCRIPTION="Mach-O 64-bit dynamically linked shared library arm64"
readonly DEPENDENCY_RECORD_PATTERN='^[[:blank:]]+([^[:blank:][:cntrl:]()]+)[[:blank:]]+\(compatibility version [0-9]+\.[0-9]+\.[0-9]+, current version [0-9]+\.[0-9]+\.[0-9]+\)$'

fail() {
    printf 'verify-package: %s\n' "$*" >&2
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

if [[ "$#" -ne 1 ]]; then
    fail "usage: $0 PACKAGE_DIRECTORY"
fi

PACKAGE_DIR="$1"
require_normalized_path_spelling "${PACKAGE_DIR}" "package directory"
[[ ! -L "${PACKAGE_DIR}" ]] || fail "package directory must not be a symlink"
[[ -d "${PACKAGE_DIR}" ]] || fail "package directory does not exist"

LIBRARY="${PACKAGE_DIR}/${LIBRARY_NAME}"
MANIFEST="${PACKAGE_DIR}/${MANIFEST_NAME}"
CHECKSUMS="${PACKAGE_DIR}/${CHECKSUM_NAME}"

for member in "${LIBRARY}" "${MANIFEST}" "${CHECKSUMS}"; do
    [[ ! -L "${member}" ]] || fail "package member must not be a symlink: ${member}"
    [[ -f "${member}" ]] || fail "package member must be a regular file: ${member}"
done

ENTRY_MARKERS="$(find "${PACKAGE_DIR}" ! -path "${PACKAGE_DIR}" -prune -exec printf x \;)"
[[ "${#ENTRY_MARKERS}" -eq 3 ]] || fail "package must contain exactly three top-level entries"

[[ "$(stat -f '%Lp' "${LIBRARY}")" == "755" ]] || fail "library mode must be 0755"
[[ "$(stat -f '%Lp' "${MANIFEST}")" == "644" ]] || fail "manifest mode must be 0644"
[[ "$(stat -f '%Lp' "${CHECKSUMS}")" == "644" ]] || fail "checksum mode must be 0644"

FILE_DESCRIPTION="$(file -b "${LIBRARY}")"
[[ "${FILE_DESCRIPTION}" == "${EXPECTED_FILE_DESCRIPTION}" ]] \
    || fail "library must be one arm64 Mach-O 64-bit dynamically linked shared library"

ARCHITECTURES="$(xcrun lipo -archs "${LIBRARY}")"
[[ "${ARCHITECTURES}" == "arm64" ]] || fail "library architecture must be exactly arm64"

HEADER_OUTPUT="$(otool -hv "${LIBRARY}")"
HEADER_FACTS="$(printf '%s\n' "${HEADER_OUTPUT}" \
    | awk '$1 == "MH_MAGIC_64" { print $1 " " $2 " " $5 }')"
[[ "${HEADER_FACTS}" == "MH_MAGIC_64 ARM64 DYLIB" ]] \
    || fail "Mach-O header must identify one 64-bit arm64 DYLIB"

BUILD_OUTPUT="$(xcrun vtool -show-build "${LIBRARY}")"
BUILD_FACTS="$(printf '%s\n' "${BUILD_OUTPUT}" \
    | awk '$1 == "platform" || $1 == "minos" { print $1 "=" $2 }')"
[[ "${BUILD_FACTS}" == $'platform=MACOS\nminos=26.0' ]] \
    || fail "LC_BUILD_VERSION must declare platform MACOS and minos 26.0 exactly once"

INSTALL_OUTPUT="$(otool -D "${LIBRARY}")"
INSTALL_LINE_COUNT="$(printf '%s\n' "${INSTALL_OUTPUT}" | awk 'NF { count++ } END { print count + 0 }')"
ACTUAL_INSTALL_NAME="$(printf '%s\n' "${INSTALL_OUTPUT}" | awk 'NR == 2 { print }')"
[[ "${INSTALL_LINE_COUNT}" == "2" && "${ACTUAL_INSTALL_NAME}" == "${INSTALL_NAME}" ]] \
    || fail "LC_ID_DYLIB install name is not exact"

LOAD_COMMANDS="$(otool -l "${LIBRARY}")"
if printf '%s\n' "${LOAD_COMMANDS}" | grep -Eq '^[[:space:]]*cmd LC_RPATH[[:space:]]*$'; then
    fail "LC_RPATH is forbidden"
fi
if printf '%s\n' "${LOAD_COMMANDS}" \
        | grep -Eq '^[[:space:]]*cmd LC_VERSION_MIN_MACOSX[[:space:]]*$'; then
    fail "legacy LC_VERSION_MIN_MACOSX is forbidden"
fi
EXPECTED_DYLIB_RECORDS="$(printf '%s\n' "${LOAD_COMMANDS}" | awk '
    $1 == "cmd" && $2 ~ /^(LC_ID_DYLIB|LC_LOAD_DYLIB|LC_LOAD_WEAK_DYLIB|LC_REEXPORT_DYLIB|LC_LOAD_UPWARD_DYLIB|LC_LAZY_LOAD_DYLIB)$/ {
        count++
    }
    END {
        print count + 0
    }
')"
[[ "${EXPECTED_DYLIB_RECORDS}" =~ ^[0-9]+$ && "${EXPECTED_DYLIB_RECORDS}" -gt 1 ]] \
    || fail "Mach-O dylib command inventory is incomplete"

LINK_OUTPUT="$(otool -L "${LIBRARY}")"
LINK_RECORD_LINES=()
LINK_LINE_INDEX=0
while IFS= read -r raw_line; do
    LINK_LINE_INDEX=$((LINK_LINE_INDEX + 1))
    if [[ "${LINK_LINE_INDEX}" -eq 1 ]]; then
        [[ "${raw_line}" == "${LIBRARY}:" ]] || fail "otool dependency header is not exact"
        continue
    fi
    [[ "${raw_line}" =~ ${DEPENDENCY_RECORD_PATTERN} ]] \
        || fail "malformed or ambiguous otool dependency record"
    LINK_RECORD_LINES[${#LINK_RECORD_LINES[@]}]="${raw_line}"
done <<< "${LINK_OUTPUT}"
[[ "${#LINK_RECORD_LINES[@]}" -eq "${EXPECTED_DYLIB_RECORDS}" ]] \
    || fail "otool dependency record count does not match Mach-O load commands"

LINK_PATHS=()
for raw_line in "${LINK_RECORD_LINES[@]}"; do
    [[ "${raw_line}" =~ ${DEPENDENCY_RECORD_PATTERN} ]] \
        || fail "validated otool dependency record changed unexpectedly"
    LINK_PATHS[${#LINK_PATHS[@]}]="${BASH_REMATCH[1]}"
done

FOUNDATION_COUNT=0
METAL_COUNT=0
MPSGRAPH_COUNT=0
LINK_INDEX=0
for dependency in "${LINK_PATHS[@]}"; do
    LINK_INDEX=$((LINK_INDEX + 1))
    if [[ "${LINK_INDEX}" -eq 1 ]]; then
        [[ "${dependency}" == "${INSTALL_NAME}" ]] || fail "otool install-name record is not exact"
        continue
    fi
    case "${dependency}" in
        *'/../'*|*'/./'*|*'//'*)
            fail "non-normalized dependency is forbidden: ${dependency}"
            ;;
    esac
    case "${dependency}" in
        /System/Library/Frameworks/Foundation.framework/Versions/*/Foundation)
            FOUNDATION_COUNT=$((FOUNDATION_COUNT + 1))
            ;;
        /System/Library/Frameworks/Metal.framework/Versions/*/Metal)
            METAL_COUNT=$((METAL_COUNT + 1))
            ;;
        /System/Library/Frameworks/MetalPerformanceShadersGraph.framework/Versions/*/MetalPerformanceShadersGraph)
            MPSGRAPH_COUNT=$((MPSGRAPH_COUNT + 1))
            ;;
        /System/Library/Frameworks/*|/usr/lib/*)
            ;;
        *)
            fail "non-system or tokenized dependency is forbidden: ${dependency}"
            ;;
    esac
done
[[ "${FOUNDATION_COUNT}" -eq 1 ]] || fail "Foundation must be linked exactly once"
[[ "${METAL_COUNT}" -eq 1 ]] || fail "Metal must be linked exactly once"
[[ "${MPSGRAPH_COUNT}" -eq 1 ]] \
    || fail "MetalPerformanceShadersGraph must be linked exactly once"

EXPECTED_EXPORTS="$(printf '%s\n' \
    synaptik_metal_foundation_abi_version \
    synaptik_metal_context_create \
    synaptik_metal_context_release \
    synaptik_metal_buffer_create \
    synaptik_metal_buffer_release \
    synaptik_metal_buffer_upload \
    synaptik_metal_buffer_download \
    synaptik_metal_mpsgraph_executable_create \
    synaptik_metal_mpsgraph_executable_release \
    synaptik_metal_mpsgraph_executable_run \
    synaptik_metal_neg_kernel_pipeline_create \
    synaptik_metal_neg_kernel_pipeline_release \
    synaptik_metal_neg_kernel_pipeline_run \
    | sort)"
ACTUAL_EXPORTS="$(nm -gUj "${LIBRARY}" | sed 's/^_//' | sort)"
[[ "${ACTUAL_EXPORTS}" == "${EXPECTED_EXPORTS}" ]] || fail "export set is not the exact ABI-v5 set"

if ! codesign --verify --strict --verbose=4 "${LIBRARY}" >/dev/null 2>&1; then
    fail "strict code-signature verification failed"
fi
SIGNATURE_OUTPUT="$(codesign --display --verbose=4 "${LIBRARY}" 2>&1)"
for exact_line in \
    "Identifier=${SIGNATURE_IDENTIFIER}" \
    "Signature=adhoc" \
    "TeamIdentifier=not set"; do
    MATCH_COUNT="$(printf '%s\n' "${SIGNATURE_OUTPUT}" | grep -Fxc -- "${exact_line}" || true)"
    [[ "${MATCH_COUNT}" == "1" ]] || fail "missing or duplicate signature fact: ${exact_line}"
done

LIBRARY_SIZE="$(stat -f '%z' "${LIBRARY}")"
LIBRARY_SHA256="$(shasum -a 256 "${LIBRARY}" | awk '{ print $1 }')"
[[ "${LIBRARY_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "library SHA-256 is malformed"

TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/synaptik-metal-verify.XXXXXX")"
cleanup() {
    rm -rf -- "${TEMP_DIR}"
}
trap cleanup EXIT

EXPECTED_MANIFEST="${TEMP_DIR}/${MANIFEST_NAME}"
printf '%s\n' \
    "{\"schemaVersion\":1,\"artifact\":{\"file\":\"${LIBRARY_NAME}\",\"size\":${LIBRARY_SIZE},\"sha256\":\"${LIBRARY_SHA256}\"},\"platform\":\"macos\",\"architecture\":\"arm64\",\"minimumMacosVersion\":\"26.0\",\"installName\":\"${INSTALL_NAME}\",\"rpaths\":[],\"nativeAbiVersion\":5,\"nodeSchemaVersion\":13,\"linkedFrameworks\":[\"Foundation\",\"Metal\",\"MetalPerformanceShadersGraph\"],\"signature\":{\"kind\":\"adhoc\",\"identifier\":\"${SIGNATURE_IDENTIFIER}\"}}" \
    > "${EXPECTED_MANIFEST}"
cmp -s "${EXPECTED_MANIFEST}" "${MANIFEST}" || fail "manifest is not canonical or does not match the dylib"

MANIFEST_SHA256="$(shasum -a 256 "${MANIFEST}" | awk '{ print $1 }')"
[[ "${MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "manifest SHA-256 is malformed"
EXPECTED_CHECKSUMS="${TEMP_DIR}/${CHECKSUM_NAME}"
printf '%s  %s\n%s  %s\n' \
    "${LIBRARY_SHA256}" "${LIBRARY_NAME}" \
    "${MANIFEST_SHA256}" "${MANIFEST_NAME}" \
    > "${EXPECTED_CHECKSUMS}"
cmp -s "${EXPECTED_CHECKSUMS}" "${CHECKSUMS}" || fail "SHA256SUMS is not canonical or complete"
(
    cd "${PACKAGE_DIR}"
    shasum -a 256 -c "${CHECKSUM_NAME}" >/dev/null
) || fail "SHA256SUMS verification failed"

printf 'Verified local Metal package: %s\n' "${PACKAGE_DIR}"
