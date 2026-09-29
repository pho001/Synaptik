#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer

fail() {
    printf 'qualify-low-precision-vendor-routes: %s\n' "$*" >&2
    exit 1
}

if [[ "$#" -ne 2 ]]; then
    fail "usage: $0 BRIDGE_DYLIB OUTPUT_TSV"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
BRIDGE="$1"
OUTPUT="$2"
[[ -f "${BRIDGE}" && ! -L "${BRIDGE}" ]] || fail "bridge must be a regular non-symlink file"
[[ -d "$(dirname "${OUTPUT}")" ]] || fail "output parent directory does not exist"
[[ ! -L "${OUTPUT}" ]] || fail "output must not be a symlink"

TEMP_DIR="$(mktemp -d "${SCRIPT_DIR}/build/.vendor-qualification.XXXXXX")"
cleanup() {
    rm -rf -- "${TEMP_DIR}"
}
trap cleanup EXIT

AIR="${TEMP_DIR}/low_precision_mpp_probe.air"
METALLIB="${TEMP_DIR}/low_precision_mpp_probe.metallib"
HOST="${TEMP_DIR}/low_precision_vendor_probe"
OBSERVATIONS="${TEMP_DIR}/observations.tsv"
STAGED_OUTPUT="${TEMP_DIR}/qualification-matrix.tsv"
MPP_SOURCE="${SCRIPT_DIR}/probes/low_precision_mpp_probe.metal"
HOST_SOURCE="${SCRIPT_DIR}/probes/low_precision_vendor_probe.m"

xcrun --sdk macosx metal \
    -std=metal4.0 \
    -mmacosx-version-min=26.2 \
    -Wall -Wextra -Werror \
    -c "${MPP_SOURCE}" \
    -o "${AIR}"
xcrun --sdk macosx metallib "${AIR}" -o "${METALLIB}"
xcrun --sdk macosx clang \
    -arch arm64 \
    -mmacosx-version-min=26.0 \
    -fobjc-arc \
    -Wall -Wextra -Werror \
    -framework Foundation \
    -framework Metal \
    -framework MetalPerformanceShaders \
    "${HOST_SOURCE}" \
    -o "${HOST}"

"${HOST}" "${BRIDGE}" "${METALLIB}" > "${OBSERVATIONS}"

observation() {
    local key="$1"
    local value
    value="$(awk -F '\t' -v key="${key}" '$1 == key { print substr($0, length($1) + 2) }' \
        "${OBSERVATIONS}")"
    [[ -n "${value}" ]] || fail "missing probe observation ${key}"
    [[ "$(awk -F '\t' -v key="${key}" '$1 == key { count++ } END { print count + 0 }' \
        "${OBSERVATIONS}")" == "1" ]] || fail "duplicate probe observation ${key}"
    printf '%s' "${value}"
}

GPU_FAMILY="$(observation gpu-family)"
OS_BUILD="$(observation os-build)"
SDK_VERSION="$(observation sdk-version)"
COMPILER_VERSION="$(observation compiler-version)"
FLAGS_OPTIONS="$(observation flags-options)"
[[ "$(observation probe-gpu-family)" == "${GPU_FAMILY}" ]] \
    || fail "runtime probe GPU family differs from bridge environment"
[[ "$(observation mps-device-support)" == "PASS" ]] \
    || fail "MPSSupportsMTLDevice gate did not pass"
[[ "$(observation mps-fp16-convolution-runtime)" == "PASS" ]] \
    || fail "FP16 MPSCNN runtime probe did not pass"
[[ "$(observation mps-fp16-accumulator-option)" == "FLOAT" ]] \
    || fail "MPSCNN FP16 accumulator option did not retain FLOAT"
[[ "$(observation mps-bfloat16-runtime)" == "REJECTED" ]] \
    || fail "classic MPSCNN unexpectedly accepted a BFLOAT16 data source"
for name in \
        mpp_matmul_f16_f32 \
        mpp_matmul_bf16_f32 \
        mpp_convolution_f16_f32 \
        mpp_convolution_bf16_f32; do
    [[ "$(observation "pipeline-${name}")" == "PASS" ]] \
        || fail "MPP target pipeline probe failed for ${name}"
done

BINARY_DIGEST="$(shasum -a 256 "${BRIDGE}" | cut -d ' ' -f 1)"
CAPABILITY_DIGEST="$(shasum -a 256 \
    "${REPOSITORY_ROOT}/testing/backend-conformance/src/test/resources/low-precision-capability-ledger-v1.tsv" \
    | cut -d ' ' -f 1)"
MPS_PROBE_DIGEST="$(shasum -a 256 "${HOST_SOURCE}" | cut -d ' ' -f 1)"
MPP_KERNEL_DIGEST="$(shasum -a 256 "${MPP_SOURCE}" | cut -d ' ' -f 1)"
MPP_PROBE_DIGEST="$(printf 'host-probe\t%s\nmpp-probe\t%s\n' \
    "${MPS_PROBE_DIGEST}" "${MPP_KERNEL_DIGEST}" | shasum -a 256 | cut -d ' ' -f 1)"
GENERATOR_DIGEST="$(shasum -a 256 \
    "${SCRIPT_DIR}/qualify-low-precision-vendor-routes.sh" | cut -d ' ' -f 1)"

printf '%s\n' \
    $'schema\troute-family\toperation\tdtype\tprobe\tobserved\tqualification\trejection-reason\tcandidate-options\tgpu-family\tos-build\tsdk-version\tcompiler-version\tbinary-digest\tflags-options\tcapability-manifest-hash\tprobe-digest\tgenerator-digest' \
    > "${STAGED_OUTPUT}"

row() {
    local family="$1" operation="$2" dtype="$3" probe="$4" observed="$5"
    local reason="$6" options="$7" digest="$8"
    printf '1\t%s\t%s\t%s\t%s\t%s\tREJECTED\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${family}" "${operation}" "${dtype}" "${probe}" "${observed}" \
        "${reason}" "${options}" "${GPU_FAMILY}" "${OS_BUILD}" "${SDK_VERSION}" \
        "${COMPILER_VERSION}" "${BINARY_DIGEST}" "${FLAGS_OPTIONS}" \
        "${CAPABILITY_DIGEST}" "${digest}" "${GENERATOR_DIGEST}" \
        >> "${STAGED_OUTPUT}"
}

row CLASSIC_MPS ANY BOTH MPSSupportsMTLDevice PASS \
    GATE_ONLY_NO_ADR0023_SEMANTIC_PROOF \
    'MPSSupportsMTLDevice=YES' "${MPS_PROBE_DIGEST}"
row CLASSIC_MPS CONV2D FLOAT16 MPSCNN_RUNTIME PASS \
    NO_COMPLETE_EXACT_LAYOUT_DOMAIN_PROOF \
    'accumulator=FLOAT;mpsimage=FLOAT16;layout=MPSIMAGE_2X2X1' "${MPS_PROBE_DIGEST}"
row CLASSIC_MPS CONV2D FLOAT16 MPSCNN_ACCUMULATOR_OPTION FLOAT \
    FP32_OPTION_WITHOUT_COMPLETE_ADR0023_FAMILY_PROOF \
    'accumulator=FLOAT;mpsimage=FLOAT16' "${MPS_PROBE_DIGEST}"
row CLASSIC_MPS CONV2D BFLOAT16 MPSCNN_RUNTIME REJECTED \
    PUBLIC_DATASOURCE_CONTRACT_EXCLUDES_BFLOAT16 \
    'datasource=MPSDataTypeBFloat16' "${MPS_PROBE_DIGEST}"
row CLASSIC_MPS ANY BFLOAT16 PUBLIC_ACCUMULATOR_GUARANTEE ABSENT \
    NO_PUBLIC_BFLOAT16_ACCUMULATOR_CONTRACT \
    'accumulator=UNSPECIFIED' "${MPS_PROBE_DIGEST}"

for operation in MATMUL2D CONVOLUTION2D; do
    for dtype in FLOAT16 BFLOAT16; do
        row MPP "${operation}" "${dtype}" COMPILE_PIPELINE PASS \
            NO_PUBLIC_INTERNAL_ACCUMULATOR_CONTRACT \
            'metal-language=4.0;deployment=26.2;__HAVE_TENSOR=1;relaxed-precision=false;destination=FLOAT32' \
            "${MPP_PROBE_DIGEST}"
        row MPP "${operation}" "${dtype}" INTERNAL_ACCUMULATOR_GUARANTEE \
            DESTINATION_FLOAT32_ONLY \
            LOW_TO_F32_DESTINATION_DOES_NOT_ESTABLISH_F32_INTERNAL_ACCUMULATION \
            'relaxed-precision=false;destination=FLOAT32;internal-accumulator=UNSPECIFIED' \
            "${MPP_PROBE_DIGEST}"
    done
done

mv -f -- "${STAGED_OUTPUT}" "${OUTPUT}"
printf 'Wrote qualified-negative vendor matrix: %s\n' "${OUTPUT}"
