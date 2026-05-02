# 29-02 Summary: Native ABI v3 DType Descriptor And Optional Probes

## Completed

- Added `MetalDTypeAbiV3Support` constants for descriptor roles and public dtype codes.
- Extended `MetalMpsBridgeCapabilities` with dtype ABI v3 support/version fields.
- Added `DTYPE_ABI_V3_UNAVAILABLE` and `DTYPE_ABI_V3_VERSION_MISMATCH` capability codes.
- Added optional FFM lookup for `synaptik_apple_mps_dtype_abi_version` and `synaptik_apple_mps_validate_dtype_abi_v3`.
- Added native Objective-C shim symbols for dtype ABI v3 discovery and conservative descriptor validation.
- Kept existing `_f32` compile/execute symbols as the only execution path.
- Added bridge tests proving dtype ABI v3 capability fields are exposed without throwing.

## Verification

- `./gradlew test --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.accelerator.residency.AcceleratorDTypeResidencyPolicyTest --tests GpuCoverageSummaryTest`
- `./gradlew metalTest`

## Outcome

`METALDTYPE-02` is covered: widened dtype capability now depends on optional/versioned dtype ABI v3 discovery instead of silent `_f32` assumptions.
