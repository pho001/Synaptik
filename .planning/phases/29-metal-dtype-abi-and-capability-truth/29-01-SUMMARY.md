# 29-01 Summary: Metal DType Capability Model And Reason Codes

## Completed

- Added role-specific Metal dtype capability decisions for storage, external input, external input role, compute, output, and operation output dtype.
- Added stable Metal dtype roles and reason codes.
- Preserved existing boolean compatibility methods on `MetalMpsCapabilities`.
- Made `FLOAT64` explicitly unsupported for native Metal compute/output.
- Kept `BOOL` legal only for predicate-style external input roles, not native BOOL-producing output.
- Added tests covering all public `DataType` values and `WHERE` role-specific external input legality.

## Verification

- `./gradlew test --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.accelerator.residency.AcceleratorDTypeResidencyPolicyTest --tests GpuCoverageSummaryTest`

## Outcome

`METALDTYPE-01` now has a concrete Java-side truth model instead of coarse dtype booleans.
