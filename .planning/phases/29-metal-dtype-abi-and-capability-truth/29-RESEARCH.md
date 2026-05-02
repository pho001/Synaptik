# Phase 29 Research: Metal DType ABI And Capability Truth

## Summary

Phase 29 should add explicit dtype capability truth without changing the supported native execution set. The existing Metal backend is FLOAT32 compute/output with BOOL accepted only as a predicate-style external input. The next phases need a richer contract so BF16, BOOL output, and INT32 index execution can be added safely.

## Codebase Findings

### Current Java capability model

`MetalMpsCapabilities` currently exposes:

- `supportsComputeDType(DataType)` - true only for `FLOAT32`.
- `supportsOutputDType(DataType)` - true only for `FLOAT32`.
- `supportsExternalInputDType(DataType)` - true for `FLOAT32` and `BOOL`.
- `supportsExternalInputRole(...)` - role-gates `WHERE` predicate BOOL, FLOAT32 branches, unmasked SDPA data inputs, and default FLOAT32 inputs.
- `abiDataTypeCode(...)` - maps FLOAT32 to `1`, BOOL to `2`, rejects everything else.

This is correct for current execution, but it is too coarse for v1.5 because it does not distinguish storage representability, native compute, output publication, per-op dtype support, and "ABI cannot describe this yet".

### Current FFM ABI

`MetalMpsFfmBridge.compile(...)` sends external input dtype codes, ranks, and shapes into `synaptik_apple_mps_compile_partition_f32`. It does not pass node output dtype metadata or final output dtype metadata. After compile, Java currently publishes `dagSpec.outputNodeIds().stream().map(ignored -> DataType.FLOAT32).toList()`.

This means older `_f32` dylibs must not be allowed to appear as BF16/BOOL/INT32/FLOAT64 compute-capable. Phase 29 should add optional dtype ABI v3 discovery and descriptor validation, while leaving the existing compile ABI unchanged until later execution phases need the widened path.

### Native optional-symbol pattern

The existing layout ABI v2 pattern is a good template:

- native symbol: `synaptik_apple_mps_layout_abi_version`;
- native symbol: `synaptik_apple_mps_validate_layout_abi_v2`;
- Java optional handles in `MetalMpsFfmBridge.init()`;
- version and validation presence exposed through `MetalMpsBridgeCapabilities`;
- stable capability code/reason for unavailable or version mismatch.

Phase 29 should add a similar dtype ABI v3 pair, for example:

- `synaptik_apple_mps_dtype_abi_version`;
- `synaptik_apple_mps_validate_dtype_abi_v3`.

The stub can conservatively validate the current truth: FLOAT32 compute/output, FLOAT32 external data input, BOOL predicate input only, and explicit unsupported decisions for BF16, INT32, FLOAT64, and BOOL output.

### Reporting and coverage hooks

Relevant report and coverage surfaces include:

- `GpuLoweringCoverageMatrix` and docs, which intentionally say semantic support is still gated by backend dtype/layout/capability checks.
- coverage summary/report classes that already report target truth, native evidence, fallback reasons, selected region metrics, and CPU exits.
- Metal planner legality diagnostics that already reject unsupported dtype/layout/capability combinations.

Phase 29 should not make the shared matrix claim wider dtype execution. It should enrich backend-owned dtype fallback/rejection detail and report text so users can see whether a tensor is merely device-resident or actually computed/output natively in that dtype.

## Implementation Implications

- Add a durable dtype capability decision model rather than more booleans.
- Preserve existing boolean methods as wrappers for compatibility.
- Add explicit reason codes that can be rendered in planner diagnostics and reports.
- Add optional native dtype ABI v3 probing and expose it in bridge capabilities.
- Add tests for every `DataType` across storage, external input, compute, output, and selected operation roles.
- Keep `FLOAT64` explicitly unsupported for Metal native compute/output.

## Verification Plan

Focused tests should cover:

- dtype role matrix and reason code decisions;
- external role decisions for `WHERE` and SDPA;
- bridge capability defaults when dtype ABI v3 symbols are missing;
- native stub dtype ABI v3 version/validation when Metal tests are available;
- coverage/report rendering for dtype residency versus native dtype compute.

Recommended commands:

```bash
./gradlew test --tests MetalMpsCapabilitiesTest --tests MetalMpsFfmBridgeTest --tests GpuLoweringCoverageMatrixTest
./gradlew classes
./gradlew metalTest
```

---

*Research completed inline because GSD planner agents are unavailable in this runtime.*
