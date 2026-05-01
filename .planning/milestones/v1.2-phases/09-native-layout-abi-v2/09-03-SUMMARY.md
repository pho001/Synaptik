---
phase: 09-native-layout-abi-v2
plan: "03"
subsystem: accelerator-runtime
tags: [fallback, reason-codes, cuda, layout-abi-v2]
requires:
  - phase: 09-01
    provides: layout ABI v2 metadata/status records
  - phase: 09-02
    provides: layout ABI v2 capability probes
provides:
  - ABI-v2-specific accelerator buffer reason codes
  - CUDA non-dense metadata rejection with layout ABI v2 diagnostics
  - documented AUTO/REQUIRE semantics
affects: [phase-10-gpu-layout-transform, phase-13-coverage-gates]
tech-stack:
  added: []
  patterns: [stable-reason-codes, visible-fallback]
key-files:
  created:
    - src/main/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2ReasonCodes.java
  modified:
    - src/main/java/backend/accelerator/buffer/AcceleratorBufferReasonCode.java
    - src/main/java/backend/cuda/buffer/CudaAcceleratorBufferBinder.java
    - src/test/java/backend/cuda/buffer/CudaAcceleratorBufferBinderTest.java
    - docs/metal-backend.md
    - docs/native-bridges-and-blas.md
requirements-completed: [GPULAYOUT-03]
duration: 14min
completed: 2026-04-30
---

# Phase 09: Native Layout ABI v2 - Plan 03 Summary

**ABI-v2-specific fallback reason codes with CUDA non-dense metadata diagnostics**

## Performance

- **Duration:** 14 min
- **Started:** 2026-04-30T12:51:00Z
- **Completed:** 2026-04-30T13:05:00Z
- **Tasks:** 4
- **Files modified:** 6

## Accomplishments

- Added six stable `NATIVE_LAYOUT_*` accelerator buffer reason codes.
- Added `AcceleratorLayoutAbiV2ReasonCodes` mapper from ABI v2 status to buffer reason code.
- Updated CUDA buffer metadata decisions so non-dense layouts reject with layout ABI v2 unavailable/mismatch/metadata-specific diagnostics.
- Documented AUTO fallback and REQUIRE failure semantics for layout ABI v2.

## Task Commits

1. **Tasks 1-4: Reason codes, CUDA policy mapping, tests, and docs** - `fbc8092` (feat)

## Files Created/Modified

- `src/main/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2ReasonCodes.java` - ABI status to reason-code mapper.
- `src/main/java/backend/accelerator/buffer/AcceleratorBufferReasonCode.java` - new stable `NATIVE_LAYOUT_*` codes.
- `src/main/java/backend/cuda/buffer/CudaAcceleratorBufferBinder.java` - CUDA layout ABI v2 rejection mapping.
- `src/test/java/backend/cuda/buffer/CudaAcceleratorBufferBinderTest.java` - AUTO/REQUIRE decision coverage for non-dense layout metadata.
- `docs/metal-backend.md` and `docs/native-bridges-and-blas.md` - fallback reason code documentation.

## Decisions Made

- Preserved Metal dense-physical logical-view behavior from the previous milestone; Phase 9 adds common reason codes and CUDA rejection diagnostics without regressing already-supported Metal logical-view flow.
- Kept dtype-specific dense-path errors unchanged; new `NATIVE_LAYOUT_*` codes apply when rejection is specifically about layout ABI v2 metadata.

## Deviations from Plan

The original plan expected Metal permuted-layout rejection to switch to layout ABI v2 unavailable. Execution preserved the existing Metal logical-view path because it is already supported through dense physical buffer metadata and is covered by prior milestone tests. CUDA still uses the new ABI-v2-specific rejection path for non-dense metadata.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Verification

Passed:

`./gradlew test --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest --tests backend.cuda.bridge.CudaFfmBridgeTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.accelerator.buffer.AcceleratorLayoutAbiV2DescriptorTest`

`./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest`

## Next Phase Readiness

Phase 10 can now distinguish "layout ABI v2 unavailable" from generic layout unsupported and can build GPU-side transform/view behavior on top of explicit capability fields.

---
*Phase: 09-native-layout-abi-v2*
*Completed: 2026-04-30*

