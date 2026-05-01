---
phase: 09-native-layout-abi-v2
plan: "02"
subsystem: native-bridge
tags: [metal, cuda, ffm, layout-abi-v2]
requires:
  - phase: 09-01
    provides: layout ABI v2 descriptor and support records
provides:
  - Metal layout ABI v2 capability record
  - CUDA layout ABI v2 capability fields
  - optional native layout ABI v2 version and validation symbols
affects: [phase-10-gpu-layout-transform, metal-backend, cuda-backend]
tech-stack:
  added: []
  patterns: [optional-ffm-symbols, layered-capabilities]
key-files:
  created:
    - src/main/java/backend/metal/bridge/MetalMpsBridgeCapabilities.java
    - src/main/java/backend/metal/bridge/MetalMpsCapabilityCode.java
  modified:
    - src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java
    - src/main/java/backend/cuda/bridge/CudaFfmBridge.java
    - src/main/java/backend/cuda/bridge/CudaBridgeCapabilities.java
    - src/main/native/apple/synaptik_apple_mps_stub.m
    - src/main/native/cuda/synaptik_cuda_graph_stub.cu
requirements-completed: [GPULAYOUT-02]
duration: 18min
completed: 2026-04-30
---

# Phase 09: Native Layout ABI v2 - Plan 02 Summary

**Metal and CUDA layout ABI v2 capability probes with optional native symbol support**

## Performance

- **Duration:** 18 min
- **Started:** 2026-04-30T12:33:00Z
- **Completed:** 2026-04-30T12:51:00Z
- **Tasks:** 4
- **Files modified:** 13

## Accomplishments

- Added layered Metal bridge capability reporting.
- Extended CUDA capabilities with layout ABI v2 support/version fields.
- Added optional Metal/CUDA native symbols for layout ABI v2 version and metadata validation.
- Documented optional-symbol behavior and native verification commands.

## Task Commits

1. **Tasks 1-4: Capability records, optional symbols, tests, and docs** - `e903fd3` (feat)

## Files Created/Modified

- `src/main/java/backend/metal/bridge/MetalMpsBridgeCapabilities.java` - Metal layered capability record.
- `src/main/java/backend/metal/bridge/MetalMpsCapabilityCode.java` - Metal capability code enum.
- `src/main/java/backend/cuda/bridge/CudaBridgeCapabilities.java` - CUDA layout ABI v2 fields.
- `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java` - optional Metal layout ABI v2 symbol lookup.
- `src/main/java/backend/cuda/bridge/CudaFfmBridge.java` - optional CUDA layout ABI v2 symbol lookup.
- `src/main/native/apple/synaptik_apple_mps_stub.m` - Metal layout ABI v2 native symbols.
- `src/main/native/cuda/synaptik_cuda_graph_stub.cu` - CUDA layout ABI v2 native symbols.
- `docs/development.md` and `docs/metal-backend.md` - optional-symbol docs.

## Decisions Made

- Missing layout ABI v2 symbols report layout ABI v2 unavailable without changing dense bridge availability.
- Metal now has a layered capability record matching the CUDA capability pattern.

## Deviations from Plan

None - plan executed as specified.

## Issues Encountered

None.

## User Setup Required

None for portable checks. Optional native checks still require local Metal/CUDA tooling.

## Verification

Passed:

`./gradlew classes`

`./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.cuda.bridge.CudaFfmBridgeTest`

## Next Phase Readiness

Plan 09-03 can now map layout ABI v2 unavailable/mismatch statuses to stable buffer reason codes and required-mode behavior.

---
*Phase: 09-native-layout-abi-v2*
*Completed: 2026-04-30*

