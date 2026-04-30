---
phase: 08-cuda-observability-and-documentation-closure
plan: "02"
subsystem: accelerator-runtime
tags: [cuda, reason-codes, required-mode, fallback]
requires:
  - phase: 08-cuda-observability-and-documentation-closure
    provides: 08-01 CUDA trace and report parity
provides:
  - Portable CUDA required-mode tests for bridge, native ABI, native execution, stale CPU, and incompatible binding failures
  - Trace assertion that CUDA reason codes and stats attributes appear in run metadata
affects: [cuda-runtime, accelerator-buffer-policy, verification]
tech-stack:
  added: []
  patterns: [fake CUDA bridge required-mode tests, trace-attribute assertions without native hardware]
key-files:
  created: []
  modified:
    - src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java
key-decisions:
  - "Required CUDA buffer mode must throw before tensor-array bridge execution or CPU fallback can hide buffer unavailability."
  - "CUDA reason-code coverage remains portable through fake bridge tests rather than native hardware assumptions."
patterns-established:
  - "Fake CUDA bridge can simulate bridge, context, executable, ABI, and native execution failure layers."
  - "PreparedExecution trace tests can wrap a fake PreparedCudaExecutable to assert run metadata contract."
requirements-completed: [CUDA-06]
duration: 7 min
completed: 2026-04-30
---

# Phase 8 Plan 02: CUDA Reason-Code Matrix Summary

**CUDA required-mode failures now have portable tests proving stable reason codes and no hidden tensor-array or CPU fallback**

## Performance

- **Duration:** 7 min
- **Started:** 2026-04-30T11:17:00Z
- **Completed:** 2026-04-30T11:19:02Z
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments

- Added required-mode tests for bridge unavailable, native buffer ABI unavailable, native buffer execution failure, and stale CPU input without a CUDA binding.
- Added a traced execution test that verifies CUDA metadata contains `acceleratorBufferReasonCode`, `cudaExecutionPath`, `cudaFallbackReason`, and backend-neutral byte counters.
- Extended the fake CUDA bridge to simulate unavailable bridge/context/executable states while staying hardware independent.

## Task Commits

1. **Task 1-3: CUDA reason-code matrix and required-mode trace coverage** - `c8fa38f` (`test(08-02)`)

## Files Created/Modified

- `src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java` - required-mode and trace metadata coverage.

## Decisions Made

- Existing production behavior already threw before fallback for required-mode failures; this plan locked that behavior with exact tests.
- `BRIDGE_UNAVAILABLE`, `NATIVE_BUFFER_ABI_UNAVAILABLE`, `INPUT_NOT_CPU_CURRENT`, `INPUT_BINDING_UNAVAILABLE`, and `NATIVE_BUFFER_EXECUTION_FAILED` are asserted through portable fake scenarios.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Verification

- `rg -n "NATIVE_BUFFER_ABI_UNAVAILABLE|BUFFER_BINDINGS_DISABLED|INPUT_DTYPE_UNSUPPORTED|OUTPUT_DTYPE_UNSUPPORTED|INPUT_LAYOUT_UNSUPPORTED|OUTPUT_LAYOUT_UNSUPPORTED|INPUT_NOT_CPU_CURRENT|INPUT_BINDING_UNAVAILABLE" src/test/java/backend/cuda/buffer/CudaAcceleratorBufferBinderTest.java src/main/java/backend/cuda/buffer/CudaAcceleratorBufferBinder.java src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java`
- `rg -n "requiredModeBridgeUnavailableThrowsBridgeUnavailableBeforeTensorArray|requiredModeNativeBufferAbiUnavailableThrowsBeforeTensorArray|requiredModeNativeBufferFailureThrowsNativeBufferExecutionFailed|Accelerator buffer path is required for GPU_CUDA" src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java src/main/java/backend/cuda/exec/PreparedCudaExecutable.java`
- `rg -n "acceleratorBufferReasonCode|cudaExecutionPath|cudaFallbackReason|GPU_CUDA" src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java`
- `./gradlew test --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest`

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

CUDA reason-code behavior is locked for Plan 08-03 documentation and Plan 08-04 final verification.

## Self-Check: PASSED

---
*Phase: 08-cuda-observability-and-documentation-closure*
*Completed: 2026-04-30*
