---
phase: 44-custom-metal-kernel-execution-route
plan: 44-01
subsystem: accelerator-runtime
tags: [metal, custom-kernel, routing, buffer-binding]
requires:
  - phase: 39-metal-backend-router-and-zero-copy-closure
    provides: route-aware Metal execution diagnostics and custom-kernel SPI seam
provides:
  - scoped custom Metal kernel candidate contract for dense FLOAT32 RELU
  - prepare-time CUSTOM_KERNEL route selection when candidate and executable are both available
  - route and prepared-execution tests for selected and rejected custom kernel paths
affects: [metal, accelerator-routing, gpu-coverage]
tech-stack:
  added: []
  patterns: [prepare-time route eligibility before execute-time buffer validation]
key-files:
  created:
    - src/main/java/backend/metal/kernel/MetalCustomKernelCandidate.java
    - src/test/java/backend/metal/exec/MetalExecutionRouterTest.java
  modified:
    - src/main/java/backend/metal/exec/MetalExecutionRouter.java
    - src/main/java/backend/metal/exec/PreparedMetalExecutable.java
    - src/main/java/backend/metal/kernel/MetalCustomKernelBridge.java
    - src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java
key-decisions:
  - "CUSTOM_KERNEL route is only eligible for a single-node dense-buffer FLOAT32 RELU candidate in this phase."
  - "MPSGraph remains the default selected route when the custom bridge, executable, dtype, or candidate family is unsupported."
patterns-established:
  - "Custom kernel route selection is gated by backend-internal candidate metadata before runtime execution."
requirements-completed: [METALKERNEL-01, METALKERNEL-02, METALKERNEL-03]
duration: 35min
completed: 2026-05-02
---

# Phase 44-01: Custom Metal Kernel Candidate Contract Summary

**Scoped `relu_f32` custom Metal route can now be selected truthfully while unsupported candidates stay on MPSGraph.**

## Performance

- **Duration:** 35 min
- **Started:** 2026-05-02T13:55:00Z
- **Completed:** 2026-05-02T14:29:13Z
- **Tasks:** 5
- **Files modified:** 9

## Accomplishments

- Added `MetalCustomKernelCandidate` as the prepare-time eligibility contract for the first custom route.
- Updated `MetalExecutionRouter` so `CUSTOM_KERNEL` is selected only when buffer binding transport, candidate eligibility, and compiled executable evidence are all present.
- Extended prepared execution so selected custom routes dispatch through `MetalCustomKernelBridge.executeBuffers`.
- Added router and prepared-executable tests covering selected custom route, MPSGraph default, unsupported multi-node candidates, unsupported dtype, unavailable executable, trace attributes, and execution path.

## Task Commits

1. **Scoped custom kernel route contract and tests** - `1acb208` (metal)

## Files Created/Modified

- `src/main/java/backend/metal/kernel/MetalCustomKernelCandidate.java` - scoped custom-kernel candidate classifier.
- `src/main/java/backend/metal/kernel/MetalCustomKernelBridge.java` - explicit buffer execution SPI for custom kernels.
- `src/main/java/backend/metal/exec/MetalCustomKernelRouteAdapter.java` - candidate-aware custom route evidence.
- `src/main/java/backend/metal/exec/MetalExecutionRouter.java` - selects `CUSTOM_KERNEL` only for supported custom candidates.
- `src/main/java/backend/metal/exec/PreparedMetalExecutable.java` - dispatches buffer-bound execution to the selected custom bridge.
- `src/main/java/backend/metal/bridge/MetalMpsBridgeExecutionPath.java` - adds `CUSTOM_KERNEL` execution path.
- `src/test/java/backend/metal/exec/MetalExecutionRouterTest.java` - route decision coverage.
- `src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java` - execution and trace coverage.

## Decisions Made

The first custom kernel route is deliberately limited to `FLOAT32` single-node `RELU`. That keeps route truthfulness independent of the broader MPSGraph DAG support and gives Phase 44-02 a concrete native target.

## Deviations from Plan

Added `MetalCustomKernelCandidate.java` as a small explicit contract file. This keeps candidate rules reusable by the native bridge work in 44-02 instead of burying the rules in the router.

## Issues Encountered

None.

## Verification

- `./gradlew test --tests backend.metal.exec.MetalExecutionRouterTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest`
- `git diff --check -- src/main/java/backend/metal src/test/java/backend/metal`

## User Setup Required

None.

## Next Phase Readiness

44-02 can now implement the native `relu_f32` Metal kernel behind the established custom bridge contract without changing public tensor APIs or broadening route claims.

---
*Phase: 44-custom-metal-kernel-execution-route*
*Completed: 2026-05-02*
