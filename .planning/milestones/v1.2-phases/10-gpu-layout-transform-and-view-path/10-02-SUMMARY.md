---
phase: 10-gpu-layout-transform-and-view-path
plan: "02"
subsystem: accelerator-runtime
tags: [gpu-layout, device-residency, metal, cuda, execution-state]

requires:
  - phase: 10-gpu-layout-transform-and-view-path
    provides: 10-01 shared layout transform decision records and reason codes
provides:
  - Metal and CUDA view binding factories that reuse native handles
  - Pre-CPU-step metadata-only device layout view propagation
  - Required-mode failure before hidden CPU materialization for rejected view propagation
affects: [phase-10, metal, cuda, prepared-execution]

tech-stack:
  added: []
  patterns: [runtime-only device view propagation, borrowed native handle view bindings]

key-files:
  created:
    - src/main/java/graph/execution/DeviceLayoutViewPropagator.java
    - src/test/java/backend/cuda/buffer/CudaBufferBindingTest.java
    - src/test/java/graph/execution/DeviceLayoutViewPropagationTest.java
  modified:
    - src/main/java/backend/metal/buffer/MetalBufferBinding.java
    - src/main/java/backend/cuda/buffer/CudaBufferBinding.java
    - src/main/java/backend/runtime/ExecutionContext.java
    - src/main/java/graph/execution/PreparedExecution.java
    - src/test/java/backend/metal/buffer/MetalBufferBindingTest.java

key-decisions:
  - "Metadata-only layout propagation runs before CPU input materialization in PreparedExecution."
  - "View bindings borrow existing Metal/CUDA handles and do not register new resources."
  - "Expand view bindings use read access; other metadata-only view bindings use read-write access."

patterns-established:
  - "DeviceLayoutViewPropagator is package-private runtime glue, not a public Tensor API."
  - "REQUIRE mode turns rejected view propagation into an early accelerator-buffer error."

requirements-completed: [GPUVIEW-01, GPUVIEW-02, GPUVIEW-03]

duration: 10 min
completed: 2026-04-30
---

# Phase 10 Plan 02: Device Layout View Propagation Summary

**Metadata-only layout nodes can reuse Metal/CUDA device bindings before CPU materialization**

## Performance

- **Duration:** 10 min
- **Started:** 2026-04-30T12:44:30Z
- **Completed:** 2026-04-30T12:54:14Z
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments

- Added Metal and CUDA `viewOf(...)` factories that reuse source native handle objects while changing node id, layout, and access.
- Added `DeviceLayoutViewPropagator` and invoked it before `requireCpuReadableInputs(...)` in `PreparedExecution`.
- Added tests proving permute/expand propagate device bindings without CPU materialization, contiguous does not use metadata-only propagation, and REQUIRE mode fails early for missing source binding.
- Preserved existing Metal/CUDA prepared executable buffer policy tests.

## Task Commits

1. **Task 1: Add backend alias and propagation tests** - `01c1555` (test)
2. **Tasks 2-3: Implement view factories and pre-CPU-step propagation** - `4409e7f` (feat)

## Files Created/Modified

- `src/main/java/backend/metal/buffer/MetalBufferBinding.java` - Added borrowed-handle `viewOf(...)`.
- `src/main/java/backend/cuda/buffer/CudaBufferBinding.java` - Added borrowed-handle `viewOf(...)`.
- `src/main/java/backend/runtime/ExecutionContext.java` - Exposes runtime config to execution-time propagation logic.
- `src/main/java/graph/execution/DeviceLayoutViewPropagator.java` - Shared runtime propagation hook for metadata-only GPU layout views.
- `src/main/java/graph/execution/PreparedExecution.java` - Runs propagation before CPU materialization.
- `src/test/java/backend/metal/buffer/MetalBufferBindingTest.java` - Verifies Metal view binding semantics.
- `src/test/java/backend/cuda/buffer/CudaBufferBindingTest.java` - Verifies CUDA view binding semantics.
- `src/test/java/graph/execution/DeviceLayoutViewPropagationTest.java` - Verifies runtime propagation and required-mode behavior.

## Decisions Made

- `DeviceLayoutViewPropagator` only claims `METADATA_ONLY_VIEW` decisions; dense materialization stays for Plan 10-03.
- Missing source binding returns false in AUTO and throws in REQUIRE when the target backend requires accelerator buffer execution.
- The propagation hook attaches target bindings as `DEVICE_OWNED` with reason `device layout view propagation`.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The contiguous negative test originally used `permute().contiguous()` with an incomplete two-node snapshot. The test now uses `input.contiguous()` because Plan 10-02 only needs to prove dense materialization candidates are not metadata-only propagation.

## Verification

- `./gradlew test --tests backend.metal.buffer.MetalBufferBindingTest --tests backend.cuda.buffer.CudaBufferBindingTest --tests graph.execution.DeviceLayoutViewPropagationTest` - PASS
- `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` - PASS
- Acceptance greps for `viewOfReusesHandleWithTargetLayoutAndNodeId`, `DeviceLayoutViewPropagator`, `device layout view propagation`, and `GPU_LAYOUT_SOURCE_BINDING_UNAVAILABLE` - PASS

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for `10-03`: dense GPU materialization can build on the same planner and pre-CPU-step hook while adding backend capability gates.

---
*Phase: 10-gpu-layout-transform-and-view-path*
*Completed: 2026-04-30*
