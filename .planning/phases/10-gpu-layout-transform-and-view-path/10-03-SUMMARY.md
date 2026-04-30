---
phase: 10-gpu-layout-transform-and-view-path
plan: "03"
subsystem: accelerator-runtime
tags: [gpu-layout, dense-materialization, metal, cuda, native-shim]

requires:
  - phase: 10-gpu-layout-transform-and-view-path
    provides: 10-01 planner decisions and 10-02 pre-CPU-step propagation
provides:
  - Optional Metal/CUDA bridge API for dense layout materialization
  - FFM symbol lookup for layout contiguous F32 buffer transforms
  - Native Metal/CUDA shim symbols for contiguous F32 layout materialization
  - Conservative dense materialization routing through a run-scoped service seam
affects: [phase-10, phase-11, metal, cuda, native-bridge]

tech-stack:
  added: []
  patterns: [optional native symbols, run-scoped runtime services, dense layout materialization capability gate]

key-files:
  created:
    - src/main/java/graph/execution/DeviceLayoutMaterializer.java
  modified:
    - src/main/java/backend/metal/bridge/MetalMpsGraphBridge.java
    - src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java
    - src/main/java/backend/cuda/bridge/CudaGraphBridge.java
    - src/main/java/backend/cuda/bridge/CudaFfmBridge.java
    - src/main/java/backend/runtime/ExecutionContext.java
    - src/main/java/graph/execution/DeviceLayoutViewPropagator.java
    - src/main/native/apple/synaptik_apple_mps_stub.m
    - src/main/native/cuda/synaptik_cuda_graph_stub.cu
    - src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java
    - src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java

key-decisions:
  - "Dense layout materialization is capability-gated separately from dense buffer execution."
  - "DeviceLayoutViewPropagator only performs dense materialization when a run-scoped DeviceLayoutMaterializer is registered."
  - "AUTO mode falls through visibly when no dense materializer is available; REQUIRE mode fails with GPU_LAYOUT_TRANSFORM_UNSUPPORTED."

patterns-established:
  - "Native layout transform symbols are optional and do not gate existing buffer execution."
  - "Dense layout materialization attaches DEVICE_OWNED target bindings with backend-specific transition reasons."

requirements-completed: [GPUVIEW-01, GPUVIEW-02, GPUVIEW-03]

duration: 6 min
completed: 2026-04-30
---

# Phase 10 Plan 03: Dense GPU Layout Materialization Summary

**Optional Metal/CUDA dense layout materialization hooks with conservative runtime service routing**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-30T12:54:15Z
- **Completed:** 2026-04-30T13:00:21Z
- **Tasks:** 3
- **Files modified:** 11

## Accomplishments

- Added `supportsLayoutMaterialization()` and `materializeLayout(...)` to Metal and CUDA bridge SPIs.
- Added optional FFM lookup for `synaptik_apple_mps_layout_contiguous_f32_buffer` and `synaptik_cuda_graph_layout_contiguous_f32_buffer`.
- Added native Metal and CUDA shim symbols for F32 contiguous layout materialization.
- Extended `DeviceLayoutViewPropagator` so dense materialization decisions can use a run-scoped `DeviceLayoutMaterializer` service.
- Added tests and acceptance evidence for dense materialization decisions and unavailable capability fallback.

## Task Commits

1. **Tasks 1-3: Dense layout materialization hooks and routing seam** - `9ea694c` (feat)

## Files Created/Modified

- `src/main/java/backend/metal/bridge/MetalMpsGraphBridge.java` - Added optional layout materialization bridge API.
- `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java` - Looks up and invokes optional Metal layout materialization symbol.
- `src/main/java/backend/cuda/bridge/CudaGraphBridge.java` - Added optional layout materialization bridge API.
- `src/main/java/backend/cuda/bridge/CudaFfmBridge.java` - Looks up and invokes optional CUDA layout materialization symbol.
- `src/main/java/graph/execution/DeviceLayoutMaterializer.java` - Run-scoped dense materialization service contract.
- `src/main/java/graph/execution/DeviceLayoutViewPropagator.java` - Routes `DENSE_GPU_MATERIALIZATION` through the service seam.
- `src/main/native/apple/synaptik_apple_mps_stub.m` - Added Metal F32 layout contiguous symbol.
- `src/main/native/cuda/synaptik_cuda_graph_stub.cu` - Added CUDA F32 layout contiguous kernel and symbol.

## Deviations from Plan

The plan expected direct active bridge/allocator wiring in the propagator. Implementation used a run-scoped `DeviceLayoutMaterializer` service seam instead, because it avoids global bridge/allocator singletons and keeps current prepared executable ownership boundaries intact. This preserves the required capability-gated behavior while leaving backend registration explicit.

## Issues Encountered

None.

## Verification

- `./gradlew classes` - PASS
- `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests graph.execution.DeviceLayoutViewPropagationTest` - PASS
- Acceptance greps for optional symbols, FFM descriptors, `supportsLayoutMaterialization`, `DENSE_GPU_MATERIALIZATION`, and backend materialization reasons - PASS

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for `10-04`: E2E tests and docs can now report the exact supported scope, including the conservative service-gated dense materialization behavior.

---
*Phase: 10-gpu-layout-transform-and-view-path*
*Completed: 2026-04-30*
