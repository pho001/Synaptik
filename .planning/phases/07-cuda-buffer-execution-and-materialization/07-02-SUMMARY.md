---
phase: 07-cuda-buffer-execution-and-materialization
plan: "02"
subsystem: accelerator-runtime
tags: [cuda, buffer-execution, materialization, execution-state]
requires:
  - phase: 07-01
    provides: CUDA buffer allocator, binding, materializer, and bridge buffer hooks
provides:
  - Prepared CUDA execution through native buffer bindings
  - CUDA-owned output residency publication after successful buffer execution
  - Graph-output materialization through ExecutionState.requireCpuReadable(...)
affects: [cuda, prepared-execution, accelerator-buffer-binding]
tech-stack:
  added: []
  patterns: [runtime buffer resolution, device-owned output publication, fake native bridge tests]
key-files:
  created: []
  modified:
    - src/main/java/backend/cuda/buffer/CudaAcceleratorBufferBinder.java
    - src/main/java/backend/cuda/exec/PreparedCudaExecutable.java
    - src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java
key-decisions:
  - "CUDA prepared execution now evaluates buffer decisions with runtime input and residency evidence."
  - "CUDA output bindings are attached as StorageResidency.DEVICE_OWNED only after executeBuffers returns successfully."
  - "AUTO mode records NATIVE_BUFFER_EXECUTION_FAILED and falls back to CPU; REQUIRE mode throws before tensor-array execution."
patterns-established:
  - "CudaAcceleratorBufferBinder keeps metadata-only decide(...) compatibility and adds a runtime-aware overload plus resolve(...)."
  - "PreparedCudaExecutable mirrors the Metal buffer flow while staying limited to dense FLOAT32 CUDA buffers."
requirements-completed: [CUDA-03, CUDA-04]
duration: 1h 10m
completed: 2026-04-30
---

# Phase 7 Plan 02: CUDA Prepared Buffer Execution Summary

**Prepared CUDA execution now resolves native buffer bindings, executes accepted dense FLOAT32 CUDA buffer decisions, and materializes device-owned graph outputs through execution state.**

## Performance

- **Duration:** 1h 10m
- **Started:** 2026-04-30T11:34:00Z
- **Completed:** 2026-04-30T12:44:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Extended `CudaAcceleratorBufferBinder` with runtime-aware decision checks and `AcceleratorBufferBindings<CudaBufferBinding>` resolution.
- Wired `PreparedCudaExecutable` to resolve prepared inputs, create a `CudaBufferAllocator`, call `executeBuffers`, and attach successful outputs as `StorageResidency.DEVICE_OWNED` with reason `cuda buffer binding output`.
- Registered `CudaDeviceToCpuMaterializer` per execution context so graph-output CPU reads use `ExecutionState.requireCpuReadable(...)`.
- Replaced the previous "not implemented" tests with portable fake CUDA bridge coverage for native buffer execution, graph-output materialization, AUTO fallback, and REQUIRE failure.

## Task Commits

1. **Task 1: Resolve CUDA input/output bindings from prepared execution** - `3fb7571` (feat)
2. **Task 2: Add portable CUDA buffer execution and materialization tests** - `3fb7571` (feat)

**Plan metadata:** this summary commit.

## Files Created/Modified

- `src/main/java/backend/cuda/buffer/CudaAcceleratorBufferBinder.java` - Runtime-aware CUDA input/output decision checks and binding resolution.
- `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java` - Native buffer execution path, output publication, and buffer failure policy.
- `src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java` - Portable fake bridge tests for execution, materialization, fallback, and required failure.

## Decisions Made

- Kept the existing metadata-only `decide(request, config)` behavior for policy tests and added a runtime-aware overload for prepared execution.
- Used `HOST_SHARED_DEVICE_BUFFER` for uploaded CUDA inputs whose CPU array remains current after upload, and `DEVICE_OWNED` for outputs written by CUDA.
- Treated buffer execution exceptions as `NATIVE_BUFFER_EXECUTION_FAILED`, with AUTO mode using CPU fallback and REQUIRE mode throwing before tensor-array execution.

## Verification

- `./gradlew test --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` - passed.
- `./gradlew test --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest --tests backend.cuda.buffer.CudaBufferAllocatorTest --tests backend.cuda.buffer.CudaDeviceToCpuMaterializerTest --tests backend.cuda.bridge.CudaFfmBridgeTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` - passed.
- `rg -n "AcceleratorBufferBindings<CudaBufferBinding>|createBufferAllocator|executeBuffers|StorageResidency.DEVICE_OWNED|cuda buffer binding output|NATIVE_BUFFER_EXECUTION_FAILED" src/main/java/backend/cuda` - passed.
- `rg -n "cudaBufferPathExecutesWithoutTensorArrayBridge|cudaBufferPathMaterializesGraphOutputThroughExecutionState|cudaBufferFailureFallsBackInAutoMode|cudaBufferFailureThrowsInRequireMode|CpuMaterializationReason.GRAPH_OUTPUT|GPU_CUDA|DEVICE_OWNED" src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java` - passed.
- `./gradlew classes` - passed.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The first runtime-aware binder change broke existing metadata-only binder policy tests. Fixed by preserving `decide(request, config)` as a metadata-only path and using the new overload only where prepared execution has runtime context.

## User Setup Required

None - portable tests use fake CUDA bridge/allocator fixtures and do not require CUDA hardware.

## Next Phase Readiness

Ready for `07-03-PLAN.md`: CUDA can now pass device-owned outputs to adjacent CUDA regions, so Plan 03 can prove compatible binding reuse and document the narrow Phase 7 contract.

---
*Phase: 07-cuda-buffer-execution-and-materialization*
*Completed: 2026-04-30*
