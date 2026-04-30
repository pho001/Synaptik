---
phase: 07-cuda-buffer-execution-and-materialization
plan: "01"
subsystem: accelerator-runtime
tags: [cuda, buffer-abi, ffm, materialization]
requires:
  - phase: 06-cuda-shim-and-capability-probe
    provides: CUDA native shim loading, capability diagnostics, and conservative buffer policy seams
provides:
  - CUDA buffer handle, binding, allocator, resource, and materializer classes under backend.cuda.buffer
  - CUDA bridge allocator and buffer execution hooks gated by native symbol availability
  - Native CUDA create/read/destroy/execute-buffer ABI symbols for dense FLOAT32 buffers
affects: [cuda, accelerator-buffer-binding, execution-state-materialization]
tech-stack:
  added: []
  patterns: [backend-owned native handles, run-scoped execution resources, capability-gated CUDA FFM symbols]
key-files:
  created:
    - src/main/java/backend/cuda/buffer/CudaBufferAccess.java
    - src/main/java/backend/cuda/buffer/CudaBufferHandle.java
    - src/main/java/backend/cuda/buffer/CudaBufferBinding.java
    - src/main/java/backend/cuda/buffer/CudaBufferResource.java
    - src/main/java/backend/cuda/buffer/CudaBufferAllocator.java
    - src/main/java/backend/cuda/buffer/CudaDeviceToCpuMaterializer.java
    - src/test/java/backend/cuda/buffer/CudaBufferAllocatorTest.java
    - src/test/java/backend/cuda/buffer/CudaDeviceToCpuMaterializerTest.java
  modified:
    - src/main/java/backend/cuda/bridge/CudaGraphBridge.java
    - src/main/java/backend/cuda/bridge/CudaFfmBridge.java
    - src/main/native/cuda/synaptik_cuda_graph_stub.cu
key-decisions:
  - "CUDA native handles and resource lifetime wrappers remain CUDA-specific under backend.cuda.*."
  - "Buffer binding support is advertised only when create, read, destroy, and execute-buffer native symbols are all available."
  - "Portable Phase 7 tests use fake native access and do not require CUDA hardware."
patterns-established:
  - "CudaBufferAllocator.NativeAccess isolates FFM/native calls from portable allocation and materialization tests."
  - "CudaDeviceToCpuMaterializer only supports matching active CUDA FLOAT32 dense bindings and delegates byte copy to the allocator."
requirements-completed: [CUDA-03, CUDA-04]
duration: 1h 4m
completed: 2026-04-30
---

# Phase 7 Plan 01: CUDA Buffer ABI Foundation Summary

**CUDA buffer ABI resources, FFM capability gating, and dense FLOAT32 materialization foundations for later prepared execution.**

## Performance

- **Duration:** 1h 4m
- **Started:** 2026-04-30T10:30:24Z
- **Completed:** 2026-04-30T11:34:00Z
- **Tasks:** 2
- **Files modified:** 11

## Accomplishments

- Added CUDA-specific buffer access, handle, binding, resource, allocator, and device-to-CPU materializer classes without changing the public `Tensor` API.
- Exposed native CUDA buffer ABI hooks through `CudaGraphBridge` and `CudaFfmBridge`, with support gated on the complete create/read/destroy/execute-buffer symbol set.
- Extended the CUDA native shim with dense FLOAT32 buffer create/read/destroy/partition execution entry points for representative RELU and ADD coverage.
- Added portable allocator/materializer tests backed by fake native access, so the required gate passes without CUDA hardware.

## Task Commits

1. **Task 1: Add CUDA buffer handle, binding, resource, allocator, and materializer classes** - `f3c9112` (feat)
2. **Task 2: Expose native CUDA buffer ABI through CudaFfmBridge** - `f3c9112` (feat)

**Plan metadata:** this summary commit.

## Files Created/Modified

- `src/main/java/backend/cuda/buffer/CudaBufferAccess.java` - CUDA read/write access enum.
- `src/main/java/backend/cuda/buffer/CudaBufferHandle.java` - Native handle identity and availability record.
- `src/main/java/backend/cuda/buffer/CudaBufferBinding.java` - CUDA implementation of shared `DeviceBufferBinding`.
- `src/main/java/backend/cuda/buffer/CudaBufferResource.java` - Idempotent run-scoped CUDA buffer cleanup resource.
- `src/main/java/backend/cuda/buffer/CudaBufferAllocator.java` - Dense FLOAT32 CUDA allocation, upload, read-back, and destroy wrapper.
- `src/main/java/backend/cuda/buffer/CudaDeviceToCpuMaterializer.java` - CUDA materializer for matching active dense FLOAT32 bindings.
- `src/main/java/backend/cuda/bridge/CudaGraphBridge.java` - Default allocator and buffer execution hooks.
- `src/main/java/backend/cuda/bridge/CudaFfmBridge.java` - FFM symbol lookup, capability gating, allocator factory, and buffer execution bridge.
- `src/main/native/cuda/synaptik_cuda_graph_stub.cu` - Native create/read/destroy/execute-buffer symbols.
- `src/test/java/backend/cuda/buffer/CudaBufferAllocatorTest.java` - Portable allocator coverage.
- `src/test/java/backend/cuda/buffer/CudaDeviceToCpuMaterializerTest.java` - Portable materializer support and delegation coverage.

## Decisions Made

- CUDA buffer ownership remains backend-specific; shared runtime surfaces only see `DeviceBufferBinding` and `ExecutionResource`.
- `CudaFfmBridge.supportsBufferBindings()` stays conservative and requires every native symbol needed for allocation, read-back, destruction, and buffer execution.
- Real CUDA hardware execution was not required for Plan 01; CPU-side fake native access is the portable correctness gate.

## Verification

- `./gradlew test --tests backend.cuda.buffer.CudaBufferAllocatorTest --tests backend.cuda.buffer.CudaDeviceToCpuMaterializerTest --tests backend.cuda.bridge.CudaFfmBridgeTest` - passed.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Early portable fake-native read-back test construction used a reinterpreted array segment and was corrected to copy directly into the destination array segment. The focused verification command passed after the fix.

## User Setup Required

None - no external service configuration required. Native CUDA hardware/tooling was not required for this portable plan gate.

## Next Phase Readiness

Ready for `07-02-PLAN.md`: prepared CUDA execution can now ask the bridge for a `CudaBufferAllocator`, resolve CUDA bindings, execute accepted buffer decisions, and materialize graph outputs through `ExecutionState`.

---
*Phase: 07-cuda-buffer-execution-and-materialization*
*Completed: 2026-04-30*
