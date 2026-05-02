---
phase: 45-metal-output-buffer-write-and-copy-closure
status: context
created: 2026-05-02
mode: auto
---

# Phase 45 Context: Metal Output Buffer Write And Copy Closure

## Goal

Prove whether MPSGraph writes into caller-provided Metal output buffers, or keep the current `MPSGRAPH_RESULT_COPY` classification while adding a lower-copy strategy and report gates that prevent false zero-copy claims.

## Locked Decisions

- Public `Tensor` remains CPU-readable/logical; device residency stays in `ExecutionState` and `DeviceBufferBinding`.
- Phase 44 custom RELU already proves `TRUE_OUTPUT_BUFFER_WRITE`, but only for `metalExecutionRoute=CUSTOM_KERNEL`.
- MPSGraph must remain classified as `MPSGRAPH_RESULT_COPY` unless a no-copy sentinel/alias probe proves true caller-output writes for a scoped operation family.
- The proof harness must run through the native shim, not by inferring from MPSGraph API shape or from normal execution that already performs `readBytes`.
- Copy strategy reports must keep route context visible: MPSGraph copy, custom direct write, tensor-array copy, or CPU fallback.
- Local benchmark/profile artifacts remain uncommitted unless deliberately promoted as fixtures.

## Code Starting Point

- `MetalMpsFfmBridge.executeBuffers(...)` calls native `synaptik_apple_mps_execute_partition_f32_buffers(...)` and reports `MetalNativeCopyStrategy.MPSGRAPH_RESULT_COPY`.
- The native MPSGraph buffer function wraps caller output buffers as `MPSGraphTensorData`, calls `runWithMTLCommandQueue:inputsArray:resultsArray:executionDescriptor:`, then conservatively copies returned `MPSNDArray` storage into the caller buffer with `readBytes:strideBytes:`.
- `MetalMpsFfmCustomKernelBridge.executeBuffers(...)` reports `TRUE_OUTPUT_BUFFER_WRITE` for scoped dense `FLOAT32` single-node `RELU`.
- Existing trace/report fields already expose `metalNativeCopyStrategy`, `metalOutputBufferWriteProven`, and `nativeDeviceCopyNs`.

## Phase Boundaries

In scope:

- Optional native no-copy probe symbol for MPSGraph buffer execution.
- Sentinel/alias tests that distinguish MPSGraph direct output writes from explicit native result copy.
- Conservative bridge classification if proof is absent or scoped only.
- Report/gate hardening for false zero-copy claims and unexpected route/copy regressions.
- Documentation and verification evidence.

Out of scope:

- Public device tensor API.
- Universal custom-kernel replacement for MPSGraph.
- Unproven zero-copy claim for every MPSGraph operation/dtype/layout.
- Committing local tuning profile output.
