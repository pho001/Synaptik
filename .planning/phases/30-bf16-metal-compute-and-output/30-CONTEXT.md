# Phase 30 Context: BF16 Metal Compute And Output

## Goal

Add legal BF16 Metal compute/output coverage for high-value supported op families while preserving Phase 29 dtype truth.

## Requirements

- `METALBF16-01`: Legal `BFLOAT16` Metal regions can execute supported matmul/linear, elementwise, softmax/log-softmax, reduction, and normalization flows or reject with stable operation-specific capability reasons.
- `METALBF16-02`: BF16 buffer binding, materialization, tolerance policy, and report evidence preserve CPU parity against the existing CPU BF16 semantics.
- `METALBF16-03`: BF16 hot-path workloads show reduced CPU exits without regressing existing `FLOAT32` Metal or CPU hot paths.

## Current Findings

- Phase 29 added role-specific dtype truth and optional dtype ABI v3 discovery, but `MetalMpsCapabilities.computeDecision(BFLOAT16)` and `outputDecision(BFLOAT16)` still reject.
- The local macOS SDK exposes `MPSDataTypeBFloat16` in MPS core headers, so native BF16 is at least compile-time representable on this lane.
- The current `synaptik_apple_mps_compile_partition_f32` ABI only carries external input dtype codes. It does not carry node output dtype or final output dtype metadata.
- `MetalMpsFfmBridge.compile(...)` currently records all executable output dtypes as `FLOAT32` regardless of the compiled node dtype.
- `synaptik_apple_mps_stub.m` stores all output dtype codes as `1` and treats dtype codes as either BOOL or FLOAT32 in tensor-array and buffer execution.
- `MetalBufferAllocator` supports FLOAT32 input/output/readback and BOOL predicate input. It does not yet copy/read raw BF16 `short[]` storage.
- Existing BF16 CPU coverage and tests already exercise BF16 matmul/linear/fused-continuation semantics, so Phase 30 must not degrade CPU BF16 paths.

## Locked Decisions

- BF16 support must be admitted only when the native dtype ABI can describe the required roles and the execution path can preserve BF16 storage semantics.
- BF16 execution must not be represented as native BF16 compute unless MPSGraph actually receives BF16 tensors and the output publication path writes BF16 bytes.
- A converted FLOAT32 transport path may be introduced later only if reports explicitly label it as conversion, not native BF16 compute. Phase 30's main path targets native BF16 MPSGraph tensors.
- Existing FLOAT32 Metal paths must remain the default stable path and cannot be slowed by per-execute dtype planning.
- Unsupported BF16 operations/layouts must reject with stable dtype/layout/capability reasons rather than falling through to hidden CPU materialization.
- CUDA behavior remains capability-gated and must not inherit Metal BF16 claims.

## Phase Direction

The correct implementation sequence is:

1. Extend the lowered DAG/native bridge contract so BF16 output dtype is known at compile and execute time.
2. Add BF16 raw storage binding and materialization for Metal buffers and tensor-array fallback where appropriate.
3. Admit scoped BF16 op families only after parity/tolerance tests exist.
4. Add coverage gates for BF16 hot paths and keep FLOAT32 regression tests in the same verification set.

## Canonical References

- `.planning/ROADMAP.md` - Phase 30 scope and success criteria.
- `.planning/REQUIREMENTS.md` - `METALBF16-01/02/03`.
- `.planning/phases/29-metal-dtype-abi-and-capability-truth/29-VERIFICATION.md` - dtype truth baseline.
- `src/main/java/backend/metal/MetalMpsCapabilities.java` - current role-specific dtype decisions.
- `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java` - current compile/execute ABI path.
- `src/main/java/backend/metal/buffer/MetalBufferAllocator.java` - current Metal buffer storage transport.
- `src/main/native/apple/synaptik_apple_mps_stub.m` - native MPSGraph dtype handling.
- `docs/metal-backend.md` - current public Metal dtype contract.

---

*Phase: 30-bf16-metal-compute-and-output*
*Context gathered: 2026-05-02*
