# Phase 2: Metal Layout-Aware Device Flow - Research

**Researched:** 2026-04-29
**Status:** Ready for planning

## Phase Summary

Phase 2 should use the Phase 1 layout-aware accelerator buffer ABI to let safe Metal values remain device-resident across view/layout operations. The practical target is not broad GPU fusion or a public device tensor API. The target is a conservative Metal execution path where layout metadata is classified, accepted, transformed on device, or rejected with visible reason codes.

No `002-CONTEXT.md` exists, so roadmap, requirements, Phase 1 artifacts, and `AGENTS.md` are the active planning source of truth.

## Current Implementation Facts

- `AcceleratorBufferLayout` already carries dtype, shape, strides, storage offset, logical element count, logical byte length, and `AcceleratorBufferLayoutClass`.
- `MetalAcceleratorBufferBinder.unsupportedBufferLayoutReason(...)` currently accepts only `DENSE_CONTIGUOUS`; every zero-offset view, non-zero-offset view, permuted/strided view, broadcast view, and unsupported layout is rejected before allocation or execution.
- `MetalBufferAllocator.createInputBinding(...)` and `createPredicateInputBinding(...)` still require contiguous zero-offset Java tensors because they upload from CPU arrays into shared buffers.
- `MetalBufferAllocator.createOutputBinding(...)` still requires `DENSE_CONTIGUOUS` outputs.
- `MetalMpsFfmBridge.validateBufferBindings(...)` validates node id, dtype, availability, and access but does not validate or pass layout metadata to native code.
- Native `synaptik_apple_mps_execute_partition_f32_buffers(...)` constructs `MPSGraphTensorData` from `MTLBuffer` plus shape only; it does not receive strides or storage offsets.
- `RuntimeMemoryBinder` aliases CPU runtime tensors for `NOOP`, `EXPAND`, `SELECT`, `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`, and contiguous-input `RESHAPE`.
- Existing tests already prove dense adjacent Metal executables reuse an intermediate `MetalBufferBinding` without CPU materialization.
- Existing tests explicitly assert conservative fallback for non-dense input and output layouts. Phase 2 must update those tests rather than adding contradictory coverage.

## Key Files

### Runtime Buffer Policy

- `src/main/java/backend/metal/buffer/MetalAcceleratorBufferBinder.java`
  - Main preflight decision point.
  - Needs layout-specific classification instead of single dense-only rejection.
  - Should produce stable reason text for represented layout, device-contiguous transform, or fallback.

- `src/main/java/backend/metal/buffer/MetalBufferAllocator.java`
  - Owns Metal input/output buffer creation and CPU materialization reads.
  - Likely needs an API to create a binding from an existing handle and alternate logical layout, or to allocate dense physical storage while preserving logical output metadata.

- `src/main/java/backend/metal/buffer/MetalBufferBinding.java`
  - Already stores shared layout metadata and backend-owned handle.
  - Can represent a logical view over a handle only if the code distinguishes logical layout from physical buffer shape when needed.

- `src/main/java/backend/metal/buffer/MetalDeviceToCpuMaterializer.java`
  - Must remain correct for graph outputs, CPU consumers, and gradient publication.
  - Current `supports(...)` requires target and binding layout equality, then allocator requires contiguous zero-offset destination. Phase 2 must avoid claiming support for non-materializable layouts unless materialization copies into the correct logical CPU view.

### Metal Execution

- `src/main/java/backend/metal/exec/PreparedMetalExecutable.java`
  - Builds `AcceleratorBufferRequest` from runtime tensors.
  - Publishes successful buffer outputs as `DEVICE_OWNED`.
  - Must preserve fallback behavior and `REQUIRE` mode errors when layout handling is unsupported.

- `src/main/java/backend/metal/bridge/MetalMpsGraphBridge.java`
  - SPI for buffer execution.
  - Additive ABI/capability methods belong here if Java needs to know native layout support.

- `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java`
  - FFM symbol loading and validation.
  - Current optional-symbol pattern is the right way to add native capability/version checks.

- `src/main/native/apple/synaptik_apple_mps_stub.m`
  - Current buffer execution receives handles and counts only.
  - If native layout metadata is required, add a new optional symbol instead of changing existing symbol semantics in place.

### Lowering And Region Shape

- `src/main/java/backend/metal/lowering/MetalPartitionSupport.java`
  - Allows `RESHAPE`, `CONTIGUOUS`, `NOOP`, `PERMUTE`, `EXPAND_DIMS`, and `SQUEEZE`.
  - Phase 2 should keep capability checks conservative for dtype/rank/layout rather than widening unrelated ops.

- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java`
  - Produces DAG shape metadata and supports layout ops.
  - Research suggests planning should avoid broad lowerer rewrites unless required by tests.

- `src/main/java/graph/execution/RuntimeMemoryBinder.java`
  - CPU runtime aliasing pattern for layout ops.
  - Device-side view metadata can mirror the idea at the binding level without changing public tensors.

### Tests

- `src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java`
  - Primary fake-bridge unit suite for Metal buffer decisions.
  - Add/update tests for zero-offset view acceptance, non-zero-offset view fallback or transform, permute/reshape flow, required-mode unsupported layout rejection, and adjacent execution with layout-preserved binding.

- `src/test/java/backend/metal/buffer/MetalBufferAllocatorTest.java`
  - Add allocator/materializer behavior for any new transform or readback support.

- `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java`
  - Add optional native capability/version checks and explicit shim tests if native ABI changes.

- `src/test/java/backend/metal/MetalBufferTraceSmokeTest.java`
  - Add trace assertions for layout-aware decisions and CPU materialization boundaries if native shim is available.

## Recommended Implementation Strategy

### 1. Start With A Metal Layout Policy

Create a small Metal-specific policy class or nested helper that maps `AcceleratorBufferLayoutClass` to a decision:

- `DENSE_CONTIGUOUS`: use native buffer binding as today.
- `ZERO_OFFSET_VIEW`: represent as layout metadata when the native consumer can use the logical shape without copying; otherwise use a device contiguous transform.
- `NON_ZERO_OFFSET_VIEW`: reject initially unless the implementation explicitly allocates/copies a dense device buffer for the logical view.
- `PERMUTED_OR_STRIDED_VIEW`: support only through an explicit contiguous transform if implemented; otherwise reject with `OUTPUT_LAYOUT_UNSUPPORTED` or `INPUT_LAYOUT_UNSUPPORTED`.
- `BROADCAST_ZERO_STRIDE_VIEW`: reject until native broadcast view semantics are implemented.
- `UNSUPPORTED`: always reject.

The plan should name the exact policy and tests rather than leaving "layout-aware" implicit.

### 2. Prefer Java-Side Device Contiguous Transform Before Native Strided ABI

MPSGraph buffer execution currently receives shape-only `MPSGraphTensorData`. Passing arbitrary strides/offsets to native code would require a new ABI and native use of APIs that may not map cleanly to all logical views.

For Phase 2, a safer implementation path is:

1. Accept layout ops inside a Metal region when the lowered DAG can compute the logical value.
2. Allocate dense physical Metal output buffers for the actual graph outputs that native MPSGraph writes.
3. Attach a `MetalBufferBinding` with the runtime tensor's logical layout only when device materialization can later read it correctly.
4. Otherwise attach a dense binding to a contiguous runtime output or fall back explicitly.

If a plan chooses native strided/view ABI instead, it must include capability/version checks and native tests before enabling it.

### 3. Make Materialization Correct Before Broadening Acceptance

`MetalDeviceToCpuMaterializer.supports(...)` currently requires exact binding/target layout match. `MetalBufferAllocator.readToCpu(...)` then rejects non-contiguous or offset destinations. Accepting a non-contiguous binding without updating readback would create a false device-current value that cannot be materialized.

Planning must include one of these concrete outcomes:

- non-contiguous logical bindings are never attached as materializable graph outputs; or
- materializer reads dense device bytes into a temporary contiguous array and scatters into the target CPU layout; or
- the executor attaches a dense physical binding to a dense runtime tensor created by an explicit `CONTIGUOUS` transform.

For Phase 2, the most conservative useful target is to support dense physical buffer handoff across `LINEAR -> RESHAPE -> PERMUTE` when the native DAG already produces the final shape into a dense output, while keeping unsupported strided/broadcast readback rejected.

### 4. Keep Fallback Visible

Every newly accepted or rejected layout must appear in `AcceleratorBufferDecision` inputs/outputs and traces. Do not silently run CPU fallback after claiming `BUFFER_BINDING_AVAILABLE`.

Expected reason-code behavior:

- Native ABI missing: `NATIVE_BUFFER_ABI_UNAVAILABLE`.
- Unsupported input layout: `INPUT_LAYOUT_UNSUPPORTED`.
- Unsupported output layout: `OUTPUT_LAYOUT_UNSUPPORTED`.
- Native buffer execution failure: `NATIVE_BUFFER_EXECUTION_FAILED`.
- Required buffer mode converts unsupported layout to `UNAVAILABLE` and throws.

## Validation Architecture

### Automated Validation Targets

- Unit tests for `MetalAcceleratorBufferBinder` behavior through `PreparedMetalExecutableBufferBindingTest`.
- Bridge validation tests for any new native ABI/capability symbols in `MetalMpsFfmBridgeTest`.
- Materializer tests for graph output, CPU consumer, and public data access readback paths.
- Trace smoke tests for native buffer path when an explicit shim is configured.
- Compile-wide sanity check with `./gradlew classes`.

### Required Test Cases

1. Dense buffer path still works and adjacent Metal executables reuse intermediate buffers without CPU materialization.
2. Zero-offset layout view is classified as representable or transformed on device and no longer rejected by the old dense-only preflight when the chosen policy says it is legal.
3. `LINEAR -> RESHAPE -> PERMUTE` or a smaller deterministic proxy graph has a Metal buffer path without CPU materialization between layout steps.
4. Unsupported layout classes still reject with stable reason codes.
5. `AcceleratorBufferBindingMode.REQUIRE` still throws before tensor-array execution when layout support is unavailable.
6. Device-to-CPU materialization succeeds for graph output and CPU/public data access where support is claimed.
7. Forward-backward graph coverage proves gradient publication boundaries materialize correctly or fall back visibly.

### Recommended Commands

```bash
./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest
./gradlew test --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest
./gradlew test --tests backend.metal.MetalBufferTraceSmokeTest
./gradlew classes
./gradlew metalTest
```

Use targeted filters during development because default `./gradlew test` can include benchmark/debug tests.

## Risks And Mitigations

| Risk | Mitigation |
|------|------------|
| Claiming a non-contiguous device binding that cannot be materialized | Update materializer first or keep that layout rejected |
| Native ABI drift between Java and Objective-C shim | Add optional symbol/capability checks and tests before execution |
| CPU hot-path regression | Keep changes inside Metal/buffer/runtime state; do not alter public `Tensor` API or CPU kernels |
| Hidden CPU fallback | Assert `lastAcceleratorBufferDecision`, `MetalMpsBridgeExecutionPath`, and CPU materialization traces in tests |
| Over-expanding Metal op coverage | Restrict to layout-aware execution for currently supported Metal DAG ops |

## Planning Guidance

Suggested plan split:

1. Metal layout policy and Java-side buffer decision changes with fake-bridge tests.
2. Native/capability/materializer work for any required device transform or readback behavior.
3. End-to-end Metal flow tests, trace assertions, and documentation updates.

The plans should cover all Phase 2 requirements: METAL-01, METAL-02, METAL-03, and METAL-04.
