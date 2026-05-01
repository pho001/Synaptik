---
phase: 10-gpu-layout-transform-and-view-path
status: complete
researched: 2026-04-30
requirements: [GPUVIEW-01, GPUVIEW-02, GPUVIEW-03]
---

# Phase 10 Research: GPU Layout Transform And View Path

## Research Goal

Answer what needs to be known to plan Phase 10 well: how Synaptik can keep legal layout transforms and view-like values device-resident across Metal and CUDA without adding a public device tensor API, hiding fallback, or weakening CPU parity.

## Phase Scope

Phase 10 owns GPU-side view/layout execution for `reshape`, `permute`, `expand`, `contiguous`, alias outputs, and legal view-like graph values. It builds on Phase 9 layout ABI v2 metadata and must produce backend-neutral layout request/decision records plus Metal/CUDA backend implementations where capability contracts allow.

Out of scope:

- Broad NN/tensor operation lowering coverage, which belongs to Phase 11.
- Fused GPU compound regions, which belong to Phase 12.
- Full coverage benchmark gates, which belong to Phase 13.
- Public user-visible device tensor APIs.

## Current Code Findings

### Existing CPU layout semantics

- `tensor.ops.layout.TensorLayoutOps` creates public layout operations:
  - `contiguous` always creates an operation node that materializes dense row-major storage during execution.
  - `reshape` creates a view for contiguous inputs and an operation node for non-contiguous inputs.
  - `expand` creates a zero-stride broadcast view.
  - `permute` creates a stride-reordered view.
  - `expandDims` and `squeeze` are shape/stride metadata alias views.
- `backend.cpu.kernels.layout.LayoutExecutor` already has the CPU truth model:
  - `alias(...)` aliases runtime storage.
  - `reshapeLike(...)` aliases contiguous input or copies linearized values.
  - `contiguous(...)` remaps logical values into dense storage via `TensorRemap.apply(...)`.
- `ExecutionState.create(...)` and `RuntimeMemoryBinder` already alias CPU runtime storage for `NOOP`, `EXPAND`, `SELECT`, `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`, and contiguous-source `RESHAPE`.

### Existing device residency and materialization

- `ExecutionState` is the right place for runtime residency. It already tracks:
  - current CPU/device state by node id,
  - active and reserved `DeviceBufferBinding` instances,
  - per-run `DeviceToCpuMaterializer` registrations,
  - CPU materialization traces for graph output, CPU consumer, and gradient publication boundaries.
- `PreparedExecution.executeSteps(...)` currently calls `requireCpuReadableInputs(...)` before CPU steps. This forces device-to-CPU materialization for CPU-owned layout nodes before a metadata-only device view propagation path can run.
- Therefore Phase 10 needs a pre-CPU-step layout propagation hook that can claim eligible layout/view nodes from an existing device binding, skip CPU kernel execution, and attach a target layout binding to the view node.

### Existing Metal behavior

- `MetalLayoutPolicy` already accepts `ZERO_OFFSET_VIEW`, `NON_ZERO_OFFSET_VIEW`, and `PERMUTED_OR_STRIDED_VIEW` for existing device inputs and outputs as `DENSE_PHYSICAL_LOGICAL_VIEW`.
- `MetalBufferAllocator.createOutputBinding(...)` can allocate dense physical bytes for logical-view outputs and keep logical layout metadata.
- `MetalBufferAllocator.readToCpu(...)` can read dense logical bytes and scatter them into a non-contiguous destination tensor for CPU materialization.
- `MetalDeviceToCpuMaterializer` supports non-broadcast logical-view layouts except `BROADCAST_ZERO_STRIDE_VIEW` and `UNSUPPORTED`.
- Existing tests in `MetalLayoutAwareDeviceFlowTest` prove a `linear -> reshape -> permute` flow can stay device-owned until graph-output materialization for Metal when the native buffer bridge is available.

### Existing CUDA behavior

- `CudaAcceleratorBufferBinder` currently accepts only dense `FLOAT32` metadata for native buffer execution.
- Phase 9 added layout ABI v2 capability/version fields and `NATIVE_LAYOUT_*` reason codes. CUDA non-dense metadata now rejects visibly as layout ABI v2 unavailable/mismatch/unsupported instead of generic layout unsupported.
- `CudaBufferAllocator.createInputBinding(...)`, `createOutputBinding(...)`, and `readToCpu(...)` validate dense contiguous `FLOAT32` layouts only.
- Adjacent dense CUDA regions already reuse `CudaBufferBinding` without CPU materialization, as proven by `PreparedCudaExecutableBufferPolicyTest.adjacentCudaRegionsReuseDeviceBufferBinding`.
- CUDA therefore needs explicit layout/view propagation and a backend-side dense materialization path before non-dense/view outputs can safely feed later CUDA work without CPU round trips.

### Native layout ABI v2 from Phase 9

- Shared metadata exists in `backend.accelerator.buffer.AcceleratorLayoutAbiV2Descriptor`.
- Metal and CUDA bridges expose optional layout ABI v2 support/version fields.
- Missing v2 symbols preserve dense v1 execution and only reject metadata that needs v2 semantics.
- Stable fallback codes include `NATIVE_LAYOUT_ABI_UNAVAILABLE`, `NATIVE_LAYOUT_ABI_VERSION_MISMATCH`, `NATIVE_LAYOUT_METADATA_UNSUPPORTED`, `NATIVE_LAYOUT_RANK_UNSUPPORTED`, `NATIVE_LAYOUT_DTYPE_UNSUPPORTED`, and `NATIVE_LAYOUT_PHYSICAL_SPAN_OVERFLOW`.

## Recommended Architecture

### 1. Backend-neutral layout/view decision model

Add shared records under `backend.accelerator.buffer` rather than a public Tensor API:

- `AcceleratorLayoutTransformKind`
- `AcceleratorLayoutTransformRequest`
- `AcceleratorLayoutTransformDecision`
- `AcceleratorLayoutTransformPlanner`

The planner should classify metadata-only view propagation versus dense materialization:

- Metadata-only view: `NOOP`, `SELECT`, `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`, `EXPAND`, and contiguous-source `RESHAPE`.
- Dense materialization: `CONTIGUOUS` and non-contiguous-source `RESHAPE`.
- Unsupported: negative strides, layout rank/dtype/backend mismatch, source binding unavailable, broadcast views when backend cannot consume/materialize them.

Use stable accelerator reason codes for traceability. New reason codes should distinguish:

- source device binding missing,
- metadata-only view binding available,
- GPU dense layout materialization available,
- GPU layout transform unsupported,
- backend mismatch.

### 2. Device view binding propagation before CPU materialization

Add a pre-CPU-step hook in `PreparedExecution.executeSteps(...)`:

1. If the next step is a layout/view operation and its source node has a compatible device binding, call a layout propagation helper.
2. If propagation succeeds, attach a target `DeviceBufferBinding` with the target node id and target layout, mark the target `DEVICE_OWNED`, and skip CPU kernel execution for that node.
3. If propagation fails, continue through the existing CPU path. In `REQUIRE`, fail with the exact reason before hidden CPU materialization.

Backend-specific binding factories can reuse the original native handle without registering an extra resource:

- `MetalBufferBinding.viewOf(int nodeId, AcceleratorBufferLayout layout, MetalBufferBinding source, MetalBufferAccess access)`
- `CudaBufferBinding.viewOf(int nodeId, AcceleratorBufferLayout layout, CudaBufferBinding source, CudaBufferAccess access)`

### 3. Backend-side dense materialization for `contiguous`

Metadata-only view propagation is not enough for `contiguous()`. A dense materialization node must produce a new dense device buffer from a source logical layout without reading to Java arrays.

Recommended staged support:

- Add optional backend layout-transform methods to bridge interfaces.
- Metal implementation can start with an optional native F32 buffer transform symbol that reads source logical metadata and writes dense logical output into the destination buffer.
- CUDA implementation can add a simple F32 strided-copy kernel behind optional native shim symbols, capability-gated like other CUDA native checks.
- Java fake bridge tests must prove behavior without native Metal/CUDA hardware.

### 4. Keep consumers conservative

Until Phase 11 broadens operation lowering, do not claim every GPU op can consume arbitrary non-dense CUDA layouts. Use one of two safe paths:

- Metadata-only view stays device-owned and can materialize to CPU correctly at a true boundary.
- `contiguous()` converts the view to dense device storage before a downstream GPU compute region consumes it.

Unsupported direct non-dense CUDA compute should remain visible through layout ABI v2 reason codes.

## Validation Architecture

Use focused JUnit/Gradle gates:

- Shared decision tests:
  - `./gradlew test --tests backend.accelerator.buffer.AcceleratorLayoutTransformPlannerTest`
- Runtime propagation tests:
  - `./gradlew test --tests graph.execution.DeviceLayoutViewPropagationTest`
- Metal/CUDA backend tests:
  - `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest`
  - `./gradlew test --tests backend.metal.MetalLayoutAwareDeviceFlowTest --tests backend.cuda.exec.CudaLayoutTransformDeviceFlowTest`
- Native/capability gates:
  - `./gradlew classes`
  - `./gradlew metalTest`
  - `./gradlew buildCudaGraphShim cudaTest`

Acceptance must include:

- at least one layout-heavy forward flow that avoids intermediate CPU materialization,
- one forward/backward or gradient-publication boundary with CPU parity,
- visible fallback/required-mode failures for unsupported layout paths,
- no source hygiene regression for local profile or native build artifacts.

## Risks And Mitigations

| Risk | Mitigation |
|---|---|
| Device view bindings reuse native handles incorrectly and double-destroy resources | Reuse backend-specific handle objects but do not register extra `ExecutionResource` for alias bindings; source allocation remains the owner. |
| CPU layout nodes force materialization before propagation | Add the propagation hook before `requireCpuReadableInputs(...)` in `PreparedExecution.executeSteps(...)`. |
| CUDA accepts non-dense metadata but native compute ignores strides | Keep non-dense CUDA compute rejected unless the node is a metadata-only view or a backend-side dense materialization transform has produced dense output. |
| Broadcast zero-stride views corrupt writes | Treat expand/broadcast as read-only view metadata unless a dense `contiguous()` transform materializes it. |
| Phase grows into operation lowering | Keep arbitrary non-dense GPU consumers out of scope; Phase 11 owns lowering coverage. |
| Native Metal/CUDA availability differs by developer machine | Portable fake bridge tests are mandatory; native tests remain capability-gated. |

## Planning Implications

Recommended waves:

1. Shared layout transform decision records and portable planner tests.
2. Runtime device view propagation plus backend-specific alias binding factories for Metal and CUDA.
3. Backend-side `contiguous` dense materialization path with optional native/fake bridge coverage.
4. End-to-end flow, gradient publication parity, docs, traces, and closure gates.

## Research Complete

Phase 10 can be planned as a four-wave implementation. The key design point is to separate metadata-only device view propagation from dense GPU materialization, while preserving explicit fallback and CPU parity.
