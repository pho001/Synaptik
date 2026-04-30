---
phase: 10-gpu-layout-transform-and-view-path
status: complete
created: 2026-04-30
---

# Phase 10 Pattern Map

## Closest Existing Analogs

| New/Modified Area | Closest Analog | Pattern To Reuse |
|---|---|---|
| Shared layout transform decision records | `backend.accelerator.buffer.AcceleratorBufferDecision`, `AcceleratorBufferLayout`, `AcceleratorLayoutAbiV2Descriptor` | Immutable records, defensive layout metadata, stable reason codes. |
| Layout transform planner tests | `AcceleratorBufferLayoutClassifierTest`, `AcceleratorLayoutAbiV2DescriptorTest` | Pure portable unit tests with fake `DeviceBufferBinding` fixtures. |
| Pre-CPU-step view propagation | `ExecutionState.create(...)`, `RuntimeMemoryBinder.aliasesInput0AtRuntime(...)`, `PreparedExecution.executeSteps(...)` | Runtime aliasing belongs in execution state and prepared execution, not public Tensor APIs. |
| Metal alias/view binding | `MetalBufferBinding`, `MetalLayoutPolicy`, `MetalDeviceToCpuMaterializer` | Preserve backend-owned handle while changing node id and logical layout metadata. |
| CUDA alias/view binding | `CudaBufferBinding`, `CudaAcceleratorBufferBinder`, `CudaDeviceToCpuMaterializer` | Keep CUDA dense compute conservative; only accept non-dense metadata through explicit layout/view path. |
| Backend contiguous materialization | `MetalBufferAllocator.readToCpu(...)`, `CudaBufferAllocator.readToCpu(...)`, native stub optional symbol patterns | Optional native symbols, capability-gated tests, fake bridge coverage. |
| E2E layout flow tests | `MetalLayoutAwareDeviceFlowTest`, `PreparedCudaExecutableBufferPolicyTest` | Real compile/prepare/execute flow with trace assertions and CPU parity. |

## Concrete Code Patterns

### Stable decision record

`AcceleratorBufferDecision` records backend, binding mode, execution path, allow/required state, reason code, reason string, and per-input/output decisions. New layout transform decisions should follow this shape and not throw for ordinary unsupported paths.

### Runtime aliasing boundary

`ExecutionState.create(...)` aliases CPU runtime tensors for view operations only when the source CPU storage is current. Phase 10 should add an equivalent device-binding propagation path but keep it runtime-only.

### Prepared execution hook point

`PreparedExecution.executeSteps(...)` currently performs:

1. `requireCpuReadableInputs(step, context)`
2. `ComputeEngine.compute(...)`
3. `markResidencyAfterStep(...)`

The layout propagation hook must run before step 1 for eligible layout/view nodes; otherwise CPU materialization has already happened.

### Backend binding aliasing

`MetalBufferBinding` and `CudaBufferBinding` are records that carry node id, layout, native handle, and access. A view binding can reuse the same handle with a new node id and logical layout. Resource ownership must remain with the originally allocated handle/resource.

### Native optional symbol pattern

Phase 9 added optional layout ABI v2 symbols in Metal/CUDA bridges without changing dense bridge availability. Phase 10 native contiguous transform symbols should follow the same pattern: missing symbols disable only GPU layout transform capability, not dense buffer execution.

## Files Likely To Change

- `src/main/java/backend/accelerator/buffer/AcceleratorBufferReasonCode.java`
- `src/main/java/backend/accelerator/buffer/AcceleratorLayoutTransformKind.java`
- `src/main/java/backend/accelerator/buffer/AcceleratorLayoutTransformRequest.java`
- `src/main/java/backend/accelerator/buffer/AcceleratorLayoutTransformDecision.java`
- `src/main/java/backend/accelerator/buffer/AcceleratorLayoutTransformPlanner.java`
- `src/main/java/graph/execution/DeviceLayoutViewPropagator.java`
- `src/main/java/graph/execution/PreparedExecution.java`
- `src/main/java/backend/metal/buffer/MetalBufferBinding.java`
- `src/main/java/backend/cuda/buffer/CudaBufferBinding.java`
- `src/main/java/backend/metal/buffer/MetalAcceleratorBufferBinder.java`
- `src/main/java/backend/cuda/buffer/CudaAcceleratorBufferBinder.java`
- `src/main/java/backend/metal/bridge/MetalMpsGraphBridge.java`
- `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java`
- `src/main/java/backend/cuda/bridge/CudaGraphBridge.java`
- `src/main/java/backend/cuda/bridge/CudaFfmBridge.java`
- `src/main/native/apple/synaptik_apple_mps_stub.m`
- `src/main/native/cuda/synaptik_cuda_graph_stub.cu`
- `src/test/java/backend/accelerator/buffer/AcceleratorLayoutTransformPlannerTest.java`
- `src/test/java/graph/execution/DeviceLayoutViewPropagationTest.java`
- `src/test/java/backend/cuda/exec/CudaLayoutTransformDeviceFlowTest.java`
- `docs/native-bridges-and-blas.md`
- `docs/metal-backend.md`

## Constraints For Executors

- Do not add public `Tensor` device residency APIs.
- Do not treat non-dense CUDA compute as supported unless the node is a metadata-only view or a dense GPU materialization has produced dense output.
- Do not register duplicate execution resources for alias/view bindings that reuse an existing native handle.
- Keep AUTO fallback visible and REQUIRE failure early.
- Do not commit local profile tuning output or `.planning/tmp/`.
