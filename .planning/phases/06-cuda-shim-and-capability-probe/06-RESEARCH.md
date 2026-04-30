# Phase 6: CUDA Shim And Capability Probe - Research

## RESEARCH COMPLETE

## Scope

Phase 6 is an infrastructure bring-up phase for CUDA. The repo already has CUDA Java scaffolding, FFM lookup, region lowering, and prepared executable fallback behavior, but no checked-in CUDA native shim and no CUDA-side consumer of the shared accelerator buffer ABI. The phase should add a minimal native shim/build/probe workflow, explicit Java capability reporting, and a conservative buffer-layout policy seam without claiming Phase 7 native device-buffer execution.

## Existing Facts

- `CudaFfmBridge` already resolves `-Dsynaptik.cuda.graph.lib`, then `SYNAPTIK_CUDA_GRAPH_LIB`, then `synaptik_cuda_graph`.
- `CudaFfmBridge` currently requires `synaptik_cuda_graph_available` and `synaptik_cuda_graph_unavailable_reason`; context/compile/execute/destroy symbols are optional handles after discovery.
- `CudaGraphBridge.supportsBufferBindings()` defaults to `false`.
- `PreparedCudaExecutable` already publishes `AcceleratorBufferDecision` and fails REQUIRED mode before tensor-list execution.
- `CudaBridgeExecutable` currently carries input/output node ids but not dtype metadata; Metal's executable carries richer metadata for buffer decisions.
- Metal's bridge pattern is the closest analog: optional buffer ABI symbols gate `supportsBufferBindings()`, `PreparedMetalExecutable` creates an `AcceleratorBufferRequest`, and `MetalAcceleratorBufferBinder` publishes reason-coded decisions.

## Recommended Implementation Shape

### Native Shim

Add `src/main/native/cuda/synaptik_cuda_graph_stub.cu` as a minimal CUDA C ABI shim. It should export the existing graph symbols that `CudaFfmBridge` expects:

- `synaptik_cuda_graph_available`
- `synaptik_cuda_graph_unavailable_reason`
- `synaptik_cuda_graph_create_context`
- `synaptik_cuda_graph_compile_partition_f32`
- `synaptik_cuda_graph_execute_partition_f32`
- `synaptik_cuda_graph_destroy_context`
- `synaptik_cuda_graph_destroy_executable`

The shim can be intentionally narrow: check CUDA runtime/device availability, return stable unavailable reasons, allocate tiny context/executable structs, and avoid broad operation coverage. When CUDA runtime is unavailable it should report unavailable instead of crashing.

Add `scripts/build-cuda-graph-shim.sh` to compile the shim to `build/native/cuda/libsynaptik_cuda_graph.{dylib|so}` using `nvcc` when present. The script should fail with a clear message if `nvcc` is missing. Gradle tasks should wrap this as optional work so portable `classes` and `test` do not require CUDA.

### Java Capability Model

Add a small bridge capability record and reason enum under `backend.cuda.bridge`, for example:

- `CudaBridgeCapabilityCode`
- `CudaBridgeCapabilities`

Minimum useful fields:

- native library discovered
- CUDA runtime available according to shim
- native context available
- graph execution ABI available
- buffer execution supported
- stable capability code
- human-readable reason

Expose it through `CudaGraphBridge.capabilities()` with a default implementation for unavailable/fake bridges. `CudaFfmBridge` should populate it from symbol lookup and shim availability results.

### Shared Buffer ABI Seam

Add a CUDA buffer policy class under `backend.cuda.buffer`, for example `CudaAcceleratorBufferBinder`. It should consume `AcceleratorBufferRequest`, `AcceleratorBufferLayout`, `AcceleratorBufferDecision`, and `AcceleratorBufferReasonCode`.

Phase 6 should validate dense `FLOAT32` input/output metadata and reject unsupported dtype/layout combinations through stable reason codes. It should not allocate real CUDA device buffers or materialize CPU results; that belongs to Phase 7. Prepared execution can evaluate/publish the policy decision, but REQUIRED mode must still fail before tensor-list fallback when actual CUDA buffer execution is unavailable.

`CudaBridgeExecutable` should carry enough dtype metadata for prepared execution to build a complete `AcceleratorBufferRequest`. Since CUDA's current native graph path is F32-oriented, output dtype can be recorded as `FLOAT32` for current DAG outputs unless the lowering model grows explicit output dtype later.

## Risks And Mitigations

| Risk | Mitigation |
|------|------------|
| Missing CUDA toolkit breaks portable builds | Keep CUDA build/test tasks opt-in; default `classes` and `test` stay Java-only. |
| Native ABI mismatch hides as generic fallback | Capability model and tests must distinguish missing required symbol, unavailable shim, and buffer ABI unavailable. |
| CUDA buffer support is overclaimed | `supportsBufferBindings()` remains false until all required Java/native pieces are present; REQUIRED mode still fails before tensor-list execution if native buffer execution is unavailable. |
| Common ABI becomes CUDA-specific | CUDA handles, allocators, resources, and access enums stay under `backend.cuda.*`; shared records remain backend-neutral. |
| Local build/profile artifacts are staged | Build outputs go under `build/native/cuda/`; plans avoid profile updates and include hygiene checks. |

## Validation Architecture

Portable gate:

- `./gradlew classes`
- `./gradlew test --tests backend.cuda.bridge.CudaFfmBridgeTest`
- `./gradlew test --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest`
- `./gradlew test --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest`
- `./gradlew test --tests SourceTreeHygieneTest`

Optional native gate:

- `./gradlew buildCudaGraphShim`
- `./gradlew cudaTest`

Optional native tests must use assumptions or Gradle `onlyIf` guards so machines without `nvcc` or CUDA hardware do not fail portable verification.

## Planning Recommendation

Use three waves:

1. Native shim/build/probe docs and Java capability model can be planned independently.
2. Shared buffer ABI policy depends on capability reporting and prepared executable metadata.
3. Final verification and docs are embedded in the relevant plans rather than a separate closure plan, because Phase 6 is narrow and should not create benchmark/report evidence yet.
