# Phase 8: CUDA Observability And Documentation Closure - Research

## RESEARCH COMPLETE

## Scope

Phase 8 closes the remaining v1.1 milestone gaps:

- `CUDA-06`: stable CUDA fallback and required-mode reason codes.
- `CUDADOC-01`: CUDA trace and benchmark report parity with the Metal accelerator evidence contract.
- `CUDADOC-02`: CUDA build/probe/fallback/troubleshooting documentation.
- `CUDADOC-03`: source hygiene and verification gates for local CUDA/native/profile artifacts.

## Existing Facts

- `PreparedExecution.buildStepMetadata(...)` already emits backend-neutral `acceleratorBuffer*` attributes from `PreparedAcceleratorExecutable.lastAcceleratorBufferDecision()`.
- `PreparedCudaExecutable` publishes `AcceleratorBufferDecision` values, but unlike `PreparedMetalExecutable` it has no per-execution stats record for path, bytes, copy timing, native execution timing, or fallback reason.
- `AcceleratorTraceSummary` aggregates backend-neutral path counts, reason codes, fallback reasons, and prepared-input usage, but byte/copy counters are currently read from `metal*` attributes only.
- `BenchmarkSessionTest` already has synthetic report-contract tests for accelerator summaries and Metal materialization details. These are the closest portable way to prove CUDA report parity without CUDA hardware.
- `PreparedCudaExecutableBufferPolicyTest` already distinguishes CUDA native buffer execution, CPU fallback, required failure, materialization, adjacent handoff, and incompatible bindings.
- `SourceTreeHygieneTest` already checks `.planning/tmp/`, root `.class` artifacts, CUDA native build outputs under `build/native/cuda`, and profile tuning artifact boundaries.
- `docs/configuration.md` still says CUDA trace/report parity is a later observability task; Phase 8 must remove that gap once implemented.
- `docs/testing.md` still contains stale "Needs verification" language saying CUDA shim build instructions are not present; Phase 8 should update this to the checked-in v1.1 state.

## Recommended Implementation Shape

### CUDA Execution Stats

Add a CUDA stats record similar to `MetalMpsBridgeExecutionStats`, but owned under `backend.cuda.bridge`, for example `CudaBridgeExecutionStats`.

Required fields:

- `usedCpuFallback`
- `fallbackReason`
- `AcceleratorBufferExecutionPath executionPath`
- `externalInputCount`
- `outputCount`
- `inputBytes`
- `outputBytes`
- `javaToNativeCopyNs`
- `nativeExecuteNs`
- `nativeDeviceCopyNs`
- `nativeToJavaCopyNs`
- `totalNs`

The first implementation can measure Java-observed boundary time with `System.nanoTime()`. Native device copy timing can be `0L` for paths where the current shim does not expose sub-timers, but the field must exist in trace/report output.

### Trace Attributes

Extend `PreparedExecution.buildStepMetadata(...)` for `PreparedCudaExecutable` with:

- CUDA-specific keys: `cudaBridgeAvailable`, `cudaBridgeContextAvailable`, `cudaBridgeExecutableAvailable`, `cudaSupportsBufferBindings`, `cudaExecutionPath`, `cudaFallbackReason`, `cudaExternalInputCount`, `cudaOutputCount`, `cudaInputBytes`, `cudaOutputBytes`, `cudaJavaToNativeCopyNs`, `cudaNativeExecuteNs`, `cudaNativeDeviceCopyNs`, `cudaNativeToJavaCopyNs`, `cudaBridgeTotalNs`.
- Backend-neutral aggregate keys: `acceleratorInputBytes`, `acceleratorOutputBytes`, `acceleratorJavaToNativeCopyNs`, `acceleratorNativeToJavaCopyNs`, `acceleratorNativeDeviceCopyNs`.

Keep existing Metal keys stable. Update `AcceleratorTraceSummary` to prefer the backend-neutral aggregate keys and fall back to existing `metal*` keys for compatibility.

### Report Tests

Use synthetic `ExecutionStepTrace` values in `BenchmarkSessionTest` to prove text and JSON report output for `GPU_CUDA`:

- `backend=GPU_CUDA`
- `bufferBindingSteps=1`
- `preparedInputSteps=1`
- `reasonCodes=[BUFFER_BINDING_AVAILABLE]`
- `fallbackReasons=[using native CUDA buffer bindings]`
- `javaToNativeMs=...`
- `nativeDeviceCopyMs=...`
- JSON `"GPU_CUDA"` accelerator summary with bytes/copy fields

### Reason-Code Matrix

Extend CUDA portable tests to cover exact reason-code strings for:

- `BRIDGE_UNAVAILABLE`
- `NATIVE_BUFFER_ABI_UNAVAILABLE`
- `INPUT_DTYPE_UNSUPPORTED`
- `INPUT_LAYOUT_UNSUPPORTED`
- `OUTPUT_DTYPE_UNSUPPORTED`
- `OUTPUT_LAYOUT_UNSUPPORTED`
- `INPUT_BINDING_UNAVAILABLE`
- `INPUT_NOT_CPU_CURRENT`
- `NATIVE_BUFFER_EXECUTION_FAILED`
- `REQUIRED_BUFFER_EXECUTION_UNAVAILABLE`

The required-mode tests should prove tensor-array execution is not called after required buffer failure.

### Documentation And Hygiene

Update:

- `docs/architecture.md`
- `docs/compute-flow.md`
- `docs/development.md`
- `docs/configuration.md`
- `docs/testing.md`
- `docs/troubleshooting.md`

Docs must state CUDA dense `FLOAT32` native buffer support remains narrow, fallback remains visible, CPU is the correctness oracle, and optional native CUDA checks pass or skip by capability.

Keep local artifacts out of project state:

- `build/native/cuda/**`
- `.planning/tmp/**`
- machine-local `profiles/platform/.../tuning/abc/*` changes
- generated native scratch files

## Validation Architecture

Portable gate:

```bash
./gradlew classes
./gradlew test --tests BenchmarkSessionTest --tests backend.cuda.bridge.CudaFfmBridgeTest --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest --tests backend.cuda.buffer.CudaBufferAllocatorTest --tests backend.cuda.buffer.CudaDeviceToCpuMaterializerTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests SourceTreeHygieneTest
```

Optional native gate:

```bash
./gradlew buildCudaGraphShim cudaTest
```

Doc/hygiene grep gate:

```bash
rg -n "GPU_CUDA|cudaExecutionPath|cudaFallbackReason|acceleratorInputBytes|CUDA trace and benchmark reports|Native CUDA tests skip" src/main/java src/test/java docs
```

## Planning Recommendation

Use four focused plans:

1. CUDA trace/report parity.
2. CUDA reason-code matrix and required-mode fallback policy.
3. CUDA docs and troubleshooting closure.
4. Hygiene and final v1.1 verification closure.
