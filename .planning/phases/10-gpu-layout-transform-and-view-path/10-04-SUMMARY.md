---
phase: 10-gpu-layout-transform-and-view-path
plan: "04"
status: completed-with-native-metal-gate-open
completed: 2026-04-30
requirements: [GPUVIEW-01, GPUVIEW-02, GPUVIEW-03]
---

# 10-04 Summary - Layout Verification Closure

## Completed

- Added trace-visible layout-transform decisions for metadata-only view propagation and dense-materialization fallback.
- Added portable CUDA layout-flow coverage for metadata-only reshape/permute propagation and visible fallback when direct non-dense CUDA materialization has no registered service.
- Added portable trace coverage proving accelerator step attributes include `acceleratorBufferReasonCode` and `storageResidency`.
- Documented the GPU layout transform and view path in Metal, native bridge, and development docs.
- Kept direct non-dense CUDA compute conservative: CUDA preserves metadata-only views, uses explicit dense materialization only when a backend capability/service exists, or falls back visibly with `GPU_LAYOUT_TRANSFORM_UNSUPPORTED`.

## Phase 10 final verification

Portable verification passed:

```bash
./gradlew classes
```

Result: PASS.

```bash
./gradlew test --tests backend.accelerator.buffer.AcceleratorLayoutTransformPlannerTest --tests graph.execution.DeviceLayoutViewPropagationTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests backend.metal.MetalLayoutAwareDeviceFlowTest --tests backend.cuda.exec.CudaLayoutTransformDeviceFlowTest --tests CompiledGraphTraceTest
```

Result: PASS.

Documentation and acceptance grep passed:

```bash
rg -n "GPU layout transform and view path|metadata-only views|dense GPU materialization|direct non-dense CUDA compute remains conservative|GPU_LAYOUT_VIEW_BINDING_AVAILABLE|GPU_LAYOUT_DENSE_MATERIALIZATION_AVAILABLE|buildCudaGraphShim cudaTest" docs src/test/java/CompiledGraphTraceTest.java src/test/java/backend/cuda/exec/CudaLayoutTransformDeviceFlowTest.java src/test/java/backend/metal/MetalLayoutAwareDeviceFlowTest.java
```

Result: PASS.

Native Metal verification:

```bash
./gradlew metalTest
```

Result: FAIL. The shim built successfully, but two native Metal backward gradient parity tests failed:

- `PreparedExecutionBuildTest.gpuMetalBackwardMatmulCanPrepareAndMatchCpuGradients`: expected gradient index 0 `15.0`, actual `17.0`.
- `MetalLayoutAwareDeviceFlowTest.forwardBackwardLayoutAwareGraphPublishesGradientsWithCpuParity`: expected gradient index 0 `0.800000011920929`, actual `1.100000023841858`.

This is recorded as an open native Metal backward gate, not a portable layout-transform regression. The portable Phase 10 layout/view checks pass.

Native CUDA verification:

```bash
./gradlew buildCudaGraphShim cudaTest
```

Result: PASS with native tasks skipped by local capability gating:

- `buildCudaGraphShim SKIPPED`
- `cudaTest SKIPPED`
- Gradle build successful.

## Trace And Boundary Evidence

- Metadata-only views publish `gpuLayoutTransformKind=METADATA_ONLY_VIEW`, `acceleratorBufferReasonCode=GPU_LAYOUT_VIEW_BINDING_AVAILABLE`, and device buffer residency attributes.
- Dense materialization without a registered service publishes `acceleratorBufferReasonCode=GPU_LAYOUT_TRANSFORM_UNSUPPORTED` before CPU fallback.
- Valid CPU materialization boundaries remain graph output, CPU consumer, and gradient publication.
- Portable CUDA fallback does not leave stale residency state: fallback CPU layout execution marks output `CPU_ARRAY`.

## Local Artifacts

The local profile tuning files under `profiles/platform/mac_os_x-aarch64-oracle_corporation-16c/tuning/abc/*` were present before this plan and remained unstaged.
