---
status: clean
phase: 25-forward-sdpa-semantic-enablement
depth: standard
files_reviewed: 19
findings:
  critical: 0
  warning: 0
  info: 0
  total: 0
created: 2026-05-01
---

# Code Review: Phase 25 Forward SDPA Semantic Enablement

## Scope

Reviewed the Phase 25 implementation surface:

- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphSignature.java`
- `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java`
- `src/main/java/backend/accelerator/lowering/GpuTargetSemanticsContract.java`
- `src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java`
- `src/main/java/backend/metal/lowering/MetalPartitionSupport.java`
- `src/main/java/tuning/benchmark/report/GpuHotPathCoverageTargets.java`
- `src/main/java/tuning/benchmark/report/GpuTargetCoverageTruth.java`
- `src/main/native/apple/synaptik_apple_mps_stub.m`
- `src/test/java/AttentionExecutionTest.java`
- `src/test/java/GpuCoverageRegressionGateTest.java`
- `src/test/java/GpuCoverageSummaryTest.java`
- `src/test/java/GpuHotPathCoverageTargetsTest.java`
- `src/test/java/GpuTargetSemanticsContractTest.java`
- `src/test/java/PreparedExecutionBuildTest.java`
- `src/test/java/backend/accelerator/lowering/GpuLoweringCoverageMatrixTest.java`
- `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java`
- `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java`
- `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java`
- Phase 25 docs and validation artifacts.

## Findings

No critical, warning, or info findings.

## Review Notes

- Metal SDPA support is narrow and evidence-backed: direct unmasked `FLOAT32` rank-3/rank-4 SDPA lowers to a primitive MPSGraph DAG and is parity-tested for explicit scale and default-scale rank-4 cases.
- The executable cache fix is necessary and appropriate: scalar DAG bits are now part of `AcceleratorSubgraphSignature`, preventing reuse of a native executable compiled with a different SDPA scale or scalar parameter.
- CUDA does not claim unsupported native execution. Legal unmasked direct SDPA reports `CAPABILITY_MISSING`; masked, dtype, layout, and shape blockers have stable earlier reason codes.
- Coverage expectations now match backend reality: Metal transformer SDPA requires native buffer evidence and zero hidden fallback; CUDA transformer SDPA expects visible capability fallback evidence.
- Local profile artifacts under `profiles/platform/...` remain dirty but are outside Phase 25 evidence and were not reviewed as source changes.

## Verification Reviewed

Passed:

```bash
./gradlew test --tests GpuTargetSemanticsContractTest --tests AttentionExecutionTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageRegressionGateTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests StandardWorkloadsTest
./gradlew metalTest
git diff --check
```

Not run:

```bash
./gradlew cudaTest
```

`nvcc` is unavailable locally, so native CUDA remains hardware/toolchain-gated.
