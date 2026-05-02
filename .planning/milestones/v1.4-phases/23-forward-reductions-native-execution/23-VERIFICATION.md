---
status: passed
phase: 23-forward-reductions-native-execution
requirements:
  GPURED-01: passed
  GPURED-02: passed
  GPURED-03: passed
created: 2026-05-02
---

# Verification: Phase 23 Forward Reductions Native Execution

## Verdict

Phase 23 passed verification.

Legal dense FLOAT32 `SUM`, `MEAN`, `REDUCE_MIN`, and `REDUCE_MAX` now lower to shared accelerator DAG primitives and execute through backend-owned Metal/CUDA paths for supported cases.

## Requirement Traceability

| Requirement | Status | Evidence |
|-------------|--------|----------|
| GPURED-01 | Passed | `AcceleratorSubgraphLowererTest` and `GpuLoweringCoverageMatrixTest` verify reduction DAG node types, axis metadata, keep-dims metadata, and supported matrix rows. |
| GPURED-02 | Passed | Metal MPSGraph reductions and CUDA dense FLOAT32 reduction kernels execute supported reduction primitives without region-internal CPU materialization. |
| GPURED-03 | Passed | Prepared execution, trace, and hot-path tests verify reduction outputs stay device-owned until true output/consumer boundaries. |

## Automated Checks

Passed:

```bash
./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuCoverageSummaryTest
./gradlew test --tests PreparedExecutionBuildTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest
./gradlew test --tests CompiledGraphTraceTest --tests PreparedExecutionBuildTest --tests GpuHotPathCoverageTargetsTest --tests StandardWorkloadsTest
./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuCoverageSummaryTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest
./gradlew metalTest
```

## Residual Risk

CUDA native execution remains environment-dependent in this local lane; portable CUDA tests verify lowering and stable behavior, while native CUDA should be run on a CUDA-equipped system.
