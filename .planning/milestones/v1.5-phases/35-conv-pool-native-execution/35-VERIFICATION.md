# Phase 35 Verification: Conv Pool Native Execution

**Status:** Passed
**Date:** 2026-05-02

## Requirement Mapping

| Requirement | Verdict | Evidence |
|---|---|---|
| `METALCONVPOOL-01` | Covered | `CONV2D` and `CONV2D_GEMM` lower to `AcceleratorDagNodeType.CONV2D`; Metal planner admits legal dense `FLOAT32` NCHW/OIHW forward conv and rejects dtype, layout, rank, grouped, dilated, and metadata-invalid variants with stable reasons. |
| `METALCONVPOOL-02` | Covered | `MAX_POOL2D` and `AVG_POOL2D` lower to native DAG nodes and MPSGraph pooling; planner rejects unsupported dtype/layout/rank and `AVG_POOL2D countIncludePad=true`. |
| `METALCONVPOOL-03` | Covered | Hot-path targets `conv2d_resnet_3x3`, `max_pool2d_small`, and `avg_pool2d_small` have hard native policies requiring buffer binding, lowered primitive evidence, no CPU fallback, no tensor-array replay, and no CPU materialization. |

## Code Evidence

- `src/main/java/backend/metal/lowering/MetalConvPoolSemantics.java`
- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java`
- `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java`
- `src/main/java/tuning/benchmark/report/GpuHotPathCoverageTargets.java`
- `src/main/java/tuning/benchmark/report/GpuTargetCoverageTruth.java`
- `src/main/java/tuning/workload/StandardWorkloads.java`
- `src/main/native/apple/synaptik_apple_mps_stub.m`

## Test Evidence

```bash
./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests PreparedExecutionBuildTest --tests Conv2dExecutionTest --tests Pool2dExecutionTest
./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests StandardWorkloadsTest --tests GpuTargetSemanticsContractTest --tests SourceTreeHygieneTest
./gradlew classes
./gradlew metalTest
git diff --check
```

All commands passed.

## Residual Scope

- CUDA conv/pool native execution remains `CAPABILITY_MISSING`.
- Conv/pool backward paths remain `CAPABILITY_MISSING`.
- Grouped/depthwise and dilated Conv2D remain unsupported.
- `AVG_POOL2D countIncludePad=true` remains unsupported until native divisor semantics are implemented.
- Local tuning outputs under `profiles/platform/...` are not canonical evidence and remain unstaged.
