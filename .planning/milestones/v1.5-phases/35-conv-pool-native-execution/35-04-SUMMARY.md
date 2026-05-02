# Summary 35-04: Coverage Gates, Docs, And Profile Hygiene

**Status:** Completed
**Date:** 2026-05-02

## Implemented

- Promoted scoped Metal `CONV2D_GEMM` to the same native MPSGraph `CONV2D` DAG path as direct `CONV2D` when descriptor semantics preserve dense `FLOAT32` NCHW/OIHW forward conv.
- Registered `avg_pool2d_small` as a separate hot-path target so max-pool and avg-pool native evidence are not conflated.
- Hardened coverage truth and regression policies for `conv2d_resnet_3x3`, `max_pool2d_small`, and `avg_pool2d_small` to require native buffer binding, lowered primitive evidence, zero CPU fallback, zero tensor-array replay, and zero CPU materialization.
- Updated Metal backend, GPU lowering coverage, architecture, troubleshooting, and workload docs/tests to reflect supported scoped conv/pool forward execution and remaining capability gates.
- Kept local profile/tuning artifacts under `profiles/platform/...` unstaged.

## Verification

```bash
./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests PreparedExecutionBuildTest --tests Conv2dExecutionTest --tests Pool2dExecutionTest
./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests StandardWorkloadsTest --tests GpuTargetSemanticsContractTest --tests SourceTreeHygieneTest
./gradlew classes
./gradlew metalTest
git diff --check
```

All commands passed.

## Notes

- Supported Metal conv/pool scope is dense `FLOAT32` forward `CONV2D`, `CONV2D_GEMM`, `MAX_POOL2D`, and `AVG_POOL2D`.
- Grouped/depthwise conv, dilated conv, conv backward, pool backward, and `AVG_POOL2D countIncludePad=true` remain explicit capability-gated work.
