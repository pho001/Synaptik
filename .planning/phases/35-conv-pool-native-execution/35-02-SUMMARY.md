# Summary 35-02: Conv2D Forward Native/Lowered Execution

**Status:** Completed
**Date:** 2026-05-02

## Implemented

- Added `CONV2D` to the shared accelerator DAG ABI as node type `54`.
- Lowered legal direct Metal `CONV2D` forward nodes into the accelerator DAG with encoded stride/padding metadata.
- Added native MPSGraph `convolution2D` execution in the Apple Metal shim for dense `FLOAT32` NCHW input and OIHW weight.
- Added optional per-output-channel bias support by reshaping bias to `[1, C_out, 1, 1]` before MPSGraph addition.
- Promoted only the scoped Metal direct `CONV2D` coverage row to `SUPPORTED`.
- Kept `CONV2D_GEMM`, grouped/depthwise Conv2D, dilated Conv2D, unsupported dtype/layout/rank cases, conv backward, and pool ops capability-gated.
- Registered `conv2d_resnet_3x3` as a Metal native coverage target while keeping CUDA visible-blocker expectations.

## Verification

```bash
./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.cuda.lowering.CudaRegionLowererTest
./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest
./gradlew test --tests PreparedExecutionBuildTest --tests Conv2dExecutionTest
./gradlew test --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageRegressionGateTest --tests GpuCoverageSummaryTest --tests GpuTargetSemanticsContractTest --tests StandardWorkloadsTest
./gradlew metalTest
```

All commands passed.

## Notes

- Native parity was verified for no-bias Conv2D and bias+stride+padding Conv2D through Metal buffer binding.
- Pool execution is intentionally deferred to Wave 35-03.
