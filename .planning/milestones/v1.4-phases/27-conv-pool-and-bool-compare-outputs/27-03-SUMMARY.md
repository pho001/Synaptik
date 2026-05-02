# 27-03 Summary: Conv Pool Support-Or-Rejection Contract

## Completed

- Added full Metal/CUDA conv/pool matrix coverage for forward, backward, and lowered GEMM operation variants.
- Kept every conv/pool target unsupported with `CAPABILITY_MISSING` instead of generic unlisted operation fallback.
- Added Metal and CUDA planner diagnostics for representative `MAX_POOL2D` and `AVG_POOL2D` paths.
- Preserved existing Metal and CUDA `CONV2D` planner diagnostics and changed their reason to the Phase 27 capability-gated contract.
- Re-ran CPU parity baselines for representative conv and pool execution.

## Verification

- `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuTargetSemanticsContractTest`
- `./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest`
- `./gradlew test --tests Conv2dExecutionTest --tests Pool2dExecutionTest --tests BoolTensorInfrastructureTest`
- Focused Phase 27 combined slice passed.

## Residual Work

- No conv/pool row is marked `SUPPORTED`.
- Native support still needs accelerator DAG ABI primitives or verified vendor-library routing plus parity and report evidence.
