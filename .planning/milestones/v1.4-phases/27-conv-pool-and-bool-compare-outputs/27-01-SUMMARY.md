# 27-01 Summary: Conv Pool And Bool Surface Coverage Contract

## Completed

- Added Phase 27 planning artifacts under `.planning/phases/27-conv-pool-and-bool-compare-outputs/`.
- Expanded `GpuLoweringCoverageMatrix` to list the full conv/pool target surface for Metal and CUDA:
  - `CONV2D`
  - `CONV2D_GEMM`
  - `CONV2D_BACKWARD_INPUT`
  - `CONV2D_BACKWARD_WEIGHT`
  - `CONV2D_BACKWARD_INPUT_GEMM`
  - `CONV2D_BACKWARD_WEIGHT_GEMM`
  - `MAX_POOL2D`
  - `MAX_POOL2D_BACKWARD_INPUT`
  - `AVG_POOL2D`
  - `AVG_POOL2D_BACKWARD_INPUT`
- Expanded `GpuLoweringCoverageMatrix` to list the full BOOL output surface for Metal and CUDA:
  - `GT`, `GE`, `LT`, `LE`, `EQ`, `NE`
  - `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_NOT`
  - `REDUCE_ALL`, `REDUCE_ANY`
- Kept conv/pool rows `UNSUPPORTED` with `CAPABILITY_MISSING` until native/lowered backend execution exists.
- Kept BOOL-producing rows `UNSUPPORTED` with `UNSUPPORTED_DTYPE` until native BOOL output compute exists.
- Updated `GpuTargetSemanticsContract` with conv/pool variants, logical BOOL ops, and BOOL reductions.
- Updated Metal planner diagnostics so BOOL-producing nodes use the matrix-backed stable rejection instead of a generic dtype message.
- Updated docs to distinguish external BOOL predicate residency for `WHERE` from native BOOL-producing GPU compute.

## Verification

- `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuTargetSemanticsContractTest`
- `./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest`
- `./gradlew test --tests Conv2dExecutionTest --tests Pool2dExecutionTest --tests BoolTensorInfrastructureTest`
- `git diff --check`

## Residual Work

- Phase 27-02 still needs compare-to-`WHERE` boundary evidence in prepared execution/coverage reports.
- Phase 27-03 still needs representative conv/pool support-or-rejection coverage gates.
- Native BOOL output and native conv/pool execution remain unclaimed until DAG ABI, lowerer mapping, backend execution, and parity evidence exist.
