# 27-02 Summary: Bool Compare Output Residency And WHERE Boundary Evidence

## Completed

- Added prepared-execution evidence for a `GT -> WHERE -> RELU` graph.
- Proved the BOOL-producing `GT` node is not claimed as native Metal compute.
- Proved the adjacent Metal `WHERE + RELU` region can still be prepared as a GPU DAG with the CPU-produced BOOL predicate as an external input.
- Verified the Metal planner emits the matrix-backed `UNSUPPORTED_DTYPE` reason for the compare node.
- Preserved the architectural distinction:
  - external BOOL predicate residency for `WHERE` is supported as a runtime input role;
  - native BOOL-producing GPU compute remains unsupported until a DAG/native ABI output path exists.

## Verification

- `./gradlew test --tests PreparedExecutionBuildTest.gpuMetalWhereUsesCpuProducedComparePredicateAsExplicitBoolBoundary`
- Focused Phase 27 slice:
  - `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuTargetSemanticsContractTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest.gpuMetalWhereUsesCpuProducedComparePredicateAsExplicitBoolBoundary --tests Conv2dExecutionTest --tests Pool2dExecutionTest --tests BoolTensorInfrastructureTest`

## Residual Work

- CUDA BOOL predicate/input boundary evidence remains conservative because CUDA native dense buffer execution is still `FLOAT32`-oriented locally.
- Native BOOL-producing GPU output support still requires DAG ABI node types, lowerer mappings, native execution, and CPU parity tests.
