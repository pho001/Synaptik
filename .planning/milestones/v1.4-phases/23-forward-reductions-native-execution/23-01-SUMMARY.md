# Summary 23-01: Shared Forward Reduction DAG Contract

**Status:** Complete
**Date:** 2026-05-01

## Completed

- Added `SUM`, `MEAN`, `REDUCE_MIN`, and `REDUCE_MAX` accelerator DAG node types.
- Lowered forward reduction descriptors into DAG primitives with encoded axis/keep-dims metadata.
- Updated the shared GPU lowering matrix to mark forward reductions supported for Metal and CUDA.
- Updated v1.4 target coverage truth so forward reductions count as native-executable targets.
- Updated lowerer and matrix tests, including candidate-shortening coverage now using normalization as the unsupported internal primitive.

## Verification

Passed:

```bash
./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuCoverageSummaryTest
```
