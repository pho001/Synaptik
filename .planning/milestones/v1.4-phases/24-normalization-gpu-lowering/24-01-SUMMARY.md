# Summary 24-01: Shared Normalization DAG Contract

**Status:** Complete
**Date:** 2026-05-01

## Completed

- Added `AcceleratorDagNodeType.ADD_SCALAR` with stable ABI code `40` for epsilon addition.
- Lowered legal `LAYER_NORM` and `RMS_NORM` into backend-neutral accelerator DAG subgraphs using repeated keep-dims `MEAN`, `ADD_SCALAR`, `SQRT`, `INV`, and elementwise primitives.
- Added pre-lowering validation for FLOAT32 dtype, rank 1-4, normalized rank, output shape, and gamma/beta tail shape contracts.
- Added normalization compound metadata via the `NORMALIZATION` GPU pattern summary.
- Added focused lowerer tests for LayerNorm, RMSNorm, epsilon scalar metadata, multi-axis keep-dims reductions, external gamma/beta inputs, and invalid parameter shape rejection.

## Verification

Passed:

```bash
./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.accelerator.lowering.GpuCompoundPatternDetectorTest
```

## Deviations from Plan

None.
