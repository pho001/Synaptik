---
phase: 41
plan: 41-03
status: complete
completed: 2026-05-02
requirements:
  - CUDAINDEX-01
---

# Plan 41-03 Summary: CUDA Forward Gather/Take Execution Or Stable Rejection

## Completed

- Added `CudaPartitionSupport` for CUDA-specific forward index legality checks.
- Routed CUDA `GATHER` and `TAKE_ALONG_AXIS` through dtype, layout, rank/shape, and static bounds validation before final support/rejection classification.
- Kept CUDA forward gather/take as `CAPABILITY_MISSING` for otherwise legal candidates because native CUDA execution is not implemented yet.
- Added tests for legal contract validation, dtype rejection, bounds rejection, layout rejection, and unchanged gradient/write blockers.

## Verification

```bash
./gradlew test --tests backend.cuda.lowering.CudaRegionLowererTest
./gradlew test --tests backend.accelerator.lowering.GpuBackendParityReportTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest
./gradlew test --tests PreparedExecutionBuildTest
```

All commands passed.

## Residual Scope

- CUDA forward gather/take native execution remains future implementation work.
- `GATHER_GRAD`, `TAKE_ALONG_AXIS_GRAD`, and `SCATTER_ADD` remain explicit `UNSUPPORTED_DUPLICATE_INDEX` blockers.
