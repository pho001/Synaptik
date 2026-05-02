---
phase: 40
plan: 40-01
status: complete
completed: 2026-05-02
requirements:
  - CUDAPARITY-01
key-files:
  - src/main/java/backend/accelerator/lowering/GpuBackendParityRow.java
  - src/main/java/backend/accelerator/lowering/GpuBackendParityReport.java
  - src/main/java/backend/accelerator/lowering/GpuBackendParityReporter.java
  - src/test/java/backend/accelerator/lowering/GpuBackendParityReportTest.java
---

# Plan 40-01 Summary

## Completed

- Added a derived CUDA-vs-Metal parity report over `GpuLoweringCoverageMatrix`.
- Added deterministic evidence buckets including `CUDA_NATIVE_EXECUTION_REQUIRED`, `CUDA_DTYPE_OR_LAYOUT_CONTRACT_REQUIRED`, `CUDA_INDEX_SEMANTICS_REQUIRED`, and `CUDA_EXPLICIT_REJECTION_OK`.
- Added tests proving Metal-supported CUDA gaps are reported and CUDA `CAPABILITY_MISSING` rows are not counted as support.

## Verification

```bash
./gradlew test --tests backend.accelerator.lowering.GpuBackendParityReportTest
./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest
git diff --check
```

All commands passed.
