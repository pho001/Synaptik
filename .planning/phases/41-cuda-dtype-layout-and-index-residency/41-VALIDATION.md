---
phase: 41
status: passed
validated: 2026-05-02
validation_gaps: 0
requirements:
  - CUDADTYPE-01
  - CUDADTYPE-02
  - CUDAINDEX-01
---

# Phase 41 Nyquist Validation

## Result

PASSED. Phase 41 has automated positive and negative evidence for each requirement, plus docs and verification artifacts that preserve the distinction between CUDA residency, native compute, layout repair, and CPU/materialization fallback.

## Requirement Coverage

| Requirement | Positive Evidence | Negative Evidence | Verdict |
|-------------|-------------------|-------------------|---------|
| CUDADTYPE-01 | `CudaDTypeRolePolicyTest` proves `FLOAT32` compute/output and role support for `INT32`, `BOOL`, and `BFLOAT16` residency. | Same test rejects `INT32`, `BOOL`, and `BFLOAT16` generic compute/output via `RESIDENCY_ONLY_NOT_COMPUTE`; binder tests reject unsupported compute dtypes. | PASS |
| CUDADTYPE-02 | `CudaDeviceLayoutMaterializerTest`, `CudaLayoutTransformDeviceFlowTest`, and shared layout tests cover metadata-only and dense `FLOAT32` materialization paths. | Tests reject unsupported dtype, non-dense target layout, CUDA broadcast materialization, and missing materializer paths with stable reasons. | PASS |
| CUDAINDEX-01 | `CudaRegionLowererTest` validates legal dense `FLOAT32` value/output plus static `INT32` indices before final CUDA `CAPABILITY_MISSING`. | Tests reject bad index dtype, out-of-bounds indices, non-dense layouts, and keep index-gradient/write ops on `UNSUPPORTED_DUPLICATE_INDEX`. | PASS |

## Task Coverage

| Plan | Evidence | Verdict |
|------|----------|---------|
| 41-01 | `41-01-SUMMARY.md`, CUDA dtype role classes, capability report tests, binder diagnostics. | PASS |
| 41-02 | `41-02-SUMMARY.md`, CUDA layout materializer diagnostics, CUDA layout tests. | PASS |
| 41-03 | `41-03-SUMMARY.md`, `CudaPartitionSupport`, CUDA index contract tests, prepared execution check. | PASS |
| 41-04 | `41-04-SUMMARY.md`, docs updates, coverage target requirement tags, coverage report/target tests. | PASS |

## Regression Gate

Validated commands:

```bash
./gradlew test --tests backend.cuda.CudaDTypeRolePolicyTest --tests backend.cuda.bridge.CudaCapabilityReportTest --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest --tests backend.cuda.buffer.CudaDeviceLayoutMaterializerTest --tests backend.cuda.exec.CudaLayoutTransformDeviceFlowTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests backend.accelerator.lowering.GpuBackendParityReportTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuCoverageTriageReportTest --tests GpuHotPathCoverageTargetsTest --tests PreparedExecutionBuildTest --tests SourceTreeHygieneTest
./gradlew classes
git diff --check
```

All commands passed.

## Nyquist Findings

- No missing test dimensions were found.
- No untested support claim was found.
- No hidden CPU fallback or tensor-array support claim was introduced.
- Local profile artifacts remain unstaged and non-canonical.
