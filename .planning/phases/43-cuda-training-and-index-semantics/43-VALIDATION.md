---
phase: 43
status: passed
validated: 2026-05-02
validation_gaps: 0
requirements:
  - CUDATRAIN-01
  - CUDATRAIN-02
  - CUDATRAIN-03
---

# Phase 43 Nyquist Validation

## Result

PASSED. Phase 43 has automated positive and negative evidence for CUDA training truth, CUDA index-write/gradient semantic rejection, and training hot-path blocker visibility.

## Requirement Coverage

| Requirement | Positive Evidence | Negative Evidence | Verdict |
|-------------|-------------------|-------------------|---------|
| CUDATRAIN-01 | `GpuCoverageSummaryTest` and `GpuHotPathCoverageTargetsTest` prove CUDA training/backward truth is explicit and not inherited from forward/Metal rows. | Tests assert unsupported and matrix-supported-only rows do not become native executable without evidence. | PASS |
| CUDATRAIN-02 | `CudaRegionLowererTest` covers valid CUDA `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD` semantic contracts before final duplicate blocker. | Tests reject bad index dtype, non-dense layout, out-of-bounds indices, and dynamic/unproven bounds before duplicate-index fallback. | PASS |
| CUDATRAIN-03 | `GpuHotPathCoverageTargetsTest` maps CUDA training/scatter targets to `CUDATRAIN` and checks supported training targets versus visible blockers. | Unsupported CUDA training targets require blocker strings; supported reduction/normalization training targets require native-buffer policy and zero hidden internal CPU materialization. | PASS |

## Task Coverage

| Plan | Evidence | Verdict |
|------|----------|---------|
| 43-01 | `43-01-SUMMARY.md`, target truth tests, training target tests. | PASS |
| 43-02 | `43-02-SUMMARY.md`, `CudaIndexWriteSemantics`, CUDA lowerer tests. | PASS |
| 43-03 | `43-03-SUMMARY.md`, `CUDATRAIN` metadata, coverage target tests. | PASS |
| 43-04 | `43-04-SUMMARY.md`, docs updates, verification artifact. | PASS |

## Regression Gate

Validated commands:

```bash
./gradlew test --tests backend.cuda.lowering.CudaRegionLowererTest
./gradlew test --tests backend.cuda.lowering.CudaRegionLowererTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageSummaryTest
./gradlew test --tests backend.cuda.lowering.CudaRegionLowererTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageRegressionGateTest --tests GpuCoverageSummaryTest --tests SourceTreeHygieneTest
./gradlew classes
git diff --check
```

All commands passed.

## Nyquist Findings

- No untested CUDA native training support claim was introduced.
- No missing negative-path coverage was found for CUDA index-write/gradient dtype, layout, bounds, or duplicate-index blockers.
- No hidden CPU fallback or tensor-array bridge path was counted as native CUDA training support.
- Local profile artifacts remain unstaged and non-canonical.

