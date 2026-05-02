---
phase: 43
type: verification
status: complete
requirements:
  - CUDATRAIN-01
  - CUDATRAIN-02
  - CUDATRAIN-03
verified_at: 2026-05-02
---

# Phase 43 Verification: CUDA Training And Index Semantics

## Result

Phase 43 is verified as CUDA training/index support-or-rejection coverage. It does not add native CUDA scatter/index-gradient or broad backward execution; it makes CUDA training blockers semantic, report-visible, and regression-tested.

## Requirement Mapping

| Requirement | Status | Evidence |
|---|---|---|
| CUDATRAIN-01 | Complete | Coverage truth and target tests distinguish CUDA backward rows from forward support and Metal native rows. |
| CUDATRAIN-02 | Complete | `CudaIndexWriteSemantics` validates CUDA `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD` dtype/layout/rank/shape/bounds before final `UNSUPPORTED_DUPLICATE_INDEX`. |
| CUDATRAIN-03 | Complete | `CUDATRAIN` hot-path metadata and tests keep unsupported CUDA training rows visible and supported training targets gated against hidden internal CPU materialization. |

## Verification Commands

```bash
./gradlew test --tests backend.cuda.lowering.CudaRegionLowererTest
./gradlew test --tests backend.cuda.lowering.CudaRegionLowererTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageSummaryTest
```

Both focused gates passed.

## Residual Scope

- CUDA duplicate-index accumulation still needs real native execution and CPU parity evidence before `SCATTER_ADD`, `GATHER_GRAD`, or `TAKE_ALONG_AXIS_GRAD` can become supported.
- CUDA SDPA backward, conv/pool backward, and index-target loss gradients remain explicit unsupported/capability blockers.

