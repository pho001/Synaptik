---
phase: 43
status: passed
verified: 2026-05-02
threats_reviewed: 7
threats_open: 0
requirements:
  - CUDATRAIN-01
  - CUDATRAIN-02
  - CUDATRAIN-03
---

# Phase 43 Security Verification

## Result

PASSED. Phase 43 keeps CUDA training/index-gradient support conservative and report-visible. It does not introduce public device tensors, native duplicate-index execution claims, or hidden CPU fallback support.

## Threat Review

| Threat | Status | Mitigation Evidence |
|--------|--------|---------------------|
| T43-01: CUDA backward support inferred from forward support | Closed | Coverage truth and hot-path tests keep CUDA backward/training rows separate from forward rows and Metal native rows. |
| T43-02: Shared matrix support counted as CUDA native backward execution | Closed | CUDA target truth remains native-evidence-gated; tests assert matrix-supported-only and unsupported rows separately. |
| T43-03: `SCATTER_ADD` duplicate indices produce nondeterministic or last-writer-wins behavior | Closed | Legal CUDA scatter candidates still end in `UNSUPPORTED_DUPLICATE_INDEX`. |
| T43-04: Index-gradient bounds or dtype errors collapse into generic fallback | Closed | `CudaIndexWriteSemantics` validates INT32 index role, dense layout, static bounds, and shape before duplicate-index rejection. |
| T43-05: Training coverage hides internal CPU materialization behind gradient publication | Closed | Training target tests keep native-buffer policy, internal CPU materialization, and gradient publication limits distinct. |
| T43-06: Unsupported CUDA training rows pass without visible blockers | Closed | `CUDATRAIN` hot-path expectations require visible blockers for SDPA backward, dense loss, index-target loss, and scatter/index-gradient targets. |
| T43-07: Local benchmark/profile artifacts staged as phase evidence | Closed | Phase 43 commits exclude local tuning/profile artifacts; status still shows them as unrelated uncommitted artifacts. |

## Security-Relevant Verification

```bash
./gradlew test --tests backend.cuda.lowering.CudaRegionLowererTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageRegressionGateTest --tests GpuCoverageSummaryTest --tests SourceTreeHygieneTest
./gradlew classes
git diff --check
```

All relevant commands passed during Phase 43 execution.

## Residual Risk

- CUDA duplicate-index accumulation remains unsupported until native execution and CPU parity evidence exist.
- CUDA SDPA backward, conv/pool backward, and index-target loss gradients remain explicit unsupported/capability blockers.

