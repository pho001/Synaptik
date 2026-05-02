---
phase: 42
status: passed
verified: 2026-05-02
threats_reviewed: 8
threats_open: 0
requirements:
  - CUDANN-01
  - CUDANN-02
  - CUDANN-03
---

# Phase 42 Security Verification

## Result

PASSED. Phase 42 adds CUDA semantic validation and report-visible blockers for NN operation families without claiming native CUDA execution, changing public `Tensor` semantics, or hiding CPU fallback paths.

## Threat Review

| Threat | Status | Mitigation Evidence |
|--------|--------|---------------------|
| T42-01: Masked or causal CUDA SDPA counted as supported without BOOL mask parity evidence | Closed | `CudaNnSemantics` validates mask dtype/layout/shape and reports `CAPABILITY_MISSING` with explicit `maskMode` instead of support. |
| T42-02: CUDA SDPA shape or dtype mismatch collapses into generic capability fallback | Closed | SDPA checks reject `FLOAT64`, non-dense layout, rank mismatch, head mismatch, and output mismatch before capability rejection. |
| T42-03: cuDNN-style conv/pool support implied before a vendor-library route exists | Closed | Legal conv/pool candidates end in `CAPABILITY_MISSING`; docs state cuDNN/custom CUDA routing is not integrated. |
| T42-04: Grouped, depthwise, or dilated conv treated as covered by the scoped forward contract | Closed | Group/channel and dilation checks have explicit capability blockers before the legal scoped forward blocker. |
| T42-05: `AVG_POOL2D countIncludePad=true` silently uses incompatible divisor semantics | Closed | CUDA pool semantics reject count-including padding with a specific blocker. |
| T42-06: Dense loss support confused with index-target loss support | Closed | Dense `NLL_LOSS` / `CROSS_ENTROPY_LOSS` use `DAG_PRIMITIVE_UNSUPPORTED`; index-target loss remains `UNSUPPORTED_INDEX_SEMANTICS`. |
| T42-07: Coverage reports pass because CUDA NN gaps are invisible | Closed | `GpuHotPathCoverageTargets` adds `CUDANN` target metadata and expected visible blockers for SDPA, conv/pool, and dense loss. |
| T42-08: Local benchmark/profile artifacts staged as phase evidence | Closed | Phase 42 commits exclude local tuning/profile artifacts; `git status --short` shows only those unrelated artifacts remain uncommitted. |

## Security-Relevant Verification

```bash
./gradlew test --tests backend.cuda.lowering.CudaRegionLowererTest
./gradlew test --tests backend.cuda.lowering.CudaRegionLowererTest --tests backend.accelerator.lowering.GpuBackendParityReportTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuCoverageTriageReportTest --tests GpuHotPathCoverageTargetsTest --tests SourceTreeHygieneTest
./gradlew classes
git diff --check
```

All relevant commands passed during Phase 42 execution.

## Residual Risk

- CUDA SDPA, conv/pool, and dense loss remain unsupported native execution paths until a future CUDA bridge, cuDNN route, or custom CUDA kernel route proves parity.
- Capability-gated CUDA native execution remains dependent on local CUDA hardware/toolchain availability.

