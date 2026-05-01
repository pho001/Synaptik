---
phase: 17-normalization-reduction-and-loss-adjacent-lowering
plan: "04"
status: complete
requirements-completed: [GPUNORM-01, GPUNORM-02, GPUNORM-03]
completed: 2026-05-01
---

# Phase 17 Plan 04: Closure Summary

## Normalization reduction and loss-adjacent closure

- [x] `GPUNORM-01`: Shared Metal/CUDA coverage rows, backend legality detail, and docs now define Phase 17 support and rejection semantics.
- [x] `GPUNORM-02`: Hot-path evidence for normalization, reduction, softmax-ish, conv, and loss-adjacent blockers is preserved in the matrix, trace, coverage, and benchmark reports.
- [x] `GPUNORM-03`: CPU parity remained the correctness oracle for numerically sensitive flows while unsupported accelerator paths stayed trace-visible.

## Verification

| Command | Result |
|---------|--------|
| `./gradlew classes` | Passed |
| `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests SourceTreeHygieneTest` | Passed |
| `rg -n "status: verified|Execution Evidence|GpuLoweringCoverageMatrixTest|MetalRegionLowererTest|CudaRegionLowererTest|PreparedExecutionBuildTest|SourceTreeHygieneTest|profiles/platform/.../tuning/abc/\\* remained unstaged" .planning/phases/17-normalization-reduction-and-loss-adjacent-lowering/17-VALIDATION.md` | Passed |
| `git status --short` | Only local tuning profile artifacts were dirty |

## Docs

Developer docs now state that `LOG_SOFTMAX` remains lowered as `SOFTMAX` followed by `LOG`, loss-adjacent fallback remained visible, and native reduction and normalization support is not implied by a fallback row.

## Hygiene

profiles/platform/.../tuning/abc/* remained unstaged

## Self-Check: PASSED
