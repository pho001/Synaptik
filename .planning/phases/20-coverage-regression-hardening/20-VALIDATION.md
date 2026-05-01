---
phase: 20-coverage-regression-hardening
status: planned
created_at: 2026-05-01
---

# Phase 20 Validation Strategy

## Validation Dimensions

| Dimension | Evidence Required |
|---|---|
| Gate semantics | `GpuCoverageRegressionGateTest` proves stable failures for coverage loss, shortened regions, lost lowered/fused counts, CPU materialization, tensor-array fallback, CPU fallback, handoff, native binding loss, and missing summaries. |
| Target coverage | `GpuHotPathCoverageTargetsTest` or a focused target gate test proves all Phase 14 workloads have deterministic expectations. |
| Report rendering | `BenchmarkSessionTest` and `BenchmarkSuiteSessionTest` prove text/JSON reports render gate inputs, gate results, and native pass/skip evidence. |
| Portable execution | Java-only tests pass without CUDA hardware or local native libraries. |
| Native capability evidence | `metalTest` and CUDA native checks are capability-gated and skipped evidence is documented when unavailable. |
| Artifact hygiene | `SourceTreeHygieneTest` and `git status --short` prove local `profiles/platform/.../tuning/abc/*` artifacts remain unstaged. |

## Final Focused Gate

```bash
./gradlew classes
./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests CompiledGraphTraceTest --tests SourceTreeHygieneTest
git status --short
```

Optional native evidence:

```bash
./gradlew metalTest
./gradlew buildCudaGraphShim cudaTest
```
