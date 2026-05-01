# Phase 14 Pattern Map: Coverage Gap Triage And Hot Path Targets

**Phase:** 14 - Coverage Gap Triage And Hot Path Targets
**Date:** 2026-05-01
**Status:** Complete

## Pattern Mapping Complete

## Files To Modify Or Extend

| Planned file | Role | Closest existing analog | Notes |
|---|---|---|---|
| `src/main/java/tuning/benchmark/report/GpuCoverageGapCategory.java` | Stable triage category enum | `GpuLoweringCoverageStatus.java` | Small enum with exact category names used by tests and renderers. |
| `src/main/java/tuning/benchmark/report/GpuCoverageGap.java` | Immutable ranked gap record | `GpuCoverageBaseline.java` | Java record with null normalization and deterministic score fields. |
| `src/main/java/tuning/benchmark/report/GpuCoverageGapTriage.java` | Report-to-gap derivation utility | `GpuCoverageSummary.java` | Static factory methods over `BenchmarkReport` and `BenchmarkSuiteReport`. |
| `src/test/java/GpuCoverageGapTriageTest.java` | Triage unit tests | `GpuCoverageSummaryTest.java` | Use synthetic traces and report fixtures to assert exact categories, reasons, and scores. |
| `src/main/java/tuning/benchmark/report/GpuHotPathCoverageTarget.java` | Hot-path workload target record | `GpuCoverageBaseline.java` | Immutable target definition with workload name, category, families, owner phase. |
| `src/main/java/tuning/benchmark/report/GpuHotPathCoverageTargets.java` | Default target registry | `StandardWorkloads.java` | Static default list for v1.3 target workloads. |
| `src/test/java/GpuHotPathCoverageTargetsTest.java` | Target registry tests | `BenchmarkSuiteSessionTest.java` | Assert target names exist in `StandardWorkloads.defaultCatalog()`. |
| `src/main/java/tuning/benchmark/report/GpuCoverageTriageReport.java` | Aggregate report object | `BenchmarkSuiteReport.java` | Derived report with targets, top gaps, category counts, family counts. |
| `src/main/java/tuning/benchmark/report/TextGpuCoverageTriageReportRenderer.java` | Human-readable triage renderer | `TextBenchmarkSuiteReportRenderer.java` | Stable headings and deterministic sorting. |
| `src/main/java/tuning/benchmark/report/JsonGpuCoverageTriageReportRenderer.java` | JSON triage renderer | `JsonBenchmarkSuiteReportRenderer.java` | Stable keys consumed by tests and docs. |
| `src/test/java/GpuCoverageTriageReportTest.java` | Triage report renderer tests | `ReportingDiffRendererTest.java` | Assert exact text headings and JSON keys. |
| `.planning/phases/14-coverage-gap-triage-and-hot-path-targets/14-HOT-PATH-TARGETS.md` | Phase handoff target list | Phase summary files | Checked-in source of truth for downstream phases. |
| `docs/gpu-coverage-triage.md` | Developer docs for triage workflow | `docs/gpu-lowering-coverage.md` | Explain target list, ranking, and artifact hygiene. |
| `docs/development.md` | Verification command docs | Existing coverage command sections | Add focused Phase 14 triage command set. |
| `docs/testing.md` | Test strategy docs | Existing native capability sections | Clarify portable triage evidence vs native execution evidence. |

## Reusable Code Patterns

### Immutable Report Records

Report records in `tuning.benchmark.report` normalize null values in compact constructors. Phase 14 records should do the same and keep mutable collection building in private helper classes.

### Static Derivation Utilities

`GpuCoverageSummary.fromTrace(...)` is the pattern for deriving report values from existing traces without mutating benchmark execution. `GpuCoverageGapTriage` should follow that style.

### Renderer Compatibility

Existing renderers add stable text/JSON fields and tests assert exact labels. New triage renderers should use stable headings and exact JSON property names rather than free-form prose.

### Synthetic Trace Fixtures

Use `GpuCoverageSummaryTest.traceFor(...)` and small `BenchmarkReport.of(...)` fixtures to avoid requiring native Metal/CUDA for triage tests.

## Landmines

- Do not rank by raw timing in Phase 14.
- Do not count tensor-array bridge execution as native buffer coverage.
- Do not create a public device tensor API.
- Do not make target workload names free-form strings without tests against `StandardWorkloads.defaultCatalog()`.
- Do not commit local `profiles/platform/.../tuning/abc/*` artifacts.
- Do not let Phase 14 implement broader GPU lowering; it creates the measured target list for later phases.
