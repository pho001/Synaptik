# Phase 13 Pattern Map: Coverage Benchmark And Regression Gate

**Phase:** 13 - Coverage Benchmark And Regression Gate
**Date:** 2026-04-30
**Status:** Complete

## Pattern Mapping Complete

## Files To Modify Or Extend

| Planned file | Role | Closest existing analog | Notes |
|---|---|---|---|
| `src/main/java/tuning/benchmark/report/GpuCoverageSummary.java` | Backend-neutral coverage aggregate | `AcceleratorTraceSummary.java` | Immutable report record with static `fromTrace(ExecutionTrace)` factory. |
| `src/main/java/tuning/benchmark/report/GpuCoverageBaseline.java` | Stable baseline contract for coverage comparison | `BenchmarkSuiteCandidateSummary.java` | Small record used by tests/report comparisons, not a profile artifact. |
| `src/main/java/tuning/benchmark/report/GpuCoverageComparison.java` | Current-vs-baseline comparison result | `ReportingDiffRendererTest` report diff patterns | Should compare coverage/materialization, not raw latency. |
| `src/main/java/tuning/benchmark/report/BenchmarkReport.java` | Per-workload report root | Existing record | Add derived coverage summary helpers without breaking constructor shape if possible. |
| `src/main/java/tuning/benchmark/report/BenchmarkCandidateReport.java` | Candidate report root | Existing record | Prefer derived methods over new constructor fields unless executor proves schema needs fields. |
| `src/main/java/tuning/benchmark/report/BenchmarkSuiteReport.java` | Suite-level aggregation | Existing suite summary methods | Add coverage summary/comparison aggregation across workload reports. |
| `src/main/java/tuning/benchmark/report/TextBenchmarkReportRenderer.java` | Human-readable report | Existing accelerator/backend selection render blocks | Add `coverage:` block beside `accelerator:` and `backendSelectionCost:`. |
| `src/main/java/tuning/benchmark/report/JsonBenchmarkReportRenderer.java` | JSON report contract | Existing accelerator JSON renderer | Add stable `coverage` object while preserving existing fields. |
| `src/main/java/tuning/benchmark/report/TextBenchmarkSuiteReportRenderer.java` | Suite report renderer | Existing candidate summary/hotspot sections | Add suite-level coverage summary. |
| `src/main/java/tuning/benchmark/report/JsonBenchmarkSuiteReportRenderer.java` | Suite JSON renderer | Existing nested report renderer | Add `coverage` summary and per-workload coverage comparisons. |
| `src/test/java/GpuCoverageSummaryTest.java` | Summary unit tests | `BenchmarkSessionTest` synthetic trace tests | Assert exact metrics from synthetic Metal/CUDA traces. |
| `src/test/java/GpuCoverageRegressionGateTest.java` | Gate utility tests | `CompiledGraphTraceTest`, `EtalonPerformanceRegressionTest` | Assert failure messages for lost coverage and hidden fallback. |
| `src/test/java/BenchmarkSessionTest.java` | Renderer/report regression tests | Existing renderer assertions | Extend to assert `gpuCoverageRatio`, region length, handoff, materialization reasons. |
| `src/test/java/BenchmarkSuiteSessionTest.java` | Suite report tests | Existing suite renderer tests | Assert suite-level coverage summary and representative workload names. |
| `src/test/java/CompiledGraphTraceTest.java` | Prepared/run trace evidence tests | Existing GPU lowering and materialization tests | Add assertions for hidden CPU exits where necessary. |
| `docs/compute-flow.md` | Trace semantics docs | Existing accelerator trace field docs | Document coverage fields and how to read CPU materialization boundaries. |
| `docs/gpu-lowering-coverage.md` | GPU coverage docs | Phase 11/12 coverage matrix doc | Link lowering/fusion coverage to Phase 13 report gates. |
| `docs/development.md` | Verification commands and hygiene | Existing focused Gradle command sections | Add coverage benchmark regression commands and artifact hygiene. |
| `docs/testing.md` | Test strategy | Existing native capability-skip explanation | Clarify portable vs native coverage gates. |

## Reusable Code Patterns

### Immutable Report Records

Report objects in `tuning.benchmark.report` use Java records, null-normalizing constructors, and static factories. Phase 13 should use the same pattern for `GpuCoverageSummary`, baseline, and comparison records.

### Renderer Compatibility

Existing renderers append new sections while preserving existing strings asserted by tests. Phase 13 should keep current `accelerator`, `backendSelectionCost`, `cpuMaterializations`, `steps`, and `hotSteps` fields intact.

### Synthetic Trace Fixtures

`BenchmarkSessionTest` already builds synthetic `ExecutionTrace`, `ExecutionStepTrace`, `StepExecutionMetadata`, and `CpuMaterializationTrace` objects. Phase 13 tests should reuse that pattern for deterministic Metal/CUDA coverage evidence without native hardware.

### Capability-Gated Native Evidence

Previous phases use portable tests for schema/fallback contracts and `metalTest` / `buildCudaGraphShim cudaTest` for native evidence. Phase 13 should keep the same split.

## Landmines

- Do not use raw latency as the only success criterion.
- Do not count tensor-array bridge execution as equivalent to native buffer GPU coverage.
- Do not hide CPU materialization behind generic accelerator presence.
- Do not require CUDA hardware for portable report/schema tests.
- Do not commit `profiles/platform/.../tuning/abc/*` local tuning artifacts.
- Do not change the public `Tensor` API to expose device residency.
