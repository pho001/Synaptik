---
phase: 44-custom-metal-kernel-execution-route
plan: 44-03
subsystem: benchmark-reporting
tags: [metal, custom-kernel, coverage, trace, reports]
requires:
  - phase: 44-01
    provides: custom-kernel route decision fields
  - phase: 44-02
    provides: native custom-kernel execution path
provides:
  - coverage route counts for MPSGraph versus custom Metal kernels
  - rejected route reason counts in GPU coverage summaries
  - text and JSON benchmark report fields for route-level evidence
affects: [gpu-coverage, benchmark-reports, metal-trace]
tech-stack:
  added: []
  patterns: [route-level coverage aggregation beside native copy strategy aggregation]
key-files:
  created: []
  modified:
    - src/main/java/tuning/benchmark/report/GpuCoverageSummary.java
    - src/main/java/tuning/benchmark/report/JsonBenchmarkReportRenderer.java
    - src/main/java/tuning/benchmark/report/TextBenchmarkReportRenderer.java
    - src/test/java/GpuCoverageSummaryTest.java
key-decisions:
  - "Coverage summaries now count selected Metal execution routes separately from native copy strategies."
  - "Rejected route reason codes are preserved in coverage summaries so custom-kernel absence remains visible."
patterns-established:
  - "Route counts and copy strategy counts are separate report axes."
requirements-completed: [METALKERNEL-03]
duration: 20min
completed: 2026-05-02
---

# Phase 44-03: Route Reporting And Coverage Evidence Summary

**GPU coverage reports now distinguish custom Metal kernels from MPSGraph and preserve rejected route evidence.**

## Performance

- **Duration:** 20 min
- **Started:** 2026-05-02T14:18:00Z
- **Completed:** 2026-05-02T14:38:09Z
- **Tasks:** 4
- **Files modified:** 4

## Accomplishments

- Added `executionRouteCounts` to `GpuCoverageSummary.BackendCoverage` so reports can count `CUSTOM_KERNEL` and `MPS_GRAPH` separately.
- Added `rejectedRouteReasonCounts` so custom-kernel unavailability or MPSGraph rejection is visible in aggregate coverage output.
- Extended text and JSON benchmark renderers with route count fields.
- Added synthetic coverage tests for `CUSTOM_KERNEL`, `TRUE_OUTPUT_BUFFER_WRITE`, and rejected route reason evidence.
- Preserved compatibility with existing coverage regression/gap triage tests through an overload for the previous `BackendCoverage` constructor shape.

## Task Commits

1. **Route-aware coverage/report aggregation** - `dbd64a2` (report)

## Files Created/Modified

- `src/main/java/tuning/benchmark/report/GpuCoverageSummary.java` - route and rejected route reason counts.
- `src/main/java/tuning/benchmark/report/TextBenchmarkReportRenderer.java` - route counts in text reports.
- `src/main/java/tuning/benchmark/report/JsonBenchmarkReportRenderer.java` - route counts in JSON reports.
- `src/test/java/GpuCoverageSummaryTest.java` - route-aware coverage regression test.

## Decisions Made

Route selection and native copy strategy are separate axes: `CUSTOM_KERNEL` says which route ran, while `TRUE_OUTPUT_BUFFER_WRITE` says how output storage was written.

## Deviations from Plan

No source change was needed in `PreparedExecution`; Phase 44-01/44-02 already populated the route fields. 44-03 focused on aggregation and renderer visibility.

## Issues Encountered

One helper used the wrong `RunTrace` accessor during test construction and was corrected before tests passed.

## Verification

- `./gradlew test --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests backend.metal.exec.MetalExecutionRouterTest`
- `./gradlew test --tests BenchmarkSessionTest --tests GpuCoverageRegressionGateTest --tests GpuCoverageGapTriageTest`
- `git diff --check -- src/main/java/graph/execution/PreparedExecution.java src/main/java/tuning/benchmark/report src/test/java/CompiledGraphTraceTest.java src/test/java/GpuCoverageSummaryTest.java src/test/java/BenchmarkSessionTest.java`

## User Setup Required

None.

## Next Phase Readiness

44-04 can update docs and planning artifacts with the scoped custom-kernel route contract, native execution proof, coverage fields, and residual limitations.

---
*Phase: 44-custom-metal-kernel-execution-route*
*Completed: 2026-05-02*
