---
phase: 003-materialization-aware-region-planning
plan: "02"
subsystem: benchmark-reporting
tags: [accelerator, backend-selection, cost-model, traces, reports]
requires:
  - phase: 003-01-static-cost-and-planner-finalist-trace-summary
    provides: Static materialization cost summaries and bounded finalist trace records
provides:
  - Prepare-time backend selection cost summaries
  - Text and JSON benchmark report diagnostics for selected and rejected accelerator candidates
affects: [backend-selection, benchmark-reporting, phase-4-tuning]
tech-stack:
  added: []
  patterns: [prepare trace summaries, bounded finalist rendering]
key-files:
  created: []
  modified:
    - src/main/java/backend/accelerator/select/AcceleratorPlanCostModel.java
    - src/main/java/backend/select/DefaultBackendSelectionPolicy.java
    - src/main/java/graph/execution/trace/BackendSelectionDecisionTrace.java
    - src/main/java/tuning/benchmark/report/TextBenchmarkReportRenderer.java
    - src/main/java/tuning/benchmark/report/JsonBenchmarkReportRenderer.java
    - src/test/java/PreparedExecutionBuildTest.java
    - src/test/java/BenchmarkSessionTest.java
key-decisions:
  - "Prepare-time accelerator selection uses static cost summaries only; profile, calibration, history, benchmark, and tuning files are not read in Phase 3."
  - "Benchmark reports expose compact backendSelectionCost sections from prepare traces instead of adding planner cost fields to runtime step traces."
patterns-established:
  - "BackendSelectionDecisionTrace carries costSummary and bounded finalists for report surfaces."
  - "Benchmark report renderers summarize selected backend candidates and up to three rejected finalists."
requirements-completed: [PLAN-01, PLAN-03]
duration: 4 min
completed: 2026-04-30
---

# Phase 3 Plan 02: Backend Selection Cost Report Summary

**Prepare-time accelerator cost decisions rendered in text and JSON benchmark reports without expanding runtime step traces**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-30T05:25:05Z
- **Completed:** 2026-04-30T05:28:42Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Added static `MaterializationCostSummary` decisions to prepare-time accelerator selection.
- Extended backend selection trace decisions with selected cost summaries and bounded rejected finalists.
- Added `backendSelectionCost` output to text and JSON benchmark reports.
- Added renderer coverage proving selected and rejected candidate score fields are visible while `ExecutionStepTrace` stays unchanged.

## Task Commits

1. **Task 1: Extend backend selection cost decisions and traces** - `b799d12`
2. **Task 2: Render compile/prepare cost summaries in benchmark reports** - `1ddeabb`

## Files Created/Modified

- `src/main/java/backend/accelerator/select/AcceleratorPlanCostModel.java` - Produces prepare-time static cost decisions and stable rejection reasons.
- `src/main/java/backend/select/DefaultBackendSelectionPolicy.java` - Attaches cost summaries to backend selection decisions.
- `src/main/java/graph/execution/trace/BackendSelectionDecisionTrace.java` - Carries cost summaries and bounded finalists.
- `src/main/java/tuning/benchmark/report/TextBenchmarkReportRenderer.java` - Renders compact `backendSelectionCost` text diagnostics.
- `src/main/java/tuning/benchmark/report/JsonBenchmarkReportRenderer.java` - Renders structured `trace.backendSelectionCost` JSON diagnostics.
- `src/test/java/PreparedExecutionBuildTest.java` - Covers selected accelerator cost summaries and minimum-work rejection traces.
- `src/test/java/BenchmarkSessionTest.java` - Covers text and JSON backend selection cost report output.

## Verification

- `rg -n "MaterializationCostSummary costSummary|costSummary\\(\\)" src/main/java/backend/accelerator/select/AcceleratorPlanCostModel.java src/main/java/graph/execution/trace/BackendSelectionDecisionTrace.java src/main/java/backend/select/DefaultBackendSelectionPolicy.java` - passed.
- `rg -n "backendSelectionCost|rejectedFinalists|selectedBackend=|estimatedTransferBytes" src/main/java/tuning/benchmark/report/TextBenchmarkReportRenderer.java src/main/java/tuning/benchmark/report/JsonBenchmarkReportRenderer.java` - passed.
- `rg -n "renderersExposeBackendSelectionCostDiagnostics|backendSelectionCost|rejectedFinalists" src/test/java/BenchmarkSessionTest.java` - passed.
- `rg -n "backendSelectionCost" src/main/java/graph/execution/trace/ExecutionStepTrace.java` - no matches.
- `./gradlew test --tests PreparedExecutionBuildTest --tests BenchmarkSessionTest` - passed.

## Decisions Made

- Renderer diagnostics are sourced from `PrepareTrace.backendSelection()` because Phase 3 should show prepare-time selection cost without runtime step bloat.
- Reported rejected finalists are capped by the trace record and renderer to three entries for deterministic, bounded output.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- `gsd-sdk query state.advance-plan` could not parse the current plan counters in `STATE.md`; roadmap progress is updated through the SDK after summary creation and state text is handled by the phase executor.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Wave 3 can add CPU safeguard regressions and final docs using backend selection cost summaries already exposed in prepare traces and benchmark reports.

---
*Phase: 003-materialization-aware-region-planning*
*Completed: 2026-04-30*
