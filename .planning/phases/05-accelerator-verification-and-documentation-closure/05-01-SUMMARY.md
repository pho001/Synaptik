---
phase: 05-accelerator-verification-and-documentation-closure
plan: "01"
subsystem: observability
tags: [benchmark-report, accelerator-trace, planner-evidence, materialization]

requires:
  - phase: 003-materialization-aware-region-planning
    provides: static accelerator planner cost summaries and selected/rejected candidate traces
  - phase: 04-tuning-and-profile-ownership-audit
    provides: runtime-derived accelerator cost model inputs
provides:
  - Stable benchmark report evidence for accelerator reason codes and fallback reasons
  - Portable report contract tests for accelerator path, copy timing, storage residency, planner finalists, and CPU materialization evidence
affects: [phase-05, benchmark-reporting, accelerator-observability]

tech-stack:
  added: []
  patterns: [in-memory benchmark report fixtures, backend-neutral accelerator evidence aggregation]

key-files:
  created:
    - .planning/phases/05-accelerator-verification-and-documentation-closure/05-01-SUMMARY.md
  modified:
    - src/main/java/tuning/benchmark/report/AcceleratorTraceSummary.java
    - src/main/java/tuning/benchmark/report/TextBenchmarkReportRenderer.java
    - src/main/java/tuning/benchmark/report/JsonBenchmarkReportRenderer.java
    - src/test/java/BenchmarkSessionTest.java
    - src/test/java/PreparedExecutionBuildTest.java

key-decisions:
  - "Aggregate accelerator reason evidence from backend-neutral acceleratorBuffer* trace attributes, with Metal fallback reasons included opportunistically."
  - "Keep Phase 5 benchmark evidence portable by constructing in-memory report fixtures rather than writing measured machine-local reports."

patterns-established:
  - "Benchmark report contract tests assert specific evidence values for accelerator execution and planner decisions."
  - "Report renderers expose backend-neutral accelerator evidence in both text and JSON outputs."

requirements-completed: [OBS-01, OBS-02]

duration: 18 min
completed: 2026-04-30
---

# Phase 5 Plan 01: Trace And Benchmark Report Evidence Contract Summary

**Benchmark reports now expose stable accelerator reason, fallback, planner, materialization, copy-time, and residency evidence through portable Java tests.**

## Performance

- **Duration:** 18 min
- **Started:** 2026-04-30T07:04:00Z
- **Completed:** 2026-04-30T07:22:31Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Added `reasonCodes` and `fallbackReasons` to `AcceleratorTraceSummary.BackendSummary`.
- Rendered accelerator reason/fallback evidence in text and JSON benchmark reports.
- Added portable in-memory report assertions for accelerator buffer path, copy timing, storage residency, CPU materialization, selected GPU backend, rejected finalists, and cost summary fields.
- Added prepare-trace coverage proving selected accelerator decisions carry planner cost evidence.

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend accelerator report summary with fallback reason contract fields** - `5c12c30` (feat)
2. **Task 2: Assert planner context and materialization evidence in report fixtures** - `6d67ee4` (test)

## Files Created/Modified

- `src/main/java/tuning/benchmark/report/AcceleratorTraceSummary.java` - Aggregates unique accelerator reason codes and fallback reasons per backend.
- `src/main/java/tuning/benchmark/report/TextBenchmarkReportRenderer.java` - Prints reason code and fallback reason lists in accelerator summaries.
- `src/main/java/tuning/benchmark/report/JsonBenchmarkReportRenderer.java` - Emits machine-readable reason and fallback arrays in accelerator summaries.
- `src/test/java/BenchmarkSessionTest.java` - Asserts accelerator report evidence, planner cost evidence, and CPU materialization report fields.
- `src/test/java/PreparedExecutionBuildTest.java` - Asserts selected accelerator prepare-trace decisions carry non-null cost summary evidence.
- `.planning/phases/05-accelerator-verification-and-documentation-closure/05-01-SUMMARY.md` - Captures plan outcome and verification.

## Decisions Made

- Reason evidence remains backend-neutral where possible: `acceleratorBufferReasonCode` feeds `reasonCodes`, while `acceleratorBufferReason` and `metalFallbackReason` feed `fallbackReasons`.
- Report evidence tests use constructed traces and benchmark reports, avoiding measured local benchmark output in version control.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The new accelerator evidence test initially used `Map.of` with more than ten pairs, which Java rejects. Replaced it with `Map.ofEntries` before committing Task 1.

## Verification

- `rg -n "fallbackReasons|reasonCodes" src/main/java/tuning/benchmark/report/AcceleratorTraceSummary.java src/main/java/tuning/benchmark/report/TextBenchmarkReportRenderer.java src/main/java/tuning/benchmark/report/JsonBenchmarkReportRenderer.java` - PASS
- `rg -n "renderersExposeAcceleratorEvidenceContract|acceleratorBufferMode|BUFFER_BINDING_AVAILABLE|DEVICE_OWNED|nativeDeviceCopyNs" src/test/java/BenchmarkSessionTest.java` - PASS
- `rg -n "boundaryCount=2|rejectedFinalists|cpuMaterializationCount=1|sourceResidency|durationNs|prepareTraceSelectedAcceleratorDecisionCarriesPlannerEvidence" src/test/java/BenchmarkSessionTest.java src/test/java/PreparedExecutionBuildTest.java` - PASS
- `./gradlew test --tests BenchmarkSessionTest` - PASS
- `./gradlew test --tests BenchmarkSessionTest --tests PreparedExecutionBuildTest` - PASS

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for `05-02`, which can build on the report contract to add closure workload and capability-gated Metal/CUDA evidence tests.

## Self-Check: PASSED

- Summary file exists.
- Task commits `5c12c30` and `6d67ee4` exist in git history.
- Plan-level verification passed.

---
*Phase: 05-accelerator-verification-and-documentation-closure*
*Completed: 2026-04-30*
