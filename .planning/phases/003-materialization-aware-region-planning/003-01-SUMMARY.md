---
phase: 003-materialization-aware-region-planning
plan: "01"
subsystem: optimizer
tags: [accelerator, partition-planning, cost-model, traces]
requires:
  - phase: 001-accelerator-buffer-layout-abi
    provides: Backend-neutral layout and fallback reason vocabulary
  - phase: 002-metal-layout-aware-device-flow
    provides: Metal device-owned layout flow and fallback trace context
provides:
  - Static materialization-aware accelerator score summaries
  - Compile-time partition cost summaries and bounded finalist traces
affects: [backend-selection, benchmark-reporting, cpu-safeguards]
tech-stack:
  added: []
  patterns: [Java record-based trace summaries, static named cost presets]
key-files:
  created: []
  modified:
    - src/main/java/graph/optimizer/partition/cost/AcceleratorPartitionScoreModel.java
    - src/main/java/graph/optimizer/partition/ScoredCandidatePartitionPlanner.java
    - src/main/java/graph/execution/trace/PartitionDecisionTrace.java
    - src/test/java/graph/optimizer/partition/cost/AcceleratorPartitionScoreModelTest.java
    - src/test/java/PreparedExecutionBuildTest.java
key-decisions:
  - "Static cost summaries use named presets and internal constants only; no profile or calibration files are read."
  - "Scored partition planning rejects non-profitable static summaries and records only the top three rejected finalists."
requirements-completed: [PLAN-01, PLAN-02, PLAN-03]
duration: 32 min
completed: 2026-04-30
---

# Phase 3 Plan 01: Static Cost And Planner Finalist Trace Summary

**Static accelerator materialization cost summaries with compile-time selected and top rejected partition finalists**

## Performance

- **Duration:** 32 min
- **Started:** 2026-04-30T04:50:00Z
- **Completed:** 2026-04-30T05:22:46Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added `MaterializationSignals`, `StaticCostPreset`, and `MaterializationCostSummary` to the accelerator partition score model.
- Preserved existing transfer-aware `acceptedScore(...)` behavior while routing it through the new summary path.
- Added compile-time `PartitionDecisionTrace.CandidateCostTrace` finalist summaries with a hard cap of three entries.
- Updated scored candidate planning to use static cost summaries for candidate selection and finalist demotion/rejection.
- Added focused JUnit coverage for static costs and compile trace cost summaries.

## Task Commits

1. **Task 1: Add static materialization cost summary records and scoring tests** - `4bef8c9`
2. **Task 2: Carry cost summaries and top rejected finalists in partition traces** - `5c62f6c`

## Files Created/Modified

- `src/main/java/graph/optimizer/partition/cost/AcceleratorPartitionScoreModel.java` - Adds static cost records, named presets, materialization-aware score summaries, and reason codes.
- `src/main/java/graph/optimizer/partition/ScoredCandidatePartitionPlanner.java` - Scores lowered candidates through static cost summaries and tracks top rejected finalists.
- `src/main/java/graph/execution/trace/PartitionDecisionTrace.java` - Adds optional cost summary and bounded finalist trace fields.
- `src/test/java/graph/optimizer/partition/cost/AcceleratorPartitionScoreModelTest.java` - Covers dispatch overhead, boundary/transfer penalties, avoided intermediate credit, fallback rejection, and preset behavior.
- `src/test/java/PreparedExecutionBuildTest.java` - Asserts scored offload compile traces carry cost summaries and bounded finalists.

## Verification

- `./gradlew test --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest` - passed.
- `./gradlew test --tests PreparedExecutionBuildTest --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest` - passed.
- Acceptance grep checks for new score records, reason codes, planner finalist strings, and compile trace assertions - passed.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- `gsd-sdk query phase-plan-index 3` could not locate the `003-...` phase directory because it looked for `03`. Execution used the on-disk plan inventory directly.
- `gsd-sdk query roadmap.update-plan-progress 3` also could not locate Phase 3 for the same numbering mismatch, so roadmap progress is updated directly.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Wave 2 can build on the shared `MaterializationCostSummary` and `CandidateCostTrace` records for backend selection and benchmark report rendering.

---
*Phase: 003-materialization-aware-region-planning*
*Completed: 2026-04-30*
