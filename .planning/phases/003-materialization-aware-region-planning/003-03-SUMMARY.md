---
phase: 003-materialization-aware-region-planning
plan: "03"
subsystem: testing
tags: [accelerator, cpu-safeguards, documentation, verification]
requires:
  - phase: 003-01-static-cost-and-planner-finalist-trace-summary
    provides: Static materialization cost summaries and compile-time finalist traces
  - phase: 003-02-backend-selection-cost-report-summary
    provides: Prepare-time backend selection cost summaries and benchmark report output
provides:
  - CPU natural region, fusion, and BLAS safeguards under accelerator offload policy
  - Architecture documentation for static materialization-aware region planning
  - Phase 4 profile/calibration-derived cost model ownership note
affects: [cpu-hot-paths, docs, phase-4-tuning]
tech-stack:
  added: []
  patterns: [focused CPU safeguard tests, static planning documentation]
key-files:
  created: []
  modified:
    - src/test/java/graph/optimizer/partition/CpuNaturalExecutionRegionPlannerTest.java
    - src/test/java/OptimizerFuseTest.java
    - src/test/java/BFloat16BlasDispatchTest.java
    - src/test/java/PreparedExecutionBuildTest.java
    - docs/architecture.md
    - .planning/ROADMAP.md
    - .planning/STATE.md
key-decisions:
  - "CPU natural regions, CPU fusion, and BLAS dispatch remain explicit regression anchors when accelerator offload policy is enabled."
  - "Profile/calibration-derived accelerator cost updates remain deferred to Phase 4 after tuning/profile ownership is audited."
patterns-established:
  - "Accelerator offload tests must prove CPU hot paths remain available when static cost or runtime gates do not select accelerator execution."
requirements-completed: [PLAN-02, PLAN-03, PLAN-04]
duration: 8 min
completed: 2026-04-30
---

# Phase 3 Plan 03: CPU Safeguards And Static Planning Documentation

**CPU hot-path regression guards and architecture docs for static materialization-aware accelerator planning**

## Performance

- **Duration:** 8 min
- **Started:** 2026-04-30T05:26:00Z
- **Completed:** 2026-04-30T05:33:54Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments

- Added CPU natural-region coverage proving `CPU_NATURAL_EXECUTION_REGION` remains present under accelerator scored offload policy.
- Added CPU fusion coverage proving fused prepared execution remains available when runtime accelerator execution is disabled under an offload-capable compile policy.
- Added BF16 BLAS coverage proving `CPU_MATMUL_BLAS` remains selected under accelerator scored offload policy.
- Added static cost rejection coverage for a boundary-heavy accelerator plan with reason `rejected-materialization-cost`.
- Documented materialization-aware region planning trace/report surfaces and Phase 4 profile/calibration-derived cost ownership.

## Task Commits

1. **Task 1: Add CPU natural, fusion, and BLAS safeguard regressions** - `28bb65a`
2. **Task 2: Document static planning and Phase 4 cost-model ownership** - `3c23e43`
3. **Task 3: Run final targeted verification and summarize Phase 3 execution readiness** - completed in this summary metadata commit.

## Files Created/Modified

- `src/test/java/graph/optimizer/partition/CpuNaturalExecutionRegionPlannerTest.java` - Adds CPU natural region guard under accelerator offload policy.
- `src/test/java/OptimizerFuseTest.java` - Adds CPU fusion guard under accelerator offload policy with runtime accelerators disabled.
- `src/test/java/BFloat16BlasDispatchTest.java` - Adds BF16 BLAS guard under accelerator offload policy.
- `src/test/java/PreparedExecutionBuildTest.java` - Adds static materialization-cost rejection and CPU execution availability guard.
- `docs/architecture.md` - Documents static materialization-aware region planning, compile/prepare trace surfaces, benchmark report output, runtime trace boundaries, and Phase 4 deferral.
- `.planning/ROADMAP.md` - Keeps the Phase 3 plan list and Phase 4 profile/calibration-derived cost model update note explicit.
- `.planning/STATE.md` - Records Phase 3 execution readiness state.

## Verification

- `./gradlew classes` - passed.
- `./gradlew test --tests graph.optimizer.partition.CpuNaturalExecutionRegionPlannerTest --tests OptimizerFuseTest --tests BFloat16BlasDispatchTest --tests PreparedExecutionBuildTest` - passed.
- `rg -n "cpuNaturalRegionsRemainAvailableWhenAcceleratorBenefitIsAmbiguous|cpuFusionRemainsAvailableWithAcceleratorOffloadPolicy|blasDispatchRemainsAvailableWithAcceleratorOffloadPolicy|staticCostDoesNotSelectAcceleratorWhenCpuPathIsClearlyCompetitive" src/test/java` - passed.
- `rg -n "CPU_NATURAL_EXECUTION_REGION|CPU_MATMUL_BLAS|fusedExecutable\\(\\) != null|rejected-materialization-cost" src/test/java/graph/optimizer/partition/CpuNaturalExecutionRegionPlannerTest.java src/test/java/OptimizerFuseTest.java src/test/java/BFloat16BlasDispatchTest.java src/test/java/PreparedExecutionBuildTest.java` - passed.
- `rg -n "Materialization-aware region planning|Phase 3 uses static named presets only|Profile/calibration-derived costs are deferred to Phase 4" docs/architecture.md` - passed.
- `rg -n "003-01-PLAN.md|003-02-PLAN.md|003-03-PLAN.md|profile/calibration-derived cost model update deferred from Phase 3|Plans: 3 plans" .planning/ROADMAP.md` - passed.
- `./gradlew test --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest --tests graph.optimizer.partition.CpuNaturalExecutionRegionPlannerTest --tests PreparedExecutionBuildTest --tests BenchmarkSessionTest --tests OptimizerFuseTest --tests BFloat16BlasDispatchTest` - passed.

`./gradlew metalTest` was skipped because Phase 3 did not change native Metal execution code or the native ABI.

## Decisions Made

- CPU fusion guard disables runtime accelerators while keeping compile-time accelerator offload policy active, proving CPU fused execution remains available under accelerator-aware graph policy.
- Native Metal tests are not part of this plan's final gate because no native Metal execution files changed in Phase 3.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The first CPU fusion guard attempt used default runtime accelerator settings and selected no fused CPU step. The test was corrected to keep accelerator offload policy at compile time but disable runtime accelerators, which matches the safeguard intent.
- Existing local profile tuning files under `profiles/platform/.../tuning/abc/*` remained modified and were not staged.
- Existing `.planning/tmp/` scratch files remained untracked and were not staged.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 3 has targeted automated verification for static cost summaries, prepare/report diagnostics, and CPU hot-path safeguards. Phase 4 can now audit tuning/profile ownership and plan the deferred profile/calibration-derived cost model update.

---
*Phase: 003-materialization-aware-region-planning*
*Completed: 2026-04-30*
