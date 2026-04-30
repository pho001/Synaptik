---
phase: 04-tuning-and-profile-ownership-audit
plan: "03"
subsystem: accelerator-runtime
tags: [accelerator, cost-model, runtime-config, cpu-safeguards]
requires:
  - phase: 04-tuning-and-profile-ownership-audit
    provides: 04-01 ownership matrix and 04-02 strict profile IO
provides:
  - Runtime-derived accelerator cost factors
  - PROFILE_DERIVED cost summaries in backend selection
  - CPU safeguard regression coverage for runtime-derived costs
affects: [phase-04, phase-05, accelerator-selection, partition-cost-model]
tech-stack:
  added: []
  patterns: [RuntimeConfig-only cost inputs, profile-derived static preset adapter]
key-files:
  created:
    - src/main/java/backend/accelerator/select/ProfileDerivedAcceleratorCostFactors.java
  modified:
    - src/main/java/backend/accelerator/select/AcceleratorPlanCostModel.java
    - src/test/java/graph/optimizer/partition/cost/AcceleratorPartitionScoreModelTest.java
    - src/test/java/PreparedExecutionBuildTest.java
key-decisions:
  - "Accelerator cost factors are derived from RuntimeConfig rather than profile/calibration files."
  - "Backend selection uses PROFILE_DERIVED summaries while preserving minimum-work and materialization-cost rejection reasons."
patterns-established:
  - "Runtime-derived cost adapter: convert audited RuntimeConfig thresholds into the existing StaticCostPreset shape."
  - "Cost model compatibility: keep conservative summarize(plan) for callers without runtime config."
requirements-completed: [TUNE-01, TUNE-02]
duration: 4 min
completed: 2026-04-30
---

# Phase 4 Plan 03: Runtime-Derived Accelerator Cost Summary

**Accelerator backend selection cost summaries derived from audited RuntimeConfig thresholds**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-30T06:22:27Z
- **Completed:** 2026-04-30T06:26:02Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Added `ProfileDerivedAcceleratorCostFactors` with `fromRuntimeConfig(...)` and `toStaticCostPreset()`.
- Derived dispatch/upload/download/layout penalties from runtime accelerator minimum work and CPU materialization threshold.
- Updated `AcceleratorPlanCostModel.decide(...)` to use runtime-aware summaries.
- Added tests proving `PROFILE_DERIVED` is surfaced and minimum-work rejection still wins.
- Re-ran CPU natural region, fusion, and BF16 BLAS safeguards.

## Task Commits

1. **Task 1: Add runtime-derived accelerator cost factors** - `d3a3fe0`
2. **Task 2: Apply runtime-derived costs in backend selection while preserving CPU safeguards** - `b50064a`

## Files Created/Modified

- `src/main/java/backend/accelerator/select/ProfileDerivedAcceleratorCostFactors.java` - Runtime-to-cost preset adapter.
- `src/main/java/backend/accelerator/select/AcceleratorPlanCostModel.java` - Runtime-aware summary path used by `decide(...)`.
- `src/test/java/graph/optimizer/partition/cost/AcceleratorPartitionScoreModelTest.java` - Runtime-derived factor tests.
- `src/test/java/PreparedExecutionBuildTest.java` - Backend selection and minimum-work rejection coverage.

## Decisions Made

- Kept profile/calibration artifact IO out of backend cost packages; runtime-derived costs consume only `RuntimeConfig`.
- Preserved the conservative static summary API for compatibility paths that do not have runtime config.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Verification

- `./gradlew test --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest`
- `./gradlew test --tests PreparedExecutionBuildTest --tests graph.optimizer.partition.CpuNaturalExecutionRegionPlannerTest --tests OptimizerFuseTest --tests BFloat16BlasDispatchTest`
- `./gradlew test --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest --tests PreparedExecutionBuildTest --tests graph.optimizer.partition.CpuNaturalExecutionRegionPlannerTest --tests OptimizerFuseTest --tests BFloat16BlasDispatchTest`

## Self-Check: PASSED

- `AcceleratorPlanCostModel.decide(...)` calls runtime-aware summary generation.
- `PROFILE_DERIVED` appears in selected/rejected backend selection cost summaries.
- No backend cost package references `PlatformRuntimeProfileIO`, `JsonFileBestProfileStore`, `CalibrationArtifactLayout`, or `profiles/platform`.
- CPU safeguard test set passes.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Wave 4 can close benchmark write-boundary tests and docs with runtime-derived cost behavior in place.

---
*Phase: 04-tuning-and-profile-ownership-audit*
*Completed: 2026-04-30*
