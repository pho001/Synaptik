---
phase: 12-fused-gpu-region-execution
plan: "01"
subsystem: accelerator-lowering
tags: [gpu, lowering, compound-regions, metal, cuda]
requires:
  - phase: 11-gpu-lowering-coverage-matrix
    provides: shared Metal/CUDA lowering coverage matrix and stable unsupported reasons
provides:
  - backend-neutral GPU compound pattern summary contract
  - explicit CPU FUSED rejection reason for GPU lowering
  - detector tests for linear+bias+activation, elementwise chain, reduction-adjacent, and CPU FUSED candidates
affects: [phase-12, phase-13, metal-lowering, cuda-lowering]
tech-stack:
  added: []
  patterns: [summary-plus-dag, stable-compound-reason-codes]
key-files:
  created:
    - src/main/java/backend/accelerator/lowering/GpuCompoundPatternType.java
    - src/main/java/backend/accelerator/lowering/GpuCompoundRegionSummary.java
    - src/main/java/backend/accelerator/lowering/GpuCompoundPatternDetector.java
    - src/main/java/backend/accelerator/lowering/GpuCompoundLoweringArtifact.java
    - src/test/java/backend/accelerator/lowering/GpuCompoundPatternDetectorTest.java
  modified:
    - src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLoweringResult.java
    - src/main/java/backend/accelerator/lowering/GpuLoweringUnsupportedReason.java
    - src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java
    - src/test/java/backend/accelerator/lowering/GpuLoweringCoverageMatrixTest.java
    - docs/gpu-lowering-coverage.md
key-decisions:
  - "GPU compound intent is represented as a backend-neutral summary beside the existing accelerator DAG."
  - "CPU Operation.OpType.FUSED is explicitly unsupported for GPU compound lowering in Phase 12."
patterns-established:
  - "Summary plus DAG: compound summaries describe traceable intent while AcceleratorDagSpec remains the execution contract."
  - "Stable compound reasons: CPU_FUSED_OPERATION_UNSUPPORTED, COMPOUND_PATTERN_UNSUPPORTED, and COMPOUND_REGION_SHORTENED are source-level reason codes."
requirements-completed: [GPUFUSE-03, GPUFUSE-04]
duration: 8 min
completed: 2026-04-30
---

# Phase 12 Plan 01: Shared GPU Compound Pattern Contract Summary

**Backend-neutral GPU compound summaries with explicit CPU FUSED rejection and stable reason codes for Metal/CUDA lowering.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-04-30T19:15:00Z
- **Completed:** 2026-04-30T19:22:52Z
- **Tasks:** 3
- **Files modified:** 10

## Accomplishments

- Added `GpuCompoundPatternType`, `GpuCompoundRegionSummary`, `GpuCompoundPatternDetector`, and `GpuCompoundLoweringArtifact`.
- Extended `AcceleratorSubgraphLoweringResult` with a compatible `compoundSummary()` field.
- Added stable reason codes for CPU FUSED and compound unsupported/shortened cases.
- Updated coverage matrix and docs so GPU paths do not imply they consume CPU fused internals.

## Task Commits

1. **Tasks 1-3: Shared compound contract, tests, reason codes, and docs** - `e30a31b` (`feat(12-01)`)

**Plan metadata:** this summary commit.

## Files Created/Modified

- `src/main/java/backend/accelerator/lowering/GpuCompoundPatternType.java` - stable compound pattern names.
- `src/main/java/backend/accelerator/lowering/GpuCompoundRegionSummary.java` - immutable summary record for trace/legal metadata.
- `src/main/java/backend/accelerator/lowering/GpuCompoundPatternDetector.java` - detector for supported and unsupported compound candidates.
- `src/main/java/backend/accelerator/lowering/GpuCompoundLoweringArtifact.java` - lowered-unit artifact wrapper for summaries.
- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLoweringResult.java` - carries summaries beside DAG specs.
- `src/main/java/backend/accelerator/lowering/GpuLoweringUnsupportedReason.java` - adds Phase 12 compound reason codes.
- `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java` - marks GPU `FUSED` as CPU-only unsupported.
- `src/test/java/backend/accelerator/lowering/GpuCompoundPatternDetectorTest.java` - detector contract tests.
- `src/test/java/backend/accelerator/lowering/GpuLoweringCoverageMatrixTest.java` - coverage row reason-code test.
- `docs/gpu-lowering-coverage.md` - documents compound reasons and CPU-only FUSED boundary.

## Decisions Made

- `NONE` summaries are explicit unsupported/non-compound metadata, not an execution path.
- `GpuCompoundPatternDetector` may inspect source operation summaries and DAG specs, but it does not replace DAG legality.

## Deviations from Plan

None - plan executed exactly as written.

---

**Total deviations:** 0 auto-fixed.
**Impact on plan:** No scope change.

## Issues Encountered

None.

## Verification

- `./gradlew test --tests backend.accelerator.lowering.GpuCompoundPatternDetectorTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest` - passed.
- `rg -n "detectsLinearBiasActivationSummaryFromExistingMatmulSpec|detectsElementwiseChainSummaryFromDagNodes|rejectsCpuFusedOpTypeForGpuCompoundLowering|compoundFusedRowsExposeStablePhaseTwelveReasons|LINEAR_BIAS_ACTIVATION|CPU_FUSED_UNSUPPORTED" src/test/java/backend/accelerator/lowering` - passed.
- `rg -n "enum GpuCompoundPatternType|record GpuCompoundRegionSummary|class GpuCompoundPatternDetector|record GpuCompoundLoweringArtifact|compoundSummary|CPU_FUSED_UNSUPPORTED|LINEAR_BIAS_ACTIVATION|ELEMENTWISE_CHAIN|REDUCTION_ADJACENT" src/main/java/backend/accelerator/lowering` - passed.
- `rg -n "CPU_FUSED_OPERATION_UNSUPPORTED|COMPOUND_PATTERN_UNSUPPORTED|COMPOUND_REGION_SHORTENED" src/main/java/backend/accelerator/lowering/GpuLoweringUnsupportedReason.java docs/gpu-lowering-coverage.md` - passed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for `12-02`: linear+bias+activation can now use the shared summary contract.

## Self-Check: PASSED

---
*Phase: 12-fused-gpu-region-execution*
*Completed: 2026-04-30*
