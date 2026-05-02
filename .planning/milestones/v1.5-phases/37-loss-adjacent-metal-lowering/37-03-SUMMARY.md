# Plan 37-03 Summary: Loss Backward And Training Boundary Gates

**Status:** Complete
**Completed:** 2026-05-02
**Requirement Coverage:** METALLOSS-02 partial, METALLOSS-03 partial

## Completed

- Added training-mode prepared-execution coverage proving scoped dense `CROSS_ENTROPY_LOSS` forward remains selected as a Metal step in `FORWARD_BACKWARD`.
- Added trace coverage proving dense Metal loss forward does not create an internal `CPU_CONSUMER` materialization boundary for the loss node.
- Verified dense CE gradients still match CPU when the forward loss step is Metal-owned.
- Added explicit index-target training boundary coverage:
  - `CROSS_ENTROPY_LOSS_INDICES` remains CPU-owned;
  - `CROSS_ENTROPY_LOSS_INDICES_GRAD` remains CPU-owned;
  - both report stable `UNSUPPORTED_INDEX_SEMANTICS` planner blockers.
- Kept gradient publication separate from hidden internal CPU fallback; tests reject `CPU_FALLBACK` materialization in the dense-loss training trace.

## Verification

```bash
./gradlew test --tests PreparedExecutionBuildTest --tests IndexTargetCrossEntropyLossExecutionTest --tests IndexTargetNllLossExecutionTest
./gradlew test --tests GpuCoverageSummaryTest --tests CompiledGraphTraceTest
./gradlew metalTest
git diff --check
```

All passed.

## Next

Phase 37-04 should close report/coverage targets and docs so dense loss support is counted separately from index-target loss blockers.
