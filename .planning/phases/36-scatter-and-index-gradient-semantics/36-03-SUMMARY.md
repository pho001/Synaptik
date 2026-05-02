# Summary 36-03: Gradient Parity And Duplicate-Index Gates

**Status:** Completed
**Date:** 2026-05-02

## Implemented

- Added CPU parity fixtures for Phase 36 index-write/index-gradient edge cases:
  - `SCATTER_ADD` repeated axis values remain scoped to their logical lane.
  - `GATHER_GRAD` repeated axis values remain scoped to their logical lane.
  - `TAKE_ALONG_AXIS_GRAD` duplicate indices within one lane accumulate into the same source position.
- Added execution-time out-of-bounds rejection tests for `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD`.
- Added prepared-execution gates proving Metal does not silently claim `SCATTER_ADD`, `GATHER_GRAD`, or `TAKE_ALONG_AXIS_GRAD`.
- Verified a supported Metal producer can still be selected while an adjacent `SCATTER_ADD` remains an explicit CPU prepared step with stable `UNSUPPORTED_DUPLICATE_INDEX` diagnostics.
- Added Metal planner gates proving gradient index paths reject static OOB and non-leaf/unproven index tensors before the generic duplicate-index blocker.

## Verification

```bash
./gradlew test --tests ScatterAddExecutionTest --tests GatherExecutionTest --tests TakeAlongAxisExecutionTest
./gradlew test --tests PreparedExecutionBuildTest --tests backend.metal.lowering.MetalRegionLowererTest
./gradlew metalTest
git diff --check
```

All commands passed.

## Deviations from Plan

The plan expected prepared-execution tests for supported index-gradient regions preserving Metal ownership. Phase 36-02 intentionally kept all index-write/index-gradient Metal paths as stable rejection because duplicate accumulation parity is not proven. This wave therefore verified the safe equivalent: supported adjacent producers remain selectable where legal, while index-write/index-gradient operations stay CPU-owned with explicit rejection reasons.

## Next Phase Readiness

Phase 36-04 can now close trace/report coverage with stable distinctions between forward index support and index-write/index-gradient rejection.
