# Summary 36-04: Trace Report And Coverage Closure

**Status:** Completed
**Date:** 2026-05-02

## Implemented

- Added `scatter_index_gradient_small` as a separate Phase 36 hot-path coverage target.
- Added `ScatterIndexGradientWorkloadSpec` to exercise `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD` without reusing the forward `gather_take_small` target.
- Updated coverage expectations so `scatter_index_gradient_small` is a visible-blocker target requiring `UNSUPPORTED_DUPLICATE_INDEX` or a named index-write/gradient operation in reports.
- Hardened `GpuTargetCoverageTruth` tests so forward `GATHER` / `TAKE_ALONG_AXIS` native status cannot imply native index-gradient or scatter support.
- Updated Metal, GPU coverage, and troubleshooting docs with the split between native forward index support and Phase 36 index-write/index-gradient rejection.
- Kept local `profiles/platform/...` tuning artifacts unstaged.

## Verification

```bash
./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests StandardWorkloadsTest
./gradlew test --tests SourceTreeHygieneTest
./gradlew classes
./gradlew metalTest
git diff --check
```

All commands passed.

## Deviations from Plan

No functional deviation. Because Phase 36 intentionally chose stable rejection over native execution, the regression gate for this target is a visible-blocker gate rather than a hard native-buffer gate.

## Next Phase Readiness

Phase 37 can now plan loss-adjacent Metal lowering with Phase 36 index-write and index-gradient blockers explicitly represented in reports.
