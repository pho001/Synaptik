# 32-04 Summary: Coverage Docs And Regression Closure

**Status:** Completed
**Completed:** 2026-05-02

## What Changed

- Added `gather_take_small` to the standard workload catalog and GPU hot-path target registry.
- Added a hard Metal coverage policy for `gather_take_small` requiring:
  - native buffer binding,
  - lowered primitive evidence,
  - no CPU materialization,
  - no CPU fallback,
  - no tensor-array fallback,
  - non-rejected `INT32` dtype residency evidence.
- Kept CUDA `gather_take_small` as a visible blocker with `CAPABILITY_MISSING` expectations.
- Updated coverage docs, Metal backend docs, and troubleshooting docs with the scoped INT32 index contract and non-goals.
- Marked `METALINTIDX-01/02/03` complete.

## Verification

```bash
./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests BenchmarkSuiteSessionTest --tests SourceTreeHygieneTest --tests StandardWorkloadsTest
./gradlew metalTest
git status --short profiles/platform
```

All tests passed. Profile artifacts remain dirty and unstaged by design.

## Remaining Work

- Phase 33 can now build on the index-forward path by improving GPU-side layout repair for non-dense/view cases.
- Scatter, gather/take gradients, and index-target loss remain deferred to their planned phases.
