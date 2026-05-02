# Summary 22-02: Semantics Contracts For Target Families

**Status:** Complete
**Date:** 2026-05-01

## Completed

- Added backend-neutral `GpuTargetSemanticsContract` records for v1.4 target families.
- Locked reduction contracts around descriptor axis, keep-dims output shape behavior, dtype/rank/layout constraints, and numerical expectations.
- Kept forward SDPA planner admission explicitly blocked until scale, mask, rank, and backward-interaction semantics are verified.
- Added BOOL compare-output contract text that distinguishes BOOL input residency from producing device-resident BOOL outputs.

## Verification

Passed:

```bash
./gradlew test --tests GpuTargetSemanticsContractTest --tests GpuCoverageSummaryTest
```

## Remaining

- 22-03 must connect these contracts to deterministic workload coverage baselines.
