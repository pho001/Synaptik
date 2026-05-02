# Summary 26-03: Gather Take Scatter Support-Or-Rejection

**Status:** Complete
**Date:** 2026-05-02

## Completed

- Kept forward `GATHER` and `TAKE_ALONG_AXIS` unsupported with `DAG_PRIMITIVE_UNSUPPORTED` until native DAG and backend execution are implemented.
- Kept `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD` unsupported with duplicate-index accumulation reasons.
- Added Metal/CUDA legality tests for representative gather, take-along-axis, and scatter-add rejected cases.
- Preserved existing CPU parity tests for gather, take-along-axis, scatter-add, duplicate-index-like gradient behavior, ignore-index loss, weighted loss, and reduction modes.

## Evidence

- `MetalRegionLowererTest.phaseTwentySixIndexFamilyUsesStableCoverageReasons`
- `CudaRegionLowererTest.phaseTwentySixIndexFamilyUsesStableCoverageReasons`
- `GatherExecutionTest`
- `TakeAlongAxisExecutionTest`
- `ScatterAddExecutionTest`
- `IndexTargetNllLossExecutionTest`
- `IndexTargetCrossEntropyLossExecutionTest`
- `IgnoreIndexLossExecutionTest`
- `WeightedIndexLossExecutionTest`
- `IndexLossReductionExecutionTest`

## Verification

Passed in focused Phase 26 gate:

```bash
./gradlew test --tests GatherExecutionTest --tests TakeAlongAxisExecutionTest --tests ScatterAddExecutionTest --tests IndexTargetNllLossExecutionTest --tests IndexTargetCrossEntropyLossExecutionTest --tests IgnoreIndexLossExecutionTest --tests WeightedIndexLossExecutionTest --tests IndexLossReductionExecutionTest
```
