# Summary 36-01: Scatter/Index-Gradient Semantics Contract

**Status:** Completed
**Date:** 2026-05-02

## Implemented

- Tightened `GpuTargetSemanticsContract` for `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD`.
- Marked Phase 36 index write/gradient contracts as planner-admission blocked until duplicate-index accumulation and static bounds semantics are proven.
- Updated `GpuLoweringCoverageMatrix` notes for stable `UNSUPPORTED_DUPLICATE_INDEX` reasons that mention duplicate accumulation, bounds, and residency blockers.
- Added Metal and CUDA planner tests for `GATHER_GRAD` and `TAKE_ALONG_AXIS_GRAD` stable duplicate-index rejection.
- Added CPU parity fixtures for repeated-index scatter/add behavior, direct `GATHER_GRAD`, and duplicate-index `TAKE_ALONG_AXIS` backward accumulation.
- Updated GPU lowering and Metal backend docs so forward index support is not counted as index-gradient support.

## Verification

```bash
./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuTargetSemanticsContractTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests ScatterAddExecutionTest --tests GatherExecutionTest --tests TakeAlongAxisExecutionTest
git diff --check
```

All commands passed.

## Notes

- No native Metal support is claimed yet for `SCATTER_ADD`, `GATHER_GRAD`, or `TAKE_ALONG_AXIS_GRAD`.
- Phase 36-02 owns the implementation decision: narrow native/lowered support only if duplicate-index parity is proven; otherwise stable rejection remains the correct behavior.
