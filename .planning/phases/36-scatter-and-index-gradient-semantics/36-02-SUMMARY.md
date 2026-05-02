# Summary 36-02: Native Or Stable-Rejection Scatter Execution

**Status:** Completed
**Date:** 2026-05-02

## Implemented

- Evaluated the Phase 36 operations against the current Metal native stack and did not claim native support because duplicate-index accumulation parity is not proven.
- Added `MetalIndexWriteSemantics` as a planner-side router for `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD`.
- Metal now validates the narrow candidate contract before duplicate rejection:
  - dense `FLOAT32` value/source/output tensors,
  - dense static `INT32` index tensors,
  - rank 1..4,
  - legal axis/shape contracts,
  - static in-bounds index values.
- Legal candidates still reject with stable `UNSUPPORTED_DUPLICATE_INDEX` details instead of hiding CPU fallback behind a supported matrix row.
- CUDA remains unchanged and capability/reason gated through the shared matrix.
- Updated docs to explain that forward index support is separate from index-write/index-gradient support.

## Verification

```bash
./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests PreparedExecutionBuildTest
./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest
./gradlew metalTest
git diff --check
```

All commands passed.

## Notes

- This wave intentionally chose stable rejection over native execution. A future native/custom Metal implementation must prove duplicate accumulation parity before changing matrix status to supported.
