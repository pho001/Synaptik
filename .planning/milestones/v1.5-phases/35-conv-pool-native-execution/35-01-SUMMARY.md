# Summary 35-01: Conv/Pool Semantic And Capability Contract

**Status:** Completed
**Date:** 2026-05-02

## Implemented

- Added `MetalConvPoolSemantics` as the Metal planner-side semantic gate for forward `CONV2D`, `CONV2D_GEMM`, `MAX_POOL2D`, and `AVG_POOL2D`.
- Replaced generic Metal conv/pool coverage-matrix fallback diagnostics with precise semantic reasons before native support is enabled.
- Classified legal direct Conv2D and pool cases as `CAPABILITY_MISSING` with scoped family/target metadata instead of claiming support.
- Added explicit rejection coverage for:
  - non-`FLOAT32` conv/pool inputs,
  - non-dense conv/pool inputs,
  - invalid rank/shape contracts,
  - grouped/depthwise Conv2D,
  - dilated Conv2D,
  - `AVG_POOL2D countIncludePad=true`,
  - `CONV2D_GEMM` staying CPU-owned until im2col/GEMM layout semantics exist in the accelerator DAG.
- Confirmed CUDA conv/pool behavior remains capability-gated.

## Verification

```bash
./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest
./gradlew test --tests Conv2dExecutionTest --tests Pool2dExecutionTest
./gradlew classes
git diff --check
```

All commands passed.

## Notes

- No Metal conv/pool coverage row was changed to supported in this wave.
- Native execution remains blocked until Wave 35-02 and Wave 35-03 add bridge/runtime parity.
