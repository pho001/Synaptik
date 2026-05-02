# Plan 38-01 Summary: Backward Coverage Matrix And Legality Update

**Status:** Complete
**Completed:** 2026-05-02
**Requirement Coverage:** METALTRAIN-01, METALTRAIN-02 partial, METALTRAIN-03 partial

## Completed

- Extended `GpuTargetCoverageTruth` with explicit training/backward rows:
  - `SOFTMAX_GRAD`
  - `LOG_SOFTMAX_GRAD`
  - `REDUCE_MIN_GRAD`
  - `REDUCE_MAX_GRAD`
  - `MIN_GRAD`
  - `MAX_GRAD`
  - `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD`
  - conv/pool backward variants
- Registered Metal native-executable truth only for backward rows with existing prepared-execution/parity evidence.
- Kept CUDA backward-adjacent rows as matrix-supported-only or explicit rejection until CUDA-native evidence exists.
- Kept conv/pool backward as `CAPABILITY_MISSING`, index gradients/scatter as `UNSUPPORTED_DUPLICATE_INDEX`, and index-target loss gradient as `UNSUPPORTED_INDEX_SEMANTICS`.
- Added tests proving backward support is tracked independently from forward support.
- Updated docs to make the per-backward-op contract explicit.

## Verification

```bash
./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuCoverageSummaryTest --tests backend.metal.lowering.MetalRegionLowererTest
./gradlew classes
git diff --check
```

All passed.

## Next

Plan 38-02 should add or verify runtime/native execution evidence for the positive Metal backward rows and keep unsupported backward rows visible.
