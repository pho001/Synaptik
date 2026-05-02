# Summary 23-02: Metal And CUDA Native Reduction Execution

**Status:** Complete
**Date:** 2026-05-01

## Completed

- Added Metal MPSGraph execution cases for `SUM`, `MEAN`, `REDUCE_MIN`, and `REDUCE_MAX`.
- Reshaped Metal reduction outputs back to the lowered DAG output contract so `keepDims` stays explicit across the native boundary.
- Added CUDA dense FLOAT32 rank 1-4 reduction kernel support for sum, mean, min, and max.
- Added CUDA reduction axis and keep-dims decoding from the shared scalar metadata contract.
- Added capability-gated Metal buffer execution coverage for keep-dims reduction output shape and value parity.

## Verification

Passed:

```bash
./gradlew test --tests PreparedExecutionBuildTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest
./gradlew metalTest
```

## Deviations from Plan

None - plan executed exactly as written.
