# Plan 38-02 Summary: Backward Native Or Stable-Rejection Execution

**Status:** Complete
**Completed:** 2026-05-02
**Requirement Coverage:** METALTRAIN-01 partial, METALTRAIN-02

## Completed

- Added explicit Apple shim trace coverage for supported Metal backward rows:
  - `SOFTMAX_GRAD`
  - `LOG_SOFTMAX_GRAD`
  - `REDUCE_MIN_GRAD`
  - `REDUCE_MAX_GRAD`
  - `MIN_GRAD`
  - `MAX_GRAD`
- The trace gate requires `BUFFER_BINDING` and `BUFFER_BINDING_AVAILABLE`, so tensor-array replay is not accepted as native buffer evidence.
- Added a required-buffer rejection assertion for `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD`; current bridge support is planner/matrix-level only and rejects native buffer execution with `BRIDGE_UNAVAILABLE`.
- Updated `GpuTargetCoverageTruth` and docs so Metal SDPA backward is not marked `NATIVE_EXECUTABLE` until the buffer bridge supports the backward SDPA DAG.
- Kept conv/pool backward, index gradients, scatter, and index-target loss gradients as explicit unsupported/backward blockers.

## Verification

```bash
./gradlew test --tests PreparedExecutionBuildTest --tests SoftmaxExecutionTest --tests LogSoftmaxExecutionTest --tests MinMaxReductionExecutionTest --tests AttentionExecutionTest
./gradlew test --tests GpuCoverageSummaryTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest
./gradlew metalTest
git diff --check
```

All passed.

## Residual Scope

- `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD` still needs native buffer bridge support before it can become `NATIVE_EXECUTABLE`.
- Conv/pool backward, index gradients, scatter, and index-target loss gradients remain explicit CPU/unsupported boundaries.
