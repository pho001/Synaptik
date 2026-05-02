# Plan 38-03 Summary: Training Trace And Gradient Publication Gates

**Status:** Complete
**Completed:** 2026-05-02 14:32 CEST

## What Changed

- Added `GpuCoverageSummary.BackendCoverage` helpers for:
  - `gradientPublicationMaterializationCount`
  - `internalCpuMaterializationCount`
- Extended `GpuCoverageGatePolicy` with separate budgets for:
  - total CPU materialization
  - avoidable internal CPU materialization
  - gradient-publication materialization
- Added `GpuCoverageGatePolicy.trainingHotPathTarget(...)` for training paths that may legitimately publish gradients while still rejecting hidden CPU exits.
- Updated coverage regression gates to fail internal materialization separately from accepted `GRADIENT_PUBLICATION`.
- Added `training_*` hot-path workloads and target expectations:
  - `training_transformer_block_hot_path`
  - `training_dense_loss_small`
  - `training_reduction_chain_small`
  - `training_layer_norm_small`
  - `training_cross_entropy_small`
- Updated text and JSON benchmark reports to expose the new materialization counters and policy budgets.

## Important Semantics

- `GRADIENT_PUBLICATION` is a public training boundary and can be allowed by training policies.
- `CPU_CONSUMER`, `CPU_FALLBACK`, `PUBLIC_DATA_ACCESS`, and `ACCELERATOR_PREPARED_INPUT` count as internal CPU materialization for coverage gates.
- Supported training targets require zero internal CPU materialization, no tensor-array replay, and no CPU fallback.
- Unsupported training targets still pass only when stable blocker reasons are visible, such as SDPA backward bridge unavailability or index-target loss gradient semantics.

## Verification

Passed:

```bash
./gradlew test --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageSummaryTest --tests CompiledGraphTraceTest
./gradlew test --tests BenchmarkSuiteSessionTest --tests StandardWorkloadsTest
./gradlew classes
./gradlew test --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageSummaryTest --tests CompiledGraphTraceTest --tests BenchmarkSuiteSessionTest --tests StandardWorkloadsTest
```

`git diff --check` is tracked as the final hygiene gate for the phase commit.
