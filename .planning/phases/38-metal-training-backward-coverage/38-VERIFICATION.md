# Phase 38 Verification: Metal Training Backward Coverage

**Status:** Verified
**Verified:** 2026-05-02
**Requirements:** METALTRAIN-01, METALTRAIN-02, METALTRAIN-03

## Requirement Evidence

| Requirement | Status | Evidence |
|---|---|---|
| METALTRAIN-01 | Complete | `GpuTargetCoverageTruth` separates backward execution truth from forward support. Metal `SOFTMAX_GRAD`, `LOG_SOFTMAX_GRAD`, `REDUCE_MIN_GRAD`, `REDUCE_MAX_GRAD`, `MIN_GRAD`, and `MAX_GRAD` are native-executable only after prepared native buffer trace and CPU parity tests in `PreparedExecutionBuildTest`. |
| METALTRAIN-02 | Complete | Unsupported or not-yet-native backward families remain explicit: SDPA backward is matrix/planner-supported but rejected by required buffer execution with `BRIDGE_UNAVAILABLE`; conv/pool backward remains `CAPABILITY_MISSING`; index gradients and scatter remain `UNSUPPORTED_DUPLICATE_INDEX`; index-target loss gradients remain `UNSUPPORTED_INDEX_SEMANTICS`. |
| METALTRAIN-03 | Complete | `GpuCoverageSummary` reports `gradientPublicationMaterializationCount` and `internalCpuMaterializationCount`; `GpuCoverageGatePolicy.trainingHotPathTarget(...)` allows bounded `GRADIENT_PUBLICATION` while rejecting hidden internal CPU materialization, tensor-array replay, and CPU fallback. |

## Supported Metal Backward Rows

| Operation | Evidence |
|---|---|
| `SOFTMAX_GRAD` | Native `BUFFER_BINDING` trace in `gpuMetalSupportedBackwardRowsUseNativeBufferBindingWithExplicitAppleShim`; CPU gradient parity in `gpuMetalBackwardSoftmaxGradCanPrepareAndMatchCpuGradients`. |
| `LOG_SOFTMAX_GRAD` | Native `BUFFER_BINDING` trace in `gpuMetalSupportedBackwardRowsUseNativeBufferBindingWithExplicitAppleShim`; CPU gradient parity in `gpuMetalBackwardLogSoftmaxGradCanPrepareAndMatchCpuGradients`. |
| `REDUCE_MIN_GRAD` | Native `BUFFER_BINDING` trace in `gpuMetalSupportedBackwardRowsUseNativeBufferBindingWithExplicitAppleShim`; CPU gradient parity in `gpuMetalBackwardReduceMinGradCanPrepareAndMatchCpuGradients`. |
| `REDUCE_MAX_GRAD` | Native `BUFFER_BINDING` trace in `gpuMetalSupportedBackwardRowsUseNativeBufferBindingWithExplicitAppleShim`; CPU gradient parity in `gpuMetalBackwardReduceMaxGradCanPrepareAndMatchCpuGradients`. |
| `MIN_GRAD` | Native `BUFFER_BINDING` trace in `gpuMetalSupportedBackwardRowsUseNativeBufferBindingWithExplicitAppleShim`; CPU gradient parity in `gpuMetalBackwardMinGradCanPrepareAndMatchCpuGradients`. |
| `MAX_GRAD` | Native `BUFFER_BINDING` trace in `gpuMetalSupportedBackwardRowsUseNativeBufferBindingWithExplicitAppleShim`; CPU gradient parity in `gpuMetalBackwardMaxGradCanPrepareAndMatchCpuGradients`. |

## Report Gate Evidence

- `training_dense_loss_small`, `training_reduction_chain_small`, and `training_layer_norm_small` use training policies that allow gradient publication but require zero internal CPU materialization.
- `training_transformer_block_hot_path` and `training_cross_entropy_small` are visible blocker targets until SDPA backward and index-target loss gradients have native execution evidence.
- Text and JSON benchmark reports expose the new materialization split and policy budgets.

## Verification Commands

Passed:

```bash
./gradlew test --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest
./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests StandardWorkloadsTest
./gradlew test --tests SourceTreeHygieneTest
./gradlew classes
./gradlew metalTest
git diff --check
```

## Residual Scope

- Metal SDPA backward is not native-buffer executable until the bridge supports the backward SDPA DAG.
- Conv/pool backward native execution remains future work.
- Index gradients, scatter, and index-target loss gradients remain blocked on duplicate-index, bounds, and reduction-denominator semantics.
- This phase does not claim full end-to-end training on Metal; it locks verified backward rows and makes remaining CPU exits report-visible.
