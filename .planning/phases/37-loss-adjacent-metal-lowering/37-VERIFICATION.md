# Phase 37 Verification: Loss-Adjacent Metal Lowering

**Status:** Verified
**Verified:** 2026-05-02
**Requirements:** METALLOSS-01, METALLOSS-02, METALLOSS-03

## Requirement Evidence

| Requirement | Status | Evidence |
|---|---|---|
| METALLOSS-01 | Complete | `MetalLossSemantics` locks the dense loss contract; `AcceleratorSubgraphLowerer` lowers dense `NLL_LOSS` and dense `CROSS_ENTROPY_LOSS`; `GpuTargetCoverageTruth` marks Metal dense loss rows native-executable; `dense_loss_small` requires hard native report evidence. |
| METALLOSS-02 | Complete | Index-target CE/NLL and gradient variants remain unsupported with `UNSUPPORTED_INDEX_SEMANTICS`; tests keep `CROSS_ENTROPY_LOSS_INDICES` and `CROSS_ENTROPY_LOSS_INDICES_GRAD` CPU-owned until INT32 target, bounds, ignore-index, weight, denominator, and scatter semantics are proven. |
| METALLOSS-03 | Complete | Training traces prove dense Metal loss forward remains GPU-owned without internal `CPU_CONSUMER` materialization; reports distinguish `dense_loss_small` native support from `cross_entropy_small` index-target blockers. |

## Verification Commands

```bash
./gradlew test --tests PreparedExecutionBuildTest --tests IndexTargetCrossEntropyLossExecutionTest --tests IndexTargetNllLossExecutionTest
./gradlew test --tests GpuCoverageSummaryTest --tests CompiledGraphTraceTest
./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests StandardWorkloadsTest
./gradlew test --tests BenchmarkSuiteSessionTest
./gradlew test --tests SourceTreeHygieneTest
./gradlew classes
./gradlew metalTest
git diff --check
```

## Residual Scope

- `CROSS_ENTROPY_LOSS_INDICES`, `CROSS_ENTROPY_LOSS_INDICES_GRAD`, and related index-target loss training paths remain visible CPU boundaries.
- CUDA dense loss remains `DAG_PRIMITIVE_UNSUPPORTED`.
- Dense Metal loss support is not generalized beyond the locked `FLOAT32`, dense zero-offset, rank 1..4, mean-reduced scalar contract.
