# Verification 26: Loss-Adjacent And Indexing GPU Coverage

**Status:** Passed
**Date:** 2026-05-02

## Result

Phase 26 satisfies GPULOSSIDX-01, GPULOSSIDX-02, and GPULOSSIDX-03 through explicit support-or-rejection evidence. No loss/indexing operation was marked `SUPPORTED` because the backend-neutral DAG ABI and native Metal/CUDA shims do not yet implement index/loss primitives.

## Requirement Mapping

| Requirement | Status | Evidence |
|-------------|--------|----------|
| GPULOSSIDX-01 | Complete | Loss-adjacent and index ops have explicit Metal/CUDA rows with stable rejection reasons tied to missing primitive, index semantics, or duplicate-index accumulation. |
| GPULOSSIDX-02 | Complete | Boundary test proves a legal Metal `LOG_SOFTMAX` region stays GPU-selected before CPU-owned `TAKE_ALONG_AXIS`; dtype residency remains diagnostic and separate from native compute support. |
| GPULOSSIDX-03 | Complete | CPU parity tests cover gather, take-along-axis, scatter-add, duplicate-index gradient behavior, ignore-index loss, weighted loss, and reduction modes; reports expose loss/index reasons. |

## Executed Gates

Passed:

```bash
./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuTargetSemanticsContractTest
./gradlew test --tests PreparedExecutionBuildTest.phaseTwentySixUnsupportedTakeBoundaryKeepsPrecedingMetalLogSoftmaxRegionSelected --tests backend.metal.lowering.MetalRegionLowererTest.phaseTwentySixIndexFamilyUsesStableCoverageReasons --tests backend.cuda.lowering.CudaRegionLowererTest.phaseTwentySixIndexFamilyUsesStableCoverageReasons
./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuTargetSemanticsContractTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests GatherExecutionTest --tests TakeAlongAxisExecutionTest --tests ScatterAddExecutionTest --tests IndexTargetNllLossExecutionTest --tests IndexTargetCrossEntropyLossExecutionTest --tests IgnoreIndexLossExecutionTest --tests WeightedIndexLossExecutionTest --tests IndexLossReductionExecutionTest
./gradlew metalTest
git diff --check
```

Native CUDA execution was not run because `nvcc` is unavailable locally.

## Residual Risk

- Forward `GATHER` and `TAKE_ALONG_AXIS` still need real DAG ABI and native backend execution before they can become supported rows.
- Scatter and index-gradient operations remain blocked on duplicate-index accumulation semantics.
- Index-target loss remains blocked on INT32 targets, bounds behavior, ignore-index masking, reduction denominator semantics, and per-class gradient scatter behavior.
