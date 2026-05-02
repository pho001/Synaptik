# Summary 26-04: Loss-Adjacent Coverage Gates And Documentation Closure

**Status:** Complete
**Date:** 2026-05-02

## Completed

- Updated benchmark/trace/report fixtures from the old `UNSUPPORTED_DTYPE` index-target loss reason to `UNSUPPORTED_INDEX_SEMANTICS`.
- Kept loss/index rows visible in coverage summaries and renderer evidence.
- Updated Phase 26 documentation and roadmap/state artifacts.
- Preserved profile artifact hygiene; local profile files remain unstaged and are not Phase 26 evidence.

## Evidence

- `CompiledGraphTraceTest`
- `GpuCoverageSummaryTest`
- `BenchmarkSessionTest`
- `docs/gpu-lowering-coverage.md`

## Verification

Passed:

```bash
./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuTargetSemanticsContractTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests GatherExecutionTest --tests TakeAlongAxisExecutionTest --tests ScatterAddExecutionTest --tests IndexTargetNllLossExecutionTest --tests IndexTargetCrossEntropyLossExecutionTest --tests IgnoreIndexLossExecutionTest --tests WeightedIndexLossExecutionTest --tests IndexLossReductionExecutionTest
```
