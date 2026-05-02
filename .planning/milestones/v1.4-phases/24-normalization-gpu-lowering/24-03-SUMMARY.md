# Summary 24-03: Planner Admission And Coverage Matrix Update

**Status:** Complete
**Date:** 2026-05-01

## Completed

- Moved `LAYER_NORM` and `RMS_NORM` matrix rows to `SUPPORTED` for Metal and CUDA legal dense FLOAT32 cases.
- Registered normalization targets as native-executable in `GpuTargetCoverageTruth`.
- Updated Metal and CUDA planner legality to admit legal normalization and reject unsupported dtype, layout, rank, or shape with stable reason prefixes.
- Updated stale fallback tests and fixtures so normalization is no longer treated as `DEFERRED_FUSED_REGION` for the legal case.
- Updated GPU lowering docs to describe supported normalization plus remaining unsupported variants.

## Verification

Passed:

```bash
./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests GpuCoverageSummaryTest --tests CompiledGraphTraceTest
```

Additional report fixture coverage passed:

```bash
./gradlew test --tests BenchmarkSessionTest --tests GpuCoverageSummaryTest --tests CompiledGraphTraceTest --tests GpuHotPathCoverageTargetsTest
```

## Deviations from Plan

None.
