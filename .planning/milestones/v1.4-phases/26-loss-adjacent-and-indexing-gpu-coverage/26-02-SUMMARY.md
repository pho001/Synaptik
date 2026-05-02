# Summary 26-02: Device Residency For Index Inputs And Boundaries

**Status:** Complete
**Date:** 2026-05-02

## Completed

- Added prepared-execution evidence that a legal Metal `LOG_SOFTMAX` producer remains selected before a CPU-owned `TAKE_ALONG_AXIS` boundary.
- The boundary rejection reports `DAG_PRIMITIVE_UNSUPPORTED`, `family=INDEX_SCATTER_GATHER`, and the concrete operation name.
- Existing dtype residency evidence remains distinct from native index compute support.

## Evidence

- `PreparedExecutionBuildTest.phaseTwentySixUnsupportedTakeBoundaryKeepsPrecedingMetalLogSoftmaxRegionSelected`

## Verification

Passed in focused Phase 26 gate:

```bash
./gradlew test --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest
```
