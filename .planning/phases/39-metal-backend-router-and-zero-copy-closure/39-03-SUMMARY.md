# Plan 39-03 Summary: Output Buffer And Native Copy Closure

**Status:** Complete
**Completed:** 2026-05-02 15:00 CEST
**Requirement Coverage:** METALROUTER-02, METALROUTER-03

## What Changed

- Added `MetalNativeCopyStrategy` with:
  - `MPSGRAPH_RESULT_COPY`
  - `TRUE_OUTPUT_BUFFER_WRITE`
  - `UNKNOWN_OR_UNPROVEN`
- Extended `MetalMpsBridgeExecutionStats` with `nativeCopyStrategy` and `outputBufferWriteProven()`.
- Classified current FFM MPSGraph execution as `MPSGRAPH_RESULT_COPY`; no zero-copy output-buffer write is claimed.
- Added Metal trace attributes:
  - `metalNativeCopyStrategy`
  - `metalOutputBufferWriteProven`
- Extended accelerator text/JSON reports with `nativeCopyStrategies`.
- Extended GPU coverage summaries with `nativeCopyStrategyCounts`.
- Updated Metal/architecture docs so `nativeDeviceCopyNs` is explicitly classified as MPSGraph-result-copy evidence, not Java copy-back and not proven output-buffer aliasing.

## Important Semantics

- Current buffer-binding execution remains device-owned from Java's residency perspective, but native output-buffer write behavior is not proven.
- The conservative path stays explicit as `MPSGRAPH_RESULT_COPY`.
- `TRUE_OUTPUT_BUFFER_WRITE` is reserved for a future sentinel/alias proof that validates direct writes into caller-provided output buffers.

## Verification

Passed:

```bash
./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.bridge.MetalMpsBridgeExecutionStatsTest
./gradlew test --tests BenchmarkSessionTest --tests GpuCoverageSummaryTest
./gradlew metalTest
git diff --check
```
