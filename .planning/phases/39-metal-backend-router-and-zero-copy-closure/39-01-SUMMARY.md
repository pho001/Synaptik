# Plan 39-01 Summary: Router Policy And Cost Evidence Model

**Status:** Complete
**Completed:** 2026-05-02 14:51 CEST
**Requirement Coverage:** METALROUTER-01, METALROUTER-03

## What Changed

- Added Metal route vocabulary:
  - `MetalExecutionRoute`
  - `MetalRouteReasonCode`
  - `MetalRouteDecision`
  - `MetalExecutionRouter`
- Wired prepare-time route decisions into `PreparedMetalExecutable` after the static transport plan is built.
- Kept router scope inside already-selected `GPU_METAL` regions; backend selection remains unchanged.
- Current supported buffer-binding Metal execution reports `MPS_GRAPH` with `MPS_GRAPH_SELECTED`.
- Tensor-array and required-unavailable transport plans map to stable route decisions.
- Added Metal route trace attributes:
  - `metalExecutionRoute`
  - `metalRouteReasonCode`
  - `metalRouteRejectedRoutes`
  - `metalRouteEstimatedCost`
  - `metalRouteEstimatedCopyCost`
  - route evidence booleans for bridge, executable, buffer ABI, custom kernel, and native copy cost.

## Important Semantics

- The router describes how an already selected Metal region will execute; it does not choose graph partitions.
- `CUSTOM_KERNEL` is visible as a rejected route for now. The actual custom-kernel SPI is intentionally left for 39-02.
- Native copy cost remains explicit and unknown in the route model through `estimatedCopyCost=-1` and `metalRouteNativeCopyCostKnown=false`.

## Verification

Passed:

```bash
./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest
./gradlew test --tests PreparedExecutionBuildTest
./gradlew test --tests BenchmarkSessionTest --tests GpuCoverageSummaryTest
./gradlew classes
git diff --check
```
