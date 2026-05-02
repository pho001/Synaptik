# Summary 35-03: Max/Avg Pool Forward Native/Lowered Execution

**Status:** Completed
**Date:** 2026-05-02

## Implemented

- Added `MAX_POOL2D` and `AVG_POOL2D` to the shared accelerator DAG ABI as node types `55` and `56`.
- Lowered legal Metal forward pool nodes into the accelerator DAG with compact kernel/stride/padding metadata.
- Added native MPSGraph max/avg pooling execution in the Apple Metal shim for dense `FLOAT32` NCHW inputs.
- Promoted scoped Metal `MAX_POOL2D` and `AVG_POOL2D` coverage rows to `SUPPORTED`.
- Registered `max_pool2d_small` as a Metal native coverage target while keeping CUDA visible-blocker expectations.
- Kept pooling backward and `AVG_POOL2D countIncludePad=true` capability-gated.

## Verification

```bash
./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests PreparedExecutionBuildTest --tests Pool2dExecutionTest --tests GpuHotPathCoverageTargetsTest
./gradlew metalTest
```

All commands passed.

## Notes

- Native parity was verified for representative max-pool and avg-pool buffer execution.
- Conv/pool backward paths remain out of scope for this phase and are still capability-gated.
