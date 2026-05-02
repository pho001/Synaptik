# 32-02 Summary: Forward Gather Take Lowering And Native Execution

**Status:** Completed
**Completed:** 2026-05-02

## What Changed

- Added shared accelerator DAG ABI node types for `GATHER` and `TAKE_ALONG_AXIS` with axis metadata in `scalarValueBits`.
- Extended shared lowering so legal forward gather/take nodes lower into backend-neutral DAG primitives.
- Promoted Metal `GATHER` and `TAKE_ALONG_AXIS` coverage to supported for dense `FLOAT32` value/output tensors with `INT32` index inputs.
- Kept CUDA forward gather/take unsupported with `CAPABILITY_MISSING`.
- Added Metal planner legality checks for dtype, dense layout, rank, axis, index shape, and output shape.
- Added native MPSGraph shim execution:
  - `TAKE_ALONG_AXIS` maps to `gatherAlongAxis`.
  - `GATHER` expands the reduced index tensor on the gathered axis, runs `gatherAlongAxis`, then squeezes the public output axis.
- Added native INT32 dtype descriptor handling for input buffers in the Metal shim.

## Verification

```bash
./gradlew classes
./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests GpuCoverageSummaryTest --tests GpuTargetSemanticsContractTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest
./gradlew metalTest
```

All commands passed.

## Remaining Work

- Phase 32-03 still needs broader parity/rejection coverage for bounds, layout, and representative rank/axis cases.
- `GATHER_GRAD`, `TAKE_ALONG_AXIS_GRAD`, and `SCATTER_ADD` remain unsupported until duplicate-index accumulation semantics are implemented.
- Index-target CE/NLL remains outside Phase 32.
