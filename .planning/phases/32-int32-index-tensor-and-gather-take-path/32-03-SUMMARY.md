# 32-03 Summary: Bounds Layout Parity And Visible Rejections

**Status:** Completed
**Completed:** 2026-05-02

## What Changed

- Added Metal planner bounds proof for forward `GATHER` and `TAKE_ALONG_AXIS`.
- Rejects unproven or out-of-range index values with stable `UNSUPPORTED_BOUNDS_CHECK` diagnostics.
- Keeps native support scoped to static dense `INT32` leaf index tensors, because MPSGraph out-of-bounds behavior returns zero instead of matching CPU exception semantics.
- Added visible rejection tests for:
  - out-of-bounds index values,
  - non-`INT32` index dtype,
  - non-dense value input,
  - non-dense index input.
- Broadened native Metal parity tests across representative axis/rank cases:
  - rank-2 gather,
  - rank-3 gather,
  - rank-2 take-along-axis on axis 1,
  - rank-2 take-along-axis on axis 0.
- Strengthened prepared-execution evidence that legal `LOG_SOFTMAX -> TAKE_ALONG_AXIS` remains one Metal-owned region.

## Verification

```bash
./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests PreparedExecutionBuildTest.phaseThirtyTwoTakeAlongAxisCanStayInsideMetalRegionAfterLogSoftmax
./gradlew metalTest
```

All commands passed.

## Remaining Work

- Phase 32-04 still needs coverage target promotion, regression gates, final docs, and `32-VERIFICATION.md`.
- Gradient/scatter index ops remain explicitly deferred.
