# Summary 34-02: Mask Conversion Lowering And Native Execution

**Phase:** 34 Masked And Causal SDPA
**Wave:** 2
**Status:** Completed
**Completed:** 2026-05-02

## What Changed

- Promoted dense external/effective `BOOL` mask direct SDPA from planner rejection to Metal-supported scope.
- Allowed SDPA external input role 3 to accept verified dense `BOOL` predicate data.
- Kept causal and external+causal mask modes rejected for Wave 34-03 with causal-specific reasons.
- Reused the existing accelerator DAG `SDPA` node fourth input as the mask operand.
- Updated the Objective-C MPSGraph shim so node type `26` applies an optional BOOL mask through `select(mask, scores, -1.0e9)` before softmax.
- Added tests proving:
  - masked SDPA lowers with `input3` as an external DAG input,
  - masked and unmasked executable signatures differ,
  - prepared execution can build a masked native SDPA DAG when Metal is available,
  - native buffer execution matches CPU parity for a rank-3 masked SDPA fixture.

## Current Supported State

- Supported:
  - dense `FLOAT32` rank-3/rank-4 direct SDPA without mask,
  - dense `FLOAT32` rank-3/rank-4 direct SDPA with dense external/effective `BOOL` mask.
- Still rejected:
  - causal-only direct SDPA,
  - external+causal direct SDPA,
  - broadcast/non-dense mask layouts until router repair is explicitly connected,
  - CUDA direct masked SDPA.

## Verification

```bash
./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests PreparedExecutionBuildTest
./gradlew metalTest
```

Both commands passed.

## Follow-Up

- Wave 34-03 should enable causal-only and external+causal modes without changing public Tensor APIs.
- Wave 34-04 should update coverage gates and docs from the previous “masked SDPA rejected” wording to the new scoped support matrix.

