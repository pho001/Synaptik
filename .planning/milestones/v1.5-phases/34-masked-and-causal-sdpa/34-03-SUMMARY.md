# Summary 34-03: Causal SDPA Support And Rejection Detail

**Phase:** 34 Masked And Causal SDPA
**Wave:** 3
**Status:** Completed
**Completed:** 2026-05-02

## What Changed

- Promoted causal-only and external+causal direct SDPA to the same Metal execution path as external BOOL masked SDPA.
- Kept the implementation aligned with the public Tensor contract: `TensorAttentionOps` already creates the effective BOOL mask, combines external and causal masks by logical AND, and expands the result to score shape.
- Reused SDPA `input3` as the effective BOOL mask input for:
  - external mask,
  - causal-only mask,
  - external+causal mask.
- Added native bridge parity fixtures for causal-only and external+causal rank-3 SDPA.
- Updated Metal lowering tests so causal and external+causal direct SDPA are planner-legal when the effective mask is dense.

## Current Supported State

- Supported:
  - dense `FLOAT32` rank-3/rank-4 direct SDPA without mask,
  - dense `FLOAT32` rank-3/rank-4 direct SDPA with dense effective `BOOL` mask,
  - causal-only direct SDPA when the effective causal mask is dense,
  - external+causal direct SDPA when the effective combined mask is dense.
- Still rejected:
  - broadcast/non-dense mask layouts until router repair is connected to SDPA mask input admission,
  - unsupported dtypes/ranks/shapes,
  - CUDA direct masked/causal SDPA.

## Verification

```bash
./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests PreparedExecutionBuildTest
./gradlew metalTest
```

Both commands passed.

## Follow-Up

- Wave 34-04 should add coverage targets and docs showing masked/causal SDPA support and remaining layout/dtype/CUDA rejection scope.
- Future layout work can admit broadcast mask repair once the SDPA mask input path consumes router-produced dense BOOL bindings without CPU materialization.

