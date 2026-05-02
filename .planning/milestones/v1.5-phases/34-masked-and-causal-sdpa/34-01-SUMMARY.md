# Summary 34-01: Mask Semantics And Parity Contract

**Phase:** 34 Masked And Causal SDPA
**Wave:** 1
**Status:** Completed
**Completed:** 2026-05-02

## What Changed

- Added `MetalSdpaMaskSemantics` as the Metal planner source of truth for direct SDPA mask semantics.
- Added explicit mask mode classification:
  - `UNMASKED`
  - `EXTERNAL_BOOL_MASK`
  - `CAUSAL_BOOL_MASK`
  - `EXTERNAL_AND_CAUSAL_BOOL_MASK`
  - `INVALID`
- Refactored `MetalPartitionSupport.sdpaUnsupportedReason(...)` so masked direct SDPA still rejects before native execution, but with precise reason details instead of the old single generic message.
- Preserved existing unmasked `FLOAT32` rank-3/rank-4 direct Metal SDPA planner support.
- Added tests for external BOOL mask, causal-only mask, external+causal combined mask, and broadcast-mask layout rejection.

## Current Supported State

- Unmasked direct Metal SDPA remains supported for the previously verified dense `FLOAT32` rank-3/rank-4 scope.
- Masked and causal direct Metal SDPA remain unsupported until Wave 2/3 add executable native/lowered support.
- Rejection is now mode-specific:
  - external BOOL mask requires native BOOL mask ABI support,
  - causal mask requires native causal support,
  - external+causal requires native causal support,
  - broadcast/non-dense mask layout rejects as mask layout unsupported.

## Verification

```bash
./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest
./gradlew test --tests AttentionExecutionTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest
```

Both commands passed.

## Follow-Up

- Wave 34-02 should extend the DAG/native contract so input 3 can carry a verified BOOL mask without reusing unmasked executable signatures.
- Wave 34-03 should decide whether causal support consumes a device-resident effective mask or generates a causal bias/mask inside the Metal path.

