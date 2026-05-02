# Summary 25-01: Forward SDPA Semantics And Parity Contract

**Status:** Complete
**Date:** 2026-05-01

## Completed

- Expanded `GpuTargetSemanticsContract` for forward SDPA with explicit Phase 25 GPU admission boundaries: FLOAT32-only GPU admission, rank 3/4, query/key/value shape relationships, resolved scale semantics, separate unmasked/causal/external mask cases, layout legality, and numerical tolerance.
- Added stable prefixed Metal rejection reasons:
  - `CAPABILITY_MISSING` for unmasked direct SDPA until native scale parity is verified.
  - `UNSUPPORTED_MASK_SEMANTICS` for direct masked SDPA until BOOL mask semantics match backend mask behavior.
  - `BACKWARD_CONTEXT_UNSUPPORTED` for forward SDPA inside Metal backward regions.
- Added SDPA parity/contract test coverage for rank-4 default scale and invalid shape/dtype/mask contracts.
- Updated SDPA docs to distinguish unmasked direct SDPA, masked direct SDPA, and CUDA support-or-rejection requirements.
- Kept `GpuLoweringCoverageMatrix` support status unchanged for forward SDPA.

## Verification

Passed:

```bash
./gradlew test --tests GpuTargetSemanticsContractTest --tests AttentionExecutionTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest
```

## Deviations from Plan

None.
