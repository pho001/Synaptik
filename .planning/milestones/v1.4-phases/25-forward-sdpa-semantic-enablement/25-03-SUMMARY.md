# Summary 25-03: CUDA Forward SDPA Support-Or-Rejection Contract

**Status:** Complete
**Date:** 2026-05-01

## Completed

- Kept CUDA direct forward SDPA as an explicit fallback because no verified CUDA native/lowered SDPA execution path exists in the local codebase.
- Changed the CUDA forward SDPA matrix reason to `CAPABILITY_MISSING` for the legal unmasked target instead of a generic unsupported-operation claim.
- Added CUDA legality detail for direct SDPA:
  - `CAPABILITY_MISSING` for legal unmasked `FLOAT32` rank-3/rank-4 SDPA.
  - `UNSUPPORTED_MASK_SEMANTICS` for masked SDPA.
  - `UNSUPPORTED_DTYPE` and `UNSUPPORTED_LAYOUT` before capability-missing for illegal inputs.
- Added prepared-execution coverage proving CUDA direct SDPA falls back to the CPU path with visible planner reason and CPU parity.
- Updated coverage docs so CUDA does not mirror Metal SDPA support without backend evidence.

## Verification

Passed:

```bash
./gradlew test --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests GpuCoverageSummaryTest --tests CompiledGraphTraceTest
```

Not run:

```bash
./gradlew cudaTest
```

`nvcc` is not available in this environment, so native CUDA compilation remains hardware/toolchain-gated.

## Deviations from Plan

- CUDA support was not implemented in this wave. The selected path is stable capability rejection because native CUDA SDPA execution lacks local implementation and compile evidence.
