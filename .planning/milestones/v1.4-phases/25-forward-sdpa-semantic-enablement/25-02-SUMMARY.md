# Summary 25-02: Metal Forward SDPA Admission Or Stable Rejection

**Status:** Complete
**Date:** 2026-05-01

## Completed

- Admitted the narrow Metal direct forward SDPA contract for unmasked `FLOAT32` rank-3/rank-4 inputs after native parity evidence.
- Replaced the native forward SDPA implementation with an explicit MPSGraph primitive DAG: `Q * K^T`, score scale, softmax, and `* V`.
- Fixed native executable cache signatures to include lowered DAG scalar bits, preventing a compiled SDPA executable for one scale from being reused for another scale.
- Kept masked direct SDPA rejected with `UNSUPPORTED_MASK_SEMANTICS` and preserved backward-context safety through `BACKWARD_CONTEXT_UNSUPPORTED`.
- Updated Metal lowering, prepared-execution, coverage truth, and docs so Metal is `SUPPORTED` only for the verified unmasked SDPA path.
- Added native Metal parity coverage for explicit scale rank-3 SDPA and default-scale rank-4 SDPA, plus a regression test proving SDPA scale changes alter the executable cache signature.

## Verification

Passed:

```bash
./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuCoverageSummaryTest --tests GpuTargetSemanticsContractTest
./gradlew metalTest
```

## Deviations from Plan

- The initial direct MPSGraph SDPA API path produced output equivalent to scale `1.0` for a scale `0.5` fixture. The final supported path uses explicit MPSGraph primitives instead of relying on the direct API.
- The failure also exposed a native executable cache-key bug: scalar DAG parameters were missing from `AcceleratorSubgraphSignature`. The fix is included because SDPA parity depends on scale-specific native executables.
