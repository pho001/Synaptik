# Summary 34-04: Transformer Coverage Closure

**Phase:** 34 Masked And Causal SDPA
**Wave:** 4
**Status:** Completed
**Completed:** 2026-05-02

## What Changed

- Added `masked_sdpa_small` as a standard workload for direct external BOOL masked SDPA coverage.
- Added `METALSDPAMASK` hot-path coverage target/gate for `masked_sdpa_small`.
- Updated GPU lowering matrix and target semantics contract to describe scoped masked/causal Metal SDPA support.
- Updated Metal/backend architecture/troubleshooting docs from “masked SDPA rejected” to the current supported matrix.
- Preserved CUDA direct masked/causal SDPA as capability-gated unsupported.
- Wrote Phase 34 verification mapping for `METALSDPAMASK-01..03`.

## Supported Scope

- Dense `FLOAT32` rank-3/rank-4 direct SDPA:
  - unmasked,
  - dense external BOOL mask,
  - causal effective BOOL mask,
  - external+causal effective BOOL mask.
- Unsupported cases remain visible:
  - non-dense/broadcast SDPA mask layouts,
  - unsupported dtypes/ranks/shapes,
  - CUDA direct masked/causal SDPA.

## Verification

```bash
./gradlew test --tests GpuHotPathCoverageTargetsTest --tests StandardWorkloadsTest --tests GpuCoverageRegressionGateTest --tests GpuCoverageSummaryTest --tests GpuTargetSemanticsContractTest
./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests PreparedExecutionBuildTest --tests AttentionExecutionTest
./gradlew metalTest
```

All commands passed.

## Follow-Up

- Phase 35 can proceed to conv/pool native execution.
- Future layout work can connect router-produced dense BOOL mask bindings to SDPA input 3 for broadcast mask repair.

