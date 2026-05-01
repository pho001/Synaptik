---
phase: 18-fused-elementwise-and-epilogue-subregions
plan: "01"
status: complete
requirements-completed: [GPUFUSEX-01, GPUFUSEX-02, GPUFUSEX-03]
completed: 2026-05-01
---

# Phase 18 Plan 01: Shared Fusion Subpattern Metadata Contract Summary

Shared GPU fusion subpattern metadata was added beside the existing compound summary contract.

## Shared fusion subpattern metadata contract

- Added `GpuFusionSubpatternSummary` for region-internal GPU fusion metadata.
- Extended `GpuLoweredRegionManifest` with `fusedSubpatterns()` while preserving `fusedSummary()`.
- Populated compatibility subpatterns from existing `ELEMENTWISE_CHAIN`, `LINEAR_BIAS_ACTIVATION`, and CPU fused rejection summaries.
- Updated manifest rendering and docs so GPU fusion remains region-internal lowering/fusion.

## Requirement Coverage

| Requirement | Evidence |
|-------------|----------|
| GPUFUSEX-01 | Elementwise chain subpatterns record original operation spans and lowered primitive counts. |
| GPUFUSEX-02 | Matmul/linear epilogue subpatterns use `LINEAR_BIAS_ACTIVATION` metadata and detail containing `epilogue`. |
| GPUFUSEX-03 | CPU `Operation.OpType.FUSED` remains rejected with `CPU_FUSED_OPERATION_UNSUPPORTED`. |

## Verification

| Command | Result |
|---------|--------|
| `./gradlew test --tests backend.accelerator.lowering.GpuCompoundPatternDetectorTest --tests backend.accelerator.lowering.GpuLoweredRegionManifestTest --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest` | Passed |

## Deviations from Plan

None - plan executed exactly as written.

## Hygiene

profiles/platform/.../tuning/abc/* remained unstaged

## Self-Check: PASSED

