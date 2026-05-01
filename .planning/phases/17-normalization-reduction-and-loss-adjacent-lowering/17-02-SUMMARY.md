---
phase: 17-normalization-reduction-and-loss-adjacent-lowering
plan: "02"
status: complete
requirements-completed: [GPUNORM-01, GPUNORM-02, GPUNORM-03]
completed: 2026-05-01
---

# Phase 17 Plan 02: Backend Legality Summary

Metal and CUDA planner legality now render Phase 17 rejection detail from the shared lowering matrix while keeping backend-owned capability, dtype, and layout gates explicit.

## Backend legality and stable rejection detail

- Routed Metal and CUDA unsupported operation diagnostics through `GpuLoweringCoverageMatrix.plannerUnsupportedDetail(...)`.
- Preserved `REDUCTION_ADJACENT` prefix behavior for reductions and normalization blockers.
- Added backend tests proving reduction, normalization, conv, and loss-adjacent rejections include family, status, operation, and hot-path target evidence.
- Added required-buffer-mode regression coverage showing unsupported reduction and normalization nodes remain CPU-prepared with precise planner detail.
- Confirmed CUDA direct non-dense compute still rejects before execution with the existing `UNSUPPORTED_LAYOUT` message; layout and dtype legality gates remained explicit.

## Verification

| Command | Result |
|---------|--------|
| `./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest` | Passed |
| `./gradlew test --tests PreparedExecutionBuildTest --tests backend.cuda.lowering.CudaRegionLowererTest` | Passed |
| `rg -n "plannerUnsupportedDetail\\(ComputeBackend.GPU_METAL|plannerUnsupportedDetail\\(ComputeBackend.GPU_CUDA|compoundPatternPrefix" src/main/java/backend/metal/lowering/MetalPartitionSupport.java src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java` | Passed |
| `rg -n "metalRequiredModeExposesPhaseSeventeenReductionRejection|cudaRequiredModeExposesPhaseSeventeenNormalizationRejection|cudaPhaseSeventeenKeepsDirectNonDenseLayoutRejectionBeforeExecution|UNSUPPORTED_LAYOUT: direct non-dense CUDA compute remains conservative" src/test/java/PreparedExecutionBuildTest.java src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java` | Passed |

## Requirement Coverage

- `GPUNORM-01`: Metal and CUDA legality now consume the shared matrix detail path for unsupported Phase 17 families.
- `GPUNORM-02`: Rejection strings expose stable family and hot-path target evidence for `layer_norm_small`, `conv2d_resnet_3x3`, and `transformer_block_hot_path`.
- `GPUNORM-03`: Required-mode tests and CUDA non-dense layout coverage show unsupported reduction/normalization/layout cases do not silently bypass existing gates.

## Commits

| Commit | Description |
|--------|-------------|
| `6769653` | Added failing-first backend rejection detail tests. |
| `b182644` | Routed backend rejection details through the shared matrix helper. |
| `1a7ace1` | Covered required-mode rejection detail and CUDA layout gate preservation. |

## Deviations from Plan

None - plan executed exactly as written.

## Hygiene

profiles/platform/.../tuning/abc/* remained unstaged

## Self-Check: PASSED
