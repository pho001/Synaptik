---
phase: 11-gpu-lowering-coverage-matrix
plan: "02"
subsystem: accelerator-lowering
tags: [gpu-lowering, legality-adapters, metal, cuda, reason-codes]

requires:
  - phase: 11-gpu-lowering-coverage-matrix
    provides: shared GPU lowering coverage matrix and stable reason codes
provides:
  - Metal planner legality routed through the shared coverage matrix
  - CUDA planner unsupported reason helper and matrix-backed support checks
  - Portable tests for supported candidates and visible unsupported reduction, normalization, loss-adjacent, and CUDA layout reasons
affects: [phase-11, phase-12, phase-13, metal, cuda]

tech-stack:
  added: []
  patterns: [matrix-backed planner legality, stable adapter diagnostics, conservative CUDA layout rejection]

key-files:
  created: []
  modified:
    - src/main/java/backend/metal/lowering/MetalPartitionSupport.java
    - src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java
    - src/test/java/backend/metal/lowering/MetalRegionLowererTest.java
    - src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java
    - docs/gpu-lowering-coverage.md

key-decisions:
  - "Metal and CUDA now consult GpuLoweringCoverageMatrix for semantic operation support after their local preconditions."
  - "CUDA exposes plannerUnsupportedReason(...) with stable messages comparable to Metal."
  - "CUDA direct compute rejects non-dense inputs visibly until Phase 10 metadata-only view propagation or dense materialization makes the layout legal."

patterns-established:
  - "Adapter unsupported messages include the shared reason enum name and backend operation family context."
  - "Backend-specific dtype, SDPA, external-input role, runtime capability, and native ABI gates remain outside the shared matrix."

requirements-completed: [GPULOWER-01, GPULOWER-03]

duration: 5 min
completed: 2026-04-30
---

# Phase 11 Plan 02: Metal/CUDA Legality Coverage Alignment Summary

**Metal and CUDA planner legality now consume the shared lowering coverage matrix while preserving backend-owned gates**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-30T16:35:58Z
- **Completed:** 2026-04-30T16:40:31Z
- **Tasks:** 4
- **Files modified:** 5

## Accomplishments

- Added adapter tests tying Metal and CUDA supported forward ops to `GpuLoweringCoverageMatrix`.
- Added tests for shared rejection reason fragments on reductions, normalization, loss-adjacent operations, and CUDA direct non-dense compute.
- Routed `MetalPartitionSupport` through shared coverage entries after existing dtype and SDPA semantic gates.
- Added `CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(...)` and switched CUDA support checks to it.
- Documented shared-vs-backend-owned planner rejection layers.

## Task Commits

1. **Task 1: Add backend adapter tests for matrix-backed support and rejection reasons** - `0952d40` (test)
2. **Task 2: Wire Metal planner support through shared matrix entries** - `1f0089e` (feat)
3. **Task 3: Add CUDA planner unsupported reasons and matrix-backed support** - `9d32266` (feat)
4. **Task 4: Update matrix docs with adapter-owned rejection behavior** - `bf8525c` (docs)

## Files Created/Modified

- `src/main/java/backend/metal/lowering/MetalPartitionSupport.java` - Uses shared coverage entries for operation support after Metal-specific gates.
- `src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java` - Adds stable planner unsupported reasons and conservative non-dense input rejection.
- `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java` - Adds matrix-backed Metal support/rejection tests.
- `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java` - Adds matrix-backed CUDA support/rejection and non-dense layout tests.
- `docs/gpu-lowering-coverage.md` - Adds planner rejection source documentation.

## Decisions Made

- Kept Metal SDPA-specific messages ahead of shared matrix handling to preserve existing semantic guardrails.
- Kept CUDA structural candidate construction unchanged; only support classification and diagnostics changed.
- Used `UNSUPPORTED_LAYOUT` for direct non-dense CUDA compute rather than pretending matrix support alone makes the layout legal.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Task 2 Metal verification needed the Task 3 CUDA method because Gradle compiles all test sources. CUDA implementation was completed before final targeted verification, while commits remained separated by task.

## Verification

- `./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest` - PASS
- `./gradlew test --tests backend.cuda.lowering.CudaRegionLowererTest --tests backend.cuda.exec.CudaLayoutTransformDeviceFlowTest` - PASS
- `rg -n "plannerUnsupportedReason|GpuLoweringCoverageMatrix|Planner Rejection Sources" src/main/java/backend src/test/java/backend docs/gpu-lowering-coverage.md` - PASS

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for `11-03`: `LOG_SOFTMAX` currently rejects with a shared `UNSUPPORTED_OPERATION` reason, so the next plan can turn that row into supported lowering and update the matrix/tests.

---
*Phase: 11-gpu-lowering-coverage-matrix*
*Completed: 2026-04-30*
