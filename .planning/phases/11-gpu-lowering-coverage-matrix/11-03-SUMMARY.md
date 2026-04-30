---
phase: 11-gpu-lowering-coverage-matrix
plan: "03"
subsystem: accelerator-lowering
tags: [log-softmax, gpu-lowering, backend-selection, metal, cuda]

requires:
  - phase: 11-gpu-lowering-coverage-matrix
    provides: shared matrix and Metal/CUDA legality alignment
provides:
  - LOG_SOFTMAX lowering as SOFTMAX followed by LOG using existing accelerator DAG primitives
  - Metal/CUDA selected and rejected candidate tests for softmax-ish support and unsupported families
  - Prepared-selection evidence for longer matmul/linear plus LOG_SOFTMAX GPU regions
affects: [phase-11, phase-12, phase-13, metal, cuda]

tech-stack:
  added: []
  patterns: [DAG decomposition without native ABI expansion, selected/rejected candidate verification]

key-files:
  created:
    - src/test/java/backend/accelerator/lowering/AcceleratorSubgraphLowererTest.java
  modified:
    - src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java
    - src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java
    - src/test/java/backend/metal/lowering/MetalRegionLowererTest.java
    - src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java
    - src/test/java/PreparedExecutionBuildTest.java
    - docs/gpu-lowering-coverage.md

key-decisions:
  - "LOG_SOFTMAX is supported by decomposing to SOFTMAX followed by LOG; no native ABI operation code was added."
  - "The generic multi-node DAG builder also handles LOG_SOFTMAX so matmul/linear -> LOG_SOFTMAX can stay in one GPU region."
  - "Reduction, normalization, and loss-adjacent families remain visibly rejected through shared coverage reasons."

patterns-established:
  - "Softmax-ish expansions should reuse existing DAG primitives before adding native ABI codes."
  - "Prepared-selection tests must prove both selected supported regions and rejected unsupported candidates."

requirements-completed: [GPULOWER-02, GPULOWER-03]

duration: 7 min
completed: 2026-04-30
---

# Phase 11 Plan 03: Softmax-Ish Lowering Expansion Summary

**LOG_SOFTMAX now stays in Metal/CUDA GPU regions by lowering to existing SOFTMAX and LOG DAG primitives**

## Performance

- **Duration:** 7 min
- **Started:** 2026-04-30T16:40:31Z
- **Completed:** 2026-04-30T16:47:16Z
- **Tasks:** 5
- **Files modified:** 7

## Accomplishments

- Added shared lowerer tests proving `LOG_SOFTMAX` decomposes into `SOFTMAX` then `LOG`.
- Implemented single-node and multi-node `LOG_SOFTMAX` DAG lowering using existing ABI node types.
- Updated the coverage matrix to mark `LOG_SOFTMAX` supported for Metal and CUDA.
- Added Metal/CUDA tests proving `matmul -> logSoftmax` can create a valid GPU partition plan.
- Added `PreparedExecutionBuildTest` coverage proving selected Metal/CUDA `matmul -> LOG_SOFTMAX` decisions and visible CPU fallback for unsupported loss/reduction nodes.
- Updated coverage docs to document supported `LOG_SOFTMAX` without a new native ABI op code.

## Task Commits

1. **Task 1: Add shared lowerer tests for LOG_SOFTMAX decomposition** - `470ff87` (test)
2. **Task 2: Implement LOG_SOFTMAX lowering through existing DAG primitives** - `9beaa80` (feat)
3. **Task 3: Add Metal and CUDA selected/rejected candidate tests** - `670cb2c` (test)
4. **Task 4: Add prepared-selection tests for longer supported GPU regions** - `b9d5411` (test)
5. **Task 5: Update coverage docs for LOG_SOFTMAX and explicit unsupported families** - `4d3e686` (docs)

## Files Created/Modified

- `src/test/java/backend/accelerator/lowering/AcceleratorSubgraphLowererTest.java` - New lowerer decomposition tests.
- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java` - Adds `tryBuildLogSoftmaxDagSpec` and multi-node LOG_SOFTMAX expansion.
- `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java` - Marks `LOG_SOFTMAX` supported for Metal/CUDA.
- `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java` - Adds accepted LOG_SOFTMAX and rejected family tests.
- `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java` - Adds accepted LOG_SOFTMAX and rejected family tests.
- `src/test/java/PreparedExecutionBuildTest.java` - Adds selected/rejected prepare trace tests.
- `docs/gpu-lowering-coverage.md` - Updates LOG_SOFTMAX coverage and ABI note.

## Decisions Made

- No new native ABI enum was introduced for `LOG_SOFTMAX`.
- Multi-node `LOG_SOFTMAX` expansion was added because selected GPU regions need to cover `matmul/linear -> logSoftmax`, not only a single isolated softmax-ish node.
- Docs keep normalization rows aligned with the source matrix reason codes to avoid source/doc drift.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The planned single-node special builder was insufficient for longer selected regions; the generic DAG builder also needed `LOG_SOFTMAX` expansion. This was implemented and verified in the Task 3 commit.

## Verification

- `./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest` - PASS
- `./gradlew test --tests PreparedExecutionBuildTest` - PASS
- `rg -n "LOG_SOFTMAX|lowered as SOFTMAX followed by LOG|UNSUPPORTED_OPERATION" src/main/java src/test/java docs/gpu-lowering-coverage.md` - PASS

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for `11-04`: trace/docs closure can now treat `LOG_SOFTMAX` as supported and focus on visible fallback, layout-heavy evidence, final docs, and profile artifact hygiene.

---
*Phase: 11-gpu-lowering-coverage-matrix*
*Completed: 2026-04-30*
