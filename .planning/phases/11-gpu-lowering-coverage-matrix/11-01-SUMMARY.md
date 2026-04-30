---
phase: 11-gpu-lowering-coverage-matrix
plan: "01"
subsystem: accelerator-lowering
tags: [gpu-lowering, coverage-matrix, metal, cuda, reason-codes]

requires:
  - phase: 09-native-layout-abi-v2
    provides: layout ABI fallback and capability contracts
  - phase: 10-gpu-layout-transform-and-view-path
    provides: layout/view residency and materialization contracts
provides:
  - Backend-neutral GPU lowering coverage statuses, operation families, unsupported reasons, entries, and matrix
  - Metal/CUDA source-level operation family matrix for supported, fallback, and unsupported rows
  - Matrix/docs drift tests for Phase 11 coverage requirements
affects: [phase-11, phase-12, phase-13, metal, cuda]

tech-stack:
  added: []
  patterns: [backend-neutral accelerator records, executable coverage matrix, stable reason-code docs]

key-files:
  created:
    - src/main/java/backend/accelerator/lowering/GpuLoweringCoverageStatus.java
    - src/main/java/backend/accelerator/lowering/GpuLoweringUnsupportedReason.java
    - src/main/java/backend/accelerator/lowering/GpuLoweringOperationFamily.java
    - src/main/java/backend/accelerator/lowering/GpuLoweringCoverageEntry.java
    - src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java
    - src/test/java/backend/accelerator/lowering/GpuLoweringCoverageMatrixTest.java
    - docs/gpu-lowering-coverage.md
  modified:
    - docs/native-bridges-and-blas.md

key-decisions:
  - "The coverage matrix is backend-neutral source under backend.accelerator.lowering; backend-owned capability, dtype, layout, and ABI gates remain in Metal/CUDA code."
  - "Supported rows use SUPPORTED reason, while fallback and unsupported rows must carry stable non-supported reasons."
  - "Forward reductions, normalization, loss-adjacent ops, and fused compound patterns are explicit fallback/unsupported rows until later Phase 11/12 work expands execution coverage."

patterns-established:
  - "Coverage claims are represented as typed source records and mirrored in docs with tests guarding required families and reason codes."
  - "Unlisted backend operations resolve to explicit unsupported rows instead of silent support."

requirements-completed: [GPULOWER-01, GPULOWER-03]

duration: 16 min
completed: 2026-04-30
---

# Phase 11 Plan 01: Shared GPU Lowering Coverage Contract Summary

**Backend-neutral Metal/CUDA lowering coverage matrix with stable operation families, statuses, and unsupported reason codes**

## Performance

- **Duration:** 16 min
- **Started:** 2026-04-30T16:20:00Z
- **Completed:** 2026-04-30T16:35:58Z
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments

- Added failing-first tests for required Phase 11 families across `GPU_METAL` and `GPU_CUDA`.
- Added `GpuLoweringCoverageStatus`, `GpuLoweringUnsupportedReason`, `GpuLoweringOperationFamily`, `GpuLoweringCoverageEntry`, and `GpuLoweringCoverageMatrix`.
- Seeded Metal/CUDA rows for matmul/linear, elementwise chains, layout/view-adjacent ops, softmax-like flows, reductions, normalization, loss-adjacent ops, attention, and deferred fused patterns.
- Added `docs/gpu-lowering-coverage.md` and linked it from `docs/native-bridges-and-blas.md`.

## Task Commits

1. **Task 1: Add failing-first tests for the GPU lowering coverage matrix** - `3e43daf` (test)
2. **Task 2: Implement shared coverage status, family, reason, entry, and matrix records** - `13eed2d` (feat)
3. **Task 3: Add checked-in documentation for the coverage matrix** - `058c193` (docs)

## Files Created/Modified

- `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageStatus.java` - Supported/fallback/unsupported status enum.
- `src/main/java/backend/accelerator/lowering/GpuLoweringUnsupportedReason.java` - Stable reason-code enum for non-supported rows.
- `src/main/java/backend/accelerator/lowering/GpuLoweringOperationFamily.java` - Shared operation-family taxonomy.
- `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageEntry.java` - Immutable coverage row with validation.
- `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java` - Metal/CUDA matrix and lookup helpers.
- `src/test/java/backend/accelerator/lowering/GpuLoweringCoverageMatrixTest.java` - Source/docs drift guard tests.
- `docs/gpu-lowering-coverage.md` - Human-readable coverage matrix and reason-code legend.
- `docs/native-bridges-and-blas.md` - Link and trace-visible fallback note.

## Decisions Made

- Kept coverage classifications semantic and backend-neutral; Metal/CUDA adapters still own final legality based on dtype, layout, capability, and native ABI.
- Marked `LOG_SOFTMAX` as fallback pending Plan 11-03, while `SOFTMAX` remains supported.
- Marked normalization and `FUSED` rows with `DEFERRED_FUSED_REGION` so Phase 12 ownership is explicit.
- Treated loss-adjacent indexed paths as `UNSUPPORTED_DTYPE` because they depend on INT32 targets outside the current accelerator DAG dtype contract.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The Task 2 source model made three of four matrix tests pass; the docs drift test failed until Task 3 added `docs/gpu-lowering-coverage.md`. This was resolved before the final plan verification.

## Verification

- `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest` - PASS
- `rg -n "enum GpuLoweringCoverageStatus|enum GpuLoweringUnsupportedReason|enum GpuLoweringOperationFamily|record GpuLoweringCoverageEntry|class GpuLoweringCoverageMatrix|DEFERRED_FUSED_REGION" src/main/java/backend/accelerator/lowering` - PASS
- `rg -n 'GPU Lowering Coverage Matrix|GPULOWER-01|GPULOWER-02|GPULOWER-03|UNSUPPORTED_OPERATION|UNSUPPORTED_DTYPE|UNSUPPORTED_LAYOUT|DEFERRED_FUSED_REGION|public \`Tensor\` remains logical' docs/gpu-lowering-coverage.md docs/native-bridges-and-blas.md` - PASS
- `rg -n "GPU Lowering Coverage Matrix|GPULOWER-01|UNSUPPORTED_OPERATION|DEFERRED_FUSED_REGION" docs src/main/java/backend/accelerator/lowering src/test/java/backend/accelerator/lowering` - PASS

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for `11-02`: Metal and CUDA legality adapters can now consume the shared coverage matrix while preserving backend-owned dtype, layout, capability, and native ABI gates.

---
*Phase: 11-gpu-lowering-coverage-matrix*
*Completed: 2026-04-30*
