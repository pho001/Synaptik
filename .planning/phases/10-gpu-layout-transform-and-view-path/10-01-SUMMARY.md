---
phase: 10-gpu-layout-transform-and-view-path
plan: "01"
subsystem: accelerator-runtime
tags: [gpu-layout, accelerator-buffer, layout-transform, traces]

requires:
  - phase: 09-native-layout-abi-v2
    provides: backend-neutral layout metadata and stable native layout fallback reasons
provides:
  - Backend-neutral GPU layout transform request and decision records
  - Stable reason codes for metadata-only view, dense materialization, backend mismatch, missing binding, and unsupported transforms
  - Planner tests covering supported and rejected layout transform categories
affects: [phase-10, phase-11, phase-12, metal, cuda]

tech-stack:
  added: []
  patterns: [backend-neutral accelerator records, pure layout transform classification, stable fallback reason codes]

key-files:
  created:
    - src/main/java/backend/accelerator/buffer/AcceleratorLayoutTransformKind.java
    - src/main/java/backend/accelerator/buffer/AcceleratorLayoutTransformRequest.java
    - src/main/java/backend/accelerator/buffer/AcceleratorLayoutTransformDecision.java
    - src/main/java/backend/accelerator/buffer/AcceleratorLayoutTransformPlanner.java
    - src/test/java/backend/accelerator/buffer/AcceleratorLayoutTransformPlannerTest.java
  modified:
    - src/main/java/backend/accelerator/buffer/AcceleratorBufferReasonCode.java
    - docs/native-bridges-and-blas.md

key-decisions:
  - "GPU layout transform classification stays backend-neutral and carries no Metal or CUDA handle objects."
  - "Metadata-only views and dense GPU materialization use separate decision kinds and reason codes."
  - "Unsupported transform paths are represented as decisions instead of hidden fallback."

patterns-established:
  - "Layout transform decisions are pure records under backend.accelerator.buffer."
  - "Reason codes distinguish available view binding, dense materialization, missing source binding, backend mismatch, and unsupported transform."

requirements-completed: [GPUVIEW-01, GPUVIEW-03]

duration: 5 min
completed: 2026-04-30
---

# Phase 10 Plan 01: Shared GPU Layout Transform Contract Summary

**Backend-neutral GPU layout transform planner with explicit metadata-only, dense materialization, and unsupported decisions**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-30T12:44:30Z
- **Completed:** 2026-04-30T12:49:26Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments

- Added failing-first planner tests for permute, expand, contiguous, non-contiguous reshape, missing source binding, backend mismatch, and unsupported negative-stride metadata.
- Added `AcceleratorLayoutTransformKind`, request, decision, and planner records/classes under `backend.accelerator.buffer`.
- Extended `AcceleratorBufferReasonCode` with GPU layout transform reason codes.
- Documented the GPU Layout Transform Contract in `docs/native-bridges-and-blas.md`.

## Task Commits

1. **Task 1: Add failing-first planner tests** - `624e953` (test)
2. **Task 2: Implement shared layout transform records and stable reason codes** - `b3ea7af` (feat)
3. **Task 3: Document the GPU layout transform contract** - `3182f7f` (docs)

## Files Created/Modified

- `src/main/java/backend/accelerator/buffer/AcceleratorLayoutTransformKind.java` - Decision kind enum for metadata-only views, dense GPU materialization, and unsupported transforms.
- `src/main/java/backend/accelerator/buffer/AcceleratorLayoutTransformRequest.java` - Backend-neutral transform request record.
- `src/main/java/backend/accelerator/buffer/AcceleratorLayoutTransformDecision.java` - Decision record plus helper factories.
- `src/main/java/backend/accelerator/buffer/AcceleratorLayoutTransformPlanner.java` - Pure classifier for layout transform decisions.
- `src/main/java/backend/accelerator/buffer/AcceleratorBufferReasonCode.java` - Added GPU layout transform reason codes.
- `src/test/java/backend/accelerator/buffer/AcceleratorLayoutTransformPlannerTest.java` - Portable planner contract tests.
- `docs/native-bridges-and-blas.md` - Added GPU layout transform contract documentation.

## Decisions Made

- Metadata-only view operations are `NOOP`, `SELECT`, `PERMUTE`, `EXPAND`, `EXPAND_DIMS`, `SQUEEZE`, and contiguous-source `RESHAPE`.
- `CONTIGUOUS` and non-contiguous-source `RESHAPE` classify as dense GPU materialization candidates.
- Backend mismatch, missing source binding, unsupported layout metadata, and unsupported operation types are explicit rejected decisions.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The RED test helper initially hardcoded `logicalElementCount=6`, which made the `[1,3]` expand source fixture classify as unsupported. The helper now computes element count from shape before the GREEN verification.

## Verification

- `./gradlew test --tests backend.accelerator.buffer.AcceleratorLayoutTransformPlannerTest --tests backend.accelerator.buffer.AcceleratorLayoutAbiV2DescriptorTest` - PASS
- `rg -n 'GPU Layout Transform Contract|metadata-only views|dense GPU materialization|GPU_LAYOUT_VIEW_BINDING_AVAILABLE|Public \`Tensor\` stays logical' docs/native-bridges-and-blas.md` - PASS
- `rg -n "GPU Layout Transform Contract|GPU_LAYOUT_VIEW_BINDING_AVAILABLE|AcceleratorLayoutTransformPlanner" src/main/java src/test/java docs` - PASS

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for `10-02`: backend-specific alias binding factories and pre-CPU-step device layout view propagation can now consume the shared decision contract.

---
*Phase: 10-gpu-layout-transform-and-view-path*
*Completed: 2026-04-30*
