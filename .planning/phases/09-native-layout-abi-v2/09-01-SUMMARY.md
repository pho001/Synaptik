---
phase: 09-native-layout-abi-v2
plan: "01"
subsystem: accelerator-runtime
tags: [layout-abi-v2, accelerator-buffer, metadata]
requires:
  - phase: v1.0-accelerator-buffer-layout-abi
    provides: shared accelerator buffer layout and binding contracts
provides:
  - backend-neutral layout ABI v2 descriptor
  - physical byte span calculation for view metadata
  - portable descriptor tests
affects: [phase-10-gpu-layout-transform, metal-backend, cuda-backend]
tech-stack:
  added: []
  patterns: [immutable-records, checked-layout-arithmetic, java-first-tests]
key-files:
  created:
    - src/main/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2Descriptor.java
    - src/main/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2Status.java
    - src/main/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2StatusCode.java
    - src/main/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2Support.java
    - src/test/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2DescriptorTest.java
  modified:
    - docs/native-bridges-and-blas.md
key-decisions:
  - "Stored only backend id, access mode, and opaque native handle identity in shared metadata."
  - "Computed physical byte span separately from logical byte length with checked arithmetic."
patterns-established:
  - "Layout ABI v2 metadata is represented by backend-neutral immutable records."
requirements-completed: [GPULAYOUT-01]
duration: 12min
completed: 2026-04-30
---

# Phase 09: Native Layout ABI v2 - Plan 01 Summary

**Backend-neutral layout ABI v2 descriptor with physical-span metadata and portable tests**

## Performance

- **Duration:** 12 min
- **Started:** 2026-04-30T12:21:00Z
- **Completed:** 2026-04-30T12:33:00Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- Added `AcceleratorLayoutAbiV2Descriptor` for rank, shape, strides, storage offset, logical bytes, physical span, access, backend id, dtype, layout class, and native handle identity.
- Added ABI v2 status/support records and stable metadata status codes.
- Added descriptor tests for dense, offset, permuted, broadcast, negative stride, overflow, and defensive-copy behavior.
- Documented the metadata boundary in `docs/native-bridges-and-blas.md`.

## Task Commits

1. **Tasks 1-3: ABI v2 metadata descriptor, tests, and docs** - `9115a7e` (feat)

## Files Created/Modified

- `src/main/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2Descriptor.java` - shared ABI v2 metadata descriptor.
- `src/main/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2Status.java` - support/rejection status record.
- `src/main/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2StatusCode.java` - stable metadata status enum.
- `src/main/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2Support.java` - required version constant and helper.
- `src/test/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2DescriptorTest.java` - portable metadata tests.
- `docs/native-bridges-and-blas.md` - layout ABI v2 metadata documentation.

## Decisions Made

- Kept backend-native handles out of shared records; common code stores only `nativeHandleIdentity`.
- Used checked arithmetic for physical span and rejected negative strides in the descriptor factory.

## Deviations from Plan

None - plan executed as specified.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Verification

Passed:

`./gradlew test --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest --tests backend.accelerator.buffer.AcceleratorLayoutAbiV2DescriptorTest`

## Next Phase Readiness

Plan 09-02 can now attach Metal/CUDA capability records and optional native symbol checks to the shared ABI v2 support model.

---
*Phase: 09-native-layout-abi-v2*
*Completed: 2026-04-30*

