---
phase: 04-tuning-and-profile-ownership-audit
plan: "01"
subsystem: tuning
tags: [tuning, autotune, calibration, ownership]
requires:
  - phase: 04-tuning-and-profile-ownership-audit
    provides: Phase 4 context and ownership decisions
provides:
  - Explicit tuning knob ownership matrix
  - Graph autotune candidate ownership validation
  - Platform calibration candidate ownership validation
affects: [phase-04, tuning, graph-autotune, platform-calibration]
tech-stack:
  added: []
  patterns: [central ownership registry, candidate-space validation]
key-files:
  created:
    - src/main/java/tuning/ownership/TuningKnobOwner.java
    - src/main/java/tuning/ownership/TuningKnobOwnership.java
    - src/test/java/TuningKnobOwnershipTest.java
  modified:
    - src/main/java/tuning/candidate/graph/GraphRuntimePolicyVariant.java
    - src/main/java/tuning/candidate/graph/GraphAutotuneCandidateSpace.java
    - src/main/java/tuning/candidate/graph/GraphPolicyMutators.java
    - src/main/java/tuning/calibration/family/CalibrationFamilyRegistry.java
    - src/test/java/CalibrationCandidateOwnershipTest.java
    - src/test/java/GraphAutotuneCandidateSpaceTest.java
    - src/test/java/tuning/candidate/graph/GraphAutotuneCandidateSpaceTest.java
key-decisions:
  - "Graph autotune candidates must declare graph/workload knob assignments before candidate creation."
  - "Calibration candidates remain constrained by family-owned knobs and the new platform/dtype ownership matrix."
patterns-established:
  - "Ownership registry: tuning.ownership.TuningKnobOwnership is the source of truth for graph/workload and platform/dtype knob classes."
  - "Candidate validation: candidate spaces validate declared knob assignments before exposing candidates."
requirements-completed: [TUNE-01, TUNE-02]
duration: 7 min
completed: 2026-04-30
---

# Phase 4 Plan 01: Tuning Knob Ownership Matrix Summary

**Central tuning knob ownership validation for graph autotune and platform calibration candidate spaces**

## Performance

- **Duration:** 7 min
- **Started:** 2026-04-30T06:12:45Z
- **Completed:** 2026-04-30T06:19:41Z
- **Tasks:** 2
- **Files modified:** 10

## Accomplishments

- Added `TuningKnobOwner` and `TuningKnobOwnership` with graph/workload and platform/dtype ownership classes.
- Added tests proving accelerator buffer mode and Metal transfer model are graph-owned while Metal selection thresholds remain platform-owned.
- Extended graph autotune variants with declared knob assignments and metadata.
- Enforced `validateGraphWorkload(...)` in graph candidate creation and `validatePlatformDtype(...)` in calibration family validation.

## Task Commits

1. **Task 1: Add the central tuning knob ownership matrix** - `3bc5fc3`
2. **Task 2: Enforce ownership in graph autotune and calibration candidate spaces** - `a4e70fc`

## Files Created/Modified

- `src/main/java/tuning/ownership/TuningKnobOwner.java` - Ownership enum with `GRAPH_WORKLOAD`, `PLATFORM_DTYPE`, and `OBSOLETE`.
- `src/main/java/tuning/ownership/TuningKnobOwnership.java` - Central matrix and validation helpers.
- `src/main/java/tuning/candidate/graph/GraphRuntimePolicyVariant.java` - Carries declared graph knob assignments.
- `src/main/java/tuning/candidate/graph/GraphAutotuneCandidateSpace.java` - Validates graph assignments and emits ownership metadata.
- `src/main/java/tuning/candidate/graph/GraphPolicyMutators.java` - Declares graph knob assignments for standard and relevant research variants.
- `src/main/java/tuning/calibration/family/CalibrationFamilyRegistry.java` - Validates calibration assignments against platform/dtype ownership.
- `src/test/java/TuningKnobOwnershipTest.java` - Matrix and cross-owner rejection tests.
- `src/test/java/CalibrationCandidateOwnershipTest.java` - Platform/dtype owner assertions for generated calibration candidates.
- `src/test/java/GraphAutotuneCandidateSpaceTest.java` - Graph ownership metadata and validation coverage.
- `src/test/java/tuning/candidate/graph/GraphAutotuneCandidateSpaceTest.java` - Package-level graph ownership coverage.

## Decisions Made

- Kept ownership keyed by the existing persisted/reporting knob strings so the same registry validates runtime profile, graph policy, and metadata-facing candidate changes.
- Included all three accelerator buffer backends in graph autotune buffer-mode assignments so Metal/CUDA/OpenCL buffer mode stays graph/workload-owned consistently.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Verification

- `./gradlew test --tests TuningKnobOwnershipTest`
- `./gradlew test --tests CalibrationCandidateOwnershipTest --tests GraphAutotuneCandidateSpaceTest --tests TuningKnobOwnershipTest`

## Self-Check: PASSED

- `TuningKnobOwnership.java` contains the required owner lookup and graph/platform validation helpers.
- Graph candidate generation calls `validateGraphWorkload(...)`.
- Calibration family validation calls `validatePlatformDtype(...)`.
- Targeted Gradle ownership suite passes.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Wave 2 can build strict platform profile IO on top of the ownership boundary established here.

---
*Phase: 04-tuning-and-profile-ownership-audit*
*Completed: 2026-04-30*
