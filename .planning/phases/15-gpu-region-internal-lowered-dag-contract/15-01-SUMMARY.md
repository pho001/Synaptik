---
phase: 15-gpu-region-internal-lowered-dag-contract
plan: "01"
subsystem: accelerator-lowering
tags: [gpu, lowered-dag, manifest, trace, reason-codes]
requires:
  - phase: 14-coverage-gap-triage-and-hot-path-targets
    provides: coverage-driven target context for GPUDAG work
provides:
  - Java-side GPU lowered-region manifest records
  - Stable DAG-level GPU lowering reason codes
  - Manifest contract tests for mapping, assumptions, rejections, and shortening
affects: [phase-15, phase-16, phase-17, phase-18, phase-19, phase-20]
tech-stack:
  added: []
  patterns: [immutable Java records, stable reason enums, backend-neutral accelerator metadata]
key-files:
  created:
    - src/main/java/backend/accelerator/lowering/GpuLoweredRegionManifest.java
    - src/main/java/backend/accelerator/lowering/GpuLoweredRegionOriginalOp.java
    - src/main/java/backend/accelerator/lowering/GpuLoweredPrimitiveManifest.java
    - src/main/java/backend/accelerator/lowering/GpuLoweredRegionValueAssumption.java
    - src/main/java/backend/accelerator/lowering/GpuLoweredRegionRejection.java
    - src/main/java/backend/accelerator/lowering/GpuLoweredRegionCandidateSpan.java
    - src/test/java/backend/accelerator/lowering/GpuLoweredRegionManifestTest.java
  modified:
    - src/main/java/backend/accelerator/lowering/GpuLoweringUnsupportedReason.java
key-decisions:
  - "The lowered-region manifest is Java-side metadata beside the native bridge DAG."
  - "DAG rejection and shortening attribution uses GpuLoweringUnsupportedReason, not free-form reason strings."
patterns-established:
  - "Manifest records copy lists/maps and normalize nulls in compact constructors."
  - "Candidate shortening evidence is represented explicitly through GpuLoweredRegionCandidateSpan."
requirements-completed: [GPUDAG-01, GPUDAG-02, GPUDAG-03]
duration: 12 min
completed: 2026-05-01
---

# Phase 15 Plan 01: Manifest Model And Reason Vocabulary Summary

**Java-side GPU lowered-region manifest records with stable DAG-level reason codes for original-op, primitive, boundary, fused-subpattern, and candidate-shortening evidence**

## Performance

- **Duration:** 12 min
- **Started:** 2026-05-01T07:40:00Z
- **Completed:** 2026-05-01T07:52:00Z
- **Tasks:** 4
- **Files modified:** 8

## Manifest model and reason vocabulary

Plan 15-01 added the backend-neutral manifest model used by the rest of Phase 15. The model records selected backend, region id, ordered original nodes, external inputs, outputs, original op metadata, lowered primitive metadata, input/output dtype-layout-storage assumptions, fused summary placeholder, rejection records, candidate-shortening evidence, and backend extension metadata.

The stable DAG reason vocabulary now includes:

- `DAG_PRIMITIVE_UNSUPPORTED`
- `DAG_REGION_BOUNDARY_MATERIALIZATION`
- `DAG_CANDIDATE_SHORTENED`
- `DAG_FUSED_SUBPATTERN_REJECTED`

## Task Commits

1. **Task 1: Add failing-first manifest model tests** - `1818924` (`test(15-01): add gpu lowered region manifest contract tests`)
2. **Tasks 2-3: Add manifest records and stable DAG reason vocabulary** - `5b15ade` (`feat(15-01): add gpu lowered region manifest model`)
3. **Task 4: Record manifest model summary and hygiene evidence** - this metadata commit

## Verification

| Command | Result |
|---------|--------|
| `./gradlew test --tests backend.accelerator.lowering.GpuLoweredRegionManifestTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest` | Passed |
| `rg -n "record GpuLoweredRegionManifest\|record GpuLoweredRegionOriginalOp\|record GpuLoweredPrimitiveManifest\|record GpuLoweredRegionValueAssumption\|record GpuLoweredRegionRejection\|record GpuLoweredRegionCandidateSpan\|backendExtensions\|selectedRegionLength\|loweredPrimitiveIds\|sourceOriginalNodeIds\|originalCandidateNodeIds" src/main/java/backend/accelerator/lowering` | Passed |
| `rg -n "DAG_PRIMITIVE_UNSUPPORTED\|DAG_REGION_BOUNDARY_MATERIALIZATION\|DAG_CANDIDATE_SHORTENED\|DAG_FUSED_SUBPATTERN_REJECTED" src/main/java/backend/accelerator/lowering/GpuLoweringUnsupportedReason.java src/main/java/backend/accelerator/lowering/GpuLoweredRegionRejection.java src/main/java/backend/accelerator/lowering/GpuLoweredRegionCandidateSpan.java src/test/java/backend/accelerator/lowering/GpuLoweredRegionManifestTest.java` | Passed |

## Requirement Coverage

- `GPUDAG-01`: Manifest can describe a GPU region as ordered original operations plus ordered lowered primitives.
- `GPUDAG-02`: Manifest records original operation ids/types, primitive ids/types, backend id, dtype/layout assumptions, fused summary, and region length.
- `GPUDAG-03`: Manifest rejection and candidate-span records use stable reason codes for primitive, boundary, fused subpattern, and shortening attribution.

## Decision Coverage

Covered decisions: `D-01`, `D-02`, `D-03`, `D-05`, `D-06`, `D-07`, `D-08`, `D-13`, `D-14`, `D-15`, and `D-16`.

## Deviations from Plan

Tasks 2 and 3 were committed together because the manifest records and new enum constants are compile-time coupled: the tests and records reference the new stable reason codes directly. No scope was added beyond the planned files.

**Total deviations:** 1 procedural merge of adjacent planned tasks. **Impact:** No behavioral scope change.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Artifact Hygiene

`profiles/platform/.../tuning/abc/* remained unstaged`.

## Next Phase Readiness

Plan 15-02 can build manifests from `AcceleratorSubgraphLowerer` and expose them through Metal/CUDA selected plans.

## Self-Check: PASSED

---
*Phase: 15-gpu-region-internal-lowered-dag-contract*
*Completed: 2026-05-01*
