---
phase: 15-gpu-region-internal-lowered-dag-contract
plan: "02"
subsystem: accelerator-lowering
tags: [gpu, lowered-dag, manifest, metal, cuda]
requires:
  - phase: 15-gpu-region-internal-lowered-dag-contract
    provides: Plan 15-01 manifest model and stable DAG reason vocabulary
provides:
  - Manifest-bearing AcceleratorSubgraphLoweringResult
  - Shared manifest construction from AcceleratorSubgraphLowerer
  - Metal and CUDA selected plan manifest exposure
affects: [phase-15, phase-16, phase-17, phase-18, phase-19, phase-20]
tech-stack:
  added: []
  patterns: [shared lowerer manifest builder, backend-neutral plan metadata]
key-files:
  created: []
  modified:
    - src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLoweringResult.java
    - src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java
    - src/main/java/backend/metal/lowering/MetalPartitionPlan.java
    - src/main/java/backend/cuda/lowering/CudaGpuPartitionPlan.java
    - src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java
    - src/test/java/backend/accelerator/lowering/AcceleratorSubgraphLowererTest.java
    - src/test/java/backend/metal/lowering/MetalRegionLowererTest.java
    - src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java
key-decisions:
  - "Manifest construction derives primitive metadata from AcceleratorDagSpec.nodes()."
  - "Metal and CUDA selected plans expose the same GpuLoweredRegionManifest contract."
patterns-established:
  - "Lowerer-built manifest uses deterministic primitive ids p0, p1, p2 in DAG order."
  - "Backend-specific details use backendExtensions instead of backend-specific manifest forks."
requirements-completed: [GPUDAG-01, GPUDAG-02]
duration: 18 min
completed: 2026-05-01
---

# Phase 15 Plan 02: Manifest Construction And Backend Plan Exposure Summary

**Shared accelerator lowering now constructs selected GPU lowered-region manifests and exposes them through Metal and CUDA partition plans**

## Performance

- **Duration:** 18 min
- **Started:** 2026-05-01T07:52:00Z
- **Completed:** 2026-05-01T08:10:00Z
- **Tasks:** 5
- **Files modified:** 9

## Manifest construction and backend plan exposure

`AcceleratorSubgraphLoweringResult` now carries a non-null `GpuLoweredRegionManifest`. `AcceleratorSubgraphLowerer` builds that manifest from the selected `AcceleratorSubgraphSpec`, actual `AcceleratorDagSpec`, compiled node context, and `GpuCompoundRegionSummary`.

The manifest builder records deterministic primitive ids (`p0`, `p1`, ...), primitive types from `AcceleratorDagNodeType`, source original node ids, original op metadata, input/output assumptions, selected region length, fused summary, and backend extension key `dagNodeCount`.

Metal and CUDA plan objects now expose the shared manifest contract:

- `MetalPartitionPlan.manifest()` returns `lowering.manifest()`.
- `CudaGpuPartitionPlan` carries `GpuLoweredRegionManifest manifest`.
- `CudaGpuRegionLegalityAdapter` passes `lowering.manifest()` when building the CUDA plan.

## Task Commits

1. **Task 1: Add lowerer manifest integration tests** - `4ab7de6` (`test(15-02): add accelerator lowerer manifest tests`)
2. **Tasks 2-3: Carry and build manifests in shared lowering** - `b2984bb` (`feat(15-02): build lowered region manifests`)
3. **Task 4: Expose manifests through Metal and CUDA selected plans** - `54018b9` (`feat(15-02): expose lowered region manifests in gpu plans`)
4. **Task 5: Record manifest construction summary and hygiene evidence** - this metadata commit

## Verification

| Command | Result |
|---------|--------|
| `./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.accelerator.lowering.GpuLoweredRegionManifestTest` | Passed |
| `./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest` | Passed |
| `./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest` | Passed |
| `rg -n "GpuLoweredRegionManifest\|manifest\\(\\)\|ComputeBackend.GPU_METAL\|ComputeBackend.GPU_CUDA\|lowering.manifest\\(\\)" src/main/java/backend/metal/lowering src/main/java/backend/cuda/lowering src/test/java/backend/metal/lowering/MetalRegionLowererTest.java src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java` | Passed |

## Requirement Coverage

- `GPUDAG-01`: Selected GPU regions now carry multi-op and multi-primitive manifest evidence from shared lowering.
- `GPUDAG-02`: Manifest construction records original op metadata, lowered primitive metadata, backend id, dtype/layout assumptions, fused summary, and selected region length.

## Decision Coverage

Covered decisions: `D-01`, `D-02`, `D-03`, `D-04`, `D-13`, `D-14`, `D-15`, and `D-16`.

## Deviations from Plan

Tasks 2 and 3 were committed together because adding the `manifest` accessor and building it in the shared lowerer were compile-time coupled by the new lowerer tests. No native Metal/CUDA ABI signatures were changed.

**Total deviations:** 1 procedural merge of adjacent planned tasks. **Impact:** No behavioral scope change.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Artifact Hygiene

`profiles/platform/.../tuning/abc/* remained unstaged`.

## Next Phase Readiness

Plan 15-03 can attach selected manifests to prepare/backend-selection traces and benchmark renderers.

## Self-Check: PASSED

---
*Phase: 15-gpu-region-internal-lowered-dag-contract*
*Completed: 2026-05-01*
