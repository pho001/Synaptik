---
phase: 12-fused-gpu-region-execution
plan: "02"
subsystem: accelerator-lowering
tags: [gpu, metal, cuda, lowering, compound-regions]

requires:
  - phase: 12-fused-gpu-region-execution
    provides: shared GPU compound pattern contract from 12-01
provides:
  - LINEAR_BIAS_ACTIVATION compound summary through accelerator lowering
  - Metal and CUDA partition/lowered-unit compound metadata
  - Prepared execution evidence and required-mode no-hidden-fallback coverage
affects: [gpu-lowering, metal-runtime, cuda-runtime, prepared-execution]

tech-stack:
  added: []
  patterns:
    - backend-aware accelerator subgraph lowering
    - compound GPU summary carried beside DAG lowering artifacts

key-files:
  created: []
  modified:
    - src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java
    - src/main/java/backend/cuda/lowering/CudaGpuPartitionPlan.java
    - src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java
    - src/main/java/backend/cuda/lowering/CudaRegionLowerer.java
    - src/main/java/backend/cuda/exec/PreparedCudaExecutable.java
    - src/main/java/backend/cuda/prepare/CudaGpuNodePreparer.java
    - src/main/java/backend/metal/lowering/MetalRegionLegalityAdapter.java
    - src/main/java/backend/metal/lowering/MetalRegionLowerer.java
    - src/main/java/backend/metal/exec/PreparedMetalExecutable.java
    - src/test/java/backend/accelerator/lowering/AcceleratorSubgraphLowererTest.java
    - src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java
    - src/test/java/backend/metal/lowering/MetalRegionLowererTest.java
    - src/test/java/PreparedExecutionBuildTest.java

key-decisions:
  - "AcceleratorSubgraphLowerer now has a backend-aware overload while preserving the original compatibility signature."
  - "CUDA partition plans carry GpuCompoundRegionSummary directly, matching Metal's lowering-backed plan metadata."
  - "Prepared Metal/CUDA executables expose compound summaries so runtime preparation can be audited without CPU fused internals."

patterns-established:
  - "Supported compound GPU patterns are metadata on accelerator DAG regions, not CPU fused Operation.OpType.FUSED reuse."
  - "LoweredExecutionUnit artifacts carry compound summaries only for non-NONE compound regions."

requirements-completed: [GPUFUSE-01, GPUFUSE-03]

duration: 6 min
completed: 2026-04-30
---

# Phase 12 Plan 02: Linear Bias Activation Compound GPU Summary

**Metal and CUDA now recognize `linear + bias + activation` as one accelerator DAG-backed `LINEAR_BIAS_ACTIVATION` compound region.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-30T19:23:00Z
- **Completed:** 2026-04-30T19:29:06Z
- **Tasks:** 3
- **Files modified:** 13

## Accomplishments

- Added backend-aware compound detection in `AcceleratorSubgraphLowerer` after DAG and legacy matmul metadata are built.
- Propagated `LINEAR_BIAS_ACTIVATION` summaries through Metal and CUDA partition plans, region lowerers, lowered-unit artifacts, and prepared executables.
- Added focused lowering, region, prepared execution, parity, and required buffer-mode tests for `input.linear(weight, bias).relu()`.

## Task Commits

1. **Tasks 1-3: linear+bias+activation tests, wiring, and prepared execution checks** - `c4eeb7f` (`feat(12-02)`)

**Plan metadata:** committed separately.

## Files Created/Modified

- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java` - added backend-aware lowering overload and detector call.
- `src/main/java/backend/cuda/lowering/CudaGpuPartitionPlan.java` - added compound summary storage with compatibility constructor.
- `src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java` - passes CUDA backend into lowering and stores summary.
- `src/main/java/backend/cuda/lowering/CudaRegionLowerer.java` - attaches `GpuCompoundLoweringArtifact` to non-NONE compound lowered units.
- `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java` - exposes prepared CUDA compound summary metadata.
- `src/main/java/backend/cuda/prepare/CudaGpuNodePreparer.java` - passes CUDA plan summary into prepared executable.
- `src/main/java/backend/metal/lowering/MetalRegionLegalityAdapter.java` - passes Metal backend into lowering.
- `src/main/java/backend/metal/lowering/MetalRegionLowerer.java` - attaches `GpuCompoundLoweringArtifact` to non-NONE compound lowered units.
- `src/main/java/backend/metal/exec/PreparedMetalExecutable.java` - exposes prepared Metal compound summary metadata.
- `src/test/java/backend/accelerator/lowering/AcceleratorSubgraphLowererTest.java` - verifies LINEAR/RELU DAG and post-op summary.
- `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java` - verifies Metal plan and lowered-unit compound artifacts.
- `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java` - verifies CUDA plan and lowered-unit compound artifacts.
- `src/test/java/PreparedExecutionBuildTest.java` - verifies prepared Metal/CUDA compound regions, parity, and required-mode failure.

## Decisions Made

- Kept compound fusion as accelerator DAG metadata rather than introducing a CPU fused ASM/vector dependency.
- Preserved old `tryLower(subgraph, context)` call sites by delegating to the backend-aware overload.
- Added concrete prepared-executable summary accessors now because Task 3 needed executable-level audit evidence.

## Deviations from Plan

None - plan executed as specified. Prepared executable metadata wiring was necessary to satisfy Task 3's executable summary assertions.

## Issues Encountered

None.

## Verification

- `./gradlew classes` - passed
- `./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest` - passed
- `./gradlew test --tests PreparedExecutionBuildTest` - passed
- Acceptance `rg` checks for test names, `LINEAR_BIAS_ACTIVATION`, compound summary wiring, and required buffer-mode diagnostic - passed
- `rg -n "backend\.cpu\.fused" src/main/java/backend/accelerator src/main/java/backend/metal src/main/java/backend/cuda src/test/java/backend/accelerator src/test/java/backend/metal src/test/java/backend/cuda` - no matches

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for 12-03. The next plan can build on executable-level compound metadata to expose elementwise-chain residency and trace evidence.

---
*Phase: 12-fused-gpu-region-execution*
*Completed: 2026-04-30*
