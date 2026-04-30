# Phase 12 Pattern Map: Fused GPU Region Execution

**Phase:** 12 - Fused GPU Region Execution
**Date:** 2026-04-30
**Status:** Complete

## Pattern Mapping Complete

## Files To Modify Or Extend

| Planned file | Role | Closest existing analog | Notes |
|---|---|---|---|
| `src/main/java/backend/accelerator/lowering/GpuCompoundPatternType.java` | Shared compound pattern taxonomy | `GpuLoweringOperationFamily.java` | Keep backend-neutral and enum-based for stable traces. |
| `src/main/java/backend/accelerator/lowering/GpuCompoundRegionSummary.java` | Traceable compound summary record | `GpuLoweringCoverageEntry.java`, `AcceleratorSubgraphLoweringResult.java` | Immutable record, no backend-native handles. |
| `src/main/java/backend/accelerator/lowering/GpuCompoundPatternDetector.java` | Pattern classifier | `AcceleratorSubgraphLowerer.java` | Should inspect compiled nodes and existing DAG specs, not CPU fused plans. |
| `src/main/java/backend/accelerator/lowering/GpuCompoundLoweringArtifact.java` | Optional lowered-unit artifact | `backend.cpu.fused.plan.FusedOperationPreparation` | Implement `LoweredUnitArtifact` without depending on CPU fused internals. |
| `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLoweringResult.java` | Carry summary with existing DAG result | existing record | Preserve compatibility constructors used by tests. |
| `src/main/java/backend/accelerator/lowering/GpuLoweringUnsupportedReason.java` | Stable reason vocabulary | existing enum | Add compound-specific reasons only if tests assert them. |
| `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java` | Matrix rows for FUSED/reduction-adjacent | existing class | Keep backend-specific rows explicit. |
| `src/main/java/backend/metal/lowering/MetalPartitionPlan.java` | Metal selected plan summary | current record | Summary can be derived from `lowering().compoundSummary()`. |
| `src/main/java/backend/cuda/lowering/CudaGpuPartitionPlan.java` | CUDA selected plan summary | current record | May need a summary field because CUDA currently stores only `dagSpec`. |
| `src/main/java/backend/metal/lowering/MetalRegionLowerer.java` | Lowered execution unit family/artifact | current lowerer | Should attach compound summary while preserving existing family selection. |
| `src/main/java/backend/cuda/lowering/CudaRegionLowerer.java` | Lowered execution unit family/artifact | current lowerer | Same contract as Metal. |
| `src/main/java/backend/accelerator/exec/PreparedAcceleratorExecutable.java` | Common summary accessor | current interface | Add default method to avoid breaking non-compound implementations. |
| `src/main/java/backend/metal/exec/PreparedMetalExecutable.java` | Runtime summary exposure and required-mode behavior | current executable | Do not add CPU array fallback between supported compound nodes. |
| `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java` | Runtime summary exposure and required-mode behavior | current executable | CUDA can be narrower but must be explicit. |
| `src/main/java/graph/execution/PreparedExecution.java` | Run trace attributes | existing accelerator trace attributes | Add generic `gpuCompound*` attributes beside buffer stats. |
| `src/test/java/backend/accelerator/lowering/GpuCompoundPatternDetectorTest.java` | Shared classifier tests | `AcceleratorSubgraphLowererTest.java`, `GpuLoweringCoverageMatrixTest.java` | First tests for summary classification and explicit FUSED rejection. |
| `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java` | Metal lowering tests | current tests | Add linear+bias+activation and summary artifact assertions. |
| `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java` | CUDA lowering tests | current tests | Match Metal where CUDA supports the minimal subset. |
| `src/test/java/PreparedExecutionBuildTest.java` | Prepared execution and parity tests | current accelerator tests | Assert one accelerator step, full node coverage, and CPU parity. |
| `src/test/java/CompiledGraphTraceTest.java` | Trace evidence | current trace tests | Assert `gpuCompoundPattern` and stable fallback/rejection reasons. |
| `docs/gpu-lowering-coverage.md` | Coverage docs | current Phase 11 doc | Update from deferral-only to Phase 12 supported/rejected compound rows. |
| `docs/graph-optimizer.md` | Architecture docs | current optimizer docs | Explain GPU compound lowering as accelerator DAG execution, not CPU fused ASM. |
| `docs/compute-flow.md` | Runtime docs | current compute flow docs | Document run trace attributes and residency proof. |
| `docs/development.md` | Verification commands | current docs | Add Phase 12 focused checks. |

## Reusable Code Patterns

### Immutable Records For Contracts

Existing lowering records validate nulls and copy lists in constructors. Use the same style for `GpuCompoundRegionSummary`.

### Backend-Neutral First, Backend-Owned Execution Second

`AcceleratorSubgraphLowerer` maps graph semantics to `AcceleratorDagSpec`. Metal and CUDA preparers compile backend-native executables from those DAGs. Phase 12 should keep this split: the summary class describes what was selected, while `MetalMpsGraphBridge` and `CudaGraphBridge` remain responsible for actual native execution.

### Stable Reason Codes

`GpuLoweringCoverageEntry` already rejects inconsistent status/reason pairs. Add compound reasons in the same enum only when they are needed by tests and docs. Do not encode free-form reasons as the only contract.

### Runtime Residency Evidence

`PreparedExecution.buildStepMetadata` already emits accelerator buffer attributes and storage residency attributes. Add compound attributes there instead of inventing a separate trace channel.

## Landmines

- Do not make `Operation.OpType.FUSED` a GPU input format in Phase 12.
- Do not import `backend.cpu.fused.*` into accelerator lowering, Metal, or CUDA code.
- Do not claim CUDA support beyond what the CUDA bridge and buffer binder can execute or capability-gate.
- Do not treat AUTO-mode CPU fallback as success for supported target patterns; required-mode tests must force failure before hidden fallback.
- Do not stage local `profiles/platform/.../tuning/abc/*` files.

