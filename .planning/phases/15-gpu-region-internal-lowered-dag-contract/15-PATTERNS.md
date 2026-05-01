# Phase 15 Pattern Map: GPU Region Internal Lowered DAG Contract

**Phase:** 15 - GPU Region Internal Lowered DAG Contract
**Date:** 2026-05-01
**Status:** Complete

## Pattern Mapping Complete

## Files To Modify Or Extend

| Planned file | Role | Closest existing analog | Notes |
|---|---|---|---|
| `src/main/java/backend/accelerator/lowering/GpuLoweredRegionManifest.java` | Top-level Java-side GPU lowered-region manifest | `GpuCompoundRegionSummary.java` | Backend-neutral Java record with null-normalizing constructor and no native ABI role. |
| `src/main/java/backend/accelerator/lowering/GpuLoweredRegionOriginalOp.java` | Original compiled-op summary and lowered primitive mapping | `GpuLoweringCoverageEntry.java` | Immutable record with op id/type, input/output ids, dtype/layout summary, and primitive ids. |
| `src/main/java/backend/accelerator/lowering/GpuLoweredPrimitiveManifest.java` | Lowered backend primitive summary | `AcceleratorDagNode.java` | Wraps primitive id/type/source op ids/value assumptions without altering `AcceleratorDagNode`. |
| `src/main/java/backend/accelerator/lowering/GpuLoweredRegionValueAssumption.java` | Dtype/layout/storage assumption record | `AcceleratorDagInput.java` | Stable value-level metadata for region inputs, primitives, and outputs. |
| `src/main/java/backend/accelerator/lowering/GpuLoweredRegionRejection.java` | Rejection/fallback/materialization attribution | `GpuLoweringCoverageEntry.java` | Uses `GpuLoweringUnsupportedReason` plus level enum/string for original op, primitive, fused subpattern, or boundary. |
| `src/main/java/backend/accelerator/lowering/GpuLoweredRegionCandidateSpan.java` | Candidate shortening metadata | `PartitionDecisionTrace.CandidateCostTrace` | Records original candidate node ids, accepted node ids, rejected node or primitive id, and reason. |
| `src/main/java/backend/accelerator/lowering/GpuLoweredRegionManifestRenderer.java` | Stable compact text renderer | `TextGpuCoverageTriageReportRenderer.java` | Exact headings and deterministic ordering for traces/reports. |
| `src/main/java/backend/accelerator/lowering/GpuLoweringUnsupportedReason.java` | Stable reason vocabulary | Existing enum | Add DAG-level reason codes; do not introduce free-form reason enums. |
| `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLoweringResult.java` | Carry manifest beside DAG and compound summary | Existing lowering result | Add manifest with constructor overloads for compatibility. |
| `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java` | Build manifest from subgraph, DAG, context, and compound summary | Existing lowerer | Derive original-op and primitive mapping from existing lowering evidence. |
| `src/main/java/graph/execution/trace/BackendSelectionDecisionTrace.java` | Prepare-time manifest attachment | Existing decision trace record | Add optional manifest field with overloads for existing call sites. |
| `src/main/java/backend/select/DefaultBackendSelectionPolicy.java` | Attach selected plan manifest to selected trace decision | Existing selection policy | Use `PartitionPlan` implementations to expose selected manifests. |
| `src/main/java/backend/metal/lowering/MetalPartitionPlan.java` | Metal selected plan manifest access | Existing Metal plan record | Expose manifest through `lowering().manifest()`; no native ABI change. |
| `src/main/java/backend/cuda/lowering/CudaGpuPartitionPlan.java` | CUDA selected plan manifest access | Existing CUDA plan record | Add manifest field or migrate to `AcceleratorSubgraphLoweringResult` without breaking current CUDA tests. |
| `src/main/java/tuning/benchmark/report/TextBenchmarkReportRenderer.java` | Human-readable selected manifest reporting | Existing renderer | Add stable lowered-region block for selected GPU backend decisions. |
| `src/main/java/tuning/benchmark/report/JsonBenchmarkReportRenderer.java` | JSON selected manifest reporting | Existing renderer | Add stable `gpuLoweredRegionManifest` object. |
| `src/test/java/backend/accelerator/lowering/GpuLoweredRegionManifestTest.java` | Manifest model tests | `GpuCompoundPatternDetectorTest.java` | Assert normalization, bidirectional mapping, candidate span, and renderer headings. |
| `src/test/java/backend/accelerator/lowering/AcceleratorSubgraphLowererTest.java` | Lowerer integration tests | Existing lowerer tests | Add LOG_SOFTMAX multi-primitive mapping and linear/bias/activation fused summary assertions. |
| `src/test/java/CompiledGraphTraceTest.java` | E2E trace contract tests | Existing trace tests | Assert prepare trace exposes manifest and run trace stays compact. |
| `docs/gpu-lowered-region-manifest.md` | Developer contract docs | `docs/gpu-lowering-coverage.md` | Explain manifest fields, reason levels, examples, and non-native-ABI boundary. |

## Reusable Code Patterns

### Java Record Contracts

Use Java records with compact constructors that copy lists/maps and normalize nulls. This matches `GpuCompoundRegionSummary`, `BackendSelectionTrace`, and report-domain records.

### Constructor Compatibility

Trace and lowering records are widely used in tests. Add overloads that preserve existing call sites and default manifest fields to an empty or generated value.

### Backend-Neutral Core With Backend Extensions

Shared fields should live in the manifest core. Backend-specific data should be exposed as `Map<String, String> backendExtensions` so Metal and CUDA do not fork the trace contract.

### Prepare Trace As Source Of Truth

Attach structured selected manifests to `BackendSelectionDecisionTrace`. Report renderers can read from prepare trace. Run trace should only include compact execution attributes from `PreparedExecution` when needed.

## Landmines

- Do not add manifest/debug fields directly to `AcceleratorDagSpec`.
- Do not change Metal or CUDA native ABI in Phase 15.
- Do not expose device residency through public `Tensor`.
- Do not make rejection reasons free-form strings where a stable enum code is required.
- Do not reuse CPU `Operation.OpType.FUSED` or CPU fused ASM internals for GPU fusion metadata.
- Do not commit local tuning profile artifacts under `profiles/platform/.../tuning/abc/*`.
