# Phase 12: Fused GPU Region Execution - Context

**Gathered:** 2026-04-30T19:01:24Z
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 12 extends the existing Metal/CUDA partition and lowering pipeline so selected GPU regions can be recognized as safe compound patterns, keep intermediates device-owned, and lower to backend primitive execution without CPU round trips between supported fused operations. It does not replace the CPU fused ASM/vector implementation, does not introduce a public GPU tensor API, and does not own Phase 13 benchmark coverage gates.

</domain>

<decisions>
## Implementation Decisions

### GPU Compound Lowering Model
- **D-01:** Phase 12 should extend the existing `LoweringPipeline`, `MetalRegionLowerer`, `CudaRegionLowerer`, and `AcceleratorSubgraphLowerer` path with GPU compound region lowering. This is not a new parallel CPU-style fusion system.
- **D-02:** A selected Metal/CUDA region may first be recognized or canonicalized as a safe compound pattern, then lowered to backend-neutral accelerator DAG primitives and finally to backend-specific Metal/CUDA execution primitives.
- **D-03:** GPU fusion means "longer device-owned compound accelerator region", not reuse of the CPU fused ASM/vector execution contract.

### Compound Representation
- **D-04:** Use both a pattern summary and a DAG. The pattern summary should expose intent such as `LINEAR_BIAS_ACTIVATION`, `ELEMENTWISE_CHAIN`, or `REDUCTION_ADJACENT` for trace, cost, fallback, and tests.
- **D-05:** Actual execution should continue to lower through accelerator DAG or backend primitive paths. The summary must not become an execution shortcut that bypasses backend legality, dtype, layout, or capability checks.
- **D-06:** Unsupported compound candidates must reject with stable reasons tied to the pattern summary, so Phase 13 can later measure coverage and fallback without reverse-engineering raw node lists.

### Existing CPU FUSED Operation
- **D-07:** `Operation.OpType.FUSED` remains CPU-only for Phase 12. GPU compound regions should arise from normal graph operations inside GPU partitions before or beside CPU fused execution, not by consuming CPU fused ASM internals.
- **D-08:** If a GPU path encounters `Operation.OpType.FUSED`, it must reject explicitly with a stable GPU fused/compound unsupported reason instead of silently using CPU fallback or tensor-array execution.
- **D-09:** Planning may protect CPU fused paths by ensuring GPU compound work does not regress existing CPU fused execution tests or metadata contracts.

### First Supported Patterns
- **D-10:** Phase 12 should implement two minimal compound GPU patterns in parallel: one `linear + bias + activation` target for `GPUFUSE-01`, and one representative elementwise-chain target for `GPUFUSE-02`.
- **D-11:** Reduction-adjacent candidates are third priority. They may be implemented only as a narrow safe subset with CPU parity tests; otherwise they must be explicitly rejected with stable coverage-matrix entries and reason codes for `GPUFUSE-04`.
- **D-12:** Pattern selection should prioritize common NN hot paths that reduce CPU materialization boundaries, not isolated operations that do not lengthen GPU-owned regions.

### Metal/CUDA Parity
- **D-13:** Metal may support a broader compound DAG through MPSGraph if parity and trace contracts hold. CUDA must provide a stable minimum subset rather than broad claims unsupported by the native provider.
- **D-14:** Phase 12 should not claim equal backend coverage when CUDA is narrower. Backend-specific coverage must remain visible in the coverage matrix, planner decisions, traces, and tests.
- **D-15:** The phase should still prove at least one minimal useful compound pattern for CUDA, or explicitly capability-gate it with stable fallback behavior if local native CUDA execution is unavailable.

### Fallback And Trace Contract
- **D-16:** For explicitly supported fused target patterns, region shortening is a failure. If the planner/lowerer promises a supported `linear + bias + activation` or supported elementwise chain, the full pattern must stay in the selected GPU region.
- **D-17:** CPU materialization or tensor-array fallback between operations of a supported fused pattern is a failure, even if the fallback is trace-visible.
- **D-18:** Unsupported or out-of-scope compound candidates may fallback, but only with stable reason codes and visible trace/report evidence. Hidden fallback remains a blocker.

### the agent's Discretion
- Exact Java type names for the compound pattern summary are left to planning. Prefer backend-neutral placement under `backend.accelerator` or lowering/partition packages, with backend-specific execution details under `backend.metal` and `backend.cuda`.
- Planner may choose whether compound pattern recognition happens inside `AcceleratorSubgraphLowerer`, adjacent helper classes, or backend lowerers, provided the result is testable, trace-visible, and does not duplicate CPU fused internals.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Scope
- `.planning/ROADMAP.md` — Phase 12 goal, success criteria, dependencies, and v1.2 sequencing.
- `.planning/REQUIREMENTS.md` — GPUFUSE-01, GPUFUSE-02, GPUFUSE-03, and GPUFUSE-04 acceptance scope.
- `.planning/PROJECT.md` — Project-level accelerator architecture constraints and v1.2 milestone intent.
- `.planning/STATE.md` — Current milestone state and local profile artifact hygiene notes.

### Prior Phase Contracts
- `.planning/phases/09-native-layout-abi-v2/09-VERIFICATION.md` — Layout ABI v2 metadata, capability, and fallback contract already validated.
- `.planning/phases/10-gpu-layout-transform-and-view-path/10-VERIFICATION.md` — Layout/view residency, dense materialization, and conservative CUDA non-dense contracts.
- `.planning/phases/11-gpu-lowering-coverage-matrix/11-CONTEXT.md` — Phase 11 decisions that Phase 12 owns fused GPU compound execution and Phase 13 owns coverage gates.
- `.planning/phases/11-gpu-lowering-coverage-matrix/11-VERIFICATION.md` — Checked-in coverage matrix, Metal/CUDA legality alignment, `LOG_SOFTMAX` lowering, trace/docs closure, and explicit deferred fused-region rows.

### Lowering, Partitioning, And Accelerator Code
- `src/main/java/backend/lowering/LoweringPipeline.java` — Existing region lowering orchestration to extend.
- `src/main/java/backend/metal/lowering/MetalRegionLowerer.java` — Metal region lowering family selection.
- `src/main/java/backend/cuda/lowering/CudaRegionLowerer.java` — CUDA region lowering family selection.
- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java` — Shared accelerator DAG lowering and existing matmul/linear post-op recognition.
- `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java` — Current source of truth for supported/fallback/unsupported GPU operation coverage.
- `src/main/java/backend/accelerator/dag/AcceleratorDagNodeType.java` — Backend-neutral accelerator DAG operation ABI codes.
- `src/main/java/backend/accelerator/dag/AcceleratorPostOpType.java` — Post-op ABI codes currently used by matmul/linear-like lowering.
- `src/main/java/backend/partition/BackendPartitionDescriptorRegistry.java` — Registration point for backend legality adapters and region lowerers.
- `src/main/java/backend/select/DefaultBackendSelectionPolicy.java` — Cost and selection gate that must keep supported compound regions whole.
- `src/main/java/graph/execution/ExecutionState.java` — Device residency and CPU materialization state that proves supported fused patterns stay device-owned.

### CPU Fused Boundary
- `src/main/java/backend/cpu/fused` — CPU fused planning/generated execution package that must remain independent from GPU compound lowering.
- `docs/mechanisms.md` — CPU fused ASM execution model and compile/prepare/runtime boundaries.
- `docs/graph-optimizer.md` — Partition, region, FUSE, and memory planning concepts.
- `docs/adding-tensor-operation.md` — Warning that marking operations fusable is not enough; fused execution support must be explicit and tested.

### Tests And Docs
- `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java` — Metal lowering coverage patterns.
- `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java` — CUDA lowering coverage patterns.
- `src/test/java/PreparedExecutionBuildTest.java` — Prepared selection and backend decision tests.
- `src/test/java/CompiledGraphTraceTest.java` — Trace metadata expectations for selected/rejected accelerator candidates.
- `src/test/java/backend/metal/MetalLayoutAwareDeviceFlowTest.java` — Metal layout-heavy device flow and parity evidence.
- `src/test/java/backend/cuda/exec/CudaLayoutTransformDeviceFlowTest.java` — CUDA layout/view and fallback evidence.
- `docs/gpu-lowering-coverage.md` — Current matrix documentation and Phase 12 fused compound deferral note.
- `docs/development.md` — Focused Gradle and optional native verification commands.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `LoweringPipeline` already runs registered backend lowerers over optimized regions after partition/fusion/memory planning.
- `MetalRegionLowerer` and `CudaRegionLowerer` already classify selected accelerator regions into graph or fused-elementwise lowering families.
- `AcceleratorSubgraphLowerer` already builds backend-neutral DAG specs and has existing `MATMUL`/`LINEAR` plus bias/post-op recognition that can seed compound pattern detection.
- `GpuLoweringCoverageMatrix` already has `DEFERRED_FUSED_REGION` rows for `FUSED` and normalization pieces; Phase 12 can turn narrow supported subsets into explicit supported/fallback entries.
- `ExecutionState` already records device bindings, device-owned residency, and CPU materialization traces needed to prove no CPU boundary appears between supported fused ops.

### Established Patterns
- Public `Tensor` stays logical; backend residency belongs in compile/prepare/execute runtime state.
- Backend-neutral contracts live under shared accelerator/lowering packages; native/provider details stay in Metal/CUDA packages.
- Unsupported accelerator behavior must be visible in reason codes, traces, tests, and benchmark reports.
- Optional native Metal/CUDA checks remain capability-gated; portable Java tests must prove planner/lowering/trace contracts.
- Local tuning/profile artifacts under `profiles/platform/.../tuning/abc/*` are dirty and must remain unstaged unless intentionally updating canonical profiles.

### Integration Points
- Region legality and selection: Metal/CUDA legality adapters, `BackendPartitionDescriptorRegistry`, and `DefaultBackendSelectionPolicy`.
- Compound lowering: `AcceleratorSubgraphLowerer`, `MetalRegionLowerer`, `CudaRegionLowerer`, `LoweredExecutionUnit`, and `LoweringFamily`.
- Prepared execution: Metal/CUDA preparers and executables must consume compound region metadata without CPU materialization between supported nodes.
- Trace/report evidence: `CompiledGraphTraceTest`, prepare/backend selection traces, accelerator execution traces, and CPU materialization traces.

</code_context>

<specifics>
## Specific Ideas

- The desired model is a region-level lowering pipeline: selected GPU region -> compound pattern recognition/fusion -> pattern summary plus accelerator DAG -> backend primitive lowering -> Metal MPSGraph or CUDA native primitive execution.
- The first pass should prove two target patterns: `linear + bias + activation` and a representative elementwise chain.
- For explicitly supported target patterns, shortening the region is a failure because the phase goal is keeping the hot path device-owned, not only making fallback visible.

</specifics>

<deferred>
## Deferred Ideas

- Full benchmark coverage ratios, workload comparisons, and regression gates remain Phase 13 scope.
- Broad universal operation, dtype, rank, layout, and provider coverage remains future accelerator coverage beyond the v1.2 target matrix.
- Sharing CPU `Operation.OpType.FUSED` internals with GPU compound execution is deferred; Phase 12 keeps CPU fused ASM independent.

</deferred>

---

*Phase: 12-fused-gpu-region-execution*
*Context gathered: 2026-04-30T19:01:24Z*
