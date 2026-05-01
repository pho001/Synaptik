# Phase 19: Multi-Op GPU Region Execution - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning
**Source:** Auto context from roadmap, requirements, Phase 14 targets, Phase 15 manifest contract, Phase 16 dtype residency, Phase 17 lowering coverage, Phase 18 fusion metadata, and codebase inspection.

<domain>
## Phase Boundary

Phase 19 turns the v1.3 contracts into runtime execution: a selected Metal or CUDA GPU region should execute as a longer device-owned lowered region containing multiple original operations, layout/view steps, dtype-resident intermediates, elementwise chains, epilogue/fusion subpatterns, and selected softmax-ish or normalization primitives when legality gates allow it.

This phase is not a public GPU tensor API, not a broad vendor-library routing phase, and not a claim of universal native operation coverage. Unsupported internal steps must split or reject the region with stable reasons before execution. Tensor-array bridge execution remains visible fallback evidence and must not count as native buffer GPU coverage.

</domain>

<decisions>
## Implementation Decisions

### Region Execution Granularity
- **D-01:** Treat one selected GPU partition as one backend-owned lowered region/executable when all internal primitives are supported. The planner/lowerer may describe multiple original operations and primitive units inside the region, but runtime should avoid decomposing supported interiors into Java-side per-op CPU-visible steps.
- **D-02:** Unsupported internal operations should split, shorten, or reject the candidate before execution with manifest candidate-span and rejection metadata. Do not silently run a CPU step inside an otherwise selected GPU region.
- **D-03:** Keep `LoweringFamily.METAL_GRAPH_REGION` and `LoweringFamily.CUDA_GRAPH_REGION` as the default multi-op family shape unless a narrowly proven fused family is required for an already-supported subpattern.

### Internal Device Handoff Semantics
- **D-04:** Supported internal values remain device-owned inside the native DAG or device buffer binding path. CPU materialization is allowed only at a true CPU consumer, graph output publication, or gradient publication boundary.
- **D-05:** `ExecutionState` and device buffer bindings are the residency boundary for Java-visible values. Public `Tensor` remains logical; Phase 19 must not add user-visible device tensor handles or device residency API.
- **D-06:** `AcceleratorPreparedInputResolver` and CPU fallback prepared-input paths must not create `ACCELERATOR_PREPARED_INPUT` CPU materialization for supported region interiors. Any such trace inside a supported multi-op region is a Phase 19 failure unless it is an explicit fallback/rejection path.

### Primitive Coverage Priority
- **D-07:** Start from already supported lowered primitives and measured hot paths: matmul/linear, layout/view steps, elementwise chains, epilogue subpatterns, and `SOFTMAX`/`LOG_SOFTMAX` style flows.
- **D-08:** For normalization, reduction, conv, and loss-adjacent pieces, Phase 19 should consume the Phase 17 support/rejection contract. Add native execution only when the implementation is narrow, legal, and covered by CPU parity plus trace evidence; otherwise preserve stable rejection reasons.
- **D-09:** Do not add broad cuBLAS/cuDNN/MPSGraph routing or a new vendor-library cost model in Phase 19. That stays deferred to `GPULIB-*`.

### Backend Contract
- **D-10:** Planning, manifest, trace, and rejection contracts stay backend-neutral. Metal and CUDA may use backend-specific bridge/executable implementations behind `PreparedMetalExecutable` and `PreparedCudaExecutable`.
- **D-11:** Native ABI changes are not the default Phase 19 path. Any native bridge widening must be additive, capability-gated, portable-test-covered, and must preserve required-mode visible failure.
- **D-12:** CPU fused execution remains isolated. GPU region-internal fusion uses `GpuFusionSubpatternSummary`, lowered primitives, and accelerator DAG metadata; it must not reuse CPU `Operation.OpType.FUSED`, CPU fused ASM, or CPU vector dispatch internals.

### Coverage Evidence
- **D-13:** Use Phase 14 targets as required evidence: `transformer_block_hot_path`, `mlp_classifier_small`, `conv2d_resnet_3x3`, and `layer_norm_small`.
- **D-14:** Implementation should focus on measurable improvements for `transformer_block_hot_path` and `mlp_classifier_small` first. `conv2d_resnet_3x3` and `layer_norm_small` can remain stable visible blockers if their unsupported primitive is outside the safe Phase 19 implementation path.
- **D-15:** Evidence must include selected region length, lowered primitive count, fused subpattern count, CPU materialization count/reasons, device handoff count, and whether runtime used native buffer binding, tensor-array bridge, or CPU fallback.

### the agent's Discretion
The agent may choose exact class decomposition, helper names, and test slicing as long as Phase 19 preserves backend-neutral contracts, visible fallback, CPU parity, and local profile artifact hygiene.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Planning And Requirements
- `.planning/ROADMAP.md` — Phase 19 goal, dependencies, success criteria, and cross-cutting constraints.
- `.planning/REQUIREMENTS.md` — `GPUMULTI-01`, `GPUMULTI-02`, and `GPUMULTI-03`.
- `.planning/PROJECT.md` — project-level accelerator/runtime constraints, public `Tensor` boundary, and milestone direction.
- `.planning/phases/14-coverage-gap-triage-and-hot-path-targets/14-HOT-PATH-TARGETS.md` — source-of-truth hot-path target list and coverage gap categories for Phases 15-20.

### Prior Phase Contracts
- `.planning/phases/15-gpu-region-internal-lowered-dag-contract/15-CONTEXT.md` — locked lowered-region manifest decisions.
- `.planning/phases/15-gpu-region-internal-lowered-dag-contract/15-VERIFICATION.md` — completed `GpuLoweredRegionManifest` contract evidence.
- `.planning/phases/16-dtype-and-storage-residency-expansion/16-CONTEXT.md` — dtype/storage residency decisions for BFLOAT16, INT32, and BOOL.
- `.planning/phases/16-dtype-and-storage-residency-expansion/16-VERIFICATION.md` — completed runtime binding and dtype residency evidence.
- `.planning/phases/17-normalization-reduction-and-loss-adjacent-lowering/17-CONTEXT.md` — lowering coverage and stable rejection decisions for normalization, reductions, softmax-ish, conv, and loss-adjacent operations.
- `.planning/phases/17-normalization-reduction-and-loss-adjacent-lowering/17-VERIFICATION.md` — completed Phase 17 coverage and parity evidence.
- `.planning/phases/18-fused-elementwise-and-epilogue-subregions/18-01-SUMMARY.md` — shared GPU fusion subpattern metadata contract.
- `.planning/phases/18-fused-elementwise-and-epilogue-subregions/18-02-SUMMARY.md` — elementwise chain subregion evidence.
- `.planning/phases/18-fused-elementwise-and-epilogue-subregions/18-03-SUMMARY.md` — matmul/linear epilogue subregion evidence.
- `.planning/phases/18-fused-elementwise-and-epilogue-subregions/18-04-SUMMARY.md` — trace/report evidence and CPU fused isolation.

### Codebase Maps
- `.planning/codebase/STACK.md` — Java/Gradle/native bridge stack and optional Metal/CUDA requirements.
- `.planning/codebase/ARCHITECTURE.md` — compile/prepare/execute layering, accelerator scaffolding, and `ExecutionState` ownership.
- `.planning/codebase/INTEGRATIONS.md` — Metal/MPS and CUDA native bridge boundaries, fallback behavior, and capability constraints.

### Lowering, Prepare, Execution, And Coverage Code
- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java` — shared accelerator DAG lowering, manifest creation, dtype residency evidence, and fusion subpattern population.
- `src/main/java/backend/accelerator/lowering/GpuLoweredRegionManifest.java` — backend-neutral region manifest consumed by traces/reports.
- `src/main/java/backend/metal/lowering/MetalRegionLowerer.java` — current Metal lowered-region unit construction.
- `src/main/java/backend/cuda/lowering/CudaRegionLowerer.java` — current CUDA lowered-region unit construction.
- `src/main/java/backend/metal/prepare/MetalNodePreparer.java` — Metal anchor/interior CPU fallback preparation.
- `src/main/java/backend/cuda/prepare/CudaGpuNodePreparer.java` — CUDA anchor/interior CPU fallback preparation.
- `src/main/java/backend/accelerator/exec/PreparedAcceleratorExecutable.java` — shared prepared accelerator executable SPI.
- `src/main/java/backend/accelerator/exec/PreparedAcceleratorExecutionSupport.java` — shared CPU fallback and runtime tensor resolution behavior.
- `src/main/java/backend/accelerator/exec/AcceleratorPreparedInputResolver.java` — prepared-input resolver that can force CPU-readable inputs.
- `src/main/java/backend/metal/exec/PreparedMetalExecutable.java` — Metal buffer-binding, tensor-array, fallback, and output residency behavior.
- `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java` — CUDA buffer-binding, tensor-array, fallback, and output residency behavior.
- `src/main/java/graph/execution/ExecutionState.java` — per-run residency, device binding, and CPU materialization trace owner.
- `src/main/java/backend/runtime/ExecutionContext.java` — runtime facade for marking device/CPU current and requiring CPU-readable values.
- `src/main/java/tuning/benchmark/report/GpuHotPathCoverageTargets.java` — checked v1.3 hot-path target registry.
- `src/main/java/tuning/workload/StandardWorkloads.java` — workload catalog containing the representative Phase 19 targets.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `AcceleratorSubgraphLowerer` already builds a backend-neutral `AcceleratorDagSpec` for supported multi-node accelerator candidates and records manifests with original ops, lowered primitives, dtype residency evidence, and fused subpatterns.
- `GpuLoweredRegionManifest` already has the fields Phase 19 needs for region length, original operation spans, lowered primitives, fused subpatterns, rejections, candidate shortening, and backend extensions.
- `PreparedMetalExecutable` and `PreparedCudaExecutable` already expose `gpuLoweredRegionManifest()`, buffer-binding decisions, CPU fallback steps, and output device binding publication.
- `GpuHotPathCoverageTargets.defaults()` and `StandardWorkloads.defaultCatalog()` already define the representative workloads needed to prove Phase 19 behavior.

### Established Patterns
- Accelerator execution is partition anchored: anchor nodes receive a prepared accelerator executable and interior nodes are marked as covered by the partition.
- Backend-neutral orchestration belongs under shared accelerator/lowering/prepare/runtime contracts; Metal and CUDA-specific native bridge behavior stays under backend packages.
- Runtime device ownership is represented by `ExecutionState`/`ExecutionContext` and device buffer bindings, not by public `Tensor` API changes.
- Fallback and materialization are trace/report evidence, not hidden control flow. REQUIRED mode should fail visibly when the required buffer/native path is unavailable.

### Integration Points
- Region granularity and legality decisions connect through partition plans, `LoweringPipeline`, `MetalRegionLowerer`, `CudaRegionLowerer`, and backend-specific node preparers.
- Internal CPU materialization risk is concentrated around CPU fallback execution and `AcceleratorPreparedInputResolver.resolveConsumerInputs(...)`, which can call `requireCpuReadable(..., ACCELERATOR_PREPARED_INPUT)`.
- Native buffer execution paths call `attachDeviceBufferBinding(...)` for outputs and mark them `DEVICE_OWNED`; tensor-array paths force CPU-readable input arrays and should remain visible as fallback/non-native coverage.
- Coverage reports and Phase 20 gates should consume manifest/run-trace fields rather than parsing prose-only fallback strings.

</code_context>

<specifics>
## Specific Ideas

- Prefer one compiled native graph/DAG per selected supported region instead of Java-side per-op GPU dispatch.
- Phase 19 should prove "hot path stayed on GPU" by showing no internal CPU materialization between supported steps, not just by showing a faster timing.
- `transformer_block_hot_path` is the primary Phase 19 target. `mlp_classifier_small` is the secondary target because it exercises linear/bias/activation and elementwise/epilogue coverage.
- `conv2d_resnet_3x3` and `layer_norm_small` are still evidence targets, but stable visible blockers are acceptable if the missing primitive belongs to future broad native coverage.

</specifics>

<deferred>
## Deferred Ideas

- Vendor library routing through cuBLAS, cuDNN, MPSGraph, or similar libraries; tracked by `GPULIB-01`.
- Backend-native primitive cost model choosing among custom kernels, vendor libraries, and CPU fallback; tracked by `GPULIB-02`.
- Universal reductions, normalizations, convolution, dynamic shape, sparse, high-rank, and advanced indexing coverage beyond v1.3 targets.
- Public GPU tensor/device API.
- Treating tensor-array bridge execution as native buffer GPU coverage.
- Native CUDA hardware evidence as canonical rather than capability-skipped portable evidence; tracked by `GPUEXT-02`.

</deferred>

---

*Phase: 19-Multi-Op GPU Region Execution*
*Context gathered: 2026-05-01*
