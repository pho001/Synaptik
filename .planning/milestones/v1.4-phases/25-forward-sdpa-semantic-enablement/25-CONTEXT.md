# Phase 25: Forward SDPA Semantic Enablement - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning
**Mode:** `$gsd-discuss-phase 25 --auto`

<domain>
## Phase Boundary

Phase 25 verifies and admits forward `SCALED_DOT_PRODUCT_ATTENTION` for supported Metal/CUDA cases without weakening the milestone rule that a `SUPPORTED` coverage row must mean real backend-lowerable GPU execution or an explicit backend-owned capability rejection.

In scope:

- Lock forward SDPA semantics for scale, rank, mask/no-mask behavior, dtype support, layout legality, numerical tolerance, and backward/training interaction.
- Enable direct forward SDPA only where backend behavior is proven against CPU semantics.
- Keep masked and unmasked SDPA as separate admission cases.
- Preserve explicit rejection for unsupported SDPA forms with stable planner and report reason codes.
- Add transformer attention coverage gates that show whether SDPA stayed on GPU and fail hidden CPU exits.

Out of scope:

- Public GPU tensor/device API.
- Full vendor-library router or cost model for cuDNN/cuBLAS/MPSGraph selection.
- Broad high-rank/dynamic-shape/sparse attention support.
- Claiming CUDA hardware performance without a CUDA-equipped native validation lane.
- Backward SDPA expansion beyond the interaction checks needed to safely admit forward SDPA.

</domain>

<decisions>
## Implementation Decisions

### Admission Strategy

- **D-01:** Treat forward SDPA as a semantics-first gate. Do not move `SCALED_DOT_PRODUCT_ATTENTION` to `SUPPORTED` in `GpuLoweringCoverageMatrix` until the admitted case has CPU parity, backend legality tests, prepared-execution evidence, and trace/report coverage.
- **D-02:** Split admission into explicit cases: unmasked forward SDPA first, masked forward SDPA only after BOOL mask semantics are mapped or intentionally rejected against backend mask semantics.
- **D-03:** Keep direct SDPA legal cases narrow: FLOAT32 query/key/value, rank 3 or 4, dense or already-legal device layouts, descriptor scale preserved exactly enough for documented FLOAT32 tolerance, output shape matching CPU SDPA.
- **D-04:** Unsupported dtype, unsupported layout, invalid rank/shape, unsupported mask kind, and backward-context conflicts must produce stable reason prefixes instead of silently shortening regions.

### Metal Direction

- **D-05:** Metal may use native MPSGraph SDPA only after the current documented scale mismatch is resolved with CPU parity tests. Existing native switch support is not enough to admit planner support.
- **D-06:** Metal masked direct SDPA remains rejected unless the phase proves a correct mask conversion contract. Synaptik public masks are BOOL; current docs/tests say verified MPSGraph SDPA mask input expects floating mask semantics.
- **D-07:** If direct MPSGraph SDPA remains semantically incompatible, the acceptable Metal closure is explicit stable rejection plus decomposed GPU region coverage for supported `matmul -> scale -> softmax -> matmul` fragments where no hidden CPU materialization occurs.

### CUDA Direction

- **D-08:** CUDA must not mirror Metal support claims by default. It either gets a real backend-owned direct SDPA/lowered primitive path for the legal case or remains an explicit `UNSUPPORTED_OPERATION` / capability rejection with report-visible evidence.
- **D-09:** Because local `nvcc` is unavailable in this environment, CUDA native execution evidence should be capability-gated. Portable Java tests still need to assert legality, lowering metadata, unavailable behavior, and stable rejection reasons.

### Backward And Training Interaction

- **D-10:** Forward SDPA admission must explicitly protect training/backward flows. Existing Metal code rejects forward SDPA nodes inside backward regions and avoids backward-pass forward-SDPA bridge execution; Phase 25 should preserve or tighten that behavior.
- **D-11:** Existing `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD` support is not a license to admit direct forward SDPA. Forward parity and backward interaction are separate gates.

### Coverage And Benchmark Evidence

- **D-12:** `transformer_block_hot_path` is the primary Phase 25 coverage target; attention-heavy and long-sequence transformer variants can be used as secondary evidence where tests stay deterministic.
- **D-13:** Coverage reports must show backend execution path, region length, lowered primitive count, fallback/rejection reason, CPU materialization count, and native evidence status for SDPA/attention targets.
- **D-14:** The gate should fail if a legal supported SDPA path regresses to tensor-array bridge or CPU fallback without an explicit capability reason.

### the agent's Discretion

Planner/researcher may choose between direct native SDPA enablement, decomposed GPU region execution, or explicit stable rejection per backend, but only if the chosen path satisfies GPUSDPA-01/02/03 and keeps the source-of-truth matrix honest.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Scope And Requirements

- `.planning/ROADMAP.md` — Phase 25 goal, dependencies, and success criteria.
- `.planning/REQUIREMENTS.md` — GPUSDPA-01, GPUSDPA-02, and GPUSDPA-03.
- `.planning/PROJECT.md` — accelerator residency rules, public Tensor API boundary, v1.4 milestone goal, and out-of-scope constraints.

### Prior Phase Contracts

- `.planning/phases/22-coverage-truth-and-semantics-lock/22-CONTEXT.md` — v1.4 target families and SDPA blocker context.
- `.planning/phases/22-coverage-truth-and-semantics-lock/22-02-SUMMARY.md` — semantics contract introduced with SDPA planner admission still blocked.
- `.planning/phases/22-coverage-truth-and-semantics-lock/22-03-SUMMARY.md` — transformer/attention coverage baselines.
- `.planning/phases/23-forward-reductions-native-execution/23-03-SUMMARY.md` — reductions moved to native coverage, normalization/loss remained explicit blockers.
- `.planning/phases/24-normalization-gpu-lowering/24-VERIFICATION.md` — recent pattern for native/fallback truth, parity, coverage, and profile hygiene closure.

### Docs

- `docs/architecture.md` — direct SDPA support matrix and current Metal rejection rationale.
- `docs/metal-backend.md` — Metal planner/native SDPA notes and mask limitation.
- `docs/gpu-lowering-coverage.md` — current Metal/CUDA attention rows and coverage rule.
- `docs/graph-optimizer.md` — attention lowering rewrite, backend selection legality, and optimizer stage order.
- `docs/tensor-api.md` — public SDPA shape, mask, scale, and gradient semantics.
- `docs/compute-flow.md` — multi-op GPU region and materialization boundary evidence.

### Source Files

- `src/main/java/backend/accelerator/lowering/GpuTargetSemanticsContract.java` — current SDPA semantics blocker contract.
- `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java` — current attention coverage rows.
- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java` — direct and decomposed SDPA DAG lowering hooks.
- `src/main/java/backend/metal/lowering/MetalPartitionSupport.java` — current Metal direct SDPA rejection reasons.
- `src/main/java/backend/metal/exec/PreparedMetalExecutable.java` — Metal bridge execution safety gates.
- `src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java` — CUDA planner legality alignment.
- `src/main/native/apple/synaptik_apple_mps_stub.m` — native Metal MPSGraph DAG switch.
- `src/main/native/cuda/synaptik_cuda_graph_stub.cu` — CUDA native DAG switch where any CUDA SDPA path would need evidence.

### Existing Tests

- `src/test/java/AttentionExecutionTest.java` — CPU/public SDPA behavior and backward coverage.
- `src/test/java/AttentionLoweringTest.java` — optimizer attention lowering behavior.
- `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java` — current direct SDPA rejection and future DAG scale encoding tests.
- `src/test/java/PreparedExecutionBuildTest.java` — prepared execution and SDPA/backward safety evidence.
- `src/test/java/GpuTargetSemanticsContractTest.java` — SDPA semantics contract tests.
- `src/test/java/GpuHotPathCoverageTargetsTest.java` — transformer hot-path target expectations.
- `src/test/java/GpuCoverageSummaryTest.java`, `src/test/java/CompiledGraphTraceTest.java`, `src/test/java/BenchmarkSessionTest.java` — trace/report/fallback visibility.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `TensorAttentionOps` and `operations.linalg.scaledDotProductAttention` already provide public SDPA and operation descriptors with scale/mask options.
- `AttentionLoweringRewrite` already canonicalizes decomposed attention patterns into SDPA primitives.
- `AcceleratorSubgraphLowerer` can already encode direct SDPA DAG nodes and preserves scale bits for future native enablement.
- `GpuTargetSemanticsContract` already has a forward SDPA blocker entry that can be flipped only after semantics verification.
- `GpuHotPathCoverageTargets` already includes `transformer_block_hot_path` with GPUSDPA family expectations.

### Established Patterns

- Matrix support status is source-of-truth but backend legality owns dtype/layout/capability detail.
- Recent Phase 23/24 work moved rows to `SUPPORTED` only after native primitive execution and CPU parity evidence existed.
- CPU remains the correctness oracle for numerically sensitive GPU enablement.
- Device residency belongs in compile/prepare/execute state; public `Tensor` stays logical.
- Local profile/calibration files under `profiles/platform/...` are not phase evidence unless intentionally promoted.

### Integration Points

- Planner admission: `MetalPartitionSupport`, `CudaGpuRegionLegalityAdapter`, `GpuLoweringCoverageMatrix`, `GpuTargetCoverageTruth`.
- Lowering: `AcceleratorSubgraphLowerer`, `AcceleratorDagNodeType.SDPA`, prepared executable manifests.
- Native execution: Metal MPSGraph shim and CUDA graph shim.
- Coverage/reporting: `GpuCoverageSummary`, `GpuHotPathCoverageTargets`, `GpuCoverageRegressionGate`, text/JSON benchmark renderers.
- Tests: focused parity, legality, trace/report, and native Metal gates.

</code_context>

<specifics>
## Specific Ideas

- Preferred implementation shape: admit a narrow unmasked FLOAT32 rank 3/4 direct SDPA path only if parity proves it, then treat masked SDPA as a separate follow-up within this phase if mask semantics can be proven without broad dtype/layout expansion.
- If native direct SDPA cannot be made semantically equivalent in Phase 25, still close the phase by proving explicit rejection plus decomposed supported GPU fragments for attention-style flows where possible.
- Do not use backend claims such as "MPSGraph has SDPA" as evidence. The evidence must be local CPU parity and trace/report behavior.

</specifics>

<deferred>
## Deferred Ideas

- Vendor-library routing and cost model for choosing between direct SDPA, decomposed primitives, cuBLAS/cuDNN/MPSGraph, or custom kernels remains future GPULIB work.
- Full CUDA hardware performance claims remain deferred until a CUDA-equipped validation lane is available.
- Broad attention variants such as sparse attention, dynamic rank > 4, BFLOAT16, and advanced mask broadcasting are outside Phase 25 unless already covered by the narrow supported contract.

</deferred>

---

*Phase: 25-forward-sdpa-semantic-enablement*
*Context gathered: 2026-05-01*
