# Phase 26: Loss-Adjacent And Indexing GPU Coverage - Context

**Gathered:** 2026-05-02
**Status:** Ready for execution
**Mode:** inline GSD planning after user requested continuation

<domain>
## Phase Boundary

Phase 26 closes the loss-adjacent and indexing truth gap without pretending that index-target loss or scatter semantics are GPU-safe before native execution exists.

In scope:

- Expand source-of-truth coverage rows for `NLL_LOSS`, `CROSS_ENTROPY_LOSS`, `CROSS_ENTROPY_LOSS_INDICES`, `CROSS_ENTROPY_LOSS_INDICES_GRAD`, `GATHER`, `GATHER_GRAD`, `TAKE_ALONG_AXIS`, `TAKE_ALONG_AXIS_GRAD`, and `SCATTER_ADD`.
- Preserve explicit stable rejection for duplicate-index accumulation, ignore-index, bounds behavior, unsupported dtype roles, and missing native primitives.
- Keep adjacent legal GPU regions selected when an indexing/loss node must exit to CPU.
- Add coverage/report tests showing whether loss/indexing flows stayed on GPU or exited with visible reason codes.
- Document what is native GPU execution, what is resident-only evidence, and what remains CPU-owned.

Out of scope:

- Public device tensor API.
- CPU fused `Operation.OpType.FUSED` reuse for GPU fusion.
- Claiming native GPU support for index/loss operations without backend DAG ABI and native execution.
- CUDA native performance evidence on this local machine while `nvcc` is unavailable.

</domain>

<decisions>
## Implementation Decisions

- **D-01:** Do not mark a loss/indexing row `SUPPORTED` unless `AcceleratorDagNodeType`, Java lowering, backend legality, native Metal/CUDA execution, parity tests, and trace evidence all exist for that exact semantic case.
- **D-02:** Use stable reason codes to separate missing primitive support from risky semantics: index-target dtype, ignore-index, duplicate-index accumulation, bounds behavior, and capability gaps should not collapse into a generic fallback.
- **D-03:** Treat `INT32` targets and index tensors as residency evidence, not compute support, until an operation has a legal backend primitive that consumes them.
- **D-04:** `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD` are high-risk because duplicate indices require accumulation semantics. They must stay explicit fallback unless backend parity covers duplicate-index behavior.
- **D-05:** `CROSS_ENTROPY_LOSS_INDICES` has a real decomposition opportunity through `LOG_SOFTMAX` plus indexed selection, but Phase 26 should only admit it if ignore-index, reduction denominator, bounds, and target dtype semantics are proven.
- **D-06:** Unsupported loss/indexing nodes must not silently shorten the preceding supported GPU region. Coverage tests should assert the preceding `matmul/logSoftmax` or elementwise region still runs as an accelerator region and the CPU exit is visible.

</decisions>

<canonical_refs>
## Canonical References

- `.planning/ROADMAP.md` - Phase 26 goal and success criteria.
- `.planning/REQUIREMENTS.md` - GPULOSSIDX-01, GPULOSSIDX-02, GPULOSSIDX-03.
- `.planning/PROJECT.md` - public Tensor API and runtime residency boundaries.
- `.planning/phases/22-coverage-truth-and-semantics-lock/22-02-SUMMARY.md` - source-of-truth coverage contracts.
- `.planning/phases/23-forward-reductions-native-execution/23-03-SUMMARY.md` - pattern for native support claims.
- `.planning/phases/24-normalization-gpu-lowering/24-VERIFICATION.md` - parity and matrix admission pattern.
- `.planning/phases/25-forward-sdpa-semantic-enablement/25-VERIFICATION.md` - support-or-rejection closure pattern.
- `docs/gpu-lowering-coverage.md` - current matrix docs.
- `docs/compute-flow.md` - residency and materialization boundary reporting.
- `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java`
- `src/main/java/backend/accelerator/lowering/GpuTargetSemanticsContract.java`
- `src/main/java/backend/accelerator/residency/AcceleratorDTypeResidencyPolicy.java`
- `src/main/java/backend/metal/lowering/MetalPartitionSupport.java`
- `src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java`
- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java`
- `src/main/java/operations/index/*`
- `src/main/java/operations/loss/*`
- `src/test/java/GatherExecutionTest.java`
- `src/test/java/TakeAlongAxisExecutionTest.java`
- `src/test/java/ScatterAddExecutionTest.java`
- `src/test/java/IndexTargetNllLossExecutionTest.java`
- `src/test/java/IndexTargetCrossEntropyLossExecutionTest.java`
- `src/test/java/IgnoreIndexLossExecutionTest.java`
- `src/test/java/WeightedIndexLossExecutionTest.java`
- `src/test/java/IndexLossReductionExecutionTest.java`

</canonical_refs>

<code_context>
## Existing Code Insights

- Current coverage matrix lists `GATHER` and `SCATTER_ADD`, but not all index-family operations already represented in `GpuTargetSemanticsContract`.
- `TAKE_ALONG_AXIS`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD` exist in CPU kernels and operation descriptors and should have explicit GPU rows even if unsupported.
- `AcceleratorDagNodeType` has no index/loss primitive codes.
- Metal native MPSGraph switch and CUDA native shim have no gather/take/scatter or loss execution path.
- `CROSS_ENTROPY_LOSS_INDICES` already has rich CPU semantics: `INT32` target handling, ignore-index masking, class dimension, optional weights, and reduction policy.
- Existing Phase 17 tests already prove visible CPU fallback for `CROSS_ENTROPY_LOSS_INDICES`; Phase 26 should make that more precise and cover the full index/loss family.

</code_context>

---
*Phase: 26-loss-adjacent-and-indexing-gpu-coverage*
