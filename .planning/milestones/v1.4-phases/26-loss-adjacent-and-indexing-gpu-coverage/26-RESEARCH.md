# Research 26: Loss-Adjacent And Indexing GPU Coverage

**Phase:** 26 Loss-Adjacent And Indexing GPU Coverage
**Date:** 2026-05-02
**Status:** Complete

## Current State

Loss-adjacent and indexing operations are correctly CPU-owned today. The backend-neutral DAG ABI and native shims can execute matmul/linear, elementwise chains, layout/view-adjacent operations, softmax-like operations, reductions, normalization, and selected attention paths, but they do not expose native indexing or loss primitives.

Current blockers:

- No `AcceleratorDagNodeType` entries for `GATHER`, `TAKE_ALONG_AXIS`, `SCATTER_ADD`, or index-target loss.
- No Metal MPSGraph switch cases for index/loss primitives.
- No CUDA native kernels for gather/take/scatter/loss in the graph execution shim.
- `INT32` targets and indices can be represented in public tensors and CPU kernels, but are not accelerator compute/output dtypes.
- Duplicate-index accumulation makes scatter and backward index ops semantically sensitive.
- Ignore-index and reduction denominator behavior make index-target losses semantically sensitive.

## Key Findings

- `GpuTargetSemanticsContract` already describes loss and index families, but coverage rows do not enumerate all index ops.
- `GpuLoweringCoverageMatrix` currently includes `GATHER` and `SCATTER_ADD`, but `GATHER_GRAD`, `TAKE_ALONG_AXIS`, and `TAKE_ALONG_AXIS_GRAD` currently fall through to generic unsupported rows.
- Existing tests cover CPU behavior for gather, take-along-axis, scatter-add, index-target NLL, index-target cross-entropy, ignore-index, weighted loss, and reduction modes.
- Existing prepared-execution tests already assert visible `CROSS_ENTROPY_LOSS_INDICES` fallback for Metal/CUDA, but reason detail is too broad for Phase 26 success criteria.
- The safest immediate closure is honest coverage expansion and adjacency preservation. Native index/loss support should be added only after a narrow primitive contract is implemented.

## Recommended Implementation Shape

1. Add operation-specific stable reason vocabulary for loss/indexing fallback.
2. List every loss/index operation in the shared coverage matrix for Metal and CUDA.
3. Add legality-adapter detail so unsupported loss/indexing cases explain dtype, ignore-index, duplicate-index accumulation, or missing primitive support.
4. Add tests proving legal adjacent GPU regions are still selected before/around a CPU-owned loss/indexing boundary.
5. Keep `SUPPORTED` rows unchanged until native DAG and backend execution exist.
6. Document that INT32/BOOL/FLOAT residency evidence is not native index/loss compute.

## Verification Strategy

- Matrix and contract tests for all Phase 26 operation rows and reason codes.
- Metal/CUDA legality tests for `TAKE_ALONG_AXIS`, `GATHER_GRAD`, `TAKE_ALONG_AXIS_GRAD`, `SCATTER_ADD`, and index-target loss.
- CPU parity tests for duplicate-index scatter/take-backward and ignore-index loss remain the correctness oracle.
- Prepared execution / coverage tests for adjacent GPU region retention plus visible CPU loss/index fallback.
- `git diff --check` and focused Gradle tests.

## Research Complete

No external research is required for this phase. The relevant semantics and backend gaps are represented in local source, docs, and tests.
