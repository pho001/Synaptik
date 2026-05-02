# Research 25: Forward SDPA Semantic Enablement

**Phase:** 25 Forward SDPA Semantic Enablement
**Date:** 2026-05-01
**Status:** Complete

## Current State

Forward `SCALED_DOT_PRODUCT_ATTENTION` exists as a public tensor operation and CPU kernel. The optimizer can lower attention-shaped expressions into the SDPA operation, and the accelerator lowerer can encode a direct `SDPA` DAG node with scale bits. At the start of Phase 25, planner admission was intentionally blocked:

- Metal direct unmasked SDPA rejects with `CAPABILITY_MISSING: direct forward SDPA disabled until native MPSGraph scale contract matches CPU semantics`.
- Metal direct masked SDPA rejects with `UNSUPPORTED_MASK_SEMANTICS: direct masked SDPA disabled until bool-mask semantics are verified against MPSGraph floating masks`.
- CUDA coverage keeps forward SDPA fallback/unsupported rather than supported.
- `GpuTargetSemanticsContract` marked forward SDPA planner admission blocked until scale, mask, rank, and backward-interaction semantics were verified.

## Key Findings

- Public SDPA computes `scores = query @ key^T * scale`, applies optional BOOL/causal mask, runs softmax over the key axis, then multiplies by value.
- `AttentionOptions` resolves default scale as `1 / sqrt(headDim)` unless an explicit positive scale is provided.
- Public masks are `BOOL`; this does not automatically match MPSGraph native mask operand semantics.
- `AcceleratorDagNodeType.SDPA` already exists and `AcceleratorSubgraphLowerer` can encode direct SDPA and an unmasked decomposed attention pattern into an `SDPA` DAG node.
- The lowerer refuses masked decomposed attention because current verified Metal mask semantics expect a floating mask tensor while Synaptik masks are BOOL.
- Metal native Objective-C switch already has a MPSGraph SDPA path, but Java planner blocks it.
- `transformer_block_hot_path` is already a GPUSDPA hot-path target.

## Recommended Implementation Shape

1. Lock executable SDPA semantics in tests first.
2. Enable only the narrow legal case that passes parity: FLOAT32 query/key/value, rank 3 or 4, unmasked first, dense or already legal device layout, and explicit stable shape contract.
3. Keep masked SDPA rejected unless a correct BOOL-to-backend-mask mapping is proven.
4. For Metal, prefer native MPSGraph SDPA only if the scale contract matches CPU parity. If not, keep direct native SDPA rejected and preserve decomposed GPU primitive support where possible.
5. For CUDA, either implement a real direct/lowered SDPA path or keep a stable capability rejection; do not mirror Metal support without native evidence.
6. Update matrix/truth/report gates only after execution evidence exists.

## Primary Risks

- Native MPSGraph SDPA may apply `scale` differently than Synaptik CPU semantics.
- BOOL mask semantics may not match native floating mask semantics.
- Direct forward SDPA admission may break backward/training prepared-execution safety.
- Coverage rows can become false positives if updated before native execution and parity evidence.
- CUDA can appear supported via Java lowering while native CUDA execution remains unavailable on this local machine.

## Verification Strategy

- CPU parity fixtures for explicit scale, default scale, rank 3, rank 4, causal/external mask, and invalid shape/dtype cases.
- Metal legality and native tests for direct unmasked admission or stable rejection.
- CUDA legality tests for supported native/lowered path or stable capability rejection.
- Prepared execution tests for inference and training/backward safety.
- Trace/report/benchmark tests for `transformer_block_hot_path` attention evidence.
- `./gradlew metalTest` when touching native Metal SDPA.
- `./gradlew cudaTest` only in a CUDA lane with `nvcc`.

## Research Complete

No external research is required for planning; the necessary constraints are already represented in local source, docs, tests, and Phase 22-24 artifacts.
