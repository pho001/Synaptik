---
phase: 42
type: research
status: complete
requirements:
  - CUDANN-01
  - CUDANN-02
  - CUDANN-03
---

# Phase 42 Research: CUDA NN Operation Parity

## Phase Goal

Add native/lowered CUDA coverage or stable rejection for high-value NN forward families: masked/causal SDPA, conv/pool, and dense loss.

## Current Evidence

- Phase 40 established CUDA parity reporting and capability dimensions.
- Phase 41 added CUDA dtype role truth, layout materialization diagnostics, and forward index support-or-rejection validation.
- CUDA lowering currently supports many dense `FLOAT32` primitives but keeps `SCALED_DOT_PRODUCT_ATTENTION` as `CAPABILITY_MISSING`, conv/pool rows as `CAPABILITY_MISSING`, and dense `NLL_LOSS` / `CROSS_ENTROPY_LOSS` as `DAG_PRIMITIVE_UNSUPPORTED`.
- Metal already has scoped contracts for all three families:
  - SDPA: dense `FLOAT32` rank-3/4 with unmasked, dense BOOL external mask, causal, and external+causal effective mask modes.
  - Conv/pool: dense `FLOAT32` NCHW/OIHW forward conv and rank-4 pooling, with explicit grouped/dilated/countIncludePad blockers.
  - Dense loss: dense `FLOAT32` rank 1..4 mean-reduced scalar NLL/CE over dense targets.

## Planning Direction

Phase 42 should not promote CUDA support just because the shared DAG has a node type. For each family:

1. Validate semantic contract first.
2. If native CUDA bridge support exists or is feasible in-scope, implement and prove parity.
3. Otherwise retain fallback/unsupported status but replace generic matrix rejection with stable, operation-specific blockers.
4. Coverage gates must prove support rows use native buffer execution and unsupported rows remain visible.

## Likely Implementation Shape

- `CudaNnSemantics` or focused helpers under `backend/cuda/lowering/` should classify CUDA SDPA, conv/pool, and dense loss before generic matrix rejection.
- CUDA capability reports and coverage reports should distinguish `DAG_PRIMITIVE_UNSUPPORTED`, `VENDOR_LIBRARY_ROUTE_NOT_INTEGRATED`, `UNSUPPORTED_MASK_SEMANTICS`, `UNSUPPORTED_RANK_OR_SHAPE`, `UNSUPPORTED_LAYOUT`, and `CAPABILITY_MISSING`.
- cuBLAS/cuDNN integration should remain `NOT_INTEGRATED` unless a real routed execution path is added.

## Verification Targets

- CUDA SDPA tests for unmasked, masked, causal, dtype, layout, rank, and broadcast shape reasons.
- CUDA conv/pool tests for dtype, layout, rank, groups, dilation, stride/pad metadata, and average-pool divisor blockers.
- CUDA dense loss tests for dense target contract, dtype/layout/rank/class-axis/output shape, and final primitive blocker.
- Coverage/docs tests proving reports expose visible CUDA blockers and do not count capability skip as support.
