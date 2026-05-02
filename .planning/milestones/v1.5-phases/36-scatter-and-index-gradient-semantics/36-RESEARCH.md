# Phase 36 Research: Scatter And Index Gradient Semantics

**Status:** Complete
**Date:** 2026-05-02

## Existing Behavior

- Public `scatterAdd(base, indices, src, dimension)` returns a base-shaped tensor and adds each `src` value into the indexed destination along `dimension`.
- `gather` backward creates a zero tensor shaped like the input and uses scatter-add style accumulation into selected positions.
- `takeAlongAxis` backward scatters upstream gradients back into the input shape while preserving non-axis dimensions.
- CPU accumulation order is deterministic for one Java execution: logical source/index order drives additive updates into the destination.

## Current GPU State

- Metal forward `GATHER` and `TAKE_ALONG_AXIS` are supported for dense `FLOAT32` values and static in-bounds `INT32` indices.
- `GATHER_GRAD`, `TAKE_ALONG_AXIS_GRAD`, and `SCATTER_ADD` remain unsupported in `GpuLoweringCoverageMatrix` with `UNSUPPORTED_DUPLICATE_INDEX`.
- Metal planner tests already assert that `SCATTER_ADD` rejects with the stable duplicate-index reason.

## Implementation Considerations

1. Duplicate indices are the core semantic risk. Native support is only valid if repeated writes accumulate rather than last-writer-wins.
2. Floating accumulation order must be documented. For `FLOAT32`, exact bitwise equality is not guaranteed if a parallel kernel changes addition order; tests should use an explicit tolerance if the native path is parallel.
3. Bounds behavior must match CPU. Phase 32 forward gather rejects unproven/out-of-range static indices before MPSGraph execution; Phase 36 should reuse the same static-bounds discipline for supported Metal index-gradient paths.
4. DType scope should start narrow: dense `FLOAT32` values/output with dense static `INT32` indices. BF16, BOOL, INT32 output, CUDA, and non-dense index-gradient compute remain later work unless independently proven.
5. Device residency evidence matters as much as numeric parity. A supported path must show `BUFFER_BINDING`, no CPU fallback, no tensor-array replay, and no CPU materialization between adjacent Metal producers/consumers.

## Recommended Plan

- Wave 36-01 should lock the semantics and matrix/diagnostic contract before native claims.
- Wave 36-02 should make an explicit implementation decision per operation:
  - implement a narrow Metal native/lowered path only where duplicate accumulation parity can be proven,
  - otherwise keep a stable rejection that reports the exact blocker.
- Wave 36-03 should add gradient and duplicate-index parity fixtures, including `gather` backward and `takeAlongAxis` backward.
- Wave 36-04 should add coverage targets/reports that distinguish forward index support from index-gradient support.

## Residual Risks

- A vendor scatter primitive may not provide CPU-compatible duplicate accumulation semantics.
- A parallel custom kernel may be faster but produce non-deterministic addition order unless its contract is explicitly tolerance-based.
- Admitting `SCATTER_ADD` without proving bounds and duplicate behavior would silently corrupt training gradients.
