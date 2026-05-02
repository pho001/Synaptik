# Phase 23 Context: Forward Reductions Native Execution

**Milestone:** v1.4 Native GPU Operation Coverage Closure
**Date:** 2026-05-01
**Status:** Ready for execution

## Goal

Add actual accelerator DAG and backend native execution support for forward reductions:

- `SUM`
- `MEAN`
- `REDUCE_MIN`
- `REDUCE_MAX`

## Phase 22 Handoff

Phase 22 established:

- `GpuTargetCoverageTruth` to prevent premature native-support claims.
- `GpuTargetSemanticsContract` for axis, keep-dims, dtype/rank/layout, and output shape semantics.
- `reduction_chain_small` as a deterministic representative workload.

## Current Gaps

- `AcceleratorDagNodeType` has no forward reduction primitive codes.
- `AcceleratorSubgraphLowerer` does not map `SUM`, `MEAN`, `REDUCE_MIN`, or `REDUCE_MAX`.
- `GpuLoweringCoverageMatrix` still marks forward reductions as fallback.
- Metal native MPSGraph already uses reduction helpers for gradient-adjacent ops, but no forward reduction DAG cases exist.
- CUDA native buffer execution currently supports a very narrow op set and needs reduction-aware shape/input validation and kernels.

## Requirements

- GPURED-01
- GPURED-02
- GPURED-03

## Constraints

- Keep public `Tensor` logical; no public GPU tensor API.
- Do not hide CPU fallback. Unsupported dtype/layout/rank cases must remain explicit.
- Keep rank 1-4 for this phase, matching current native ABI.
- Use FLOAT32 first. Non-floating and unsupported dtype cases remain explicit future work.
- Do not commit local benchmark/profile artifacts.
