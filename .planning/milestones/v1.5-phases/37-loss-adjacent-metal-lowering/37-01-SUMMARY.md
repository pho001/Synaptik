# Plan 37-01 Summary: Dense Loss Lowering Contract

**Status:** Complete
**Completed:** 2026-05-02
**Requirement Coverage:** METALLOSS-01, METALLOSS-03 partial

## Completed

- Added `MetalLossSemantics` as the Metal planner gate for dense `NLL_LOSS` and dense `CROSS_ENTROPY_LOSS`.
- Locked the dense Metal loss candidate scope to `FLOAT32`, dense zero-offset input/target layout, rank 1..4, valid class axis, same-shape dense targets, and public mean-reduced output shape `[1]`.
- Kept dense losses explicitly unsupported with `DAG_PRIMITIVE_UNSUPPORTED` until Phase 37-02 adds backend-owned lowering/execution.
- Preserved separation from index-target loss rows; `CROSS_ENTROPY_LOSS_INDICES` and `CROSS_ENTROPY_LOSS_INDICES_GRAD` remain `UNSUPPORTED_INDEX_SEMANTICS`.
- Updated GPU semantics contracts, coverage matrix notes, Metal docs, and lowering coverage docs.
- Added planner and contract tests for dense loss scope and index-target separation.

## Verification

```bash
./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuTargetSemanticsContractTest
./gradlew test --tests NllLossExecutionTest --tests CrossEntropyLossExecutionTest
git diff --check
```

All passed.

## Next

Phase 37-02 should implement or admit scoped dense loss lowering/execution against this contract, then decide index-target forward admission versus stable rejection without bypassing Phase 36 scatter/index-gradient blockers.
