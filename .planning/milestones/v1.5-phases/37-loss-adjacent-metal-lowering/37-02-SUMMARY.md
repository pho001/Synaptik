# Plan 37-02 Summary: Dense And Index-Target Loss Admission

**Status:** Complete
**Completed:** 2026-05-02
**Requirement Coverage:** METALLOSS-01, METALLOSS-02 partial, METALLOSS-03 partial

## Completed

- Promoted scoped Metal dense `NLL_LOSS` and dense `CROSS_ENTROPY_LOSS` coverage rows to `SUPPORTED`.
- Added dense loss DAG lowering through existing Metal-supported primitives:
  - dense CE: `SOFTMAX -> LOG -> MUL(targets) -> SUM(all axes) -> MUL_SCALAR(-1/sampleCount)`;
  - dense NLL: `MUL(targets) -> SUM(all axes) -> MUL_SCALAR(-1/sampleCount)`.
- Kept support scoped to `FLOAT32`, dense zero-offset layouts, rank 1..4, matching dense target shape, valid class axis, and output shape `[1]`.
- Added prepared/native parity coverage for dense CE and NLL through the explicit Apple shim path.
- Left index-target CE/NLL and index-target gradients explicitly rejected with existing `UNSUPPORTED_INDEX_SEMANTICS`; no forward index-target admission was claimed because ignore-index, class weights, reduction denominators, and Phase 36 scatter/index-gradient blockers remain unresolved.
- Updated Metal backend docs and GPU lowering coverage docs.

## Verification

```bash
./gradlew test --tests PreparedExecutionBuildTest --tests backend.metal.lowering.MetalRegionLowererTest
./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuTargetSemanticsContractTest --tests backend.metal.lowering.MetalRegionLowererTest --tests PreparedExecutionBuildTest
./gradlew test --tests NllLossExecutionTest --tests CrossEntropyLossExecutionTest --tests IndexTargetNllLossExecutionTest --tests IndexTargetCrossEntropyLossExecutionTest --tests IgnoreIndexLossExecutionTest --tests WeightedIndexLossExecutionTest
./gradlew metalTest
git diff --check
```

All passed.

## Next

Phase 37-03 should focus on training/backward boundaries: dense loss forward can now stay Metal-owned, but dense loss backward and index-target gradient paths still need explicit trace gates so they do not masquerade as supported GPU training.
