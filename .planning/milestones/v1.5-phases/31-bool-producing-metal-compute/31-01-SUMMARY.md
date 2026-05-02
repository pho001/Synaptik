# 31-01 Summary: BOOL Output ABI And Compare Primitive Contract

## Completed

- Added accelerator DAG node types and ABI codes for `GT`, `GE`, `LT`, `LE`, `EQ`, and `NE`.
- Extended accelerator subgraph lowering so compare ops can lower to BOOL-output DAG nodes with dtype metadata.
- Updated Metal dtype capability truth for scoped BOOL compare outputs while keeping logical BOOL ops and BOOL reductions operation-rejected.
- Added Metal compare input-role legality for FLOAT32/BFLOAT16 data inputs only.
- Added planner-side compare contract checks for BOOL output, two inputs, supported numeric input dtype, and dense layout.
- Kept Metal compare rows as `FALLBACK/CAPABILITY_MISSING` until native Phase 31-02 execution evidence lands.
- Kept CUDA compare/logical/reduction BOOL rows at `UNSUPPORTED_DTYPE`.

## Verification

- `./gradlew classes`
- `./gradlew test --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.lowering.MetalRegionLowererTest`
- `git diff --check`

## Outcome

`METALBOOL-01` and `METALBOOL-02` now have the Java-side compare ABI and legality contract needed for native Metal BOOL output work. The planner still rejects compare execution with `CAPABILITY_MISSING`, so no runtime path claims native BOOL-producing Metal compute before Phase 31-02.
