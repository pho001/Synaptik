# Phase 31 Research: BOOL-Producing Metal Compute

## Summary

Metal BOOL support is already partially present as storage and predicate input support: `WHERE` can consume an external BOOL predicate buffer. The missing piece is native BOOL-producing compute/output. Phase 31 should add this in a narrow, evidence-driven order: compare outputs first, logical BOOL chains second, `WHERE` mask-chain residency third, and BOOL reductions either supported or explicitly rejected.

## Existing Architecture

- DType ABI v3 can describe `BOOL` roles, but `MetalMpsCapabilities.computeDecision(BOOL)` and `outputDecision(BOOL)` reject today.
- `MetalBufferAllocator.createPredicateInputBinding(...)` already handles BOOL predicate inputs.
- Runtime materialization can read/write dtype-matched buffers, but output validation and native executable descriptors must be widened for BOOL outputs.
- Shared coverage rows already name compare/logical/reduction BOOL operations, currently as `UNSUPPORTED_DTYPE`.

## Implementation Risks

1. **MPSGraph API surface:** MPSGraph comparison and logical operations may differ by SDK/runtime. Native support must be capability-gated or tests must skip cleanly when unavailable.
2. **BOOL storage representation:** Synaptik BOOL storage is one byte per element. Native MPSGraph BOOL tensor data must map to the same byte convention before output readback is claimed.
3. **Mask-chain hidden CPU exits:** A compare may execute natively but still materialize before `WHERE` unless planner/lowering/runtime keeps the BOOL output device-owned.
4. **Reduction identity semantics:** `REDUCE_ALL` and `REDUCE_ANY` need exact CPU parity for identity, axis, keepDims, rank, and empty-dimension behavior. If unproven, they should remain stable rejection rows.
5. **CUDA isolation:** Shared matrix changes must not mark CUDA native support unless CUDA execution and tests exist.

## Recommended Slice

1. Add BOOL output role capability and native DAG node types for compare/logical operations behind Metal-only legality.
2. Implement native MPSGraph compare/logical branches and Java FFM descriptor handling for BOOL outputs.
3. Prove compare/logical parity with `metalTest`.
4. Make `compare -> WHERE` a selected Metal region with no CPU materialization between the BOOL producer and consumer.
5. Decide `REDUCE_ALL/ANY` based on native feasibility:
   - if feasible, implement scoped rank/axis/keepDims support and parity tests;
   - otherwise, keep stable explicit `CAPABILITY_MISSING` or `UNSUPPORTED_RANK_OR_SHAPE` rejection.
6. Update coverage reports and gates so `bool_compare_where_small` requires native BOOL evidence when supported.

## Verification Targets

```bash
./gradlew classes
./gradlew test --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.lowering.MetalRegionLowererTest
./gradlew test --tests PreparedExecutionBuildTest --tests GpuCoverageSummaryTest --tests GpuHotPathCoverageTargetsTest --tests BenchmarkSuiteSessionTest
./gradlew metalTest
git diff --check
```

## Planning Implication

Do not treat all BOOL rows as supported in one step. Compare/logical output support and `WHERE` mask-chain residency are the value path. BOOL reductions are acceptable as supported or stable rejected as long as `METALBOOL-03` is explicitly satisfied.
