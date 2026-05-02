# Phase 31 Verification: BOOL-Producing Metal Compute

**Phase:** 31 BOOL-Producing Metal Compute
**Verified:** 2026-05-02

## Requirement Coverage

| Requirement | Status | Evidence |
|---|---|---|
| `METALBOOL-01` | Covered | `AcceleratorDagNodeType` and `AcceleratorSubgraphLowerer` lower compare/logical BOOL outputs; `MetalMpsCapabilities` and `MetalPartitionSupport` admit scoped dense BOOL operations; `MetalMpsFfmBridgeTest` verifies native BOOL compare/logical bytes. |
| `METALBOOL-02` | Covered | `AcceleratorDTypeResidencyPolicy` allows Metal internal `BOOL` values; `PreparedExecutionBuildTest.gpuMetalWhereCanKeepComparePredicateInsideGpuRegion` verifies `compare -> WHERE -> elementwise` stays inside one Metal lowered region with compute/internalValue BOOL residency evidence; `bool_compare_where_small` coverage gates reject missing BOOL evidence or hidden CPU/tensor-array exits. |
| `METALBOOL-03` | Covered | `REDUCE_ALL` and `REDUCE_ANY` lower through the shared BOOL DAG ABI and execute in the native Metal shim; native bridge tests verify exact BOOL reduction output bytes, while planner legality still rejects unsupported dtype/rank/layout variants visibly. |

## Verification Commands

```bash
./gradlew test --tests backend.accelerator.residency.AcceleratorDTypeResidencyPolicyTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests PreparedExecutionBuildTest --tests GpuCoverageSummaryTest --tests BenchmarkSuiteSessionTest
./gradlew metalTest
git diff --check
```

Phase 31-04 additionally ran:

```bash
./gradlew test --tests SourceTreeHygieneTest
./gradlew test --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.metal.lowering.MetalRegionLowererTest --tests GpuCoverageSummaryTest
./gradlew metalTest
git diff --check
git status --short profiles/platform
```

## Boundaries

- Phase 31 does not admit direct masked or causal SDPA; public BOOL mask semantics remain Phase 34 scope.
- CUDA BOOL-producing compute remains `UNSUPPORTED_DTYPE` until a CUDA-native implementation provides equivalent lowering, execution, parity, and coverage evidence.
- Local profile artifacts under `profiles/platform/...` remain uncommitted.
