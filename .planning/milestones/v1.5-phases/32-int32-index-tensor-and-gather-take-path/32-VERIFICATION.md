# Phase 32 Verification: INT32 Index Tensor And Gather Take Path

**Status:** Complete
**Verified:** 2026-05-02

## Requirement Mapping

| Requirement | Status | Evidence |
|---|---|---|
| `METALINTIDX-01` | Complete | `MetalMpsCapabilities` admits `INT32` only for index input roles; `MetalBufferAllocator`, `MetalMpsFfmBridge`, native dtype ABI v3, and `gather_take_small` coverage gates prove buffer/residency/report evidence without claiming generic INT32 compute/output. |
| `METALINTIDX-02` | Complete | `GATHER` and `TAKE_ALONG_AXIS` have shared DAG node types, axis metadata, Metal planner legality, native MPSGraph execution, rank/axis parity tests, and stable `UNSUPPORTED_DTYPE`, `UNSUPPORTED_LAYOUT`, and `UNSUPPORTED_BOUNDS_CHECK` rejections. |
| `METALINTIDX-03` | Complete | Prepared-execution tests prove `LOG_SOFTMAX -> TAKE_ALONG_AXIS` can stay in one Metal-owned region; coverage gates require no CPU materialization, no CPU fallback, and no tensor-array fallback for `gather_take_small`. |

## Verification Commands

```bash
./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests GpuCoverageSummaryTest --tests GpuTargetSemanticsContractTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest
./gradlew test --tests GatherExecutionTest --tests PreparedExecutionBuildTest --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.metal.lowering.MetalRegionLowererTest
./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests BenchmarkSuiteSessionTest --tests SourceTreeHygieneTest --tests StandardWorkloadsTest
./gradlew metalTest
git diff --check
git status --short profiles/platform
```

All Gradle commands passed. `git diff --check` passed. Local profile artifacts remain unstaged.

## Explicit Non-Goals

- No generic Metal `INT32` arithmetic or `INT32` output support.
- No CUDA gather/take implementation.
- No `SCATTER_ADD`, `GATHER_GRAD`, or `TAKE_ALONG_AXIS_GRAD` support.
- No index-target CE/NLL support.
- No broad non-dense/strided index compute support before the layout-router phase.
