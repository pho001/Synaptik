# Phase 39 Verification: Metal Backend Router And Zero-Copy Closure

**Status:** Verified
**Verified:** 2026-05-02 15:03 CEST

## Requirement Mapping

| Requirement | Status | Evidence |
|---|---|---|
| `METALROUTER-01` | Complete | `MetalExecutionRouter`, `MetalRouteDecision`, route trace attrs, custom-kernel SPI, and `CUSTOM_KERNEL_UNAVAILABLE` rejected-route evidence. |
| `METALROUTER-02` | Complete | `MetalNativeCopyStrategy` classifies current output behavior as `MPSGRAPH_RESULT_COPY`; `TRUE_OUTPUT_BUFFER_WRITE` remains reserved for future proof. |
| `METALROUTER-03` | Complete | Coverage/report summaries expose route counts, rejected route reasons, native copy strategies, tensor-array/CPU fallback counts, selected region length, and native-copy strategy gate checks. |

## Code Evidence

- Router and route metadata:
  - `src/main/java/backend/metal/exec/MetalExecutionRouter.java`
  - `src/main/java/backend/metal/exec/MetalRouteDecision.java`
  - `src/main/java/backend/metal/exec/PreparedMetalExecutable.java`
- Custom-kernel seam:
  - `src/main/java/backend/metal/kernel/MetalCustomKernelBridge.java`
  - `src/main/java/backend/metal/exec/MetalCustomKernelRouteAdapter.java`
- Native copy classification:
  - `src/main/java/backend/metal/bridge/MetalNativeCopyStrategy.java`
  - `src/main/java/backend/metal/bridge/MetalMpsBridgeExecutionStats.java`
  - `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java`
- Reports and gates:
  - `src/main/java/tuning/benchmark/report/AcceleratorTraceSummary.java`
  - `src/main/java/tuning/benchmark/report/GpuCoverageSummary.java`
  - `src/main/java/tuning/benchmark/report/GpuCoverageRegressionGate.java`
  - `src/main/java/tuning/benchmark/report/GpuCoverageGatePolicy.java`

## Verification Commands

Passed:

```bash
./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest
./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest
./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.bridge.MetalMpsBridgeExecutionStatsTest
./gradlew test --tests BenchmarkSessionTest --tests GpuCoverageSummaryTest
./gradlew test --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest
./gradlew test --tests BenchmarkSuiteSessionTest --tests SourceTreeHygieneTest
./gradlew classes
./gradlew metalTest
git diff --check
```

## Residual Scope

- Custom Metal kernels are not implemented yet; the seam is intentionally unavailable and report-visible.
- Output-buffer direct write is not claimed; current native behavior remains `MPSGRAPH_RESULT_COPY`.
- Local profile/tuning artifacts remain unstaged and uncommitted.
