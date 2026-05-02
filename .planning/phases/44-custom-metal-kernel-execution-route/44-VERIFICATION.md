---
phase: 44-custom-metal-kernel-execution-route
status: passed
verified: 2026-05-02
requirements:
  - METALKERNEL-01
  - METALKERNEL-02
  - METALKERNEL-03
gaps: 0
---

# Phase 44 Verification: Custom Metal Kernel Execution Route

## Verdict

Passed. Phase 44 turns the custom Metal kernel route from an unavailable seam into a real scoped native route without overclaiming broad custom-kernel support.

## Requirement Evidence

| Requirement | Status | Evidence |
|---|---|---|
| `METALKERNEL-01` | Passed | `MetalCustomKernelCandidate` admits the single-node dense `FLOAT32` `RELU` candidate; `MetalMpsFfmCustomKernelBridge` compiles it; native `synaptik_apple_mps_custom_relu_f32_buffer` executes it through buffer handles. |
| `METALKERNEL-02` | Passed | `MetalExecutionRouterTest`, `PreparedMetalExecutableBufferBindingTest`, `MetalMpsFfmBridgeTest`, and `MetalBufferTraceSmokeTest` cover CPU parity, dtype/candidate legality, dense runtime binding checks, buffer binding execution, fallback to MPSGraph when custom is ineligible, and native Metal execution. |
| `METALKERNEL-03` | Passed | `PreparedExecution` trace metadata exposes route fields; `GpuCoverageSummary` aggregates `executionRouteCounts`, `rejectedRouteReasonCounts`, and `nativeCopyStrategyCounts`; text/JSON renderers include these fields. |

## Scope Truth

Supported custom kernel route:

- `metalExecutionRoute=CUSTOM_KERNEL`
- `metalExecutionPath=CUSTOM_KERNEL`
- kernel id `relu_f32`
- single lowered DAG node: `RELU`
- one external input, one output
- dense contiguous `FLOAT32` runtime bindings
- buffer transport only
- native copy strategy: `TRUE_OUTPUT_BUFFER_WRITE`

Explicitly not claimed:

- no universal custom-kernel replacement for MPSGraph
- no CPU `Operation.OpType.FUSED` reuse
- no BF16/BOOL/INT32/FLOAT64 custom kernel support
- no custom execution for multi-node regions yet
- no MPSGraph true output-buffer write proof

## Automated Checks

- `./gradlew test --tests backend.metal.exec.MetalExecutionRouterTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests SourceTreeHygieneTest` - passed
- `./gradlew test --tests backend.metal.exec.MetalExecutionRouterTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.MetalBufferTraceSmokeTest` - passed during 44-02
- `./gradlew test --tests BenchmarkSessionTest --tests GpuCoverageRegressionGateTest --tests GpuCoverageGapTriageTest` - passed during 44-03
- Runtime route metadata regression after review: `PreparedMetalExecutableBufferBindingTest.customKernelRouteReportsMpsGraphWhenRuntimeBindingsAreNotDense` - passed
- `./gradlew classes` - passed
- `./gradlew metalTest` - passed
- `git diff --check` - passed

## Review Closure

Advisory code review is recorded in `44-REVIEW.md` with `findings_open: 0`. The only issue found during review was fixed in `a4fc53b`: runtime view/non-dense bindings now report `MPS_GRAPH` rather than leaving stale prepare-time `CUSTOM_KERNEL` route metadata after execution falls back to the MPSGraph buffer route.

## Source Hygiene

Local benchmark/profile artifacts under `profiles/platform/mac_os_x-aarch64-oracle_corporation-16c/tuning/` remain intentionally uncommitted.

## Residual Risk

The custom-kernel route is intentionally narrow. Phase 45 still owns MPSGraph output-buffer write/copy closure, and Phase 46 still owns broader router calibration across MPSGraph, custom Metal kernels, CUDA, tensor-array fallback, and CPU fallback.
