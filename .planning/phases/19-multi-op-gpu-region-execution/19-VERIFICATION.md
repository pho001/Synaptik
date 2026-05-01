---
phase: 19-multi-op-gpu-region-execution
status: verified
verified_at: 2026-05-01
requirements: [GPUMULTI-01, GPUMULTI-02, GPUMULTI-03]
---

# Phase 19 Verification

## Verdict

Phase 19 is verified. All five execution plans are complete, all Phase 19 requirements are marked complete, and focused
verification passed for lowering, backend legality, prepared execution, trace/report coverage, buffer-binding handoff,
device-layout propagation, documentation, and source hygiene.

## Requirement Evidence

| Requirement | Verdict | Evidence |
|---|---|---|
| `GPUMULTI-01` | Pass | `AcceleratorSubgraphLowerer`, Metal, and CUDA tests cover multi-op lowered regions with layout/view, elementwise, matmul/linear, and log-softmax-ish primitive sequences. Coverage docs now state that a selected GPU partition can execute as one backend-owned lowered region only after backend gates accept it. |
| `GPUMULTI-02` | Pass | Prepared Metal/CUDA executable tests and `DeviceLayoutViewPropagationTest` verify supported native-buffer paths keep interior values device-owned while graph output, CPU consumer, and gradient publication remain real CPU materialization boundaries. |
| `GPUMULTI-03` | Pass | Shared accelerator lowering/manifest contracts remain backend-neutral while Metal/CUDA choose backend-specific prepared executables. `SourceTreeHygieneTest` rejects CPU fused internals and public `Tensor` device-residency API drift. |

## Verification Commands

| Command | Result |
|---|---|
| `./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest` | Passed in Wave 1 |
| `./gradlew test --tests PreparedExecutionBuildTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests graph.execution.DeviceLayoutViewPropagationTest` | Passed in Wave 2 |
| `./gradlew test --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest` | Passed in Wave 3 |
| `./gradlew test --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests GpuHotPathCoverageTargetsTest` | Passed in Wave 4 |
| `./gradlew test --tests SourceTreeHygieneTest` | Passed in Wave 5 |
| `rg -n "Phase 19 multi-op GPU region execution\|selected GPU partition can execute as one backend-owned lowered region\|ExecutionState and device buffer bindings carry supported internal values\|tensor-array bridge execution is not native buffer GPU coverage\|GPU fusion remains region-internal lowering/fusion, not CPU fused ASM reuse\|vendor library routing is deferred to GPULIB-\\*\|profiles/platform/.../tuning/abc/\\* remained unstaged" docs/gpu-lowering-coverage.md docs/compute-flow.md docs/development.md` | Passed in Wave 5 |
| `./gradlew classes` | Passed final verification |
| `./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests graph.execution.DeviceLayoutViewPropagationTest --tests SourceTreeHygieneTest` | Passed final verification |

## Review Notes

Inline review found no additional Phase 19 blockers. Remaining unsupported normalization, reduction, conv, and
loss-adjacent GPU blockers are intentionally documented as visible fallback/rejection outcomes, not hidden support.
`tensor-array bridge execution is not native buffer GPU coverage`, and `GPU fusion remains region-internal lowering/fusion, not CPU fused ASM reuse`.

## Hygiene

Local profile tuning artifacts under `profiles/platform/.../tuning/abc/*` remained unstaged and are not Phase 19
closure evidence.
