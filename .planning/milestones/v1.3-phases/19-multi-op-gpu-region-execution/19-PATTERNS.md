# Phase 19: Multi-Op GPU Region Execution - Pattern Map

**Date:** 2026-05-01
**Status:** Complete

## Closest Existing Patterns

| Planned Area | Closest Existing Files | Pattern To Follow |
|--------------|------------------------|-------------------|
| Multi-op lowering contract | `AcceleratorSubgraphLowerer.java`, `GpuLoweredRegionManifest.java`, `GpuLoweredRegionManifestRenderer.java` | Backend-neutral records and renderer-backed trace evidence; avoid native ABI changes for debug metadata. |
| Metal/CUDA lowerer parity | `MetalRegionLowerer.java`, `CudaRegionLowerer.java`, `MetalRegionLowererTest.java`, `CudaRegionLowererTest.java` | Keep shared region shape and backend-specific package implementations symmetric. |
| Prepared executable residency | `PreparedMetalExecutable.java`, `PreparedCudaExecutable.java`, `PreparedAcceleratorExecutionSupport.java` | Native buffer binding first, explicit fallback paths, output bindings marked `DEVICE_OWNED`. |
| Prepared input materialization risk | `AcceleratorPreparedInputResolver.java`, `DeviceLayoutViewPropagationTest.java` | CPU-readable prepared inputs are visible materialization reasons; tests assert supported device paths do not trigger them. |
| Trace/report evidence | `PreparedExecution.java`, `GpuCoverageSummary.java`, `JsonBenchmarkReportRenderer.java`, `CompiledGraphTraceTest.java`, `GpuCoverageSummaryTest.java`, `BenchmarkSessionTest.java` | Add stable fields beside existing accelerator buffer/materialization/device handoff evidence. |
| Hot path targets | `GpuHotPathCoverageTargets.java`, `StandardWorkloads.java`, `BenchmarkSuiteSessionTest.java` | Use checked workload registry instead of ad hoc workload names. |
| Source hygiene | `SourceTreeHygieneTest.java` | Add grep-style constraints for package boundary and CPU fused isolation. |

## Files Likely Modified

- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java`
- `src/main/java/backend/accelerator/lowering/GpuLoweredRegionManifest.java`
- `src/main/java/backend/accelerator/exec/AcceleratorPreparedInputResolver.java`
- `src/main/java/backend/accelerator/exec/PreparedAcceleratorExecutionSupport.java`
- `src/main/java/backend/metal/exec/PreparedMetalExecutable.java`
- `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java`
- `src/main/java/graph/execution/PreparedExecution.java`
- `src/main/java/tuning/benchmark/report/GpuCoverageSummary.java`
- `src/main/java/tuning/benchmark/report/JsonBenchmarkReportRenderer.java`
- `docs/gpu-lowering-coverage.md`
- `docs/compute-flow.md`
- `docs/development.md`

## Test Files Likely Modified

- `src/test/java/backend/accelerator/lowering/AcceleratorSubgraphLowererTest.java`
- `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java`
- `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java`
- `src/test/java/PreparedExecutionBuildTest.java`
- `src/test/java/CompiledGraphTraceTest.java`
- `src/test/java/GpuCoverageSummaryTest.java`
- `src/test/java/BenchmarkSessionTest.java`
- `src/test/java/BenchmarkSuiteSessionTest.java`
- `src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java`
- `src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java`
- `src/test/java/graph/execution/DeviceLayoutViewPropagationTest.java`
- `src/test/java/SourceTreeHygieneTest.java`

## Guardrails

- Do not add public `Tensor` device APIs.
- Do not count tensor-array bridge execution as native buffer GPU coverage.
- Do not reuse CPU fused ASM or `Operation.OpType.FUSED` for GPU fusion.
- Keep Metal and CUDA aligned through shared accelerator contracts, not copy-pasted backend forks.
- Leave `profiles/platform/.../tuning/abc/*` unstaged.
