---
phase: 18-fused-elementwise-and-epilogue-subregions
plan: "04"
status: complete
requirements-completed: [GPUFUSEX-01, GPUFUSEX-02, GPUFUSEX-03]
completed: 2026-05-01
---

# Phase 18 Plan 04: Trace Report And CPU Fusion Isolation Closure

## Trace report and CPU fusion isolation closure

Phase 18 now exposes fused GPU subpattern evidence in run traces and benchmark coverage reports without replacing the existing accelerator buffer, fallback, CPU materialization, tensor-array fallback, or device handoff fields.

`GPU fusion is region-internal lowering/fusion, not CPU fused ASM reuse`. The stable evidence fields are `gpuFusedSubpatternCount`, `gpuFusedSubpatternTypes`, `gpuFusedSubpatternOriginalNodeIds`, `gpuFusedSubpatternLoweredPrimitiveCount`, and `gpuFusedSubpatternReasons`.

CPU fused execution remains covered by CPU-focused tests, and source hygiene checks reject imports from CPU fused internals in accelerator, Metal, and CUDA production packages.

## Requirement Notes

| Requirement | Evidence |
|-------------|----------|
| GPUFUSEX-01 | Elementwise subpattern trace/report fields expose original operation spans, lowered primitive counts, and CPU materialization evidence. |
| GPUFUSEX-02 | Epilogue subpattern evidence flows through the same manifest-backed trace/report fields and keeps rejection detail visible. |
| GPUFUSEX-03 | CPU fused execution still uses `Operation.OpType.FUSED` only through CPU fused planning paths, while GPU fusion source hygiene rejects CPU fused internals. |

Native Metal/CUDA execution remains capability-gated.

## Verification

| Command | Result |
|---------|--------|
| `./gradlew classes` | Passed |
| `./gradlew test --tests backend.accelerator.lowering.GpuCompoundPatternDetectorTest --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.accelerator.lowering.GpuLoweredRegionManifestTest --tests graph.optimizer.region.DefaultRegionOptimizerTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests OptimizerFuseTest --tests backend.cpu.fused.plan.LoweredFusedOperationBuilderTest --tests SourceTreeHygieneTest` | Passed |

## Hygiene

`profiles/platform/.../tuning/abc/* remained unstaged`
