# Phase 18 Pattern Map: Fused Elementwise And Epilogue Subregions

**Phase:** 18 - Fused Elementwise And Epilogue Subregions
**Date:** 2026-05-01
**Status:** Complete

## Pattern Mapping Complete

## Files To Modify Or Extend

| Planned file | Role | Closest existing analog | Notes |
|---|---|---|---|
| `src/main/java/backend/accelerator/lowering/GpuCompoundPatternType.java` | Stable fusion taxonomy | Existing Phase 12 enum | Add epilogue/subpattern names only if needed; do not replace existing names. |
| `src/main/java/backend/accelerator/lowering/GpuFusionSubpatternSummary.java` | New list-capable subpattern metadata | `GpuCompoundRegionSummary.java`, `GpuLoweredPrimitiveManifest.java` | Records type, original operation span, lowered primitive ids/count, support state, and rejection reason. |
| `src/main/java/backend/accelerator/lowering/GpuLoweredRegionManifest.java` | Manifest carrier for subpatterns | Existing `fusedSummary` field | Preserve existing `fusedSummary()` for compatibility and add `fusedSubpatterns()` as the richer contract. |
| `src/main/java/backend/accelerator/lowering/GpuCompoundPatternDetector.java` | Shared detector/classifier | Existing compound detector | Detect supported/rejected subpatterns from normal graph ops and `AcceleratorDagSpec`, never CPU fused internals. |
| `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java` | Manifest construction and primitive mapping | Existing manifest builder | Populate `GpuFusionSubpatternSummary` using original nodes and lowered primitive ids. |
| `src/main/java/graph/optimizer/region/GenericGpuRegionOptimizationPolicy.java` | GPU unit selection | Existing whole-partition fusion policy | Add maximal elementwise subchain and epilogue unit selection while preserving selected region boundaries. |
| `src/main/java/graph/optimizer/region/RegionOptimizationUnitSupport.java` | Unit helpers | Existing fused whole partition/subchain helpers | Add helpers for GPU elementwise subchains and matmul/linear epilogue spans. |
| `src/main/java/graph/optimizer/region/ExecutionUnitKind.java` | Unit taxonomy | Existing `FUSED_ELEMENTWISE` and `SPECIALIZED_PRIMITIVE` | Reuse existing kinds where possible; add a new kind only if tests need stable distinction. |
| `src/main/java/backend/metal/lowering/MetalRegionLowerer.java` | Metal lowered family/artifact | Existing compound artifact attachment | Preserve backend-neutral artifact and choose precise lowering family only for legal fused subpatterns. |
| `src/main/java/backend/cuda/lowering/CudaRegionLowerer.java` | CUDA lowered family/artifact | Metal lowerer | Mirror Metal metadata while keeping CUDA capability/layout gates conservative. |
| `src/main/java/backend/metal/exec/PreparedMetalExecutable.java` | Runtime behavior and metadata exposure | Existing compound summary and buffer decisions | Keep supported interiors device-owned; REQUIRED mode must fail instead of hidden CPU fallback. |
| `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java` | Runtime behavior and metadata exposure | Metal executable | Same contract, with CUDA native availability capability-gated. |
| `src/main/java/graph/execution/PreparedExecution.java` | Run trace attributes | Existing `gpuCompound*` attributes | Add `gpuFusedSubpatternCount`, span, primitive count, and rejection detail fields. |
| `src/main/java/tuning/benchmark/report/GpuCoverageSummary.java` | Coverage summary | Existing compound/manifest counters | Count fused subpatterns without hiding CPU exits. |
| `src/main/java/tuning/benchmark/report/JsonBenchmarkReportRenderer.java` | JSON/text report rendering | Existing `fusedSubpatternsJson` helper | Render subpattern count, type, primitive count, span, and rejection reason. |
| `src/test/java/backend/accelerator/lowering/GpuCompoundPatternDetectorTest.java` | Detector tests | Existing Phase 12 tests | Extend for supported subpatterns, rejection reasons, and CPU `FUSED` rejection. |
| `src/test/java/backend/accelerator/lowering/AcceleratorSubgraphLowererTest.java` | Manifest tests | Existing Phase 15 tests | Assert subpattern primitive ids/count and original op spans. |
| `src/test/java/graph/optimizer/region/DefaultRegionOptimizerTest.java` | Region unit tests | Existing GPU/CPU region tests | Assert fused subchains/epilogues do not shorten selected region. |
| `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java` | Metal lowerer tests | Existing compound tests | Add elementwise subchain and epilogue metadata assertions. |
| `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java` | CUDA lowerer tests | Metal lowerer tests | Mirror shared metadata and CUDA-specific rejection evidence. |
| `src/test/java/PreparedExecutionBuildTest.java` | Prepared execution tests | Existing accelerator tests | Assert single accelerator region, device-owned interiors, CPU parity, and required-mode behavior. |
| `src/test/java/CompiledGraphTraceTest.java` | Trace tests | Existing `gpuCompound*` tests | Assert fused subpattern trace fields. |
| `src/test/java/GpuCoverageSummaryTest.java` | Coverage summary tests | Existing report tests | Assert fused subpattern count and no hidden materialization. |
| `src/test/java/BenchmarkSessionTest.java` | Benchmark report tests | Existing JSON/text report tests | Assert Phase 18 fields are rendered. |
| `src/test/java/SourceTreeHygieneTest.java` | Architecture guard | Existing CPU-fused isolation checks | Add or keep checks preventing CPU fused imports in accelerator packages. |
| `docs/gpu-lowering-coverage.md` | Lowering docs | Existing Phase 17 coverage docs | Document `GPUFUSEX-01/02/03` and supported/rejected subpatterns. |
| `docs/compute-flow.md` | Runtime flow docs | Existing accelerator flow docs | Explain region-internal fusion and true materialization boundaries. |
| `.planning/phases/18-fused-elementwise-and-epilogue-subregions/18-*-SUMMARY.md` | Execution evidence | Phase 17 summaries | Record commands, requirement coverage, and profile artifact hygiene per wave. |

## Reusable Code Patterns

### Preserve Compatibility Records

Existing tests and renderers use `GpuCompoundRegionSummary`. Add richer subpattern metadata beside it and keep legacy accessors stable.

### Shared Contract, Backend-Owned Execution

The shared lowerer should identify semantic subpatterns and metadata. Metal/CUDA executables remain responsible for native bridge execution, buffer binding decisions, and visible fallback.

### Existing Unit Selection Style

`RegionOptimizationUnitSupport` already has `buildFusedSubchainUnit(...)` and `unitOutputsForChain(...)`. Reuse that style for GPU elementwise subchains and epilogue spans.

### Visible Fallback

Prepared execution traces already expose accelerator buffer decisions and compound metadata. Extend that channel instead of creating a second trace system.

## Landmines

- Do not modify public `tensor.Tensor` APIs.
- Do not create GPU public `FUSED` operations.
- Do not import `backend.cpu.fused` from accelerator, Metal, or CUDA production code.
- Do not treat AUTO-mode CPU fallback as proof of supported GPU fusion; REQUIRED-mode tests must catch hidden fallback.
- Do not stage local tuning profile artifacts.

