---
phase: 18
slug: fused-elementwise-and-epilogue-subregions
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-01
---

# Phase 18 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | JUnit Jupiter via Gradle |
| **Config file** | `build.gradle` |
| **Quick run command** | `./gradlew test --tests backend.accelerator.lowering.GpuCompoundPatternDetectorTest --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests graph.optimizer.region.DefaultRegionOptimizerTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest` |
| **Full suite command** | `./gradlew classes && ./gradlew test --tests backend.accelerator.lowering.GpuCompoundPatternDetectorTest --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.accelerator.lowering.GpuLoweredRegionManifestTest --tests graph.optimizer.region.DefaultRegionOptimizerTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests OptimizerFuseTest --tests backend.cpu.fused.plan.LoweredFusedOperationBuilderTest --tests SourceTreeHygieneTest` |
| **Estimated runtime** | ~150 seconds focused; native Metal/CUDA execution remains capability-gated |

## Sampling Rate

- **After every task commit:** Run the focused test class for the touched area.
- **After every plan wave:** Run the quick run command.
- **Before `$gsd-verify-work`:** Run the full suite command and `git status --short`.
- **Max feedback latency:** 150 seconds for focused portable gates.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 18-01-01 | 01 | 1 | GPUFUSEX-01, GPUFUSEX-02, GPUFUSEX-03 | T-18-01, T-18-02 | Shared subpattern metadata records spans/counts/rejections without CPU fused internals. | unit/model/docs | `./gradlew test --tests backend.accelerator.lowering.GpuCompoundPatternDetectorTest --tests backend.accelerator.lowering.GpuLoweredRegionManifestTest` | W0 | pending |
| 18-02-01 | 02 | 2 | GPUFUSEX-01, GPUFUSEX-03 | T-18-03, T-18-04 | Elementwise chain interiors stay GPU-owned and do not use Java array round trips. | optimizer/lowering/runtime | `./gradlew test --tests graph.optimizer.region.DefaultRegionOptimizerTest --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest` | W0 | pending |
| 18-03-01 | 03 | 3 | GPUFUSEX-02, GPUFUSEX-03 | T-18-05, T-18-06 | Matmul/linear epilogues are gated by dtype/layout/backend legality and visible fallback. | lowering/runtime/parity | `./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest` | W0 | pending |
| 18-04-01 | 04 | 4 | GPUFUSEX-01, GPUFUSEX-02, GPUFUSEX-03 | T-18-07 | Trace/report evidence exposes subpatterns and CPU fused execution remains isolated. | trace/report/docs/hygiene | `./gradlew classes && ./gradlew test --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests OptimizerFuseTest --tests backend.cpu.fused.plan.LoweredFusedOperationBuilderTest --tests SourceTreeHygieneTest` | W0 | pending |

## Wave 0 Requirements

Existing infrastructure covers the phase starting point:

- `src/main/java/backend/accelerator/lowering/GpuCompoundRegionSummary.java`
- `src/main/java/backend/accelerator/lowering/GpuCompoundPatternDetector.java`
- `src/main/java/backend/accelerator/lowering/GpuLoweredRegionManifest.java`
- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java`
- `src/main/java/graph/optimizer/region/GenericGpuRegionOptimizationPolicy.java`
- `src/main/java/backend/metal/lowering/MetalRegionLowerer.java`
- `src/main/java/backend/cuda/lowering/CudaRegionLowerer.java`
- `src/main/java/graph/execution/PreparedExecution.java`
- `src/test/java/backend/accelerator/lowering/GpuCompoundPatternDetectorTest.java`
- `src/test/java/backend/accelerator/lowering/AcceleratorSubgraphLowererTest.java`
- `src/test/java/PreparedExecutionBuildTest.java`

## Manual-Only Verifications

Native Metal and CUDA execution remain optional and capability-gated outside portable tests. Run `./gradlew metalTest` and CUDA native tasks where local shims/devices are available before claiming native execution evidence.

## Validation Sign-Off

- [x] All tasks have automated verify commands or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all existing references.
- [x] No watch-mode flags.
- [x] Feedback latency target < 150s.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** draft

