---
phase: 17
slug: normalization-reduction-and-loss-adjacent-lowering
status: planned
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-01
---

# Phase 17 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | JUnit Jupiter via Gradle |
| **Config file** | `build.gradle` |
| **Quick run command** | `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest` |
| **Full suite command** | `./gradlew classes && ./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests SourceTreeHygieneTest` |
| **Estimated runtime** | ~120 seconds focused; native Metal/CUDA execution remains capability-gated |

## Sampling Rate

- **After every task commit:** Run the focused test class for the touched area.
- **After every plan wave:** Run the quick run command.
- **Before `$gsd-verify-work`:** Run the full suite command and `git status --short`.
- **Max feedback latency:** 120 seconds for focused tests.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 17-01-01 | 01 | 1 | GPUNORM-01, GPUNORM-02 | T-17-01, T-17-02 | Shared matrix covers Phase 17 families and target evidence before backend changes. | unit/docs | `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest` | W0 | planned |
| 17-02-01 | 02 | 2 | GPUNORM-01, GPUNORM-02 | T-17-03, T-17-04 | Metal/CUDA legality rejections stay precise and do not bypass dtype/layout gates. | unit | `./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest` | W0 | planned |
| 17-03-01 | 03 | 3 | GPUNORM-02, GPUNORM-03 | T-17-05, T-17-06 | Softmax-ish support and loss-adjacent rejection remain trace-visible with CPU parity. | trace/report/parity | `./gradlew test --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest` | W0 | planned |
| 17-04-01 | 04 | 4 | GPUNORM-01, GPUNORM-02, GPUNORM-03 | T-17-07 | Docs, validation evidence, and artifact hygiene are verified before closure. | docs/test | `./gradlew classes && ./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests SourceTreeHygieneTest` | W0 | planned |

## Wave 0 Requirements

Existing infrastructure covers the phase starting point:

- `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java`
- `src/main/java/backend/accelerator/lowering/GpuLoweringUnsupportedReason.java`
- `src/main/java/backend/metal/lowering/MetalPartitionSupport.java`
- `src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java`
- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java`
- `src/test/java/backend/accelerator/lowering/GpuLoweringCoverageMatrixTest.java`
- `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java`
- `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java`
- `src/test/java/CompiledGraphTraceTest.java`
- `src/test/java/GpuCoverageSummaryTest.java`

## Manual-Only Verifications

Native CUDA and Metal execution remain optional and capability-gated outside portable tests. Run native Metal/CUDA test tasks where local shims and devices are available.

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all existing references.
- [x] No watch-mode flags.
- [x] Feedback latency target < 120s.
- [x] `nyquist_compliant: true` set in frontmatter.
