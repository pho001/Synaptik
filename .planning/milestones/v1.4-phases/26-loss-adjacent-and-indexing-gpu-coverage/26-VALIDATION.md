---
phase: 26
slug: loss-adjacent-and-indexing-gpu-coverage
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-02
---

# Phase 26 - Validation Strategy

## Requirement Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| GPULOSSIDX-01 | COVERED | Loss-adjacent and index ops have explicit Metal/CUDA rows with stable rejection reasons tied to missing primitives, index semantics, or duplicate-index accumulation. |
| GPULOSSIDX-02 | COVERED | Boundary tests prove legal GPU producers/consumers remain selected around CPU-owned loss/index boundaries. |
| GPULOSSIDX-03 | COVERED | CPU parity tests cover gather, take-along-axis, scatter-add, duplicate-index gradient behavior, ignore-index loss, weighted loss, and reduction modes. |

## Execution Evidence

| Command | Result |
|---------|--------|
| `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuTargetSemanticsContractTest` | Passed |
| `./gradlew test --tests PreparedExecutionBuildTest.phaseTwentySixUnsupportedTakeBoundaryKeepsPrecedingMetalLogSoftmaxRegionSelected --tests backend.metal.lowering.MetalRegionLowererTest.phaseTwentySixIndexFamilyUsesStableCoverageReasons --tests backend.cuda.lowering.CudaRegionLowererTest.phaseTwentySixIndexFamilyUsesStableCoverageReasons` | Passed |
| `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuTargetSemanticsContractTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests GatherExecutionTest --tests TakeAlongAxisExecutionTest --tests ScatterAddExecutionTest --tests IndexTargetNllLossExecutionTest --tests IndexTargetCrossEntropyLossExecutionTest --tests IgnoreIndexLossExecutionTest --tests WeightedIndexLossExecutionTest --tests IndexLossReductionExecutionTest` | Passed |
| `./gradlew metalTest` | Passed |
| `git diff --check` | Passed |
| `./gradlew cudaTest` | Not run locally; `nvcc` unavailable |

## Validation Audit 2026-05-02

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

**Approval:** verified 2026-05-02
