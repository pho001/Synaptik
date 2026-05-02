---
phase: 24
slug: normalization-gpu-lowering
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-01
---

# Phase 24 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | JUnit Jupiter via Gradle |
| **Config file** | `build.gradle` |
| **Quick run command** | `./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest` |
| **Full focused command** | `./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.accelerator.lowering.GpuCompoundPatternDetectorTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests PreparedExecutionBuildTest --tests NormalizationExecutionTest --tests GpuCoverageSummaryTest --tests CompiledGraphTraceTest --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageRegressionGateTest --tests StandardWorkloadsTest --tests BenchmarkSessionTest` |
| **Native Metal command** | `./gradlew metalTest` |
| **Optional CUDA native command** | `./gradlew cudaTest` when `nvcc` and CUDA hardware are available |
| **Estimated runtime** | Focused Java gate ~2-3 minutes locally; Metal native gate depends on shim/device availability. |

## Sampling Rate

- **After every task commit:** Run the focused test class for the touched lowering/backend/report area.
- **After every plan wave:** Run the plan verification command from the relevant `24-0x-PLAN.md`.
- **Before phase verification:** Run the full focused command, `./gradlew metalTest`, `git diff --check`, and `git status --short profiles/platform`.
- **Max feedback latency:** 180 seconds for focused gates, excluding native Metal build/device variability.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 24-01-01 | 01 | 1 | GPUNORMX-01, GPUNORMX-02 | T-24-01, T-24-02 | Normalization lowers to a shared DAG with epsilon metadata, repeated keep-dims reductions, and gamma/beta shape validation before backend admission. | unit/lowering | `./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.accelerator.lowering.GpuCompoundPatternDetectorTest` | W0 | passed |
| 24-02-01 | 02 | 2 | GPUNORMX-01, GPUNORMX-02 | T-24-03, T-24-04 | Metal and CUDA support or explicitly reject the primitive set needed by the lowered normalization DAG. | native/portable | `./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests PreparedExecutionBuildTest` and `./gradlew metalTest` | W0 | passed |
| 24-03-01 | 03 | 3 | GPUNORMX-01, GPUNORMX-02, GPUNORMX-03 | T-24-05, T-24-06 | Legal dense FLOAT32 normalization is planner-supported; unsupported dtype, layout, rank, and shape variants reject with stable reason prefixes. | matrix/planner/trace | `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests GpuCoverageSummaryTest --tests CompiledGraphTraceTest` | W0 | passed |
| 24-04-01 | 04 | 4 | GPUNORMX-02, GPUNORMX-03 | T-24-07, T-24-08, T-24-09 | Selected-GPU normalization matches CPU output, hot-path gates require native evidence, and profile artifacts remain outside phase evidence. | parity/report/hygiene | `./gradlew test --tests PreparedExecutionBuildTest --tests NormalizationExecutionTest --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageRegressionGateTest --tests StandardWorkloadsTest` | W0 | passed |

## Wave 0 Requirements

Existing infrastructure covers the phase starting point:

- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java`
- `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java`
- `src/main/java/backend/metal/lowering/MetalPartitionSupport.java`
- `src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java`
- `src/main/java/tuning/benchmark/report/GpuTargetCoverageTruth.java`
- `src/main/java/tuning/benchmark/report/GpuHotPathCoverageTargets.java`
- `src/test/java/backend/accelerator/lowering/AcceleratorSubgraphLowererTest.java`
- `src/test/java/backend/accelerator/lowering/GpuLoweringCoverageMatrixTest.java`
- `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java`
- `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java`
- `src/test/java/PreparedExecutionBuildTest.java`
- `src/test/java/NormalizationExecutionTest.java`
- `src/test/java/GpuCoverageSummaryTest.java`
- `src/test/java/CompiledGraphTraceTest.java`

## Requirement Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| GPUNORMX-01 | COVERED | `AcceleratorSubgraphLowererTest`, `GpuCompoundPatternDetectorTest`, `GpuLoweringCoverageMatrixTest`, `MetalRegionLowererTest`, and `CudaRegionLowererTest` cover shared normalization DAG lowering, `ADD_SCALAR`, supported coverage rows, and backend legality. |
| GPUNORMX-02 | COVERED | `NormalizationExecutionTest`, `MetalMpsFfmBridgeTest`, and `PreparedExecutionBuildTest` cover epsilon, gamma/beta broadcasting, selected-GPU parity, Metal native execution, and unsupported shape handling. |
| GPUNORMX-03 | COVERED | `PreparedExecutionBuildTest`, `GpuHotPathCoverageTargetsTest`, `GpuCoverageRegressionGateTest`, `GpuCoverageSummaryTest`, `CompiledGraphTraceTest`, `StandardWorkloadsTest`, and `BenchmarkSessionTest` cover native evidence, lowered primitive counts, visible materialization boundaries, coverage reports, and representative normalization workloads. |

## Manual-Only Verifications

CUDA native compile/execution remains manual-only in this environment because `nvcc` is not installed locally. Portable CUDA Java tests passed and assert stable capability/unavailable behavior, but canonical CUDA native execution should be run in a CUDA-equipped lane with:

```bash
./gradlew cudaTest
```

## Validation Sign-Off

- [x] All tasks have automated verification or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all existing references.
- [x] No watch-mode flags.
- [x] Feedback latency target documented.
- [x] `nyquist_compliant: true` set in frontmatter.

## Execution Evidence

| Command | Result |
|---------|--------|
| `./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.accelerator.lowering.GpuCompoundPatternDetectorTest` | Passed |
| `./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests PreparedExecutionBuildTest` | Passed |
| `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests GpuCoverageSummaryTest --tests CompiledGraphTraceTest` | Passed |
| `./gradlew test --tests PreparedExecutionBuildTest --tests NormalizationExecutionTest --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageRegressionGateTest --tests StandardWorkloadsTest` | Passed |
| `./gradlew test --tests BenchmarkSessionTest --tests GpuCoverageSummaryTest --tests CompiledGraphTraceTest --tests GpuHotPathCoverageTargetsTest` | Passed |
| `./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.accelerator.lowering.GpuCompoundPatternDetectorTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests PreparedExecutionBuildTest --tests NormalizationExecutionTest --tests GpuCoverageSummaryTest --tests CompiledGraphTraceTest --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageRegressionGateTest --tests StandardWorkloadsTest --tests BenchmarkSessionTest` | Passed |
| `./gradlew metalTest` | Passed |
| `git diff --check` | Passed |
| `command -v nvcc` | Not available locally; CUDA native gate not run |
| `git status --short profiles/platform` | Local profile artifacts remain dirty but are not staged phase evidence |

## Validation Audit 2026-05-01

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

Phase state detected as State A: existing `24-VALIDATION.md` plus completed plan summaries. No additional Nyquist tests were required after requirement-to-test cross-reference.

**Approval:** verified 2026-05-01
