---
phase: 27
slug: conv-pool-and-bool-compare-outputs
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-02
---

# Phase 27 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | JUnit Jupiter via Gradle |
| **Config file** | `build.gradle` |
| **Quick run command** | `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuTargetSemanticsContractTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest` |
| **Full focused command** | `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuTargetSemanticsContractTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest.gpuMetalWhereUsesCpuProducedComparePredicateAsExplicitBoolBoundary --tests Conv2dExecutionTest --tests Pool2dExecutionTest --tests BoolTensorInfrastructureTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest` |
| **Native Metal command** | `./gradlew metalTest` |
| **Optional CUDA native command** | `./gradlew cudaTest` when `nvcc` and CUDA hardware are available |
| **Estimated runtime** | Focused Java gate ~2-3 minutes locally; Metal native gate depends on shim/device availability. |

## Sampling Rate

- **After every task commit:** Run the focused test class for the touched matrix, planner, prepared-execution, or report area.
- **After every plan wave:** Run the verification commands listed in the corresponding `27-0x-PLAN.md`.
- **Before phase verification:** Run the full focused command, `./gradlew metalTest`, `git diff --check`, and profile artifact status checks.
- **Max feedback latency:** 180 seconds for focused Java gates, excluding native Metal build/device variability.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 27-01-01 | 01 | 1 | GPUCONVBOOL-01, GPUCONVBOOL-02 | T-27-01, T-27-02, T-27-03 | Every conv/pool and BOOL-producing target has an explicit Metal/CUDA matrix row and semantics contract before planner admission. | matrix/contract | `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuTargetSemanticsContractTest` | W0 | passed |
| 27-02-01 | 02 | 2 | GPUCONVBOOL-02, GPUCONVBOOL-03 | T-27-02, T-27-04 | `GT -> WHERE -> RELU` keeps the BOOL-producing compare as a visible CPU boundary while the adjacent supported GPU region can accept an external BOOL predicate. | prepared-execution/planner | `./gradlew test --tests PreparedExecutionBuildTest.gpuMetalWhereUsesCpuProducedComparePredicateAsExplicitBoolBoundary --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest` | W0 | passed |
| 27-03-01 | 03 | 3 | GPUCONVBOOL-01 | T-27-01, T-27-03, T-27-07 | Conv/pool targets reject with `CAPABILITY_MISSING` and CPU conv/pool parity remains the baseline. | planner/parity | `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests Conv2dExecutionTest --tests Pool2dExecutionTest` | W0 | passed |
| 27-04-01 | 04 | 4 | GPUCONVBOOL-01, GPUCONVBOOL-02, GPUCONVBOOL-03 | T-27-05, T-27-06, T-27-08 | Docs, reports, and phase evidence agree that native conv/pool and native BOOL output compute remain unclaimed; local profile artifacts are not proof. | report/docs/hygiene | `./gradlew test --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest` plus `./gradlew metalTest` and `git diff --check` | W0 | passed |

## Wave 0 Requirements

Existing infrastructure covers the phase starting point:

- `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java`
- `src/main/java/backend/accelerator/lowering/GpuTargetSemanticsContract.java`
- `src/main/java/backend/metal/lowering/MetalPartitionSupport.java`
- `src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java`
- `src/main/java/tuning/benchmark/report/GpuCoverageSummary.java`
- `src/test/java/backend/accelerator/lowering/GpuLoweringCoverageMatrixTest.java`
- `src/test/java/GpuTargetSemanticsContractTest.java`
- `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java`
- `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java`
- `src/test/java/PreparedExecutionBuildTest.java`
- `src/test/java/Conv2dExecutionTest.java`
- `src/test/java/Pool2dExecutionTest.java`
- `src/test/java/BoolTensorInfrastructureTest.java`

## Requirement Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| GPUCONVBOOL-01 | COVERED | Matrix rows and planner tests cover conv/pool support-or-rejection with `CAPABILITY_MISSING`; `Conv2dExecutionTest` and `Pool2dExecutionTest` keep CPU parity baselines green. |
| GPUCONVBOOL-02 | COVERED | BOOL-producing rows reject with `UNSUPPORTED_DTYPE`; prepared execution proves external BOOL predicate residency for `WHERE` is separate from native BOOL-producing GPU compute. |
| GPUCONVBOOL-03 | COVERED | `GpuCoverageSummaryTest`, `BenchmarkSessionTest`, and docs verify the existing report schema exposes selected region length, lowered primitive count, backend path, CPU exits, and fallback evidence for the current state. |

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
| `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuTargetSemanticsContractTest` | Passed |
| `./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest` | Passed |
| `./gradlew test --tests PreparedExecutionBuildTest.gpuMetalWhereUsesCpuProducedComparePredicateAsExplicitBoolBoundary` | Passed |
| `./gradlew test --tests Conv2dExecutionTest --tests Pool2dExecutionTest --tests BoolTensorInfrastructureTest` | Passed |
| `./gradlew test --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest` | Passed |
| Focused combined Phase 27 slice | Passed |
| `./gradlew metalTest` | Passed |
| `git diff --check` | Passed |
| `command -v nvcc` | Not available locally; CUDA native gate not run |
| `git status --short profiles/platform` | Local profile artifacts remain dirty but are not staged phase evidence |

## Validation Audit 2026-05-02

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

Phase state detected as State A: existing `27-VALIDATION.md` plus completed plan summaries. No additional Nyquist tests were required after requirement-to-test cross-reference.

**Approval:** verified 2026-05-02
