---
phase: 28
slug: coverage-regression-closure
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-02
---

# Phase 28 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | JUnit Jupiter via Gradle |
| **Config file** | `build.gradle` |
| **Quick run command** | `./gradlew test --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageRegressionGateTest --tests GpuCoverageSummaryTest` |
| **Full focused command** | `./gradlew test --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageRegressionGateTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests StandardWorkloadsTest` |
| **Native Metal command** | `./gradlew metalTest` |
| **Optional CUDA native command** | `./gradlew cudaTest` when `nvcc` and CUDA hardware are available |
| **Estimated runtime** | Focused Java gate ~2-3 minutes locally; Metal native gate depends on shim/device availability. |

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 28-01-01 | 01 | 1 | GPUCLOSE-01, GPUCLOSE-02 | policy/truth | `./gradlew test --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageSummaryTest` | passed |
| 28-02-01 | 02 | 2 | GPUCLOSE-01, GPUCLOSE-02 | regression gates | `./gradlew test --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests BenchmarkSuiteSessionTest` | passed |
| 28-03-01 | 03 | 3 | GPUCLOSE-01, GPUCLOSE-03 | report/docs | `./gradlew test --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests GpuCoverageSummaryTest` | passed |
| 28-04-01 | 04 | 4 | GPUCLOSE-01, GPUCLOSE-02, GPUCLOSE-03 | closure/native/hygiene | `./gradlew test --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageRegressionGateTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests StandardWorkloadsTest` plus `./gradlew metalTest` | passed |

## Requirement Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| GPUCLOSE-01 | COVERED | Target policies and suite report fields cover region length, fallback count, CPU materialization count, lowered primitive count, backend path, and handoff evidence. |
| GPUCLOSE-02 | COVERED | Regression gate tests fail supported targets on tensor-array, CPU fallback, materialization, lost native buffer binding, lost selected region, and lost lowered primitive evidence. |
| GPUCLOSE-03 | COVERED | Deterministic baseline delta rendering, docs, and artifact hygiene checks keep local profile outputs out of closure proof. |

## Manual-Only Verifications

CUDA native compile/execution remains manual-only in this environment because `nvcc` is not installed locally. Portable CUDA Java tests cover the report/gate semantics; canonical CUDA native execution should be run in a CUDA-equipped lane with:

```bash
./gradlew cudaTest
```

## Execution Evidence

| Command | Result |
|---------|--------|
| `./gradlew test --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageRegressionGateTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests StandardWorkloadsTest` | Passed |
| `./gradlew metalTest` | Passed |
| `git diff --check` | Passed |
| `command -v nvcc` | Not available locally; CUDA native gate not run |
| `git status --short profiles/platform` | Local profile artifacts remain dirty but are not staged Phase 28 evidence |

## Validation Sign-Off

- [x] All tasks have automated verification or documented native-lane limits.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all existing references.
- [x] No watch-mode flags.
- [x] Feedback latency target documented.
- [x] `nyquist_compliant: true` set in frontmatter.

## Validation Audit 2026-05-02

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

Phase state detected as State A: existing `28-VALIDATION.md` plus completed plan summaries. No additional Nyquist tests were required after requirement-to-test cross-reference.

**Approval:** verified 2026-05-02
