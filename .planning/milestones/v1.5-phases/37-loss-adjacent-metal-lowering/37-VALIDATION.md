---
phase: 37
slug: loss-adjacent-metal-lowering
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-02
---

# Phase 37 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | Gradle / JUnit |
| Config file | `build.gradle` |
| Quick run command | `./gradlew test --tests PreparedExecutionBuildTest --tests IndexTargetCrossEntropyLossExecutionTest --tests IndexTargetNllLossExecutionTest` |
| Full suite command | `./gradlew classes && ./gradlew metalTest` |
| Estimated runtime | focused tests: seconds; native suite: environment-dependent |

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 37-01 | 37-01 | 1 | `METALLOSS-01` | T-37-01 | Dense loss lowering follows CPU-compatible formula, output shape, and class-axis behavior. | unit/native | `./gradlew test --tests PreparedExecutionBuildTest --tests GpuCoverageSummaryTest && ./gradlew metalTest` | yes | green |
| 37-02 | 37-02 | 2 | `METALLOSS-02` | T-37-02 | Index-target CE/NLL and gradient variants reject visibly until ignore-index/weight/denominator semantics are proven. | parity/rejection | `./gradlew test --tests IndexTargetCrossEntropyLossExecutionTest --tests IndexTargetNllLossExecutionTest` | yes | green |
| 37-03 | 37-03..04 | 3-4 | `METALLOSS-03` | T-37-03 | Training traces separate dense native loss support from index-target CPU blockers. | trace/coverage | `./gradlew test --tests CompiledGraphTraceTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests StandardWorkloadsTest --tests BenchmarkSuiteSessionTest` | yes | green |

## Wave 0 Requirements

Existing Gradle/JUnit and Metal native test infrastructure covers the phase requirements.

## Manual-Only Verifications

All phase behaviors have automated verification.

## Validation Sign-Off

- [x] All requirements map to automated verification.
- [x] Sampling continuity is satisfied by per-wave Gradle commands in the plan summaries.
- [x] No watch-mode flags are required.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-02
