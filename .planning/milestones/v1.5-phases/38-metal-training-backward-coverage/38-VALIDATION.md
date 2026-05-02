---
phase: 38
slug: metal-training-backward-coverage
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-02
---

# Phase 38 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | Gradle / JUnit |
| Config file | `build.gradle` |
| Quick run command | `./gradlew test --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest` |
| Full suite command | `./gradlew classes && ./gradlew metalTest` |
| Estimated runtime | focused tests: seconds; native suite: environment-dependent |

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 38-01 | 38-01 | 1 | `METALTRAIN-01` | T-38-01 | Backward execution truth is separated from forward support and requires native-buffer trace plus CPU parity. | integration/parity | `./gradlew test --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest && ./gradlew metalTest` | yes | green |
| 38-02 | 38-02 | 2 | `METALTRAIN-02` | T-38-02 | Unsupported backward families remain explicit with stable blocker reasons. | coverage/rejection | `./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest` | yes | green |
| 38-03 | 38-03..04 | 3-4 | `METALTRAIN-03` | T-38-03 | Reports distinguish allowed gradient publication from hidden internal CPU materialization. | report/gate | `./gradlew test --tests StandardWorkloadsTest --tests SourceTreeHygieneTest --tests BenchmarkSuiteSessionTest` | yes | green |

## Wave 0 Requirements

Existing Gradle/JUnit and Metal native test infrastructure covers the phase requirements.

## Manual-Only Verifications

All phase behaviors have automated verification. Full end-to-end training on Metal remains explicitly outside this phase.

## Validation Sign-Off

- [x] All requirements map to automated verification.
- [x] Sampling continuity is satisfied by per-wave Gradle commands in the plan summaries.
- [x] No watch-mode flags are required.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-02
