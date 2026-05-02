---
phase: 39
slug: metal-backend-router-and-zero-copy-closure
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-02
---

# Phase 39 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | Gradle / JUnit |
| Config file | `build.gradle` |
| Quick run command | `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.bridge.MetalMpsBridgeExecutionStatsTest` |
| Full suite command | `./gradlew classes && ./gradlew metalTest` |
| Estimated runtime | focused tests: seconds; native suite: environment-dependent |

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 39-01 | 39-01..02 | 1-2 | `METALROUTER-01` | T-39-01 | Router decisions stay inside already-selected Metal regions and expose rejected custom-kernel evidence. | unit/integration | `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.lowering.MetalRegionLowererTest` | yes | green |
| 39-02 | 39-03 | 3 | `METALROUTER-02` | T-39-02 | Native output behavior is classified as `MPSGRAPH_RESULT_COPY`; zero-copy is not claimed without alias proof. | native/report | `./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.bridge.MetalMpsBridgeExecutionStatsTest && ./gradlew metalTest` | yes | green |
| 39-03 | 39-04 | 4 | `METALROUTER-03` | T-39-03 | Coverage/report gates expose route counts, rejected route reasons, copy strategy, and fallback counters. | coverage/docs | `./gradlew test --tests BenchmarkSessionTest --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests BenchmarkSuiteSessionTest --tests SourceTreeHygieneTest` | yes | green |

## Wave 0 Requirements

Existing Gradle/JUnit and Metal native test infrastructure covers the phase requirements.

## Manual-Only Verifications

All phase behaviors have automated verification. Direct MPSGraph output-buffer writes are not accepted as manual evidence; current behavior remains explicitly classified as `MPSGRAPH_RESULT_COPY`.

## Validation Sign-Off

- [x] All requirements map to automated verification.
- [x] Sampling continuity is satisfied by per-wave Gradle commands in the plan summaries.
- [x] No watch-mode flags are required.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-02
