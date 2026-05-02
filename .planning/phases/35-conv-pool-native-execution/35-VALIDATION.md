---
phase: 35
slug: conv-pool-native-execution
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-02
---

# Phase 35 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | Gradle / JUnit |
| Config file | `build.gradle` |
| Quick run command | `./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest` |
| Full suite command | `./gradlew classes && ./gradlew metalTest` |
| Estimated runtime | focused tests: seconds; native suite: environment-dependent |

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 35-01 | 35-01..02 | 1-2 | `METALCONVPOOL-01` | T-35-01 | Dense FLOAT32 NCHW/OIHW conv admits only legal forward cases and rejects unsupported variants. | unit/native | `./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests Conv2dExecutionTest && ./gradlew metalTest` | yes | green |
| 35-02 | 35-03 | 3 | `METALCONVPOOL-02` | T-35-02 | Max/avg pool native paths verify supported divisor/window semantics and reject unsupported padding modes. | native/parity | `./gradlew test --tests Pool2dExecutionTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest && ./gradlew metalTest` | yes | green |
| 35-03 | 35-04 | 4 | `METALCONVPOOL-03` | T-35-03 | Conv/pool hot-path targets require native buffer binding and no CPU/tensor-array fallbacks. | coverage | `./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests StandardWorkloadsTest` | yes | green |

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
