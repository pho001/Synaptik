---
phase: 36
slug: scatter-and-index-gradient-semantics
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-02
---

# Phase 36 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | Gradle / JUnit |
| Config file | `build.gradle` |
| Quick run command | `./gradlew test --tests ScatterAddExecutionTest --tests GatherExecutionTest --tests TakeAlongAxisExecutionTest` |
| Full suite command | `./gradlew classes && ./gradlew metalTest` |
| Estimated runtime | focused tests: seconds; native suite: environment-dependent |

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 36-01 | 36-01..02 | 1-2 | `METALSCATTER-01` | T-36-01 | Scatter/index-gradient ops reject native support until duplicate-index semantics are proven. | unit/contract | `./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests GpuCoverageSummaryTest` | yes | green |
| 36-02 | 36-02 | 2 | `METALSCATTER-02` | T-36-02 | CPU parity fixtures cover duplicate accumulation and out-of-bounds behavior. | parity | `./gradlew test --tests ScatterAddExecutionTest --tests GatherExecutionTest --tests TakeAlongAxisExecutionTest` | yes | green |
| 36-03 | 36-03..04 | 3-4 | `METALSCATTER-03` | T-36-03 | Adjacent Metal producers can stay selected while scatter/index gradients remain visible CPU steps. | integration/coverage | `./gradlew test --tests PreparedExecutionBuildTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests StandardWorkloadsTest` | yes | green |

## Wave 0 Requirements

Existing Gradle/JUnit and Metal native test infrastructure covers the phase requirements.

## Manual-Only Verifications

All phase behaviors have automated verification. Native duplicate-index execution remains future work and is explicitly rejected, not manually accepted.

## Validation Sign-Off

- [x] All requirements map to automated verification.
- [x] Sampling continuity is satisfied by per-wave Gradle commands in the plan summaries.
- [x] No watch-mode flags are required.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-02
