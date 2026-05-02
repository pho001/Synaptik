---
phase: 31
slug: bool-producing-metal-compute
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-02
---

# Phase 31 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | Gradle / JUnit |
| Config file | `build.gradle` |
| Quick run command | `./gradlew test --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.metal.lowering.MetalRegionLowererTest` |
| Full suite command | `./gradlew classes && ./gradlew metalTest` |
| Estimated runtime | focused tests: seconds; native suite: environment-dependent |

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 31-01 | 31-01..02 | 1-2 | `METALBOOL-01` | T-31-01 | Compare/logical BOOL outputs are admitted only through operation-specific decisions and descriptor tests. | unit/native | `./gradlew test --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest && ./gradlew metalTest` | yes | green |
| 31-02 | 31-03 | 3 | `METALBOOL-02` | T-31-02 | `compare -> WHERE -> elementwise` stays inside a Metal lowered region with BOOL residency evidence. | integration/coverage | `./gradlew test --tests PreparedExecutionBuildTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest` | yes | green |
| 31-03 | 31-02..04 | 2-4 | `METALBOOL-03` | T-31-03 | BOOL reductions execute natively only for scoped supported semantics; unsupported variants reject visibly. | native/report | `./gradlew test --tests GpuCoverageSummaryTest && ./gradlew metalTest` | yes | green |

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
