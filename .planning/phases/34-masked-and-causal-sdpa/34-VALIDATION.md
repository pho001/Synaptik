---
phase: 34
slug: masked-and-causal-sdpa
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-02
---

# Phase 34 - Validation Strategy

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
| 34-01 | 34-01 | 1 | `METALSDPAMASK-01` | T-34-01 | SDPA mask semantics classify unmasked, external BOOL, causal, combined, and unsupported cases before admission. | unit/contract | `./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests GpuTargetSemanticsContractTest` | yes | green |
| 34-02 | 34-02..03 | 2-3 | `METALSDPAMASK-02` | T-34-02 | Native MPSGraph SDPA applies BOOL mask polarity and causal convention without CPU materialization. | native/parity | `./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests AttentionExecutionTest --tests PreparedExecutionBuildTest && ./gradlew metalTest` | yes | green |
| 34-03 | 34-04 | 4 | `METALSDPAMASK-03` | T-34-03 | `masked_sdpa_small` and transformer gates rely on trace/report evidence, not timing. | coverage/docs | `./gradlew test --tests GpuHotPathCoverageTargetsTest --tests StandardWorkloadsTest --tests GpuCoverageRegressionGateTest --tests GpuCoverageSummaryTest` | yes | green |

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
