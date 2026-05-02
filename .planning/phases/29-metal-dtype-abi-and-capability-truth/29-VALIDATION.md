---
phase: 29
slug: metal-dtype-abi-and-capability-truth
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-02
---

# Phase 29 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | Gradle / JUnit |
| Config file | `build.gradle` |
| Quick run command | `./gradlew test --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest` |
| Full suite command | `./gradlew classes && ./gradlew metalTest` |
| Estimated runtime | focused tests: seconds; native suite: environment-dependent |

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 29-01 | 29-01..04 | 1-4 | `METALDTYPE-01` | T-29-01 | DType residency and native compute/output decisions remain role-specific. | unit | `./gradlew test --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.accelerator.residency.AcceleratorDTypeResidencyPolicyTest` | yes | green |
| 29-02 | 29-02 | 2 | `METALDTYPE-02` | T-29-02 | Missing dtype ABI v3 symbols do not widen support or disable FLOAT32. | native/unit | `./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest && ./gradlew metalTest` | yes | green |
| 29-03 | 29-03..04 | 3-4 | `METALDTYPE-03` | T-29-03 | Coverage/docs expose dtype reason codes without overclaiming execution. | report/docs | `./gradlew test --tests GpuCoverageSummaryTest && ./gradlew classes` | yes | green |

## Wave 0 Requirements

Existing Gradle/JUnit and Metal native test infrastructure covers the phase requirements.

## Manual-Only Verifications

All phase behaviors have automated verification. Local profile artifact exclusion was checked through `git status --short profiles/platform` during phase execution.

## Validation Sign-Off

- [x] All requirements map to automated verification.
- [x] Sampling continuity is satisfied by per-wave Gradle commands in the plan summaries.
- [x] No watch-mode flags are required.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-02
