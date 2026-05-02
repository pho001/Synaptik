---
phase: 33
slug: gpu-layout-router-and-strided-materialization
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-02
---

# Phase 33 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | Gradle / JUnit |
| Config file | `build.gradle` |
| Quick run command | `./gradlew test --tests backend.accelerator.buffer.AcceleratorLayoutTransformPlannerTest --tests backend.metal.MetalLayoutAwareDeviceFlowTest` |
| Full suite command | `./gradlew classes && ./gradlew metalTest` |
| Estimated runtime | focused tests: seconds; native suite: environment-dependent |

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 33-01 | 33-01 | 1 | `METALLAYOUT-01` | T-33-01 | Layout router route kinds and rejection reasons distinguish metadata, dense repair, broadcast repair, and unsupported strided native compute. | unit | `./gradlew test --tests backend.accelerator.buffer.AcceleratorLayoutTransformPlannerTest` | yes | green |
| 33-02 | 33-02..03 | 2-3 | `METALLAYOUT-02` | T-33-02 | Metal FLOAT32 dense/broadcast GPU-side materialization validates physical spans and trace metadata. | native/integration | `./gradlew test --tests backend.metal.MetalLayoutAwareDeviceFlowTest --tests PreparedExecutionBuildTest && ./gradlew metalTest` | yes | green |
| 33-03 | 33-04 | 4 | `METALLAYOUT-03` | T-33-03 | Coverage gates require buffer-binding execution and visible layout materialization evidence. | coverage | `./gradlew test --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests StandardWorkloadsTest` | yes | green |

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
