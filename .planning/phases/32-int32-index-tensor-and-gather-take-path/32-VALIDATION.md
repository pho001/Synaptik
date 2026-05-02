---
phase: 32
slug: int32-index-tensor-and-gather-take-path
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-02
---

# Phase 32 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | Gradle / JUnit |
| Config file | `build.gradle` |
| Quick run command | `./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.metal.MetalMpsCapabilitiesTest` |
| Full suite command | `./gradlew classes && ./gradlew metalTest` |
| Estimated runtime | focused tests: seconds; native suite: environment-dependent |

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 32-01 | 32-01 | 1 | `METALINTIDX-01` | T-32-01 | INT32 is admitted for index input roles without claiming generic INT32 compute/output. | unit/buffer | `./gradlew test --tests backend.accelerator.residency.AcceleratorDTypeResidencyPolicyTest --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.metal.buffer.MetalBufferAllocatorTest` | yes | green |
| 32-02 | 32-02..03 | 2-3 | `METALINTIDX-02` | T-32-02 | `GATHER` and `TAKE_ALONG_AXIS` native paths match CPU axis/rank/index semantics or reject unsafe cases. | native/parity | `./gradlew test --tests GatherExecutionTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest && ./gradlew metalTest` | yes | green |
| 32-03 | 32-03..04 | 3-4 | `METALINTIDX-03` | T-32-03 | `LOG_SOFTMAX -> TAKE_ALONG_AXIS` remains one Metal-owned region without CPU materialization. | integration/coverage | `./gradlew test --tests PreparedExecutionBuildTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests BenchmarkSuiteSessionTest` | yes | green |

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
