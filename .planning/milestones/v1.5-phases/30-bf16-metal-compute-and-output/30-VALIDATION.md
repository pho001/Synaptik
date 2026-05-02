---
phase: 30
slug: bf16-metal-compute-and-output
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-02
---

# Phase 30 - Validation Strategy

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
| 30-01 | 30-01..02 | 1-2 | `METALBF16-01` | T-30-01 | BF16 descriptor/cache support is dtype-aware and scoped to admitted operation families. | unit/native | `./gradlew test --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.metal.lowering.MetalRegionLowererTest && ./gradlew metalTest` | yes | green |
| 30-02 | 30-01..03 | 1-3 | `METALBF16-02` | T-30-02 | BF16 raw storage, upload/readback, and tolerance policy are verified against CPU-compatible behavior. | native/parity | `./gradlew metalTest` | yes | green |
| 30-03 | 30-03..04 | 3-4 | `METALBF16-03` | T-30-03 | BF16 hot paths require native buffer binding and reject hidden CPU/tensor-array exits. | coverage | `./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests BenchmarkSuiteSessionTest` | yes | green |

## Wave 0 Requirements

Existing Gradle/JUnit and Metal native test infrastructure covers the phase requirements.

## Manual-Only Verifications

All phase behaviors have automated verification. Local tuning/profile artifacts are intentionally excluded from canonical evidence.

## Validation Sign-Off

- [x] All requirements map to automated verification.
- [x] Sampling continuity is satisfied by per-wave Gradle commands in the plan summaries.
- [x] No watch-mode flags are required.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-02
