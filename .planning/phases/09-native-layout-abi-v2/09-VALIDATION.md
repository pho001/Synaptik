---
phase: 09
slug: native-layout-abi-v2
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-30
---

# Phase 09 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | JUnit 5 via Gradle |
| Config file | `build.gradle` |
| Quick run command | `./gradlew classes` |
| Full suite command | `./gradlew test --tests backend.accelerator.buffer.* --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.cuda.bridge.CudaFfmBridgeTest` |
| Estimated runtime | 60-180 seconds for focused portable tests |

## Sampling Rate

- After every task commit: run the task-specific focused Gradle filter.
- After every plan wave: run the plan verification command listed in the PLAN.md.
- Before phase verification: run `./gradlew classes` and all focused accelerator buffer, Metal bridge, CUDA bridge, and prepared-executable tests touched by the phase.
- Max feedback latency: one plan wave.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 09-01-01 | 01 | 1 | GPULAYOUT-01 | T-09-01 | descriptor rejects invalid or overflowing layout metadata | unit | `./gradlew test --tests backend.accelerator.buffer.AcceleratorLayoutAbiV2DescriptorTest` | yes | pending |
| 09-01-02 | 01 | 1 | GPULAYOUT-01 | T-09-01 | descriptor exposes physical span separately from logical bytes | unit | `./gradlew test --tests backend.accelerator.buffer.AcceleratorLayoutAbiV2DescriptorTest` | yes | pending |
| 09-02-01 | 02 | 2 | GPULAYOUT-02 | T-09-02 | missing layout ABI v2 symbols do not break dense bridge availability | unit | `./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.cuda.bridge.CudaFfmBridgeTest` | yes | pending |
| 09-02-02 | 02 | 2 | GPULAYOUT-02 | T-09-02 | native stubs export versioned optional layout ABI symbols | compile/optional native | `./gradlew classes` | yes | pending |
| 09-03-01 | 03 | 3 | GPULAYOUT-03 | T-09-03 | AUTO fallback and REQUIRE failure use ABI-specific reason codes | unit | `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` | yes | pending |
| 09-03-02 | 03 | 3 | GPULAYOUT-03 | T-09-03 | docs explain supported, unavailable, mismatch, and unsupported outcomes | grep | `rg -n "layout ABI v2|NATIVE_LAYOUT_ABI_UNAVAILABLE|NATIVE_LAYOUT_ABI_VERSION_MISMATCH" docs src/test/java` | yes | pending |

## Wave 0 Requirements

Existing JUnit/Gradle infrastructure covers all phase requirements.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Metal native layout ABI symbol load on a local Apple machine | GPULAYOUT-02 | depends on local Metal native shim build | run `./gradlew metalTest` after building or configuring the Metal shim |
| CUDA native layout ABI symbol load on a CUDA machine | GPULAYOUT-02 | depends on local CUDA toolkit and driver | run `./gradlew buildCudaGraphShim cudaTest` |

## Validation Sign-Off

- [x] All tasks have automated verify commands or explicit optional-native notes.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing test infrastructure.
- [x] No watch-mode flags.
- [x] Feedback latency below one plan wave.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** pending execution

