---
phase: 2
slug: metal-layout-aware-device-flow
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-29
updated: 2026-04-30
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for Metal layout-aware device flow.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | JUnit 5 via Gradle |
| **Config file** | `build.gradle` |
| **Quick run command** | `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.buffer.MetalBufferAllocatorTest` |
| **Full suite command** | `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.MetalBufferTraceSmokeTest --tests backend.metal.MetalLayoutAwareDeviceFlowTest` |
| **Native gate command** | `./gradlew metalTest` |
| **Estimated runtime** | ~30-120 seconds for targeted tests, depending on native shim availability and Gradle warmup |

---

## Sampling Rate

- **After every task commit:** Run the quick targeted Gradle command above.
- **After every plan wave:** Run the full targeted Gradle command above.
- **Before `$gsd-verify-work`:** Run the full targeted command, `./gradlew classes`, and `./gradlew metalTest` when Metal native work changed.
- **Max feedback latency:** 120 seconds for targeted feedback on a warm Gradle daemon.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 2-01-01 | 01 | 1 | METAL-01, METAL-03 | T-002-01-01 | Layout policy classifies dense, zero-offset, permuted/strided, broadcast, and unsupported layouts with stable reason text before native execution. | unit | `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest` | ✅ | ✅ green |
| 2-01-02 | 01 | 1 | METAL-01, METAL-03 | T-002-01-02 | Metal binder preflight accepts compatible existing device bindings only after policy, dtype, shape, stride, offset, logical count, availability, and access checks. | unit | `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest` | ✅ | ✅ green |
| 2-01-03 | 01 | 1 | METAL-03 | T-002-01-03 | Unsupported layouts and required-mode failures stay visible through stable fallback reason codes and policy detail fragments. | unit | `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest` | ✅ | ✅ green |
| 2-02-01 | 02 | 1 | METAL-02, METAL-03 | T-002-02-01 | Legal logical-view outputs allocate dense physical Metal buffers and preserve logical metadata for later materialization. | unit | `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest` | ✅ | ✅ green |
| 2-02-02 | 02 | 1 | METAL-04 | T-002-02-02, T-002-02-05 | Device-to-CPU materialization supports only policy-approved matching layouts and scatters dense readback by shape, strides, and storage offset. | unit | `./gradlew test --tests backend.metal.buffer.MetalBufferAllocatorTest` | ✅ | ✅ green |
| 2-02-03 | 02 | 1 | METAL-02, METAL-03 | T-002-02-04 | Dense physical logical-view flow requires no new native layout ABI symbol or stride/offset native parameter. | unit/native | `./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest` | ✅ | ✅ green |
| 2-03-01 | 03 | 2 | METAL-01, METAL-02, METAL-04 | T-002-03-02 | End-to-end layout-aware Metal forward and forward-backward graphs preserve CPU parity, gradient publication, and visible fallback behavior. | integration/native | `./gradlew test --tests backend.metal.MetalLayoutAwareDeviceFlowTest` | ✅ | ✅ green |
| 2-03-02 | 03 | 2 | METAL-01, METAL-02, METAL-03, METAL-04 | T-002-03-01, T-002-03-03 | Trace smoke coverage reports buffer path, storage residency, CPU materialization reasons, native copy fields, and unsupported-layout fallback. | integration/native | `./gradlew test --tests backend.metal.MetalBufferTraceSmokeTest` | ✅ | ✅ green |
| 2-03-03 | 03 | 2 | METAL-01, METAL-02, METAL-03, METAL-04 | T-002-03-05 | Phase verification gate runs focused Java tests, `classes`, and full native `metalTest` without relying on default debug benchmark tests. | integration/native | `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.MetalBufferTraceSmokeTest --tests backend.metal.MetalLayoutAwareDeviceFlowTest` and `./gradlew metalTest` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] Update `src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java` so old dense-only fallback assertions become explicit unsupported-layout tests, and add legal layout-aware success tests.
- [x] Add or update `src/test/java/backend/metal/buffer/MetalBufferAllocatorTest.java` for any new materialization or device-contiguous transform behavior.
- [x] Add native capability/version tests in `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java` proving no new native layout ABI symbol is required.

---

## Validation Audit 2026-04-30

| Metric | Count |
|--------|-------|
| Requirements audited | 4 |
| Plan tasks audited | 9 |
| Automated test files verified | 5 |
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

## Audit Evidence

| Evidence | Result |
|----------|--------|
| `002-01-PLAN.md`, `002-02-PLAN.md`, and `002-03-PLAN.md` each contain automated verify commands for every task. | PASS |
| `002-01-SUMMARY.md`, `002-02-SUMMARY.md`, and `002-03-SUMMARY.md` record passed task-level verification and no unhandled threat flags. | PASS |
| `src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java` covers policy actions, legal logical-view success, unsupported-layout rejection, required-mode behavior, and adjacent device handoff. | PASS |
| `src/test/java/backend/metal/buffer/MetalBufferAllocatorTest.java` covers logical-view output allocation, materializer gates, scatter readback, and rank/broadcast rejection. | PASS |
| `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java` proves dense physical logical-view bridge execution does not require native stride/offset ABI changes. | PASS |
| `src/test/java/backend/metal/MetalLayoutAwareDeviceFlowTest.java` covers end-to-end forward and forward-backward CPU parity, fallback visibility, and gradient publication. | PASS |
| `src/test/java/backend/metal/MetalBufferTraceSmokeTest.java` covers buffer path diagnostics, storage residency, materialization reasons, native copy fields, and unsupported-layout fallback. | PASS |
| `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.MetalBufferTraceSmokeTest --tests backend.metal.MetalLayoutAwareDeviceFlowTest` | PASS |
| `./gradlew metalTest` | PASS |

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| None | All Phase 2 requirements | All requirement-level behaviors have automated JUnit or Gradle Metal coverage. | N/A |

## Environment Notes

- Native Metal execution still depends on local Apple hardware and the Metal shim build path. The 2026-04-30 audit ran `./gradlew metalTest`, which built `build/native/apple/libsynaptik_apple_mps.dylib` and passed.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing or changed test references.
- [x] No watch-mode flags.
- [x] Feedback latency target is under 120 seconds for targeted test commands.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** verified 2026-04-30
