---
phase: 2
slug: metal-layout-aware-device-flow
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-04-29
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
| **Full suite command** | `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.MetalBufferTraceSmokeTest` |
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
| 2-01-01 | 01 | 1 | METAL-01, METAL-03 | T-2-01 | Layout policy accepts only explicitly legal Metal layouts and rejects unsupported layouts before native execution | unit | `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest` | ✅ | ⬜ pending |
| 2-01-02 | 01 | 1 | METAL-01 | T-2-02 | Zero-offset/device-representable layout decisions do not allocate or publish invalid bindings | unit | `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest` | ✅ | ⬜ pending |
| 2-02-01 | 02 | 1 | METAL-02, METAL-03 | T-2-03 | Native ABI/capability checks gate any new layout-aware Metal execution before buffers are used | unit/native | `./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest` | ✅ | ⬜ pending |
| 2-02-02 | 02 | 1 | METAL-04 | T-2-04 | Device-to-CPU materialization is supported only for layouts that can be copied correctly into runtime tensor storage | unit | `./gradlew test --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest` | ✅ | ⬜ pending |
| 2-03-01 | 03 | 2 | METAL-01, METAL-02, METAL-04 | T-2-05 | Representative layout-heavy Metal graph avoids accidental CPU materialization and reports real buffer path diagnostics | integration/native | `./gradlew test --tests backend.metal.MetalBufferTraceSmokeTest` | ✅ | ⬜ pending |
| 2-03-02 | 03 | 2 | METAL-04 | T-2-06 | Forward-backward graph materializes graph outputs and gradients correctly or falls back visibly | integration | `./gradlew metalTest` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Update `src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java` so old dense-only fallback assertions become explicit unsupported-layout tests, and add legal layout-aware success tests.
- [ ] Add or update `src/test/java/backend/metal/buffer/MetalBufferAllocatorTest.java` for any new materialization or device-contiguous transform behavior.
- [ ] Add native capability/version tests in `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java` if any native ABI symbol changes.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Native Metal shim availability on local Apple hardware | METAL-02, METAL-04 | CI or developer machines may not have Metal shim configured | Run `./gradlew metalTest` with `synaptik.metal.mps.lib` or the default native build path configured |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies.
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify.
- [ ] Wave 0 covers all missing or changed test references.
- [ ] No watch-mode flags.
- [ ] Feedback latency target is under 120 seconds for targeted test commands.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** pending
