---
phase: 1
slug: accelerator-buffer-layout-abi
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-04-29
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for accelerator buffer layout ABI work.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | JUnit 5 via Gradle |
| **Config file** | `build.gradle` |
| **Quick run command** | `./gradlew test --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest --tests graph.execution.ExecutionStateResidencyTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` |
| **Full suite command** | `./gradlew test --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.buffer.MetalBufferAllocatorTest --tests graph.execution.ExecutionStateResidencyTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` |
| **Estimated runtime** | ~30-90 seconds for targeted tests, depending on Gradle/JVM warmup |

---

## Sampling Rate

- **After every task commit:** Run the quick targeted Gradle command above.
- **After every plan wave:** Run the full targeted Gradle command above.
- **Before `$gsd-verify-work`:** Targeted suite must be green; run `./gradlew classes` as a compile-wide sanity check.
- **Max feedback latency:** 90 seconds for targeted feedback on a warm Gradle daemon.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | ABI-01, ABI-03 | T-1-01 | Layout byte length and class are derived without native handle access | unit | `./gradlew test --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | ABI-01, ABI-02 | T-1-02 | Device binding exposes shared layout while backend handles stay backend-owned | unit | `./gradlew test --tests graph.execution.ExecutionStateResidencyTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest` | ✅ | ⬜ pending |
| 1-01-03 | 01 | 1 | ABI-03, ABI-04 | T-1-03 | Metal buffer preflight reports precise layout reason codes without unsafe native execution | unit | `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest` | ✅ | ⬜ pending |
| 1-01-04 | 01 | 1 | ABI-02, ABI-04 | T-1-04 | CUDA required-buffer mode remains explicit unavailable instead of silent fallback | unit | `./gradlew test --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` | ✅ | ⬜ pending |
| 1-01-05 | 01 | 2 | ABI-01, ABI-02, ABI-03, ABI-04 | T-1-05 | Profile/trace-visible diagnostics preserve stable reason names and layout details | integration | `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.buffer.MetalBufferAllocatorTest` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `src/test/java/backend/accelerator/buffer/AcceleratorBufferLayoutClassifierTest.java` — classification tests for dense contiguous, zero-offset view, non-zero-offset view, permuted/strided view, broadcast/zero-stride view, and unsupported layout cases.
- [ ] Update fake `DeviceBufferBinding` implementations in existing tests after the interface gains shared layout/access methods.

---

## Manual-Only Verifications

All phase behaviors have automated verification.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies.
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify.
- [ ] Wave 0 covers all missing test references.
- [ ] No watch-mode flags.
- [ ] Feedback latency target is under 90 seconds for targeted test commands.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** pending
