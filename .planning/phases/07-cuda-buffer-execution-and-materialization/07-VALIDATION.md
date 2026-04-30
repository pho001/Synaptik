---
phase: 07
slug: cuda-buffer-execution-and-materialization
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-30
---

# Phase 07 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | JUnit Jupiter 5.11.2 / Gradle 9.4.1 |
| **Config file** | `build.gradle` |
| **Quick run command** | `./gradlew test --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` |
| **Full suite command** | `./gradlew classes && ./gradlew test --tests backend.cuda.buffer.CudaBufferAllocatorTest --tests backend.cuda.buffer.CudaDeviceToCpuMaterializerTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests backend.cuda.bridge.CudaFfmBridgeTest --tests SourceTreeHygieneTest` |
| **Estimated runtime** | ~60-180 seconds |

---

## Sampling Rate

- **After every task commit:** Run the focused test command named in that task.
- **After every plan wave:** Run the full Phase 7 portable suite.
- **Before `$gsd-verify-work`:** Full portable suite must be green; optional native CUDA gate must pass or skip with a clear unavailable reason.
- **Max feedback latency:** 180 seconds for portable checks.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | CUDA-03 | T-7-01 | Buffer ABI advertised only when all symbols and Java support exist | unit | `./gradlew test --tests backend.cuda.bridge.CudaFfmBridgeTest --tests backend.cuda.buffer.CudaBufferAllocatorTest` | ✅ | ⬜ pending |
| 07-01-02 | 01 | 1 | CUDA-04 | T-7-02 | CUDA materializer copies bytes before CPU-current transition | unit | `./gradlew test --tests backend.cuda.buffer.CudaDeviceToCpuMaterializerTest` | ✅ | ⬜ pending |
| 07-02-01 | 02 | 2 | CUDA-03 | T-7-03 | Accepted buffer path does not call tensor-array execution | integration | `./gradlew test --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` | ✅ | ⬜ pending |
| 07-02-02 | 02 | 2 | CUDA-04 | T-7-04 | Graph output and CPU consumer materialize through `requireCpuReadable` | integration | `./gradlew test --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` | ✅ | ⬜ pending |
| 07-03-01 | 03 | 3 | CUDA-05 | T-7-05 | Compatible CUDA binding is reused across adjacent regions | integration | `./gradlew test --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` | ✅ | ⬜ pending |
| 07-03-02 | 03 | 3 | CUDA-03/CUDA-04/CUDA-05 | T-7-06 | Optional native CUDA checks skip cleanly without hardware/tooling | native-gated | `./gradlew buildCudaGraphShim cudaTest` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠ flaky*

---

## Wave 0 Requirements

Existing JUnit/Gradle infrastructure covers all Phase 7 requirements.

---

## Manual-Only Verifications

All required Phase 7 behaviors have automated or capability-gated verification.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify commands.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency target is under 180 seconds for portable checks.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-04-30
