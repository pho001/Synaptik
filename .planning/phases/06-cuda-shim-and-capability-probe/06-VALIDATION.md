---
phase: 06
slug: cuda-shim-and-capability-probe
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-30
---

# Phase 06 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | JUnit 5 via Gradle |
| **Config file** | `build.gradle` |
| **Quick run command** | `./gradlew test --tests backend.cuda.bridge.CudaFfmBridgeTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` |
| **Full suite command** | `./gradlew classes && ./gradlew test --tests backend.cuda.* --tests SourceTreeHygieneTest` |
| **Estimated runtime** | ~60-180 seconds |

---

## Sampling Rate

- **After every task commit:** Run the task's focused Gradle filter.
- **After every plan wave:** Run `./gradlew classes` and the CUDA-focused portable tests touched by the wave.
- **Before `$gsd-verify-work`:** Portable Java gate must be green; optional native CUDA gate may be skipped with explicit reason when local CUDA tooling/hardware is unavailable.
- **Max feedback latency:** 180 seconds for portable checks.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 1 | CUDA-01 | T-6-01 | Missing CUDA toolkit/hardware is explicit and non-crashing | unit/build | `./gradlew test --tests backend.cuda.bridge.CudaFfmBridgeTest` | ✅ | ⬜ pending |
| 06-01-02 | 01 | 1 | CUDA-01 | T-6-02 | CUDA native build outputs stay under `build/native/cuda/` | build/docs | `./gradlew classes` | ✅ | ⬜ pending |
| 06-02-01 | 02 | 1 | CUDA-01 | T-6-03 | ABI mismatch and unavailable CUDA use stable capability codes | unit | `./gradlew test --tests backend.cuda.bridge.CudaFfmBridgeTest` | ✅ | ⬜ pending |
| 06-02-02 | 02 | 1 | CUDA-01 | T-6-04 | Bridge capability state does not require CUDA hardware | unit | `./gradlew test --tests backend.cuda.bridge.CudaFfmBridgeTest` | ✅ | ⬜ pending |
| 06-03-01 | 03 | 2 | CUDA-02 | T-6-05 | Dense shared layout metadata is accepted without CUDA-specific common fields | unit | `./gradlew test --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest` | ❌ W0 | ⬜ pending |
| 06-03-02 | 03 | 2 | CUDA-02 | T-6-06 | REQUIRED mode fails before tensor-list fallback when native buffer execution is unavailable | unit | `./gradlew test --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements. `CudaFfmBridgeTest` and `PreparedCudaExecutableBufferPolicyTest` already exist; Plan 06-03 adds `CudaAcceleratorBufferBinderTest`.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Optional native CUDA shim build on a CUDA-capable machine | CUDA-01 | Local machine may not have CUDA toolkit or GPU | Run `./gradlew buildCudaGraphShim` and `./gradlew cudaTest`; record pass or skip reason in the plan summary. |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or existing test dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all existing references
- [x] No watch-mode flags
- [x] Feedback latency < 180s for portable checks
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-30
