---
phase: 06
slug: cuda-shim-and-capability-probe
status: verified
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
| 06-01-01 | 01 | 1 | CUDA-01 | T-6-01 | Missing CUDA toolkit/hardware is explicit and non-crashing | unit/build | `./gradlew test --tests backend.cuda.bridge.CudaFfmBridgeTest` | ✅ | ✅ green |
| 06-01-02 | 01 | 1 | CUDA-01 | T-6-02 | CUDA native build outputs stay under `build/native/cuda/` | build/docs | `./gradlew classes` | ✅ | ✅ green |
| 06-02-01 | 02 | 1 | CUDA-01 | T-6-03 | ABI mismatch and unavailable CUDA use stable capability codes | unit | `./gradlew test --tests backend.cuda.bridge.CudaFfmBridgeTest` | ✅ | ✅ green |
| 06-02-02 | 02 | 1 | CUDA-01 | T-6-04 | Bridge capability state does not require CUDA hardware | unit | `./gradlew test --tests backend.cuda.bridge.CudaFfmBridgeTest` | ✅ | ✅ green |
| 06-03-01 | 03 | 2 | CUDA-02 | T-6-05 | Dense shared layout metadata is accepted without CUDA-specific common fields | unit | `./gradlew test --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest` | ✅ | ✅ green |
| 06-03-02 | 03 | 2 | CUDA-02 | T-6-06 | REQUIRED mode fails before tensor-list fallback when native buffer execution is unavailable | unit | `./gradlew test --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements. `CudaFfmBridgeTest` and `PreparedCudaExecutableBufferPolicyTest` already exist; Plan 06-03 adds `CudaAcceleratorBufferBinderTest`.

---

## Manual-Only Verifications

No required manual-only verifications remain. Optional native CUDA execution is capability-gated; on machines without CUDA tooling or hardware the native gate may skip when the portable Java gate is green and the skip reason is recorded.

## Validation Audit 2026-04-30

| Metric | Count |
|--------|-------|
| Gaps found | 1 |
| Resolved | 1 |
| Escalated | 0 |

Resolved gap:
- Updated stale pending task rows to green using the Phase 6 portable gate evidence.
- Resolved the stale W0 file-exists marker for `CudaAcceleratorBufferBinderTest`; the test file exists and is covered by the portable suite.

Portable evidence:
- `./gradlew classes` - green.
- `./gradlew test --tests backend.cuda.bridge.CudaFfmBridgeTest --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests SourceTreeHygieneTest` - green.

Native-gated evidence:
- `./gradlew buildCudaGraphShim cudaTest` - build successful with `buildCudaGraphShim` and `cudaTest` skipped because `command -v nvcc` returned no path in this environment.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or existing test dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all existing references
- [x] No watch-mode flags
- [x] Feedback latency < 180s for portable checks
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-30
