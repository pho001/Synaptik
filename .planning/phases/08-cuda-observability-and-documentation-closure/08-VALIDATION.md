---
phase: 08
slug: cuda-observability-and-documentation-closure
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-30
---

# Phase 08 - Validation Strategy

Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | JUnit Jupiter / Gradle |
| **Config file** | `build.gradle` |
| **Quick run command** | `./gradlew test --tests BenchmarkSessionTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` |
| **Full suite command** | `./gradlew classes && ./gradlew test --tests BenchmarkSessionTest --tests backend.cuda.bridge.CudaFfmBridgeTest --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest --tests backend.cuda.buffer.CudaBufferAllocatorTest --tests backend.cuda.buffer.CudaDeviceToCpuMaterializerTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests SourceTreeHygieneTest` |
| **Estimated runtime** | ~60-180 seconds |

---

## Sampling Rate

- **After every task commit:** Run the focused test command named in that task.
- **After every plan wave:** Run the Phase 8 portable suite.
- **Before `$gsd-verify-work`:** Full portable suite must be green; optional native CUDA gate must pass or skip with an explicit unavailable reason.
- **Max feedback latency:** 180 seconds for portable checks.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 08-01-01 | 01 | 1 | CUDADOC-01 | T-8-01 | CUDA report summary exposes backend, path, reason, prepared-input, bytes, copy timing, and residency fields | unit | `./gradlew test --tests BenchmarkSessionTest` | ✅ | ⬜ pending |
| 08-01-02 | 01 | 1 | CUDADOC-01 | T-8-02 | CUDA execution publishes trace attrs without changing public Tensor API | unit | `./gradlew test --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests BenchmarkSessionTest` | ✅ | ⬜ pending |
| 08-02-01 | 02 | 1 | CUDA-06 | T-8-03 | CUDA unavailable, unsupported dtype/layout, native failure, and required-unavailable paths use stable reason codes | unit | `./gradlew test --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` | ✅ | ⬜ pending |
| 08-02-02 | 02 | 1 | CUDA-06 | T-8-04 | REQUIRED mode fails before tensor-array or CPU fallback can hide required buffer unavailability | unit | `./gradlew test --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` | ✅ | ⬜ pending |
| 08-03-01 | 03 | 2 | CUDADOC-02 | T-8-05 | CUDA docs explain build, probing, fallback interpretation, troubleshooting, and ABI symmetry without broad-coverage overclaim | docs | `rg -n "CUDA trace and benchmark reports|CUDA fallback interpretation|SYNAPTIK_CUDA_GRAPH_LIB|./gradlew buildCudaGraphShim cudaTest|Native CUDA tests skip" docs` | ✅ | ⬜ pending |
| 08-04-01 | 04 | 3 | CUDADOC-03 | T-8-06 | Hygiene checks prevent native output, scratch, benchmark/profile artifact pollution | unit/hygiene | `./gradlew test --tests SourceTreeHygieneTest` | ✅ | ⬜ pending |
| 08-04-02 | 04 | 3 | CUDA-06/CUDADOC-01/CUDADOC-02/CUDADOC-03 | T-8-07 | Final gate proves portable CUDA observability closure and records optional native pass/skip status | integration | `./gradlew classes && ./gradlew test --tests BenchmarkSessionTest --tests backend.cuda.bridge.CudaFfmBridgeTest --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest --tests backend.cuda.buffer.CudaBufferAllocatorTest --tests backend.cuda.buffer.CudaDeviceToCpuMaterializerTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests SourceTreeHygieneTest` | ✅ | ⬜ pending |

*Status: ⬜ pending - ✅ green - ❌ red - ⚠ flaky*

---

## Wave 0 Requirements

Existing Gradle/JUnit infrastructure covers all Phase 8 requirements.

---

## Manual-Only Verifications

Optional native CUDA checks are capability-gated. On machines without `nvcc` or CUDA hardware, `./gradlew buildCudaGraphShim cudaTest` may skip and must record the unavailable reason.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify commands.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency target is under 180 seconds for portable checks.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-04-30
