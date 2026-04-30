---
phase: 08
slug: cuda-observability-and-documentation-closure
status: verified
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
| 08-01-01 | 01 | 1 | CUDADOC-01 | T-8-01 | CUDA report summary exposes backend, path, reason, prepared-input, bytes, copy timing, and residency fields | unit | `./gradlew test --tests BenchmarkSessionTest` | ✅ | ✅ green |
| 08-01-02 | 01 | 1 | CUDADOC-01 | T-8-02 | CUDA execution publishes trace attrs without changing public Tensor API | unit | `./gradlew test --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests BenchmarkSessionTest` | ✅ | ✅ green |
| 08-02-01 | 02 | 1 | CUDA-06 | T-8-03 | CUDA unavailable, unsupported dtype/layout, native failure, and required-unavailable paths use stable reason codes | unit | `./gradlew test --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` | ✅ | ✅ green |
| 08-02-02 | 02 | 1 | CUDA-06 | T-8-04 | REQUIRED mode fails before tensor-array or CPU fallback can hide required buffer unavailability | unit | `./gradlew test --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` | ✅ | ✅ green |
| 08-03-01 | 03 | 2 | CUDADOC-02 | T-8-05 | CUDA docs explain build, probing, fallback interpretation, troubleshooting, and ABI symmetry without broad-coverage overclaim | docs | `rg -n "CUDA trace and benchmark reports|CUDA fallback interpretation|SYNAPTIK_CUDA_GRAPH_LIB|./gradlew buildCudaGraphShim cudaTest|Native CUDA tests skip" docs` | ✅ | ✅ green |
| 08-04-01 | 04 | 3 | CUDADOC-03 | T-8-06 | Hygiene checks prevent native output, scratch, benchmark/profile artifact pollution | unit/hygiene | `./gradlew test --tests SourceTreeHygieneTest` | ✅ | ✅ green |
| 08-04-02 | 04 | 3 | CUDA-06/CUDADOC-01/CUDADOC-02/CUDADOC-03 | T-8-07 | Final gate proves portable CUDA observability closure and records optional native pass/skip status | integration | `./gradlew classes && ./gradlew test --tests BenchmarkSessionTest --tests backend.cuda.bridge.CudaFfmBridgeTest --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest --tests backend.cuda.buffer.CudaBufferAllocatorTest --tests backend.cuda.buffer.CudaDeviceToCpuMaterializerTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests SourceTreeHygieneTest` | ✅ | ✅ green |

*Status: ⬜ pending - ✅ green - ❌ red - ⚠ flaky*

---

## Wave 0 Requirements

Existing Gradle/JUnit infrastructure covers all Phase 8 requirements.

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
- Updated stale pending task rows to green using the Phase 8 portable gate, docs/source grep, and native-gated evidence.
- Confirmed each Phase 8 requirement has automated or grep-based verification: `CUDA-06`, `CUDADOC-01`, `CUDADOC-02`, and `CUDADOC-03`.

Portable evidence:
- `./gradlew classes` - green.
- `./gradlew test --tests BenchmarkSessionTest --tests backend.cuda.bridge.CudaFfmBridgeTest --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest --tests backend.cuda.buffer.CudaBufferAllocatorTest --tests backend.cuda.buffer.CudaDeviceToCpuMaterializerTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests SourceTreeHygieneTest` - green.
- `rg -n "GPU_CUDA|cudaExecutionPath|cudaFallbackReason|acceleratorInputBytes|CUDA trace and benchmark reports|Native CUDA tests skip|SYNAPTIK_CUDA_GRAPH_LIB|CUDA fallback interpretation|NATIVE_BUFFER_ABI_UNAVAILABLE|REQUIRED_BUFFER_EXECUTION_UNAVAILABLE|NATIVE_BUFFER_EXECUTION_FAILED|build/native/cuda|do not stage local profile tuning changes accidentally" src/main/java src/test/java docs .gitignore` - green.

Native-gated evidence:
- `./gradlew buildCudaGraphShim cudaTest` - build successful with `buildCudaGraphShim` and `cudaTest` skipped because `command -v nvcc` returned no path in this environment.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify commands.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency target is under 180 seconds for portable checks.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-04-30
