---
phase: 001
slug: accelerator-buffer-layout-abi
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-29
updated: 2026-04-30
---

# Phase 001 - Validation Strategy

> Per-phase validation contract for accelerator buffer layout ABI work.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | JUnit 5 via Gradle |
| **Config file** | `build.gradle` |
| **Quick run command** | `./gradlew test --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest --tests graph.execution.ExecutionStateResidencyTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` |
| **Full suite command** | `./gradlew classes && ./gradlew test --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest --tests graph.execution.ExecutionStateResidencyTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.buffer.MetalBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` |
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
| 001-01-01 | 001-01 | 1 | ABI-01, ABI-03 | T-001-01-01, T-001-01-03 | Shared layout descriptor validates metadata, uses checked byte length, and classifies dense/view/strided/broadcast/unsupported layouts | unit | `./gradlew test --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest` | yes | green |
| 001-01-02 | 001-01 | 1 | ABI-02, ABI-04 | T-001-01-04 | Stable reason taxonomy compiles and remains available to accelerator decisions | compile/docs | `./gradlew classes` | yes | green |
| 001-01-03 | 001-01 | 1 | ABI-01, ABI-02 | T-001-01-02 | DeviceBufferBinding and MetalBufferBinding expose shared layout/access/identity without common native handle leakage | unit | `./gradlew test --tests backend.metal.buffer.MetalBufferBindingTest --tests graph.execution.ExecutionStateResidencyTest` | yes | green |
| 001-02-01 | 001-02 | 2 | ABI-01, ABI-03, ABI-04 | T-001-02-01 | Metal buffer requests and preflight decisions compare exact layout facts and report native ABI unavailability | unit | `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest` | yes | green |
| 001-02-02 | 001-02 | 2 | ABI-01, ABI-03 | T-001-02-02 | Metal allocator/materializer route through binding layouts and reject unsupported allocation before native calls | unit | `./gradlew test --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.buffer.MetalBufferBindingTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest` | yes | green |
| 001-02-03 | 001-02 | 2 | ABI-02, ABI-04 | T-001-02-03, T-001-02-04 | CUDA required-buffer mode remains explicit unavailable; CUDA buffer SPI stays accepted-risk internal/local | unit | `./gradlew test --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` | yes | green |
| 001-03-01 | 001-03 | 3 | ABI-01, ABI-02, ABI-03 | T-001-01-01, T-001-01-02 | Shared ABI and binding unit coverage asserts classifier behavior, Metal binding diagnostics, and allocator byte lengths | unit | `./gradlew test --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest --tests backend.metal.buffer.MetalBufferBindingTest --tests backend.metal.buffer.MetalBufferAllocatorTest` | yes | green |
| 001-03-02 | 001-03 | 3 | ABI-03, ABI-04 | T-001-03-01, T-001-03-02 | Metal/CUDA decision regressions assert stable reason codes, layout details, and no unsafe output promotion | regression | `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` | yes | green |
| 001-03-03 | 001-03 | 3 | ABI-01, ABI-02, ABI-03, ABI-04 | T-001-03-03, T-001-03-04 | Trace/report reason docs and phase gate preserve stable diagnostics without exposing native handle values | integration/docs | `./gradlew classes && ./gradlew test --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest --tests graph.execution.ExecutionStateResidencyTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.buffer.MetalBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` | yes | green |

*Status values: pending, green, red, flaky.*

---

## Wave 0 Requirements

- [x] `src/test/java/backend/accelerator/buffer/AcceleratorBufferLayoutClassifierTest.java` covers dense contiguous, zero-offset view, non-zero-offset view, permuted/strided view, broadcast/zero-stride view, unsupported layout, defensive copies, and byte lengths.
- [x] Fake `DeviceBufferBinding` implementations in existing tests implement shared layout/access/native identity methods.

---

## Manual-Only Verifications

All phase behaviors have automated verification.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing test references.
- [x] No watch-mode flags.
- [x] Feedback latency target is under 90 seconds for targeted test commands.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** verified 2026-04-30

## Evidence

Validation status is based on the completed Phase 1 summaries, `001-VERIFICATION.md`, `001-SECURITY.md`, and a fresh validation rerun:

- `rg -n "dense|zeroOffset|nonZeroOffset|permuted|broadcast|unsupported|defensive|byteLength|logicalByteLength|DeviceBufferBinding|nativeHandleIdentity|ExecutionStateResidency|REQUIRED_BUFFER_EXECUTION_UNAVAILABLE|NATIVE_BUFFER_ABI_UNAVAILABLE|OUTPUT_LAYOUT_UNSUPPORTED|INPUT_LAYOUT_UNSUPPORTED|layoutClass=" src/test/java/backend/accelerator/buffer/AcceleratorBufferLayoutClassifierTest.java src/test/java/backend/metal/buffer/MetalBufferBindingTest.java src/test/java/backend/metal/buffer/MetalBufferAllocatorTest.java src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java src/test/java/graph/execution/ExecutionStateResidencyTest.java` - PASS
- `./gradlew classes` - PASS
- `./gradlew test --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest --tests graph.execution.ExecutionStateResidencyTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.buffer.MetalBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` - PASS

No generated validation test files were needed; Phase 1 requirements already had automated coverage after execution.

## Validation Audit 2026-04-30

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
