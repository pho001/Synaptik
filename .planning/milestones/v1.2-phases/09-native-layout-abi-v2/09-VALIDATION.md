---
phase: 09
slug: native-layout-abi-v2
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-30
updated: 2026-05-01
---

# Phase 09 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | JUnit 5 via Gradle |
| Config file | `build.gradle` |
| Quick run command | `./gradlew classes` |
| Full suite command | `./gradlew test --tests 'backend.accelerator.buffer.*' --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.cuda.bridge.CudaFfmBridgeTest --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` |
| Native Metal gate | `./gradlew metalTest` |
| Native CUDA gate | `./gradlew buildCudaGraphShim cudaTest` |
| Estimated runtime | 60-180 seconds for focused portable tests |

## Sampling Rate

- After every task commit: run the task-specific focused Gradle filter.
- After every plan wave: run the plan verification command listed in the PLAN.md.
- Before phase verification: run `./gradlew classes` and all focused accelerator buffer, Metal bridge, CUDA bridge, and prepared-executable tests touched by the phase.
- Max feedback latency: one plan wave.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 09-01-01 | 01 | 1 | GPULAYOUT-01 | T-09-01 | descriptor rejects invalid or overflowing layout metadata | unit | `./gradlew test --tests backend.accelerator.buffer.AcceleratorLayoutAbiV2DescriptorTest` | yes | green |
| 09-01-02 | 01 | 1 | GPULAYOUT-01 | T-09-01 | descriptor exposes physical span separately from logical bytes | unit | `./gradlew test --tests backend.accelerator.buffer.AcceleratorLayoutAbiV2DescriptorTest` | yes | green |
| 09-02-01 | 02 | 2 | GPULAYOUT-02 | T-09-02 | missing layout ABI v2 symbols do not break dense bridge availability | unit | `./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.cuda.bridge.CudaFfmBridgeTest` | yes | green |
| 09-02-02 | 02 | 2 | GPULAYOUT-02 | T-09-02 | native stubs export versioned optional layout ABI symbols | compile/optional native | `./gradlew classes`; `./gradlew metalTest`; `./gradlew buildCudaGraphShim cudaTest` | yes | green |
| 09-03-01 | 03 | 3 | GPULAYOUT-03 | T-09-03 | AUTO fallback and REQUIRE failure use ABI-specific reason codes | unit | `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest` | yes | green |
| 09-03-02 | 03 | 3 | GPULAYOUT-03 | T-09-03 | docs explain supported, unavailable, mismatch, and unsupported outcomes | grep | `rg -n "layout ABI v2|NATIVE_LAYOUT_ABI_UNAVAILABLE|NATIVE_LAYOUT_ABI_VERSION_MISMATCH" docs src/test/java` | yes | green |

Status: green = targeted command passed or was covered by the consolidated successful validation gate.

## Requirement Coverage

| Requirement | Coverage | Status |
|-------------|----------|--------|
| GPULAYOUT-01 | `AcceleratorLayoutAbiV2Descriptor` and accelerator buffer tests cover rank, shape, strides, storage offset, logical element/byte counts, physical byte span, access mode, backend id, dtype, layout class, native handle identity, defensive copies, and invalid metadata rejection. | covered |
| GPULAYOUT-02 | Metal and CUDA bridge capability tests cover optional layout ABI v2 symbols, version reporting, and preservation of dense bridge availability when layout ABI v2 symbols are unavailable; native Metal gate passed and native CUDA gate is capability-gated locally. | covered |
| GPULAYOUT-03 | Shared `NATIVE_LAYOUT_*` reason codes, CUDA binder tests, prepared executable buffer policy tests, and docs cover stable AUTO fallback and REQUIRED-mode diagnostics. | covered |

## Wave 0 Requirements

Existing JUnit/Gradle infrastructure covers all phase requirements.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CUDA native layout ABI symbol load on a CUDA machine | GPULAYOUT-02 | depends on local CUDA toolkit and driver | run `./gradlew buildCudaGraphShim cudaTest` |

Native Metal execution is not manual-only on this host; `./gradlew metalTest` completed successfully during validation.

Native CUDA execution remains manual-only for hardware-backed evidence on this host because `buildCudaGraphShim` and `cudaTest` were capability-skipped locally.

## Validation Audit 2026-05-01

| Metric | Count |
|--------|-------|
| Requirements audited | 3 |
| Task verification rows | 6 |
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
| Generated test files | 0 |

Verification commands:

- `./gradlew test --tests 'backend.accelerator.buffer.*' --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.cuda.bridge.CudaFfmBridgeTest --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` - PASS
- `rg -n "layout ABI v2|NATIVE_LAYOUT_ABI_UNAVAILABLE|NATIVE_LAYOUT_ABI_VERSION_MISMATCH|NATIVE_LAYOUT_PHYSICAL_SPAN_OVERFLOW|record AcceleratorLayoutAbiV2Descriptor|REQUIRED_VERSION = 2|layoutAbiV2Supported|synaptik_apple_mps_layout_abi_version|synaptik_cuda_graph_layout_abi_version" docs src/main/java src/main/native src/test/java` - PASS
- `./gradlew metalTest` - PASS
- `./gradlew buildCudaGraphShim cudaTest` - PASS with `buildCudaGraphShim SKIPPED` and `cudaTest SKIPPED` on this host

The first sandboxed CUDA gate attempt failed before Gradle execution because the wrapper could not access the `~/.gradle` lock file. The escalated Gradle run completed successfully.

## Validation Sign-Off

- [x] All tasks have automated verify commands or explicit optional-native notes.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing test infrastructure.
- [x] No watch-mode flags.
- [x] Feedback latency below one plan wave.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** verified 2026-05-01
