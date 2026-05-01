---
phase: 16
slug: dtype-and-storage-residency-expansion
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-01
---

# Phase 16 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | JUnit Jupiter via Gradle |
| **Config file** | `build.gradle` |
| **Quick run command** | `./gradlew test --tests graph.execution.RuntimeMemoryBinderTest --tests backend.accelerator.residency.AcceleratorDTypeResidencyPolicyTest --tests GpuCoverageSummaryTest` |
| **Full suite command** | `./gradlew classes && ./gradlew test --tests graph.execution.RuntimeMemoryBinderTest --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest --tests backend.accelerator.residency.AcceleratorDTypeResidencyPolicyTest --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.cuda.buffer.CudaBufferAllocatorTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests SourceTreeHygieneTest` |
| **Estimated runtime** | ~120 seconds focused; native Metal/CUDA execution remains capability-gated |

## Sampling Rate

- **After every task commit:** Run the focused test class for the touched area.
- **After every plan wave:** Run the quick run command.
- **Before `$gsd-verify-work`:** Run the full suite command and `git status --short`.
- **Max feedback latency:** 120 seconds for focused tests.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 16-01-01 | 01 | 1 | GPUSTORAGE-01, GPUSTORAGE-03 | T-16-01, T-16-05 | Runtime slots preserve dtype storage without changing public tensor semantics. | unit | `./gradlew test --tests graph.execution.RuntimeMemoryBinderTest` | W0 | passed |
| 16-02-01 | 02 | 2 | GPUSTORAGE-01, GPUSTORAGE-02 | T-16-02, T-16-03 | Backend dtype residency decisions are role-gated and explicit. | unit | `./gradlew test --tests backend.accelerator.residency.AcceleratorDTypeResidencyPolicyTest --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.cuda.buffer.CudaBufferAllocatorTest` | W0 | passed |
| 16-03-01 | 03 | 3 | GPUSTORAGE-02, GPUSTORAGE-03 | T-16-04, T-16-06 | Trace/report evidence exposes dtype materialization and rejection reasons. | trace/report | `./gradlew test --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest` | W0 | passed |
| 16-04-01 | 04 | 4 | GPUSTORAGE-01, GPUSTORAGE-02, GPUSTORAGE-03 | T-16-07 | Docs, source hygiene, CPU guardrails, and artifact hygiene are verified before closure. | docs/test | `./gradlew classes && ./gradlew test --tests graph.execution.RuntimeMemoryBinderTest --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest --tests backend.accelerator.residency.AcceleratorDTypeResidencyPolicyTest --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.cuda.buffer.CudaBufferAllocatorTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests SourceTreeHygieneTest` | W0 | passed |

## Wave 0 Requirements

Existing infrastructure covers the phase starting point:

- `src/main/java/graph/execution/RuntimeMemoryBinder.java`
- `src/main/java/graph/execution/ExecutionState.java`
- `src/main/java/backend/accelerator/buffer/AcceleratorBufferLayout.java`
- `src/main/java/backend/metal/MetalMpsCapabilities.java`
- `src/main/java/backend/cuda/buffer/CudaBufferAllocator.java`
- `src/main/java/backend/accelerator/lowering/GpuLoweredRegionManifest.java`
- `src/test/java/graph/execution/RuntimeMemoryBinderTest.java`
- `src/test/java/GpuCoverageSummaryTest.java`

## Manual-Only Verifications

Native CUDA execution remains optional and capability-gated outside portable tests. Metal native tests should run where the local shim is available.

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all existing references.
- [x] No watch-mode flags.
- [x] Feedback latency target < 120s.
- [x] `nyquist_compliant: true` set in frontmatter.

## Execution Evidence

| Command | Result |
|---------|--------|
| `./gradlew classes` | Passed |
| `./gradlew test --tests graph.execution.RuntimeMemoryBinderTest --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest --tests backend.accelerator.residency.AcceleratorDTypeResidencyPolicyTest --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.cuda.buffer.CudaBufferAllocatorTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests SourceTreeHygieneTest` | Passed |
| `git status --short` | Source/docs/planning changes visible; profiles/platform/.../tuning/abc/* remained unstaged |

**Approval:** verified
