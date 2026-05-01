---
phase: 19
slug: multi-op-gpu-region-execution
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-01
---

# Phase 19 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | JUnit Jupiter via Gradle |
| **Config file** | `build.gradle` |
| **Quick run command** | `./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests PreparedExecutionBuildTest` |
| **Full suite command** | `./gradlew classes && ./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests graph.execution.DeviceLayoutViewPropagationTest --tests SourceTreeHygieneTest` |
| **Estimated runtime** | ~120-240 seconds |

## Sampling Rate

- **After every task commit:** Run the plan-specific focused Gradle command.
- **After every plan wave:** Run the wave verification command listed in the plan.
- **Before `$gsd-verify-work`:** Full suite must be green or capability-skipped with visible native skip evidence.
- **Max feedback latency:** 240 seconds for the focused Phase 19 set.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 19-01-01 | 01 | 1 | GPUMULTI-01/03 | T-19-01/T-19-02 | Unsupported internals reject before execution | unit | `./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest` | yes | pending |
| 19-02-01 | 02 | 2 | GPUMULTI-02 | T-19-03/T-19-04 | Supported interiors avoid CPU materialization | unit/integration | `./gradlew test --tests PreparedExecutionBuildTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests graph.execution.DeviceLayoutViewPropagationTest` | yes | pending |
| 19-03-01 | 03 | 3 | GPUMULTI-01/03 | T-19-05/T-19-06 | Metal/CUDA execution stays backend-neutral at planning contract | integration | `./gradlew test --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest` | yes | pending |
| 19-04-01 | 04 | 4 | GPUMULTI-01/02/03 | T-19-07 | Coverage evidence distinguishes native buffer from fallback | report | `./gradlew test --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest` | yes | pending |
| 19-05-01 | 05 | 5 | GPUMULTI-01/02/03 | T-19-08/T-19-09 | Docs/source hygiene preserve boundaries | docs/hygiene | `./gradlew classes && ./gradlew test --tests SourceTreeHygieneTest` | yes | pending |

## Wave 0 Requirements

Existing infrastructure covers all Phase 19 requirements. No new test framework setup is needed.

## Manual-Only Verifications

All Phase 19 behaviors have automated or capability-gated verification.

## Validation Sign-Off

- [x] All tasks have `<automated>` verify commands.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency target is under 240 seconds for focused tests.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** pending
