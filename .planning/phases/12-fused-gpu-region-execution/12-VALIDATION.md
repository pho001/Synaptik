---
phase: 12
slug: fused-gpu-region-execution
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-30
---

# Phase 12 - Validation Strategy

> Per-phase validation contract for fused GPU region execution.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | JUnit 5 via Gradle |
| **Config file** | `build.gradle` |
| **Quick run command** | `./gradlew test --tests backend.accelerator.lowering.* --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest` |
| **Full suite command** | `./gradlew test --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest` |
| **Estimated runtime** | ~60-180 seconds focused; native checks capability-dependent |

## Sampling Rate

- **After every task commit:** Run the task's focused `./gradlew test --tests ...` command.
- **After every plan wave:** Run the quick run command.
- **Before `$gsd-verify-work`:** Run `./gradlew classes` and the full suite command.
- **Max feedback latency:** one focused Gradle invocation per task.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 12-01-01 | 01 | 1 | GPUFUSE-03, GPUFUSE-04 | T-12-01 / T-12-02 | CPU `FUSED` rejects explicitly on GPU path | unit | `./gradlew test --tests backend.accelerator.lowering.*` | W0 | pending |
| 12-01-02 | 01 | 1 | GPUFUSE-03 | T-12-03 | Traceable summary is backend-neutral and not CPU fused | unit | `./gradlew test --tests backend.accelerator.lowering.*` | W0 | pending |
| 12-02-01 | 02 | 2 | GPUFUSE-01 | T-12-04 | Full linear+bias+activation region stays selected | unit/integration | `./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest` | W0 | pending |
| 12-02-02 | 02 | 2 | GPUFUSE-01 | T-12-05 | Required buffer mode fails before hidden CPU fallback | integration | `./gradlew test --tests PreparedExecutionBuildTest` | W0 | pending |
| 12-03-01 | 03 | 3 | GPUFUSE-02 | T-12-06 | Elementwise intermediates remain partition interiors/device-owned | integration | `./gradlew test --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest` | W0 | pending |
| 12-04-01 | 04 | 4 | GPUFUSE-04 | T-12-07 | Reduction-adjacent candidates support or reject visibly | unit/integration | `./gradlew test --tests backend.accelerator.lowering.* --tests CompiledGraphTraceTest` | W0 | pending |
| 12-04-02 | 04 | 4 | GPUFUSE-03 | T-12-08 | CPU fused tests continue to pass | regression | `./gradlew test --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest` | W0 | pending |

## Wave 0 Requirements

Existing infrastructure covers all phase requirements:

- `build.gradle` provides Gradle/JUnit execution.
- Existing accelerator lowering tests exist under `src/test/java/backend/metal/lowering` and `src/test/java/backend/cuda/lowering`.
- Existing prepared execution and trace tests exist in `PreparedExecutionBuildTest` and `CompiledGraphTraceTest`.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Native Metal execution | GPUFUSE-01, GPUFUSE-02 | Requires local Metal native shim availability | Run `./gradlew metalTest` when `synaptik.metal.mps.lib` is configured. |
| Native CUDA execution | GPUFUSE-01, GPUFUSE-02 | Requires local CUDA runtime/toolchain availability | Run `./gradlew buildCudaGraphShim cudaTest` when CUDA is installed. |

## Validation Sign-Off

- [x] All tasks have `<automated>` verify commands.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency bounded by focused Gradle filters.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-04-30

