---
phase: 11
slug: gpu-lowering-coverage-matrix
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-30
---

# Phase 11 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | JUnit 5 via Gradle |
| Config file | `build.gradle` |
| Quick run command | `./gradlew classes` |
| Full suite command | `./gradlew test --tests backend.accelerator.lowering.* --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest` |
| Estimated runtime | 90-240 seconds for focused portable tests |

## Sampling Rate

- After every task commit: run the task-specific focused Gradle filter.
- After every plan wave: run the plan verification command listed in the PLAN.md.
- Before phase verification: run `./gradlew classes` and all focused accelerator lowering, Metal, CUDA, prepared-execution, and trace tests touched by the phase.
- Max feedback latency: one plan wave.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 11-01-01 | 01 | 1 | GPULOWER-01 | T-11-01 | coverage matrix rows classify supported/fallback/unsupported with stable reasons | unit | `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest` | yes | covered |
| 11-01-02 | 01 | 1 | GPULOWER-01 | T-11-02 | docs matrix cannot drift from source-level shared matrix unnoticed | grep/unit | `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest` | yes | covered |
| 11-02-01 | 02 | 2 | GPULOWER-01, GPULOWER-03 | T-11-03 | Metal and CUDA planner adapters expose common stable unsupported reasons | unit | `./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest` | yes | covered |
| 11-02-02 | 02 | 2 | GPULOWER-02, GPULOWER-03 | T-11-04 | CUDA keeps direct non-dense compute conservative and visible | unit | `./gradlew test --tests backend.cuda.lowering.CudaRegionLowererTest --tests backend.cuda.exec.CudaLayoutTransformDeviceFlowTest` | yes | covered |
| 11-03-01 | 03 | 3 | GPULOWER-02 | T-11-05 | softmax-ish lowering expansion uses supported DAG primitives only | unit | `./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest` | yes | covered |
| 11-03-02 | 03 | 3 | GPULOWER-02, GPULOWER-03 | T-11-06 | selected and rejected Metal/CUDA candidates are visible in prepare traces | integration | `./gradlew test --tests PreparedExecutionBuildTest` | yes | covered |
| 11-04-01 | 04 | 4 | GPULOWER-01, GPULOWER-02, GPULOWER-03 | T-11-07 | trace/docs closure exposes lowering reason and support scope | integration/grep | `./gradlew test --tests CompiledGraphTraceTest` | yes | covered |
| 11-04-02 | 04 | 4 | GPULOWER-03 | T-11-08 | local profile artifacts remain unstaged and native checks remain capability-gated | compile/optional native | `./gradlew classes` | yes | covered |

## Wave 0 Requirements

Existing JUnit/Gradle infrastructure covers the phase. New tests are created inside the relevant plan waves.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Native Metal lowering parity on local Apple runtime | GPULOWER-02 | depends on local Metal shim and hardware | run `./gradlew metalTest` after the focused portable tests |
| Native CUDA lowering parity on CUDA host | GPULOWER-02 | depends on CUDA toolkit/driver availability | run `./gradlew buildCudaGraphShim cudaTest` on a CUDA-capable host |

## Optional Native Evidence

| Behavior | Command | Result |
|----------|---------|--------|
| Native Metal lowering parity on local Apple runtime | `./gradlew metalTest` | PASS, recorded in `11-04-SUMMARY.md` |
| Native CUDA lowering parity on local host | `./gradlew buildCudaGraphShim cudaTest` | BUILD SUCCESSFUL with CUDA native tasks skipped because local CUDA native capability/toolchain tasks were not active, recorded in `11-04-SUMMARY.md` |

## Validation Audit 2026-04-30

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

Verification commands:

- `./gradlew classes` - PASS
- `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest` - PASS
- `./gradlew test --tests backend.metal.MetalLayoutAwareDeviceFlowTest --tests backend.cuda.exec.CudaLayoutTransformDeviceFlowTest` - PASS

## Validation Sign-Off

- [x] All tasks have automated verify commands or explicit optional-native notes.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing test infrastructure.
- [x] No watch-mode flags.
- [x] Feedback latency below one plan wave.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** verified 2026-04-30
