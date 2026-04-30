---
phase: 10
slug: gpu-layout-transform-and-view-path
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-30
---

# Phase 10 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | JUnit Jupiter 5.11.2 via Gradle |
| Config file | `build.gradle` |
| Quick run command | `./gradlew test --tests backend.accelerator.buffer.AcceleratorLayoutTransformPlannerTest --tests graph.execution.DeviceLayoutViewPropagationTest` |
| Full suite command | `./gradlew test --tests backend.accelerator.buffer.AcceleratorLayoutTransformPlannerTest --tests graph.execution.DeviceLayoutViewPropagationTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests backend.metal.MetalLayoutAwareDeviceFlowTest --tests backend.cuda.exec.CudaLayoutTransformDeviceFlowTest` |
| Estimated runtime | ~30-90 seconds for portable slice; native Metal/CUDA gates depend on local tooling |

## Sampling Rate

- After every task commit: run the plan's focused `./gradlew test --tests ...` command.
- After every plan wave: run the accumulated focused portable slice.
- Before phase verification: run the full portable slice plus `./gradlew classes`; run `./gradlew metalTest` and `./gradlew buildCudaGraphShim cudaTest` when local capability gates allow.
- Max feedback latency: 90 seconds for portable gates.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 10-01-01 | 01 | 1 | GPUVIEW-01/GPUVIEW-03 | T-10-01 | Layout decisions cannot hide fallback | unit | `./gradlew test --tests backend.accelerator.buffer.AcceleratorLayoutTransformPlannerTest` | W0 | pending |
| 10-01-02 | 01 | 1 | GPUVIEW-01/GPUVIEW-03 | T-10-01 | Shared reason codes distinguish metadata view vs unsupported transform | unit | `./gradlew test --tests backend.accelerator.buffer.AcceleratorLayoutTransformPlannerTest` | W0 | pending |
| 10-01-03 | 01 | 1 | GPUVIEW-03 | - | Docs define backend-neutral contract | grep | `rg -n "GPU Layout Transform Contract|metadata-only view|dense GPU materialization" docs/native-bridges-and-blas.md` | W0 | pending |
| 10-02-01 | 02 | 2 | GPUVIEW-01/GPUVIEW-02 | T-10-02 | Device view propagation runs before CPU materialization | unit | `./gradlew test --tests graph.execution.DeviceLayoutViewPropagationTest` | W0 | pending |
| 10-02-02 | 02 | 2 | GPUVIEW-03 | T-10-03 | Backend alias bindings do not double-own native resources | unit | `./gradlew test --tests backend.metal.buffer.MetalBufferBindingTest --tests backend.cuda.buffer.CudaBufferBindingTest` | W0 | pending |
| 10-02-03 | 02 | 2 | GPUVIEW-02 | T-10-02 | Unsupported propagation falls back visibly | unit | `./gradlew test --tests graph.execution.DeviceLayoutViewPropagationTest --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest` | W0 | pending |
| 10-03-01 | 03 | 3 | GPUVIEW-01/GPUVIEW-02 | T-10-04 | `contiguous()` can materialize dense device output without Java array round trip when supported | unit/integration | `./gradlew test --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest` | W0 | pending |
| 10-03-02 | 03 | 3 | GPUVIEW-02/GPUVIEW-03 | T-10-05 | Optional native symbols are capability-gated | compile/native | `./gradlew classes` | W0 | pending |
| 10-04-01 | 04 | 4 | GPUVIEW-01/GPUVIEW-02 | T-10-06 | Layout-heavy forward flow avoids intermediate CPU materialization | integration | `./gradlew test --tests backend.metal.MetalLayoutAwareDeviceFlowTest --tests backend.cuda.exec.CudaLayoutTransformDeviceFlowTest` | W0 | pending |
| 10-04-02 | 04 | 4 | GPUVIEW-02 | T-10-06 | Gradient publication boundary has CPU parity | integration | `./gradlew test --tests backend.metal.MetalLayoutAwareDeviceFlowTest --tests backend.cuda.exec.CudaLayoutTransformDeviceFlowTest` | W0 | pending |
| 10-04-03 | 04 | 4 | GPUVIEW-03 | - | Docs and trace fields expose decisions | grep/test | `rg -n "gpuLayoutTransform|GPU layout transform|device view propagation" src/main/java src/test/java docs` | W0 | pending |

## Wave 0 Requirements

Existing Gradle/JUnit infrastructure covers all phase requirements.

## Manual-Only Verifications

All phase behaviors have automated verification. Native Metal/CUDA execution is capability-gated and may skip on hosts without the required native tooling.

## Validation Sign-Off

- [x] All tasks have automated verify commands.
- [x] Sampling continuity has no three-task gap.
- [x] Wave 0 covers all missing test references.
- [x] No watch-mode flags.
- [x] Feedback latency target is under 90 seconds for portable gates.
- [x] `nyquist_compliant: true` set in frontmatter.

Approval: approved 2026-04-30
