---
phase: 44-custom-metal-kernel-execution-route
plan: 44-02
subsystem: accelerator-runtime
tags: [metal, native-shim, custom-kernel, relu, buffer-binding]
requires:
  - phase: 44-01
    provides: scoped custom Metal kernel candidate and routing contract
provides:
  - native custom Metal `relu_f32` buffer kernel
  - FFM custom kernel bridge using existing Metal context and MTLBuffer handles
  - capability-gated native parity test and trace smoke update for custom execution path
affects: [metal, native-shim, accelerator-routing, gpu-trace]
tech-stack:
  added: []
  patterns: [separate MPSGraph bridge and custom-kernel bridge contracts]
key-files:
  created:
    - src/main/java/backend/metal/bridge/MetalMpsFfmCustomKernelBridge.java
  modified:
    - src/main/native/apple/synaptik_apple_mps_stub.m
    - src/main/java/backend/metal/prepare/MetalNodePreparer.java
    - src/main/java/backend/metal/exec/PreparedMetalExecutable.java
    - src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java
    - src/test/java/backend/metal/MetalBufferTraceSmokeTest.java
key-decisions:
  - "The FFM custom kernel bridge is separate from the MPSGraph bridge because their capability contracts differ."
  - "Only dense contiguous FLOAT32 bindings execute through the custom RELU route; other runtime layouts use the MPSGraph buffer route."
patterns-established:
  - "Scoped custom kernels can report TRUE_OUTPUT_BUFFER_WRITE without changing the broader MPSGraph copy classification."
requirements-completed: [METALKERNEL-01, METALKERNEL-02]
duration: 30min
completed: 2026-05-02
---

# Phase 44-02: First Scoped Custom Kernel Execution Summary

**Native Metal `relu_f32` now executes through a custom buffer-bound kernel for the scoped dense FLOAT32 route.**

## Performance

- **Duration:** 30 min
- **Started:** 2026-05-02T14:04:00Z
- **Completed:** 2026-05-02T14:34:08Z
- **Tasks:** 5
- **Files modified:** 6

## Accomplishments

- Added `MetalMpsFfmCustomKernelBridge`, a backend-internal custom-kernel bridge that compiles the Phase 44 candidate and executes against existing Metal context/buffer handles.
- Added native Objective-C/Metal `synaptik_apple_mps_custom_relu_f32_buffer` with a simple `FLOAT32` RELU compute pipeline.
- Wired the default `MetalNodePreparer` to pass the FFM custom bridge into `PreparedMetalExecutable`.
- Kept runtime layout safety: selected custom route executes custom only for dense contiguous `FLOAT32` bindings; non-dense runtime bindings continue through MPSGraph buffer execution.
- Added native and trace evidence that the custom route reports `CUSTOM_KERNEL` and `TRUE_OUTPUT_BUFFER_WRITE`.

## Task Commits

1. **Scoped native custom RELU kernel execution** - `e42c95d` (metal)

## Files Created/Modified

- `src/main/java/backend/metal/bridge/MetalMpsFfmCustomKernelBridge.java` - FFM custom-kernel bridge implementation.
- `src/main/native/apple/synaptik_apple_mps_stub.m` - native `relu_f32` Metal compute pipeline and exported buffer symbol.
- `src/main/java/backend/metal/prepare/MetalNodePreparer.java` - default custom bridge wiring.
- `src/main/java/backend/metal/exec/PreparedMetalExecutable.java` - dense binding guard before custom execution.
- `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java` - capability-gated custom kernel parity test.
- `src/test/java/backend/metal/MetalBufferTraceSmokeTest.java` - trace expectation for custom RELU smoke path.

## Decisions Made

Custom kernel execution is not folded into `MetalMpsGraphBridge`; it remains a separate SPI so MPSGraph route reporting and custom kernel capability reporting cannot be conflated.

## Deviations from Plan

The implementation added a runtime dense-binding guard in `PreparedMetalExecutable`. This avoids turning a view/layout mismatch into a CPU fallback when MPSGraph can still handle the already prepared buffer route.

## Issues Encountered

`metalTest` initially failed because the existing RELU smoke test still expected `BUFFER_BINDING` as the Metal execution path. The test was updated to assert the new `CUSTOM_KERNEL` path and true output-buffer write classification for the scoped route.

## Verification

- `./gradlew test --tests backend.metal.exec.MetalExecutionRouterTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.MetalBufferTraceSmokeTest`
- `./gradlew metalTest`
- `git diff --check -- src/main/java/backend/metal src/test/java/backend/metal src/main/native/apple/synaptik_apple_mps_stub.m`

## User Setup Required

None.

## Next Phase Readiness

44-03 can now close trace/report coverage around `CUSTOM_KERNEL`, `TRUE_OUTPUT_BUFFER_WRITE`, rejected alternatives, and route-level coverage summaries.

---
*Phase: 44-custom-metal-kernel-execution-route*
*Completed: 2026-05-02*
