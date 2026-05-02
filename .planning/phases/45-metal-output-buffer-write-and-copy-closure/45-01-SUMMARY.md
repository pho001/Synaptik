---
phase: 45-metal-output-buffer-write-and-copy-closure
plan: 45-01-output-buffer-write-proof-harness
status: completed
completed: 2026-05-02
requirements:
  - METALCOPY-01
---

# 45-01 Summary: Output Buffer Write Proof Harness

## Completed

- Added native `synaptik_apple_mps_probe_output_buffer_write_f32_buffers(...)`, an internal MPSGraph buffer execution path that passes caller output buffers but skips the final `MPSNDArray.readBytes(...)` result copy.
- Refactored the native MPSGraph buffer path through a shared helper so production execution still copies returned MPSGraph result storage and reports `MPSGRAPH_RESULT_COPY`.
- Added `MetalMpsFfmBridge.probeOutputBufferWriteWithoutResultCopy(...)` and `supportsOutputBufferWriteProbe()` for Phase 45 proof tests.
- Added sentinel coverage in `MetalMpsFfmBridgeTest`: the probe runs without native result-copy timing, reports `UNKNOWN_OR_UNPROVEN`, and lets the test inspect whether caller output bytes changed before the explicit copy.
- Added trace/docs visibility for `metalOutputBufferWriteProbeSupported` while keeping `metalOutputBufferWriteProven=false` for conservative MPSGraph execution.

## Scope Truth

- This is a proof harness, not a zero-copy promotion.
- Normal MPSGraph `executeBuffers(...)` still reports `MetalNativeCopyStrategy.MPSGRAPH_RESULT_COPY`.
- `TRUE_OUTPUT_BUFFER_WRITE` remains scoped to the Phase 44 custom RELU route unless a later Phase 45 plan promotes a route-specific no-copy result.

## Verification

- `./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest` - passed
- `./gradlew metalTest` - passed
- `./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest` - passed
