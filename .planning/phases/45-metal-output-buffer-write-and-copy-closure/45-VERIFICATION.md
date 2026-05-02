---
phase: 45-metal-output-buffer-write-and-copy-closure
status: passed
verified: 2026-05-02
requirements:
  - METALCOPY-01
  - METALCOPY-02
  - METALCOPY-03
gaps: 0
---

# Phase 45 Verification: Metal Output Buffer Write And Copy Closure

## Verdict

Passed. Phase 45 adds a real MPSGraph no-copy proof seam while keeping production MPSGraph execution conservatively classified as `MPSGRAPH_RESULT_COPY` / `COPY_REQUIRED`. The scoped custom RELU route remains the only accepted `TRUE_OUTPUT_BUFFER_WRITE` / `PROVEN_TRUE_WRITE` route.

## Requirement Evidence

| Requirement | Status | Evidence |
|---|---|---|
| `METALCOPY-01` | Passed | Native `synaptik_apple_mps_probe_output_buffer_write_f32_buffers(...)` executes MPSGraph buffer bindings without the final `readBytes` copy. `MetalMpsFfmBridgeTest.explicitShimOutputBufferWriteProbeRunsWithoutResultCopy` initializes a sentinel output buffer and inspects it after the no-copy probe. Normal `executeBuffers(...)` still reports `MPSGRAPH_RESULT_COPY`. |
| `METALCOPY-02` | Passed | Copy-elision remains route-proof scoped: custom RELU reports `TRUE_OUTPUT_BUFFER_WRITE` / `PROVEN_TRUE_WRITE`; MPSGraph reports `MPSGRAPH_RESULT_COPY` / `COPY_REQUIRED`; tensor-array/CPU fallback stay outside direct-write claims. |
| `METALCOPY-03` | Passed | `PreparedExecution` now emits `metalOutputBufferWriteProbeSupported` and `metalOutputBufferWriteStatus`; `AcceleratorTraceSummary`, text reports, and JSON reports expose `outputBufferWriteStatuses`; focused tests assert MPSGraph `COPY_REQUIRED` reporting. |

## Scope Truth

- The MPSGraph production path still performs the explicit native result copy.
- `metalOutputBufferWriteProbeSupported=true` is probe availability, not proof.
- `metalOutputBufferWriteStatus=PROVEN_TRUE_WRITE` is reserved for accepted direct-write routes.
- No public `Tensor` API or graph-output publication semantics changed.

## Automated Checks

- `./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest` - passed
- `./gradlew metalTest` - passed
- `./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest` - passed
- `./gradlew test --tests backend.metal.bridge.MetalMpsBridgeExecutionStatsTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests BenchmarkSessionTest` - passed
- `./gradlew classes` - passed
- `git diff --check` - passed

## Source Hygiene

Local benchmark/profile artifacts under `profiles/platform/mac_os_x-aarch64-oracle_corporation-16c/tuning/` remain intentionally uncommitted.
