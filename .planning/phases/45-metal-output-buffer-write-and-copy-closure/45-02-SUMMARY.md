---
phase: 45-metal-output-buffer-write-and-copy-closure
plan: 45-02-lower-copy-or-alias-strategy
status: completed
completed: 2026-05-02
requirements:
  - METALCOPY-02
---

# 45-02 Summary: Lower-Copy Or Alias Strategy

## Completed

- Kept MPSGraph on the conservative result-copy strategy because no normal execution path has been promoted from the no-copy probe into a scoped true-write contract.
- Made output-buffer write status explicit through `MetalMpsBridgeExecutionStats.outputBufferWriteStatus()`.
- Preserved the lower-copy route distinction already introduced by Phase 44: scoped custom RELU reports `TRUE_OUTPUT_BUFFER_WRITE` / `PROVEN_TRUE_WRITE`; MPSGraph reports `MPSGRAPH_RESULT_COPY` / `COPY_REQUIRED`.
- Kept tensor-array and CPU fallback behavior outside the direct-write claim path.

## Scope Truth

- This plan does not remove the MPSGraph `readBytes` copy.
- The lower-copy strategy is route-specific: use proven custom direct-write routes where available; otherwise keep MPSGraph copy explicit and measured.
- Public `Tensor` semantics and graph-output materialization are unchanged.

## Verification

- `./gradlew test --tests backend.metal.bridge.MetalMpsBridgeExecutionStatsTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests BenchmarkSessionTest` - passed
