---
phase: 45-metal-output-buffer-write-and-copy-closure
plan: 45-03-copy-strategy-reporting-and-gates
status: completed
completed: 2026-05-02
requirements:
  - METALCOPY-03
---

# 45-03 Summary: Copy Strategy Reporting And Gates

## Completed

- Added `metalOutputBufferWriteStatus` trace metadata beside `metalNativeCopyStrategy` and `metalOutputBufferWriteProven`.
- Added benchmark accelerator summaries for `outputBufferWriteStatuses` in text and JSON reports.
- Updated focused trace/report tests so MPSGraph paths report `COPY_REQUIRED` instead of looking like zero-copy, while existing native-copy strategy gates still distinguish `MPSGRAPH_RESULT_COPY` from `TRUE_OUTPUT_BUFFER_WRITE`.
- Updated Metal docs and troubleshooting guidance with the probe-vs-proof distinction.

## Scope Truth

- `metalOutputBufferWriteProbeSupported=true` is only proof harness availability.
- `metalOutputBufferWriteStatus=PROVEN_TRUE_WRITE` is reserved for accepted direct-write routes.
- MPSGraph remains `COPY_REQUIRED` until a later proof-backed change removes the explicit result copy.

## Verification

- `./gradlew test --tests backend.metal.bridge.MetalMpsBridgeExecutionStatsTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests BenchmarkSessionTest` - passed
