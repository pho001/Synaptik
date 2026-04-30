---
status: testing
phase: 10-gpu-layout-transform-and-view-path
source:
  - 10-01-SUMMARY.md
  - 10-02-SUMMARY.md
  - 10-03-SUMMARY.md
  - 10-04-SUMMARY.md
started: 2026-04-30T13:30:38Z
updated: 2026-04-30T15:35:21Z
---

## Current Test
<!-- OVERWRITE each test - shows where we are -->

number: 1
name: Shared Layout Transform Decisions
expected: |
  Running the planner contract tests shows metadata-only views, dense GPU materialization, missing source binding, backend mismatch, and unsupported transform paths as explicit decisions with stable reason codes such as `GPU_LAYOUT_VIEW_BINDING_AVAILABLE` and `GPU_LAYOUT_DENSE_MATERIALIZATION_AVAILABLE`.
awaiting: user response

## Tests

### 1. Shared Layout Transform Decisions
expected: Running the planner contract tests shows metadata-only views, dense GPU materialization, missing source binding, backend mismatch, and unsupported transform paths as explicit decisions with stable reason codes such as `GPU_LAYOUT_VIEW_BINDING_AVAILABLE` and `GPU_LAYOUT_DENSE_MATERIALIZATION_AVAILABLE`.
result: [pending]

### 2. Metadata-Only Device View Propagation
expected: Running the device layout propagation tests shows compatible Metal/CUDA metadata-only views can reuse existing device bindings before CPU materialization, and rejected REQUIRED-mode propagation fails before hidden CPU fallback.
result: [pending]

### 3. Dense Materialization Capability Gate
expected: Running the Metal/CUDA buffer policy tests shows `contiguous()` and non-contiguous-source `reshape` classify as dense GPU materialization candidates, while execution only uses dense materialization when backend capability and a run-scoped materializer service are available.
result: [pending]

### 4. Trace And Documentation Closure
expected: Running the Phase 10 portable verification gate succeeds, trace attributes include `acceleratorBufferReasonCode`, `storageResidency`, and `gpuLayoutTransformKind`, docs describe the GPU layout transform and view path, CUDA native checks capability-skip successfully, and the native Metal backward parity gate passes after the host-shared metadata-only view fix.
result: [pending]

## Summary

total: 4
passed: 0
issues: 0
pending: 4
skipped: 0
blocked: 0

## Gaps

- Fixed during UAT: native Metal backward parity was failing because metadata-only layout views over host-shared Metal inputs were marked `DEVICE_OWNED`, forcing prepared-input CPU materialization of an alias view back into overlapping CPU storage. `DeviceLayoutViewPropagator` now preserves `HOST_SHARED_DEVICE_BUFFER` residency when the source CPU representation is current, and the focused native Metal backward parity tests pass.
