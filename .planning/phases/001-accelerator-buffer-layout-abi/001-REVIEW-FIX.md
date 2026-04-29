---
phase: 001-accelerator-buffer-layout-abi
source_review: 001-REVIEW.md
status: fixed
fixed: 2026-04-29
findings_fixed:
  warnings: 4
  info: 1
---

# Phase 1 Code Review Fix Summary

## Fixed Findings

- `WR-01`: `AcceleratorBufferLayout` now rejects constructor-provided byte lengths that do not match `dataType * logicalElementCount`.
- `WR-02`: `PreparedCudaExecutable` now fails every CUDA `REQUIRE` buffer mode until CUDA has a real buffer execution implementation, even if a future bridge advertises buffer support.
- `WR-03`: `MetalMpsFfmBridge` buffer validation now verifies input and output binding `nodeId` values against the compiled executable node mapping.
- `WR-04`: `AcceleratorBufferRequest` now requires dtype list sizes to match node-id list sizes, matching the strict layout-list ABI contract.
- `IN-01`: Metal layout fallback diagnostics now include `shape=` so runtime messages match `docs/compute-flow.md`.

## Verification

Passed:

```bash
./gradlew test --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.buffer.MetalBufferBindingTest --tests graph.execution.ExecutionStateResidencyTest
./gradlew classes
```

## Scope Notes

The fix did not touch the pre-existing local tuning profile changes under `profiles/platform/.../tuning/abc/*` and did not stage `.planning/tmp/`.
