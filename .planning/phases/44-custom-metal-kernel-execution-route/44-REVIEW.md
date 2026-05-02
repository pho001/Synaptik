---
phase: 44-custom-metal-kernel-execution-route
status: clean
reviewed: 2026-05-02
findings_open: 0
---

# Phase 44 Code Review

## Findings

No open findings.

## Review Notes

One trace accuracy issue was found during review and fixed before closing the phase:

- If a prepared custom-kernel route was selected but concrete runtime bindings were not dense `FLOAT32`, execution correctly used MPSGraph, but route metadata still reported `CUSTOM_KERNEL`.
- Fixed in `a4fc53b` by tracking the latest route decision separately from the prepare-time decision and switching runtime metadata to `MPS_GRAPH` with `UNSUPPORTED_LAYOUT` rejected custom evidence when bindings are not custom-kernel eligible.
- Added regression coverage in `PreparedMetalExecutableBufferBindingTest.customKernelRouteReportsMpsGraphWhenRuntimeBindingsAreNotDense`.

## Checks Reviewed

- Route eligibility is scoped to single-node `FLOAT32` `RELU`.
- Native custom bridge is separate from the MPSGraph bridge.
- MPSGraph copy strategy remains `MPSGRAPH_RESULT_COPY`.
- Custom RELU route alone reports `TRUE_OUTPUT_BUFFER_WRITE`.
- Non-dense runtime bindings do not falsely report custom execution.
- Local tuning/profile artifacts remain uncommitted.

## Residual Risk

Custom-kernel support remains intentionally narrow. Broader custom kernel routing and calibrated route selection belong to Phase 46; MPSGraph copy closure belongs to Phase 45.
