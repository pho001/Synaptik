---
status: complete
phase: 11-gpu-lowering-coverage-matrix
source:
  - .planning/phases/11-gpu-lowering-coverage-matrix/11-01-SUMMARY.md
  - .planning/phases/11-gpu-lowering-coverage-matrix/11-02-SUMMARY.md
  - .planning/phases/11-gpu-lowering-coverage-matrix/11-03-SUMMARY.md
  - .planning/phases/11-gpu-lowering-coverage-matrix/11-04-SUMMARY.md
started: 2026-04-30T18:30:04Z
updated: 2026-04-30T18:33:54Z
---

## Current Test
<!-- OVERWRITE each test - shows where we are -->

[testing complete]

## Tests

### 1. GPU Lowering Coverage Matrix
expected: The checked-in Metal/CUDA coverage matrix and docs classify common NN/tensor operation families as supported, fallback, or unsupported with stable reason codes. The matrix includes matmul/linear, elementwise chains, layout/view-adjacent ops, softmax-like flows, reductions, normalization, loss-adjacent ops, attention, and deferred fused patterns.
result: pass

### 2. Metal And CUDA Legality Routing
expected: Metal and CUDA planner legality consume the shared coverage matrix while preserving backend-owned dtype, layout, SDPA, capability, and native ABI gates. Unsupported reductions, normalization, loss-adjacent operations, and CUDA direct non-dense inputs reject visibly with stable reason fragments.
result: pass

### 3. LOG_SOFTMAX GPU Lowering
expected: LOG_SOFTMAX is marked supported for Metal and CUDA and lowers as SOFTMAX followed by LOG using existing accelerator DAG primitives, without adding a new native ABI op code. Matmul or linear followed by LOG_SOFTMAX can remain in selected GPU regions when other contracts allow it.
result: pass

### 4. Trace, Layout-Heavy Flow, And Docs Closure
expected: Phase 11 verification evidence shows selected MATMUL/LINEAR to LOG_SOFTMAX GPU coverage, visible unsupported-family rejection in traces, layout-heavy Metal/CUDA flow coverage, developer docs for adding GPU-lowerable operation families, and no committed local tuning/profile artifacts.
result: pass

## Summary

total: 4
passed: 4
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none yet]
