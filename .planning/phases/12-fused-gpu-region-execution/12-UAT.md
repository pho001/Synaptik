---
status: testing
phase: 12-fused-gpu-region-execution
source:
  - 12-01-SUMMARY.md
  - 12-02-SUMMARY.md
  - 12-03-SUMMARY.md
  - 12-04-SUMMARY.md
started: 2026-04-30T19:44:35Z
updated: 2026-04-30T19:44:35Z
---

## Current Test
<!-- OVERWRITE each test - shows where we are -->

number: 1
name: Linear Bias Activation Stays GPU Compound
expected: |
  Running the focused Phase 12 lowering/prepared-execution checks shows Metal and CUDA reporting a `LINEAR_BIAS_ACTIVATION` compound summary for a `linear(weight, bias).relu()` style region, without relying on CPU fused internals.
awaiting: user response

## Tests

### 1. Linear Bias Activation Stays GPU Compound
expected: Running the focused Phase 12 lowering/prepared-execution checks shows Metal and CUDA reporting a `LINEAR_BIAS_ACTIVATION` compound summary for a `linear(weight, bias).relu()` style region, without relying on CPU fused internals.
result: [pending]

### 2. Elementwise Chain Publishes GPU Compound Trace
expected: A representative `ADD -> RELU -> EXP` Metal/CUDA region publishes `ELEMENTWISE_CHAIN` through prepared executable metadata and `gpuCompound*` run trace attributes, with device-owned intermediate residency covered by buffer-binding tests.
result: [pending]

### 3. Reduction Adjacent Rejection Is Explicit
expected: Reduction-adjacent candidates such as `LAYER_NORM` and `RMS_NORM` do not silently fall back; planner or trace diagnostics name `REDUCTION_ADJACENT` and stable reasons such as `COMPOUND_PATTERN_UNSUPPORTED` or `DEFERRED_FUSED_REGION`.
result: [pending]

### 4. CPU Fused Operation Remains CPU-Only
expected: GPU Metal/CUDA paths reject `Operation.OpType.FUSED` with `CPU_FUSED_OPERATION_UNSUPPORTED`, and accelerator/Metal/CUDA production packages have no `backend.cpu.fused` imports.
result: [pending]

### 5. Phase 12 Verification And Hygiene Evidence Is Recorded
expected: `12-04-SUMMARY.md` records the final focused verification results, optional native Metal/CUDA status, and confirms local `profiles/platform/.../tuning/abc/*` files remained unstaged.
result: [pending]

## Summary

total: 5
passed: 0
issues: 0
pending: 5
skipped: 0
blocked: 0

## Gaps

[none yet]
