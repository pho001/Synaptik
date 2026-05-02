---
phase: 46-cross-backend-router-calibration-and-regression-gates
status: context
created: 2026-05-02
mode: auto
---

# Phase 46 Context: Cross-Backend Router Calibration And Regression Gates

## Goal

Make backend route decisions auditably evidence-driven across MPSGraph, scoped custom Metal kernels, CUDA, tensor-array fallback, and CPU fallback. Representative workload gates must fail hidden CPU exits, tensor-array replay, unsupported route overclaims, and unexpected copy regressions.

## Locked Decisions

- Public `Tensor` remains logical; backend residency and native handles stay in runtime execution state.
- Existing route execution paths are not rewritten in this phase. Phase 46 hardens evidence and gates around them.
- CUDA capability skips remain evidence, not support.
- MPSGraph output remains `MPSGRAPH_RESULT_COPY` / `COPY_REQUIRED`; scoped custom RELU remains `TRUE_OUTPUT_BUFFER_WRITE` / `PROVEN_TRUE_WRITE`.
- Reports must distinguish route selection, rejected alternatives, copy/write status, region length, lowered primitive count, fallback reasons, and backend ownership.
- Local benchmark/calibration profile artifacts remain unstaged.

## Starting Point

- `GpuCoverageSummary` already tracks GPU coverage ratio, region length, lowered primitive count, native buffer steps, tensor-array fallback, CPU fallback, materializations, native copy strategy, Metal route counts, and rejected route reasons.
- `AcceleratorTraceSummary` already reports accelerator transport, route counts, rejected route reasons, native copy strategies, output write statuses, and native copy timings.
- Phase 46 needs a clearer cross-backend evidence model and hard gates that can be used by representative workload suites.

## Phase Boundaries

In scope:

- Router evidence aggregation from existing trace attributes.
- Gate policies for hidden exits, route overclaim, tensor-array replay, CPU fallback, and native-copy/write-status regressions.
- Tests and docs proving the evidence contract.

Out of scope:

- Universal CUDA native implementation.
- Public GPU tensor/device API.
- Replacing MPSGraph with custom kernels.
- Committing local tuning profile output.
